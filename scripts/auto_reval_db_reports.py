#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

EXIT_ACTIONS = ("EXIT", "REBAL_EXIT", "KILL", "MANUAL", "EXTERNAL")


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_float_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default) or default)
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default) or default)
    except Exception:
        return float(default)


def _parse_batch_kpi(obj: Any) -> dict[str, float] | None:
    if not isinstance(obj, dict):
        return None
    out: dict[str, float] = {}
    for k in ("direction_hit", "entry_issue_ratio", "avg_exit_regret"):
        try:
            out[k] = float(obj.get(k))
        except Exception:
            return None
    return out


def _load_progress(path: Path) -> dict[str, Any]:
    base = {
        "completed_batches": 0,
        "completed_reports_total": 0,
        "processed_new_closed_total": 0,
        "last_ready_ts_ms": 0,
        "last_ready_closed_total": 0,
        "last_batch_kpi": None,
    }
    if not path.exists():
        return dict(base)
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return dict(base)
    except Exception:
        return dict(base)
    out = dict(base)
    for k in ("completed_batches", "completed_reports_total", "processed_new_closed_total", "last_ready_ts_ms", "last_ready_closed_total"):
        out[k] = max(0, _safe_int(payload.get(k), out[k]))
    out["last_batch_kpi"] = _parse_batch_kpi(payload.get("last_batch_kpi"))
    return out


def _save_progress(path: Path, progress: dict[str, Any]) -> None:
    payload = {
        "completed_batches": max(0, _safe_int(progress.get("completed_batches"), 0)),
        "completed_reports_total": max(0, _safe_int(progress.get("completed_reports_total"), 0)),
        "processed_new_closed_total": max(0, _safe_int(progress.get("processed_new_closed_total"), 0)),
        "last_ready_ts_ms": max(0, _safe_int(progress.get("last_ready_ts_ms"), 0)),
        "last_ready_closed_total": max(0, _safe_int(progress.get("last_ready_closed_total"), 0)),
        "saved_ts_ms": int(time.time() * 1000),
    }
    last_batch_kpi = _parse_batch_kpi(progress.get("last_batch_kpi"))
    if isinstance(last_batch_kpi, dict):
        payload["last_batch_kpi"] = last_batch_kpi
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _load_baseline_closed(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return max(0, _safe_int((payload or {}).get("baseline_closed"), 0))
    except Exception:
        return 0


def _count_closed_total(conn: sqlite3.Connection) -> int:
    q = (
        "SELECT COUNT(*) FROM trades "
        "WHERE action IN ('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL')"
    )
    cur = conn.cursor()
    cur.execute(q)
    row = cur.fetchone()
    return max(0, _safe_int(row[0] if row else 0, 0))


def _baseline_since_id(conn: sqlite3.Connection, baseline_closed: int) -> int:
    if baseline_closed <= 0:
        return 0
    q = (
        "SELECT id FROM trades "
        "WHERE action IN ('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL') "
        "ORDER BY id LIMIT 1 OFFSET ?"
    )
    cur = conn.cursor()
    cur.execute(q, (max(0, int(baseline_closed) - 1),))
    row = cur.fetchone()
    if not row:
        return 0
    return max(0, _safe_int(row[0], 0))


def _run(cmd: list[str], cwd: Path) -> dict[str, Any]:
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
        return {
            "cmd": cmd,
            "code": int(proc.returncode),
            "elapsed_sec": float(time.time() - t0),
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "code": -1,
            "elapsed_sec": float(time.time() - t0),
            "stdout_tail": "",
            "stderr_tail": str(e),
        }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _lower_text(v: Any) -> str:
    try:
        return str(v or "").strip().lower()
    except Exception:
        return ""


def _extract_stage4_metrics(db_path: Path, since_id: int, cf: dict[str, Any]) -> dict[str, Any]:
    liq_keywords = ("liquidat", "unrealized_dd", "emergency_stop", "kill", "drawdown")
    event_keywords = ("event_mc_exit",)
    hold_keywords = ("hold_vs_exit",)
    total_exits = 0
    liq_like_n = 0
    event_mc_exit_n = 0
    hold_vs_exit_n = 0
    sum_roe = 0.0
    roe_n = 0

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT entry_reason, roe
            FROM trades
            WHERE id > ?
              AND action IN ('EXIT','REBAL_EXIT')
            """,
            (int(since_id),),
        ).fetchall()

    for reason, roe in rows:
        total_exits += 1
        rr = _lower_text(reason)
        if any(k in rr for k in liq_keywords):
            liq_like_n += 1
        if any(k in rr for k in event_keywords):
            event_mc_exit_n += 1
        if any(k in rr for k in hold_keywords):
            hold_vs_exit_n += 1
        if roe is not None:
            sum_roe += _safe_float(roe, 0.0)
            roe_n += 1

    exit_cf = (cf or {}).get("exit_counterfactual") or {}
    avg_exit_regret = exit_cf.get("avg_exit_regret")
    early_like_rate = exit_cf.get("early_like_rate")
    improvable_rate = exit_cf.get("improvable_rate_regret_gt_10bps")
    p90_exit_regret = exit_cf.get("p90_exit_regret")

    return {
        "since_id": int(since_id),
        "exit_count": int(total_exits),
        "avg_roe": (_safe_float(sum_roe / roe_n, 0.0) if roe_n > 0 else None),
        "liq_like_count": int(liq_like_n),
        "liq_like_rate": (float(liq_like_n / total_exits) if total_exits > 0 else None),
        "event_mc_exit_count": int(event_mc_exit_n),
        "event_mc_exit_rate": (float(event_mc_exit_n / total_exits) if total_exits > 0 else None),
        "hold_vs_exit_count": int(hold_vs_exit_n),
        "hold_vs_exit_rate": (float(hold_vs_exit_n / total_exits) if total_exits > 0 else None),
        "avg_exit_regret": (_safe_float(avg_exit_regret) if avg_exit_regret is not None else None),
        "p90_exit_regret": (_safe_float(p90_exit_regret) if p90_exit_regret is not None else None),
        "early_like_rate": (_safe_float(early_like_rate) if early_like_rate is not None else None),
        "improvable_rate_regret_gt_10bps": (_safe_float(improvable_rate) if improvable_rate is not None else None),
    }


def _read_stage4_baseline(path: Path) -> dict[str, Any] | None:
    obj = _load_json(path)
    if not isinstance(obj, dict):
        return None
    m = obj.get("metrics")
    if not isinstance(m, dict):
        return None
    return obj


def _metric_delta(curr: Any, base: Any) -> float | None:
    if curr is None or base is None:
        return None
    try:
        return float(curr) - float(base)
    except Exception:
        return None


def _build_stage4_compare(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    base_m = (baseline or {}).get("metrics") or {}
    cur_m = current or {}
    delta = {
        "liq_like_rate": _metric_delta(cur_m.get("liq_like_rate"), base_m.get("liq_like_rate")),
        "avg_exit_regret": _metric_delta(cur_m.get("avg_exit_regret"), base_m.get("avg_exit_regret")),
        "early_like_rate": _metric_delta(cur_m.get("early_like_rate"), base_m.get("early_like_rate")),
        "event_mc_exit_rate": _metric_delta(cur_m.get("event_mc_exit_rate"), base_m.get("event_mc_exit_rate")),
        "hold_vs_exit_rate": _metric_delta(cur_m.get("hold_vs_exit_rate"), base_m.get("hold_vs_exit_rate")),
        "avg_roe": _metric_delta(cur_m.get("avg_roe"), base_m.get("avg_roe")),
    }
    improvement = {
        "liq_like_rate_down": (delta["liq_like_rate"] < 0.0) if delta["liq_like_rate"] is not None else None,
        "avg_exit_regret_down": (delta["avg_exit_regret"] < 0.0) if delta["avg_exit_regret"] is not None else None,
        "early_like_rate_down": (delta["early_like_rate"] < 0.0) if delta["early_like_rate"] is not None else None,
        "avg_roe_up": (delta["avg_roe"] > 0.0) if delta["avg_roe"] is not None else None,
    }
    return {
        "baseline": baseline,
        "current": {
            "timestamp_ms": int(time.time() * 1000),
            "metrics": cur_m,
        },
        "delta": delta,
        "improvement": improvement,
    }


def _safe_text(v: Any) -> str:
    try:
        return str(v or "").strip()
    except Exception:
        return ""


def _parse_float_list(raw: Any) -> list[float]:
    txt = _safe_text(raw)
    out: list[float] = []
    if not txt:
        return out
    for tok in txt.split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except Exception:
            continue
    return out


def _default_capital_tier_thresholds() -> list[float]:
    tiers = _parse_float_list(os.environ.get("CAPITAL_TIER_USDT", ""))
    if not tiers:
        tiers = [500.0, 1500.0, 3000.0, 6000.0, 9000.0]
    tiers = sorted(float(max(0.0, x)) for x in tiers)
    out: list[float] = []
    prev = -1.0
    for x in tiers:
        if x > prev:
            out.append(float(x))
            prev = float(x)
    return out or [500.0, 1500.0, 3000.0, 6000.0, 9000.0]


def _default_min_notional_tiers(n: int) -> list[float]:
    nn = int(max(2, n))
    base_seed = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    if nn <= len(base_seed):
        return [float(x) for x in base_seed[:nn]]
    out = [float(x) for x in base_seed]
    while len(out) < nn:
        out.append(float(round(out[-1] * 1.7, 3)))
    return out[:nn]


def _avg(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(sum(vals) / float(len(vals)))


def _quantile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    x = sorted(float(v) for v in vals)
    qq = float(max(0.0, min(1.0, q)))
    if len(x) == 1:
        return float(x[0])
    pos = qq * float(len(x) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(x[lo])
    w = float(pos - lo)
    return float(x[lo] * (1.0 - w) + x[hi] * w)


def _exec_group_from_actual(exec_type: Any) -> str:
    et = _safe_text(exec_type).lower()
    if "maker_fallback" in et:
        return "taker"
    if "maker" in et:
        return "maker"
    return "taker"


def _exec_group_from_selected(selected_exec_type: Any, actual_exec_type: Any) -> str:
    st = _safe_text(selected_exec_type).lower()
    if st:
        if "maker" in st:
            return "maker"
        if "market" in st or "taker" in st:
            return "taker"
    # Backward-compatible inference for older slippage rows without selected_exec_type.
    at = _safe_text(actual_exec_type).lower()
    if "maker_fallback" in at:
        return "maker"
    if "maker" in at:
        return "maker"
    return "taker"


def _extract_slippage_exit_report(
    db_path: Path,
    *,
    since_id: int,
    exit_limit: int = 200,
    match_window_ms: int = 120_000,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "source": "none",
        "exit_limit": int(max(20, exit_limit)),
        "match_window_ms": int(max(5_000, match_window_ms)),
        "exit_sample_n": 0,
        "matched_n": 0,
        "match_rate": None,
        "estimation_n": 0,
        "estimation_mae_bps": None,
        "estimation_rmse_bps": None,
        "estimation_bias_bps": None,
        "actual_slippage_avg_bps": None,
        "actual_abs_slippage_avg_bps": None,
        "latency_ms_avg": None,
        "selected_exec_ab": {
            "maker": {"n": 0, "avg_abs_slippage_bps": None, "avg_error_mae_bps": None},
            "taker": {"n": 0, "avg_abs_slippage_bps": None, "avg_error_mae_bps": None},
            "delta_abs_slippage_bps_taker_minus_maker": None,
            "delta_error_mae_bps_taker_minus_maker": None,
            "selected_vs_actual_match_rate": None,
            "maker_selected_fallback_rate": None,
        },
        "by_actual_exec_group": {
            "maker": {"n": 0, "avg_abs_slippage_bps": None},
            "taker": {"n": 0, "avg_abs_slippage_bps": None},
        },
        "top_exec_mismatch_symbols": [],
    }
    if not db_path.exists():
        return out

    lim = int(max(20, exit_limit))
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            slip_cols = {str(r[1]) for r in cur.execute("PRAGMA table_info(slippage_analysis)").fetchall()}
            selected_col_sql = "selected_exec_type" if "selected_exec_type" in slip_cols else "NULL AS selected_exec_type"
            exits = cur.execute(
                """
                SELECT id, symbol, side, timestamp_ms
                FROM trades
                WHERE id > ?
                  AND action IN ('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL')
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(max(0, since_id)), int(lim)),
            ).fetchall()
            source = "since_id"
            if len(exits) < max(40, int(0.25 * lim)):
                exits = cur.execute(
                    """
                    SELECT id, symbol, side, timestamp_ms
                    FROM trades
                    WHERE action IN ('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL')
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(lim),),
                ).fetchall()
                source = "global_recent"
            exits = list(exits)
            if not exits:
                out["source"] = source
                return out
            min_ts = min(int(_safe_int(r[3], 0)) for r in exits)
            max_ts = max(int(_safe_int(r[3], 0)) for r in exits)
            t0 = int(min_ts - max(5_000, match_window_ms))
            t1 = int(max_ts + max(5_000, match_window_ms))
            slips = cur.execute(
                f"""
                SELECT id, symbol, timestamp_ms, slippage_bps, estimated_slippage_bps, estimation_error_bps,
                       side, exec_type, {selected_col_sql}, latency_ms
                FROM slippage_analysis
                WHERE timestamp_ms BETWEEN ? AND ?
                ORDER BY timestamp_ms ASC
                """,
                (int(t0), int(t1)),
            ).fetchall()
    except Exception:
        return out

    out["source"] = source
    out["exit_sample_n"] = int(len(exits))
    if not slips:
        return out

    by_symbol: dict[str, list[tuple[Any, ...]]] = {}
    for row in slips:
        sym = _safe_text(row[1])
        if not sym:
            continue
        by_symbol.setdefault(sym, []).append(row)

    used_slip_ids: set[int] = set()
    matched: list[dict[str, Any]] = []
    w_ms = int(max(5_000, match_window_ms))
    for ex in exits:
        ex_id = _safe_int(ex[0], 0)
        sym = _safe_text(ex[1])
        ex_ts = _safe_int(ex[3], 0)
        if not sym or ex_ts <= 0:
            continue
        cands = by_symbol.get(sym) or []
        best = None
        best_dt = None
        for r in cands:
            sid = _safe_int(r[0], 0)
            if sid <= 0 or sid in used_slip_ids:
                continue
            st = _safe_int(r[2], 0)
            dt = abs(int(st) - int(ex_ts))
            if dt > w_ms:
                continue
            if best is None or dt < int(best_dt):
                best = r
                best_dt = dt
        if best is None:
            continue
        sid = _safe_int(best[0], 0)
        used_slip_ids.add(sid)
        slip_bps = _safe_float(best[3], 0.0)
        est_bps = _safe_float_or_none(best[4])
        err_bps = _safe_float_or_none(best[5])
        if err_bps is None and est_bps is not None:
            err_bps = float(slip_bps - float(est_bps))
        actual_exec = _safe_text(best[7])
        selected_exec = _safe_text(best[8])
        selected_group = _exec_group_from_selected(selected_exec, actual_exec)
        actual_group = _exec_group_from_actual(actual_exec)
        matched.append(
            {
                "exit_id": int(ex_id),
                "symbol": sym,
                "dt_ms": int(best_dt or 0),
                "slippage_bps": float(slip_bps),
                "estimated_slippage_bps": est_bps,
                "error_bps": err_bps,
                "latency_ms": _safe_float_or_none(best[9]),
                "actual_exec_type": actual_exec,
                "selected_exec_type": selected_exec,
                "selected_group": selected_group,
                "actual_group": actual_group,
            }
        )

    m_n = int(len(matched))
    out["matched_n"] = m_n
    out["match_rate"] = float(m_n / max(1, int(len(exits))))
    if m_n <= 0:
        return out

    errs = [float(r["error_bps"]) for r in matched if r.get("error_bps") is not None]
    slips_signed = [float(r.get("slippage_bps", 0.0)) for r in matched]
    slips_abs = [abs(float(r.get("slippage_bps", 0.0))) for r in matched]
    lats = [float(r["latency_ms"]) for r in matched if r.get("latency_ms") is not None]
    out["estimation_n"] = int(len(errs))
    out["estimation_mae_bps"] = _avg([abs(x) for x in errs])
    out["estimation_rmse_bps"] = (float(math.sqrt(sum(x * x for x in errs) / max(1, len(errs)))) if errs else None)
    out["estimation_bias_bps"] = _avg(errs)
    out["actual_slippage_avg_bps"] = _avg(slips_signed)
    out["actual_abs_slippage_avg_bps"] = _avg(slips_abs)
    out["latency_ms_avg"] = _avg(lats)

    selected_stats: dict[str, dict[str, Any]] = {
        "maker": {"n": 0, "abs_slips": [], "err_abs": []},
        "taker": {"n": 0, "abs_slips": [], "err_abs": []},
    }
    actual_stats: dict[str, dict[str, Any]] = {
        "maker": {"n": 0, "abs_slips": []},
        "taker": {"n": 0, "abs_slips": []},
    }
    match_n = 0
    maker_sel_n = 0
    maker_fallback_n = 0
    mismatch_by_symbol: dict[str, int] = {}
    for r in matched:
        sg = str(r.get("selected_group") or "taker")
        ag = str(r.get("actual_group") or "taker")
        abs_slip = abs(float(r.get("slippage_bps") or 0.0))
        e = r.get("error_bps")
        if sg in selected_stats:
            selected_stats[sg]["n"] = int(selected_stats[sg]["n"]) + 1
            selected_stats[sg]["abs_slips"].append(float(abs_slip))
            if e is not None:
                selected_stats[sg]["err_abs"].append(abs(float(e)))
        if ag in actual_stats:
            actual_stats[ag]["n"] = int(actual_stats[ag]["n"]) + 1
            actual_stats[ag]["abs_slips"].append(float(abs_slip))
        if sg == ag:
            match_n += 1
        else:
            sym = _safe_text(r.get("symbol"))
            if sym:
                mismatch_by_symbol[sym] = int(mismatch_by_symbol.get(sym, 0) + 1)
        if sg == "maker":
            maker_sel_n += 1
            if ag != "maker":
                maker_fallback_n += 1

    maker_abs = _avg(selected_stats["maker"]["abs_slips"])
    taker_abs = _avg(selected_stats["taker"]["abs_slips"])
    maker_err = _avg(selected_stats["maker"]["err_abs"])
    taker_err = _avg(selected_stats["taker"]["err_abs"])
    out["selected_exec_ab"] = {
        "maker": {
            "n": int(selected_stats["maker"]["n"]),
            "avg_abs_slippage_bps": maker_abs,
            "avg_error_mae_bps": maker_err,
        },
        "taker": {
            "n": int(selected_stats["taker"]["n"]),
            "avg_abs_slippage_bps": taker_abs,
            "avg_error_mae_bps": taker_err,
        },
        "delta_abs_slippage_bps_taker_minus_maker": (
            float(taker_abs - maker_abs) if (taker_abs is not None and maker_abs is not None) else None
        ),
        "delta_error_mae_bps_taker_minus_maker": (
            float(taker_err - maker_err) if (taker_err is not None and maker_err is not None) else None
        ),
        "selected_vs_actual_match_rate": float(match_n / max(1, m_n)),
        "maker_selected_fallback_rate": float(maker_fallback_n / max(1, maker_sel_n)) if maker_sel_n > 0 else None,
    }
    out["by_actual_exec_group"] = {
        "maker": {
            "n": int(actual_stats["maker"]["n"]),
            "avg_abs_slippage_bps": _avg(actual_stats["maker"]["abs_slips"]),
        },
        "taker": {
            "n": int(actual_stats["taker"]["n"]),
            "avg_abs_slippage_bps": _avg(actual_stats["taker"]["abs_slips"]),
        },
    }
    out["top_exec_mismatch_symbols"] = [
        {"symbol": sym, "n": int(n)}
        for sym, n in sorted(mismatch_by_symbol.items(), key=lambda kv: kv[1], reverse=True)[:8]
    ]
    out["available"] = True
    return out


def _extract_min_notional_tuning(
    db_path: Path,
    *,
    since_id: int,
    recent_limit: int = 1200,
) -> dict[str, Any]:
    tier_thresholds = _default_capital_tier_thresholds()
    expected_tiers = int(len(tier_thresholds) + 1)
    current_tiers = _parse_float_list(os.environ.get("CAPITAL_TIER_MIN_NOTIONAL", ""))
    default_tiers = _default_min_notional_tiers(expected_tiers)
    if not current_tiers:
        current_tiers = list(default_tiers)
    if len(current_tiers) < expected_tiers:
        current_tiers.extend([float(current_tiers[-1] if current_tiers else default_tiers[-1])] * (expected_tiers - len(current_tiers)))
    elif len(current_tiers) > expected_tiers:
        current_tiers = current_tiers[:expected_tiers]
    try:
        tier_cap = float(os.environ.get("CAPITAL_TIER_MIN_NOTIONAL_MAX", 20.0) or 20.0)
    except Exception:
        tier_cap = 20.0
    tier_cap = float(max(2.0, tier_cap))
    current_tiers = [float(max(0.1, min(tier_cap, x))) for x in current_tiers]
    out: dict[str, Any] = {
        "available": False,
        "source": "none",
        "recent_limit": int(max(80, recent_limit)),
        "reject_sample_n": 0,
        "min_notional_blocked_n": 0,
        "min_notional_blocked_share": None,
        "tier_thresholds": [float(x) for x in tier_thresholds],
        "current_tiers": list(current_tiers),
        "recommended_tiers": list(current_tiers),
        "apply_hint": False,
        "top_blocked_symbols": [],
    }
    if not db_path.exists():
        return out

    lim = int(max(80, recent_limit))
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT id, symbol, entry_reason, raw_data
                FROM trades
                WHERE id > ?
                  AND action='ORDER_REJECT'
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(max(0, since_id)), int(lim)),
            ).fetchall()
            source = "since_id"
            if len(rows) < max(80, int(0.25 * lim)):
                rows = cur.execute(
                    """
                    SELECT id, symbol, entry_reason, raw_data
                    FROM trades
                    WHERE action='ORDER_REJECT'
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(lim),),
                ).fetchall()
                source = "global_recent"
    except Exception:
        return out

    out["source"] = source
    out["reject_sample_n"] = int(len(rows))
    if not rows:
        return out

    min_tokens = ("min_notional", "min notional", "minimum amount precision", "min_qty", "min qty")
    min_blocks: list[dict[str, Any]] = []
    by_symbol: dict[str, dict[str, Any]] = {}
    for rid, symbol, reason_raw, raw_txt in rows:
        reason = _safe_text(reason_raw).lower()
        raw_obj: dict[str, Any] = {}
        try:
            raw_obj = json.loads(raw_txt) if raw_txt else {}
        except Exception:
            raw_obj = {}
        raw_s = _safe_text(raw_txt).lower()
        explicit_min = any(tok in reason for tok in min_tokens) or any(tok in raw_s for tok in min_tokens)
        insufficient_min_qty = "insufficient_balance_or_min_qty" in reason
        est_notional = _safe_float_or_none(raw_obj.get("est_notional"))
        if est_notional is None:
            est_notional = _safe_float_or_none(raw_obj.get("notional"))
        min_floor = _safe_float_or_none(raw_obj.get("min_entry_notional"))
        if min_floor is None:
            min_floor = _safe_float_or_none(raw_obj.get("min_notional"))
        if min_floor is None:
            min_floor = _safe_float_or_none(raw_obj.get("policy_min_notional"))
        inferred_min = bool(
            insufficient_min_qty
            and est_notional is not None
            and (
                (min_floor is not None and float(est_notional) <= float(min_floor) * 1.01)
                or (float(est_notional) <= float(max(0.1, current_tiers[0])) * 1.25)
            )
        )
        if not (explicit_min or inferred_min):
            continue
        sym = _safe_text(symbol)
        if not sym:
            continue
        rec = {
            "id": int(_safe_int(rid, 0)),
            "symbol": sym,
            "est_notional": est_notional,
            "min_floor": min_floor,
            "explicit_min": bool(explicit_min),
        }
        min_blocks.append(rec)
        st = by_symbol.setdefault(sym, {"n": 0, "notionals": [], "explicit_n": 0})
        st["n"] = int(st["n"]) + 1
        if est_notional is not None:
            st["notionals"].append(float(est_notional))
        if explicit_min:
            st["explicit_n"] = int(st["explicit_n"]) + 1

    blocked_n = int(len(min_blocks))
    out["min_notional_blocked_n"] = blocked_n
    out["min_notional_blocked_share"] = float(blocked_n / max(1, len(rows)))
    if blocked_n <= 0:
        return out

    top_rows = []
    all_notionals: list[float] = []
    for sym, st in by_symbol.items():
        ns = list(st.get("notionals") or [])
        all_notionals.extend(float(x) for x in ns)
        top_rows.append(
            {
                "symbol": sym,
                "n": int(st.get("n", 0)),
                "explicit_ratio": float(st.get("explicit_n", 0) / max(1, st.get("n", 0))),
                "avg_est_notional": _avg(ns),
                "p50_est_notional": _quantile(ns, 0.50),
                "p75_est_notional": _quantile(ns, 0.75),
            }
        )
    top_rows.sort(key=lambda x: int(x.get("n", 0)), reverse=True)
    out["top_blocked_symbols"] = top_rows[:12]

    rec_tiers = list(current_tiers)
    if len(all_notionals) >= max(12, expected_tiers * 3):
        # Distribution-driven proposal; works with variable tier count.
        quantiles = [float((i + 1) / (expected_tiers + 1)) for i in range(expected_tiers)]
        cand = [_quantile(all_notionals, q) for q in quantiles]
        if all(v is not None for v in cand):
            tuned: list[float] = []
            for i, raw_v in enumerate(cand):
                vv = float(max(0.1, min(tier_cap, float(raw_v))))
                floor_i = 0.1 if i == 0 else float(tuned[i - 1] + 0.15)
                vv = float(max(floor_i, vv))
                tuned.append(vv)
            rec_tiers = [float(round(v, 3)) for v in tuned]
    out["recommended_tiers"] = rec_tiers
    out["apply_hint"] = bool(blocked_n >= 10 and float(out["min_notional_blocked_share"] or 0.0) >= 0.05)
    out["available"] = True
    return out


def _extract_min_notional_effectiveness(
    db_path: Path,
    *,
    since_id: int,
    recent_limit: int = 1200,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "source": "none",
        "sample_n": 0,
        "current_floor": None,
        "overall_avg_roe": None,
        "overall_win_rate": None,
        "recommended_base_floor": None,
        "recommended_score": None,
        "current_score": None,
        "score_gain_vs_current": None,
        "low_notional_avg_roe": None,
        "low_notional_win_rate": None,
        "low_minus_overall_roe": None,
        "p25_notional": None,
        "p50_notional": None,
        "p75_notional": None,
    }
    if not db_path.exists():
        return out

    cur_tiers = _parse_float_list(os.environ.get("CAPITAL_TIER_MIN_NOTIONAL", ""))
    current_floor = float(max(0.1, cur_tiers[0])) if cur_tiers else 0.1
    out["current_floor"] = float(current_floor)

    lim = int(max(100, recent_limit))
    rows: list[tuple[Any, Any, Any]] = []
    source = "since_id"
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT notional, roe, raw_data
                FROM trades
                WHERE id > ?
                  AND action IN ('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL')
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(max(0, since_id)), int(lim)),
            ).fetchall()
            if len(rows) < max(80, int(0.30 * lim)):
                rows = cur.execute(
                    """
                    SELECT notional, roe, raw_data
                    FROM trades
                    WHERE action IN ('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL')
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(lim),),
                ).fetchall()
                source = "global_recent"
    except Exception:
        return out
    out["source"] = source
    if not rows:
        return out

    samples: list[tuple[float, float]] = []
    for notional_col, roe_col, raw_txt in rows:
        notional = _safe_float_or_none(notional_col)
        roe = _safe_float_or_none(roe_col)
        raw_obj: dict[str, Any] = {}
        if (notional is None or notional <= 0) or (roe is None):
            try:
                raw_obj = json.loads(raw_txt) if raw_txt else {}
            except Exception:
                raw_obj = {}
        if notional is None or notional <= 0:
            notional = _safe_float_or_none(raw_obj.get("notional"))
            if notional is None or notional <= 0:
                notional = _safe_float_or_none(raw_obj.get("entry_notional"))
            if notional is None or notional <= 0:
                notional = _safe_float_or_none(raw_obj.get("est_notional"))
        if roe is None:
            roe = _safe_float_or_none(raw_obj.get("roe"))
        if notional is None or roe is None:
            continue
        n_v = float(notional)
        r_v = float(roe)
        if (not math.isfinite(n_v)) or (not math.isfinite(r_v)):
            continue
        if n_v <= 0:
            continue
        if abs(r_v) > 20.0:
            continue
        samples.append((n_v, r_v))

    if len(samples) < 60:
        return out

    notionals = [x for x, _ in samples]
    roes = [y for _, y in samples]
    n_total = len(samples)
    overall_avg = _avg(roes)
    overall_win = float(sum(1 for r in roes if r > 0.0) / max(1, n_total))
    out["sample_n"] = int(n_total)
    out["overall_avg_roe"] = overall_avg
    out["overall_win_rate"] = overall_win
    out["p25_notional"] = _quantile(notionals, 0.25)
    out["p50_notional"] = _quantile(notionals, 0.50)
    out["p75_notional"] = _quantile(notionals, 0.75)

    # Evaluate floor candidates by throughput-adjusted expectancy score.
    candidate_q = [0.00, 0.08, 0.16, 0.24, 0.32, 0.40, 0.50, 0.60]
    candidates = [current_floor]
    for q in candidate_q:
        vv = _quantile(notionals, q)
        if vv is not None:
            candidates.append(float(max(0.1, vv)))
    candidates = sorted(set(round(v, 6) for v in candidates))

    def _score_for_floor(floor_v: float) -> tuple[float | None, float | None, int]:
        picked = [r for n, r in samples if n >= float(floor_v)]
        n_pick = len(picked)
        if n_pick < max(40, int(0.15 * n_total)):
            return None, None, int(n_pick)
        avg_roe = _avg(picked)
        if avg_roe is None:
            return None, None, int(n_pick)
        coverage = float(n_pick / max(1, n_total))
        return float(avg_roe * coverage), float(avg_roe), int(n_pick)

    best_floor = float(current_floor)
    best_score, _, _ = _score_for_floor(best_floor)
    cur_score = best_score
    if cur_score is None:
        cur_score = 0.0
    if best_score is None:
        best_score = 0.0
    for c in candidates:
        sc, _, _ = _score_for_floor(float(c))
        if sc is None:
            continue
        if sc > float(best_score) + 1e-12:
            best_score = float(sc)
            best_floor = float(c)

    low_cap = _quantile(notionals, 0.30)
    if low_cap is None:
        low_cap = float(max(current_floor * 4.0, current_floor + 0.2))
    low_rows = [r for n, r in samples if n <= float(low_cap)]
    low_avg = _avg(low_rows)
    low_win = float(sum(1 for r in low_rows if r > 0.0) / max(1, len(low_rows))) if low_rows else None

    out["recommended_base_floor"] = float(best_floor)
    out["recommended_score"] = float(best_score)
    out["current_score"] = float(cur_score)
    out["score_gain_vs_current"] = float(best_score - cur_score)
    out["low_notional_avg_roe"] = low_avg
    out["low_notional_win_rate"] = low_win
    if low_avg is not None and overall_avg is not None:
        out["low_minus_overall_roe"] = float(low_avg - overall_avg)
    out["available"] = True
    return out


def _extract_batch_kpi(diag: dict[str, Any], cf: dict[str, Any]) -> dict[str, float | None]:
    entry_direction = (diag or {}).get("entry_direction") or {}
    attribution = (diag or {}).get("attribution") or {}
    exit_cf = (cf or {}).get("exit_counterfactual") or {}
    miss_rate = _safe_float(entry_direction.get("miss_rate"), 0.0)
    direction_hit = max(0.0, min(1.0, 1.0 - float(miss_rate)))
    entry_issue_ratio = _safe_float(attribution.get("entry_issue_rate"), 0.0)
    avg_exit_regret = _safe_float(exit_cf.get("avg_exit_regret"), 0.0)
    return {
        "direction_hit": float(direction_hit),
        "entry_issue_ratio": float(entry_issue_ratio),
        "avg_exit_regret": float(avg_exit_regret),
    }


def _extract_prev_batch_kpi(status: dict[str, Any], progress: dict[str, Any] | None = None) -> dict[str, float] | None:
    if not isinstance(status, dict):
        status = {}
    summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
    kpi = _parse_batch_kpi(summary.get("batch_kpi"))
    if isinstance(kpi, dict):
        return kpi
    if isinstance(progress, dict):
        kpi = _parse_batch_kpi(progress.get("last_batch_kpi"))
        if isinstance(kpi, dict):
            return kpi
    return None


def _kpi_delta(current: dict[str, float | None], prev: dict[str, float] | None) -> dict[str, float | None]:
    if not isinstance(prev, dict):
        return {
            "direction_hit": None,
            "entry_issue_ratio": None,
            "avg_exit_regret": None,
        }
    out: dict[str, float | None] = {}
    for k in ("direction_hit", "entry_issue_ratio", "avg_exit_regret"):
        cv = current.get(k)
        pv = prev.get(k)
        out[k] = _metric_delta(cv, pv)
    return out


def _evaluate_exit_kpi_gate(
    *,
    batch_kpi: dict[str, float | None],
    batch_kpi_delta: dict[str, float | None],
    cf: dict[str, Any],
    reval_loss_driver: dict[str, Any],
) -> dict[str, Any]:
    # Fixed pass/fail gate for exit-quality tuning decisions.
    # PASS: all core targets satisfied.
    # IMPROVING: not yet at target but 2+ key deltas moving in right direction and no severe fail.
    # FAIL: severe breach or no improving evidence.
    direction_hit = _safe_float(batch_kpi.get("direction_hit"), None)
    entry_issue_ratio = _safe_float(batch_kpi.get("entry_issue_ratio"), None)
    avg_exit_regret = _safe_float(batch_kpi.get("avg_exit_regret"), None)
    exit_cf = (cf or {}).get("exit_counterfactual") if isinstance((cf or {}).get("exit_counterfactual"), dict) else {}
    early_like_rate = _safe_float(exit_cf.get("early_like_rate"), None)
    if early_like_rate is None:
        tv = (reval_loss_driver or {}).get("timing_validation")
        if isinstance(tv, dict):
            early_like_rate = _safe_float(tv.get("early_like_rate"), None)

    direction_hit_delta = _safe_float(batch_kpi_delta.get("direction_hit"), None)
    entry_issue_delta = _safe_float(batch_kpi_delta.get("entry_issue_ratio"), None)
    exit_regret_delta = _safe_float(batch_kpi_delta.get("avg_exit_regret"), None)

    tgt_dir_hit = _env_float("AUTO_REVAL_GATE_MIN_DIRECTION_HIT", 0.53)
    tgt_entry_issue = _env_float("AUTO_REVAL_GATE_MAX_ENTRY_ISSUE_RATIO", 0.50)
    tgt_exit_regret = _env_float("AUTO_REVAL_GATE_MAX_AVG_EXIT_REGRET", 0.0080)
    tgt_early_like = _env_float("AUTO_REVAL_GATE_MAX_EARLY_LIKE_RATE", 0.45)

    # Delta guard tolerances: small noise allowed around zero.
    delta_dir_hit_min = _env_float("AUTO_REVAL_GATE_MIN_DIRECTION_HIT_DELTA", -0.005)
    delta_entry_issue_max = _env_float("AUTO_REVAL_GATE_MAX_ENTRY_ISSUE_DELTA", 0.010)
    delta_exit_regret_max = _env_float("AUTO_REVAL_GATE_MAX_EXIT_REGRET_DELTA", 0.0010)

    # Severe fail thresholds.
    hard_exit_regret = _env_float("AUTO_REVAL_GATE_HARD_MAX_AVG_EXIT_REGRET", 0.0150)
    hard_early_like = _env_float("AUTO_REVAL_GATE_HARD_MAX_EARLY_LIKE_RATE", 0.65)

    checks = {
        "direction_hit_ok": (direction_hit is not None and direction_hit >= tgt_dir_hit),
        "entry_issue_ok": (entry_issue_ratio is not None and entry_issue_ratio <= tgt_entry_issue),
        "exit_regret_ok": (avg_exit_regret is not None and avg_exit_regret <= tgt_exit_regret),
        "early_like_ok": (early_like_rate is not None and early_like_rate <= tgt_early_like),
    }
    core_available = int(sum(1 for v in (direction_hit, entry_issue_ratio, avg_exit_regret, early_like_rate) if v is not None))
    pass_all = bool(core_available >= 3 and all(v for v in checks.values() if isinstance(v, bool)))

    improving_flags = {
        "direction_hit_delta_ok": (direction_hit_delta is not None and direction_hit_delta >= delta_dir_hit_min),
        "entry_issue_delta_ok": (entry_issue_delta is not None and entry_issue_delta <= delta_entry_issue_max),
        "exit_regret_delta_ok": (exit_regret_delta is not None and exit_regret_delta <= delta_exit_regret_max),
    }
    improving_count = int(sum(1 for v in improving_flags.values() if bool(v)))
    severe_fail = bool(
        (avg_exit_regret is not None and avg_exit_regret >= hard_exit_regret)
        or (early_like_rate is not None and early_like_rate >= hard_early_like)
    )

    if pass_all:
        grade = "PASS"
    elif (not severe_fail) and improving_count >= 2:
        grade = "IMPROVING"
    else:
        grade = "FAIL"

    fail_reasons: list[str] = []
    if direction_hit is not None and direction_hit < tgt_dir_hit:
        fail_reasons.append("direction_hit_below_target")
    if entry_issue_ratio is not None and entry_issue_ratio > tgt_entry_issue:
        fail_reasons.append("entry_issue_above_target")
    if avg_exit_regret is not None and avg_exit_regret > tgt_exit_regret:
        fail_reasons.append("exit_regret_above_target")
    if early_like_rate is not None and early_like_rate > tgt_early_like:
        fail_reasons.append("early_like_above_target")
    if severe_fail:
        fail_reasons.append("severe_fail_guard")

    if grade == "PASS":
        message = "목표 충족: 현재 청산 품질 기준 통과"
    elif grade == "IMPROVING":
        message = "목표 미달이나 개선 추세 확인: 튜닝 유지"
    else:
        message = "목표 미충족/악화: 청산 보수화 및 원인축 재튜닝 필요"

    return {
        "grade": grade,
        "pass": bool(grade == "PASS"),
        "improving": bool(grade == "IMPROVING"),
        "severe_fail": bool(severe_fail),
        "message": message,
        "targets": {
            "direction_hit_min": float(tgt_dir_hit),
            "entry_issue_ratio_max": float(tgt_entry_issue),
            "avg_exit_regret_max": float(tgt_exit_regret),
            "early_like_rate_max": float(tgt_early_like),
        },
        "hard_limits": {
            "avg_exit_regret_max": float(hard_exit_regret),
            "early_like_rate_max": float(hard_early_like),
        },
        "values": {
            "direction_hit": direction_hit,
            "entry_issue_ratio": entry_issue_ratio,
            "avg_exit_regret": avg_exit_regret,
            "early_like_rate": early_like_rate,
            "direction_hit_delta": direction_hit_delta,
            "entry_issue_ratio_delta": entry_issue_delta,
            "avg_exit_regret_delta": exit_regret_delta,
        },
        "delta_guards": {
            "direction_hit_delta_min": float(delta_dir_hit_min),
            "entry_issue_ratio_delta_max": float(delta_entry_issue_max),
            "avg_exit_regret_delta_max": float(delta_exit_regret_max),
        },
        "checks": checks,
        "improving_checks": improving_flags,
        "improving_count": int(improving_count),
        "fail_reasons": fail_reasons,
    }


def _extract_last_ready_snapshot(status: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(status, dict):
        return None
    ready = bool(status.get("ready") is True)
    summary = status.get("summary") if isinstance(status.get("summary"), dict) else None
    runs = status.get("runs") if isinstance(status.get("runs"), list) else None
    if not ready and not isinstance(summary, dict):
        return None
    snap = {
        "timestamp_ms": _safe_int(status.get("timestamp_ms"), 0),
        "heartbeat_ts_ms": _safe_int(status.get("heartbeat_ts_ms"), 0),
        "baseline_closed": _safe_int(status.get("baseline_closed"), 0),
        "closed_total": _safe_int(status.get("closed_total"), 0),
        "new_closed_total": _safe_int(status.get("new_closed_total"), 0),
        "new_closed_total_cum": _safe_int(status.get("new_closed_total_cum"), 0),
        "since_id": _safe_int(status.get("since_id"), 0),
        "reports_ok": _safe_int(status.get("reports_ok"), 0),
        "summary": dict(summary) if isinstance(summary, dict) else {},
        "runs": list(runs) if isinstance(runs, list) else [],
    }
    return snap


def main() -> int:
    ap = argparse.ArgumentParser(description="Wait until new_closed_total reaches threshold, then run DB report suite.")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--baseline-file", default="state/reval_baseline.json")
    ap.add_argument("--target-new", type=int, default=_env_int("AUTO_REVAL_TARGET_NEW", 120))
    ap.add_argument("--poll-sec", type=float, default=_env_float("AUTO_REVAL_POLL_SEC", 15.0))
    ap.add_argument("--timeout-sec", type=float, default=_env_float("AUTO_REVAL_TIMEOUT_SEC", 0.0), help="0 means no timeout")
    ap.add_argument("--recent", type=int, default=_env_int("AUTO_REVAL_RECENT_ROWS", 5000))
    ap.add_argument("--obs-out", default="state/trade_observability_report_new.json")
    ap.add_argument("--diag-out", default="state/entry_exit_diagnosis_report.json")
    ap.add_argument("--cf-out", default="state/counterfactual_replay_report.json")
    ap.add_argument("--reason-matrix-out", default="state/entry_exit_reason_matrix_report.json")
    ap.add_argument("--entry-score-driver-out", default="state/entry_score_driver_report.json")
    ap.add_argument("--reval-loss-driver-out", default="state/reval_loss_driver_report.json")
    ap.add_argument("--reval-loss-driver-top-csv", default="state/reval_loss_driver_top_loss_positions.csv")
    ap.add_argument("--reval-loss-driver-history-out", default="state/reval_loss_driver_history.json")
    ap.add_argument("--capital-steps", default="1500,3000,6000,9000")
    ap.add_argument("--slippage-exit-out", default="state/slippage_exit_200_report.json")
    ap.add_argument("--slippage-exit-limit", type=int, default=_env_int("AUTO_REVAL_SLIPPAGE_EXIT_LIMIT", 200))
    ap.add_argument("--slippage-match-window-sec", type=float, default=_env_float("AUTO_REVAL_SLIPPAGE_MATCH_WINDOW_SEC", 120.0))
    ap.add_argument("--min-notional-tune-out", default="state/min_notional_tuning_report.json")
    ap.add_argument("--min-notional-effectiveness-out", default="state/min_notional_effectiveness_report.json")
    ap.add_argument("--min-notional-recent-limit", type=int, default=_env_int("AUTO_REVAL_MIN_NOTIONAL_RECENT_LIMIT", 1200))
    ap.add_argument("--reason-matrix-max-symbols", type=int, default=_env_int("AUTO_REVAL_REASON_MAX_SYMBOLS", 40))
    ap.add_argument("--reason-matrix-max-cell-top", type=int, default=_env_int("AUTO_REVAL_REASON_MAX_CELL_TOP", 25))
    ap.add_argument("--status-out", default="state/auto_reval_db_report.json")
    ap.add_argument("--max-hold-min", type=int, default=_env_int("AUTO_REVAL_MAX_HOLD_MIN", 90))
    ap.add_argument("--entry-sample-limit", type=int, default=_env_int("AUTO_REVAL_ENTRY_SAMPLE_LIMIT", 1200))
    ap.add_argument("--stage4-baseline-out", default="state/stage4_liq_regret_baseline.json")
    ap.add_argument("--stage4-compare-out", default="state/stage4_liq_regret_compare.json")
    ap.add_argument("--progress-file", default="state/auto_reval_progress.json")
    ap.add_argument("--stage4-set-baseline-if-missing", type=int, default=1)
    ap.add_argument("--stage4-reset-baseline", type=int, default=0)
    ap.add_argument("--roll-baseline", type=int, default=0, help="1 to set baseline_closed=closed_total after run")
    ap.add_argument(
        "--exclude-symbols",
        default=(os.environ.get("AUTO_REVAL_EXCLUDE_SYMBOLS") or os.environ.get("RESEARCH_EXCLUDE_SYMBOLS") or ""),
        help="Comma-separated symbols to exclude from CF/reason/loss-driver reports.",
    )
    args = ap.parse_args()

    cwd = Path(__file__).resolve().parents[1]
    db_path = (cwd / args.db).resolve() if not os.path.isabs(args.db) else Path(args.db)
    baseline_path = (cwd / args.baseline_file).resolve() if not os.path.isabs(args.baseline_file) else Path(args.baseline_file)
    obs_out = (cwd / args.obs_out).resolve() if not os.path.isabs(args.obs_out) else Path(args.obs_out)
    diag_out = (cwd / args.diag_out).resolve() if not os.path.isabs(args.diag_out) else Path(args.diag_out)
    cf_out = (cwd / args.cf_out).resolve() if not os.path.isabs(args.cf_out) else Path(args.cf_out)
    reason_matrix_out = (cwd / args.reason_matrix_out).resolve() if not os.path.isabs(args.reason_matrix_out) else Path(args.reason_matrix_out)
    entry_score_driver_out = (cwd / args.entry_score_driver_out).resolve() if not os.path.isabs(args.entry_score_driver_out) else Path(args.entry_score_driver_out)
    reval_loss_driver_out = (cwd / args.reval_loss_driver_out).resolve() if not os.path.isabs(args.reval_loss_driver_out) else Path(args.reval_loss_driver_out)
    reval_loss_driver_top_csv = (cwd / args.reval_loss_driver_top_csv).resolve() if not os.path.isabs(args.reval_loss_driver_top_csv) else Path(args.reval_loss_driver_top_csv)
    reval_loss_driver_history_out = (cwd / args.reval_loss_driver_history_out).resolve() if not os.path.isabs(args.reval_loss_driver_history_out) else Path(args.reval_loss_driver_history_out)
    slippage_exit_out = (cwd / args.slippage_exit_out).resolve() if not os.path.isabs(args.slippage_exit_out) else Path(args.slippage_exit_out)
    min_notional_tune_out = (cwd / args.min_notional_tune_out).resolve() if not os.path.isabs(args.min_notional_tune_out) else Path(args.min_notional_tune_out)
    min_notional_effect_out = (cwd / args.min_notional_effectiveness_out).resolve() if not os.path.isabs(args.min_notional_effectiveness_out) else Path(args.min_notional_effectiveness_out)
    status_out = (cwd / args.status_out).resolve() if not os.path.isabs(args.status_out) else Path(args.status_out)
    stage4_baseline_out = (cwd / args.stage4_baseline_out).resolve() if not os.path.isabs(args.stage4_baseline_out) else Path(args.stage4_baseline_out)
    stage4_compare_out = (cwd / args.stage4_compare_out).resolve() if not os.path.isabs(args.stage4_compare_out) else Path(args.stage4_compare_out)
    progress_path = (cwd / args.progress_file).resolve() if not os.path.isabs(args.progress_file) else Path(args.progress_file)

    progress = _load_progress(progress_path)
    prev_status = _load_json(status_out) or {}
    prev_batch_kpi = _extract_prev_batch_kpi(prev_status, progress)
    last_ready_snapshot = (
        _extract_last_ready_snapshot(prev_status.get("last_ready"))
        if isinstance(prev_status.get("last_ready"), dict)
        else _extract_last_ready_snapshot(prev_status)
    )

    baseline_closed = _load_baseline_closed(baseline_path)
    exclude_symbols_raw = str(args.exclude_symbols or "").strip()
    start = time.time()
    closed_total = 0
    new_closed_total = 0

    status: dict[str, Any] = {
        "timestamp_ms": int(time.time() * 1000),
        "config": {
            "db": str(db_path),
            "baseline_file": str(baseline_path),
            "target_new": int(args.target_new),
            "poll_sec": float(args.poll_sec),
            "timeout_sec": float(args.timeout_sec),
            "recent": int(args.recent),
            "obs_out": str(obs_out),
                "diag_out": str(diag_out),
                "cf_out": str(cf_out),
                "reason_matrix_out": str(reason_matrix_out),
                "entry_score_driver_out": str(entry_score_driver_out),
                "reval_loss_driver_out": str(reval_loss_driver_out),
                "reval_loss_driver_top_csv": str(reval_loss_driver_top_csv),
                "reval_loss_driver_history_out": str(reval_loss_driver_history_out),
                "slippage_exit_out": str(slippage_exit_out),
                "min_notional_tune_out": str(min_notional_tune_out),
                "min_notional_effectiveness_out": str(min_notional_effect_out),
                "stage4_baseline_out": str(stage4_baseline_out),
                "capital_steps": str(args.capital_steps),
            "stage4_compare_out": str(stage4_compare_out),
            "progress_file": str(progress_path),
            "roll_baseline": bool(int(args.roll_baseline) == 1),
            "exclude_symbols": exclude_symbols_raw,
            },
        "baseline_closed": int(baseline_closed),
    }

    if not db_path.exists():
        status["error"] = f"db_not_found: {db_path}"
        status_out.parent.mkdir(parents=True, exist_ok=True)
        status_out.write_text(json.dumps(status, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 1

    while True:
        with sqlite3.connect(str(db_path)) as conn:
            closed_total = _count_closed_total(conn)
        new_closed_total = max(0, int(closed_total) - int(baseline_closed))

        if new_closed_total >= int(max(1, args.target_new)):
            break

        status.update(
            {
                "ready": False,
                "timeout": False,
                "closed_total": int(closed_total),
                "new_closed_total": int(new_closed_total),
                "new_closed_total_cum": int(max(0, _safe_int(progress.get("processed_new_closed_total"), 0) + int(new_closed_total))),
                "remaining_to_target": int(max(0, int(args.target_new) - int(new_closed_total))),
                "elapsed_sec": float(time.time() - start),
                "heartbeat_ts_ms": int(time.time() * 1000),
                "progress": dict(progress),
            }
        )
        if isinstance(last_ready_snapshot, dict):
            status["last_ready"] = dict(last_ready_snapshot)
        status_out.parent.mkdir(parents=True, exist_ok=True)
        status_out.write_text(json.dumps(status, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

        if float(args.timeout_sec) > 0 and (time.time() - start) >= float(args.timeout_sec):
            status.update(
                {
                    "ready": False,
                    "timeout": True,
                    "closed_total": int(closed_total),
                    "new_closed_total": int(new_closed_total),
                    "new_closed_total_cum": int(max(0, _safe_int(progress.get("processed_new_closed_total"), 0) + int(new_closed_total))),
                    "elapsed_sec": float(time.time() - start),
                    "progress": dict(progress),
                }
            )
            if isinstance(last_ready_snapshot, dict):
                status["last_ready"] = dict(last_ready_snapshot)
            status_out.parent.mkdir(parents=True, exist_ok=True)
            status_out.write_text(json.dumps(status, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0

        time.sleep(max(0.2, float(args.poll_sec)))

    with sqlite3.connect(str(db_path)) as conn:
        since_id = _baseline_since_id(conn, baseline_closed)
    next_batch_id = int(max(1, _safe_int(progress.get("completed_batches"), 0) + 1))

    runs = []
    runs.append(
        _run(
            [
                "python3",
                "scripts/check_trade_observability.py",
                "--db",
                str(args.db),
                "--recent",
                str(int(args.recent)),
                "--since-id",
                str(int(since_id)),
                "--out",
                str(args.obs_out),
            ],
            cwd,
        )
    )
    runs.append(
        _run(
            [
                "python3",
                "scripts/diagnose_entry_exit.py",
                "--db",
                str(args.db),
                "--since-id",
                str(int(since_id)),
                "--recent",
                str(int(args.recent)),
                "--out",
                str(args.diag_out),
            ],
            cwd,
        )
    )
    cf_cmd = [
        "python3",
        "scripts/counterfactual_replay.py",
        "--db",
        str(args.db),
        "--out",
        str(args.cf_out),
        "--max-hold-min",
        str(int(args.max_hold_min)),
        "--entry-sample-limit",
        str(int(args.entry_sample_limit)),
        "--since-id",
        str(int(since_id)),
    ]
    if exclude_symbols_raw:
        cf_cmd += ["--exclude-symbols", exclude_symbols_raw]
    runs.append(_run(cf_cmd, cwd))
    reason_matrix_cmd = [
        "python3",
        "scripts/analyze_entry_exit_reason_matrix.py",
        "--db",
        str(args.db),
        "--since-id",
        str(int(since_id)),
        "--recent-exits",
        str(int(args.recent)),
        "--max-symbols",
        str(int(args.reason_matrix_max_symbols)),
        "--max-cell-top",
        str(int(args.reason_matrix_max_cell_top)),
        "--out",
        str(reason_matrix_out),
    ]
    if exclude_symbols_raw:
        reason_matrix_cmd += ["--exclude-symbols", exclude_symbols_raw]
    runs.append(_run(reason_matrix_cmd, cwd))
    runs.append(
        _run(
            [
                "python3",
                "scripts/analyze_entry_score_drivers.py",
                "--db",
                str(args.db),
                "--since-id",
                str(int(since_id)),
                "--recent-exits",
                str(int(args.recent)),
                "--out",
                str(entry_score_driver_out),
            ],
            cwd,
        )
    )
    reval_loss_cmd = [
        "python3",
        "scripts/analyze_reval_batch_loss_drivers.py",
        "--db",
        str(args.db),
        "--since-id",
        str(int(since_id)),
        "--batch-id",
        str(int(next_batch_id)),
        "--target-new",
        str(int(args.target_new)),
        "--out",
        str(args.reval_loss_driver_out),
        "--top-csv",
        str(args.reval_loss_driver_top_csv),
        "--history-out",
        str(args.reval_loss_driver_history_out),
        "--cf-file",
        str(args.cf_out),
        "--reason-matrix-file",
        str(args.reason_matrix_out),
        "--env-file",
        "state/bybit.env",
        "--capital-steps",
        str(args.capital_steps),
    ]
    if exclude_symbols_raw:
        reval_loss_cmd += ["--exclude-symbols", exclude_symbols_raw]
    runs.append(_run(reval_loss_cmd, cwd))
    reports_ok = 0
    for r in runs:
        if not isinstance(r, dict):
            continue
        try:
            if int(r.get("code", -1)) == 0:
                reports_ok += 1
        except Exception:
            continue

    progress["completed_batches"] = int(max(0, _safe_int(progress.get("completed_batches"), 0) + 1))
    progress["completed_reports_total"] = int(max(0, _safe_int(progress.get("completed_reports_total"), 0) + int(reports_ok)))
    progress["processed_new_closed_total"] = int(max(0, _safe_int(progress.get("processed_new_closed_total"), 0) + int(new_closed_total)))
    progress["last_ready_ts_ms"] = int(time.time() * 1000)
    progress["last_ready_closed_total"] = int(max(0, int(closed_total)))
    _save_progress(progress_path, progress)

    obs = _load_json(obs_out) or {}
    diag = _load_json(diag_out) or {}
    cf = _load_json(cf_out) or {}
    reason_matrix = _load_json(reason_matrix_out) or {}
    entry_score_driver = _load_json(entry_score_driver_out) or {}
    reval_loss_driver = _load_json(reval_loss_driver_out) or {}
    batch_kpi = _extract_batch_kpi(diag, cf)
    batch_kpi_delta = _kpi_delta(batch_kpi, prev_batch_kpi)
    exit_kpi_gate = _evaluate_exit_kpi_gate(
        batch_kpi=batch_kpi,
        batch_kpi_delta=batch_kpi_delta,
        cf=cf,
        reval_loss_driver=reval_loss_driver,
    )
    progress["last_batch_kpi"] = dict(batch_kpi)
    _save_progress(progress_path, progress)
    stage4_metrics = _extract_stage4_metrics(db_path, since_id=int(since_id), cf=cf)
    slippage_exit_report = _extract_slippage_exit_report(
        db_path=db_path,
        since_id=int(since_id),
        exit_limit=int(max(20, args.slippage_exit_limit)),
        match_window_ms=int(max(5_000.0, float(args.slippage_match_window_sec) * 1000.0)),
    )
    min_notional_tune = _extract_min_notional_tuning(
        db_path=db_path,
        since_id=int(since_id),
        recent_limit=int(max(80, args.min_notional_recent_limit)),
    )
    min_notional_effect = _extract_min_notional_effectiveness(
        db_path=db_path,
        since_id=int(since_id),
        recent_limit=int(max(120, args.entry_sample_limit)),
    )
    try:
        slippage_exit_out.parent.mkdir(parents=True, exist_ok=True)
        slippage_exit_out.write_text(json.dumps(slippage_exit_report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    try:
        min_notional_tune_out.parent.mkdir(parents=True, exist_ok=True)
        min_notional_tune_out.write_text(json.dumps(min_notional_tune, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    try:
        min_notional_effect_out.parent.mkdir(parents=True, exist_ok=True)
        min_notional_effect_out.write_text(json.dumps(min_notional_effect, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    stage4_baseline = _read_stage4_baseline(stage4_baseline_out)
    stage4_compare = None
    stage4_baseline_action = "none"
    if int(args.stage4_reset_baseline) == 1:
        stage4_baseline = None
    if stage4_baseline is None and int(args.stage4_set_baseline_if_missing) == 1:
        stage4_baseline = {
            "timestamp_ms": int(time.time() * 1000),
            "baseline_closed": int(baseline_closed),
            "since_id": int(since_id),
            "metrics": stage4_metrics,
        }
        stage4_baseline_out.parent.mkdir(parents=True, exist_ok=True)
        stage4_baseline_out.write_text(json.dumps(stage4_baseline, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        stage4_baseline_action = "created"
    elif stage4_baseline is not None:
        stage4_compare = _build_stage4_compare(stage4_baseline, stage4_metrics)
        stage4_compare_out.parent.mkdir(parents=True, exist_ok=True)
        stage4_compare_out.write_text(json.dumps(stage4_compare, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        stage4_baseline_action = "reused"

    status.update(
        {
            "ready": True,
            "timeout": False,
            "closed_total": int(closed_total),
            "new_closed_total": int(new_closed_total),
            "new_closed_total_cum": int(max(0, _safe_int(progress.get("processed_new_closed_total"), 0))),
            "remaining_to_target": int(max(0, int(args.target_new) - int(new_closed_total))),
            "since_id": int(since_id),
            "elapsed_sec": float(time.time() - start),
            "heartbeat_ts_ms": int(time.time() * 1000),
            "runs": runs,
            "reports_ok": int(reports_ok),
            "progress": dict(progress),
            "summary": {
                "observability_counts": obs.get("counts"),
                "observability_linkage": obs.get("linkage"),
                "diagnosis_direction": diag.get("entry_direction"),
                "diagnosis_attribution": diag.get("attribution"),
                "counterfactual_entry": (cf.get("entry_counterfactual") or {}),
                "counterfactual_exit": (cf.get("exit_counterfactual") or {}),
                "counterfactual_tstar": (cf.get("tstar_counterfactual") or {}),
                "reason_matrix_root_cause": (reason_matrix.get("root_cause_direction_miss") or []),
                "reason_matrix_by_exit_reason_top": list((reason_matrix.get("by_exit_reason") or []))[:8],
                "entry_score_driver_top_bad": list((entry_score_driver.get("bad_driver_candidates") or []))[:8],
                "reval_loss_driver": {
                    "batch_window": (reval_loss_driver.get("batch_window") or {}),
                    "post_period_summary": (reval_loss_driver.get("post_period_summary") or {}),
                    "delta_vs_prev_same_exit_count": (reval_loss_driver.get("delta_vs_prev_same_exit_count") or {}),
                    "timing_validation": (reval_loss_driver.get("timing_validation") or {}),
                    "cumulative_compare": (reval_loss_driver.get("cumulative_compare") or {}),
                    "top_loss_symbols": list((reval_loss_driver.get("loss_concentration") or {}).get("top_loss_symbols") or [])[:8],
                    "top_loss_reasons": list((reval_loss_driver.get("loss_concentration") or {}).get("top_loss_reasons") or [])[:8],
                    "capital_scale_plan": list(reval_loss_driver.get("capital_scale_plan") or []),
                },
                "batch_kpi": batch_kpi,
                "batch_kpi_delta": batch_kpi_delta,
                "exit_kpi_gate": exit_kpi_gate,
                "stage4_metrics": stage4_metrics,
                "stage4_compare_ready": bool(stage4_compare is not None),
                "slippage_exit_200": slippage_exit_report,
                "min_notional_tuning": min_notional_tune,
                "min_notional_effectiveness": min_notional_effect,
            },
            "stage4_baseline_action": stage4_baseline_action,
            "stage4_baseline_out": str(stage4_baseline_out),
            "stage4_compare_out": str(stage4_compare_out),
        }
    )
    current_ready_snapshot = _extract_last_ready_snapshot(status)
    if isinstance(current_ready_snapshot, dict):
        status["last_ready"] = dict(current_ready_snapshot)
        last_ready_snapshot = dict(current_ready_snapshot)
    elif isinstance(last_ready_snapshot, dict):
        status["last_ready"] = dict(last_ready_snapshot)

    if int(args.roll_baseline) == 1:
        try:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_payload = {
                "baseline_closed": int(closed_total),
                "baseline_set_ts": int(time.time() * 1000),
            }
            baseline_path.write_text(json.dumps(baseline_payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            status["baseline_rolled"] = True
        except Exception as e:
            status["baseline_rolled"] = False
            status["baseline_roll_error"] = str(e)

    status_out.parent.mkdir(parents=True, exist_ok=True)
    status_out.write_text(json.dumps(status, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

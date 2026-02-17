#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _now_ms() -> int:
    return int(time.time() * 1000)


_OVERRIDE_RUNTIME_ALIAS: dict[str, list[str]] = {
    # Legacy leverage keys -> runtime-consumed keys
    "LEVERAGE_TARGET_MAX": ["MAX_LEVERAGE", "UNI_LEV_MAX"],
    "LEVERAGE_REGIME_MAX_BULL": ["UNI_LEV_MAX_TREND"],
    "LEVERAGE_REGIME_MAX_TREND": ["UNI_LEV_MAX_TREND"],
    "LEVERAGE_REGIME_MAX_BEAR": ["UNI_LEV_MAX_BEAR"],
    "LEVERAGE_REGIME_MAX_CHOP": ["UNI_LEV_MAX_CHOP"],
    "LEVERAGE_REGIME_MAX_VOLATILE": ["UNI_LEV_MAX_VOLATILE"],
    "LEVERAGE_TARGET_MAX_BULL": ["UNI_LEV_MAX_TREND"],
    "LEVERAGE_TARGET_MAX_TREND": ["UNI_LEV_MAX_TREND"],
    "LEVERAGE_TARGET_MAX_BEAR": ["UNI_LEV_MAX_BEAR"],
    "LEVERAGE_TARGET_MAX_CHOP": ["UNI_LEV_MAX_CHOP"],
    "LEVERAGE_TARGET_MAX_VOLATILE": ["UNI_LEV_MAX_VOLATILE"],
    # Legacy direction gate alias
    "ALPHA_DIRECTION_MIN_CONFIDENCE": ["ALPHA_DIRECTION_GATE_MIN_CONF"],
    # Exposure aliases
    "KELLY_TOTAL_EXPOSURE": ["MAX_NOTIONAL_EXPOSURE", "LIVE_MAX_NOTIONAL_EXPOSURE", "UNI_MAX_TOTAL_EXPOSURE"],
    "MAX_NOTIONAL_EXPOSURE": ["MAX_NOTIONAL_EXPOSURE", "UNI_MAX_TOTAL_EXPOSURE"],
    "LIVE_MAX_NOTIONAL_EXPOSURE": ["LIVE_MAX_NOTIONAL_EXPOSURE", "UNI_MAX_TOTAL_EXPOSURE"],
}


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _is_true(v: Any) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _to_env_str(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if abs(v) >= 1e6 or (0 < abs(v) < 1e-4):
            return f"{v:.10g}"
        return f"{v:.8f}".rstrip("0").rstrip(".")
    return str(v)


def _sanitize_override_value(v: Any) -> Any:
    if isinstance(v, (bool, int, float)):
        return v
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return s


def _expand_runtime_aliases(overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    out: dict[str, Any] = {}
    alias_applied: dict[str, list[str]] = {}
    source_keys: set[str] = set()
    for k, v in (overrides or {}).items():
        key = str(k or "").strip().upper()
        if not key:
            continue
        if not all(ch.isalnum() or ch == "_" for ch in key):
            continue
        sv = _sanitize_override_value(v)
        if sv is None:
            continue
        source_keys.add(key)
        out[key] = sv

    for source_key, source_val in list(out.items()):
        for target in _OVERRIDE_RUNTIME_ALIAS.get(source_key, []):
            target_key = str(target or "").strip().upper()
            if not target_key:
                continue
            if target_key in source_keys:
                continue
            if target_key not in out:
                out[target_key] = source_val
                alias_applied.setdefault(source_key, []).append(target_key)
    return out, alias_applied


def _pick_metric(status: dict[str, Any], path: list[str]) -> Any:
    cur: Any = status
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _get_effective_float(
    key: str,
    overrides: dict[str, Any],
    default: float,
) -> float:
    if key in overrides:
        v = _safe_float(overrides.get(key), None)
        if v is not None:
            return float(v)
    v = _safe_float(os.environ.get(key), None)
    if v is not None:
        return float(v)
    return float(default)


def _get_effective_int(
    key: str,
    overrides: dict[str, Any],
    default: int,
) -> int:
    if key in overrides:
        return _safe_int(overrides.get(key), default)
    return _safe_int(os.environ.get(key), default)


def _parse_float_list(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, list):
        for it in value:
            fv = _safe_float(it, None)
            if fv is not None:
                out.append(float(fv))
        return out
    txt = str(value or "").strip()
    if not txt:
        return out
    for tok in txt.split(","):
        s = tok.strip()
        if not s:
            continue
        fv = _safe_float(s, None)
        if fv is not None:
            out.append(float(fv))
    return out


def _get_effective_float_list(
    key: str,
    overrides: dict[str, Any],
    default: list[float],
    expected_len: int | None = None,
) -> list[float]:
    vals = _parse_float_list(overrides.get(key))
    if not vals:
        vals = _parse_float_list(os.environ.get(key, ""))
    if not vals:
        vals = [float(x) for x in default]
    if expected_len is not None and expected_len > 0:
        exp = int(expected_len)
        if len(vals) < exp:
            vals = list(vals) + [float(default[min(i, len(default) - 1)]) for i in range(len(vals), exp)]
        elif len(vals) > exp:
            vals = vals[:exp]
    return [float(x) for x in vals]


def _float_list_to_env(vals: list[float]) -> str:
    out = []
    for v in vals:
        out.append(_to_env_str(float(v)))
    return ",".join(out)


def _normalize_sorted_unique_floats(vals: list[float], *, min_v: float = 0.0) -> list[float]:
    out: list[float] = []
    prev = None
    for x in sorted(float(max(min_v, v)) for v in vals):
        if prev is None or abs(float(x) - float(prev)) > 1e-12:
            out.append(float(x))
            prev = float(x)
    return out


def _expand_float_list(vals: list[float], expected_len: int, fallback: list[float]) -> list[float]:
    expected = int(max(1, expected_len))
    base = [float(x) for x in vals] if vals else [float(x) for x in fallback]
    if not base:
        base = [0.0]
    if len(base) < expected:
        base.extend([float(base[-1])] * (expected - len(base)))
    elif len(base) > expected:
        base = base[:expected]
    return [float(x) for x in base]


def _read_capital_balance_usdt(repo_root: Path, overrides: dict[str, Any]) -> float | None:
    # Priority: explicit override/env -> live balance snapshot -> model balance snapshot.
    for key in ("AUTO_TUNE_CAPITAL_USDT",):
        if key in overrides:
            v = _safe_float(overrides.get(key), None)
            if v is not None and v > 0:
                return float(v)
    v = _safe_float(os.environ.get("AUTO_TUNE_CAPITAL_USDT"), None)
    if v is not None and v > 0:
        return float(v)

    src = str(os.environ.get("AUTO_TUNE_CAPITAL_SOURCE", "live")).strip().lower()
    candidates: list[Path] = []
    if src in ("live", "live_balance", "auto"):
        candidates.append((repo_root / "state/balance_live.json").resolve())
    if src in ("model", "paper", "auto", "live", "live_balance"):
        candidates.append((repo_root / "state/balance.json").resolve())
    for p in candidates:
        try:
            if not p.exists():
                continue
            obj = json.loads(p.read_text(encoding="utf-8"))
            val = None
            if isinstance(obj, (int, float, str)):
                val = _safe_float(obj, None)
            elif isinstance(obj, dict):
                for k in ("balance", "wallet_balance", "equity", "total_equity", "value"):
                    vv = _safe_float(obj.get(k), None)
                    if vv is not None:
                        val = vv
                        break
            if val is not None and float(val) > 0:
                return float(val)
        except Exception:
            continue
    return None


def _set_float_step(
    out: dict[str, Any],
    key: str,
    cur: float,
    delta: float,
    min_v: float | None = None,
    max_v: float | None = None,
) -> bool:
    nxt = float(cur + delta)
    if min_v is not None:
        nxt = max(float(min_v), nxt)
    if max_v is not None:
        nxt = min(float(max_v), nxt)
    if abs(nxt - float(cur)) < 1e-12:
        return False
    out[key] = float(nxt)
    return True


def _set_int_step(
    out: dict[str, Any],
    key: str,
    cur: int,
    delta: int,
    min_v: int | None = None,
    max_v: int | None = None,
) -> bool:
    nxt = int(cur + delta)
    if min_v is not None:
        nxt = max(int(min_v), nxt)
    if max_v is not None:
        nxt = min(int(max_v), nxt)
    if nxt == int(cur):
        return False
    out[key] = int(nxt)
    return True


def _extract_metrics(
    status: dict[str, Any],
    diag: dict[str, Any],
    cf: dict[str, Any],
    stage4_compare: dict[str, Any],
    last_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status_kpi = _pick_metric(status, ["summary", "batch_kpi"]) if isinstance(status, dict) else {}
    status_kpi = status_kpi if isinstance(status_kpi, dict) else {}
    status_kpi_delta = _pick_metric(status, ["summary", "batch_kpi_delta"]) if isinstance(status, dict) else {}
    status_kpi_delta = status_kpi_delta if isinstance(status_kpi_delta, dict) else {}
    entry_issue_rate = _safe_float(_pick_metric(diag, ["attribution", "entry_issue_rate"]), None)
    entry_issue_ratio = _safe_float(status_kpi.get("entry_issue_ratio"), entry_issue_rate)
    miss_rate = _safe_float(_pick_metric(diag, ["entry_direction", "miss_rate"]), None)
    direction_hit = _safe_float(status_kpi.get("direction_hit"), None)
    if direction_hit is None and miss_rate is not None:
        direction_hit = float(max(0.0, min(1.0, 1.0 - float(miss_rate))))
    avg_exit_regret = _safe_float(_pick_metric(cf, ["exit_counterfactual", "avg_exit_regret"]), None)
    if avg_exit_regret is None:
        avg_exit_regret = _safe_float(status_kpi.get("avg_exit_regret"), None)
    early_like_rate = _safe_float(_pick_metric(cf, ["exit_counterfactual", "early_like_rate"]), None)
    improvable_rate = _safe_float(_pick_metric(cf, ["exit_counterfactual", "improvable_rate_regret_gt_10bps"]), None)
    event_mc_exit_rate = _safe_float(_pick_metric(stage4_compare, ["current", "metrics", "event_mc_exit_rate"]), None)
    liq_like_rate = _safe_float(_pick_metric(stage4_compare, ["current", "metrics", "liq_like_rate"]), None)
    avg_roe = _safe_float(_pick_metric(stage4_compare, ["current", "metrics", "avg_roe"]), None)
    new_closed_total = _safe_int(status.get("new_closed_total"), 0)
    target_new = _safe_int(_pick_metric(status, ["config", "target_new"]), 0)
    completed_batches = _safe_int(_pick_metric(status, ["progress", "completed_batches"]), 0)
    prev_metrics = (last_state or {}).get("last_metrics") if isinstance((last_state or {}).get("last_metrics"), dict) else {}
    prev_direction_hit = _safe_float(prev_metrics.get("direction_hit"), None)
    prev_entry_issue_ratio = _safe_float(prev_metrics.get("entry_issue_ratio"), None)
    prev_avg_exit_regret = _safe_float(prev_metrics.get("avg_exit_regret"), None)
    direction_hit_delta = _safe_float(status_kpi_delta.get("direction_hit"), None)
    entry_issue_ratio_delta = _safe_float(status_kpi_delta.get("entry_issue_ratio"), None)
    avg_exit_regret_delta = _safe_float(status_kpi_delta.get("avg_exit_regret"), None)
    if direction_hit_delta is None and direction_hit is not None and prev_direction_hit is not None:
        direction_hit_delta = float(direction_hit) - float(prev_direction_hit)
    if entry_issue_ratio_delta is None and entry_issue_ratio is not None and prev_entry_issue_ratio is not None:
        entry_issue_ratio_delta = float(entry_issue_ratio) - float(prev_entry_issue_ratio)
    if avg_exit_regret_delta is None and avg_exit_regret is not None and prev_avg_exit_regret is not None:
        avg_exit_regret_delta = float(avg_exit_regret) - float(prev_avg_exit_regret)

    return {
        "entry_issue_rate": entry_issue_rate,
        "entry_issue_ratio": entry_issue_ratio,
        "miss_rate": miss_rate,
        "direction_hit": direction_hit,
        "avg_exit_regret": avg_exit_regret,
        "early_like_rate": early_like_rate,
        "improvable_rate_regret_gt_10bps": improvable_rate,
        "event_mc_exit_rate": event_mc_exit_rate,
        "liq_like_rate": liq_like_rate,
        "avg_roe": avg_roe,
        "direction_hit_delta": direction_hit_delta,
        "entry_issue_ratio_delta": entry_issue_ratio_delta,
        "avg_exit_regret_delta": avg_exit_regret_delta,
        "new_closed_total": int(new_closed_total),
        "target_new": int(target_new),
        "completed_batches": int(completed_batches),
    }


def _extract_reason_matrix_metrics(reason_matrix: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "coverage_eval_n": 0,
        "cause_share_entry_direction_mismatch": None,
        "cause_share_exit_timing_regret": None,
        "cause_share_exit_timing_or_cost_flip": None,
        "cause_share_mixed_or_noise": None,
        "cause_n_entry_direction_mismatch": 0,
        "cause_n_exit_timing_regret": 0,
        "cause_n_exit_timing_or_cost_flip": 0,
        "cause_n_mixed_or_noise": 0,
        "event_mc_exit": {},
        "unified_flip": {},
        "ev_drop": {},
        "hold_vs_exit": {},
        "external_sync": {},
    }
    if not isinstance(reason_matrix, dict):
        return out

    cov = reason_matrix.get("coverage") if isinstance(reason_matrix.get("coverage"), dict) else {}
    out["coverage_eval_n"] = int(_safe_int(cov.get("direction_miss_cause_eval_n"), 0))

    root = reason_matrix.get("root_cause_direction_miss") if isinstance(reason_matrix.get("root_cause_direction_miss"), list) else []
    by_exit = reason_matrix.get("by_exit_reason") if isinstance(reason_matrix.get("by_exit_reason"), list) else []
    if not root and not by_exit:
        return out

    total_n = 0
    cause_map: dict[str, dict[str, Any]] = {}
    for row in root:
        if not isinstance(row, dict):
            continue
        c = str(row.get("cause") or "").strip()
        if not c:
            continue
        n = int(_safe_int(row.get("n"), 0))
        total_n += n
        share = _safe_float(row.get("share_over_direction_miss"), None)
        cause_map[c] = {"n": n, "share": share}
    if total_n > 0:
        for c, v in cause_map.items():
            if v.get("share") is None:
                v["share"] = float(v.get("n", 0) / max(1, total_n))

    def _cause_share(name: str) -> float | None:
        v = cause_map.get(name)
        if not isinstance(v, dict):
            return None
        return _safe_float(v.get("share"), None)

    def _cause_n(name: str) -> int:
        v = cause_map.get(name)
        if not isinstance(v, dict):
            return 0
        return int(_safe_int(v.get("n"), 0))

    def _pick_exit_row(token: str) -> dict[str, Any]:
        best: dict[str, Any] = {}
        best_n = -1
        needle = str(token).strip().lower()
        for row in by_exit:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("exit_reason") or "").strip().lower()
            if needle not in reason:
                continue
            n = int(_safe_int(row.get("n"), 0))
            if n > best_n:
                best = dict(row)
                best_n = n
        if not best:
            return {}
        return {
            "exit_reason": str(best.get("exit_reason") or ""),
            "n": int(_safe_int(best.get("n"), 0)),
            "avg_roe": _safe_float(best.get("avg_roe"), None),
            "direction_miss_rate": _safe_float(best.get("direction_miss_rate"), None),
            "opp_side_better_rate": _safe_float(best.get("opp_side_better_rate"), None),
            "profitable_side_loss_rate": _safe_float(best.get("profitable_side_loss_rate"), None),
            "early_like_rate": _safe_float(best.get("early_like_rate"), None),
            "avg_exit_regret": _safe_float(best.get("avg_exit_regret"), None),
        }

    out["available"] = True
    out["cause_share_entry_direction_mismatch"] = _cause_share("entry_direction_mismatch")
    out["cause_share_exit_timing_regret"] = _cause_share("exit_timing_regret")
    out["cause_share_exit_timing_or_cost_flip"] = _cause_share("exit_timing_or_cost_flip")
    out["cause_share_mixed_or_noise"] = _cause_share("mixed_or_noise")
    out["cause_n_entry_direction_mismatch"] = _cause_n("entry_direction_mismatch")
    out["cause_n_exit_timing_regret"] = _cause_n("exit_timing_regret")
    out["cause_n_exit_timing_or_cost_flip"] = _cause_n("exit_timing_or_cost_flip")
    out["cause_n_mixed_or_noise"] = _cause_n("mixed_or_noise")
    out["event_mc_exit"] = _pick_exit_row("event_mc_exit")
    out["unified_flip"] = _pick_exit_row("unified_flip")
    out["ev_drop"] = _pick_exit_row("ev_drop")
    out["hold_vs_exit"] = _pick_exit_row("hold_vs_exit")
    out["external_sync"] = _pick_exit_row("external_sync")
    if not out["external_sync"]:
        out["external_sync"] = _pick_exit_row("exchange_close_external_sync")
    return out


def _extract_entry_score_driver_metrics(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "matched_pairs": 0,
        "win_rate": None,
        "avg_roe": None,
        "direction_miss_rate": None,
        "top_bad_feature": None,
        "top_bad_score": None,
        "pred_mu_dir_conf_bad_score": None,
        "entry_quality_bad_score": None,
        "entry_ev_bad_score": None,
        "event_exit_score_bad_score": None,
        "leverage_signal_bad_score": None,
    }
    if not isinstance(report, dict):
        return out
    cov = report.get("coverage") if isinstance(report.get("coverage"), dict) else {}
    agg = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    bad = report.get("bad_driver_candidates") if isinstance(report.get("bad_driver_candidates"), list) else []
    out["matched_pairs"] = int(_safe_int(cov.get("matched_pairs"), 0))
    out["win_rate"] = _safe_float(agg.get("win_rate"), None)
    out["avg_roe"] = _safe_float(agg.get("avg_roe"), None)
    out["direction_miss_rate"] = _safe_float(agg.get("direction_miss_rate"), None)
    out["available"] = bool(int(out["matched_pairs"]) > 0 and bool(bad))
    if not bad:
        return out

    top = None
    top_score = -1.0
    for row in bad:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or "").strip().lower()
        score = _safe_float(row.get("bad_driver_score"), None)
        if score is None:
            continue
        if float(score) > float(top_score):
            top_score = float(score)
            top = feat
        if feat == "pred_mu_dir_conf":
            out["pred_mu_dir_conf_bad_score"] = float(score)
        elif feat == "entry_quality_score":
            out["entry_quality_bad_score"] = float(score)
        elif feat in ("entry_ev", "entry_ev_excess"):
            prev = _safe_float(out.get("entry_ev_bad_score"), None)
            if prev is None or float(score) > float(prev):
                out["entry_ev_bad_score"] = float(score)
        elif feat == "event_exit_score":
            out["event_exit_score_bad_score"] = float(score)
        elif feat == "leverage_signal_score":
            out["leverage_signal_bad_score"] = float(score)
    if top is not None:
        out["top_bad_feature"] = str(top)
        out["top_bad_score"] = float(top_score)
    return out


def _extract_reval_loss_driver_metrics(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "timing_issue_validated": False,
        "avg_exit_regret": None,
        "early_like_rate": None,
        "window_exit_n": 0,
        "history_snapshots": 0,
        "exit_regret_by_reason": {},
        "exit_regret_by_reason_delta": {},
        "by_reason": {
            "unified_flip": {},
            "hold_vs_exit": {},
            "exchange_close_external_sync": {},
        },
    }
    if not isinstance(report, dict):
        return out
    tv = report.get("timing_validation") if isinstance(report.get("timing_validation"), dict) else {}
    by_reason = tv.get("by_reason") if isinstance(tv.get("by_reason"), dict) else {}
    batch_window = report.get("batch_window") if isinstance(report.get("batch_window"), dict) else {}
    cumulative = report.get("cumulative_compare") if isinstance(report.get("cumulative_compare"), dict) else {}
    delta_prev = cumulative.get("delta_vs_prev_snapshot") if isinstance(cumulative.get("delta_vs_prev_snapshot"), dict) else {}
    out["window_exit_n"] = int(_safe_int(batch_window.get("window_exit_n"), 0))
    out["history_snapshots"] = int(_safe_int(cumulative.get("total_snapshots"), 0))
    out["timing_issue_validated"] = bool(tv.get("timing_issue_validated") is True)
    out["avg_exit_regret"] = _safe_float(tv.get("avg_exit_regret"), None)
    out["early_like_rate"] = _safe_float(tv.get("early_like_rate"), None)
    out["exit_regret_by_reason"] = tv.get("exit_regret_by_reason") if isinstance(tv.get("exit_regret_by_reason"), dict) else {}
    out["exit_regret_by_reason_delta"] = (
        delta_prev.get("exit_regret_by_reason_delta") if isinstance(delta_prev.get("exit_regret_by_reason_delta"), dict) else {}
    )
    for axis in ("unified_flip", "hold_vs_exit", "exchange_close_external_sync"):
        ar = by_reason.get(axis) if isinstance(by_reason.get(axis), dict) else {}
        out["by_reason"][axis] = {
            "timing_issue": bool(ar.get("timing_issue") is True),
            "timing_gap_ret": _safe_float(ar.get("timing_gap_ret"), None),
            "window_n": int(_safe_int(ar.get("window_n"), 0)),
            "window_avg_roe": _safe_float(ar.get("window_avg_roe"), None),
            "window_loss_rate": _safe_float(ar.get("window_loss_rate"), None),
            "window_avg_hold_sec": _safe_float(ar.get("window_avg_hold_sec"), None),
            "recommended_hold_increase_ratio": _safe_float(ar.get("recommended_hold_increase_ratio"), None),
            "recommended_hold_sec": _safe_float(ar.get("recommended_hold_sec"), None),
            "recommended_param_deltas": ar.get("recommended_param_deltas") if isinstance(ar.get("recommended_param_deltas"), dict) else {},
        }
    out["available"] = bool(tv or batch_window or cumulative)
    return out


def _normalize_regime(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if ("trend" in s) or ("bull" in s) or ("bear" in s):
        return "trend"
    if ("volatile" in s) or ("random" in s) or ("noise" in s):
        return "volatile"
    if ("mean" in s) or ("revert" in s) or ("chop" in s) or ("range" in s):
        return "chop"
    return "chop"


def _extract_regime_metrics(db_path: Path, since_id: int) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {
        "trend": {
            "exit_n": 0,
            "loss_n": 0,
            "event_n": 0,
            "dd_n": 0,
            "liq_n": 0,
            "sum_roe": 0.0,
            "roe_n": 0,
            "entry_n": 0,
            "reject_n": 0,
            "reject_110007_n": 0,
            "precision_reject_n": 0,
        },
        "chop": {
            "exit_n": 0,
            "loss_n": 0,
            "event_n": 0,
            "dd_n": 0,
            "liq_n": 0,
            "sum_roe": 0.0,
            "roe_n": 0,
            "entry_n": 0,
            "reject_n": 0,
            "reject_110007_n": 0,
            "precision_reject_n": 0,
        },
        "volatile": {
            "exit_n": 0,
            "loss_n": 0,
            "event_n": 0,
            "dd_n": 0,
            "liq_n": 0,
            "sum_roe": 0.0,
            "roe_n": 0,
            "entry_n": 0,
            "reject_n": 0,
            "reject_110007_n": 0,
            "precision_reject_n": 0,
        },
    }
    if not db_path.exists():
        return out
    q = (
        "SELECT regime, entry_reason, roe FROM trades "
        "WHERE id > ? AND action IN ('EXIT','REBAL_EXIT')"
    )
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(q, (int(max(0, since_id)),)).fetchall()
    except Exception:
        return out
    for regime_raw, reason_raw, roe_raw in rows:
        reg = _normalize_regime(regime_raw)
        rtxt = str(reason_raw or "").strip().lower()
        m = out.get(reg)
        if not isinstance(m, dict):
            continue
        m["exit_n"] = int(m.get("exit_n", 0) or 0) + 1
        rr = _safe_float(roe_raw, None)
        if rr is not None:
            m["sum_roe"] = float(m.get("sum_roe", 0.0) or 0.0) + float(rr)
            m["roe_n"] = int(m.get("roe_n", 0) or 0) + 1
            if float(rr) < 0.0:
                m["loss_n"] = int(m.get("loss_n", 0) or 0) + 1
        if "event_mc_exit" in rtxt:
            m["event_n"] = int(m.get("event_n", 0) or 0) + 1
        if "unrealized_dd" in rtxt:
            m["dd_n"] = int(m.get("dd_n", 0) or 0) + 1
        if ("liquidat" in rtxt) or ("kill" in rtxt) or ("emergency" in rtxt):
            m["liq_n"] = int(m.get("liq_n", 0) or 0) + 1
    q_rejects = (
        "SELECT regime, action, entry_reason FROM trades "
        "WHERE id > ? AND action IN ('ENTER','ORDER_REJECT')"
    )
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows2 = conn.execute(q_rejects, (int(max(0, since_id)),)).fetchall()
    except Exception:
        rows2 = []
    for regime_raw, action_raw, reason_raw in rows2:
        reg = _normalize_regime(regime_raw)
        a = str(action_raw or "").strip().upper()
        reason = str(reason_raw or "").strip().lower()
        m = out.get(reg)
        if not isinstance(m, dict):
            continue
        if a == "ENTER":
            m["entry_n"] = int(m.get("entry_n", 0) or 0) + 1
            continue
        if a != "ORDER_REJECT":
            continue
        m["reject_n"] = int(m.get("reject_n", 0) or 0) + 1
        if ("110007" in reason) or ("ab not enough" in reason):
            m["reject_110007_n"] = int(m.get("reject_110007_n", 0) or 0) + 1
        if ("minimum amount precision" in reason) or ("min qty" in reason) or ("min_qty" in reason):
            m["precision_reject_n"] = int(m.get("precision_reject_n", 0) or 0) + 1
    # ── regime별 direction_hit (방향 적중률) ──────────────────────
    q_dir = (
        "SELECT regime, side, roe FROM trades "
        "WHERE id > ? AND action IN ('EXIT','REBAL_EXIT') AND roe IS NOT NULL"
    )
    try:
        with sqlite3.connect(str(db_path)) as conn:
            dir_rows = conn.execute(q_dir, (int(max(0, since_id)),)).fetchall()
    except Exception:
        dir_rows = []
    for regime_raw, side_raw, roe_raw in dir_rows:
        reg = _normalize_regime(regime_raw)
        m = out.get(reg)
        if not isinstance(m, dict):
            continue
        rr = _safe_float(roe_raw, None)
        if rr is None:
            continue
        m["dir_total_n"] = int(m.get("dir_total_n", 0) or 0) + 1
        # direction hit = roe > 0 이면 방향 적중
        if float(rr) > 0:
            m["dir_hit_n"] = int(m.get("dir_hit_n", 0) or 0) + 1
        # regime 소급 정확도: trend에서 진입한 거래가 실제로 추세 방향으로 이익을 냈는지
        side = str(side_raw or "").strip().upper()
        if reg == "trend":
            # trend 분류가 맞았으면 방향과 roe가 일치해야 함
            if (side == "LONG" and float(rr) > 0) or (side == "SHORT" and float(rr) > 0):
                m["regime_correct_n"] = int(m.get("regime_correct_n", 0) or 0) + 1
            m["regime_eval_n"] = int(m.get("regime_eval_n", 0) or 0) + 1
        elif reg == "chop":
            # chop에서는 방향 무관하게 손실이 자연스러움 → 손실 비율이 높을수록 분류 정확
            m["regime_eval_n"] = int(m.get("regime_eval_n", 0) or 0) + 1
            # chop 정확도: |roe| < 0.5% (작은 움직임) = chop 분류 맞음
            if abs(float(rr)) < 0.005:
                m["regime_correct_n"] = int(m.get("regime_correct_n", 0) or 0) + 1
        elif reg == "volatile":
            # volatile에서는 |roe|가 클수록 분류 정확
            m["regime_eval_n"] = int(m.get("regime_eval_n", 0) or 0) + 1
            if abs(float(rr)) > 0.005:
                m["regime_correct_n"] = int(m.get("regime_correct_n", 0) or 0) + 1

    for reg in ("trend", "chop", "volatile"):
        m = out.get(reg, {})
        n = int(m.get("exit_n", 0) or 0)
        rn = int(m.get("roe_n", 0) or 0)
        en = int(m.get("entry_n", 0) or 0)
        rj = int(m.get("reject_n", 0) or 0)
        m["loss_rate"] = float((m.get("loss_n", 0) or 0) / n) if n > 0 else None
        m["event_rate"] = float((m.get("event_n", 0) or 0) / n) if n > 0 else None
        m["dd_rate"] = float((m.get("dd_n", 0) or 0) / n) if n > 0 else None
        m["liq_rate"] = float((m.get("liq_n", 0) or 0) / n) if n > 0 else None
        m["avg_roe"] = float((m.get("sum_roe", 0.0) or 0.0) / rn) if rn > 0 else None
        m["reject_rate"] = float(rj / max(1, en + rj))
        m["reject_110007_rate"] = float((m.get("reject_110007_n", 0) or 0) / max(1, rj))
        m["precision_reject_rate"] = float((m.get("precision_reject_n", 0) or 0) / max(1, rj))
        # regime별 방향 적중률
        dir_n = int(m.get("dir_total_n", 0) or 0)
        m["direction_hit"] = float((m.get("dir_hit_n", 0) or 0) / dir_n) if dir_n > 0 else None
        # regime 소급 정확도 (regime 분류가 실제 시장 행동과 일치했는지)
        regime_eval = int(m.get("regime_eval_n", 0) or 0)
        m["regime_accuracy"] = float((m.get("regime_correct_n", 0) or 0) / regime_eval) if regime_eval > 0 else None
        out[reg] = m
    return out


def _extract_recent_reason_metrics(
    db_path: Path,
    *,
    since_id: int,
    limit_per_reason: int = 200,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "limit_per_reason": int(max(20, limit_per_reason)),
        "event_mc_exit": {"n": 0, "avg_roe": None, "loss_rate": None, "source": "none"},
        "ev_drop_exit": {"n": 0, "avg_roe": None, "loss_rate": None, "source": "none"},
        "unified_flip": {"n": 0, "avg_roe": None, "loss_rate": None, "source": "none"},
        "hold_vs_exit": {"n": 0, "avg_roe": None, "loss_rate": None, "source": "none"},
        "external_sync": {"n": 0, "avg_roe": None, "loss_rate": None, "source": "none"},
    }
    if not db_path.exists():
        return out

    def _query_stats(token: str | list[str] | tuple[str, ...]) -> dict[str, Any]:
        tokens: list[str]
        if isinstance(token, (list, tuple)):
            tokens = [str(t).strip().lower() for t in token if str(t).strip()]
        else:
            tokens = [str(token).strip().lower()]
        if not tokens:
            return {"n": 0, "avg_roe": None, "loss_rate": None, "source": "none"}
        lim = int(max(20, limit_per_reason))
        q_base = (
            "SELECT roe FROM trades "
            "WHERE action IN ('EXIT','REBAL_EXIT') "
            "  AND ({cond}) "
            "  AND id > ? "
            "ORDER BY id DESC "
            "LIMIT ?"
        )
        q_all_base = (
            "SELECT roe FROM trades "
            "WHERE action IN ('EXIT','REBAL_EXIT') "
            "  AND ({cond}) "
            "ORDER BY id DESC "
            "LIMIT ?"
        )
        cond = " OR ".join(["lower(entry_reason) LIKE ?"] * len(tokens))
        q = q_base.format(cond=cond)
        q_all = q_all_base.format(cond=cond)
        rows: list[tuple[Any, ...]] = []
        src = "since_id"
        try:
            with sqlite3.connect(str(db_path)) as conn:
                params = tuple([f"%{t}%" for t in tokens] + [int(max(0, since_id)), int(lim)])
                rows = conn.execute(q, params).fetchall()
                if len(rows) < max(40, int(0.25 * lim)):
                    params_all = tuple([f"%{t}%" for t in tokens] + [int(lim)])
                    rows = conn.execute(q_all, params_all).fetchall()
                    src = "global_recent"
        except Exception:
            rows = []
            src = "error"
        vals: list[float] = []
        for (roe_raw,) in rows:
            rr = _safe_float(roe_raw, None)
            if rr is None:
                continue
            vals.append(float(rr))
        n = int(len(vals))
        if n <= 0:
            return {"n": 0, "avg_roe": None, "loss_rate": None, "source": src}
        loss_n = int(sum(1 for v in vals if float(v) < 0.0))
        return {
            "n": int(n),
            "avg_roe": float(sum(vals) / max(1, n)),
            "loss_rate": float(loss_n / max(1, n)),
            "source": src,
        }

    out["event_mc_exit"] = _query_stats("event_mc_exit")
    out["ev_drop_exit"] = _query_stats("ev_drop_exit")
    out["unified_flip"] = _query_stats("unified_flip")
    out["hold_vs_exit"] = _query_stats("hold_vs_exit")
    out["external_sync"] = _query_stats(["exchange_close_external_sync", "external_sync"])
    out["available"] = bool(
        int(out.get("event_mc_exit", {}).get("n", 0) or 0) > 0
        or int(out.get("ev_drop_exit", {}).get("n", 0) or 0) > 0
        or int(out.get("unified_flip", {}).get("n", 0) or 0) > 0
        or int(out.get("hold_vs_exit", {}).get("n", 0) or 0) > 0
        or int(out.get("external_sync", {}).get("n", 0) or 0) > 0
    )
    return out


def _run_retrain(
    repo_root: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    cmd = [sys.executable, "scripts/train_alpha_weights.py"]
    env = os.environ.copy()
    # Keep runtime retraining calibration-on, even if parent shell omitted these vars.
    env.setdefault("ALPHA_DIRECTION_CALIBRATION_ENABLED", "1")
    env.setdefault("ALPHA_DIRECTION_CALIBRATION_MIN_SAMPLES", "320")
    env.setdefault("ALPHA_DIRECTION_CALIBRATION_MIN_IMPROVE", "0.0")
    env.setdefault("ALPHA_DIRECTION_LABEL_SOURCE", "auto")
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(30.0, float(timeout_sec)),
        )
        return {
            "ran": True,
            "rc": int(proc.returncode),
            "elapsed_sec": float(time.time() - start),
            "stdout_tail": (proc.stdout or "")[-3000:],
            "stderr_tail": (proc.stderr or "")[-3000:],
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ran": True,
            "rc": -9,
            "elapsed_sec": float(time.time() - start),
            "stdout_tail": str((e.stdout or "")[-3000:]),
            "stderr_tail": str((e.stderr or "")[-3000:]),
            "error": "timeout",
        }
    except Exception as e:
        return {
            "ran": True,
            "rc": -1,
            "elapsed_sec": float(time.time() - start),
            "stdout_tail": "",
            "stderr_tail": str(e),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reval-driven auto tuning policy updater")
    ap.add_argument("--status-in", default="state/auto_reval_db_report.json")
    ap.add_argument("--diag-in", default="state/entry_exit_diagnosis_report.json")
    ap.add_argument("--cf-in", default="state/counterfactual_replay_report_new_window.json")
    ap.add_argument("--reason-matrix-in", default="state/entry_exit_reason_matrix_report.json")
    ap.add_argument("--entry-score-driver-in", default="state/entry_score_driver_report.json")
    ap.add_argument("--reval-loss-driver-in", default="state/reval_loss_driver_report.json")
    ap.add_argument("--stage4-compare-in", default="state/stage4_liq_regret_compare.json")
    ap.add_argument("--override-in", default="state/auto_tune_overrides.json")
    ap.add_argument("--override-out", default="state/auto_tune_overrides.json")
    ap.add_argument("--state-file", default="state/auto_tune_state.json")
    ap.add_argument("--status-out", default="state/auto_tune_status.json")
    ap.add_argument("--run-retrain", type=int, default=1)
    ap.add_argument("--retrain-timeout-sec", type=float, default=240.0)
    ap.add_argument("--min-entry-issue", type=float, default=0.50)
    ap.add_argument("--min-miss-rate", type=float, default=0.52)
    ap.add_argument("--high-exit-regret", type=float, default=0.015)
    ap.add_argument("--high-early-like", type=float, default=0.55)
    ap.add_argument("--high-event-mc-rate", type=float, default=0.30)
    ap.add_argument("--high-liq-rate", type=float, default=0.03)
    ap.add_argument("--min-regime-samples", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    status_path = (repo_root / args.status_in).resolve() if not os.path.isabs(args.status_in) else Path(args.status_in)
    diag_path = (repo_root / args.diag_in).resolve() if not os.path.isabs(args.diag_in) else Path(args.diag_in)
    cf_path = (repo_root / args.cf_in).resolve() if not os.path.isabs(args.cf_in) else Path(args.cf_in)
    reason_matrix_path = (repo_root / args.reason_matrix_in).resolve() if not os.path.isabs(args.reason_matrix_in) else Path(args.reason_matrix_in)
    entry_score_driver_path = (repo_root / args.entry_score_driver_in).resolve() if not os.path.isabs(args.entry_score_driver_in) else Path(args.entry_score_driver_in)
    reval_loss_driver_path = (repo_root / args.reval_loss_driver_in).resolve() if not os.path.isabs(args.reval_loss_driver_in) else Path(args.reval_loss_driver_in)
    stage4_path = (repo_root / args.stage4_compare_in).resolve() if not os.path.isabs(args.stage4_compare_in) else Path(args.stage4_compare_in)
    override_in_path = (repo_root / args.override_in).resolve() if not os.path.isabs(args.override_in) else Path(args.override_in)
    override_out_path = (repo_root / args.override_out).resolve() if not os.path.isabs(args.override_out) else Path(args.override_out)
    state_file_path = (repo_root / args.state_file).resolve() if not os.path.isabs(args.state_file) else Path(args.state_file)
    status_out_path = (repo_root / args.status_out).resolve() if not os.path.isabs(args.status_out) else Path(args.status_out)

    status = _load_json(status_path)
    diag = _load_json(diag_path)
    cf = _load_json(cf_path)
    reason_matrix = _load_json(reason_matrix_path)
    entry_score_driver = _load_json(entry_score_driver_path)
    reval_loss_driver = _load_json(reval_loss_driver_path)
    stage4 = _load_json(stage4_path)
    existing_override_payload = _load_json(override_in_path)
    existing_overrides_raw = (
        existing_override_payload.get("overrides")
        if isinstance(existing_override_payload.get("overrides"), dict)
        else {}
    )
    existing_overrides, existing_aliases = _expand_runtime_aliases(existing_overrides_raw)
    last_state = _load_json(state_file_path)

    ready = bool(status.get("ready") is True)
    metrics = _extract_metrics(status, diag, cf, stage4, last_state=last_state)
    reason_matrix_metrics = _extract_reason_matrix_metrics(reason_matrix)
    entry_score_driver_metrics = _extract_entry_score_driver_metrics(entry_score_driver)
    reval_loss_driver_metrics = _extract_reval_loss_driver_metrics(reval_loss_driver)
    status_cfg = status.get("config") if isinstance(status.get("config"), dict) else {}
    db_cfg = status_cfg.get("db")
    if db_cfg is None:
        db_path = (repo_root / "state/bot_data_live.db").resolve()
    else:
        db_path = (repo_root / str(db_cfg)).resolve() if not os.path.isabs(str(db_cfg)) else Path(str(db_cfg))
    since_id = _safe_int(status.get("since_id"), 0)
    regime_metrics = _extract_regime_metrics(db_path=db_path, since_id=since_id)
    reason_recent_limit = _safe_int(os.environ.get("AUTO_TUNE_REASON_RECENT_LIMIT"), 0)
    if reason_recent_limit <= 0:
        reason_recent_limit = int(max(80, min(200, _safe_int(metrics.get("target_new"), 200))))
    reason_recent_metrics = _extract_recent_reason_metrics(
        db_path=db_path,
        since_id=since_id,
        limit_per_reason=int(reason_recent_limit),
    )
    status_summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
    exit_kpi_gate = status_summary.get("exit_kpi_gate") if isinstance(status_summary.get("exit_kpi_gate"), dict) else {}
    exit_kpi_grade = str(exit_kpi_gate.get("grade") or "").strip().upper()
    exit_kpi_pass = bool(exit_kpi_gate.get("pass") is True)
    exit_kpi_improving = bool(exit_kpi_gate.get("improving") is True)
    exit_kpi_severe = bool(exit_kpi_gate.get("severe_fail") is True)
    min_notional_tuning = status_summary.get("min_notional_tuning") if isinstance(status_summary.get("min_notional_tuning"), dict) else {}
    min_notional_effectiveness = (
        status_summary.get("min_notional_effectiveness")
        if isinstance(status_summary.get("min_notional_effectiveness"), dict)
        else {}
    )
    batch_id = int(max(0, metrics.get("completed_batches") or 0))
    last_batch_id = int(max(0, _safe_int(last_state.get("last_batch_id"), 0)))
    already_processed = bool(ready and batch_id > 0 and batch_id <= last_batch_id)

    result: dict[str, Any] = {
        "timestamp_ms": _now_ms(),
        "ready": bool(ready),
        "batch_id": int(batch_id),
        "last_batch_id": int(last_batch_id),
        "already_processed": bool(already_processed),
        "metrics": metrics,
        "regime_metrics": regime_metrics,
        "actions": [],
        "overrides": {},
        "retrain": {
            "requested": False,
            "reason": "",
            "ran": False,
            "rc": None,
            "elapsed_sec": 0.0,
        },
        "status_paths": {
            "status_in": str(status_path),
            "diag_in": str(diag_path),
            "cf_in": str(cf_path),
            "reason_matrix_in": str(reason_matrix_path),
            "entry_score_driver_in": str(entry_score_driver_path),
            "reval_loss_driver_in": str(reval_loss_driver_path),
            "stage4_compare_in": str(stage4_path),
            "override_out": str(override_out_path),
            "state_file": str(state_file_path),
            "db": str(db_path),
            "since_id": int(since_id),
        },
        "runtime_key_aliases_existing": existing_aliases,
        "reason_matrix_metrics": reason_matrix_metrics,
        "entry_score_driver_metrics": entry_score_driver_metrics,
        "reval_loss_driver_metrics": reval_loss_driver_metrics,
        "reason_recent_metrics": reason_recent_metrics,
        "exit_kpi_gate": {
            "grade": exit_kpi_grade,
            "pass": bool(exit_kpi_pass),
            "improving": bool(exit_kpi_improving),
            "severe_fail": bool(exit_kpi_severe),
            "message": str(exit_kpi_gate.get("message") or ""),
        },
        "min_notional_tuning": min_notional_tuning,
        "min_notional_effectiveness": min_notional_effectiveness,
    }

    if not ready:
        result["actions"].append("skip:not_ready")
        _write_json(status_out_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if already_processed:
        result["actions"].append("skip:already_processed_batch")
        _write_json(status_out_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    out_overrides: dict[str, Any] = {}
    actions: list[str] = []
    retrain_reason: list[str] = []

    entry_issue_rate = _safe_float(metrics.get("entry_issue_ratio"), None)
    if entry_issue_rate is None:
        entry_issue_rate = _safe_float(metrics.get("entry_issue_rate"), None)
    miss_rate = _safe_float(metrics.get("miss_rate"), None)
    direction_hit = _safe_float(metrics.get("direction_hit"), None)
    avg_exit_regret = _safe_float(metrics.get("avg_exit_regret"), None)
    early_like_rate = _safe_float(metrics.get("early_like_rate"), None)
    event_mc_exit_rate = _safe_float(metrics.get("event_mc_exit_rate"), None)
    liq_like_rate = _safe_float(metrics.get("liq_like_rate"), None)
    avg_roe = _safe_float(metrics.get("avg_roe"), None)
    direction_hit_delta = _safe_float(metrics.get("direction_hit_delta"), None)
    entry_issue_ratio_delta = _safe_float(metrics.get("entry_issue_ratio_delta"), None)
    avg_exit_regret_delta = _safe_float(metrics.get("avg_exit_regret_delta"), None)
    batch_new_closed = _safe_int(metrics.get("new_closed_total"), 0)
    top_regret_rows = []
    try:
        top_regret_rows = list((cf.get("exit_counterfactual") or {}).get("top_regret_reasons") or [])
    except Exception:
        top_regret_rows = []
    reason_counts: dict[str, int] = {}
    total_reason_n = 0
    for row in top_regret_rows:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip().lower()
        n = _safe_int(row.get("n"), 0)
        if not reason or n <= 0:
            continue
        reason_counts[reason] = int(reason_counts.get(reason, 0) + n)
        total_reason_n += int(n)

    def _reason_ratio(token: str) -> float:
        if total_reason_n <= 0:
            return 0.0
        needle = str(token).strip().lower()
        hit = 0
        for key, n in reason_counts.items():
            if needle in str(key):
                hit += int(n)
        return float(hit) / float(max(1, total_reason_n))

    event_reason_ratio = _reason_ratio("event_mc_exit")
    unified_flip_ratio = _reason_ratio("unified_flip")
    hold_vs_exit_ratio = _reason_ratio("hold_vs_exit")
    external_sync_ratio = _reason_ratio("external_sync")
    if external_sync_ratio <= 0.0:
        external_sync_ratio = _reason_ratio("exchange_close_external_sync")
    ev_drop_ratio = _reason_ratio("ev_drop_exit")
    dd_reason_ratio = _reason_ratio("unrealized_dd")
    rm_event = reason_matrix_metrics.get("event_mc_exit") if isinstance(reason_matrix_metrics.get("event_mc_exit"), dict) else {}
    rm_unified = reason_matrix_metrics.get("unified_flip") if isinstance(reason_matrix_metrics.get("unified_flip"), dict) else {}
    rm_hold = reason_matrix_metrics.get("hold_vs_exit") if isinstance(reason_matrix_metrics.get("hold_vs_exit"), dict) else {}
    rm_external = reason_matrix_metrics.get("external_sync") if isinstance(reason_matrix_metrics.get("external_sync"), dict) else {}
    rm_entry_share = _safe_float(reason_matrix_metrics.get("cause_share_entry_direction_mismatch"), 0.0) or 0.0
    rm_exit_regret_share = _safe_float(reason_matrix_metrics.get("cause_share_exit_timing_regret"), 0.0) or 0.0
    rm_exit_cost_flip_share = _safe_float(reason_matrix_metrics.get("cause_share_exit_timing_or_cost_flip"), 0.0) or 0.0
    rm_exit_timing_total_share = float(rm_exit_regret_share + rm_exit_cost_flip_share)
    esd_top_bad = str(entry_score_driver_metrics.get("top_bad_feature") or "")
    esd_top_bad_score = _safe_float(entry_score_driver_metrics.get("top_bad_score"), 0.0) or 0.0
    esd_dir_conf_bad = _safe_float(entry_score_driver_metrics.get("pred_mu_dir_conf_bad_score"), 0.0) or 0.0
    esd_entry_q_bad = _safe_float(entry_score_driver_metrics.get("entry_quality_bad_score"), 0.0) or 0.0
    esd_entry_ev_bad = _safe_float(entry_score_driver_metrics.get("entry_ev_bad_score"), 0.0) or 0.0
    esd_leverage_bad = _safe_float(entry_score_driver_metrics.get("leverage_signal_bad_score"), 0.0) or 0.0
    rr_event = reason_recent_metrics.get("event_mc_exit") if isinstance(reason_recent_metrics.get("event_mc_exit"), dict) else {}
    rr_ev_drop = reason_recent_metrics.get("ev_drop_exit") if isinstance(reason_recent_metrics.get("ev_drop_exit"), dict) else {}
    rr_unified = reason_recent_metrics.get("unified_flip") if isinstance(reason_recent_metrics.get("unified_flip"), dict) else {}
    rr_hold = reason_recent_metrics.get("hold_vs_exit") if isinstance(reason_recent_metrics.get("hold_vs_exit"), dict) else {}
    rr_external = reason_recent_metrics.get("external_sync") if isinstance(reason_recent_metrics.get("external_sync"), dict) else {}
    rr_event_n = int(_safe_int(rr_event.get("n"), 0))
    rr_event_avg_roe = _safe_float(rr_event.get("avg_roe"), None)
    rr_event_loss = _safe_float(rr_event.get("loss_rate"), None)
    rr_ev_drop_n = int(_safe_int(rr_ev_drop.get("n"), 0))
    rr_ev_drop_avg_roe = _safe_float(rr_ev_drop.get("avg_roe"), None)
    rr_ev_drop_loss = _safe_float(rr_ev_drop.get("loss_rate"), None)
    rr_unified_n = int(_safe_int(rr_unified.get("n"), 0))
    rr_unified_avg_roe = _safe_float(rr_unified.get("avg_roe"), None)
    rr_unified_loss = _safe_float(rr_unified.get("loss_rate"), None)
    rr_hold_n = int(_safe_int(rr_hold.get("n"), 0))
    rr_hold_avg_roe = _safe_float(rr_hold.get("avg_roe"), None)
    rr_hold_loss = _safe_float(rr_hold.get("loss_rate"), None)
    rr_external_n = int(_safe_int(rr_external.get("n"), 0))
    rr_external_avg_roe = _safe_float(rr_external.get("avg_roe"), None)
    rr_external_loss = _safe_float(rr_external.get("loss_rate"), None)
    ld_by_reason = reval_loss_driver_metrics.get("by_reason") if isinstance(reval_loss_driver_metrics.get("by_reason"), dict) else {}
    ld_unified = ld_by_reason.get("unified_flip") if isinstance(ld_by_reason.get("unified_flip"), dict) else {}
    ld_hold = ld_by_reason.get("hold_vs_exit") if isinstance(ld_by_reason.get("hold_vs_exit"), dict) else {}
    ld_external = ld_by_reason.get("exchange_close_external_sync") if isinstance(ld_by_reason.get("exchange_close_external_sync"), dict) else {}
    ld_regret_delta_map = (
        reval_loss_driver_metrics.get("exit_regret_by_reason_delta")
        if isinstance(reval_loss_driver_metrics.get("exit_regret_by_reason_delta"), dict)
        else {}
    )

    def _sum_regret_delta_for_token(token: str) -> float | None:
        needle = str(token).strip().lower()
        if not needle:
            return None
        found = False
        tot = 0.0
        for k, v in ld_regret_delta_map.items():
            kk = str(k or "").strip().lower()
            if needle not in kk:
                continue
            dv = _safe_float(v, None)
            if dv is None:
                continue
            tot += float(dv)
            found = True
        return float(tot) if found else None

    event_regret_delta = _sum_regret_delta_for_token("event_mc_exit")
    dd_regret_delta = _sum_regret_delta_for_token("unrealized_dd")
    mn_blocked_n = int(_safe_int(min_notional_tuning.get("min_notional_blocked_n"), 0))
    mn_blocked_share = _safe_float(min_notional_tuning.get("min_notional_blocked_share"), None)
    mn_apply_hint = bool(min_notional_tuning.get("apply_hint") is True)
    mn_eff_available = bool(min_notional_effectiveness.get("available") is True)
    mn_eff_sample_n = int(_safe_int(min_notional_effectiveness.get("sample_n"), 0))
    mn_eff_reco_floor = _safe_float(min_notional_effectiveness.get("recommended_base_floor"), None)
    mn_eff_score_gain = _safe_float(min_notional_effectiveness.get("score_gain_vs_current"), None)
    mn_eff_low_delta = _safe_float(min_notional_effectiveness.get("low_minus_overall_roe"), None)
    tier_thresholds = _normalize_sorted_unique_floats(
        _parse_float_list(
            os.environ.get("AUTO_TUNE_CAPITAL_STAGE_THRESHOLDS", os.environ.get("CAPITAL_TIER_USDT", "500,1500,3000,6000,9000"))
        ),
        min_v=0.0,
    )
    if not tier_thresholds:
        tier_thresholds = [500.0, 1500.0, 3000.0, 6000.0, 9000.0]
    tier_expected_len = int(len(tier_thresholds) + 1)
    default_tiers = _expand_float_list([0.1, 0.2, 0.5, 1.0, 2.0, 4.0], tier_expected_len, [0.1])
    mn_rec_tiers_raw = min_notional_tuning.get("recommended_tiers")
    mn_rec_tiers = _expand_float_list(_parse_float_list(mn_rec_tiers_raw), tier_expected_len, default_tiers)
    try:
        mn_cap = float(os.environ.get("CAPITAL_TIER_MIN_NOTIONAL_MAX", 20.0) or 20.0)
    except Exception:
        mn_cap = 20.0
    mn_cap = float(max(2.0, mn_cap))
    mn_rec_tiers = [float(max(0.1, min(mn_cap, v))) for v in mn_rec_tiers]
    result["reason_breakdown"] = {
        "event_reason_ratio": float(event_reason_ratio),
        "unified_flip_ratio": float(unified_flip_ratio),
        "hold_vs_exit_ratio": float(hold_vs_exit_ratio),
        "external_sync_ratio": float(external_sync_ratio),
        "ev_drop_ratio": float(ev_drop_ratio),
        "dd_reason_ratio": float(dd_reason_ratio),
        "reason_counts": dict(reason_counts),
        "reason_total": int(total_reason_n),
        "reason_matrix_entry_direction_share": float(rm_entry_share),
        "reason_matrix_exit_timing_share": float(rm_exit_timing_total_share),
        "reason_matrix_hold_vs_exit_n": int(_safe_int(rm_hold.get("n"), 0)),
        "reason_matrix_hold_vs_exit_avg_roe": _safe_float(rm_hold.get("avg_roe"), None),
        "reason_matrix_external_sync_n": int(_safe_int(rm_external.get("n"), 0)),
        "reason_matrix_external_sync_avg_roe": _safe_float(rm_external.get("avg_roe"), None),
        "entry_score_top_bad_feature": esd_top_bad,
        "entry_score_top_bad_score": float(esd_top_bad_score),
        "entry_score_dir_conf_bad_score": float(esd_dir_conf_bad),
        "entry_score_entry_quality_bad_score": float(esd_entry_q_bad),
        "entry_score_entry_ev_bad_score": float(esd_entry_ev_bad),
        "entry_score_leverage_signal_bad_score": float(esd_leverage_bad),
        "recent_event_mc_exit_n": int(rr_event_n),
        "recent_event_mc_exit_avg_roe": rr_event_avg_roe,
        "recent_event_mc_exit_loss_rate": rr_event_loss,
        "recent_ev_drop_exit_n": int(rr_ev_drop_n),
        "recent_ev_drop_exit_avg_roe": rr_ev_drop_avg_roe,
        "recent_ev_drop_exit_loss_rate": rr_ev_drop_loss,
        "recent_unified_flip_n": int(rr_unified_n),
        "recent_unified_flip_avg_roe": rr_unified_avg_roe,
        "recent_unified_flip_loss_rate": rr_unified_loss,
        "recent_hold_vs_exit_n": int(rr_hold_n),
        "recent_hold_vs_exit_avg_roe": rr_hold_avg_roe,
        "recent_hold_vs_exit_loss_rate": rr_hold_loss,
        "recent_external_sync_n": int(rr_external_n),
        "recent_external_sync_avg_roe": rr_external_avg_roe,
        "recent_external_sync_loss_rate": rr_external_loss,
        "loss_driver_timing_issue": bool(reval_loss_driver_metrics.get("timing_issue_validated") is True),
        "loss_driver_window_exit_n": int(_safe_int(reval_loss_driver_metrics.get("window_exit_n"), 0)),
        "loss_driver_history_snapshots": int(_safe_int(reval_loss_driver_metrics.get("history_snapshots"), 0)),
        "min_notional_eff_available": bool(mn_eff_available),
        "min_notional_eff_sample_n": int(mn_eff_sample_n),
        "min_notional_eff_recommended_floor": mn_eff_reco_floor,
        "min_notional_eff_score_gain": mn_eff_score_gain,
        "min_notional_eff_low_minus_overall_roe": mn_eff_low_delta,
    }

    # Reason-specific auto tuning (recent 200 exits):
    # avg_roe < 0 => relax sensitivity (reduce early exits)
    # avg_roe > 0 => keep/slightly strengthen if stable.
    if rr_event_n >= 40 and rr_event_avg_roe is not None:
        if float(rr_event_avg_roe) < 0.0:
            actions.append("reason_recent_event_mc_relax")
            cur = _get_effective_float("EVENT_EXIT_SHOCK_MODE_THRESHOLD", existing_overrides, 1.0)
            _set_float_step(out_overrides, "EVENT_EXIT_SHOCK_MODE_THRESHOLD", cur, +0.10, min_v=0.8, max_v=3.0)
            cur = _get_effective_float("EVENT_EXIT_SHOCK_FAST_THRESHOLD", existing_overrides, 1.0)
            _set_float_step(out_overrides, "EVENT_EXIT_SHOCK_FAST_THRESHOLD", cur, +0.10, min_v=0.8, max_v=3.0)
            cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_SHOCK", existing_overrides, 1)
            _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_SHOCK", cur, +1, min_v=1, max_v=8)
            cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 3)
            _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=2, max_v=12)
            cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 4)
            _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=3, max_v=14)
            cur = _get_effective_float("EVENT_EXIT_SHOCK_MIN_PSL", existing_overrides, 0.88)
            _set_float_step(out_overrides, "EVENT_EXIT_SHOCK_MIN_PSL", cur, +0.01, min_v=0.70, max_v=0.995)
            cur = _get_effective_float("EVENT_EXIT_SHOCK_MIN_CVAR", existing_overrides, 0.025)
            _set_float_step(out_overrides, "EVENT_EXIT_SHOCK_MIN_CVAR", cur, +0.002, min_v=0.002, max_v=0.200)
            cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", existing_overrides, 0.75)
            _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", cur, +0.02, min_v=0.45, max_v=0.99)
            cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_MEAN_REVERT", existing_overrides, 0.85)
            _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_MEAN_REVERT", cur, +0.02, min_v=0.50, max_v=0.99)
            cur = _get_effective_float("EVENT_MC_TSTAR_BYPASS_SHOCK", existing_overrides, 1.20)
            _set_float_step(out_overrides, "EVENT_MC_TSTAR_BYPASS_SHOCK", cur, +0.05, min_v=0.8, max_v=3.0)
            cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_CHOP", existing_overrides, -0.0008)
            _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_CHOP", cur, -0.00015, min_v=-0.0060, max_v=0.0060)
            cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_RANDOM", existing_overrides, -0.0003)
            _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_RANDOM", cur, -0.00015, min_v=-0.0060, max_v=0.0060)
        elif rr_event_n >= 80 and float(rr_event_avg_roe) > 0.002:
            actions.append("reason_recent_event_mc_strengthen")
            cur = _get_effective_float("EVENT_EXIT_SHOCK_MODE_THRESHOLD", existing_overrides, 1.0)
            _set_float_step(out_overrides, "EVENT_EXIT_SHOCK_MODE_THRESHOLD", cur, -0.05, min_v=0.8, max_v=3.0)
            cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_CHOP", existing_overrides, -0.0008)
            _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_CHOP", cur, +0.00008, min_v=-0.0060, max_v=0.0060)

    if rr_ev_drop_n >= 30 and rr_ev_drop_avg_roe is not None:
        if float(rr_ev_drop_avg_roe) < 0.0:
            actions.append("reason_recent_ev_drop_relax")
            cur = _get_effective_float("EXIT_SCORE_DROP", existing_overrides, 0.0004)
            _set_float_step(out_overrides, "EXIT_SCORE_DROP", cur, +0.00015, min_v=0.0, max_v=0.0100)
            cur = _get_effective_float("EV_DROP_OPP_CONF_MIN", existing_overrides, 0.58)
            _set_float_step(out_overrides, "EV_DROP_OPP_CONF_MIN", cur, +0.01, min_v=0.52, max_v=0.90)
            cur = _get_effective_float("EV_DROP_OPP_CONF_RISE_MIN", existing_overrides, 0.010)
            _set_float_step(out_overrides, "EV_DROP_OPP_CONF_RISE_MIN", cur, +0.003, min_v=0.0, max_v=0.08)
            cur = _get_effective_float("EV_DROP_OPP_EDGE_MIN", existing_overrides, 0.0004)
            _set_float_step(out_overrides, "EV_DROP_OPP_EDGE_MIN", cur, +0.0001, min_v=0.0, max_v=0.01)
            cur = _get_effective_float("EV_DROP_MIN_PROGRESS_RANDOM", existing_overrides, 0.80)
            _set_float_step(out_overrides, "EV_DROP_MIN_PROGRESS_RANDOM", cur, +0.02, min_v=0.45, max_v=0.99)
            cur = _get_effective_float("EV_DROP_MIN_PROGRESS_MEAN_REVERT", existing_overrides, 0.88)
            _set_float_step(out_overrides, "EV_DROP_MIN_PROGRESS_MEAN_REVERT", cur, +0.02, min_v=0.55, max_v=0.99)
            cur = _get_effective_int("EV_DROP_MIN_CONFIRM_NON_SHOCK", existing_overrides, 3)
            _set_int_step(out_overrides, "EV_DROP_MIN_CONFIRM_NON_SHOCK", cur, +1, min_v=2, max_v=12)
        elif rr_ev_drop_n >= 60 and float(rr_ev_drop_avg_roe) > 0.002:
            actions.append("reason_recent_ev_drop_strengthen")
            cur = _get_effective_float("EXIT_SCORE_DROP", existing_overrides, 0.0004)
            _set_float_step(out_overrides, "EXIT_SCORE_DROP", cur, -0.00008, min_v=0.0, max_v=0.0100)
            cur = _get_effective_float("EV_DROP_OPP_CONF_MIN", existing_overrides, 0.58)
            _set_float_step(out_overrides, "EV_DROP_OPP_CONF_MIN", cur, -0.005, min_v=0.52, max_v=0.90)

    if rr_unified_n >= 8 and rr_unified_avg_roe is not None:
        if float(rr_unified_avg_roe) < -0.0015 or (rr_unified_loss is not None and float(rr_unified_loss) >= 0.55):
            actions.append("reason_recent_unified_flip_relax")
            cur = _get_effective_int("UNIFIED_FLIP_CONFIRM_TICKS_NORMAL", existing_overrides, 3)
            _set_int_step(out_overrides, "UNIFIED_FLIP_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=12)
            cur = _get_effective_int("UNIFIED_FLIP_CONFIRM_TICKS_NOISE", existing_overrides, 4)
            _set_int_step(out_overrides, "UNIFIED_FLIP_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=14)
            cur = _get_effective_float("UNIFIED_FLIP_MIN_PROGRESS_RANDOM", existing_overrides, 0.75)
            _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_PROGRESS_RANDOM", cur, +0.02, min_v=0.45, max_v=0.99)
            cur = _get_effective_float("UNIFIED_FLIP_MIN_PROGRESS_MEAN_REVERT", existing_overrides, 0.85)
            _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_PROGRESS_MEAN_REVERT", cur, +0.02, min_v=0.55, max_v=0.99)
            cur = _get_effective_float("UNIFIED_FLIP_MIN_REVERSE_EDGE", existing_overrides, 0.0008)
            _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_REVERSE_EDGE", cur, +0.0002, min_v=0.0002, max_v=0.0100)
            cur = _get_effective_float("UNIFIED_FLIP_MIN_OPPOSITE_SIDE_PROB", existing_overrides, 0.56)
            _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_OPPOSITE_SIDE_PROB", cur, +0.01, min_v=0.50, max_v=0.90)
            cur = _get_effective_float("UNIFIED_FLIP_BYPASS_SHOCK", existing_overrides, 1.15)
            _set_float_step(out_overrides, "UNIFIED_FLIP_BYPASS_SHOCK", cur, +0.05, min_v=0.80, max_v=3.00)
            cur = _get_effective_float("EXIT_RESPECT_ENTRY_MIN_PROGRESS", existing_overrides, 0.65)
            _set_float_step(out_overrides, "EXIT_RESPECT_ENTRY_MIN_PROGRESS", cur, +0.02, min_v=0.45, max_v=0.95)
        elif rr_unified_n >= 20 and float(rr_unified_avg_roe) > 0.0015:
            actions.append("reason_recent_unified_flip_strengthen")
            cur = _get_effective_int("UNIFIED_FLIP_CONFIRM_TICKS_NORMAL", existing_overrides, 3)
            _set_int_step(out_overrides, "UNIFIED_FLIP_CONFIRM_TICKS_NORMAL", cur, -1, min_v=1, max_v=12)
            cur = _get_effective_float("UNIFIED_FLIP_MIN_REVERSE_EDGE", existing_overrides, 0.0008)
            _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_REVERSE_EDGE", cur, -0.0001, min_v=0.0002, max_v=0.0100)

    # Dedicated reason loop #2: hold_vs_exit
    hold_timing_issue = bool(ld_hold.get("timing_issue") is True)
    hold_inc_ratio = _safe_float(ld_hold.get("recommended_hold_increase_ratio"), 0.0) or 0.0
    if rr_hold_n >= 20 and rr_hold_avg_roe is not None:
        if (
            float(rr_hold_avg_roe) < -0.001
            or (rr_hold_loss is not None and float(rr_hold_loss) >= 0.55)
            or hold_timing_issue
        ):
            actions.append("reason_recent_hold_vs_exit_relax")
            cur = _get_effective_float("HOLD_EVAL_MIN_PROGRESS_TO_EXIT", existing_overrides, 0.85)
            _set_float_step(out_overrides, "HOLD_EVAL_MIN_PROGRESS_TO_EXIT", cur, +max(0.03, float(hold_inc_ratio) * 0.40), min_v=0.45, max_v=1.20)
            cur = _get_effective_int("HOLD_EVAL_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
            _set_int_step(out_overrides, "HOLD_EVAL_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=12)
            cur = _get_effective_int("HOLD_EVAL_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
            _set_int_step(out_overrides, "HOLD_EVAL_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=14)
            cur = _get_effective_float("HOLD_EVAL_EXIT_MARGIN", existing_overrides, 0.0)
            _set_float_step(out_overrides, "HOLD_EVAL_EXIT_MARGIN", cur, -0.0001, min_v=-0.0050, max_v=0.0050)
        elif (
            rr_hold_n >= 60
            and float(rr_hold_avg_roe) > 0.0015
            and (rr_hold_loss is None or float(rr_hold_loss) <= 0.45)
            and (not hold_timing_issue)
        ):
            actions.append("reason_recent_hold_vs_exit_strengthen")
            cur = _get_effective_float("HOLD_EVAL_MIN_PROGRESS_TO_EXIT", existing_overrides, 0.85)
            _set_float_step(out_overrides, "HOLD_EVAL_MIN_PROGRESS_TO_EXIT", cur, -0.02, min_v=0.45, max_v=1.20)
            cur = _get_effective_int("HOLD_EVAL_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
            _set_int_step(out_overrides, "HOLD_EVAL_EXIT_CONFIRM_TICKS_NORMAL", cur, -1, min_v=1, max_v=12)

    # Dedicated reason loop #3: external_sync
    ext_timing_issue = bool(ld_external.get("timing_issue") is True)
    if rr_external_n >= 8 and rr_external_avg_roe is not None:
        if (
            float(rr_external_avg_roe) < -0.001
            or (rr_external_loss is not None and float(rr_external_loss) >= 0.55)
            or ext_timing_issue
        ):
            actions.append("reason_recent_external_sync_guard")
            cur = _get_effective_int("LIVE_LIQUIDATION_MISS_COUNT", existing_overrides, 2)
            _set_int_step(out_overrides, "LIVE_LIQUIDATION_MISS_COUNT", cur, +1, min_v=1, max_v=8)
            cur = _get_effective_float("EXTERNAL_CLOSE_LIQ_BUFFER_PCT", existing_overrides, 0.0030)
            _set_float_step(out_overrides, "EXTERNAL_CLOSE_LIQ_BUFFER_PCT", cur, -0.0002, min_v=0.0005, max_v=0.0200)
        elif (
            rr_external_n >= 20
            and float(rr_external_avg_roe) > 0.001
            and (rr_external_loss is None or float(rr_external_loss) <= 0.40)
            and (not ext_timing_issue)
        ):
            actions.append("reason_recent_external_sync_relax")
            cur = _get_effective_int("LIVE_LIQUIDATION_MISS_COUNT", existing_overrides, 2)
            _set_int_step(out_overrides, "LIVE_LIQUIDATION_MISS_COUNT", cur, -1, min_v=1, max_v=8)
            cur = _get_effective_float("EXTERNAL_CLOSE_LIQ_BUFFER_PCT", existing_overrides, 0.0030)
            _set_float_step(out_overrides, "EXTERNAL_CLOSE_LIQ_BUFFER_PCT", cur, +0.0002, min_v=0.0005, max_v=0.0200)

    # Loss-driver timing recommendations (counterfactual-validated).
    # Apply only to the three dedicated axes.
    for axis_name, axis_data in (
        ("unified_flip", ld_unified),
        ("hold_vs_exit", ld_hold),
        ("exchange_close_external_sync", ld_external),
    ):
        if not isinstance(axis_data, dict):
            continue
        if not bool(axis_data.get("timing_issue") is True):
            continue
        rec = axis_data.get("recommended_param_deltas") if isinstance(axis_data.get("recommended_param_deltas"), dict) else {}
        if not rec:
            continue
        actions.append(f"loss_driver_timing_rec:{axis_name}")
        for k, delta_raw in rec.items():
            key = str(k or "").strip()
            if not key:
                continue
            dv = _safe_float(delta_raw, None)
            if dv is None or abs(float(dv)) <= 1e-12:
                continue
            key_u = key.upper()
            if ("_TICKS" in key_u) or ("_COUNT" in key_u):
                cur_i = _get_effective_int(key, out_overrides if key in out_overrides else existing_overrides, 0)
                _set_int_step(out_overrides, key, cur_i, int(round(float(dv))), min_v=0, max_v=64)
                continue
            cur_f = _get_effective_float(key, out_overrides if key in out_overrides else existing_overrides, 0.0)
            # Keep deltas bounded to avoid unstable jumps from a single window.
            step = float(max(-0.25, min(0.25, float(dv))))
            _set_float_step(out_overrides, key, cur_f, step, min_v=-5.0, max_v=5.0)

    if (
        mn_apply_hint
        and mn_blocked_n >= 10
        and mn_blocked_share is not None
        and float(mn_blocked_share) >= 0.05
        and len(mn_rec_tiers) == int(tier_expected_len)
    ):
        actions.append("min_notional_tier_micro_tune")
        cur_tiers = _get_effective_float_list(
            "CAPITAL_TIER_MIN_NOTIONAL",
            existing_overrides,
            default_tiers,
            expected_len=tier_expected_len,
        )
        # Stronger blend when blocked share is high.
        blend = 0.50 if float(mn_blocked_share) >= 0.12 else 0.35
        tuned = []
        for i in range(int(tier_expected_len)):
            c = float(cur_tiers[i])
            r = float(mn_rec_tiers[i])
            tuned.append(float((1.0 - blend) * c + blend * r))
        tuned.sort()
        for i in range(len(tuned)):
            low = 0.1 if i == 0 else float(tuned[i - 1] + 0.15)
            tuned[i] = max(low, tuned[i])
        tuned = [float(min(mn_cap, round(v, 3))) for v in tuned]
        out_overrides["CAPITAL_TIER_MIN_NOTIONAL_OVERRIDE_BASE"] = 1
        out_overrides["CAPITAL_TIER_USDT"] = _float_list_to_env(tier_thresholds)
        out_overrides["CAPITAL_TIER_MIN_NOTIONAL"] = _float_list_to_env(tuned)
        result["min_notional_tier_tuned"] = {
            "blocked_n": int(mn_blocked_n),
            "blocked_share": float(mn_blocked_share),
            "tier_thresholds": [float(x) for x in tier_thresholds],
            "current_tiers": [float(x) for x in cur_tiers],
            "recommended_tiers": [float(x) for x in mn_rec_tiers],
            "tuned_tiers": [float(x) for x in tuned],
        }

    # Profitability-aware min_notional adjustment:
    # use exit-based notional/ROE effectiveness to avoid over-trading tiny notionals.
    if mn_eff_available and mn_eff_sample_n >= 120 and mn_eff_reco_floor is not None:
        cur_tiers_eff = _get_effective_float_list(
            "CAPITAL_TIER_MIN_NOTIONAL",
            out_overrides if out_overrides else existing_overrides,
            default_tiers,
            expected_len=tier_expected_len,
        )
        cur_floor = float(max(0.1, cur_tiers_eff[0]))
        reco_floor = float(max(0.1, min(mn_cap, mn_eff_reco_floor)))
        gain = float(mn_eff_score_gain or 0.0)
        low_delta = float(mn_eff_low_delta or 0.0)
        need_raise = bool(
            (gain > 0.00020 and reco_floor > cur_floor * 1.10)
            or (low_delta < -0.0015 and reco_floor > cur_floor * 1.05)
        )
        need_relax = bool(
            (gain > 0.00010 and reco_floor < cur_floor * 0.92)
            and (mn_blocked_share is not None and float(mn_blocked_share) >= 0.06)
            and (low_delta > -0.0006)
        )
        if need_raise or need_relax:
            actions.append("min_notional_effectiveness_tune")
            # Move gradually to avoid oscillation.
            w = 0.22 if need_raise else 0.18
            shift = float((reco_floor - cur_floor) * w)
            tuned = []
            for i, tv in enumerate(cur_tiers_eff):
                decay = float(0.65 ** i)
                nv = float(tv + shift * decay)
                tuned.append(float(max(0.1, min(mn_cap, nv))))
            tuned.sort()
            for i in range(len(tuned)):
                low = 0.1 if i == 0 else float(tuned[i - 1] + 0.10)
                tuned[i] = max(low, tuned[i])
            tuned = [float(round(v, 3)) for v in tuned]
            out_overrides["CAPITAL_TIER_MIN_NOTIONAL_OVERRIDE_BASE"] = 1
            out_overrides["CAPITAL_TIER_USDT"] = _float_list_to_env(tier_thresholds)
            out_overrides["CAPITAL_TIER_MIN_NOTIONAL"] = _float_list_to_env(tuned)
            result["min_notional_effectiveness_tuned"] = {
                "sample_n": int(mn_eff_sample_n),
                "current_floor": float(cur_floor),
                "recommended_floor": float(reco_floor),
                "score_gain_vs_current": float(gain),
                "low_minus_overall_roe": float(low_delta),
                "mode": ("raise" if need_raise else "relax"),
                "current_tiers": [float(x) for x in cur_tiers_eff],
                "tuned_tiers": [float(x) for x in tuned],
            }

    # Capital-stage baseline: keep sizing/leverage profile aligned with account growth steps.
    capital_stage_enabled = _is_true(os.environ.get("AUTO_TUNE_CAPITAL_STAGE_ENABLED", "1"))
    capital_usdt = _read_capital_balance_usdt(repo_root, existing_overrides)
    stage_total_caps = _expand_float_list(
        _parse_float_list(os.environ.get("AUTO_TUNE_CAPITAL_STAGE_TOTAL_CAP", os.environ.get("CAPITAL_TIER_TOTAL_CAP", ""))),
        tier_expected_len,
        _expand_float_list([2.0, 2.8, 4.0, 5.5, 7.5, 9.5], tier_expected_len, [2.0]),
    )
    stage_min_notional = _expand_float_list(
        _parse_float_list(os.environ.get("AUTO_TUNE_CAPITAL_STAGE_MIN_NOTIONAL", os.environ.get("CAPITAL_TIER_MIN_NOTIONAL", ""))),
        tier_expected_len,
        default_tiers,
    )
    stage_lev_max = _expand_float_list(
        _parse_float_list(os.environ.get("AUTO_TUNE_CAPITAL_STAGE_LEVERAGE_MAX", "")),
        tier_expected_len,
        _expand_float_list([8.0, 12.0, 20.0, 28.0, 38.0, 50.0], tier_expected_len, [12.0]),
    )
    stage_low_conf_cap = _expand_float_list(
        _parse_float_list(os.environ.get("AUTO_TUNE_CAPITAL_STAGE_LOW_CONF_CAP", "")),
        tier_expected_len,
        _expand_float_list([1.0, 1.4, 1.8, 2.3, 2.8, 3.3], tier_expected_len, [1.0]),
    )
    stage_high_vpin_cap = _expand_float_list(
        _parse_float_list(os.environ.get("AUTO_TUNE_CAPITAL_STAGE_HIGH_VPIN_CAP", "")),
        tier_expected_len,
        _expand_float_list([1.0, 1.2, 1.5, 1.9, 2.3, 2.8], tier_expected_len, [1.0]),
    )
    stage_train_trigger = [int(max(40, round(v))) for v in _expand_float_list(
        _parse_float_list(os.environ.get("AUTO_TUNE_CAPITAL_STAGE_TRAIN_TRIGGER", "")),
        tier_expected_len,
        _expand_float_list([80.0, 90.0, 110.0, 130.0, 150.0, 170.0], tier_expected_len, [120.0]),
    )]
    stage_idx = 0
    if capital_usdt is not None:
        for th in tier_thresholds:
            if float(capital_usdt) < float(th):
                break
            stage_idx += 1
    stage_idx = int(max(0, min(stage_idx, tier_expected_len - 1)))
    result["capital_stage"] = {
        "enabled": bool(capital_stage_enabled),
        "capital_usdt": (float(capital_usdt) if capital_usdt is not None else None),
        "tier_thresholds": [float(x) for x in tier_thresholds],
        "stage_index": int(stage_idx),
        "stage_total_cap": float(stage_total_caps[stage_idx]),
        "stage_min_notional": float(stage_min_notional[stage_idx]),
        "stage_leverage_max": float(stage_lev_max[stage_idx]),
        "stage_low_conf_cap": float(stage_low_conf_cap[stage_idx]),
        "stage_high_vpin_cap": float(stage_high_vpin_cap[stage_idx]),
        "stage_train_trigger_new_exits": int(stage_train_trigger[stage_idx]),
    }
    if capital_stage_enabled and capital_usdt is not None:
        actions.append(f"capital_stage_profile:{stage_idx}")
        out_overrides["CAPITAL_TIER_USDT"] = _float_list_to_env(tier_thresholds)
        out_overrides["CAPITAL_TIER_TOTAL_CAP_OVERRIDE_BASE"] = 1
        out_overrides["CAPITAL_TIER_TOTAL_CAP"] = _float_list_to_env(stage_total_caps)
        out_overrides["CAPITAL_TIER_MIN_NOTIONAL_OVERRIDE_BASE"] = 1
        if "CAPITAL_TIER_MIN_NOTIONAL" not in out_overrides:
            out_overrides["CAPITAL_TIER_MIN_NOTIONAL"] = _float_list_to_env(stage_min_notional)
        # Keep base caps aligned so tier caps are effective.
        out_overrides["MAX_NOTIONAL_EXPOSURE"] = float(stage_total_caps[stage_idx])
        out_overrides["LIVE_MAX_NOTIONAL_EXPOSURE"] = float(stage_total_caps[stage_idx])
        out_overrides["KELLY_TOTAL_EXPOSURE"] = float(stage_total_caps[stage_idx])
        out_overrides["ALPHA_TRAIN_TRIGGER_NEW_EXITS"] = int(stage_train_trigger[stage_idx])

        out_overrides["LEVERAGE_TARGET_MAX"] = float(max(1.0, min(50.0, stage_lev_max[stage_idx])))
        out_overrides["LEVERAGE_LOW_CONF_CAP"] = float(max(1.0, min(out_overrides["LEVERAGE_TARGET_MAX"], stage_low_conf_cap[stage_idx])))
        out_overrides["LEVERAGE_HIGH_VPIN_CAP"] = float(max(1.0, min(out_overrides["LEVERAGE_TARGET_MAX"], stage_high_vpin_cap[stage_idx])))

    direction_bad = bool(
        (entry_issue_rate is not None and entry_issue_rate >= float(args.min_entry_issue))
        or (miss_rate is not None and miss_rate >= float(args.min_miss_rate))
    )
    exit_bad = bool(
        (avg_exit_regret is not None and avg_exit_regret >= float(args.high_exit_regret))
        or (early_like_rate is not None and early_like_rate >= float(args.high_early_like))
        or (event_mc_exit_rate is not None and event_mc_exit_rate >= float(args.high_event_mc_rate))
    )
    if exit_kpi_grade == "FAIL":
        exit_bad = True
    risk_bad = bool(
        (liq_like_rate is not None and liq_like_rate >= float(args.high_liq_rate))
        or (avg_roe is not None and avg_roe < -0.0025)
    )
    direction_good = bool(
        (direction_hit is not None and direction_hit >= 0.56)
        and (entry_issue_rate is not None and entry_issue_rate <= 0.45)
        and (not risk_bad)
    )
    try:
        delta_sample_min = int(os.environ.get("AUTO_TUNE_DELTA_SAMPLE_MIN", 40) or 40)
    except Exception:
        delta_sample_min = 40
    delta_sample_min = int(max(20, delta_sample_min))
    try:
        delta_hit_drop = float(os.environ.get("AUTO_TUNE_DELTA_HIT_DROP", 0.010) or 0.010)
    except Exception:
        delta_hit_drop = 0.010
    try:
        delta_entry_rise = float(os.environ.get("AUTO_TUNE_DELTA_ENTRY_ISSUE_RISE", 0.015) or 0.015)
    except Exception:
        delta_entry_rise = 0.015
    try:
        delta_regret_rise = float(os.environ.get("AUTO_TUNE_DELTA_EXIT_REGRET_RISE", 0.0025) or 0.0025)
    except Exception:
        delta_regret_rise = 0.0025
    delta_bad = bool(
        int(batch_new_closed) >= int(delta_sample_min)
        and (
            (direction_hit_delta is not None and float(direction_hit_delta) <= -abs(float(delta_hit_drop)))
            or (entry_issue_ratio_delta is not None and float(entry_issue_ratio_delta) >= abs(float(delta_entry_rise)))
            or (avg_exit_regret_delta is not None and float(avg_exit_regret_delta) >= abs(float(delta_regret_rise)))
        )
    )
    result["delta_trigger"] = {
        "enabled": True,
        "sample_min": int(delta_sample_min),
        "batch_new_closed": int(batch_new_closed),
        "direction_hit_delta": direction_hit_delta,
        "entry_issue_ratio_delta": entry_issue_ratio_delta,
        "avg_exit_regret_delta": avg_exit_regret_delta,
        "direction_hit_drop_threshold": -abs(float(delta_hit_drop)),
        "entry_issue_rise_threshold": abs(float(delta_entry_rise)),
        "avg_exit_regret_rise_threshold": abs(float(delta_regret_rise)),
        "delta_bad": bool(delta_bad),
    }
    result["fixed_gate_trigger"] = {
        "enabled": True,
        "grade": exit_kpi_grade,
        "pass": bool(exit_kpi_pass),
        "improving": bool(exit_kpi_improving),
        "severe_fail": bool(exit_kpi_severe),
        "message": str(exit_kpi_gate.get("message") or ""),
    }

    if direction_bad:
        actions.append("direction_quality_guard_up")
        cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_CONF", existing_overrides, 0.56)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_CONF", cur, +0.005, min_v=0.52, max_v=0.66)
        cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_EDGE", existing_overrides, 0.08)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_EDGE", cur, +0.005, min_v=0.05, max_v=0.12)
        cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", existing_overrides, 0.56)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", cur, +0.005, min_v=0.52, max_v=0.66)
        cur = _get_effective_float("ALPHA_DIRECTION_SCORE_MIN_SIDE_PROB", existing_overrides, 0.52)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_SCORE_MIN_SIDE_PROB", cur, +0.005, min_v=0.52, max_v=0.66)
        cur = _get_effective_float("POLICY_SMALL_GAP_CONFIDENCE", existing_overrides, 0.62)
        _set_float_step(out_overrides, "POLICY_SMALL_GAP_CONFIDENCE", cur, +0.01, min_v=0.55, max_v=0.72)
        cur = _get_effective_float("POLICY_SMALL_GAP_DIR_CONFIDENCE", existing_overrides, 0.58)
        _set_float_step(out_overrides, "POLICY_SMALL_GAP_DIR_CONFIDENCE", cur, +0.01, min_v=0.52, max_v=0.70)
        cur = _get_effective_float("POLICY_SMALL_GAP_DIR_EDGE", existing_overrides, 0.08)
        _set_float_step(out_overrides, "POLICY_SMALL_GAP_DIR_EDGE", cur, +0.005, min_v=0.05, max_v=0.12)
        cur = _get_effective_float("POLICY_SMALL_GAP_SIDE_PROB", existing_overrides, 0.56)
        _set_float_step(out_overrides, "POLICY_SMALL_GAP_SIDE_PROB", cur, +0.005, min_v=0.52, max_v=0.70)
        cur_trigger = _get_effective_int("ALPHA_TRAIN_TRIGGER_NEW_EXITS", existing_overrides, 200)
        _set_int_step(out_overrides, "ALPHA_TRAIN_TRIGGER_NEW_EXITS", cur_trigger, -20, min_v=120, max_v=400)
        out_overrides["ALPHA_DIRECTION_LABEL_SOURCE"] = "auto"
        cur = _get_effective_float("ALPHA_DIRECTION_LABEL_FEE_BPS", existing_overrides, 1.5)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_LABEL_FEE_BPS", cur, +0.5, min_v=0.0, max_v=8.0)
        cur = _get_effective_float("ALPHA_DIRECTION_LABEL_SLIPPAGE_BPS", existing_overrides, 1.0)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_LABEL_SLIPPAGE_BPS", cur, +0.5, min_v=0.0, max_v=10.0)
        cur = _get_effective_float("ALPHA_DIRECTION_NEUTRAL_BAND_BPS", existing_overrides, 1.0)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_NEUTRAL_BAND_BPS", cur, +0.5, min_v=0.0, max_v=8.0)
        out_overrides["ALPHA_DIRECTION_LABEL_EXCLUDE_TOKENS"] = "external,manual_cleanup,rebal,rebal_exit,cleanup,manual_close"
        out_overrides["ALPHA_DIRECTION_HSTAR_LABEL_ENABLED"] = 1
        out_overrides["ALPHA_DIRECTION_HSTAR_BARS"] = "1,2,3,5,8,13,21,34"
        out_overrides["ALPHA_DIRECTION_LGBM_TUNE_ENABLED"] = 1
        # Expectancy-first objective for direction model selection.
        out_overrides["ALPHA_DIRECTION_SCORE_EXP_WEIGHT"] = 1.0
        out_overrides["ALPHA_DIRECTION_SCORE_HIT_WEIGHT"] = 0.04
        out_overrides["ALPHA_DIRECTION_SCORE_AUC_WEIGHT"] = 0.02
        out_overrides["ALPHA_DIRECTION_SCORE_BALACC_WEIGHT"] = 0.02
        out_overrides["ALPHA_DIRECTION_SCORE_LOGLOSS_PENALTY"] = 0.04
        cur = _get_effective_float("ALPHA_DIRECTION_SCORE_COVERAGE_TARGET", existing_overrides, 0.35)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_SCORE_COVERAGE_TARGET", cur, +0.01, min_v=0.15, max_v=0.80)
        cur = _get_effective_float("ALPHA_DIRECTION_SCORE_COVERAGE_PENALTY_BPS", existing_overrides, 8.0)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_SCORE_COVERAGE_PENALTY_BPS", cur, +0.5, min_v=1.0, max_v=25.0)
        cur = _get_effective_float("ALPHA_DIRECTION_SCORE_MIN_EXPECTANCY_BPS", existing_overrides, -2.0)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_SCORE_MIN_EXPECTANCY_BPS", cur, +0.5, min_v=-8.0, max_v=8.0)
        # Respect explicit operator disable in env/override instead of force-enabling.
        sq_enabled = _is_true(existing_overrides.get("SYMBOL_QUALITY_FILTER_ENABLED", os.environ.get("SYMBOL_QUALITY_FILTER_ENABLED", "1")))
        sq_time_enabled = _is_true(existing_overrides.get("SYMBOL_QUALITY_TIME_FILTER_ENABLED", os.environ.get("SYMBOL_QUALITY_TIME_FILTER_ENABLED", "1")))
        if sq_enabled:
            out_overrides["SYMBOL_QUALITY_FILTER_ENABLED"] = 1
            out_overrides["SYMBOL_QUALITY_TIME_FILTER_ENABLED"] = 1 if sq_time_enabled else 0
            cur = _get_effective_int("SYMBOL_QUALITY_LOOKBACK_EXITS", existing_overrides, 320)
            _set_int_step(out_overrides, "SYMBOL_QUALITY_LOOKBACK_EXITS", cur, +40, min_v=120, max_v=600)
            cur = _get_effective_int("SYMBOL_QUALITY_MIN_EXITS", existing_overrides, 40)
            _set_int_step(out_overrides, "SYMBOL_QUALITY_MIN_EXITS", cur, +4, min_v=16, max_v=120)
            cur = _get_effective_float("SYMBOL_QUALITY_TIME_WEIGHT", existing_overrides, 0.30)
            _set_float_step(out_overrides, "SYMBOL_QUALITY_TIME_WEIGHT", cur, +0.03, min_v=0.10, max_v=0.70)
            cur = _get_effective_float("SYMBOL_QUALITY_MIN_SCORE", existing_overrides, 0.42)
            _set_float_step(out_overrides, "SYMBOL_QUALITY_MIN_SCORE", cur, +0.01, min_v=0.30, max_v=0.80)
        else:
            out_overrides["SYMBOL_QUALITY_FILTER_ENABLED"] = 0
            out_overrides["SYMBOL_QUALITY_TIME_FILTER_ENABLED"] = 0
        # Preserve high-quality entry opportunities when event entry filter is too strict.
        out_overrides["ENTRY_EVENT_EXIT_BYPASS_ENABLED"] = 1
        retrain_reason.append("entry_direction_quality")

    # Entry-score-driver guided direction layer correction:
    # - If dir_conf / entry_quality are paradoxical, reduce chop dependence on dir_conf
    #   and shift weight toward edge/gap; keep trend stricter.
    esd_available = bool(entry_score_driver_metrics.get("available"))
    esd_pairs = int(_safe_int(entry_score_driver_metrics.get("matched_pairs"), 0))
    if esd_available and esd_pairs >= 200:
        if esd_leverage_bad >= 0.08 or (esd_top_bad == "leverage_signal_score" and esd_top_bad_score >= 0.05):
            # If leverage_signal_score is a dominant bad driver, reduce its influence
            # and lean more on pre-ROE proxy components.
            actions.append("entry_score_driver_leverage_signal_downweight")
            cur = _get_effective_float("LEVERAGE_SIGNAL_BLEND", existing_overrides, 0.55)
            _set_float_step(out_overrides, "LEVERAGE_SIGNAL_BLEND", cur, -0.05, min_v=0.15, max_v=0.90)
            cur = _get_effective_float("LEVERAGE_SIGNAL_SCORE_POW", existing_overrides, 1.20)
            _set_float_step(out_overrides, "LEVERAGE_SIGNAL_SCORE_POW", cur, +0.05, min_v=0.40, max_v=2.50)
            cur = _get_effective_float("LEVERAGE_SIGNAL_QUALITY_WEIGHT", existing_overrides, 0.62)
            _set_float_step(out_overrides, "LEVERAGE_SIGNAL_QUALITY_WEIGHT", cur, -0.05, min_v=0.20, max_v=0.90)
            cur = _get_effective_float("LEVERAGE_PRE_ROE_PROXY_BLEND", existing_overrides, 0.65)
            _set_float_step(out_overrides, "LEVERAGE_PRE_ROE_PROXY_BLEND", cur, +0.05, min_v=0.20, max_v=0.95)
            cur = _get_effective_float("LEVERAGE_PRE_ROE_W_ONEWAY", existing_overrides, 0.06)
            _set_float_step(out_overrides, "LEVERAGE_PRE_ROE_W_ONEWAY", cur, -0.01, min_v=0.0, max_v=0.40)
            cur = _get_effective_float("LEVERAGE_PRE_ROE_W_DIR_CONF", existing_overrides, 0.40)
            _set_float_step(out_overrides, "LEVERAGE_PRE_ROE_W_DIR_CONF", cur, +0.02, min_v=0.05, max_v=0.75)
            cur = _get_effective_float("LEVERAGE_PRE_ROE_W_ENTRY_Q", existing_overrides, 0.32)
            _set_float_step(out_overrides, "LEVERAGE_PRE_ROE_W_ENTRY_Q", cur, +0.02, min_v=0.05, max_v=0.75)
            retrain_reason.append("entry_score_driver_leverage_signal")
        if (esd_dir_conf_bad >= 0.08) or (esd_entry_q_bad >= 0.08):
            actions.append("entry_score_driver_regime_reweight")
            out_overrides["ALPHA_DIRECTION_CALIBRATION_ENABLED"] = 1
            cur = _get_effective_int("ALPHA_DIRECTION_CALIBRATION_MIN_SAMPLES", existing_overrides, 400)
            _set_int_step(out_overrides, "ALPHA_DIRECTION_CALIBRATION_MIN_SAMPLES", cur, -40, min_v=160, max_v=1200)
            cur = _get_effective_float("ALPHA_DIRECTION_CALIBRATION_MIN_IMPROVE", existing_overrides, 0.0001)
            _set_float_step(out_overrides, "ALPHA_DIRECTION_CALIBRATION_MIN_IMPROVE", cur, -0.00005, min_v=0.0, max_v=0.01)
            cur = _get_effective_float("ENTRY_QUALITY_W_DIR_CONF_CHOP", existing_overrides, 0.16)
            _set_float_step(out_overrides, "ENTRY_QUALITY_W_DIR_CONF_CHOP", cur, -0.02, min_v=0.05, max_v=0.50)
            cur = _get_effective_float("LEVERAGE_ENTRY_QUALITY_W_DIR_CONF_CHOP", existing_overrides, 0.16)
            _set_float_step(out_overrides, "LEVERAGE_ENTRY_QUALITY_W_DIR_CONF_CHOP", cur, -0.02, min_v=0.05, max_v=0.50)
            cur = _get_effective_float("ENTRY_QUALITY_W_EDGE_CHOP", existing_overrides, 0.20)
            _set_float_step(out_overrides, "ENTRY_QUALITY_W_EDGE_CHOP", cur, +0.01, min_v=0.05, max_v=0.60)
            cur = _get_effective_float("LEVERAGE_ENTRY_QUALITY_W_EDGE_CHOP", existing_overrides, 0.20)
            _set_float_step(out_overrides, "LEVERAGE_ENTRY_QUALITY_W_EDGE_CHOP", cur, +0.01, min_v=0.05, max_v=0.60)
            cur = _get_effective_float("ENTRY_QUALITY_W_GAP_CHOP", existing_overrides, 0.14)
            _set_float_step(out_overrides, "ENTRY_QUALITY_W_GAP_CHOP", cur, +0.01, min_v=0.05, max_v=0.60)
            cur = _get_effective_float("LEVERAGE_ENTRY_QUALITY_W_GAP_CHOP", existing_overrides, 0.14)
            _set_float_step(out_overrides, "LEVERAGE_ENTRY_QUALITY_W_GAP_CHOP", cur, +0.01, min_v=0.05, max_v=0.60)
            cur = _get_effective_float("ENTRY_QUALITY_W_CONF_CHOP", existing_overrides, 0.50)
            _set_float_step(out_overrides, "ENTRY_QUALITY_W_CONF_CHOP", cur, 0.00, min_v=0.10, max_v=0.80)
            cur = _get_effective_float("LEVERAGE_ENTRY_QUALITY_W_CONF_CHOP", existing_overrides, 0.40)
            _set_float_step(out_overrides, "LEVERAGE_ENTRY_QUALITY_W_CONF_CHOP", cur, +0.005, min_v=0.10, max_v=0.80)

            cur = _get_effective_float("ENTRY_QUALITY_W_DIR_CONF_TREND", existing_overrides, 0.32)
            _set_float_step(out_overrides, "ENTRY_QUALITY_W_DIR_CONF_TREND", cur, +0.01, min_v=0.10, max_v=0.70)
            cur = _get_effective_float("LEVERAGE_ENTRY_QUALITY_W_DIR_CONF_TREND", existing_overrides, 0.34)
            _set_float_step(out_overrides, "LEVERAGE_ENTRY_QUALITY_W_DIR_CONF_TREND", cur, +0.01, min_v=0.10, max_v=0.70)
            cur = _get_effective_float("ENTRY_QUALITY_W_CONF_TREND", existing_overrides, 0.36)
            _set_float_step(out_overrides, "ENTRY_QUALITY_W_CONF_TREND", cur, -0.01, min_v=0.10, max_v=0.80)
            cur = _get_effective_float("LEVERAGE_ENTRY_QUALITY_W_CONF_TREND", existing_overrides, 0.26)
            _set_float_step(out_overrides, "LEVERAGE_ENTRY_QUALITY_W_CONF_TREND", cur, -0.01, min_v=0.10, max_v=0.80)

            # Reduce over-blocking in chop while keeping trend confidence stricter.
            cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_CONF_CHOP", existing_overrides, 0.56)
            _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_CONF_CHOP", cur, -0.005, min_v=0.50, max_v=0.70)
            cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_EDGE_CHOP", existing_overrides, 0.08)
            _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_EDGE_CHOP", cur, +0.002, min_v=0.04, max_v=0.20)
            cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_CONF_TREND", existing_overrides, 0.56)
            _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_CONF_TREND", cur, +0.003, min_v=0.50, max_v=0.75)
            retrain_reason.append("entry_score_driver_regime_reweight")
        elif esd_entry_ev_bad >= 0.06:
            actions.append("entry_score_driver_small_gap_guard")
            cur = _get_effective_float("POLICY_MIN_EV_GAP_CHOP", existing_overrides, 0.0006)
            _set_float_step(out_overrides, "POLICY_MIN_EV_GAP_CHOP", cur, +0.00005, min_v=0.0002, max_v=0.0030)
            cur = _get_effective_float("POLICY_SMALL_GAP_DIR_EDGE_CHOP", existing_overrides, 0.08)
            _set_float_step(out_overrides, "POLICY_SMALL_GAP_DIR_EDGE_CHOP", cur, +0.003, min_v=0.05, max_v=0.20)
            cur = _get_effective_float("POLICY_SMALL_GAP_SIDE_PROB_CHOP", existing_overrides, 0.56)
            _set_float_step(out_overrides, "POLICY_SMALL_GAP_SIDE_PROB_CHOP", cur, +0.003, min_v=0.52, max_v=0.75)

    if delta_bad:
        actions.append("batch_delta_degrade_guard_up")
        cur = _get_effective_int("ALPHA_TRAIN_TRIGGER_NEW_EXITS", existing_overrides, 200)
        _set_int_step(out_overrides, "ALPHA_TRAIN_TRIGGER_NEW_EXITS", cur, -30, min_v=80, max_v=400)
        cur = _get_effective_int("ALPHA_DIRECTION_REGIME_MIN_SAMPLES", existing_overrides, 250)
        _set_int_step(out_overrides, "ALPHA_DIRECTION_REGIME_MIN_SAMPLES", cur, -30, min_v=120, max_v=500)
        cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_CONF", existing_overrides, 0.56)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_CONF", cur, +0.005, min_v=0.52, max_v=0.68)
        cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", existing_overrides, 0.56)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", cur, +0.005, min_v=0.52, max_v=0.68)
        cur = _get_effective_float("POLICY_SMALL_GAP_DIR_EDGE", existing_overrides, 0.08)
        _set_float_step(out_overrides, "POLICY_SMALL_GAP_DIR_EDGE", cur, +0.005, min_v=0.05, max_v=0.18)
        if avg_exit_regret_delta is not None and float(avg_exit_regret_delta) > 0.0:
            cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
            _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=10)
            cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", existing_overrides, 0.60)
            _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", cur, +0.02, min_v=0.45, max_v=0.99)
        out_overrides["ALPHA_DIRECTION_LGBM_TUNE_ENABLED"] = 1
        out_overrides["ALPHA_DIRECTION_HSTAR_LABEL_ENABLED"] = 1
        out_overrides["ALPHA_DIRECTION_LABEL_SOURCE"] = "auto"
        retrain_reason.append("batch_delta_degrade")

    if exit_bad:
        actions.append("exit_early_guard_up")
        cur = _get_effective_int("HYBRID_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "HYBRID_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=6)
        cur = _get_effective_int("HYBRID_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
        _set_int_step(out_overrides, "HYBRID_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=8)
        cur = _get_effective_int("UNIFIED_CASH_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "UNIFIED_CASH_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=6)
        cur = _get_effective_int("UNIFIED_CASH_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
        _set_int_step(out_overrides, "UNIFIED_CASH_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=8)
        cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=8)
        cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
        _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=10)
        cur = _get_effective_int("UNREALIZED_DD_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "UNREALIZED_DD_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=8)
        cur = _get_effective_int("UNREALIZED_DD_CONFIRM_TICKS_NOISE", existing_overrides, 3)
        _set_int_step(out_overrides, "UNREALIZED_DD_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=10)
        cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_RANDOM", existing_overrides, -0.0003)
        _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_RANDOM", cur, -0.0002, min_v=-0.0035, max_v=0.005)
        cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_TREND", existing_overrides, -0.0012)
        _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_TREND", cur, -0.0002, min_v=-0.0040, max_v=0.005)
        cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_CHOP", existing_overrides, -0.0008)
        _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_CHOP", cur, -0.0002, min_v=-0.0040, max_v=0.005)
        cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_VOLATILE", existing_overrides, -0.0003)
        _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_VOLATILE", cur, -0.0002, min_v=-0.0040, max_v=0.005)
        cur = _get_effective_float("EVENT_EXIT_SHOCK_FAST_THRESHOLD", existing_overrides, 1.0)
        _set_float_step(out_overrides, "EVENT_EXIT_SHOCK_FAST_THRESHOLD", cur, +0.10, min_v=0.8, max_v=2.2)
        cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_TREND", existing_overrides, 0.45)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_TREND", cur, +0.03, min_v=0.35, max_v=0.98)
        cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", existing_overrides, 0.60)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", cur, +0.03, min_v=0.45, max_v=0.99)
        cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_MEAN_REVERT", existing_overrides, 0.70)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_MEAN_REVERT", cur, +0.03, min_v=0.50, max_v=0.99)
        cur = _get_effective_float("EVENT_MC_TSTAR_BYPASS_SHOCK", existing_overrides, 1.20)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_BYPASS_SHOCK", cur, +0.05, min_v=0.8, max_v=3.0)
        cur = _get_effective_float("EVENT_MC_TSTAR_BYPASS_PSL", existing_overrides, 0.92)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_BYPASS_PSL", cur, +0.01, min_v=0.70, max_v=0.99)
        cur = _get_effective_float("EXIT_SCORE_DROP", existing_overrides, 0.0004)
        _set_float_step(out_overrides, "EXIT_SCORE_DROP", cur, +0.0001, min_v=0.0, max_v=0.005)
        cur = _get_effective_float("EXIT_RESPECT_ENTRY_MIN_PROGRESS", existing_overrides, 0.65)
        _set_float_step(out_overrides, "EXIT_RESPECT_ENTRY_MIN_PROGRESS", cur, +0.03, min_v=0.45, max_v=0.95)
        cur = _get_effective_float("EXIT_RESPECT_ENTRY_MIN_Q", existing_overrides, 0.72)
        _set_float_step(out_overrides, "EXIT_RESPECT_ENTRY_MIN_Q", cur, +0.01, min_v=0.55, max_v=0.90)
        cur = _get_effective_float("EXIT_RESPECT_ENTRY_MIN_SIGNAL", existing_overrides, 0.70)
        _set_float_step(out_overrides, "EXIT_RESPECT_ENTRY_MIN_SIGNAL", cur, +0.01, min_v=0.55, max_v=0.90)
        if event_reason_ratio >= 0.20:
            cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
            _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=12)
        if unified_flip_ratio >= 0.20:
            cur = _get_effective_int("UNIFIED_CASH_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
            _set_int_step(out_overrides, "UNIFIED_CASH_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=8)
        if ev_drop_ratio >= 0.20:
            cur = _get_effective_float("EXIT_SCORE_DROP", existing_overrides, 0.0004)
            _set_float_step(out_overrides, "EXIT_SCORE_DROP", cur, +0.0001, min_v=0.0, max_v=0.006)
        if dd_reason_ratio >= 0.15:
            cur = _get_effective_int("UNREALIZED_DD_CONFIRM_TICKS_NOISE", existing_overrides, 3)
            _set_int_step(out_overrides, "UNREALIZED_DD_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=12)
            cur = _get_effective_float("UNREALIZED_DD_MULT_NOISE", existing_overrides, 1.12)
            _set_float_step(out_overrides, "UNREALIZED_DD_MULT_NOISE", cur, +0.03, min_v=1.0, max_v=2.0)
        retrain_reason.append("exit_regret_or_early_exit")

    # Reason-level regret-delta -> shock gate tuning (direct link).
    # If event/dd regret worsens batch-over-batch, require more confirmation in shock mode and
    # demand more severe DD before allowing shock exits.
    try:
        regret_delta_tighten = float(os.environ.get("AUTO_TUNE_REGRET_DELTA_TIGHTEN", 0.0008) or 0.0008)
    except Exception:
        regret_delta_tighten = 0.0008
    if (event_regret_delta is not None and float(event_regret_delta) >= float(regret_delta_tighten)) or (
        dd_regret_delta is not None and float(dd_regret_delta) >= float(regret_delta_tighten)
    ):
        actions.append("reason_regret_delta_shock_guard")
        if event_regret_delta is not None and float(event_regret_delta) >= float(regret_delta_tighten):
            cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_SHOCK", existing_overrides, 1)
            _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_SHOCK", cur, +1, min_v=1, max_v=10)
            cur = _get_effective_float("EVENT_EXIT_SHOCK_FAST_THRESHOLD", existing_overrides, 1.0)
            _set_float_step(out_overrides, "EVENT_EXIT_SHOCK_FAST_THRESHOLD", cur, +0.05, min_v=0.8, max_v=2.5)
        if dd_regret_delta is not None and float(dd_regret_delta) >= float(regret_delta_tighten):
            cur = _get_effective_int("UNREALIZED_DD_CONFIRM_TICKS_SHOCK", existing_overrides, 1)
            _set_int_step(out_overrides, "UNREALIZED_DD_CONFIRM_TICKS_SHOCK", cur, +1, min_v=1, max_v=10)
            cur = _get_effective_float("UNREALIZED_DD_SEVERE_MULT", existing_overrides, 1.40)
            _set_float_step(out_overrides, "UNREALIZED_DD_SEVERE_MULT", cur, +0.03, min_v=1.10, max_v=2.00)

    # Reason-matrix guided exit tuning:
    # - event_mc_exit dominates with high early-loss signature -> guard up event fast-exit path.
    # - unified_flip dominates with loss/regret signature -> require stronger reversal proof.
    # - exit timing share dominates over direction mismatch -> strengthen hold-progress + confirmations.
    rm_event_n = int(_safe_int(rm_event.get("n"), 0))
    rm_event_roe = _safe_float(rm_event.get("avg_roe"), None)
    rm_event_early = _safe_float(rm_event.get("early_like_rate"), None)
    rm_event_flip = _safe_float(rm_event.get("profitable_side_loss_rate"), None)
    rm_event_miss = _safe_float(rm_event.get("direction_miss_rate"), None)
    if rm_event_n >= 60 and (
        (rm_event_roe is not None and rm_event_roe <= -0.003)
        or (rm_event_early is not None and rm_event_early >= 0.85)
        or (rm_event_flip is not None and rm_event_flip >= 0.45)
    ):
        actions.append("reason_matrix_event_mc_exit_guard")
        cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=10)
        cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
        _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=12)
        cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_TREND", existing_overrides, 0.45)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_TREND", cur, +0.01, min_v=0.35, max_v=0.99)
        cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", existing_overrides, 0.60)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", cur, +0.02, min_v=0.45, max_v=0.99)
        cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_MEAN_REVERT", existing_overrides, 0.70)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_MEAN_REVERT", cur, +0.02, min_v=0.50, max_v=0.99)
        cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_CHOP", existing_overrides, -0.0008)
        _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_CHOP", cur, -0.0002, min_v=-0.0050, max_v=0.0050)
        cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_RANDOM", existing_overrides, -0.0003)
        _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_RANDOM", cur, -0.0002, min_v=-0.0050, max_v=0.0050)
        if rm_event_miss is not None and rm_event_miss >= 0.55:
            cur = _get_effective_float("EVENT_EXIT_SHOCK_FAST_THRESHOLD", existing_overrides, 1.0)
            _set_float_step(out_overrides, "EVENT_EXIT_SHOCK_FAST_THRESHOLD", cur, +0.10, min_v=0.8, max_v=2.5)
        retrain_reason.append("reason_matrix_event_mc_exit")

    rm_unified_n = int(_safe_int(rm_unified.get("n"), 0))
    rm_unified_roe = _safe_float(rm_unified.get("avg_roe"), None)
    rm_unified_regret = _safe_float(rm_unified.get("avg_exit_regret"), None)
    rm_unified_miss = _safe_float(rm_unified.get("direction_miss_rate"), None)
    if rm_unified_n >= 50 and (
        (rm_unified_roe is not None and rm_unified_roe <= -0.003)
        or (rm_unified_regret is not None and rm_unified_regret >= 0.012)
        or (rm_unified_miss is not None and rm_unified_miss >= 0.50)
    ):
        actions.append("reason_matrix_unified_flip_guard")
        cur = _get_effective_int("UNIFIED_FLIP_CONFIRM_TICKS_NORMAL", existing_overrides, 3)
        _set_int_step(out_overrides, "UNIFIED_FLIP_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=10)
        cur = _get_effective_int("UNIFIED_FLIP_CONFIRM_TICKS_NOISE", existing_overrides, 4)
        _set_int_step(out_overrides, "UNIFIED_FLIP_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=12)
        cur = _get_effective_float("UNIFIED_FLIP_MIN_REVERSE_EDGE", existing_overrides, 0.0008)
        _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_REVERSE_EDGE", cur, +0.0002, min_v=0.0002, max_v=0.0100)
        cur = _get_effective_float("UNIFIED_FLIP_MIN_OPPOSITE_SIDE_PROB", existing_overrides, 0.56)
        _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_OPPOSITE_SIDE_PROB", cur, +0.01, min_v=0.50, max_v=0.90)
        cur = _get_effective_float("UNIFIED_FLIP_MIN_DIR_CONF", existing_overrides, 0.58)
        _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_DIR_CONF", cur, +0.01, min_v=0.50, max_v=0.90)
        cur = _get_effective_float("UNIFIED_FLIP_MIN_DIR_EDGE", existing_overrides, 0.06)
        _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_DIR_EDGE", cur, +0.005, min_v=0.03, max_v=0.25)
        cur = _get_effective_float("UNIFIED_FLIP_MIN_PROGRESS_RANDOM", existing_overrides, 0.75)
        _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_PROGRESS_RANDOM", cur, +0.02, min_v=0.45, max_v=0.99)
        cur = _get_effective_float("UNIFIED_FLIP_MIN_PROGRESS_MEAN_REVERT", existing_overrides, 0.85)
        _set_float_step(out_overrides, "UNIFIED_FLIP_MIN_PROGRESS_MEAN_REVERT", cur, +0.01, min_v=0.55, max_v=0.99)
        retrain_reason.append("reason_matrix_unified_flip")

    rm_hold_n = int(_safe_int(rm_hold.get("n"), 0))
    rm_hold_roe = _safe_float(rm_hold.get("avg_roe"), None)
    rm_hold_early = _safe_float(rm_hold.get("early_like_rate"), None)
    if rm_hold_n >= 40 and (
        (rm_hold_roe is not None and rm_hold_roe <= -0.0025)
        or (rm_hold_early is not None and rm_hold_early >= 0.60)
    ):
        actions.append("reason_matrix_hold_vs_exit_guard")
        cur = _get_effective_float("HOLD_EVAL_MIN_PROGRESS_TO_EXIT", existing_overrides, 0.85)
        _set_float_step(out_overrides, "HOLD_EVAL_MIN_PROGRESS_TO_EXIT", cur, +0.03, min_v=0.45, max_v=1.20)
        cur = _get_effective_int("HOLD_EVAL_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "HOLD_EVAL_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=12)
        cur = _get_effective_int("HOLD_EVAL_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
        _set_int_step(out_overrides, "HOLD_EVAL_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=14)
        cur = _get_effective_float("HOLD_EVAL_EXIT_MARGIN", existing_overrides, 0.0)
        _set_float_step(out_overrides, "HOLD_EVAL_EXIT_MARGIN", cur, -0.0001, min_v=-0.0050, max_v=0.0050)

    rm_external_n = int(_safe_int(rm_external.get("n"), 0))
    rm_external_roe = _safe_float(rm_external.get("avg_roe"), None)
    rm_external_loss = _safe_float(rm_external.get("profitable_side_loss_rate"), None)
    if rm_external_n >= 12 and (
        (rm_external_roe is not None and rm_external_roe <= -0.0015)
        or (rm_external_loss is not None and rm_external_loss >= 0.45)
    ):
        actions.append("reason_matrix_external_sync_guard")
        cur = _get_effective_int("LIVE_LIQUIDATION_MISS_COUNT", existing_overrides, 2)
        _set_int_step(out_overrides, "LIVE_LIQUIDATION_MISS_COUNT", cur, +1, min_v=1, max_v=8)
        cur = _get_effective_float("EXTERNAL_CLOSE_LIQ_BUFFER_PCT", existing_overrides, 0.0030)
        _set_float_step(out_overrides, "EXTERNAL_CLOSE_LIQ_BUFFER_PCT", cur, -0.0002, min_v=0.0005, max_v=0.0200)

    if rm_exit_timing_total_share >= 0.45:
        actions.append("reason_matrix_exit_timing_dominant_guard")
        cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=10)
        cur = _get_effective_int("UNIFIED_CASH_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "UNIFIED_CASH_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=10)
        cur = _get_effective_float("EXIT_SCORE_DROP", existing_overrides, 0.0004)
        _set_float_step(out_overrides, "EXIT_SCORE_DROP", cur, +0.0001, min_v=0.0, max_v=0.0060)

    if risk_bad:
        actions.append("tail_risk_guard_up")
        min_target_max_floor = _safe_float(os.environ.get("LEVERAGE_AUTOTUNE_MIN_TARGET_MAX", 25.0), 25.0) or 25.0
        min_dynamic_scale_floor = _safe_float(os.environ.get("LEVERAGE_AUTOTUNE_MIN_DYNAMIC_SCALE", 0.25), 0.25) or 0.25
        min_vpin_soft_floor = _safe_float(os.environ.get("LEVERAGE_AUTOTUNE_MIN_VPIN_SOFT", 0.80), 0.80) or 0.80
        min_signal_blend_floor = _safe_float(os.environ.get("LEVERAGE_AUTOTUNE_MIN_SIGNAL_BLEND", 0.60), 0.60) or 0.60
        max_signal_pow_cap = _safe_float(os.environ.get("LEVERAGE_AUTOTUNE_MAX_SIGNAL_POW", 1.40), 1.40) or 1.40
        min_target_max_floor = max(8.0, min(50.0, float(min_target_max_floor)))
        min_dynamic_scale_floor = max(0.10, min(0.60, float(min_dynamic_scale_floor)))
        min_vpin_soft_floor = max(0.52, min(0.95, float(min_vpin_soft_floor)))
        min_signal_blend_floor = max(0.20, min(0.90, float(min_signal_blend_floor)))
        max_signal_pow_cap = max(0.60, min(2.50, float(max_signal_pow_cap)))
        cur = _get_effective_float("LEVERAGE_TARGET_MAX", existing_overrides, 15.0)
        _set_float_step(out_overrides, "LEVERAGE_TARGET_MAX", cur, -1.0, min_v=min_target_max_floor, max_v=50.0)
        cur = _get_effective_float("LEVERAGE_DYNAMIC_MIN_SCALE", existing_overrides, 0.20)
        _set_float_step(out_overrides, "LEVERAGE_DYNAMIC_MIN_SCALE", cur, +0.02, min_v=min_dynamic_scale_floor, max_v=0.60)
        cur = _get_effective_float("LEVERAGE_VPIN_SOFT", existing_overrides, 0.65)
        _set_float_step(out_overrides, "LEVERAGE_VPIN_SOFT", cur, +0.01, min_v=min_vpin_soft_floor, max_v=0.95)
        cur = _get_effective_float("LEVERAGE_SIGNAL_BLEND", existing_overrides, 0.55)
        _set_float_step(out_overrides, "LEVERAGE_SIGNAL_BLEND", cur, +0.03, min_v=min_signal_blend_floor, max_v=0.90)
        cur = _get_effective_float("LEVERAGE_SIGNAL_SCORE_POW", existing_overrides, 1.20)
        _set_float_step(out_overrides, "LEVERAGE_SIGNAL_SCORE_POW", cur, -0.05, min_v=0.50, max_v=max_signal_pow_cap)
        cur = _get_effective_float("UNREALIZED_DD_SEVERE_MULT", existing_overrides, 1.40)
        _set_float_step(out_overrides, "UNREALIZED_DD_SEVERE_MULT", cur, -0.05, min_v=1.10, max_v=2.00)
        cur = _get_effective_float("LEVERAGE_LOW_CONF_CAP", existing_overrides, 3.0)
        _set_float_step(out_overrides, "LEVERAGE_LOW_CONF_CAP", cur, -0.25, min_v=1.0, max_v=8.0)
        cur = _get_effective_float("LEVERAGE_HIGH_VPIN_CAP", existing_overrides, 2.0)
        _set_float_step(out_overrides, "LEVERAGE_HIGH_VPIN_CAP", cur, -0.25, min_v=1.0, max_v=6.0)
        retrain_reason.append("tail_risk_control")
    elif direction_good:
        actions.append("quality_leverage_expand")
        cur = _get_effective_float("LEVERAGE_TARGET_MAX", existing_overrides, 50.0)
        _set_float_step(out_overrides, "LEVERAGE_TARGET_MAX", cur, +1.0, min_v=8.0, max_v=50.0)
        cur = _get_effective_float("LEVERAGE_SIGNAL_BLEND", existing_overrides, 0.55)
        _set_float_step(out_overrides, "LEVERAGE_SIGNAL_BLEND", cur, +0.03, min_v=0.20, max_v=0.90)
        cur = _get_effective_float("LEVERAGE_SIGNAL_SCORE_POW", existing_overrides, 1.20)
        _set_float_step(out_overrides, "LEVERAGE_SIGNAL_SCORE_POW", cur, -0.05, min_v=0.50, max_v=2.50)
        cur = _get_effective_float("LEVERAGE_LOW_CONF_CAP", existing_overrides, 3.0)
        _set_float_step(out_overrides, "LEVERAGE_LOW_CONF_CAP", cur, +0.25, min_v=1.0, max_v=10.0)
        cur = _get_effective_float("LEVERAGE_HIGH_VPIN_CAP", existing_overrides, 2.0)
        _set_float_step(out_overrides, "LEVERAGE_HIGH_VPIN_CAP", cur, +0.25, min_v=1.0, max_v=8.0)

    # Regime-aware secondary tuning loop.
    # - direction quality -> tighten small-gap gates by regime
    # - early exits/regret -> relax regime-specific EVENT_EXIT score thresholds
    # - liquidation/drawdown pressure -> tighten UNREALIZED_DD cut by regime
    min_regime_samples = int(max(1, args.min_regime_samples))
    for reg in ("trend", "chop", "volatile"):
        rm = regime_metrics.get(reg) if isinstance(regime_metrics, dict) else {}
        if not isinstance(rm, dict):
            continue
        n = int(rm.get("exit_n", 0) or 0)
        if n < min_regime_samples:
            continue
        loss_rate_reg = _safe_float(rm.get("loss_rate"), None)
        event_rate_reg = _safe_float(rm.get("event_rate"), None)
        dd_rate_reg = _safe_float(rm.get("dd_rate"), None)
        liq_rate_reg = _safe_float(rm.get("liq_rate"), None)
        avg_roe_reg = _safe_float(rm.get("avg_roe"), None)
        reject_rate_reg = _safe_float(rm.get("reject_rate"), None)
        reject_110007_rate_reg = _safe_float(rm.get("reject_110007_rate"), None)
        precision_reject_rate_reg = _safe_float(rm.get("precision_reject_rate"), None)
        reg_key = str(reg).upper()

        if direction_bad and loss_rate_reg is not None and loss_rate_reg >= 0.52:
            actions.append(f"regime_direction_guard:{reg}")
            for key, step, lo, hi, dflt in (
                (f"POLICY_SMALL_GAP_CONFIDENCE_{reg_key}", +0.01, 0.55, 0.84, 0.62),
                (f"POLICY_SMALL_GAP_DIR_CONFIDENCE_{reg_key}", +0.01, 0.52, 0.80, 0.58),
                (f"POLICY_SMALL_GAP_DIR_EDGE_{reg_key}", +0.005, 0.05, 0.20, 0.08),
                (f"POLICY_SMALL_GAP_SIDE_PROB_{reg_key}", +0.01, 0.50, 0.80, 0.56),
            ):
                cur_r = _get_effective_float(key, existing_overrides, dflt)
                _set_float_step(out_overrides, key, cur_r, step, min_v=lo, max_v=hi)

        if event_rate_reg is not None and event_rate_reg >= 0.30:
            ev_key = f"EVENT_EXIT_SCORE_OFFSET_{reg_key}"
            default_off = -0.0008 if reg == "chop" else (-0.0012 if reg == "trend" else -0.0003)
            cur_off = _get_effective_float(ev_key, existing_overrides, default_off)
            if avg_roe_reg is not None and avg_roe_reg < -0.002 and loss_rate_reg is not None and loss_rate_reg >= 0.55:
                actions.append(f"regime_exit_tighten:{reg}")
                _set_float_step(out_overrides, ev_key, cur_off, +0.0002, min_v=-0.0050, max_v=0.0050)
            elif exit_bad:
                actions.append(f"regime_exit_guard:{reg}")
                _set_float_step(out_overrides, ev_key, cur_off, -0.0002, min_v=-0.0050, max_v=0.0050)

        # Regime-specific entry net expectancy floor tuning:
        # negative regime expectancy -> tighten floor, stable positive regime -> relax floor.
        if avg_roe_reg is not None and (avg_roe_reg < -0.0015 or (loss_rate_reg is not None and loss_rate_reg >= 0.58)):
            actions.append(f"regime_net_expectancy_guard:{reg}")
            for key, step, lo, hi, dflt in (
                (f"ENTRY_NET_EXPECTANCY_MIN_{reg_key}", +0.00015, -0.0030, 0.0050, 0.0),
                (f"ENTRY_NET_EXPECTANCY_VPIN_BUMP_{reg_key}", +0.00008, 0.0, 0.0030, 0.0004),
                (f"ENTRY_NET_EXPECTANCY_DIR_CONF_BUMP_{reg_key}", +0.00008, 0.0, 0.0030, 0.0004),
                (f"ENTRY_NET_EXPECTANCY_DIR_CONF_REF_{reg_key}", +0.0100, 0.50, 0.86, 0.58),
            ):
                cur_r = _get_effective_float(key, existing_overrides, dflt)
                _set_float_step(out_overrides, key, cur_r, step, min_v=lo, max_v=hi)
        elif (
            avg_roe_reg is not None
            and avg_roe_reg > 0.002
            and loss_rate_reg is not None
            and loss_rate_reg <= 0.45
            and not direction_bad
        ):
            actions.append(f"regime_net_expectancy_relax:{reg}")
            for key, step, lo, hi, dflt in (
                (f"ENTRY_NET_EXPECTANCY_MIN_{reg_key}", -0.00008, -0.0030, 0.0050, 0.0),
                (f"ENTRY_NET_EXPECTANCY_VPIN_BUMP_{reg_key}", -0.00005, 0.0, 0.0030, 0.0004),
                (f"ENTRY_NET_EXPECTANCY_DIR_CONF_BUMP_{reg_key}", -0.00005, 0.0, 0.0030, 0.0004),
            ):
                cur_r = _get_effective_float(key, existing_overrides, dflt)
                _set_float_step(out_overrides, key, cur_r, step, min_v=lo, max_v=hi)

        # Symbol/time quality guard by reject pressure.
        if reject_rate_reg is not None and reject_rate_reg >= 0.35:
            actions.append(f"regime_reject_guard:{reg}")
            cur = _get_effective_float("SYMBOL_QUALITY_REJECT_PENALTY", existing_overrides, 0.18)
            _set_float_step(out_overrides, "SYMBOL_QUALITY_REJECT_PENALTY", cur, +0.02, min_v=0.05, max_v=0.60)
            cur = _get_effective_float("SYMBOL_QUALITY_REJECT_HARD", existing_overrides, 0.55)
            _set_float_step(out_overrides, "SYMBOL_QUALITY_REJECT_HARD", cur, -0.02, min_v=0.20, max_v=0.90)
            cur = _get_effective_float("SYMBOL_QUALITY_MIN_SCORE", existing_overrides, 0.42)
            _set_float_step(out_overrides, "SYMBOL_QUALITY_MIN_SCORE", cur, +0.01, min_v=0.30, max_v=0.85)
        if reject_110007_rate_reg is not None and reject_110007_rate_reg >= 0.30:
            actions.append(f"regime_balance_reject_guard:{reg}")
            cur = _get_effective_float("LEVERAGE_TARGET_MAX", existing_overrides, 15.0)
            _set_float_step(out_overrides, "LEVERAGE_TARGET_MAX", cur, -1.0, min_v=8.0, max_v=50.0)
            cur = _get_effective_float("LEVERAGE_BALANCE_REJECT_SCALE", existing_overrides, 0.82)
            _set_float_step(out_overrides, "LEVERAGE_BALANCE_REJECT_SCALE", cur, -0.03, min_v=0.35, max_v=0.95)
            cur = _get_effective_float("LEVERAGE_BALANCE_REJECT_MIN_SCALE", existing_overrides, 0.45)
            _set_float_step(out_overrides, "LEVERAGE_BALANCE_REJECT_MIN_SCALE", cur, -0.03, min_v=0.20, max_v=0.90)
        if precision_reject_rate_reg is not None and precision_reject_rate_reg >= 0.15:
            actions.append(f"regime_precision_reject_guard:{reg}")
            cur = _get_effective_float("LIVE_MIN_QTY_MAX_UPSIZE", existing_overrides, 5.0)
            _set_float_step(out_overrides, "LIVE_MIN_QTY_MAX_UPSIZE", cur, +0.5, min_v=1.0, max_v=20.0)
            key = f"LIVE_MIN_QTY_MAX_UPSIZE_{reg_key}"
            cur = _get_effective_float(key, existing_overrides, 5.0)
            _set_float_step(out_overrides, key, cur, +0.5, min_v=1.0, max_v=20.0)

        # Regime 분류 정확도 기반 보수화: 정확도가 낮으면 해당 regime 진입 임계치 상향
        regime_accuracy_reg = _safe_float(rm.get("regime_accuracy"), None)
        regime_dir_hit_reg = _safe_float(rm.get("direction_hit"), None)
        regime_eval_n_reg = int(rm.get("regime_eval_n", 0) or 0)
        if regime_eval_n_reg >= 10 and regime_accuracy_reg is not None and regime_accuracy_reg < 0.40:
            actions.append(f"regime_accuracy_low_guard:{reg}")
            # 분류 정확도가 40% 미만이면 해당 regime에서 진입 기준 강화
            for key, step, lo, hi, dflt in (
                (f"UNIFIED_ENTRY_FLOOR_{reg_key}", +0.0002, -0.005, 0.010, 0.0),
                (f"POLICY_SMALL_GAP_CONFIDENCE_{reg_key}", +0.02, 0.55, 0.84, 0.62),
            ):
                cur_r = _get_effective_float(key, existing_overrides, dflt)
                _set_float_step(out_overrides, key, cur_r, step, min_v=lo, max_v=hi)
        elif regime_eval_n_reg >= 10 and regime_accuracy_reg is not None and regime_accuracy_reg >= 0.65:
            # 분류 정확도가 높으면 해당 regime에서 진입 기준 완화
            actions.append(f"regime_accuracy_good_relax:{reg}")
            for key, step, lo, hi, dflt in (
                (f"UNIFIED_ENTRY_FLOOR_{reg_key}", -0.0001, -0.005, 0.010, 0.0),
            ):
                cur_r = _get_effective_float(key, existing_overrides, dflt)
                _set_float_step(out_overrides, key, cur_r, step, min_v=lo, max_v=hi)

        # regime별 방향 적중률이 30% 미만이면 해당 regime에서 레버리지 축소
        if regime_eval_n_reg >= 10 and regime_dir_hit_reg is not None and regime_dir_hit_reg < 0.30:
            actions.append(f"regime_dir_hit_low_guard:{reg}")
            lev_key = f"LEVERAGE_TARGET_MAX_{reg_key}"
            cur_lev_r = _get_effective_float(lev_key, existing_overrides, 15.0)
            _set_float_step(out_overrides, lev_key, cur_lev_r, -1.0, min_v=3.0, max_v=50.0)

        # Tighten DD floor only when tail risk is actually elevated in that regime.
        if (risk_bad or (liq_rate_reg is not None and liq_rate_reg >= 0.02)) and (
            (dd_rate_reg is not None and dd_rate_reg >= 0.08) or (liq_rate_reg is not None and liq_rate_reg >= 0.02)
        ):
            actions.append(f"regime_dd_guard:{reg}")
            dd_key = f"UNREALIZED_DD_REGIME_MULT_{reg_key}"
            cur_dd = _get_effective_float(dd_key, existing_overrides, 1.0)
            _set_float_step(out_overrides, dd_key, cur_dd, -0.05, min_v=0.60, max_v=1.40)
        elif (
            exit_bad
            and avg_exit_regret is not None
            and avg_exit_regret >= float(args.high_exit_regret)
            and dd_rate_reg is not None
            and dd_rate_reg >= 0.10
            and avg_roe_reg is not None
            and avg_roe_reg > 0.0
        ):
            # If DD exits are frequent despite positive regime ROE, slightly loosen to reduce premature cuts.
            actions.append(f"regime_dd_relax:{reg}")
            dd_key = f"UNREALIZED_DD_REGIME_MULT_{reg_key}"
            cur_dd = _get_effective_float(dd_key, existing_overrides, 1.0)
            _set_float_step(out_overrides, dd_key, cur_dd, +0.03, min_v=0.60, max_v=1.40)

    if capital_stage_enabled and capital_usdt is not None:
        stage_cap_lev = float(max(1.0, min(50.0, stage_lev_max[stage_idx])))
        stage_cap_total = float(max(0.1, stage_total_caps[stage_idx]))
        stage_cap_lc = float(max(1.0, min(stage_cap_lev, stage_low_conf_cap[stage_idx])))
        stage_cap_vpin = float(max(1.0, min(stage_cap_lev, stage_high_vpin_cap[stage_idx])))
        lev_cur = _safe_float(out_overrides.get("LEVERAGE_TARGET_MAX"), None)
        if lev_cur is None:
            lev_cur = _get_effective_float("LEVERAGE_TARGET_MAX", existing_overrides, stage_cap_lev)
        if float(lev_cur) > stage_cap_lev + 1e-12:
            out_overrides["LEVERAGE_TARGET_MAX"] = float(stage_cap_lev)
            actions.append("capital_stage_leverage_clamp")
        out_overrides["LEVERAGE_LOW_CONF_CAP"] = float(stage_cap_lc)
        out_overrides["LEVERAGE_HIGH_VPIN_CAP"] = float(stage_cap_vpin)
        out_overrides["MAX_NOTIONAL_EXPOSURE"] = float(stage_cap_total)
        out_overrides["LIVE_MAX_NOTIONAL_EXPOSURE"] = float(stage_cap_total)
        out_overrides["KELLY_TOTAL_EXPOSURE"] = float(stage_cap_total)

    if not actions:
        actions.append("no_change:metrics_within_thresholds")

    payload_overrides: dict[str, Any] = {}
    runtime_aliases_out: dict[str, list[str]] = {}
    if out_overrides:
        payload_raw = {
            k: _sanitize_override_value(v)
            for k, v in out_overrides.items()
            if _sanitize_override_value(v) is not None
        }
        payload_overrides, runtime_aliases_out = _expand_runtime_aliases(payload_raw)
        if runtime_aliases_out:
            actions.append(f"runtime_key_alias_expand:{sum(len(v) for v in runtime_aliases_out.values())}")

    # Recovery mode: allow entry-gate relaxation only.
    # Tightening (higher value) of UNIFIED_ENTRY_FLOOR / ENTRY_NET_EXPECTANCY_* is blocked.
    recovery_relax_only = bool(
        _is_true(os.environ.get("AUTO_TUNE_RECOVERY_ENTRY_RELAX_ONLY", "0"))
        or _is_true(existing_overrides.get("AUTO_TUNE_RECOVERY_ENTRY_RELAX_ONLY"))
    )
    recovery_blocked: list[str] = []
    if recovery_relax_only and payload_overrides:
        guarded_exact = {"UNIFIED_ENTRY_FLOOR"}
        for key in list(payload_overrides.keys()):
            if (key not in guarded_exact) and (not str(key).startswith("ENTRY_NET_EXPECTANCY_")):
                continue
            nxt = _safe_float(payload_overrides.get(key), None)
            if nxt is None:
                continue
            cur = _get_effective_float(key, existing_overrides, float(nxt))
            # For these keys, larger value means stricter entry gate.
            if float(nxt) > float(cur) + 1e-12:
                payload_overrides.pop(key, None)
                recovery_blocked.append(str(key))
        if recovery_blocked:
            actions.append(f"recovery_relax_only_blocked:{len(recovery_blocked)}")
    result["recovery_relax_only"] = {
        "enabled": bool(recovery_relax_only),
        "blocked_count": int(len(recovery_blocked)),
        "blocked_keys": sorted(set(recovery_blocked)),
    }

    retrain_requested = bool(direction_bad or exit_bad or delta_bad)
    retrain_info = {
        "requested": bool(retrain_requested),
        "reason": ",".join(sorted(set(retrain_reason))) if retrain_reason else "",
        "ran": False,
        "rc": None,
        "elapsed_sec": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if retrain_requested and int(args.run_retrain) == 1 and not args.dry_run:
        r = _run_retrain(repo_root=repo_root, timeout_sec=float(args.retrain_timeout_sec))
        retrain_info.update(r)
        actions.append(f"retrain:rc={retrain_info.get('rc')}")

    override_payload = {
        "timestamp_ms": _now_ms(),
        "source": "auto_reval_policy_tuner",
        "batch_id": int(batch_id),
        "actions": list(actions),
        "metrics": metrics,
        "regime_metrics": regime_metrics,
        "reason_breakdown": {
            "event_reason_ratio": float(event_reason_ratio),
            "unified_flip_ratio": float(unified_flip_ratio),
            "ev_drop_ratio": float(ev_drop_ratio),
            "dd_reason_ratio": float(dd_reason_ratio),
            "reason_counts": dict(reason_counts),
            "reason_total": int(total_reason_n),
        },
        "reason_matrix_metrics": reason_matrix_metrics,
        "reval_loss_driver_metrics": reval_loss_driver_metrics,
        "overrides": {k: _sanitize_override_value(v) for k, v in payload_overrides.items()},
        "runtime_key_aliases_out": runtime_aliases_out,
        "retrain": {
            "requested": bool(retrain_info.get("requested")),
            "ran": bool(retrain_info.get("ran")),
            "rc": retrain_info.get("rc"),
            "elapsed_sec": retrain_info.get("elapsed_sec"),
            "reason": retrain_info.get("reason"),
        },
    }

    if not args.dry_run:
        _write_json(override_out_path, override_payload)
        _write_json(
            state_file_path,
            {
                "timestamp_ms": _now_ms(),
                "last_batch_id": int(batch_id),
                "last_actions": list(actions),
                "last_retrain_rc": retrain_info.get("rc"),
                "last_retrain_ran": bool(retrain_info.get("ran")),
                "last_metrics": {
                    "direction_hit": _safe_float(metrics.get("direction_hit"), None),
                    "entry_issue_ratio": _safe_float(metrics.get("entry_issue_ratio"), None),
                    "avg_exit_regret": _safe_float(metrics.get("avg_exit_regret"), None),
                },
            },
        )

    result["actions"] = list(actions)
    result["overrides"] = dict(payload_overrides)
    result["override_count"] = int(len(payload_overrides))
    result["runtime_key_aliases_out"] = dict(runtime_aliases_out)
    result["retrain"] = dict(retrain_info)
    result["applied"] = bool((not args.dry_run) and bool(payload_overrides))

    _write_json(status_out_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

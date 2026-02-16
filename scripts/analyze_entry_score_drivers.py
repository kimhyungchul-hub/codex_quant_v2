#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EntryRow:
    id: int
    symbol: str
    side: str
    ts_ms: int
    entry_link_id: str | None
    entry_ev: float | None
    pred_mu_alpha: float | None
    pred_mu_dir_conf: float | None
    alpha_vpin: float | None
    alpha_hurst: float | None
    policy_score_threshold: float | None
    policy_event_exit_min_score: float | None
    policy_unrealized_dd_floor: float | None
    entry_quality_score: float | None
    one_way_move_score: float | None
    leverage_signal_score: float | None
    raw: dict[str, Any]


@dataclass
class ExitRow:
    id: int
    symbol: str
    side: str
    ts_ms: int
    entry_link_id: str | None
    roe: float | None
    realized_pnl: float | None
    notional: float | None
    entry_reason: str
    raw: dict[str, Any]


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _safe_rate(n: int, d: int) -> float | None:
    return float(n / d) if d > 0 else None


def _sign(v: float | None) -> int:
    if v is None:
        return 0
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _to_dict_json(raw_txt: Any) -> dict[str, Any]:
    if not raw_txt:
        return {}
    try:
        obj = json.loads(raw_txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _target_value(ex: ExitRow, target_mode: str) -> float | None:
    mode = str(target_mode or "pre_roe").strip().lower()
    if mode in ("roe", "raw_roe", "realized_roe"):
        return _safe_float(ex.roe, None)
    notional = _safe_float(ex.notional, None)
    pnl = _safe_float(ex.realized_pnl, None)
    if notional is not None and abs(float(notional)) > 1e-12 and pnl is not None:
        return float(pnl / abs(float(notional)))
    # Fallback when notional is missing: de-leverage ROE.
    lev = _safe_float(ex.raw.get("entry_leverage"), None)
    if lev is None:
        lev = _safe_float(ex.raw.get("leverage_effective"), None)
    if lev is None:
        lev = _safe_float(ex.raw.get("leverage"), None)
    roe = _safe_float(ex.roe, None)
    if roe is None:
        return None
    if lev is None or abs(float(lev)) <= 1e-9:
        return None
    return float(roe / float(lev))


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    arr = sorted(float(v) for v in vals)
    if len(arr) == 1:
        return arr[0]
    pos = max(0.0, min(1.0, float(p))) * (len(arr) - 1)
    i0 = int(math.floor(pos))
    i1 = int(math.ceil(pos))
    if i0 == i1:
        return arr[i0]
    w = pos - i0
    return arr[i0] * (1.0 - w) + arr[i1] * w


def _bin_idx(v: float, edges: list[float]) -> int:
    # 0..len(edges)
    for i, e in enumerate(edges):
        if v <= e:
            return i
    return len(edges)


def _load_rows(db_path: Path, since_id: int, recent_exits: int) -> tuple[list[EntryRow], list[ExitRow]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    entry_rows = cur.execute(
        """
        SELECT
          id, symbol, side, timestamp_ms, entry_link_id,
          entry_ev, pred_mu_alpha, pred_mu_dir_conf,
          alpha_vpin, alpha_hurst,
          policy_score_threshold, policy_event_exit_min_score, policy_unrealized_dd_floor,
          entry_quality_score, one_way_move_score, leverage_signal_score,
          raw_data
        FROM trades
        WHERE action IN ('ENTER','SPREAD')
        ORDER BY timestamp_ms ASC, id ASC
        """
    ).fetchall()

    exit_rows = cur.execute(
        """
        SELECT id, symbol, side, timestamp_ms, entry_link_id, roe, realized_pnl, notional, entry_reason, raw_data
        FROM trades
        WHERE action IN ('EXIT','REBAL_EXIT')
          AND id > ?
        ORDER BY timestamp_ms DESC, id DESC
        LIMIT ?
        """,
        (int(max(0, since_id)), int(max(1, recent_exits))),
    ).fetchall()
    conn.close()

    entries: list[EntryRow] = []
    for r in entry_rows:
        entries.append(
            EntryRow(
                id=int(r["id"]),
                symbol=str(r["symbol"]),
                side=str(r["side"]).upper(),
                ts_ms=int(r["timestamp_ms"]),
                entry_link_id=str(r["entry_link_id"]).strip() if r["entry_link_id"] is not None and str(r["entry_link_id"]).strip() else None,
                entry_ev=_safe_float(r["entry_ev"]),
                pred_mu_alpha=_safe_float(r["pred_mu_alpha"]),
                pred_mu_dir_conf=_safe_float(r["pred_mu_dir_conf"]),
                alpha_vpin=_safe_float(r["alpha_vpin"]),
                alpha_hurst=_safe_float(r["alpha_hurst"]),
                policy_score_threshold=_safe_float(r["policy_score_threshold"]),
                policy_event_exit_min_score=_safe_float(r["policy_event_exit_min_score"]),
                policy_unrealized_dd_floor=_safe_float(r["policy_unrealized_dd_floor"]),
                entry_quality_score=_safe_float(r["entry_quality_score"]),
                one_way_move_score=_safe_float(r["one_way_move_score"]),
                leverage_signal_score=_safe_float(r["leverage_signal_score"]),
                raw=_to_dict_json(r["raw_data"]),
            )
        )

    exits: list[ExitRow] = []
    for r in list(exit_rows)[::-1]:
        exits.append(
            ExitRow(
                id=int(r["id"]),
                symbol=str(r["symbol"]),
                side=str(r["side"]).upper(),
                ts_ms=int(r["timestamp_ms"]),
                entry_link_id=str(r["entry_link_id"]).strip() if r["entry_link_id"] is not None and str(r["entry_link_id"]).strip() else None,
                roe=_safe_float(r["roe"]),
                realized_pnl=_safe_float(r["realized_pnl"]),
                notional=_safe_float(r["notional"]),
                entry_reason=str(r["entry_reason"] or ""),
                raw=_to_dict_json(r["raw_data"]),
            )
        )

    return entries, exits


def _match(entries: list[EntryRow], exits: list[ExitRow], fallback_lookback_sec: int) -> list[tuple[EntryRow, ExitRow, str]]:
    by_link: dict[str, EntryRow] = {}
    by_sym_side: dict[tuple[str, str], list[EntryRow]] = {}
    for e in entries:
        if e.entry_link_id:
            by_link[e.entry_link_id] = e
        key = (e.symbol, e.side)
        by_sym_side.setdefault(key, []).append(e)
    for arr in by_sym_side.values():
        arr.sort(key=lambda x: (x.ts_ms, x.id))

    out: list[tuple[EntryRow, ExitRow, str]] = []
    lookback_ms = int(max(60, fallback_lookback_sec)) * 1000
    for ex in exits:
        if ex.entry_link_id and ex.entry_link_id in by_link:
            out.append((by_link[ex.entry_link_id], ex, "entry_link_id"))
            continue
        arr = by_sym_side.get((ex.symbol, ex.side))
        if not arr:
            continue
        # fallback: nearest previous entry in same symbol/side
        i = len(arr) - 1
        while i >= 0 and arr[i].ts_ms > ex.ts_ms:
            i -= 1
        if i < 0:
            continue
        en = arr[i]
        if ex.ts_ms - en.ts_ms <= lookback_ms:
            out.append((en, ex, "fallback_symbol_side"))
    return out


def _pick_feature(en: EntryRow, name: str) -> float | None:
    if name == "entry_ev":
        return en.entry_ev
    if name == "entry_ev_excess":
        if en.entry_ev is None or en.policy_score_threshold is None:
            return None
        return float(en.entry_ev - en.policy_score_threshold)
    if name == "pred_mu_alpha":
        return en.pred_mu_alpha
    if name == "pred_mu_abs":
        if en.pred_mu_alpha is None:
            return None
        return abs(float(en.pred_mu_alpha))
    if name == "pred_mu_dir_conf":
        return en.pred_mu_dir_conf
    if name == "alpha_vpin":
        return en.alpha_vpin
    if name == "alpha_hurst":
        return en.alpha_hurst
    if name == "entry_quality_score":
        return en.entry_quality_score
    if name == "one_way_move_score":
        return en.one_way_move_score
    if name == "leverage_signal_score":
        return en.leverage_signal_score
    if name == "event_exit_score":
        return _safe_float(en.raw.get("event_exit_score"))
    if name == "event_p_tp":
        return _safe_float(en.raw.get("event_p_tp"))
    if name == "event_p_sl":
        return _safe_float(en.raw.get("event_p_sl"))
    if name == "event_cvar_pct":
        return _safe_float(en.raw.get("event_cvar_pct"))
    return None


def _feature_report(
    rows: list[tuple[EntryRow, ExitRow, str]],
    feature: str,
    *,
    high_is_good: bool,
    target_mode: str,
) -> dict[str, Any]:
    vals = []
    for en, ex, _m in rows:
        fv = _pick_feature(en, feature)
        if fv is None or not math.isfinite(float(fv)):
            continue
        vals.append(float(fv))
    if len(vals) < 40:
        return {
            "feature": feature,
            "n": int(len(vals)),
            "insufficient": True,
        }

    q20 = _pct(vals, 0.2)
    q40 = _pct(vals, 0.4)
    q60 = _pct(vals, 0.6)
    q80 = _pct(vals, 0.8)
    edges = [q20, q40, q60, q80]
    labels = ["q1", "q2", "q3", "q4", "q5"]
    bins = {k: {"n": 0, "win_n": 0, "sum_roe": 0.0, "roe_n": 0, "miss_n": 0, "miss_eval_n": 0} for k in labels}

    top = {"n": 0, "loss_n": 0, "sum_roe": 0.0, "roe_n": 0, "miss_n": 0, "miss_eval_n": 0}
    bot = {"n": 0, "loss_n": 0, "sum_roe": 0.0, "roe_n": 0, "miss_n": 0, "miss_eval_n": 0}

    for en, ex, _m in rows:
        fv = _pick_feature(en, feature)
        if fv is None or not math.isfinite(float(fv)):
            continue
        x = float(fv)
        bi = _bin_idx(x, edges)
        key = labels[bi]
        b = bins[key]
        b["n"] += 1
        rr = _target_value(ex, target_mode)
        if rr is not None:
            b["sum_roe"] += float(rr)
            b["roe_n"] += 1
            if rr > 0:
                b["win_n"] += 1
        pred_s = _sign(en.pred_mu_alpha)
        real_s = _sign(rr)
        if pred_s != 0 and real_s != 0:
            b["miss_eval_n"] += 1
            if pred_s != real_s:
                b["miss_n"] += 1

        tgt = top if bi == 4 else bot if bi == 0 else None
        if tgt is not None:
            tgt["n"] += 1
            if rr is not None:
                tgt["sum_roe"] += float(rr)
                tgt["roe_n"] += 1
                if rr < 0:
                    tgt["loss_n"] += 1
            if pred_s != 0 and real_s != 0:
                tgt["miss_eval_n"] += 1
                if pred_s != real_s:
                    tgt["miss_n"] += 1

    out_bins = []
    for k in labels:
        b = bins[k]
        out_bins.append(
            {
                "bin": k,
                "n": int(b["n"]),
                "win_rate": _safe_rate(int(b["win_n"]), int(b["n"])),
                "avg_roe": (float(b["sum_roe"] / b["roe_n"]) if int(b["roe_n"]) > 0 else None),
                "direction_miss_rate": _safe_rate(int(b["miss_n"]), int(b["miss_eval_n"])),
            }
        )

    top_loss = _safe_rate(int(top["loss_n"]), int(top["n"]))
    bot_loss = _safe_rate(int(bot["loss_n"]), int(bot["n"]))
    top_miss = _safe_rate(int(top["miss_n"]), int(top["miss_eval_n"]))
    bot_miss = _safe_rate(int(bot["miss_n"]), int(bot["miss_eval_n"]))
    top_avg_roe = float(top["sum_roe"] / top["roe_n"]) if int(top["roe_n"]) > 0 else None
    bot_avg_roe = float(bot["sum_roe"] / bot["roe_n"]) if int(bot["roe_n"]) > 0 else None

    if high_is_good:
        paradox_loss_lift = (top_loss - bot_loss) if (top_loss is not None and bot_loss is not None) else None
        paradox_miss_lift = (top_miss - bot_miss) if (top_miss is not None and bot_miss is not None) else None
        paradox_roe_delta = (top_avg_roe - bot_avg_roe) if (top_avg_roe is not None and bot_avg_roe is not None) else None
    else:
        paradox_loss_lift = (bot_loss - top_loss) if (top_loss is not None and bot_loss is not None) else None
        paradox_miss_lift = (bot_miss - top_miss) if (top_miss is not None and bot_miss is not None) else None
        paradox_roe_delta = (bot_avg_roe - top_avg_roe) if (top_avg_roe is not None and bot_avg_roe is not None) else None

    return {
        "feature": feature,
        "n": int(len(vals)),
        "quantile_edges": {
            "q20": q20,
            "q40": q40,
            "q60": q60,
            "q80": q80,
        },
        "bins": out_bins,
        "top_vs_bottom": {
            "high_bin": {
                "n": int(top["n"]),
                "loss_rate": top_loss,
                "avg_roe": top_avg_roe,
                "direction_miss_rate": top_miss,
            },
            "low_bin": {
                "n": int(bot["n"]),
                "loss_rate": bot_loss,
                "avg_roe": bot_avg_roe,
                "direction_miss_rate": bot_miss,
            },
            "paradox_loss_lift": paradox_loss_lift,
            "paradox_direction_miss_lift": paradox_miss_lift,
            "paradox_avg_roe_delta": paradox_roe_delta,
            "high_is_good": bool(high_is_good),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze entry score drivers vs realized outcome/direction mismatch.")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--since-id", type=int, default=0)
    ap.add_argument("--recent-exits", type=int, default=1500)
    ap.add_argument("--fallback-lookback-sec", type=int, default=6 * 3600)
    ap.add_argument("--target-mode", default="pre_roe", choices=["pre_roe", "roe"])
    ap.add_argument("--out", default="state/entry_score_driver_report.json")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    entries, exits = _load_rows(db_path, int(args.since_id), int(args.recent_exits))
    matched = _match(entries, exits, int(args.fallback_lookback_sec))
    if not matched:
        out = {
            "timestamp_ms": int(time.time() * 1000),
            "db": str(db_path),
            "message": "no matched rows",
        }
        Path(args.out).write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=True, indent=2))
        return

    total_n = len(matched)
    win_n = 0
    sum_roe = 0.0
    roe_n = 0
    miss_n = 0
    miss_eval_n = 0
    for en, ex, _m in matched:
        rr = _target_value(ex, args.target_mode)
        if rr is not None:
            sum_roe += float(rr)
            roe_n += 1
            if rr > 0:
                win_n += 1
        pred_s = _sign(en.pred_mu_alpha)
        real_s = _sign(rr)
        if pred_s != 0 and real_s != 0:
            miss_eval_n += 1
            if pred_s != real_s:
                miss_n += 1

    feature_specs = [
        ("entry_ev", True),
        ("entry_ev_excess", True),
        ("pred_mu_abs", True),
        ("pred_mu_dir_conf", True),
        ("entry_quality_score", True),
        ("one_way_move_score", True),
        ("leverage_signal_score", True),
        ("event_exit_score", True),
        ("event_p_tp", True),
        ("event_p_sl", False),
        ("event_cvar_pct", False),
        ("alpha_vpin", False),
        ("alpha_hurst", True),
    ]
    reports = []
    for name, high_good in feature_specs:
        reports.append(_feature_report(matched, name, high_is_good=high_good, target_mode=args.target_mode))

    # Rank likely "bad drivers":
    # high values expected to help entry, but high-bin outcomes are worse.
    bad_drivers = []
    for r in reports:
        t = (r or {}).get("top_vs_bottom")
        if not isinstance(t, dict):
            continue
        if not bool(t.get("high_is_good")):
            continue
        n_hi = int(((t.get("high_bin") or {}).get("n") or 0))
        if n_hi < 40:
            continue
        loss_lift = _safe_float(t.get("paradox_loss_lift"), None)
        miss_lift = _safe_float(t.get("paradox_direction_miss_lift"), None)
        roe_delta = _safe_float(t.get("paradox_avg_roe_delta"), None)
        if loss_lift is None or miss_lift is None or roe_delta is None:
            continue
        score = float(loss_lift * 1.5 + miss_lift * 1.0 + max(0.0, -roe_delta) * 10.0)
        bad_drivers.append(
            {
                "feature": r.get("feature"),
                "high_bin_n": n_hi,
                "loss_rate_lift_high_vs_low": loss_lift,
                "direction_miss_lift_high_vs_low": miss_lift,
                "avg_roe_delta_high_minus_low": roe_delta,
                "bad_driver_score": score,
            }
        )
    bad_drivers.sort(key=lambda x: float(x.get("bad_driver_score") or 0.0), reverse=True)

    out = {
        "timestamp_ms": int(time.time() * 1000),
        "db": str(db_path),
        "config": {
            "since_id": int(args.since_id),
            "recent_exits": int(args.recent_exits),
            "fallback_lookback_sec": int(args.fallback_lookback_sec),
            "target_mode": str(args.target_mode),
        },
        "coverage": {
            "entries_total": int(len(entries)),
            "exits_total_recent": int(len(exits)),
            "matched_pairs": int(total_n),
            "match_mode_entry_link_id": int(sum(1 for _en, _ex, m in matched if m == "entry_link_id")),
            "match_mode_fallback_symbol_side": int(sum(1 for _en, _ex, m in matched if m != "entry_link_id")),
        },
        "aggregate": {
            "win_rate": _safe_rate(int(win_n), int(total_n)),
            "avg_target": (float(sum_roe / roe_n) if roe_n > 0 else None),
            "direction_miss_rate": _safe_rate(int(miss_n), int(miss_eval_n)),
        },
        "entry_score_structure_hint": {
            "entry_score_primary": "decision.ev (entry_ev)",
            "entry_gate_context": [
                "policy_score_threshold/unified_floor",
                "pred_mu_alpha + pred_mu_dir_conf(dir_gate)",
                "event_exit_score/event_p_tp/event_p_sl(event precheck)",
                "alpha_vpin/alpha_hurst",
                "entry_quality_score/one_way_move_score",
            ],
            "note": "This report evaluates which high-score components are paradoxically associated with poor realized outcomes.",
        },
        "feature_reports": reports,
        "bad_driver_candidates": bad_drivers[:10],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

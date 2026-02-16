#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _sign(x: float | None, eps: float = 1e-12) -> int:
    if x is None:
        return 0
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values[int(k)])
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return float(d0 + d1)


def _fetch_exits(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    q = """
    SELECT
      id, timestamp_ms, symbol, side, action,
      fill_price, qty, notional,
      fee, fee_rate,
      realized_pnl, roe, hold_duration_sec,
      entry_reason, entry_ev, entry_confidence,
      regime, alpha_vpin,
      pred_mu_alpha, pred_mu_dir_conf,
      entry_quality_score, one_way_move_score, leverage_signal_score,
      raw_data
    FROM trades
    WHERE action IN ('EXIT', 'REBAL_EXIT', 'KILL', 'MANUAL', 'EXTERNAL')
    ORDER BY timestamp_ms DESC
    LIMIT ?
    """
    cur = conn.cursor()
    cur.execute(q, (int(limit),))
    return cur.fetchall()


def _extract_reason(row: sqlite3.Row) -> str:
    reason = None
    raw = row["raw_data"]
    if raw:
        try:
            j = json.loads(raw)
            reason = j.get("reason") or j.get("exit_reason")
        except Exception:
            reason = None
    if not reason:
        reason = row["entry_reason"]
    return str(reason or "unknown")


def _extract_lev_bucket(row: sqlite3.Row) -> str:
    raw = row["raw_data"]
    lev = None
    if raw:
        try:
            j = json.loads(raw)
            lev = _safe_float(j.get("leverage"), None)
        except Exception:
            lev = None
    if lev is None:
        return "unknown"
    if lev < 2:
        return "<2x"
    if lev < 5:
        return "2-5x"
    if lev < 10:
        return "5-10x"
    return ">=10x"


def analyze(rows: list[sqlite3.Row], worst_n: int) -> dict[str, Any]:
    exits = []
    for r in rows:
        pnl = _safe_float(r["realized_pnl"], None)
        if pnl is None:
            continue
        exits.append(r)

    exits_sorted = sorted(exits, key=lambda x: _safe_float(x["realized_pnl"], 0.0))
    worst = exits_sorted[: max(1, min(int(worst_n), len(exits_sorted)))]

    all_pnls = [_safe_float(r["realized_pnl"], 0.0) for r in exits_sorted]
    loss_pnls = [p for p in all_pnls if p < 0]

    reason_all = Counter(_extract_reason(r) for r in exits_sorted)
    reason_worst = Counter(_extract_reason(r) for r in worst)
    regime_all = Counter(str(r["regime"] or "unknown") for r in exits_sorted)
    regime_worst = Counter(str(r["regime"] or "unknown") for r in worst)
    lev_worst = Counter(_extract_lev_bucket(r) for r in worst)

    hold_worst = [_safe_float(r["hold_duration_sec"], None) for r in worst]
    hold_worst = [v for v in hold_worst if v is not None]

    vpin_worst = [_safe_float(r["alpha_vpin"], None) for r in worst]
    vpin_worst = [v for v in vpin_worst if v is not None]

    conf_worst = [_safe_float(r["pred_mu_dir_conf"], None) for r in worst]
    conf_worst = [v for v in conf_worst if v is not None]

    mismatch_worst = 0
    mismatch_all = 0
    total_mu_worst = 0
    total_mu_all = 0

    for r in exits_sorted:
        mu = _safe_float(r["pred_mu_alpha"], None)
        roe = _safe_float(r["roe"], None)
        if mu is None or roe is None:
            continue
        total_mu_all += 1
        if _sign(mu) != 0 and _sign(roe) != 0 and _sign(mu) != _sign(roe):
            mismatch_all += 1

    for r in worst:
        mu = _safe_float(r["pred_mu_alpha"], None)
        roe = _safe_float(r["roe"], None)
        if mu is None or roe is None:
            continue
        total_mu_worst += 1
        if _sign(mu) != 0 and _sign(roe) != 0 and _sign(mu) != _sign(roe):
            mismatch_worst += 1

    worst_rows = []
    for r in worst:
        worst_rows.append(
            {
                "id": _safe_int(r["id"]),
                "timestamp_ms": _safe_int(r["timestamp_ms"]),
                "symbol": r["symbol"],
                "side": r["side"],
                "reason": _extract_reason(r),
                "regime": r["regime"],
                "realized_pnl": _safe_float(r["realized_pnl"], 0.0),
                "roe": _safe_float(r["roe"], None),
                "hold_duration_sec": _safe_float(r["hold_duration_sec"], None),
                "entry_ev": _safe_float(r["entry_ev"], None),
                "pred_mu_alpha": _safe_float(r["pred_mu_alpha"], None),
                "pred_mu_dir_conf": _safe_float(r["pred_mu_dir_conf"], None),
                "alpha_vpin": _safe_float(r["alpha_vpin"], None),
                "entry_quality_score": _safe_float(r["entry_quality_score"], None),
                "one_way_move_score": _safe_float(r["one_way_move_score"], None),
                "leverage_signal_score": _safe_float(r["leverage_signal_score"], None),
                "lev_bucket": _extract_lev_bucket(r),
            }
        )

    report = {
        "timestamp_ms": int(time.time() * 1000),
        "sample": {
            "exits_total": len(exits_sorted),
            "worst_n": len(worst),
        },
        "pnl_summary": {
            "sum": float(sum(all_pnls)) if all_pnls else None,
            "avg": float(statistics.mean(all_pnls)) if all_pnls else None,
            "loss_count": len(loss_pnls),
            "p10": _pct(all_pnls, 0.10),
            "p25": _pct(all_pnls, 0.25),
            "p50": _pct(all_pnls, 0.50),
        },
        "common_patterns": {
            "reason_top_all": reason_all.most_common(8),
            "reason_top_worst": reason_worst.most_common(8),
            "regime_top_all": regime_all.most_common(6),
            "regime_top_worst": regime_worst.most_common(6),
            "leverage_bucket_worst": lev_worst.most_common(6),
            "hold_sec_worst": {
                "avg": float(statistics.mean(hold_worst)) if hold_worst else None,
                "p50": _pct(hold_worst, 0.50),
                "p90": _pct(hold_worst, 0.90),
            },
            "vpin_worst": {
                "avg": float(statistics.mean(vpin_worst)) if vpin_worst else None,
                "p50": _pct(vpin_worst, 0.50),
                "p90": _pct(vpin_worst, 0.90),
            },
            "dir_conf_worst": {
                "avg": float(statistics.mean(conf_worst)) if conf_worst else None,
                "p50": _pct(conf_worst, 0.50),
                "p90": _pct(conf_worst, 0.90),
            },
            "mu_sign_mismatch_rate": {
                "all": float(mismatch_all / total_mu_all) if total_mu_all else None,
                "worst": float(mismatch_worst / total_mu_worst) if total_mu_worst else None,
                "all_n": int(total_mu_all),
                "worst_n": int(total_mu_worst),
            },
        },
        "worst_trades": worst_rows,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze largest loss trades and extract common patterns.")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--worst-n", type=int, default=80)
    ap.add_argument("--out", default="state/large_loss_common_patterns.json")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_exits(conn, limit=int(args.limit))
    finally:
        conn.close()

    report = analyze(rows, worst_n=int(args.worst_n))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

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


EXIT_SQL = "('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL')"


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


def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    arr = sorted(vals)
    pos = (len(arr) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    return float(arr[lo] * (hi - pos) + arr[hi] * (pos - lo))


def _sign(x: float | None, eps: float = 1e-12) -> int:
    if x is None:
        return 0
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _fetch_rows(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
          id, timestamp_ms, symbol, side, action,
          realized_pnl, roe, hold_duration_sec,
          entry_reason, regime,
          pred_mu_alpha, pred_mu_dir_conf,
          alpha_vpin,
          entry_quality_score, one_way_move_score, leverage_signal_score,
          raw_data
        FROM trades
        WHERE action IN {EXIT_SQL}
        ORDER BY timestamp_ms DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    return cur.fetchall()


def _reason_from_row(r: sqlite3.Row) -> str:
    reason = str(r["entry_reason"] or "").strip()
    raw = r["raw_data"]
    if raw:
        try:
            j = json.loads(raw)
            reason = str(j.get("reason") or j.get("exit_reason") or reason).strip()
        except Exception:
            pass
    return reason or "unknown"


def _leverage_from_row(r: sqlite3.Row) -> float | None:
    raw = r["raw_data"]
    if not raw:
        return None
    try:
        j = json.loads(raw)
        return _safe_float(j.get("leverage"), None)
    except Exception:
        return None


def _bucket_hold(sec: float | None) -> str:
    if sec is None:
        return "unknown"
    if sec <= 30:
        return "<=30s"
    if sec <= 120:
        return "30-120s"
    if sec <= 600:
        return "120-600s"
    return ">600s"


def _bucket_conf(conf: float | None) -> str:
    if conf is None:
        return "unknown"
    if conf < 0.55:
        return "<0.55"
    if conf < 0.65:
        return "0.55-0.65"
    return ">=0.65"


def _bucket_mu_abs(mu: float | None) -> str:
    if mu is None:
        return "unknown"
    a = abs(mu)
    if a < 0.25:
        return "<0.25"
    if a < 1.0:
        return "0.25-1"
    if a < 5.0:
        return "1-5"
    if a < 15.0:
        return "5-15"
    return ">=15"


def analyze(rows: list[sqlite3.Row], worst_n: int) -> dict[str, Any]:
    rows = [r for r in rows if _safe_float(r["realized_pnl"], None) is not None]
    rows_sorted = sorted(rows, key=lambda x: _safe_float(x["realized_pnl"], 0.0))
    worst = rows_sorted[: max(1, min(int(worst_n), len(rows_sorted)))]

    reason_ctr = Counter()
    regime_ctr = Counter()
    sym_ctr = Counter()
    sym_pnl = defaultdict(float)
    reason_regime = Counter()
    hold_ctr = Counter()
    conf_ctr = Counter()
    mu_ctr = Counter()
    external_like = 0
    mismatch = 0
    mismatch_n = 0
    high_vpin = 0
    high_vpin_n = 0

    roes = []
    holds = []
    vpins = []
    confs = []
    levs = []

    detail_rows = []
    for r in worst:
        pnl = _safe_float(r["realized_pnl"], 0.0) or 0.0
        roe = _safe_float(r["roe"], None)
        hold = _safe_float(r["hold_duration_sec"], None)
        mu = _safe_float(r["pred_mu_alpha"], None)
        conf = _safe_float(r["pred_mu_dir_conf"], None)
        vpin = _safe_float(r["alpha_vpin"], None)
        lev = _leverage_from_row(r)
        reason = _reason_from_row(r)
        regime = str(r["regime"] or "unknown")
        sym = str(r["symbol"] or "unknown")

        reason_ctr[reason] += 1
        regime_ctr[regime] += 1
        sym_ctr[sym] += 1
        sym_pnl[sym] += float(pnl)
        reason_regime[(reason, regime)] += 1
        hold_ctr[_bucket_hold(hold)] += 1
        conf_ctr[_bucket_conf(conf)] += 1
        mu_ctr[_bucket_mu_abs(mu)] += 1

        rl = reason.lower()
        if ("exchange_close" in rl) or ("external" in rl) or ("manual" in rl):
            external_like += 1

        if mu is not None and roe is not None:
            mismatch_n += 1
            if _sign(mu) != 0 and _sign(roe) != 0 and _sign(mu) != _sign(roe):
                mismatch += 1

        if vpin is not None:
            high_vpin_n += 1
            if vpin >= 0.8:
                high_vpin += 1

        if roe is not None:
            roes.append(float(roe))
        if hold is not None:
            holds.append(float(hold))
        if vpin is not None:
            vpins.append(float(vpin))
        if conf is not None:
            confs.append(float(conf))
        if lev is not None:
            levs.append(float(lev))

        detail_rows.append(
            {
                "id": _safe_int(r["id"]),
                "timestamp_ms": _safe_int(r["timestamp_ms"]),
                "symbol": sym,
                "side": r["side"],
                "reason": reason,
                "regime": regime,
                "realized_pnl": pnl,
                "roe": roe,
                "hold_duration_sec": hold,
                "pred_mu_alpha": mu,
                "pred_mu_dir_conf": conf,
                "alpha_vpin": vpin,
                "leverage": lev,
                "entry_quality_score": _safe_float(r["entry_quality_score"], None),
                "one_way_move_score": _safe_float(r["one_way_move_score"], None),
                "leverage_signal_score": _safe_float(r["leverage_signal_score"], None),
            }
        )

    top_reason_regime = [
        {"reason": k[0], "regime": k[1], "n": int(v)}
        for k, v in reason_regime.most_common(15)
    ]
    top_symbol_loss = [
        {"symbol": s, "n": int(sym_ctr[s]), "sum_pnl": float(sym_pnl[s]), "avg_pnl": float(sym_pnl[s] / max(1, sym_ctr[s]))}
        for s in sorted(sym_pnl.keys(), key=lambda x: sym_pnl[x])[:15]
    ]

    return {
        "timestamp_ms": int(time.time() * 1000),
        "sample": {
            "exits_total": int(len(rows_sorted)),
            "worst_n": int(len(worst)),
        },
        "aggregates": {
            "roe_avg": float(statistics.mean(roes)) if roes else None,
            "roe_p10": _pct(roes, 0.10),
            "roe_p50": _pct(roes, 0.50),
            "hold_sec_avg": float(statistics.mean(holds)) if holds else None,
            "hold_sec_p50": _pct(holds, 0.50),
            "hold_sec_p90": _pct(holds, 0.90),
            "vpin_avg": float(statistics.mean(vpins)) if vpins else None,
            "vpin_p90": _pct(vpins, 0.90),
            "dir_conf_avg": float(statistics.mean(confs)) if confs else None,
            "dir_conf_p50": _pct(confs, 0.50),
            "leverage_avg": float(statistics.mean(levs)) if levs else None,
            "leverage_p50": _pct(levs, 0.50),
        },
        "common_patterns": {
            "top_reasons": reason_ctr.most_common(12),
            "top_regimes": regime_ctr.most_common(8),
            "top_symbols_by_count": sym_ctr.most_common(12),
            "top_symbols_by_loss_sum": top_symbol_loss,
            "reason_regime_matrix_top": top_reason_regime,
            "hold_bucket": hold_ctr,
            "dir_conf_bucket": conf_ctr,
            "mu_abs_bucket": mu_ctr,
            "mu_sign_mismatch_rate": (float(mismatch / mismatch_n) if mismatch_n else None),
            "external_or_sync_reason_share": float(external_like / max(1, len(worst))),
            "high_vpin_share_ge_0_8": float(high_vpin / max(1, high_vpin_n)),
        },
        "worst_trades": detail_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep-dive analysis on largest loss exits")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--worst-n", type=int, default=120)
    ap.add_argument("--out", default="state/large_loss_deepdive.json")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_rows(conn, int(args.limit))
    finally:
        conn.close()

    report = analyze(rows, int(args.worst_n))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
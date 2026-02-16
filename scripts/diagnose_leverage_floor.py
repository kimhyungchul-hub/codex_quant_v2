#!/usr/bin/env python3
"""Diagnose why entry leverage is sticking to 1x from recent DB trades.

Outputs a JSON report with cause buckets and per-symbol breakdown.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _pick(raw: dict[str, Any], row: sqlite3.Row, key: str, default: Any = None) -> Any:
    if key in row.keys() and row[key] is not None:
        return row[key]
    return raw.get(key, default)


def _classify(entry: dict[str, Any]) -> str:
    lev = _safe_float(entry.get("leverage"), 1.0)
    if lev > 1.000001:
        return "lev_gt_1"

    eq = _safe_float(entry.get("entry_quality_score"), 0.0)
    ls = _safe_float(entry.get("leverage_signal_score"), 0.0)
    vpin = _safe_float(entry.get("alpha_vpin"), 0.0)
    reject_cnt = _safe_int(entry.get("lev_balance_reject_count"), 0)
    lev_source = str(entry.get("lev_source") or "")
    lev_raw = _safe_float(entry.get("lev_raw_before_caps"), 0.0)
    lev_target_max = _safe_float(entry.get("lev_target_max"), 0.0)
    lev_liq_cap = _safe_float(entry.get("lev_liq_cap"), 0.0)
    sigma_stress = _safe_float(entry.get("lev_dynamic_sigma_stress"), 0.0)

    if eq <= 0.0 or ls <= 0.0:
        return "score_missing_or_zero"
    if reject_cnt >= 2:
        return "recent_110007_balance_reject"
    if vpin >= 0.95:
        return "vpin_extreme_delev"
    if sigma_stress >= 0.95:
        return "sigma_stress_delev"
    if lev_liq_cap > 0.0 and lev_liq_cap <= 1.05:
        return "liquidation_distance_cap"
    if lev_target_max > 0.0 and lev_target_max <= 1.05:
        return "target_cap_floor"
    if lev_source.startswith("dynamic_risk") and 0.0 < lev_raw <= 1.05:
        return "dynamic_raw_floor"
    return "other_floor_or_risk_cap"


def _load_recent_entries(db_path: Path, lookback_entries: int, min_id: int = 0) -> list[dict[str, Any]]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT id, symbol, side, action, timestamp_ms,
                   entry_quality_score, leverage_signal_score, alpha_vpin, raw_data
            FROM trades
            WHERE action='ENTER' AND id >= ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(max(0, min_id)), int(max(1, lookback_entries))),
        ).fetchall()
    finally:
        con.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        raw = {}
        raw_text = row["raw_data"]
        if raw_text:
            try:
                raw = json.loads(raw_text)
            except Exception:
                raw = {}
        item = {
            "id": _safe_int(row["id"]),
            "symbol": row["symbol"],
            "side": row["side"],
            "timestamp_ms": _safe_int(row["timestamp_ms"]),
            "leverage": _pick(raw, row, "leverage", 1.0),
            "entry_quality_score": _pick(raw, row, "entry_quality_score", 0.0),
            "leverage_signal_score": _pick(raw, row, "leverage_signal_score", 0.0),
            "alpha_vpin": _pick(raw, row, "alpha_vpin", 0.0),
            "lev_source": _pick(raw, row, "lev_source"),
            "lev_raw_before_caps": _pick(raw, row, "lev_raw_before_caps"),
            "lev_target_max": _pick(raw, row, "lev_target_max"),
            "lev_liq_cap": _pick(raw, row, "lev_liq_cap"),
            "lev_dynamic_sigma_stress": _pick(raw, row, "lev_dynamic_sigma_stress"),
            "lev_balance_reject_count": _pick(raw, row, "lev_balance_reject_count"),
        }
        item["cause"] = _classify(item)
        out.append(item)
    return out


def _build_streaks(entries_desc: list[dict[str, Any]], min_len: int = 3) -> list[dict[str, Any]]:
    # Work in chronological order per symbol.
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in reversed(entries_desc):
        by_symbol[str(e.get("symbol") or "")].append(e)

    streaks: list[dict[str, Any]] = []
    for sym, arr in by_symbol.items():
        cur: list[dict[str, Any]] = []
        for e in arr:
            lev = _safe_float(e.get("leverage"), 1.0)
            if lev <= 1.000001:
                cur.append(e)
            else:
                if len(cur) >= min_len:
                    streaks.append(_summarize_streak(sym, cur))
                cur = []
        if len(cur) >= min_len:
            streaks.append(_summarize_streak(sym, cur))

    streaks.sort(key=lambda x: int(x.get("length", 0)), reverse=True)
    return streaks[:20]


def _summarize_streak(sym: str, streak: list[dict[str, Any]]) -> dict[str, Any]:
    causes = Counter(str(x.get("cause") or "") for x in streak)
    eq_avg = sum(_safe_float(x.get("entry_quality_score"), 0.0) for x in streak) / max(1, len(streak))
    ls_avg = sum(_safe_float(x.get("leverage_signal_score"), 0.0) for x in streak) / max(1, len(streak))
    vpin_avg = sum(_safe_float(x.get("alpha_vpin"), 0.0) for x in streak) / max(1, len(streak))
    return {
        "symbol": sym,
        "length": len(streak),
        "start_id": _safe_int(streak[0].get("id")),
        "end_id": _safe_int(streak[-1].get("id")),
        "avg_entry_quality_score": eq_avg,
        "avg_leverage_signal_score": ls_avg,
        "avg_alpha_vpin": vpin_avg,
        "cause_top": causes.most_common(3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose 1x leverage floor causes from recent entries")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--lookback-entries", type=int, default=1500)
    ap.add_argument("--min-id", type=int, default=0)
    ap.add_argument("--out", default="state/leverage_floor_diagnosis.json")
    args = ap.parse_args()

    db_path = Path(args.db)
    entries = _load_recent_entries(db_path, args.lookback_entries, min_id=int(args.min_id))

    total = len(entries)
    lev1_entries = [e for e in entries if _safe_float(e.get("leverage"), 1.0) <= 1.000001]
    levgt1_entries = [e for e in entries if _safe_float(e.get("leverage"), 1.0) > 1.000001]

    cause_counts = Counter(str(e.get("cause") or "") for e in lev1_entries)
    by_symbol_lev1 = Counter(str(e.get("symbol") or "") for e in lev1_entries)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db": str(db_path),
        "lookback_entries": int(args.lookback_entries),
        "min_id": int(args.min_id),
        "entries_total": total,
        "lev1_entries": len(lev1_entries),
        "lev_gt1_entries": len(levgt1_entries),
        "lev1_share": (len(lev1_entries) / total) if total > 0 else 0.0,
        "cause_counts": dict(cause_counts),
        "cause_shares": {k: (v / max(1, len(lev1_entries))) for k, v in cause_counts.items()},
        "top_symbols_lev1": by_symbol_lev1.most_common(20),
        "lev1_streaks": _build_streaks(entries_desc=entries, min_len=3),
        "sample_latest": entries[:20],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "entries_total": report["entries_total"],
        "lev1_entries": report["lev1_entries"],
        "lev_gt1_entries": report["lev_gt1_entries"],
        "lev1_share": report["lev1_share"],
        "top_causes": cause_counts.most_common(5),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

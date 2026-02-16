#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path


def _rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Check observability coverage for trade logs.")
    ap.add_argument("--db", default="state/bot_data_live.db", help="SQLite DB path")
    ap.add_argument("--recent", type=int, default=2000, help="Recent rows to inspect")
    ap.add_argument("--since-id", type=int, default=0, help="Only include rows with id > since_id")
    ap.add_argument("--since-ts-ms", type=int, default=0, help="Only include rows with timestamp_ms >= since_ts_ms")
    ap.add_argument("--out", default="state/trade_observability_report.json", help="Output JSON path")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where = []
    params = []
    if int(args.since_id) > 0:
        where.append("id > ?")
        params.append(int(args.since_id))
    if int(args.since_ts_ms) > 0:
        where.append("timestamp_ms >= ?")
        params.append(int(args.since_ts_ms))
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT * FROM trades {where_sql} ORDER BY id DESC LIMIT ?"
    params.append(int(args.recent))
    rows = cur.execute(sql, tuple(params)).fetchall()
    entry_links_all_rows = cur.execute(
        """
        SELECT DISTINCT COALESCE(entry_link_id, entry_id, trade_uid) AS lk
        FROM trades
        WHERE action IN ('ENTER','SPREAD')
          AND COALESCE(entry_link_id, entry_id, trade_uid) IS NOT NULL
        """
    ).fetchall()
    conn.close()

    rows = list(rows)
    if not rows:
        out = {
            "timestamp_ms": int(time.time() * 1000),
            "db": str(db_path),
            "recent": int(args.recent),
            "message": "no rows",
        }
        Path(args.out).write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=True, indent=2))
        return

    exit_actions = {"EXIT", "REBAL_EXIT", "KILL", "MANUAL", "EXTERNAL"}
    entry_actions = {"ENTER", "SPREAD"}
    reject_actions = {"ORDER_REJECT"}

    entries = [r for r in rows if str(r["action"]).upper() in entry_actions]
    exits = [r for r in rows if str(r["action"]).upper() in exit_actions]
    active_rows = [r for r in rows if str(r["action"]).upper() not in reject_actions]

    def _text(v) -> str:
        try:
            return str(v or "").strip().lower()
        except Exception:
            return ""

    def _is_managed_exit(row) -> bool:
        act = _text(row["action"]).upper()
        reason = _text(row["entry_reason"]) if "entry_reason" in row.keys() else ""
        if act in ("MANUAL", "EXTERNAL"):
            return False
        # Exchange-side sync/manual cleanup/liquidation records are excluded from managed quality metrics.
        if reason.startswith("exchange_") or "exchange_" in reason:
            return False
        if "manual_close" in reason or "external_sync" in reason:
            return False
        return True

    managed_exits = [r for r in exits if _is_managed_exit(r)]

    def _entry_key(row) -> str:
        keys = row.keys() if hasattr(row, "keys") else []
        v_link = row["entry_link_id"] if "entry_link_id" in keys else None
        v_id = row["entry_id"] if "entry_id" in keys else None
        v = v_link if v_link is not None else v_id
        return str(v or "").strip()

    entry_links = {_entry_key(r) for r in entries if _entry_key(r)}
    entry_links_all = {str(r["lk"]).strip() for r in entry_links_all_rows if r["lk"] is not None and str(r["lk"]).strip()}

    exit_with_link = 0
    exit_link_match = 0
    for r in exits:
        lk = _entry_key(r)
        if not lk:
            continue
        exit_with_link += 1
        if lk in entry_links:
            exit_link_match += 1

    exit_with_link_all = 0
    exit_link_match_all = 0
    for r in exits:
        lk = _entry_key(r)
        if not lk:
            continue
        exit_with_link_all += 1
        if lk in entry_links_all:
            exit_link_match_all += 1

    managed_with_link = 0
    managed_match_all = 0
    for r in managed_exits:
        lk = _entry_key(r)
        if not lk:
            continue
        managed_with_link += 1
        if lk in entry_links_all:
            managed_match_all += 1

    def _nonnull_count(rs, col: str) -> int:
        if rs and hasattr(rs[0], "keys") and col not in rs[0].keys():
            return 0
        c = 0
        for r in rs:
            v = r[col]
            if v is None:
                continue
            if isinstance(v, str) and (not v.strip()):
                continue
            c += 1
        return c

    required_cols = [
        "entry_id",
        "entry_link_id",
        "trade_uid",
        "regime",
        "alpha_vpin",
        "alpha_hurst",
        "pred_mu_alpha",
        "pred_mu_dir_conf",
        "policy_score_threshold",
        "policy_event_exit_min_score",
        "policy_unrealized_dd_floor",
    ]

    coverage_exit = {}
    for col in required_cols:
        n = _nonnull_count(exits, col)
        coverage_exit[col] = {
            "nonnull": int(n),
            "total": int(len(exits)),
            "rate": _rate(n, len(exits)),
            "null_rate": 1.0 - _rate(n, len(exits)),
        }

    coverage_exit_managed = {}
    for col in required_cols:
        n = _nonnull_count(managed_exits, col)
        coverage_exit_managed[col] = {
            "nonnull": int(n),
            "total": int(len(managed_exits)),
            "rate": _rate(n, len(managed_exits)),
            "null_rate": 1.0 - _rate(n, len(managed_exits)),
        }

    coverage_active = {}
    for col in ("entry_id", "entry_link_id", "trade_uid"):
        n = _nonnull_count(active_rows, col)
        coverage_active[col] = {
            "nonnull": int(n),
            "total": int(len(active_rows)),
            "rate": _rate(n, len(active_rows)),
            "null_rate": 1.0 - _rate(n, len(active_rows)),
        }

    out = {
        "timestamp_ms": int(time.time() * 1000),
        "db": str(db_path),
        "recent": int(args.recent),
        "since_id": int(args.since_id),
        "since_ts_ms": int(args.since_ts_ms),
        "counts": {
            "rows": int(len(rows)),
            "entries": int(len(entries)),
            "exits": int(len(exits)),
            "managed_exits": int(len(managed_exits)),
            "active_non_reject": int(len(active_rows)),
        },
        "linkage": {
            "entry_links_unique": int(len(entry_links)),
            "exit_with_entry_link": int(exit_with_link),
            "exit_entry_link_match": int(exit_link_match),
            "exit_entry_link_match_rate": _rate(exit_link_match, exit_with_link),
            "exit_entry_link_missing_rate": 1.0 - _rate(exit_with_link, len(exits)),
        },
        "linkage_all_entries": {
            "entry_links_unique_all": int(len(entry_links_all)),
            "exit_with_entry_link": int(exit_with_link_all),
            "exit_entry_link_match": int(exit_link_match_all),
            "exit_entry_link_match_rate": _rate(exit_link_match_all, exit_with_link_all),
            "exit_entry_link_missing_rate": 1.0 - _rate(exit_with_link_all, len(exits)),
        },
        "linkage_managed": {
            "managed_exit_with_entry_link": int(managed_with_link),
            "managed_exit_entry_link_match": int(managed_match_all),
            "managed_exit_entry_link_match_rate": _rate(managed_match_all, managed_with_link),
            "managed_exit_entry_link_missing_rate": 1.0 - _rate(managed_with_link, len(managed_exits)),
        },
        "coverage_exit": coverage_exit,
        "coverage_exit_managed": coverage_exit_managed,
        "coverage_non_reject": coverage_active,
    }

    managed_null_rates = []
    for col in ("entry_id", "pred_mu_alpha", "pred_mu_dir_conf", "alpha_vpin", "alpha_hurst", "regime"):
        if col in coverage_exit_managed:
            managed_null_rates.append(float(coverage_exit_managed[col]["null_rate"]))
    max_managed_null = max(managed_null_rates) if managed_null_rates else 1.0
    managed_match_rate = _rate(managed_match_all, managed_with_link)
    out["targets"] = {
        "managed_exit_null_rate_lt_1pct": bool(max_managed_null < 0.01),
        "managed_exit_match_rate_gt_95pct": bool(managed_match_rate > 0.95),
        "max_managed_null_rate": float(max_managed_null),
        "managed_match_rate": float(managed_match_rate),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path


def _sign(x: float | None) -> int:
    if x is None:
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _is_forced_reason(reason: str) -> bool:
    r = (reason or "").lower()
    return any(k in r for k in ("liquidation", "unrealized_dd", "emergency_stop", "kill", "drawdown"))


def _safe_rate(n: int, d: int) -> float:
    return float(n / d) if d > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose entry-vs-exit issues on linked trade cohort.")
    ap.add_argument("--db", default="state/bot_data_live.db", help="SQLite DB path")
    ap.add_argument("--since-id", type=int, default=0, help="Only include rows with id > since_id")
    ap.add_argument("--recent", type=int, default=5000, help="Max rows to inspect")
    ap.add_argument("--out", default="state/entry_exit_diagnosis_report.json", help="Output JSON path")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cols = {str(r[1]) for r in cur.execute("PRAGMA table_info(trades)").fetchall()}
    entry_id_sql = "entry_id" if "entry_id" in cols else "NULL AS entry_id"
    entry_link_sql = "entry_link_id" if "entry_link_id" in cols else "NULL AS entry_link_id"

    rows = cur.execute(
        f"""
        SELECT
            id, action, symbol, side, timestamp_ms, {entry_id_sql}, {entry_link_sql}, trade_uid,
            roe, entry_reason, pred_mu_alpha, pred_mu_dir_conf,
            json_extract(raw_data, '$.opt_hold_entry_sec') AS opt_hold_entry_sec,
            hold_duration_sec
        FROM trades
        WHERE id > ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(args.since_id), int(args.recent)),
    ).fetchall()
    conn.close()

    rows = list(rows)[::-1]  # ASC
    if not rows:
        out = {
            "timestamp_ms": int(time.time() * 1000),
            "db": str(db_path),
            "since_id": int(args.since_id),
            "recent": int(args.recent),
            "message": "no rows",
        }
        Path(args.out).write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=True, indent=2))
        return

    entry_actions = {"ENTER", "SPREAD"}
    exit_actions = {"EXIT", "REBAL_EXIT"}

    entries_by_link = {}
    exits = []
    def _entry_key(row) -> str:
        v_link = row["entry_link_id"] if "entry_link_id" in row.keys() else None
        v_id = row["entry_id"] if "entry_id" in row.keys() else None
        v = v_link if v_link is not None else v_id
        return str(v or "").strip()

    for r in rows:
        act = str(r["action"]).upper()
        link = _entry_key(r)
        if act in entry_actions and link:
            entries_by_link[link] = r
        elif act in exit_actions:
            exits.append(r)

    linked_exits = []
    unlinked_exits = []
    for ex in exits:
        link = _entry_key(ex)
        if link and link in entries_by_link:
            linked_exits.append((ex, entries_by_link[link]))
        else:
            unlinked_exits.append(ex)

    reason_stats = defaultdict(lambda: {"n": 0, "sum_roe": 0.0, "loss_n": 0})

    direction_eval_n = 0
    direction_miss_n = 0
    forced_loss_n = 0
    exit_timing_issue_n = 0
    entry_issue_n = 0
    hold_ratio_n = 0
    hold_ratio_under_n = 0
    hold_ratio_over_n = 0
    hold_ratio_near_n = 0

    for ex, en in linked_exits:
        rr = ex["roe"]
        rr_f = float(rr) if rr is not None else 0.0
        reason = str(ex["entry_reason"] or "")
        rs = reason_stats[reason]
        rs["n"] += 1
        rs["sum_roe"] += rr_f
        if rr_f < 0:
            rs["loss_n"] += 1

        pred_mu = en["pred_mu_alpha"]
        pred_s = _sign(float(pred_mu) if pred_mu is not None else None)
        real_s = _sign(rr_f)
        if pred_s != 0 and real_s != 0:
            direction_eval_n += 1
            if pred_s != real_s:
                direction_miss_n += 1

        if rr_f < 0 and _is_forced_reason(reason):
            forced_loss_n += 1

        # Hold-based heuristic
        hold_sec = ex["hold_duration_sec"]
        opt_hold = ex["opt_hold_entry_sec"]
        hold_ratio = None
        try:
            if hold_sec is not None and opt_hold is not None and float(opt_hold) > 0:
                hold_ratio = float(hold_sec) / float(opt_hold)
        except Exception:
            hold_ratio = None

        if hold_ratio is not None and rr_f < 0:
            hold_ratio_n += 1
            if hold_ratio < 0.8:
                hold_ratio_under_n += 1
                exit_timing_issue_n += 1
            elif hold_ratio > 1.2:
                hold_ratio_over_n += 1
                exit_timing_issue_n += 1
            else:
                hold_ratio_near_n += 1
                if pred_s != 0 and pred_s != real_s:
                    entry_issue_n += 1
                else:
                    exit_timing_issue_n += 1
        elif rr_f < 0:
            # Fallback when hold info is missing.
            if _is_forced_reason(reason):
                exit_timing_issue_n += 1
            elif pred_s != 0 and pred_s != real_s:
                entry_issue_n += 1
            else:
                exit_timing_issue_n += 1

    top_reasons = []
    for reason, v in reason_stats.items():
        n = int(v["n"])
        sum_roe = float(v["sum_roe"])
        top_reasons.append(
            {
                "reason": reason,
                "n": n,
                "avg_roe": float(sum_roe / n) if n > 0 else 0.0,
                "loss_rate": _safe_rate(int(v["loss_n"]), n),
                "sum_roe": sum_roe,
            }
        )
    top_reasons.sort(key=lambda x: x["sum_roe"])
    top_reasons = top_reasons[:15]

    linked_n = len(linked_exits)
    linked_loss_n = sum(1 for ex, _ in linked_exits if (ex["roe"] is not None and float(ex["roe"]) < 0))
    out = {
        "timestamp_ms": int(time.time() * 1000),
        "db": str(db_path),
        "since_id": int(args.since_id),
        "recent": int(args.recent),
        "counts": {
            "rows": int(len(rows)),
            "entries_with_link": int(len(entries_by_link)),
            "exits_total": int(len(exits)),
            "exits_linked": int(linked_n),
            "exits_unlinked": int(len(unlinked_exits)),
            "linked_rate": _safe_rate(linked_n, len(exits)),
        },
        "entry_direction": {
            "eval_n": int(direction_eval_n),
            "miss_n": int(direction_miss_n),
            "miss_rate": _safe_rate(direction_miss_n, direction_eval_n),
        },
        "exit_timing": {
            "forced_loss_n": int(forced_loss_n),
            "forced_loss_rate_over_linked_losses": _safe_rate(forced_loss_n, linked_loss_n),
            "hold_ratio_n": int(hold_ratio_n),
            "hold_under_n": int(hold_ratio_under_n),
            "hold_over_n": int(hold_ratio_over_n),
            "hold_near_n": int(hold_ratio_near_n),
        },
        "attribution": {
            "linked_loss_n": int(linked_loss_n),
            "entry_issue_n": int(entry_issue_n),
            "exit_timing_issue_n": int(exit_timing_issue_n),
            "entry_issue_rate": _safe_rate(entry_issue_n, linked_loss_n),
            "exit_timing_issue_rate": _safe_rate(exit_timing_issue_n, linked_loss_n),
            "dominant_issue": (
                "entry_direction_or_signal_quality"
                if entry_issue_n > exit_timing_issue_n
                else "exit_timing_or_risk_management"
                if exit_timing_issue_n > entry_issue_n
                else "mixed"
            ),
        },
        "top_reasons_by_sum_roe": top_reasons,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


EXIT_ACTIONS = ("EXIT", "REBAL_EXIT", "KILL", "MANUAL", "EXTERNAL")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _fetch_closed_total(conn: sqlite3.Connection) -> int:
    q = (
        "SELECT COUNT(*) AS n FROM trades "
        "WHERE UPPER(action) IN ('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL')"
    )
    row = conn.execute(q).fetchone()
    return int(row[0] if row else 0)


def _fetch_new_exits(conn: sqlite3.Connection, ts_ms_from: int) -> list[sqlite3.Row]:
    q = (
        "SELECT id, timestamp_ms, symbol, action, entry_reason, hold_duration_sec, roe "
        "FROM trades "
        "WHERE UPPER(action) IN ('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL') "
        "  AND timestamp_ms >= ? "
        "ORDER BY timestamp_ms ASC, id ASC"
    )
    cur = conn.execute(q, (int(ts_ms_from),))
    return cur.fetchall()


def _parse_filter_log_delta(log_path: Path, offset: int) -> tuple[int, dict[str, int]]:
    counts = {
        "total_filter_rows": 0,
        "blocked_rows": 0,
        "nx_blocked_rows": 0,
        "sq_blocked_rows": 0,
    }
    if not log_path.exists():
        return offset, counts
    try:
        with log_path.open("rb") as f:
            file_size = f.seek(0, os.SEEK_END)
            if offset < 0 or offset > file_size:
                offset = file_size
            f.seek(offset, os.SEEK_SET)
            blob = f.read()
            new_offset = f.tell()
    except Exception:
        return offset, counts

    if not blob:
        return offset, counts
    try:
        text = blob.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    for line in text.splitlines():
        if "[FILTER]" not in line:
            continue
        if ("blocked:" not in line) and ("all_pass" not in line):
            continue
        counts["total_filter_rows"] += 1
        if "blocked:" in line:
            ll = line.lower()
            counts["blocked_rows"] += 1
            if "net_expectancy" in ll:
                counts["nx_blocked_rows"] += 1
            if "symbol_quality" in ll:
                counts["sq_blocked_rows"] += 1
    return new_offset, counts


def _maybe_set_baseline(state: dict[str, Any], report: dict[str, Any]) -> None:
    if state.get("baseline_set"):
        return
    if int(report.get("new_exits", 0) or 0) <= 0:
        return
    state["baseline_set"] = True
    state["baseline_event_mc_share"] = float(report.get("event_mc_share", 0.0) or 0.0)
    state["baseline_unrealized_dd_share"] = float(report.get("unrealized_dd_share", 0.0) or 0.0)
    state["baseline_event_mc_early_share"] = float(report.get("event_mc_early_share", 0.0) or 0.0)
    state["baseline_unrealized_dd_early_share"] = float(report.get("unrealized_dd_early_share", 0.0) or 0.0)


def _build_report(
    *,
    state: dict[str, Any],
    new_exits_rows: list[sqlite3.Row],
    closed_total_now: int,
    early_hold_sec: float,
    progress_payload: dict[str, Any],
    status_payload: dict[str, Any],
) -> dict[str, Any]:
    def _pick(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    cfg = status_payload.get("config") if isinstance(status_payload, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}
    prog = status_payload.get("progress") if isinstance(status_payload, dict) else {}
    if not isinstance(prog, dict):
        prog = {}

    new_exits = int(len(new_exits_rows))
    event_mc_n = 0
    dd_n = 0
    event_mc_early_n = 0
    dd_early_n = 0
    for r in new_exits_rows:
        reason = str(r["entry_reason"] or "").strip().lower()
        hold_sec = _safe_float(r["hold_duration_sec"], None)
        is_early = hold_sec is not None and hold_sec <= float(early_hold_sec)
        if "event_mc_exit" in reason:
            event_mc_n += 1
            if is_early:
                event_mc_early_n += 1
        if "unrealized_dd" in reason:
            dd_n += 1
            if is_early:
                dd_early_n += 1

    total_filter_rows = int(state.get("total_filter_rows", 0) or 0)
    blocked_rows = int(state.get("blocked_rows", 0) or 0)
    nx_blocked_rows = int(state.get("nx_blocked_rows", 0) or 0)
    sq_blocked_rows = int(state.get("sq_blocked_rows", 0) or 0)

    event_mc_share = float(event_mc_n / max(1, new_exits))
    dd_share = float(dd_n / max(1, new_exits))
    event_mc_early_share = float(event_mc_early_n / max(1, event_mc_n))
    dd_early_share = float(dd_early_n / max(1, dd_n))

    report = {
        "ts_ms": _now_ms(),
        "monitor_start_ms": int(state["monitor_start_ms"]),
        "elapsed_sec": round((_now_ms() - int(state["monitor_start_ms"])) / 1000.0, 1),
        "closed_total_baseline": int(state["closed_total_baseline"]),
        "closed_total_now": int(closed_total_now),
        "new_exits": int(new_exits),
        "early_hold_sec": float(early_hold_sec),
        "entry_filter": {
            "total_filter_rows": int(total_filter_rows),
            "blocked_rows": int(blocked_rows),
            "nx_blocked_rows": int(nx_blocked_rows),
            "sq_blocked_rows": int(sq_blocked_rows),
            "blocked_rate_total": float(blocked_rows / max(1, total_filter_rows)),
            "nx_block_rate_total": float(nx_blocked_rows / max(1, total_filter_rows)),
            "sq_block_rate_total": float(sq_blocked_rows / max(1, total_filter_rows)),
            "nx_share_in_blocked": float(nx_blocked_rows / max(1, blocked_rows)),
            "sq_share_in_blocked": float(sq_blocked_rows / max(1, blocked_rows)),
        },
        "exit_mix": {
            "event_mc_exit_n": int(event_mc_n),
            "unrealized_dd_n": int(dd_n),
            "event_mc_share": float(event_mc_share),
            "unrealized_dd_share": float(dd_share),
            "event_mc_early_n": int(event_mc_early_n),
            "unrealized_dd_early_n": int(dd_early_n),
            "event_mc_early_share": float(event_mc_early_share),
            "unrealized_dd_early_share": float(dd_early_share),
        },
        "reval_progress": {
            "target_new": _pick(status_payload.get("target_new"), cfg.get("target_new"), progress_payload.get("target_new")),
            "new_closed_total": _pick(status_payload.get("new_closed_total"), progress_payload.get("new_closed_total")),
            "new_closed_total_cum": _pick(status_payload.get("new_closed_total_cum"), progress_payload.get("new_closed_total_cum")),
            "reports_ready": _pick(status_payload.get("reports_ready"), prog.get("completed_reports_total"), progress_payload.get("reports_ready"), progress_payload.get("completed_reports_total")),
            "reports_total": _pick(status_payload.get("reports_total"), progress_payload.get("reports_total"), 3),
            "batch_id": _pick(status_payload.get("batch_id"), prog.get("completed_batches"), progress_payload.get("batch_id"), progress_payload.get("completed_batches")),
            "ready": _pick(status_payload.get("ready"), progress_payload.get("ready")),
        },
    }

    if bool(state.get("baseline_set")):
        report["delta_from_baseline"] = {
            "event_mc_share_delta": float(event_mc_share - float(state.get("baseline_event_mc_share", 0.0) or 0.0)),
            "unrealized_dd_share_delta": float(dd_share - float(state.get("baseline_unrealized_dd_share", 0.0) or 0.0)),
            "event_mc_early_share_delta": float(event_mc_early_share - float(state.get("baseline_event_mc_early_share", 0.0) or 0.0)),
            "unrealized_dd_early_share_delta": float(dd_early_share - float(state.get("baseline_unrealized_dd_early_share", 0.0) or 0.0)),
        }
    else:
        report["delta_from_baseline"] = {
            "event_mc_share_delta": None,
            "unrealized_dd_share_delta": None,
            "event_mc_early_share_delta": None,
            "unrealized_dd_early_share_delta": None,
        }
    return report


def _print_snapshot(report: dict[str, Any]) -> None:
    ef = report["entry_filter"]
    ex = report["exit_mix"]
    rv = report["reval_progress"]
    d = report["delta_from_baseline"]
    print(
        "[LIVE_EXIT_MON] "
        f"new_exits={report['new_exits']} "
        f"NX={ef['nx_block_rate_total']*100:.1f}% "
        f"SQ={ef['sq_block_rate_total']*100:.1f}% "
        f"event_mc={ex['event_mc_share']*100:.1f}% "
        f"dd={ex['unrealized_dd_share']*100:.1f}% "
        f"event_early={ex['event_mc_early_share']*100:.1f}% "
        f"dd_early={ex['unrealized_dd_early_share']*100:.1f}% "
        f"delta(event_mc={_safe_float(d.get('event_mc_share_delta'), 0.0)*100:+.1f}%p, dd={_safe_float(d.get('unrealized_dd_share_delta'), 0.0)*100:+.1f}%p) "
        f"reval={rv.get('new_closed_total')}/{rv.get('target_new')}",
        flush=True,
    )


def _append_history_line(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")


def _extract_summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    ef = report.get("entry_filter") or {}
    ex = report.get("exit_mix") or {}
    return {
        "nx_block_rate_total": float(_safe_float(ef.get("nx_block_rate_total"), 0.0) or 0.0),
        "sq_block_rate_total": float(_safe_float(ef.get("sq_block_rate_total"), 0.0) or 0.0),
        "event_mc_share": float(_safe_float(ex.get("event_mc_share"), 0.0) or 0.0),
        "unrealized_dd_share": float(_safe_float(ex.get("unrealized_dd_share"), 0.0) or 0.0),
        "event_mc_early_share": float(_safe_float(ex.get("event_mc_early_share"), 0.0) or 0.0),
        "unrealized_dd_early_share": float(_safe_float(ex.get("unrealized_dd_early_share"), 0.0) or 0.0),
    }


def _metric_delta(curr: dict[str, float], ref: dict[str, float] | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    keys = (
        "nx_block_rate_total",
        "sq_block_rate_total",
        "event_mc_share",
        "unrealized_dd_share",
        "event_mc_early_share",
        "unrealized_dd_early_share",
    )
    for k in keys:
        c = _safe_float(curr.get(k), None)
        r = _safe_float((ref or {}).get(k) if isinstance(ref, dict) else None, None)
        out[k] = None if (c is None or r is None) else float(c - r)
    return out


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{float(v)*100:.1f}%"


def _fmt_delta_pp(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{float(v)*100:+.1f}%p"


def _print_summary(summary: dict[str, Any]) -> None:
    m = summary.get("metrics") or {}
    d_prev = summary.get("delta_prev") or {}
    d_start = summary.get("delta_since_start") or {}
    print(
        "[LIVE_EXIT_MON_SUMMARY] "
        f"new_exits={summary.get('new_exits')} "
        f"NX={_fmt_pct(_safe_float(m.get('nx_block_rate_total'), None))} "
        f"SQ={_fmt_pct(_safe_float(m.get('sq_block_rate_total'), None))} "
        f"event={_fmt_pct(_safe_float(m.get('event_mc_share'), None))} "
        f"dd={_fmt_pct(_safe_float(m.get('unrealized_dd_share'), None))} "
        f"prevΔ(event={_fmt_delta_pp(_safe_float(d_prev.get('event_mc_share'), None))}, dd={_fmt_delta_pp(_safe_float(d_prev.get('unrealized_dd_share'), None))}, NX={_fmt_delta_pp(_safe_float(d_prev.get('nx_block_rate_total'), None))}, SQ={_fmt_delta_pp(_safe_float(d_prev.get('sq_block_rate_total'), None))}) "
        f"startΔ(event={_fmt_delta_pp(_safe_float(d_start.get('event_mc_share'), None))}, dd={_fmt_delta_pp(_safe_float(d_start.get('unrealized_dd_share'), None))}, NX={_fmt_delta_pp(_safe_float(d_start.get('nx_block_rate_total'), None))}, SQ={_fmt_delta_pp(_safe_float(d_start.get('sq_block_rate_total'), None))})",
        flush=True,
    )


def run(args: argparse.Namespace) -> int:
    state_path = Path(args.state_file)
    log_path = Path(args.engine_log)
    out_path = Path(args.out)
    history_path = Path(args.history_file)
    summary_out_path = Path(args.summary_out)
    summary_history_path = Path(args.summary_history_file)
    progress_path = Path(args.progress_file)
    status_path = Path(args.status_file)
    db_path = Path(args.db)

    state = _read_json(state_path, {})
    if not state:
        state = {
            "monitor_start_ms": _now_ms(),
            "closed_total_baseline": 0,
            "log_offset": 0,
            "total_filter_rows": 0,
            "blocked_rows": 0,
            "nx_blocked_rows": 0,
            "sq_blocked_rows": 0,
            "baseline_set": False,
            "baseline_event_mc_share": 0.0,
            "baseline_unrealized_dd_share": 0.0,
            "baseline_event_mc_early_share": 0.0,
            "baseline_unrealized_dd_early_share": 0.0,
            "summary_active": False,
            "summary_start_ts_ms": 0,
            "summary_start_exits": 0,
            "summary_start_metrics": {},
            "summary_prev_ts_ms": 0,
            "summary_prev_metrics": {},
            "summary_last_emit_ms": 0,
            "summary_last_emit_exits": 0,
        }
        if log_path.exists():
            try:
                state["log_offset"] = int(log_path.stat().st_size)
            except Exception:
                state["log_offset"] = 0

    interval_sec = max(1.0, float(args.interval_sec))
    early_hold_sec = float(args.early_hold_sec)
    summary_start_exits = max(1, int(args.summary_start_exits))
    summary_interval_ms = int(max(1.0, float(args.summary_interval_sec)) * 1000.0)

    while True:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            closed_total_now = _fetch_closed_total(conn)
            if int(state.get("closed_total_baseline", 0) or 0) <= 0:
                state["closed_total_baseline"] = int(closed_total_now)
            new_exits_rows = _fetch_new_exits(conn, int(state["monitor_start_ms"]))
            conn.close()
        except Exception as e:
            print(f"[LIVE_EXIT_MON][ERR] db read failed: {e}")
            if args.run_once:
                return 1
            time.sleep(interval_sec)
            continue

        new_offset, delta_counts = _parse_filter_log_delta(log_path, int(state.get("log_offset", 0) or 0))
        state["log_offset"] = int(new_offset)
        for k, v in delta_counts.items():
            state[k] = int(state.get(k, 0) or 0) + int(v)

        progress_payload = _read_json(progress_path, {})
        status_payload = _read_json(status_path, {})
        report = _build_report(
            state=state,
            new_exits_rows=new_exits_rows,
            closed_total_now=closed_total_now,
            early_hold_sec=early_hold_sec,
            progress_payload=progress_payload if isinstance(progress_payload, dict) else {},
            status_payload=status_payload if isinstance(status_payload, dict) else {},
        )
        _maybe_set_baseline(state, report)
        if bool(state.get("baseline_set")):
            report = _build_report(
                state=state,
                new_exits_rows=new_exits_rows,
                closed_total_now=closed_total_now,
                early_hold_sec=early_hold_sec,
                progress_payload=progress_payload if isinstance(progress_payload, dict) else {},
                status_payload=status_payload if isinstance(status_payload, dict) else {},
            )

        now_ms = _now_ms()
        new_exits_now = int(report.get("new_exits", 0) or 0)
        curr_metrics = _extract_summary_metrics(report)
        summary_active = bool(state.get("summary_active", False))
        auto_summary = {
            "enabled": True,
            "active": False,
            "start_exits_threshold": int(summary_start_exits),
            "remaining_to_start": int(max(0, summary_start_exits - new_exits_now)),
        }

        if new_exits_now >= summary_start_exits:
            if not summary_active:
                state["summary_active"] = True
                state["summary_start_ts_ms"] = int(now_ms)
                state["summary_start_exits"] = int(new_exits_now)
                state["summary_start_metrics"] = dict(curr_metrics)
                state["summary_prev_ts_ms"] = int(now_ms)
                state["summary_prev_metrics"] = dict(curr_metrics)
                state["summary_last_emit_ms"] = 0
                state["summary_last_emit_exits"] = int(new_exits_now)

            prev_metrics = state.get("summary_prev_metrics") if isinstance(state.get("summary_prev_metrics"), dict) else {}
            start_metrics = state.get("summary_start_metrics") if isinstance(state.get("summary_start_metrics"), dict) else {}
            delta_prev = _metric_delta(curr_metrics, prev_metrics if prev_metrics else None)
            delta_start = _metric_delta(curr_metrics, start_metrics if start_metrics else None)

            auto_summary = {
                "enabled": True,
                "active": True,
                "start_exits_threshold": int(summary_start_exits),
                "start_ts_ms": int(state.get("summary_start_ts_ms", 0) or 0),
                "start_exits": int(state.get("summary_start_exits", 0) or 0),
                "new_exits": int(new_exits_now),
                "since_start_exits": int(max(0, new_exits_now - int(state.get("summary_start_exits", 0) or 0))),
                "metrics": dict(curr_metrics),
                "delta_prev": delta_prev,
                "delta_since_start": delta_start,
                "prev_ts_ms": int(state.get("summary_prev_ts_ms", 0) or 0),
                "last_emit_ts_ms": int(state.get("summary_last_emit_ms", 0) or 0),
                "last_emit_exits": int(state.get("summary_last_emit_exits", 0) or 0),
            }

            last_emit_ms = int(state.get("summary_last_emit_ms", 0) or 0)
            should_emit = (last_emit_ms <= 0) or ((now_ms - last_emit_ms) >= summary_interval_ms)
            if should_emit:
                emit_payload = {
                    "ts_ms": int(now_ms),
                    "summary_interval_sec": float(args.summary_interval_sec),
                    **auto_summary,
                }
                _write_json_atomic(summary_out_path, emit_payload)
                _append_history_line(summary_history_path, emit_payload)
                _print_summary(emit_payload)
                state["summary_last_emit_ms"] = int(now_ms)
                state["summary_last_emit_exits"] = int(new_exits_now)
                auto_summary["last_emit_ts_ms"] = int(now_ms)
                auto_summary["last_emit_exits"] = int(new_exits_now)

            state["summary_prev_ts_ms"] = int(now_ms)
            state["summary_prev_metrics"] = dict(curr_metrics)
        else:
            state["summary_active"] = False
            state["summary_start_ts_ms"] = 0
            state["summary_start_exits"] = 0
            state["summary_start_metrics"] = {}
            state["summary_prev_ts_ms"] = 0
            state["summary_prev_metrics"] = {}
            state["summary_last_emit_ms"] = 0
            state["summary_last_emit_exits"] = 0

        report["auto_summary"] = auto_summary

        _write_json_atomic(out_path, report)
        _append_history_line(history_path, report)
        _write_json_atomic(state_path, state)
        _print_snapshot(report)

        if args.run_once:
            return 0
        time.sleep(interval_sec)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live monitor for NX/SQ block rates and early exit mix.")
    p.add_argument("--db", type=str, default="state/bot_data_live.db")
    p.add_argument("--engine-log", type=str, default="state/codex_engine.log")
    p.add_argument("--progress-file", type=str, default="state/auto_reval_progress.json")
    p.add_argument("--status-file", type=str, default="state/auto_reval_db_report.json")
    p.add_argument("--out", type=str, default="state/live_exit_quality_report.json")
    p.add_argument("--state-file", type=str, default="state/live_exit_quality_monitor_state.json")
    p.add_argument("--history-file", type=str, default="state/live_exit_quality_report.jsonl")
    p.add_argument("--interval-sec", type=float, default=30.0)
    p.add_argument("--early-hold-sec", type=float, default=float(os.environ.get("EXIT_MIN_HOLD_SEC", 60.0) or 60.0))
    p.add_argument("--summary-start-exits", type=int, default=20)
    p.add_argument("--summary-interval-sec", type=float, default=120.0)
    p.add_argument("--summary-out", type=str, default="state/live_exit_quality_summary.json")
    p.add_argument("--summary-history-file", type=str, default="state/live_exit_quality_summary.jsonl")
    p.add_argument("--run-once", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run(_build_arg_parser().parse_args()))

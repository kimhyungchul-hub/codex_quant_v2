#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Wait until new_closed_total reaches threshold, then run DB report suite.")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--baseline-file", default="state/reval_baseline.json")
    ap.add_argument("--target-new", type=int, default=200)
    ap.add_argument("--poll-sec", type=float, default=15.0)
    ap.add_argument("--timeout-sec", type=float, default=0.0, help="0 means no timeout")
    ap.add_argument("--recent", type=int, default=5000)
    ap.add_argument("--obs-out", default="state/trade_observability_report_new.json")
    ap.add_argument("--diag-out", default="state/entry_exit_diagnosis_report.json")
    ap.add_argument("--cf-out", default="state/counterfactual_replay_report.json")
    ap.add_argument("--status-out", default="state/auto_reval_db_report.json")
    ap.add_argument("--max-hold-min", type=int, default=90)
    ap.add_argument("--entry-sample-limit", type=int, default=1200)
    ap.add_argument("--stage4-baseline-out", default="state/stage4_liq_regret_baseline.json")
    ap.add_argument("--stage4-compare-out", default="state/stage4_liq_regret_compare.json")
    ap.add_argument("--progress-file", default="state/auto_reval_progress.json")
    ap.add_argument("--stage4-set-baseline-if-missing", type=int, default=1)
    ap.add_argument("--stage4-reset-baseline", type=int, default=0)
    ap.add_argument("--roll-baseline", type=int, default=0, help="1 to set baseline_closed=closed_total after run")
    args = ap.parse_args()

    cwd = Path(__file__).resolve().parents[1]
    db_path = (cwd / args.db).resolve() if not os.path.isabs(args.db) else Path(args.db)
    baseline_path = (cwd / args.baseline_file).resolve() if not os.path.isabs(args.baseline_file) else Path(args.baseline_file)
    obs_out = (cwd / args.obs_out).resolve() if not os.path.isabs(args.obs_out) else Path(args.obs_out)
    diag_out = (cwd / args.diag_out).resolve() if not os.path.isabs(args.diag_out) else Path(args.diag_out)
    cf_out = (cwd / args.cf_out).resolve() if not os.path.isabs(args.cf_out) else Path(args.cf_out)
    status_out = (cwd / args.status_out).resolve() if not os.path.isabs(args.status_out) else Path(args.status_out)
    stage4_baseline_out = (cwd / args.stage4_baseline_out).resolve() if not os.path.isabs(args.stage4_baseline_out) else Path(args.stage4_baseline_out)
    stage4_compare_out = (cwd / args.stage4_compare_out).resolve() if not os.path.isabs(args.stage4_compare_out) else Path(args.stage4_compare_out)
    progress_path = (cwd / args.progress_file).resolve() if not os.path.isabs(args.progress_file) else Path(args.progress_file)

    progress = _load_progress(progress_path)
    prev_status = _load_json(status_out) or {}
    prev_batch_kpi = _extract_prev_batch_kpi(prev_status, progress)

    baseline_closed = _load_baseline_closed(baseline_path)
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
                "stage4_baseline_out": str(stage4_baseline_out),
            "stage4_compare_out": str(stage4_compare_out),
            "progress_file": str(progress_path),
            "roll_baseline": bool(int(args.roll_baseline) == 1),
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
            status_out.parent.mkdir(parents=True, exist_ok=True)
            status_out.write_text(json.dumps(status, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0

        time.sleep(max(0.2, float(args.poll_sec)))

    with sqlite3.connect(str(db_path)) as conn:
        since_id = _baseline_since_id(conn, baseline_closed)

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
    runs.append(
        _run(
            [
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
            ],
            cwd,
        )
    )
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
    batch_kpi = _extract_batch_kpi(diag, cf)
    batch_kpi_delta = _kpi_delta(batch_kpi, prev_batch_kpi)
    progress["last_batch_kpi"] = dict(batch_kpi)
    _save_progress(progress_path, progress)
    stage4_metrics = _extract_stage4_metrics(db_path, since_id=int(since_id), cf=cf)

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
                "counterfactual_exit": (cf.get("exit_counterfactual") or {}),
                "batch_kpi": batch_kpi,
                "batch_kpi_delta": batch_kpi_delta,
                "stage4_metrics": stage4_metrics,
                "stage4_compare_ready": bool(stage4_compare is not None),
            },
            "stage4_baseline_action": stage4_baseline_action,
            "stage4_baseline_out": str(stage4_baseline_out),
            "stage4_compare_out": str(stage4_compare_out),
        }
    )

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

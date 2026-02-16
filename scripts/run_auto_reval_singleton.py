#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _now_ms() -> int:
    return int(time.time() * 1000)


def _acquire_singleton_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return None
    fd.seek(0)
    fd.truncate()
    fd.write(str(os.getpid()))
    fd.flush()
    return fd


def _build_reval_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/auto_reval_db_reports.py",
        "--db",
        str(args.db),
        "--target-new",
        str(int(args.target_new)),
        "--poll-sec",
        str(float(args.poll_sec)),
        "--timeout-sec",
        str(float(args.timeout_sec)),
        "--diag-out",
        str(args.diag_out),
        "--cf-out",
        str(args.cf_out),
        "--status-out",
        str(args.status_out),
        "--stage4-baseline-out",
        str(args.stage4_baseline_out),
        "--stage4-compare-out",
        str(args.stage4_compare_out),
        "--progress-file",
        str(args.progress_file),
        "--roll-baseline",
        str(int(args.roll_baseline)),
    ]


def _build_tune_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(args.auto_tune_script),
        "--status-in",
        str(args.status_out),
        "--diag-in",
        str(args.diag_out),
        "--cf-in",
        str(args.cf_out),
        "--stage4-compare-in",
        str(args.stage4_compare_out),
        "--override-out",
        str(args.auto_tune_overrides_out),
        "--state-file",
        str(args.auto_tune_state_file),
        "--status-out",
        str(args.auto_tune_status_out),
        "--run-retrain",
        "1" if int(args.auto_tune_run_retrain) == 1 else "0",
        "--retrain-timeout-sec",
        str(float(args.auto_tune_retrain_timeout_sec)),
    ]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Single-instance auto reval watcher. Prevents duplicate loops and reruns per +N EXIT batch."
    )
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--target-new", type=int, default=200)
    ap.add_argument("--poll-sec", type=float, default=60.0)
    ap.add_argument("--timeout-sec", type=float, default=0.0)
    ap.add_argument("--diag-out", default="state/entry_exit_diagnosis_report.json")
    ap.add_argument("--cf-out", default="state/counterfactual_replay_report_new_window.json")
    ap.add_argument("--status-out", default="state/auto_reval_db_report.json")
    ap.add_argument("--stage4-baseline-out", default="state/stage4_liq_regret_baseline.json")
    ap.add_argument("--stage4-compare-out", default="state/stage4_liq_regret_compare.json")
    ap.add_argument("--progress-file", default="state/auto_reval_progress.json")
    ap.add_argument("--roll-baseline", type=int, default=1)
    ap.add_argument("--restart-sleep-sec", type=float, default=20.0)
    ap.add_argument("--error-sleep-sec", type=float, default=45.0)
    ap.add_argument("--lock-file", default="state/auto_reval_singleton.lock")
    ap.add_argument("--auto-tune-enabled", type=int, default=1)
    ap.add_argument("--auto-tune-script", default="scripts/auto_reval_policy_tuner.py")
    ap.add_argument("--auto-tune-overrides-out", default="state/auto_tune_overrides.json")
    ap.add_argument("--auto-tune-state-file", default="state/auto_tune_state.json")
    ap.add_argument("--auto-tune-status-out", default="state/auto_tune_status.json")
    ap.add_argument("--auto-tune-run-retrain", type=int, default=1)
    ap.add_argument("--auto-tune-retrain-timeout-sec", type=float, default=240.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    lock_path = (repo_root / args.lock_file).resolve()
    lock_fd = _acquire_singleton_lock(lock_path)
    if lock_fd is None:
        print(
            json.dumps(
                {
                    "timestamp_ms": _now_ms(),
                    "status": "skipped",
                    "reason": "lock_held",
                    "lock_file": str(lock_path),
                },
                ensure_ascii=False,
            )
        )
        return 0

    running = True

    def _stop(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running:
        cmd = _build_reval_cmd(args)
        start = time.time()
        rc = subprocess.run(cmd, cwd=str(repo_root)).returncode
        elapsed = time.time() - start
        status = _read_json((repo_root / args.status_out).resolve())
        tune_rc = None
        tune_status = {}
        if int(args.auto_tune_enabled) == 1 and bool(status.get("ready") is True):
            tune_cmd = _build_tune_cmd(args)
            try:
                tune_rc = int(subprocess.run(tune_cmd, cwd=str(repo_root)).returncode)
            except Exception:
                tune_rc = -1
            tune_status = _read_json((repo_root / args.auto_tune_status_out).resolve())
        print(
            json.dumps(
                {
                    "timestamp_ms": _now_ms(),
                    "rc": int(rc),
                    "elapsed_sec": float(elapsed),
                    "new_closed_total": status.get("new_closed_total"),
                    "new_closed_total_cum": status.get("new_closed_total_cum"),
                    "target_new": status.get("config", {}).get("target_new", args.target_new),
                    "ready": status.get("ready"),
                    "timeout": status.get("timeout"),
                    "completed_batches": ((status.get("progress") or {}).get("completed_batches") if isinstance(status, dict) else None),
                    "completed_reports_total": ((status.get("progress") or {}).get("completed_reports_total") if isinstance(status, dict) else None),
                    "auto_tune_rc": tune_rc,
                    "auto_tune_batch": (tune_status.get("batch_id") if isinstance(tune_status, dict) else None),
                    "auto_tune_actions": (tune_status.get("actions") if isinstance(tune_status, dict) else None),
                    "auto_tune_retrain_rc": ((tune_status.get("retrain") or {}).get("rc") if isinstance(tune_status, dict) else None),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.once:
            break
        sleep_sec = float(args.restart_sleep_sec if rc == 0 else args.error_sleep_sec)
        for _ in range(int(max(1, sleep_sec * 2))):
            if not running:
                break
            time.sleep(0.5)

    try:
        lock_fd.seek(0)
        lock_fd.truncate()
        lock_fd.flush()
    except Exception:
        pass
    try:
        lock_fd.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

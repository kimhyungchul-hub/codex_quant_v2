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


EXIT_ACTIONS_SQL = "('EXIT','REBAL_EXIT','KILL','MANUAL','EXTERNAL')"


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
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _count_closed_total(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            f"SELECT COUNT(*) FROM trades WHERE action IN {EXIT_ACTIONS_SQL}"
        ).fetchone()
        return int(row[0] if row else 0)
    finally:
        conn.close()


def _run(cmd: list[str], cwd: Path) -> dict[str, Any]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
        return {
            "cmd": cmd,
            "rc": int(p.returncode),
            "elapsed_sec": float(time.time() - t0),
            "stdout_tail": (p.stdout or "")[-4000:],
            "stderr_tail": (p.stderr or "")[-4000:],
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "rc": -1,
            "elapsed_sec": float(time.time() - t0),
            "stdout_tail": "",
            "stderr_tail": str(e),
        }


def _extract_validation_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    rm = report.get("runtime_metrics") if isinstance(report, dict) else {}
    opp = report.get("opp_side") if isinstance(report, dict) else {}
    cs = report.get("mu_alpha_mc_consistency") if isinstance(report, dict) else {}
    if not isinstance(rm, dict):
        rm = {}
    if not isinstance(opp, dict):
        opp = {}
    if not isinstance(cs, dict):
        cs = {}
    return {
        "mu_sign_flip_ratio": _safe_float(rm.get("mu_sign_flip_ratio"), None),
        "opp_side_better_rate": _safe_float(opp.get("opp_side_better_rate"), None),
        "direction_mismatch_rate": _safe_float(cs.get("direction_mismatch_rate"), None),
        "mu_over_cap_rate": _safe_float(cs.get("mu_over_cap_rate"), None),
        "tp_hit_count": _safe_float(rm.get("tp_hit_count"), None),
        "sl_hit_count": _safe_float(rm.get("sl_hit_count"), None),
    }


def _extract_loss_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    cm = report.get("common_patterns") if isinstance(report, dict) else {}
    sm = report.get("sample") if isinstance(report, dict) else {}
    pnl = report.get("pnl_summary") if isinstance(report, dict) else {}
    if not isinstance(cm, dict):
        cm = {}
    if not isinstance(sm, dict):
        sm = {}
    if not isinstance(pnl, dict):
        pnl = {}
    mis = cm.get("mu_sign_mismatch_rate") if isinstance(cm.get("mu_sign_mismatch_rate"), dict) else {}
    return {
        "exits_total": _safe_float(sm.get("exits_total"), None),
        "worst_n": _safe_float(sm.get("worst_n"), None),
        "pnl_avg": _safe_float(pnl.get("avg"), None),
        "pnl_p10": _safe_float(pnl.get("p10"), None),
        "worst_mu_sign_mismatch_rate": _safe_float(mis.get("worst"), None),
        "all_mu_sign_mismatch_rate": _safe_float(mis.get("all"), None),
    }


def _delta(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None:
        return None
    return float(curr - prev)


def _build_compare(pre_v: dict[str, Any], post_v: dict[str, Any], pre_l: dict[str, Any], post_l: dict[str, Any]) -> dict[str, Any]:
    a = _extract_validation_metrics(pre_v)
    b = _extract_validation_metrics(post_v)
    c = _extract_loss_metrics(pre_l)
    d = _extract_loss_metrics(post_l)

    compare_val = {
        k: {
            "pre": a.get(k),
            "post": b.get(k),
            "delta": _delta(_safe_float(b.get(k), None), _safe_float(a.get(k), None)),
        }
        for k in sorted(set(a.keys()) | set(b.keys()))
    }
    compare_loss = {
        k: {
            "pre": c.get(k),
            "post": d.get(k),
            "delta": _delta(_safe_float(d.get(k), None), _safe_float(c.get(k), None)),
        }
        for k in sorted(set(c.keys()) | set(d.keys()))
    }

    return {
        "validation": compare_val,
        "loss_patterns": compare_loss,
    }


def _start_live_monitor_if_needed(repo_root: Path, db: str, force_restart: bool = False) -> dict[str, Any]:
    log_path = Path("/tmp/live_exit_quality_monitor.log")
    pid_path = repo_root / "state/live_exit_quality_monitor.pid"
    if force_restart and pid_path.exists():
        try:
            pid = int((pid_path.read_text(encoding="utf-8") or "0").strip() or "0")
            if pid > 0:
                os.kill(pid, 15)
        except Exception:
            pass

    if pid_path.exists() and (not force_restart):
        try:
            pid = int((pid_path.read_text(encoding="utf-8") or "0").strip() or "0")
            if pid > 0:
                os.kill(pid, 0)
                return {"started": False, "already_running": True, "pid": pid, "log": str(log_path)}
        except Exception:
            pass

    cmd = [
        "python3",
        "scripts/live_exit_quality_monitor.py",
        "--db",
        db,
        "--engine-log",
        "/tmp/engine.log",
        "--interval-sec",
        "30",
        "--summary-start-exits",
        "20",
        "--summary-interval-sec",
        "120",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), stdout=lf, stderr=lf)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    return {"started": True, "already_running": False, "pid": int(proc.pid), "log": str(log_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor new EXIT batch and auto run pre/post validation compare")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--min-new", type=int, default=80)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--poll-sec", type=float, default=15.0)
    ap.add_argument("--timeout-sec", type=float, default=0.0)
    ap.add_argument("--start-live-monitor", type=int, default=1)
    ap.add_argument("--restart-live-monitor", type=int, default=0)
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--worst-n", type=int, default=120)
    ap.add_argument("--pre-validate-out", default="state/mu_mc_runtime_validation_pre_batch.json")
    ap.add_argument("--post-validate-out", default="state/mu_mc_runtime_validation_post_batch.json")
    ap.add_argument("--pre-loss-out", default="state/large_loss_common_patterns_pre_batch.json")
    ap.add_argument("--post-loss-out", default="state/large_loss_common_patterns_post_batch.json")
    ap.add_argument("--compare-out", default="state/mu_mc_prepost_compare.json")
    ap.add_argument("--status-out", default="state/mu_mc_batch_monitor_status.json")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    db_path = (repo_root / args.db).resolve() if not os.path.isabs(args.db) else Path(args.db)
    pre_validate_out = (repo_root / args.pre_validate_out).resolve() if not os.path.isabs(args.pre_validate_out) else Path(args.pre_validate_out)
    post_validate_out = (repo_root / args.post_validate_out).resolve() if not os.path.isabs(args.post_validate_out) else Path(args.post_validate_out)
    pre_loss_out = (repo_root / args.pre_loss_out).resolve() if not os.path.isabs(args.pre_loss_out) else Path(args.pre_loss_out)
    post_loss_out = (repo_root / args.post_loss_out).resolve() if not os.path.isabs(args.post_loss_out) else Path(args.post_loss_out)
    compare_out = (repo_root / args.compare_out).resolve() if not os.path.isabs(args.compare_out) else Path(args.compare_out)
    status_out = (repo_root / args.status_out).resolve() if not os.path.isabs(args.status_out) else Path(args.status_out)

    status: dict[str, Any] = {
        "ts_ms": _now_ms(),
        "config": {
            "db": str(db_path),
            "min_new": int(args.min_new),
            "max_new": int(args.max_new),
            "poll_sec": float(args.poll_sec),
            "timeout_sec": float(args.timeout_sec),
            "limit": int(args.limit),
            "worst_n": int(args.worst_n),
        },
        "phase": "init",
    }
    _write_json(status_out, status)

    if int(args.start_live_monitor) == 1:
        mon = _start_live_monitor_if_needed(
            repo_root,
            db=str(args.db),
            force_restart=(int(args.restart_live_monitor) == 1),
        )
        status["live_monitor"] = mon
        status["phase"] = "live_monitor_started"
        _write_json(status_out, status)

    baseline_closed = _count_closed_total(db_path)
    status["baseline_closed"] = int(baseline_closed)
    status["phase"] = "pre_snapshot"
    _write_json(status_out, status)

    run_pre = []
    run_pre.append(
        _run(
            [
                "python3",
                "scripts/validate_mu_mc_runtime.py",
                "--db",
                str(args.db),
                "--log",
                "/tmp/engine.log",
                "--out",
                str(args.pre_validate_out),
            ],
            repo_root,
        )
    )
    run_pre.append(
        _run(
            [
                "python3",
                "scripts/analyze_large_loss_patterns.py",
                "--db",
                str(args.db),
                "--limit",
                str(int(args.limit)),
                "--worst-n",
                str(int(args.worst_n)),
                "--out",
                str(args.pre_loss_out),
            ],
            repo_root,
        )
    )
    status["runs_pre"] = run_pre
    status["phase"] = "waiting_batch"
    status["wait_start_ms"] = _now_ms()
    _write_json(status_out, status)

    start_ts = time.time()
    done = False
    new_closed = 0
    while True:
        closed_now = _count_closed_total(db_path)
        new_closed = max(0, int(closed_now) - int(baseline_closed))
        status.update(
            {
                "heartbeat_ts_ms": _now_ms(),
                "closed_total_now": int(closed_now),
                "new_closed": int(new_closed),
                "remaining_to_min": int(max(0, int(args.min_new) - int(new_closed))),
            }
        )
        _write_json(status_out, status)

        if new_closed >= int(max(1, args.min_new)):
            done = True
            break

        if float(args.timeout_sec) > 0 and (time.time() - start_ts) >= float(args.timeout_sec):
            break

        time.sleep(max(0.2, float(args.poll_sec)))

    status["phase"] = "post_snapshot" if done else "timeout"
    status["batch_reached"] = bool(done)
    status["new_closed_final"] = int(new_closed)
    _write_json(status_out, status)

    if not done:
        return 0

    run_post = []
    run_post.append(
        _run(
            [
                "python3",
                "scripts/validate_mu_mc_runtime.py",
                "--db",
                str(args.db),
                "--log",
                "/tmp/engine.log",
                "--out",
                str(args.post_validate_out),
            ],
            repo_root,
        )
    )
    run_post.append(
        _run(
            [
                "python3",
                "scripts/analyze_large_loss_patterns.py",
                "--db",
                str(args.db),
                "--limit",
                str(int(args.limit)),
                "--worst-n",
                str(int(args.worst_n)),
                "--out",
                str(args.post_loss_out),
            ],
            repo_root,
        )
    )

    pre_v = _read_json(pre_validate_out, {})
    post_v = _read_json(post_validate_out, {})
    pre_l = _read_json(pre_loss_out, {})
    post_l = _read_json(post_loss_out, {})

    compare = {
        "ts_ms": _now_ms(),
        "baseline_closed": int(baseline_closed),
        "new_closed": int(new_closed),
        "target_range": [int(args.min_new), int(args.max_new)],
        "runs_pre": run_pre,
        "runs_post": run_post,
        "compare": _build_compare(pre_v, post_v, pre_l, post_l),
        "paths": {
            "pre_validate": str(pre_validate_out),
            "post_validate": str(post_validate_out),
            "pre_loss": str(pre_loss_out),
            "post_loss": str(post_loss_out),
            "status": str(status_out),
        },
    }
    _write_json(compare_out, compare)

    status["phase"] = "completed"
    status["compare_out"] = str(compare_out)
    status["runs_post"] = run_post
    _write_json(status_out, status)

    print(json.dumps(compare, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
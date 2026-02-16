#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_alpha_pipeline import EXIT_TYPES, _analyze, _extract_rows, _load_npz_count


def _load_baseline(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("baseline_closed", 0) or 0)
    except Exception:
        return 0


def _read_trade_tape(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _current_exits(tape: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in tape if str(r.get("ttype") or r.get("type") or "").upper() in EXIT_TYPES]


def _wait_samples(mlofi_npz: Path, causal_npz: Path, target: int, timeout_sec: float, poll_sec: float) -> dict[str, Any]:
    start = time.time()
    while True:
        m = _load_npz_count(mlofi_npz)
        c = _load_npz_count(causal_npz)
        ok = (m >= target) and (c >= target)
        status = {
            "mlofi_samples": int(m),
            "causal_samples": int(c),
            "target_samples": int(target),
            "ready": bool(ok),
            "elapsed_sec": float(time.time() - start),
        }
        if ok:
            return status
        if timeout_sec > 0 and (time.time() - start) >= timeout_sec:
            status["timeout"] = True
            return status
        time.sleep(max(0.2, float(poll_sec)))


def main() -> None:
    p = argparse.ArgumentParser(description="Wait for 200~300 new exits after baseline, then re-evaluate long/short performance.")
    p.add_argument("--trade-tape", default="state/trade_tape_live.json")
    p.add_argument("--baseline-file", default="state/reval_baseline.json")
    p.add_argument("--mlofi-npz", default="state/mlofi_train_samples.npz")
    p.add_argument("--causal-npz", default="state/causal_train_samples.npz")
    p.add_argument("--target-samples", type=int, default=2000)
    p.add_argument("--min-new", type=int, default=200)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--timeout-sec", type=float, default=0.0, help="0 means no timeout")
    p.add_argument("--out", default="state/alpha_pipeline_report_new_exits.json")
    args = p.parse_args()

    trade_path = Path(args.trade_tape)
    baseline_path = Path(args.baseline_file)
    mlofi_npz = Path(args.mlofi_npz)
    causal_npz = Path(args.causal_npz)

    baseline_closed = _load_baseline(baseline_path)
    if baseline_closed < 0:
        baseline_closed = 0

    readiness = _wait_samples(mlofi_npz, causal_npz, int(args.target_samples), timeout_sec=30.0, poll_sec=max(0.2, args.poll_sec))

    start = time.time()
    timeout_sec = float(args.timeout_sec)
    report: dict[str, Any] = {
        "timestamp": int(time.time() * 1000),
        "baseline_closed": int(baseline_closed),
        "target_new_range": [int(args.min_new), int(args.max_new)],
        "readiness": readiness,
        "config": {
            "trade_tape": str(trade_path),
            "baseline_file": str(baseline_path),
            "poll_sec": float(args.poll_sec),
            "timeout_sec": float(timeout_sec),
        },
    }

    while True:
        tape = _read_trade_tape(trade_path)
        exits = _current_exits(tape)
        now_closed = int(len(exits))
        new_count = max(0, now_closed - int(baseline_closed))

        if new_count >= int(args.min_new):
            new_exits_all = exits[int(baseline_closed):]
            use_count = min(int(args.max_new), len(new_exits_all))
            new_exits = new_exits_all[:use_count]
            rows = _extract_rows(new_exits, limit=max(use_count, 1))
            report.update(
                {
                    "timestamp": int(time.time() * 1000),
                    "current_closed": int(now_closed),
                    "new_closed_total": int(new_count),
                    "new_closed_used": int(use_count),
                    "performance": _analyze(rows),
                    "ready": True,
                }
            )
            break

        if timeout_sec > 0 and (time.time() - start) >= timeout_sec:
            report.update(
                {
                    "timestamp": int(time.time() * 1000),
                    "current_closed": int(now_closed),
                    "new_closed_total": int(new_count),
                    "ready": False,
                    "timeout": True,
                }
            )
            break

        time.sleep(max(0.2, float(args.poll_sec)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt = json.dumps(report, ensure_ascii=False, indent=2)
    out_path.write_text(out_txt + "\n", encoding="utf-8")
    print(out_txt)


if __name__ == "__main__":
    main()

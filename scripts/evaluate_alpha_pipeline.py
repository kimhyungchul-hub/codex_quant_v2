#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXIT_TYPES = {"EXIT", "REBAL_EXIT", "KILL", "MANUAL", "EXTERNAL"}


def _load_npz_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with np.load(path, allow_pickle=True) as data:
            x = data.get("X")
            y = data.get("y")
            if x is None or y is None:
                return 0
            return int(min(len(x), len(y)))
    except Exception:
        return 0


def _sign(x: float | None) -> int:
    if x is None:
        return 0
    try:
        v = float(x)
    except Exception:
        return 0
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


@dataclass
class SideStats:
    n: int = 0
    win: int = 0
    sum_r: float = 0.0

    def add(self, r: float) -> None:
        self.n += 1
        self.sum_r += float(r)
        if r > 0:
            self.win += 1

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": int(self.n),
            "win_rate": float(self.win / self.n) if self.n else 0.0,
            "avg_realized_r": float(self.sum_r / self.n) if self.n else 0.0,
        }


def _extract_rows(trade_tape: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in trade_tape[-max(limit, 1) :]:
        ttype = str(r.get("ttype") or r.get("type") or "").upper()
        if ttype not in EXIT_TYPES:
            continue
        side = str(r.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            continue
        rr = r.get("realized_r")
        if rr is None:
            pnl = r.get("pnl")
            notional = r.get("notional")
            lev = r.get("leverage")
            try:
                if pnl is not None and notional and lev:
                    base = float(notional) / max(float(lev), 1e-9)
                    rr = float(pnl) / max(base, 1e-9)
            except Exception:
                rr = None
        if rr is None:
            continue
        x = dict(r)
        x["realized_r"] = float(rr)
        if x.get("hold_duration_sec") is None:
            try:
                age = x.get("age_sec")
                if age is not None:
                    x["hold_duration_sec"] = float(age)
            except Exception:
                pass
        rows.append(x)
    return rows


def _analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["closed_trades"] = int(len(rows))
    if not rows:
        out["error"] = "no_closed_trades"
        return out

    by_side = {"LONG": SideStats(), "SHORT": SideStats()}
    all_stats = SideStats()
    reason_counter: Counter[str] = Counter()

    dir_eval_n = 0
    dir_eval_hit = 0

    # Entry vs exit attribution
    entry_issue = 0
    exit_timing_issue = 0
    good = 0

    hold_ratio_vals: list[float] = []
    hold_ratio_loss_vals: list[float] = []
    loss_hold_under = 0
    loss_hold_over = 0
    loss_hold_near = 0

    for r in rows:
        side = str(r.get("side") or "").upper()
        rr = float(r.get("realized_r", 0.0))
        by_side[side].add(rr)
        all_stats.add(rr)

        reason = str(r.get("reason") or "")
        if reason:
            reason_counter[reason] += 1

        pred_ev_r = r.get("pred_event_ev_r")
        pred_s = _sign(pred_ev_r)
        if pred_s == 0:
            pred_s = _sign(r.get("pred_mu_alpha"))
        if pred_s == 0:
            pred_s = _sign(r.get("pred_mu_alpha_raw"))
        real_s = _sign(rr)
        if pred_s != 0:
            dir_eval_n += 1
            if pred_s == real_s:
                dir_eval_hit += 1

        opt_hold = r.get("opt_hold_entry_sec")
        hold_sec = r.get("hold_duration_sec")
        hold_ratio = None
        try:
            if opt_hold is not None and hold_sec is not None and float(opt_hold) > 0:
                hold_ratio = float(hold_sec) / float(opt_hold)
                hold_ratio_vals.append(hold_ratio)
                if rr < 0:
                    hold_ratio_loss_vals.append(hold_ratio)
        except Exception:
            hold_ratio = None

        # Attribution heuristic
        if rr > 0:
            good += 1
            continue

        if pred_s != 0 and pred_s != real_s:
            entry_issue += 1
            continue

        # Predicted sign was not clearly wrong, but realized loss -> likely exit/management side.
        if hold_ratio is not None:
            if rr < 0:
                if hold_ratio < 0.7:
                    loss_hold_under += 1
                elif hold_ratio > 1.5:
                    loss_hold_over += 1
                else:
                    loss_hold_near += 1
            if hold_ratio < 0.7 or hold_ratio > 1.5:
                exit_timing_issue += 1
            else:
                # within hold band but still lost: treat as entry/model miss
                entry_issue += 1
        else:
            # no horizon info -> classify by reason
            reason_l = reason.lower()
            if any(k in reason_l for k in ("unrealized_dd", "event_mc_exit", "hold_vs_exit", "hybrid_exit", "flip")):
                exit_timing_issue += 1
            else:
                entry_issue += 1

    out["overall"] = all_stats.as_dict()
    out["by_side"] = {k: v.as_dict() for k, v in by_side.items()}
    out["direction_eval"] = {
        "n": int(dir_eval_n),
        "hit_rate": float(dir_eval_hit / dir_eval_n) if dir_eval_n else None,
    }
    out["top_exit_reasons"] = reason_counter.most_common(12)

    loss_total = max(int(sum(1 for r in rows if float(r.get("realized_r", 0.0)) < 0)), 1)
    out["attribution"] = {
        "good_trades": int(good),
        "loss_trades": int(loss_total),
        "entry_issue": int(entry_issue),
        "exit_timing_issue": int(exit_timing_issue),
        "entry_issue_ratio": float(entry_issue / loss_total),
        "exit_timing_issue_ratio": float(exit_timing_issue / loss_total),
    }

    if hold_ratio_vals:
        out["hold_ratio"] = {
            "n": int(len(hold_ratio_vals)),
            "median": float(np.median(np.asarray(hold_ratio_vals))),
            "p25": float(np.percentile(np.asarray(hold_ratio_vals), 25)),
            "p75": float(np.percentile(np.asarray(hold_ratio_vals), 75)),
        }
    if hold_ratio_loss_vals:
        out["hold_ratio_loss"] = {
            "n": int(len(hold_ratio_loss_vals)),
            "median": float(np.median(np.asarray(hold_ratio_loss_vals))),
            "p25": float(np.percentile(np.asarray(hold_ratio_loss_vals), 25)),
            "p75": float(np.percentile(np.asarray(hold_ratio_loss_vals), 75)),
        }
    out["hold_loss_breakdown"] = {
        "under_hold": int(loss_hold_under),
        "over_hold": int(loss_hold_over),
        "near_opt_hold": int(loss_hold_near),
    }
    if entry_issue > exit_timing_issue:
        out["dominant_issue"] = "entry_direction_or_signal_quality"
    elif exit_timing_issue > entry_issue:
        out["dominant_issue"] = "exit_timing_or_risk_management"
    else:
        out["dominant_issue"] = "mixed"
    return out


def _wait_for_samples(mlofi_npz: Path, causal_npz: Path, target: int, timeout_sec: float, poll_sec: float) -> dict[str, Any]:
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
        time.sleep(max(poll_sec, 0.2))


def main() -> None:
    p = argparse.ArgumentParser(description="Wait for MLOFI/CAUSAL sample readiness and evaluate long/short performance.")
    p.add_argument("--mlofi-npz", default=os.environ.get("ALPHA_MLOFI_TRAIN_PATH", "state/mlofi_train_samples.npz"))
    p.add_argument("--causal-npz", default=os.environ.get("ALPHA_CAUSAL_TRAIN_PATH", "state/causal_train_samples.npz"))
    p.add_argument("--trade-tape", default="state/trade_tape_live.json")
    p.add_argument("--target-samples", type=int, default=int(os.environ.get("ALPHA_WEIGHT_TRAIN_MIN_SAMPLES", 2000) or 2000))
    p.add_argument("--timeout-sec", type=float, default=0.0, help="0이면 무한 대기")
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--recent", type=int, default=1500)
    p.add_argument("--out", default="")
    args = p.parse_args()

    mlofi_npz = Path(args.mlofi_npz)
    causal_npz = Path(args.causal_npz)
    tape_path = Path(args.trade_tape)

    readiness = _wait_for_samples(mlofi_npz, causal_npz, int(args.target_samples), float(args.timeout_sec), float(args.poll_sec))

    report: dict[str, Any] = {
        "timestamp": int(time.time() * 1000),
        "readiness": readiness,
        "config": {
            "trade_tape": str(tape_path),
            "recent": int(args.recent),
        },
    }

    if not tape_path.exists():
        report["error"] = f"trade_tape_not_found: {tape_path}"
    else:
        try:
            with open(tape_path, "r", encoding="utf-8") as f:
                tape = json.load(f)
            if not isinstance(tape, list):
                raise ValueError("trade_tape must be a list")
            rows = _extract_rows(tape, int(args.recent))
            report["performance"] = _analyze(rows)
        except Exception as e:
            report["error"] = f"analysis_failed: {e}"

    out_txt = json.dumps(report, ensure_ascii=False, indent=2)
    print(out_txt)

    if args.out:
        Path(args.out).write_text(out_txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

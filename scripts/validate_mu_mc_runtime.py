#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


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


def _sign(x: float | None, eps: float = 1e-12) -> int:
    if x is None:
        return 0
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _parse_log_metrics(lines: list[str]) -> dict[str, Any]:
    mu_lines = [ln for ln in lines if "MU_SIGN_FLIP_EXIT" in ln]
    exit_lines = [ln for ln in lines if re.search(r"\bEXIT\b", ln)]

    tp_hit = sum(1 for ln in lines if re.search(r"\btp_hit\b|\bTP_HIT\b", ln))
    sl_hit = sum(1 for ln in lines if re.search(r"\bsl_hit\b|\bSL_HIT\b", ln))

    age_vals: list[float] = []
    need_vals: list[int] = []
    for ln in mu_lines:
        m_age = re.search(r"age=([0-9]+(?:\.[0-9]+)?)", ln)
        if m_age:
            age_vals.append(float(m_age.group(1)))
        m_need = re.search(r"need=([0-9]+)", ln)
        if m_need:
            need_vals.append(int(m_need.group(1)))

    mu_ratio = float(len(mu_lines) / len(exit_lines)) if exit_lines else None

    return {
        "mu_sign_flip_exit_count": len(mu_lines),
        "exit_count": len(exit_lines),
        "mu_sign_flip_ratio": mu_ratio,
        "tp_hit_count": int(tp_hit),
        "sl_hit_count": int(sl_hit),
        "age_sec": {
            "n": len(age_vals),
            "avg": (float(statistics.mean(age_vals)) if age_vals else None),
            "min": (float(min(age_vals)) if age_vals else None),
            "max": (float(max(age_vals)) if age_vals else None),
            "lte_2sec_count": int(sum(1 for v in age_vals if v <= 2.0)),
        },
        "need_ticks": {
            "n": len(need_vals),
            "avg": (float(statistics.mean(need_vals)) if need_vals else None),
            "min": (int(min(need_vals)) if need_vals else None),
            "max": (int(max(need_vals)) if need_vals else None),
        },
        "sample_last_mu_lines": mu_lines[-8:],
    }


def _load_opp_side_better_rate(auto_reval_path: Path, cf_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": None,
        "opp_side_better_rate": None,
        "opp_side_better_rate_net": None,
    }

    if auto_reval_path.exists():
        try:
            d = json.loads(auto_reval_path.read_text(encoding="utf-8"))
            summary = d.get("summary") if isinstance(d, dict) else None
            if not isinstance(summary, dict):
                lr = d.get("last_ready") if isinstance(d, dict) else None
                if isinstance(lr, dict):
                    summary = lr.get("summary")
            if isinstance(summary, dict):
                cf = summary.get("counterfactual_entry") or {}
                if cf.get("opp_side_better_rate") is not None:
                    out["source"] = "auto_reval_db_report.summary.counterfactual_entry"
                    out["opp_side_better_rate"] = _safe_float(cf.get("opp_side_better_rate"), None)
                    out["opp_side_better_rate_net"] = _safe_float(cf.get("opp_side_better_rate_net"), None)
                    return out
        except Exception:
            pass

    if cf_path.exists():
        try:
            d = json.loads(cf_path.read_text(encoding="utf-8"))
            entry_cf = d.get("entry_counterfactual") or {}
            out["source"] = "counterfactual_replay_report.entry_counterfactual"
            out["opp_side_better_rate"] = _safe_float(entry_cf.get("opp_side_better_rate"), None)
            out["opp_side_better_rate_net"] = _safe_float(entry_cf.get("opp_side_better_rate_net"), None)
            return out
        except Exception:
            pass

    return out


def _audit_mu_mc_consistency(db_path: Path, sample_n: int = 3000, mu_cap: float = 15.0) -> dict[str, Any]:
    if not db_path.exists():
        return {"error": f"DB not found: {db_path}"}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              id, symbol, regime, realized_pnl, roe,
              pred_mu_alpha, pred_mu_dir_conf, entry_ev,
              hold_duration_sec
            FROM trades
            WHERE action IN ('EXIT', 'REBAL_EXIT', 'KILL', 'MANUAL', 'EXTERNAL')
            ORDER BY timestamp_ms DESC
            LIMIT ?
            """,
            (int(sample_n),),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    total = 0
    total_chop = 0
    mismatch = 0
    mismatch_chop = 0
    low_mu_high_loss = 0
    low_mu_n = 0

    conf_bins = {"low<0.55": [0, 0], "mid0.55-0.65": [0, 0], "high>=0.65": [0, 0]}  # [n, losses]

    extreme_mu_n = 0

    for r in rows:
        mu = _safe_float(r["pred_mu_alpha"], None)
        roe = _safe_float(r["roe"], None)
        if mu is None or roe is None:
            continue
        if abs(mu) > float(mu_cap):
            extreme_mu_n += 1
        total += 1
        reg = str(r["regime"] or "unknown").lower()
        if reg == "chop":
            total_chop += 1

        mu_s = _sign(mu)
        roe_s = _sign(roe)
        if mu_s != 0 and roe_s != 0 and mu_s != roe_s:
            mismatch += 1
            if reg == "chop":
                mismatch_chop += 1

        if abs(mu) < 0.25:
            low_mu_n += 1
            if roe < 0:
                low_mu_high_loss += 1

        conf = _safe_float(r["pred_mu_dir_conf"], None)
        if conf is not None:
            if conf < 0.55:
                key = "low<0.55"
            elif conf < 0.65:
                key = "mid0.55-0.65"
            else:
                key = "high>=0.65"
            conf_bins[key][0] += 1
            if roe < 0:
                conf_bins[key][1] += 1

    conf_loss_rate = {
        k: {
            "n": v[0],
            "loss_rate": (float(v[1] / v[0]) if v[0] else None),
        }
        for k, v in conf_bins.items()
    }

    return {
        "sample_n": int(total),
        "mu_cap_for_audit": float(mu_cap),
        "mu_over_cap_rate": (float(extreme_mu_n / total) if total else None),
        "direction_mismatch_rate": (float(mismatch / total) if total else None),
        "direction_mismatch_rate_chop": (float(mismatch_chop / total_chop) if total_chop else None),
        "low_mu_under_0_25": {
            "n": int(low_mu_n),
            "loss_rate": (float(low_mu_high_loss / low_mu_n) if low_mu_n else None),
        },
        "dir_conf_loss_rate": conf_loss_rate,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto validation for mu_sign_flip/TP-SL/opp_side + mu_alpha-MC consistency audit")
    ap.add_argument("--log", default="/tmp/engine.log")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--auto-reval", default="state/auto_reval_db_report.json")
    ap.add_argument("--cf-report", default="state/counterfactual_replay_report_latest500.json")
    ap.add_argument("--run-counterfactual", action="store_true", help="Refresh counterfactual report before validation")
    ap.add_argument("--cf-limit", type=int, default=500, help="recent exits limit for counterfactual replay")
    ap.add_argument("--out", default="state/mu_mc_runtime_validation.json")
    ap.add_argument("--max-mu-flip-ratio", type=float, default=0.60)
    ap.add_argument("--max-opp-side-better-rate", type=float, default=0.55)
    ap.add_argument("--max-dir-mismatch-rate", type=float, default=0.45)
    ap.add_argument("--mu-cap", type=float, default=15.0)
    args = ap.parse_args()

    if args.run_counterfactual:
        cmd = [
            "python3",
            "scripts/counterfactual_replay.py",
            "--db",
            str(args.db),
            "--out",
            str(args.cf_report),
            "--entry-sample-limit",
            str(int(args.cf_limit)),
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(json.dumps({"counterfactual_refresh_error": str(e), "cmd": cmd}, ensure_ascii=False))

    lines = _load_lines(Path(args.log))
    log_metrics = _parse_log_metrics(lines)
    opp = _load_opp_side_better_rate(Path(args.auto_reval), Path(args.cf_report))
    consistency = _audit_mu_mc_consistency(Path(args.db), sample_n=3000, mu_cap=float(args.mu_cap))

    checks = {
        "mu_sign_flip_ratio_ok": (
            None
            if log_metrics.get("mu_sign_flip_ratio") is None
            else bool(float(log_metrics["mu_sign_flip_ratio"]) <= float(args.max_mu_flip_ratio))
        ),
        "opp_side_better_rate_ok": (
            None
            if opp.get("opp_side_better_rate") is None
            else bool(float(opp["opp_side_better_rate"]) <= float(args.max_opp_side_better_rate))
        ),
        "direction_mismatch_rate_ok": (
            None
            if consistency.get("direction_mismatch_rate") is None
            else bool(float(consistency["direction_mismatch_rate"]) <= float(args.max_dir_mismatch_rate))
        ),
    }

    report = {
        "timestamp_ms": int(time.time() * 1000),
        "inputs": {
            "log": str(args.log),
            "db": str(args.db),
            "auto_reval": str(args.auto_reval),
            "cf_report": str(args.cf_report),
        },
        "runtime_metrics": log_metrics,
        "opp_side": opp,
        "mu_alpha_mc_consistency": consistency,
        "thresholds": {
            "max_mu_flip_ratio": float(args.max_mu_flip_ratio),
            "max_opp_side_better_rate": float(args.max_opp_side_better_rate),
            "max_dir_mismatch_rate": float(args.max_dir_mismatch_rate),
        },
        "checks": checks,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

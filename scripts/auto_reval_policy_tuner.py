#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _is_true(v: Any) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _to_env_str(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if abs(v) >= 1e6 or (0 < abs(v) < 1e-4):
            return f"{v:.10g}"
        return f"{v:.8f}".rstrip("0").rstrip(".")
    return str(v)


def _sanitize_override_value(v: Any) -> Any:
    if isinstance(v, (bool, int, float)):
        return v
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return s


def _pick_metric(status: dict[str, Any], path: list[str]) -> Any:
    cur: Any = status
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _get_effective_float(
    key: str,
    overrides: dict[str, Any],
    default: float,
) -> float:
    if key in overrides:
        v = _safe_float(overrides.get(key), None)
        if v is not None:
            return float(v)
    v = _safe_float(os.environ.get(key), None)
    if v is not None:
        return float(v)
    return float(default)


def _get_effective_int(
    key: str,
    overrides: dict[str, Any],
    default: int,
) -> int:
    if key in overrides:
        return _safe_int(overrides.get(key), default)
    return _safe_int(os.environ.get(key), default)


def _set_float_step(
    out: dict[str, Any],
    key: str,
    cur: float,
    delta: float,
    min_v: float | None = None,
    max_v: float | None = None,
) -> bool:
    nxt = float(cur + delta)
    if min_v is not None:
        nxt = max(float(min_v), nxt)
    if max_v is not None:
        nxt = min(float(max_v), nxt)
    if abs(nxt - float(cur)) < 1e-12:
        return False
    out[key] = float(nxt)
    return True


def _set_int_step(
    out: dict[str, Any],
    key: str,
    cur: int,
    delta: int,
    min_v: int | None = None,
    max_v: int | None = None,
) -> bool:
    nxt = int(cur + delta)
    if min_v is not None:
        nxt = max(int(min_v), nxt)
    if max_v is not None:
        nxt = min(int(max_v), nxt)
    if nxt == int(cur):
        return False
    out[key] = int(nxt)
    return True


def _extract_metrics(
    status: dict[str, Any],
    diag: dict[str, Any],
    cf: dict[str, Any],
    stage4_compare: dict[str, Any],
    last_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status_kpi = _pick_metric(status, ["summary", "batch_kpi"]) if isinstance(status, dict) else {}
    status_kpi = status_kpi if isinstance(status_kpi, dict) else {}
    status_kpi_delta = _pick_metric(status, ["summary", "batch_kpi_delta"]) if isinstance(status, dict) else {}
    status_kpi_delta = status_kpi_delta if isinstance(status_kpi_delta, dict) else {}
    entry_issue_rate = _safe_float(_pick_metric(diag, ["attribution", "entry_issue_rate"]), None)
    entry_issue_ratio = _safe_float(status_kpi.get("entry_issue_ratio"), entry_issue_rate)
    miss_rate = _safe_float(_pick_metric(diag, ["entry_direction", "miss_rate"]), None)
    direction_hit = _safe_float(status_kpi.get("direction_hit"), None)
    if direction_hit is None and miss_rate is not None:
        direction_hit = float(max(0.0, min(1.0, 1.0 - float(miss_rate))))
    avg_exit_regret = _safe_float(_pick_metric(cf, ["exit_counterfactual", "avg_exit_regret"]), None)
    if avg_exit_regret is None:
        avg_exit_regret = _safe_float(status_kpi.get("avg_exit_regret"), None)
    early_like_rate = _safe_float(_pick_metric(cf, ["exit_counterfactual", "early_like_rate"]), None)
    improvable_rate = _safe_float(_pick_metric(cf, ["exit_counterfactual", "improvable_rate_regret_gt_10bps"]), None)
    event_mc_exit_rate = _safe_float(_pick_metric(stage4_compare, ["current", "metrics", "event_mc_exit_rate"]), None)
    liq_like_rate = _safe_float(_pick_metric(stage4_compare, ["current", "metrics", "liq_like_rate"]), None)
    avg_roe = _safe_float(_pick_metric(stage4_compare, ["current", "metrics", "avg_roe"]), None)
    new_closed_total = _safe_int(status.get("new_closed_total"), 0)
    target_new = _safe_int(_pick_metric(status, ["config", "target_new"]), 0)
    completed_batches = _safe_int(_pick_metric(status, ["progress", "completed_batches"]), 0)
    prev_metrics = (last_state or {}).get("last_metrics") if isinstance((last_state or {}).get("last_metrics"), dict) else {}
    prev_direction_hit = _safe_float(prev_metrics.get("direction_hit"), None)
    prev_entry_issue_ratio = _safe_float(prev_metrics.get("entry_issue_ratio"), None)
    prev_avg_exit_regret = _safe_float(prev_metrics.get("avg_exit_regret"), None)
    direction_hit_delta = _safe_float(status_kpi_delta.get("direction_hit"), None)
    entry_issue_ratio_delta = _safe_float(status_kpi_delta.get("entry_issue_ratio"), None)
    avg_exit_regret_delta = _safe_float(status_kpi_delta.get("avg_exit_regret"), None)
    if direction_hit_delta is None and direction_hit is not None and prev_direction_hit is not None:
        direction_hit_delta = float(direction_hit) - float(prev_direction_hit)
    if entry_issue_ratio_delta is None and entry_issue_ratio is not None and prev_entry_issue_ratio is not None:
        entry_issue_ratio_delta = float(entry_issue_ratio) - float(prev_entry_issue_ratio)
    if avg_exit_regret_delta is None and avg_exit_regret is not None and prev_avg_exit_regret is not None:
        avg_exit_regret_delta = float(avg_exit_regret) - float(prev_avg_exit_regret)

    return {
        "entry_issue_rate": entry_issue_rate,
        "entry_issue_ratio": entry_issue_ratio,
        "miss_rate": miss_rate,
        "direction_hit": direction_hit,
        "avg_exit_regret": avg_exit_regret,
        "early_like_rate": early_like_rate,
        "improvable_rate_regret_gt_10bps": improvable_rate,
        "event_mc_exit_rate": event_mc_exit_rate,
        "liq_like_rate": liq_like_rate,
        "avg_roe": avg_roe,
        "direction_hit_delta": direction_hit_delta,
        "entry_issue_ratio_delta": entry_issue_ratio_delta,
        "avg_exit_regret_delta": avg_exit_regret_delta,
        "new_closed_total": int(new_closed_total),
        "target_new": int(target_new),
        "completed_batches": int(completed_batches),
    }


def _normalize_regime(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if ("trend" in s) or ("bull" in s) or ("bear" in s):
        return "trend"
    if ("volatile" in s) or ("random" in s) or ("noise" in s):
        return "volatile"
    if ("mean" in s) or ("revert" in s) or ("chop" in s) or ("range" in s):
        return "chop"
    return "chop"


def _extract_regime_metrics(db_path: Path, since_id: int) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {
        "trend": {
            "exit_n": 0,
            "loss_n": 0,
            "event_n": 0,
            "dd_n": 0,
            "liq_n": 0,
            "sum_roe": 0.0,
            "roe_n": 0,
            "entry_n": 0,
            "reject_n": 0,
            "reject_110007_n": 0,
            "precision_reject_n": 0,
        },
        "chop": {
            "exit_n": 0,
            "loss_n": 0,
            "event_n": 0,
            "dd_n": 0,
            "liq_n": 0,
            "sum_roe": 0.0,
            "roe_n": 0,
            "entry_n": 0,
            "reject_n": 0,
            "reject_110007_n": 0,
            "precision_reject_n": 0,
        },
        "volatile": {
            "exit_n": 0,
            "loss_n": 0,
            "event_n": 0,
            "dd_n": 0,
            "liq_n": 0,
            "sum_roe": 0.0,
            "roe_n": 0,
            "entry_n": 0,
            "reject_n": 0,
            "reject_110007_n": 0,
            "precision_reject_n": 0,
        },
    }
    if not db_path.exists():
        return out
    q = (
        "SELECT regime, entry_reason, roe FROM trades "
        "WHERE id > ? AND action IN ('EXIT','REBAL_EXIT')"
    )
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(q, (int(max(0, since_id)),)).fetchall()
    except Exception:
        return out
    for regime_raw, reason_raw, roe_raw in rows:
        reg = _normalize_regime(regime_raw)
        rtxt = str(reason_raw or "").strip().lower()
        m = out.get(reg)
        if not isinstance(m, dict):
            continue
        m["exit_n"] = int(m.get("exit_n", 0) or 0) + 1
        rr = _safe_float(roe_raw, None)
        if rr is not None:
            m["sum_roe"] = float(m.get("sum_roe", 0.0) or 0.0) + float(rr)
            m["roe_n"] = int(m.get("roe_n", 0) or 0) + 1
            if float(rr) < 0.0:
                m["loss_n"] = int(m.get("loss_n", 0) or 0) + 1
        if "event_mc_exit" in rtxt:
            m["event_n"] = int(m.get("event_n", 0) or 0) + 1
        if "unrealized_dd" in rtxt:
            m["dd_n"] = int(m.get("dd_n", 0) or 0) + 1
        if ("liquidat" in rtxt) or ("kill" in rtxt) or ("emergency" in rtxt):
            m["liq_n"] = int(m.get("liq_n", 0) or 0) + 1
    q_rejects = (
        "SELECT regime, action, entry_reason FROM trades "
        "WHERE id > ? AND action IN ('ENTER','ORDER_REJECT')"
    )
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows2 = conn.execute(q_rejects, (int(max(0, since_id)),)).fetchall()
    except Exception:
        rows2 = []
    for regime_raw, action_raw, reason_raw in rows2:
        reg = _normalize_regime(regime_raw)
        a = str(action_raw or "").strip().upper()
        reason = str(reason_raw or "").strip().lower()
        m = out.get(reg)
        if not isinstance(m, dict):
            continue
        if a == "ENTER":
            m["entry_n"] = int(m.get("entry_n", 0) or 0) + 1
            continue
        if a != "ORDER_REJECT":
            continue
        m["reject_n"] = int(m.get("reject_n", 0) or 0) + 1
        if ("110007" in reason) or ("ab not enough" in reason):
            m["reject_110007_n"] = int(m.get("reject_110007_n", 0) or 0) + 1
        if ("minimum amount precision" in reason) or ("min qty" in reason) or ("min_qty" in reason):
            m["precision_reject_n"] = int(m.get("precision_reject_n", 0) or 0) + 1
    for reg in ("trend", "chop", "volatile"):
        m = out.get(reg, {})
        n = int(m.get("exit_n", 0) or 0)
        rn = int(m.get("roe_n", 0) or 0)
        en = int(m.get("entry_n", 0) or 0)
        rj = int(m.get("reject_n", 0) or 0)
        m["loss_rate"] = float((m.get("loss_n", 0) or 0) / n) if n > 0 else None
        m["event_rate"] = float((m.get("event_n", 0) or 0) / n) if n > 0 else None
        m["dd_rate"] = float((m.get("dd_n", 0) or 0) / n) if n > 0 else None
        m["liq_rate"] = float((m.get("liq_n", 0) or 0) / n) if n > 0 else None
        m["avg_roe"] = float((m.get("sum_roe", 0.0) or 0.0) / rn) if rn > 0 else None
        m["reject_rate"] = float(rj / max(1, en + rj))
        m["reject_110007_rate"] = float((m.get("reject_110007_n", 0) or 0) / max(1, rj))
        m["precision_reject_rate"] = float((m.get("precision_reject_n", 0) or 0) / max(1, rj))
        out[reg] = m
    return out


def _run_retrain(
    repo_root: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    cmd = [sys.executable, "scripts/train_alpha_weights.py"]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(30.0, float(timeout_sec)),
        )
        return {
            "ran": True,
            "rc": int(proc.returncode),
            "elapsed_sec": float(time.time() - start),
            "stdout_tail": (proc.stdout or "")[-3000:],
            "stderr_tail": (proc.stderr or "")[-3000:],
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ran": True,
            "rc": -9,
            "elapsed_sec": float(time.time() - start),
            "stdout_tail": str((e.stdout or "")[-3000:]),
            "stderr_tail": str((e.stderr or "")[-3000:]),
            "error": "timeout",
        }
    except Exception as e:
        return {
            "ran": True,
            "rc": -1,
            "elapsed_sec": float(time.time() - start),
            "stdout_tail": "",
            "stderr_tail": str(e),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reval-driven auto tuning policy updater")
    ap.add_argument("--status-in", default="state/auto_reval_db_report.json")
    ap.add_argument("--diag-in", default="state/entry_exit_diagnosis_report.json")
    ap.add_argument("--cf-in", default="state/counterfactual_replay_report_new_window.json")
    ap.add_argument("--stage4-compare-in", default="state/stage4_liq_regret_compare.json")
    ap.add_argument("--override-in", default="state/auto_tune_overrides.json")
    ap.add_argument("--override-out", default="state/auto_tune_overrides.json")
    ap.add_argument("--state-file", default="state/auto_tune_state.json")
    ap.add_argument("--status-out", default="state/auto_tune_status.json")
    ap.add_argument("--run-retrain", type=int, default=1)
    ap.add_argument("--retrain-timeout-sec", type=float, default=240.0)
    ap.add_argument("--min-entry-issue", type=float, default=0.50)
    ap.add_argument("--min-miss-rate", type=float, default=0.52)
    ap.add_argument("--high-exit-regret", type=float, default=0.015)
    ap.add_argument("--high-early-like", type=float, default=0.55)
    ap.add_argument("--high-event-mc-rate", type=float, default=0.30)
    ap.add_argument("--high-liq-rate", type=float, default=0.03)
    ap.add_argument("--min-regime-samples", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    status_path = (repo_root / args.status_in).resolve() if not os.path.isabs(args.status_in) else Path(args.status_in)
    diag_path = (repo_root / args.diag_in).resolve() if not os.path.isabs(args.diag_in) else Path(args.diag_in)
    cf_path = (repo_root / args.cf_in).resolve() if not os.path.isabs(args.cf_in) else Path(args.cf_in)
    stage4_path = (repo_root / args.stage4_compare_in).resolve() if not os.path.isabs(args.stage4_compare_in) else Path(args.stage4_compare_in)
    override_in_path = (repo_root / args.override_in).resolve() if not os.path.isabs(args.override_in) else Path(args.override_in)
    override_out_path = (repo_root / args.override_out).resolve() if not os.path.isabs(args.override_out) else Path(args.override_out)
    state_file_path = (repo_root / args.state_file).resolve() if not os.path.isabs(args.state_file) else Path(args.state_file)
    status_out_path = (repo_root / args.status_out).resolve() if not os.path.isabs(args.status_out) else Path(args.status_out)

    status = _load_json(status_path)
    diag = _load_json(diag_path)
    cf = _load_json(cf_path)
    stage4 = _load_json(stage4_path)
    existing_override_payload = _load_json(override_in_path)
    existing_overrides = existing_override_payload.get("overrides") if isinstance(existing_override_payload.get("overrides"), dict) else {}
    last_state = _load_json(state_file_path)

    ready = bool(status.get("ready") is True)
    metrics = _extract_metrics(status, diag, cf, stage4, last_state=last_state)
    status_cfg = status.get("config") if isinstance(status.get("config"), dict) else {}
    db_cfg = status_cfg.get("db")
    if db_cfg is None:
        db_path = (repo_root / "state/bot_data_live.db").resolve()
    else:
        db_path = (repo_root / str(db_cfg)).resolve() if not os.path.isabs(str(db_cfg)) else Path(str(db_cfg))
    since_id = _safe_int(status.get("since_id"), 0)
    regime_metrics = _extract_regime_metrics(db_path=db_path, since_id=since_id)
    batch_id = int(max(0, metrics.get("completed_batches") or 0))
    last_batch_id = int(max(0, _safe_int(last_state.get("last_batch_id"), 0)))
    already_processed = bool(ready and batch_id > 0 and batch_id <= last_batch_id)

    result: dict[str, Any] = {
        "timestamp_ms": _now_ms(),
        "ready": bool(ready),
        "batch_id": int(batch_id),
        "last_batch_id": int(last_batch_id),
        "already_processed": bool(already_processed),
        "metrics": metrics,
        "regime_metrics": regime_metrics,
        "actions": [],
        "overrides": {},
        "retrain": {
            "requested": False,
            "reason": "",
            "ran": False,
            "rc": None,
            "elapsed_sec": 0.0,
        },
        "status_paths": {
            "status_in": str(status_path),
            "diag_in": str(diag_path),
            "cf_in": str(cf_path),
            "stage4_compare_in": str(stage4_path),
            "override_out": str(override_out_path),
            "state_file": str(state_file_path),
            "db": str(db_path),
            "since_id": int(since_id),
        },
    }

    if not ready:
        result["actions"].append("skip:not_ready")
        _write_json(status_out_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if already_processed:
        result["actions"].append("skip:already_processed_batch")
        _write_json(status_out_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    out_overrides: dict[str, Any] = {}
    actions: list[str] = []
    retrain_reason: list[str] = []

    entry_issue_rate = _safe_float(metrics.get("entry_issue_ratio"), None)
    if entry_issue_rate is None:
        entry_issue_rate = _safe_float(metrics.get("entry_issue_rate"), None)
    miss_rate = _safe_float(metrics.get("miss_rate"), None)
    direction_hit = _safe_float(metrics.get("direction_hit"), None)
    avg_exit_regret = _safe_float(metrics.get("avg_exit_regret"), None)
    early_like_rate = _safe_float(metrics.get("early_like_rate"), None)
    event_mc_exit_rate = _safe_float(metrics.get("event_mc_exit_rate"), None)
    liq_like_rate = _safe_float(metrics.get("liq_like_rate"), None)
    avg_roe = _safe_float(metrics.get("avg_roe"), None)
    direction_hit_delta = _safe_float(metrics.get("direction_hit_delta"), None)
    entry_issue_ratio_delta = _safe_float(metrics.get("entry_issue_ratio_delta"), None)
    avg_exit_regret_delta = _safe_float(metrics.get("avg_exit_regret_delta"), None)
    batch_new_closed = _safe_int(metrics.get("new_closed_total"), 0)
    top_regret_rows = []
    try:
        top_regret_rows = list((cf.get("exit_counterfactual") or {}).get("top_regret_reasons") or [])
    except Exception:
        top_regret_rows = []
    reason_counts: dict[str, int] = {}
    total_reason_n = 0
    for row in top_regret_rows:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip().lower()
        n = _safe_int(row.get("n"), 0)
        if not reason or n <= 0:
            continue
        reason_counts[reason] = int(reason_counts.get(reason, 0) + n)
        total_reason_n += int(n)

    def _reason_ratio(token: str) -> float:
        if total_reason_n <= 0:
            return 0.0
        needle = str(token).strip().lower()
        hit = 0
        for key, n in reason_counts.items():
            if needle in str(key):
                hit += int(n)
        return float(hit) / float(max(1, total_reason_n))

    event_reason_ratio = _reason_ratio("event_mc_exit")
    unified_flip_ratio = _reason_ratio("unified_flip")
    ev_drop_ratio = _reason_ratio("ev_drop_exit")
    dd_reason_ratio = _reason_ratio("unrealized_dd")
    result["reason_breakdown"] = {
        "event_reason_ratio": float(event_reason_ratio),
        "unified_flip_ratio": float(unified_flip_ratio),
        "ev_drop_ratio": float(ev_drop_ratio),
        "dd_reason_ratio": float(dd_reason_ratio),
        "reason_counts": dict(reason_counts),
        "reason_total": int(total_reason_n),
    }

    direction_bad = bool(
        (entry_issue_rate is not None and entry_issue_rate >= float(args.min_entry_issue))
        or (miss_rate is not None and miss_rate >= float(args.min_miss_rate))
    )
    exit_bad = bool(
        (avg_exit_regret is not None and avg_exit_regret >= float(args.high_exit_regret))
        or (early_like_rate is not None and early_like_rate >= float(args.high_early_like))
        or (event_mc_exit_rate is not None and event_mc_exit_rate >= float(args.high_event_mc_rate))
    )
    risk_bad = bool(
        (liq_like_rate is not None and liq_like_rate >= float(args.high_liq_rate))
        or (avg_roe is not None and avg_roe < -0.0025)
    )
    direction_good = bool(
        (direction_hit is not None and direction_hit >= 0.56)
        and (entry_issue_rate is not None and entry_issue_rate <= 0.45)
        and (not risk_bad)
    )
    try:
        delta_sample_min = int(os.environ.get("AUTO_TUNE_DELTA_SAMPLE_MIN", 40) or 40)
    except Exception:
        delta_sample_min = 40
    delta_sample_min = int(max(20, delta_sample_min))
    try:
        delta_hit_drop = float(os.environ.get("AUTO_TUNE_DELTA_HIT_DROP", 0.010) or 0.010)
    except Exception:
        delta_hit_drop = 0.010
    try:
        delta_entry_rise = float(os.environ.get("AUTO_TUNE_DELTA_ENTRY_ISSUE_RISE", 0.015) or 0.015)
    except Exception:
        delta_entry_rise = 0.015
    try:
        delta_regret_rise = float(os.environ.get("AUTO_TUNE_DELTA_EXIT_REGRET_RISE", 0.0025) or 0.0025)
    except Exception:
        delta_regret_rise = 0.0025
    delta_bad = bool(
        int(batch_new_closed) >= int(delta_sample_min)
        and (
            (direction_hit_delta is not None and float(direction_hit_delta) <= -abs(float(delta_hit_drop)))
            or (entry_issue_ratio_delta is not None and float(entry_issue_ratio_delta) >= abs(float(delta_entry_rise)))
            or (avg_exit_regret_delta is not None and float(avg_exit_regret_delta) >= abs(float(delta_regret_rise)))
        )
    )
    result["delta_trigger"] = {
        "enabled": True,
        "sample_min": int(delta_sample_min),
        "batch_new_closed": int(batch_new_closed),
        "direction_hit_delta": direction_hit_delta,
        "entry_issue_ratio_delta": entry_issue_ratio_delta,
        "avg_exit_regret_delta": avg_exit_regret_delta,
        "direction_hit_drop_threshold": -abs(float(delta_hit_drop)),
        "entry_issue_rise_threshold": abs(float(delta_entry_rise)),
        "avg_exit_regret_rise_threshold": abs(float(delta_regret_rise)),
        "delta_bad": bool(delta_bad),
    }

    if direction_bad:
        actions.append("direction_quality_guard_up")
        cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_CONF", existing_overrides, 0.56)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_CONF", cur, +0.01, min_v=0.52, max_v=0.72)
        cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_EDGE", existing_overrides, 0.08)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_EDGE", cur, +0.01, min_v=0.05, max_v=0.16)
        cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", existing_overrides, 0.56)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", cur, +0.01, min_v=0.52, max_v=0.72)
        cur = _get_effective_float("ALPHA_DIRECTION_SCORE_MIN_SIDE_PROB", existing_overrides, 0.52)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_SCORE_MIN_SIDE_PROB", cur, +0.01, min_v=0.52, max_v=0.72)
        cur = _get_effective_float("POLICY_SMALL_GAP_CONFIDENCE", existing_overrides, 0.62)
        _set_float_step(out_overrides, "POLICY_SMALL_GAP_CONFIDENCE", cur, +0.02, min_v=0.55, max_v=0.78)
        cur = _get_effective_float("POLICY_SMALL_GAP_DIR_CONFIDENCE", existing_overrides, 0.58)
        _set_float_step(out_overrides, "POLICY_SMALL_GAP_DIR_CONFIDENCE", cur, +0.02, min_v=0.52, max_v=0.75)
        cur = _get_effective_float("POLICY_SMALL_GAP_DIR_EDGE", existing_overrides, 0.08)
        _set_float_step(out_overrides, "POLICY_SMALL_GAP_DIR_EDGE", cur, +0.01, min_v=0.05, max_v=0.15)
        cur = _get_effective_float("POLICY_SMALL_GAP_SIDE_PROB", existing_overrides, 0.56)
        _set_float_step(out_overrides, "POLICY_SMALL_GAP_SIDE_PROB", cur, +0.01, min_v=0.52, max_v=0.80)
        cur_trigger = _get_effective_int("ALPHA_TRAIN_TRIGGER_NEW_EXITS", existing_overrides, 200)
        _set_int_step(out_overrides, "ALPHA_TRAIN_TRIGGER_NEW_EXITS", cur_trigger, -20, min_v=120, max_v=400)
        out_overrides["ALPHA_DIRECTION_LABEL_SOURCE"] = "auto"
        cur = _get_effective_float("ALPHA_DIRECTION_LABEL_FEE_BPS", existing_overrides, 1.5)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_LABEL_FEE_BPS", cur, +0.5, min_v=0.0, max_v=8.0)
        cur = _get_effective_float("ALPHA_DIRECTION_LABEL_SLIPPAGE_BPS", existing_overrides, 1.0)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_LABEL_SLIPPAGE_BPS", cur, +0.5, min_v=0.0, max_v=10.0)
        cur = _get_effective_float("ALPHA_DIRECTION_NEUTRAL_BAND_BPS", existing_overrides, 1.0)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_NEUTRAL_BAND_BPS", cur, +0.5, min_v=0.0, max_v=8.0)
        out_overrides["ALPHA_DIRECTION_LABEL_EXCLUDE_TOKENS"] = "external,manual_cleanup,rebal,rebal_exit,cleanup,manual_close"
        out_overrides["ALPHA_DIRECTION_HSTAR_LABEL_ENABLED"] = 1
        out_overrides["ALPHA_DIRECTION_HSTAR_BARS"] = "1,2,3,5,8,13,21,34"
        out_overrides["ALPHA_DIRECTION_LGBM_TUNE_ENABLED"] = 1
        # Expectancy-first objective for direction model selection.
        out_overrides["ALPHA_DIRECTION_SCORE_EXP_WEIGHT"] = 1.0
        out_overrides["ALPHA_DIRECTION_SCORE_HIT_WEIGHT"] = 0.04
        out_overrides["ALPHA_DIRECTION_SCORE_AUC_WEIGHT"] = 0.02
        out_overrides["ALPHA_DIRECTION_SCORE_BALACC_WEIGHT"] = 0.02
        out_overrides["ALPHA_DIRECTION_SCORE_LOGLOSS_PENALTY"] = 0.04
        cur = _get_effective_float("ALPHA_DIRECTION_SCORE_COVERAGE_TARGET", existing_overrides, 0.35)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_SCORE_COVERAGE_TARGET", cur, +0.01, min_v=0.15, max_v=0.80)
        cur = _get_effective_float("ALPHA_DIRECTION_SCORE_COVERAGE_PENALTY_BPS", existing_overrides, 8.0)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_SCORE_COVERAGE_PENALTY_BPS", cur, +0.5, min_v=1.0, max_v=25.0)
        cur = _get_effective_float("ALPHA_DIRECTION_SCORE_MIN_EXPECTANCY_BPS", existing_overrides, -2.0)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_SCORE_MIN_EXPECTANCY_BPS", cur, +0.5, min_v=-8.0, max_v=8.0)
        # Respect explicit operator disable in env/override instead of force-enabling.
        sq_enabled = _is_true(existing_overrides.get("SYMBOL_QUALITY_FILTER_ENABLED", os.environ.get("SYMBOL_QUALITY_FILTER_ENABLED", "1")))
        sq_time_enabled = _is_true(existing_overrides.get("SYMBOL_QUALITY_TIME_FILTER_ENABLED", os.environ.get("SYMBOL_QUALITY_TIME_FILTER_ENABLED", "1")))
        if sq_enabled:
            out_overrides["SYMBOL_QUALITY_FILTER_ENABLED"] = 1
            out_overrides["SYMBOL_QUALITY_TIME_FILTER_ENABLED"] = 1 if sq_time_enabled else 0
            cur = _get_effective_int("SYMBOL_QUALITY_LOOKBACK_EXITS", existing_overrides, 320)
            _set_int_step(out_overrides, "SYMBOL_QUALITY_LOOKBACK_EXITS", cur, +40, min_v=120, max_v=600)
            cur = _get_effective_int("SYMBOL_QUALITY_MIN_EXITS", existing_overrides, 40)
            _set_int_step(out_overrides, "SYMBOL_QUALITY_MIN_EXITS", cur, +4, min_v=16, max_v=120)
            cur = _get_effective_float("SYMBOL_QUALITY_TIME_WEIGHT", existing_overrides, 0.30)
            _set_float_step(out_overrides, "SYMBOL_QUALITY_TIME_WEIGHT", cur, +0.03, min_v=0.10, max_v=0.70)
            cur = _get_effective_float("SYMBOL_QUALITY_MIN_SCORE", existing_overrides, 0.42)
            _set_float_step(out_overrides, "SYMBOL_QUALITY_MIN_SCORE", cur, +0.01, min_v=0.30, max_v=0.80)
        else:
            out_overrides["SYMBOL_QUALITY_FILTER_ENABLED"] = 0
            out_overrides["SYMBOL_QUALITY_TIME_FILTER_ENABLED"] = 0
        # Preserve high-quality entry opportunities when event entry filter is too strict.
        out_overrides["ENTRY_EVENT_EXIT_BYPASS_ENABLED"] = 1
        retrain_reason.append("entry_direction_quality")

    if delta_bad:
        actions.append("batch_delta_degrade_guard_up")
        cur = _get_effective_int("ALPHA_TRAIN_TRIGGER_NEW_EXITS", existing_overrides, 200)
        _set_int_step(out_overrides, "ALPHA_TRAIN_TRIGGER_NEW_EXITS", cur, -30, min_v=80, max_v=400)
        cur = _get_effective_int("ALPHA_DIRECTION_REGIME_MIN_SAMPLES", existing_overrides, 250)
        _set_int_step(out_overrides, "ALPHA_DIRECTION_REGIME_MIN_SAMPLES", cur, -30, min_v=120, max_v=500)
        cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_CONF", existing_overrides, 0.56)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_CONF", cur, +0.01, min_v=0.52, max_v=0.76)
        cur = _get_effective_float("ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", existing_overrides, 0.56)
        _set_float_step(out_overrides, "ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", cur, +0.01, min_v=0.52, max_v=0.76)
        cur = _get_effective_float("POLICY_SMALL_GAP_DIR_EDGE", existing_overrides, 0.08)
        _set_float_step(out_overrides, "POLICY_SMALL_GAP_DIR_EDGE", cur, +0.005, min_v=0.05, max_v=0.18)
        if avg_exit_regret_delta is not None and float(avg_exit_regret_delta) > 0.0:
            cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
            _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=10)
            cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", existing_overrides, 0.60)
            _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", cur, +0.02, min_v=0.45, max_v=0.99)
        out_overrides["ALPHA_DIRECTION_LGBM_TUNE_ENABLED"] = 1
        out_overrides["ALPHA_DIRECTION_HSTAR_LABEL_ENABLED"] = 1
        out_overrides["ALPHA_DIRECTION_LABEL_SOURCE"] = "auto"
        retrain_reason.append("batch_delta_degrade")

    if exit_bad:
        actions.append("exit_early_guard_up")
        cur = _get_effective_int("HYBRID_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "HYBRID_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=6)
        cur = _get_effective_int("HYBRID_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
        _set_int_step(out_overrides, "HYBRID_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=8)
        cur = _get_effective_int("UNIFIED_CASH_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "UNIFIED_CASH_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=6)
        cur = _get_effective_int("UNIFIED_CASH_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
        _set_int_step(out_overrides, "UNIFIED_CASH_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=8)
        cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=8)
        cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
        _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=10)
        cur = _get_effective_int("UNREALIZED_DD_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
        _set_int_step(out_overrides, "UNREALIZED_DD_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=8)
        cur = _get_effective_int("UNREALIZED_DD_CONFIRM_TICKS_NOISE", existing_overrides, 3)
        _set_int_step(out_overrides, "UNREALIZED_DD_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=10)
        cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_RANDOM", existing_overrides, -0.0003)
        _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_RANDOM", cur, -0.0002, min_v=-0.0035, max_v=0.005)
        cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_TREND", existing_overrides, -0.0012)
        _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_TREND", cur, -0.0002, min_v=-0.0040, max_v=0.005)
        cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_CHOP", existing_overrides, -0.0008)
        _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_CHOP", cur, -0.0002, min_v=-0.0040, max_v=0.005)
        cur = _get_effective_float("EVENT_EXIT_SCORE_OFFSET_VOLATILE", existing_overrides, -0.0003)
        _set_float_step(out_overrides, "EVENT_EXIT_SCORE_OFFSET_VOLATILE", cur, -0.0002, min_v=-0.0040, max_v=0.005)
        cur = _get_effective_float("EVENT_EXIT_SHOCK_FAST_THRESHOLD", existing_overrides, 1.0)
        _set_float_step(out_overrides, "EVENT_EXIT_SHOCK_FAST_THRESHOLD", cur, +0.10, min_v=0.8, max_v=2.2)
        cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_TREND", existing_overrides, 0.45)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_TREND", cur, +0.03, min_v=0.35, max_v=0.98)
        cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", existing_overrides, 0.60)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_RANDOM", cur, +0.03, min_v=0.45, max_v=0.99)
        cur = _get_effective_float("EVENT_MC_TSTAR_MIN_PROGRESS_MEAN_REVERT", existing_overrides, 0.70)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_MIN_PROGRESS_MEAN_REVERT", cur, +0.03, min_v=0.50, max_v=0.99)
        cur = _get_effective_float("EVENT_MC_TSTAR_BYPASS_SHOCK", existing_overrides, 1.20)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_BYPASS_SHOCK", cur, +0.05, min_v=0.8, max_v=3.0)
        cur = _get_effective_float("EVENT_MC_TSTAR_BYPASS_PSL", existing_overrides, 0.92)
        _set_float_step(out_overrides, "EVENT_MC_TSTAR_BYPASS_PSL", cur, +0.01, min_v=0.70, max_v=0.99)
        cur = _get_effective_float("EXIT_SCORE_DROP", existing_overrides, 0.0004)
        _set_float_step(out_overrides, "EXIT_SCORE_DROP", cur, +0.0001, min_v=0.0, max_v=0.005)
        cur = _get_effective_float("EXIT_RESPECT_ENTRY_MIN_PROGRESS", existing_overrides, 0.65)
        _set_float_step(out_overrides, "EXIT_RESPECT_ENTRY_MIN_PROGRESS", cur, +0.03, min_v=0.45, max_v=0.95)
        cur = _get_effective_float("EXIT_RESPECT_ENTRY_MIN_Q", existing_overrides, 0.72)
        _set_float_step(out_overrides, "EXIT_RESPECT_ENTRY_MIN_Q", cur, +0.01, min_v=0.55, max_v=0.90)
        cur = _get_effective_float("EXIT_RESPECT_ENTRY_MIN_SIGNAL", existing_overrides, 0.70)
        _set_float_step(out_overrides, "EXIT_RESPECT_ENTRY_MIN_SIGNAL", cur, +0.01, min_v=0.55, max_v=0.90)
        if event_reason_ratio >= 0.20:
            cur = _get_effective_int("EVENT_EXIT_CONFIRM_TICKS_NOISE", existing_overrides, 3)
            _set_int_step(out_overrides, "EVENT_EXIT_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=12)
        if unified_flip_ratio >= 0.20:
            cur = _get_effective_int("UNIFIED_CASH_EXIT_CONFIRM_TICKS_NORMAL", existing_overrides, 2)
            _set_int_step(out_overrides, "UNIFIED_CASH_EXIT_CONFIRM_TICKS_NORMAL", cur, +1, min_v=1, max_v=8)
        if ev_drop_ratio >= 0.20:
            cur = _get_effective_float("EXIT_SCORE_DROP", existing_overrides, 0.0004)
            _set_float_step(out_overrides, "EXIT_SCORE_DROP", cur, +0.0001, min_v=0.0, max_v=0.006)
        if dd_reason_ratio >= 0.15:
            cur = _get_effective_int("UNREALIZED_DD_CONFIRM_TICKS_NOISE", existing_overrides, 3)
            _set_int_step(out_overrides, "UNREALIZED_DD_CONFIRM_TICKS_NOISE", cur, +1, min_v=1, max_v=12)
            cur = _get_effective_float("UNREALIZED_DD_MULT_NOISE", existing_overrides, 1.12)
            _set_float_step(out_overrides, "UNREALIZED_DD_MULT_NOISE", cur, +0.03, min_v=1.0, max_v=2.0)
        retrain_reason.append("exit_regret_or_early_exit")

    if risk_bad:
        actions.append("tail_risk_guard_up")
        min_target_max_floor = _safe_float(os.environ.get("LEVERAGE_AUTOTUNE_MIN_TARGET_MAX", 25.0), 25.0) or 25.0
        min_dynamic_scale_floor = _safe_float(os.environ.get("LEVERAGE_AUTOTUNE_MIN_DYNAMIC_SCALE", 0.25), 0.25) or 0.25
        min_vpin_soft_floor = _safe_float(os.environ.get("LEVERAGE_AUTOTUNE_MIN_VPIN_SOFT", 0.80), 0.80) or 0.80
        min_signal_blend_floor = _safe_float(os.environ.get("LEVERAGE_AUTOTUNE_MIN_SIGNAL_BLEND", 0.60), 0.60) or 0.60
        max_signal_pow_cap = _safe_float(os.environ.get("LEVERAGE_AUTOTUNE_MAX_SIGNAL_POW", 1.40), 1.40) or 1.40
        min_target_max_floor = max(8.0, min(50.0, float(min_target_max_floor)))
        min_dynamic_scale_floor = max(0.10, min(0.60, float(min_dynamic_scale_floor)))
        min_vpin_soft_floor = max(0.52, min(0.95, float(min_vpin_soft_floor)))
        min_signal_blend_floor = max(0.20, min(0.90, float(min_signal_blend_floor)))
        max_signal_pow_cap = max(0.60, min(2.50, float(max_signal_pow_cap)))
        cur = _get_effective_float("LEVERAGE_TARGET_MAX", existing_overrides, 15.0)
        _set_float_step(out_overrides, "LEVERAGE_TARGET_MAX", cur, -1.0, min_v=min_target_max_floor, max_v=50.0)
        cur = _get_effective_float("LEVERAGE_DYNAMIC_MIN_SCALE", existing_overrides, 0.20)
        _set_float_step(out_overrides, "LEVERAGE_DYNAMIC_MIN_SCALE", cur, +0.02, min_v=min_dynamic_scale_floor, max_v=0.60)
        cur = _get_effective_float("LEVERAGE_VPIN_SOFT", existing_overrides, 0.65)
        _set_float_step(out_overrides, "LEVERAGE_VPIN_SOFT", cur, +0.01, min_v=min_vpin_soft_floor, max_v=0.95)
        cur = _get_effective_float("LEVERAGE_SIGNAL_BLEND", existing_overrides, 0.55)
        _set_float_step(out_overrides, "LEVERAGE_SIGNAL_BLEND", cur, +0.03, min_v=min_signal_blend_floor, max_v=0.90)
        cur = _get_effective_float("LEVERAGE_SIGNAL_SCORE_POW", existing_overrides, 1.20)
        _set_float_step(out_overrides, "LEVERAGE_SIGNAL_SCORE_POW", cur, -0.05, min_v=0.50, max_v=max_signal_pow_cap)
        cur = _get_effective_float("UNREALIZED_DD_SEVERE_MULT", existing_overrides, 1.40)
        _set_float_step(out_overrides, "UNREALIZED_DD_SEVERE_MULT", cur, -0.05, min_v=1.10, max_v=2.00)
        cur = _get_effective_float("LEVERAGE_LOW_CONF_CAP", existing_overrides, 3.0)
        _set_float_step(out_overrides, "LEVERAGE_LOW_CONF_CAP", cur, -0.25, min_v=1.0, max_v=8.0)
        cur = _get_effective_float("LEVERAGE_HIGH_VPIN_CAP", existing_overrides, 2.0)
        _set_float_step(out_overrides, "LEVERAGE_HIGH_VPIN_CAP", cur, -0.25, min_v=1.0, max_v=6.0)
        retrain_reason.append("tail_risk_control")
    elif direction_good:
        actions.append("quality_leverage_expand")
        cur = _get_effective_float("LEVERAGE_TARGET_MAX", existing_overrides, 50.0)
        _set_float_step(out_overrides, "LEVERAGE_TARGET_MAX", cur, +1.0, min_v=8.0, max_v=50.0)
        cur = _get_effective_float("LEVERAGE_SIGNAL_BLEND", existing_overrides, 0.55)
        _set_float_step(out_overrides, "LEVERAGE_SIGNAL_BLEND", cur, +0.03, min_v=0.20, max_v=0.90)
        cur = _get_effective_float("LEVERAGE_SIGNAL_SCORE_POW", existing_overrides, 1.20)
        _set_float_step(out_overrides, "LEVERAGE_SIGNAL_SCORE_POW", cur, -0.05, min_v=0.50, max_v=2.50)
        cur = _get_effective_float("LEVERAGE_LOW_CONF_CAP", existing_overrides, 3.0)
        _set_float_step(out_overrides, "LEVERAGE_LOW_CONF_CAP", cur, +0.25, min_v=1.0, max_v=10.0)
        cur = _get_effective_float("LEVERAGE_HIGH_VPIN_CAP", existing_overrides, 2.0)
        _set_float_step(out_overrides, "LEVERAGE_HIGH_VPIN_CAP", cur, +0.25, min_v=1.0, max_v=8.0)

    # Regime-aware secondary tuning loop.
    # - direction quality -> tighten small-gap gates by regime
    # - early exits/regret -> relax regime-specific EVENT_EXIT score thresholds
    # - liquidation/drawdown pressure -> tighten UNREALIZED_DD cut by regime
    min_regime_samples = int(max(1, args.min_regime_samples))
    for reg in ("trend", "chop", "volatile"):
        rm = regime_metrics.get(reg) if isinstance(regime_metrics, dict) else {}
        if not isinstance(rm, dict):
            continue
        n = int(rm.get("exit_n", 0) or 0)
        if n < min_regime_samples:
            continue
        loss_rate_reg = _safe_float(rm.get("loss_rate"), None)
        event_rate_reg = _safe_float(rm.get("event_rate"), None)
        dd_rate_reg = _safe_float(rm.get("dd_rate"), None)
        liq_rate_reg = _safe_float(rm.get("liq_rate"), None)
        avg_roe_reg = _safe_float(rm.get("avg_roe"), None)
        reject_rate_reg = _safe_float(rm.get("reject_rate"), None)
        reject_110007_rate_reg = _safe_float(rm.get("reject_110007_rate"), None)
        precision_reject_rate_reg = _safe_float(rm.get("precision_reject_rate"), None)
        reg_key = str(reg).upper()

        if direction_bad and loss_rate_reg is not None and loss_rate_reg >= 0.52:
            actions.append(f"regime_direction_guard:{reg}")
            for key, step, lo, hi, dflt in (
                (f"POLICY_SMALL_GAP_CONFIDENCE_{reg_key}", +0.01, 0.55, 0.84, 0.62),
                (f"POLICY_SMALL_GAP_DIR_CONFIDENCE_{reg_key}", +0.01, 0.52, 0.80, 0.58),
                (f"POLICY_SMALL_GAP_DIR_EDGE_{reg_key}", +0.005, 0.05, 0.20, 0.08),
                (f"POLICY_SMALL_GAP_SIDE_PROB_{reg_key}", +0.01, 0.50, 0.80, 0.56),
            ):
                cur_r = _get_effective_float(key, existing_overrides, dflt)
                _set_float_step(out_overrides, key, cur_r, step, min_v=lo, max_v=hi)

        if event_rate_reg is not None and event_rate_reg >= 0.30:
            ev_key = f"EVENT_EXIT_SCORE_OFFSET_{reg_key}"
            default_off = -0.0008 if reg == "chop" else (-0.0012 if reg == "trend" else -0.0003)
            cur_off = _get_effective_float(ev_key, existing_overrides, default_off)
            if avg_roe_reg is not None and avg_roe_reg < -0.002 and loss_rate_reg is not None and loss_rate_reg >= 0.55:
                actions.append(f"regime_exit_tighten:{reg}")
                _set_float_step(out_overrides, ev_key, cur_off, +0.0002, min_v=-0.0050, max_v=0.0050)
            elif exit_bad:
                actions.append(f"regime_exit_guard:{reg}")
                _set_float_step(out_overrides, ev_key, cur_off, -0.0002, min_v=-0.0050, max_v=0.0050)

        # Regime-specific entry net expectancy floor tuning:
        # negative regime expectancy -> tighten floor, stable positive regime -> relax floor.
        if avg_roe_reg is not None and (avg_roe_reg < -0.0015 or (loss_rate_reg is not None and loss_rate_reg >= 0.58)):
            actions.append(f"regime_net_expectancy_guard:{reg}")
            for key, step, lo, hi, dflt in (
                (f"ENTRY_NET_EXPECTANCY_MIN_{reg_key}", +0.00015, -0.0030, 0.0050, 0.0),
                (f"ENTRY_NET_EXPECTANCY_VPIN_BUMP_{reg_key}", +0.00008, 0.0, 0.0030, 0.0004),
                (f"ENTRY_NET_EXPECTANCY_DIR_CONF_BUMP_{reg_key}", +0.00008, 0.0, 0.0030, 0.0004),
                (f"ENTRY_NET_EXPECTANCY_DIR_CONF_REF_{reg_key}", +0.0100, 0.50, 0.86, 0.58),
            ):
                cur_r = _get_effective_float(key, existing_overrides, dflt)
                _set_float_step(out_overrides, key, cur_r, step, min_v=lo, max_v=hi)
        elif (
            avg_roe_reg is not None
            and avg_roe_reg > 0.002
            and loss_rate_reg is not None
            and loss_rate_reg <= 0.45
            and not direction_bad
        ):
            actions.append(f"regime_net_expectancy_relax:{reg}")
            for key, step, lo, hi, dflt in (
                (f"ENTRY_NET_EXPECTANCY_MIN_{reg_key}", -0.00008, -0.0030, 0.0050, 0.0),
                (f"ENTRY_NET_EXPECTANCY_VPIN_BUMP_{reg_key}", -0.00005, 0.0, 0.0030, 0.0004),
                (f"ENTRY_NET_EXPECTANCY_DIR_CONF_BUMP_{reg_key}", -0.00005, 0.0, 0.0030, 0.0004),
            ):
                cur_r = _get_effective_float(key, existing_overrides, dflt)
                _set_float_step(out_overrides, key, cur_r, step, min_v=lo, max_v=hi)

        # Symbol/time quality guard by reject pressure.
        if reject_rate_reg is not None and reject_rate_reg >= 0.35:
            actions.append(f"regime_reject_guard:{reg}")
            cur = _get_effective_float("SYMBOL_QUALITY_REJECT_PENALTY", existing_overrides, 0.18)
            _set_float_step(out_overrides, "SYMBOL_QUALITY_REJECT_PENALTY", cur, +0.02, min_v=0.05, max_v=0.60)
            cur = _get_effective_float("SYMBOL_QUALITY_REJECT_HARD", existing_overrides, 0.55)
            _set_float_step(out_overrides, "SYMBOL_QUALITY_REJECT_HARD", cur, -0.02, min_v=0.20, max_v=0.90)
            cur = _get_effective_float("SYMBOL_QUALITY_MIN_SCORE", existing_overrides, 0.42)
            _set_float_step(out_overrides, "SYMBOL_QUALITY_MIN_SCORE", cur, +0.01, min_v=0.30, max_v=0.85)
        if reject_110007_rate_reg is not None and reject_110007_rate_reg >= 0.30:
            actions.append(f"regime_balance_reject_guard:{reg}")
            cur = _get_effective_float("LEVERAGE_TARGET_MAX", existing_overrides, 15.0)
            _set_float_step(out_overrides, "LEVERAGE_TARGET_MAX", cur, -1.0, min_v=8.0, max_v=50.0)
            cur = _get_effective_float("LEVERAGE_BALANCE_REJECT_SCALE", existing_overrides, 0.82)
            _set_float_step(out_overrides, "LEVERAGE_BALANCE_REJECT_SCALE", cur, -0.03, min_v=0.35, max_v=0.95)
            cur = _get_effective_float("LEVERAGE_BALANCE_REJECT_MIN_SCALE", existing_overrides, 0.45)
            _set_float_step(out_overrides, "LEVERAGE_BALANCE_REJECT_MIN_SCALE", cur, -0.03, min_v=0.20, max_v=0.90)
        if precision_reject_rate_reg is not None and precision_reject_rate_reg >= 0.15:
            actions.append(f"regime_precision_reject_guard:{reg}")
            cur = _get_effective_float("LIVE_MIN_QTY_MAX_UPSIZE", existing_overrides, 5.0)
            _set_float_step(out_overrides, "LIVE_MIN_QTY_MAX_UPSIZE", cur, +0.5, min_v=1.0, max_v=20.0)
            key = f"LIVE_MIN_QTY_MAX_UPSIZE_{reg_key}"
            cur = _get_effective_float(key, existing_overrides, 5.0)
            _set_float_step(out_overrides, key, cur, +0.5, min_v=1.0, max_v=20.0)

        # Tighten DD floor only when tail risk is actually elevated in that regime.
        if (risk_bad or (liq_rate_reg is not None and liq_rate_reg >= 0.02)) and (
            (dd_rate_reg is not None and dd_rate_reg >= 0.08) or (liq_rate_reg is not None and liq_rate_reg >= 0.02)
        ):
            actions.append(f"regime_dd_guard:{reg}")
            dd_key = f"UNREALIZED_DD_REGIME_MULT_{reg_key}"
            cur_dd = _get_effective_float(dd_key, existing_overrides, 1.0)
            _set_float_step(out_overrides, dd_key, cur_dd, -0.05, min_v=0.60, max_v=1.40)
        elif (
            exit_bad
            and avg_exit_regret is not None
            and avg_exit_regret >= float(args.high_exit_regret)
            and dd_rate_reg is not None
            and dd_rate_reg >= 0.10
            and avg_roe_reg is not None
            and avg_roe_reg > 0.0
        ):
            # If DD exits are frequent despite positive regime ROE, slightly loosen to reduce premature cuts.
            actions.append(f"regime_dd_relax:{reg}")
            dd_key = f"UNREALIZED_DD_REGIME_MULT_{reg_key}"
            cur_dd = _get_effective_float(dd_key, existing_overrides, 1.0)
            _set_float_step(out_overrides, dd_key, cur_dd, +0.03, min_v=0.60, max_v=1.40)

    if not actions:
        actions.append("no_change:metrics_within_thresholds")

    payload_overrides: dict[str, Any] = {}
    if out_overrides:
        payload_overrides = {
            k: _sanitize_override_value(v)
            for k, v in out_overrides.items()
            if _sanitize_override_value(v) is not None
        }

    # Recovery mode: allow entry-gate relaxation only.
    # Tightening (higher value) of UNIFIED_ENTRY_FLOOR / ENTRY_NET_EXPECTANCY_* is blocked.
    recovery_relax_only = bool(
        _is_true(os.environ.get("AUTO_TUNE_RECOVERY_ENTRY_RELAX_ONLY", "0"))
        or _is_true(existing_overrides.get("AUTO_TUNE_RECOVERY_ENTRY_RELAX_ONLY"))
    )
    recovery_blocked: list[str] = []
    if recovery_relax_only and payload_overrides:
        guarded_exact = {"UNIFIED_ENTRY_FLOOR"}
        for key in list(payload_overrides.keys()):
            if (key not in guarded_exact) and (not str(key).startswith("ENTRY_NET_EXPECTANCY_")):
                continue
            nxt = _safe_float(payload_overrides.get(key), None)
            if nxt is None:
                continue
            cur = _get_effective_float(key, existing_overrides, float(nxt))
            # For these keys, larger value means stricter entry gate.
            if float(nxt) > float(cur) + 1e-12:
                payload_overrides.pop(key, None)
                recovery_blocked.append(str(key))
        if recovery_blocked:
            actions.append(f"recovery_relax_only_blocked:{len(recovery_blocked)}")
    result["recovery_relax_only"] = {
        "enabled": bool(recovery_relax_only),
        "blocked_count": int(len(recovery_blocked)),
        "blocked_keys": sorted(set(recovery_blocked)),
    }

    retrain_requested = bool(direction_bad or exit_bad or delta_bad)
    retrain_info = {
        "requested": bool(retrain_requested),
        "reason": ",".join(sorted(set(retrain_reason))) if retrain_reason else "",
        "ran": False,
        "rc": None,
        "elapsed_sec": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if retrain_requested and int(args.run_retrain) == 1 and not args.dry_run:
        r = _run_retrain(repo_root=repo_root, timeout_sec=float(args.retrain_timeout_sec))
        retrain_info.update(r)
        actions.append(f"retrain:rc={retrain_info.get('rc')}")

    override_payload = {
        "timestamp_ms": _now_ms(),
        "source": "auto_reval_policy_tuner",
        "batch_id": int(batch_id),
        "actions": list(actions),
        "metrics": metrics,
        "regime_metrics": regime_metrics,
        "reason_breakdown": {
            "event_reason_ratio": float(event_reason_ratio),
            "unified_flip_ratio": float(unified_flip_ratio),
            "ev_drop_ratio": float(ev_drop_ratio),
            "dd_reason_ratio": float(dd_reason_ratio),
            "reason_counts": dict(reason_counts),
            "reason_total": int(total_reason_n),
        },
        "overrides": {k: _sanitize_override_value(v) for k, v in payload_overrides.items()},
        "retrain": {
            "requested": bool(retrain_info.get("requested")),
            "ran": bool(retrain_info.get("ran")),
            "rc": retrain_info.get("rc"),
            "elapsed_sec": retrain_info.get("elapsed_sec"),
            "reason": retrain_info.get("reason"),
        },
    }

    if not args.dry_run:
        _write_json(override_out_path, override_payload)
        _write_json(
            state_file_path,
            {
                "timestamp_ms": _now_ms(),
                "last_batch_id": int(batch_id),
                "last_actions": list(actions),
                "last_retrain_rc": retrain_info.get("rc"),
                "last_retrain_ran": bool(retrain_info.get("ran")),
                "last_metrics": {
                    "direction_hit": _safe_float(metrics.get("direction_hit"), None),
                    "entry_issue_ratio": _safe_float(metrics.get("entry_issue_ratio"), None),
                    "avg_exit_regret": _safe_float(metrics.get("avg_exit_regret"), None),
                },
            },
        )

    result["actions"] = list(actions)
    result["overrides"] = dict(payload_overrides)
    result["override_count"] = int(len(payload_overrides))
    result["retrain"] = dict(retrain_info)
    result["applied"] = bool((not args.dry_run) and bool(payload_overrides))

    _write_json(status_out_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

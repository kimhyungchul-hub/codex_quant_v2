"""
research/runner.py — Main Research Engine Runner (v2)
======================================================
5분마다 CF 시뮬레이션 → 조건 충족 시 자동 적용 (보수적 모드)
1시간마다 Gemini 코드 리뷰 → git push

Usage:
  python -m research.runner              # 기본 (5분 주기 + auto-apply)
  python -m research.runner --once       # 1회만 실행
  python -m research.runner --stage leverage  # 특정 스테이지만
  python -m research.runner --no-dashboard    # 대시보드 없이
  python -m research.runner --no-auto-apply   # 자동 적용 비활성화
  python -m research.runner --no-gemini       # Gemini 리뷰 비활성화
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.cf_engine import (
    CFEngine,
    TradeLoader,
    ALL_SIMULATORS,
    compute_metrics_by_regime,
    DEFAULT_DL_REPORT,
    DEFAULT_DL_SCORES,
    DEFAULT_DL_MODEL,
)
from research.documenter import (
    save_findings_json,
    generate_findings_markdown,
    generate_changelog_entry,
)
from utils.singleton_process_lock import SingletonProcessLock

logger = logging.getLogger("research.runner")

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
DEFAULT_DB = os.path.join(PROJECT_ROOT, "state", "bot_data_live.db")
CF_INTERVAL_SEC = 300          # 5분마다 CF 분석
GEMINI_INTERVAL_SEC = 3600     # 1시간마다 Gemini 리뷰
MAX_COMBOS_PER_STAGE = 220
DASHBOARD_PORT = 9998
FINDINGS_OUTPUT = os.path.join(PROJECT_ROOT, "docs", "RESEARCH_FINDINGS.md")
FINDINGS_JSON = os.path.join(PROJECT_ROOT, "state", "research_findings.json")
RESEARCH_CYCLES_JSON = os.path.join(PROJECT_ROOT, "state", "research_cycles.json")
LAST_ANALYZED_JSON = os.path.join(PROJECT_ROOT, "state", "research_last_analyzed.json")
MAX_CYCLE_HISTORY = 80
MIN_NEW_TRADES_FOR_REANALYSIS = 3  # 최소 3개의 새 거래가 있어야 재분석
MTF_STAGE_NAME = "mtf_image_dl_gate"


def _env_bool(name: str, default: bool = False) -> bool:
    try:
        v = os.environ.get(name)
        if v is None:
            return bool(default)
        return str(v).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return bool(default)


def _file_env_bool(name: str, default: bool = False) -> bool:
    """Read boolean env from state/bybit.env when process env is not exported."""
    try:
        env_path = Path(PROJECT_ROOT) / "state" / "bybit.env"
        if not env_path.exists():
            return bool(default)
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if (not s) or s.startswith("#") or ("=" not in s):
                continue
            k, v = s.split("=", 1)
            if str(k).strip() == name:
                return str(v).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return bool(default)
    return bool(default)


def _file_env_float(name: str, default: float = 0.0) -> float:
    """Read float env from state/bybit.env when process env is not exported."""
    try:
        env_path = Path(PROJECT_ROOT) / "state" / "bybit.env"
        if not env_path.exists():
            return float(default)
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if (not s) or s.startswith("#") or ("=" not in s):
                continue
            k, v = s.split("=", 1)
            if str(k).strip() == name:
                try:
                    return float(str(v).strip())
                except Exception:
                    return float(default)
    except Exception:
        return float(default)
    return float(default)


def _file_env_int(name: str, default: int = 0) -> int:
    try:
        return int(round(float(_file_env_float(name, float(default)))))
    except Exception:
        return int(default)


def _runtime_interval_sec(default_sec: int, *, deep_mode: bool = False) -> int:
    """
    Resolve cycle interval from env (hot-reload friendly).

    Priority:
    - RESEARCH_CF_INTERVAL_SEC_DEEP (when deep_mode)
    - RESEARCH_CF_INTERVAL_SEC
    - CF_INTERVAL_SEC
    - CLI/default
    """
    try:
        if deep_mode:
            v = _file_env_int("RESEARCH_CF_INTERVAL_SEC_DEEP", 0)
            if v > 0:
                return int(max(60, min(v, 24 * 3600)))
        v = _file_env_int("RESEARCH_CF_INTERVAL_SEC", 0)
        if v <= 0:
            v = _file_env_int("CF_INTERVAL_SEC", int(default_sec))
        return int(max(60, min(int(v), 24 * 3600)))
    except Exception:
        return int(max(60, int(default_sec)))


def _load_cycle_history() -> list[dict]:
    p = Path(RESEARCH_CYCLES_JSON)
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def _save_cycle_history(rows: list[dict]) -> None:
    p = Path(RESEARCH_CYCLES_JSON)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows[-MAX_CYCLE_HISTORY:], ensure_ascii=False, indent=2), encoding="utf-8")


def _load_last_analyzed() -> dict:
    """Load last analyzed trade info (timestamp, trade count, hash)."""
    p = Path(LAST_ANALYZED_JSON)
    if not p.exists():
        return {"last_trade_ts": 0, "last_trade_count": 0, "last_pnl_hash": ""}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_trade_ts": 0, "last_trade_count": 0, "last_pnl_hash": ""}


def _save_last_analyzed(data: dict) -> None:
    """Save last analyzed trade info."""
    p = Path(LAST_ANALYZED_JSON)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_trades_hash(trades: list) -> str:
    """Compute a hash of trade PnLs to detect if data changed."""
    import hashlib
    if not trades:
        return ""
    pnl_str = ",".join([f"{getattr(t, 'realized_pnl', 0):.6f}" for t in trades[-50:]])
    return hashlib.md5(pnl_str.encode()).hexdigest()[:12]


def _get_latest_equity() -> float | None:
    try:
        conn = sqlite3.connect(DEFAULT_DB)
        row = conn.execute("SELECT total_equity FROM equity_history ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        return None
    return None


def _latest_apply_lookup(apply_history: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rec in (apply_history or []):
        if not isinstance(rec, dict):
            continue
        fid = str(rec.get("finding_id") or "").strip()
        if not fid:
            continue
        prev = out.get(fid)
        if (prev is None) or (float(rec.get("timestamp") or 0.0) >= float(prev.get("timestamp") or 0.0)):
            out[fid] = rec
    return out


def _resolve_path(path_like: str | os.PathLike | None, fallback: str | os.PathLike) -> Path:
    raw = str(path_like or "").strip()
    p = Path(raw) if raw else Path(fallback)
    if not p.is_absolute():
        p = Path(PROJECT_ROOT) / p
    return p


def _path_mtime(path: Path) -> float | None:
    try:
        if path.exists():
            return float(path.stat().st_mtime)
    except Exception:
        return None
    return None


def _path_size(path: Path) -> int | None:
    try:
        if path.exists():
            return int(path.stat().st_size)
    except Exception:
        return None
    return None


def _read_json(path: Path) -> dict | list | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_mtf_score_stats(
    scores_path: Path,
    *,
    side_threshold: float,
    win_threshold: float,
) -> dict:
    out = {
        "n_scores": 0,
        "prob_min": None,
        "prob_max": None,
        "prob_mean": None,
        "n_ge_side_threshold": 0,
        "n_ge_win_threshold": 0,
        "ratio_ge_side_threshold": 0.0,
        "ratio_ge_win_threshold": 0.0,
        "entry_ts_min_ms": None,
        "entry_ts_max_ms": None,
        "entry_span_hours": None,
    }
    rows = _read_json(scores_path)
    if not isinstance(rows, list):
        return out
    probs: list[float] = []
    entry_ts_vals: list[int] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            p = float(r.get("mtf_dl_prob"))
            if p == p:  # NaN-safe
                probs.append(p)
        except Exception:
            pass
        try:
            ts = int(float(r.get("entry_ts_ms") or 0))
            if ts > 0:
                entry_ts_vals.append(ts)
        except Exception:
            pass
    n = len(probs)
    if n <= 0:
        return out
    p_min = min(probs)
    p_max = max(probs)
    p_mean = sum(probs) / float(max(1, n))
    n_ge_side = sum(1 for p in probs if p >= float(side_threshold))
    n_ge_win = sum(1 for p in probs if p >= float(win_threshold))
    out.update(
        {
            "n_scores": int(n),
            "prob_min": float(p_min),
            "prob_max": float(p_max),
            "prob_mean": float(p_mean),
            "n_ge_side_threshold": int(n_ge_side),
            "n_ge_win_threshold": int(n_ge_win),
            "ratio_ge_side_threshold": float(n_ge_side / float(max(1, n))),
            "ratio_ge_win_threshold": float(n_ge_win / float(max(1, n))),
        }
    )
    if entry_ts_vals:
        ts_min = int(min(entry_ts_vals))
        ts_max = int(max(entry_ts_vals))
        span_h = float(max(0.0, (ts_max - ts_min) / 3_600_000.0))
        out.update(
            {
                "entry_ts_min_ms": ts_min,
                "entry_ts_max_ms": ts_max,
                "entry_span_hours": span_h,
            }
        )
    return out


def _collect_mtf_imageh_state(
    *,
    result: dict,
    findings: list[dict],
    apply_history: list[dict],
) -> dict:
    report_path = _resolve_path(os.environ.get("CF_MTF_DL_REPORT_PATH"), DEFAULT_DL_REPORT)
    scores_path = _resolve_path(os.environ.get("CF_MTF_DL_SCORES_PATH"), DEFAULT_DL_SCORES)
    model_path = _resolve_path(os.environ.get("CF_MTF_DL_MODEL_PATH"), DEFAULT_DL_MODEL)

    report_raw = _read_json(report_path)
    report = report_raw if isinstance(report_raw, dict) else {}
    training = report.get("training") if isinstance(report.get("training"), dict) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    best = report.get("best") if isinstance(report.get("best"), dict) else {}
    baseline = report.get("baseline") if isinstance(report.get("baseline"), dict) else {}
    sweep = report.get("threshold_sweep") if isinstance(report.get("threshold_sweep"), dict) else {}
    input_cfg = report.get("input") if isinstance(report.get("input"), dict) else {}
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}

    stage_prog = {}
    try:
        stage_prog = dict((result.get("sweep_progress") or {}).get(MTF_STAGE_NAME) or {})
    except Exception:
        stage_prog = {}

    mtf_findings = []
    for f in findings or []:
        if not isinstance(f, dict):
            continue
        if str(f.get("stage") or "").strip() == MTF_STAGE_NAME:
            mtf_findings.append({
                "finding_id": str(f.get("finding_id") or ""),
                "title": f.get("title"),
                "confidence": float(f.get("confidence") or 0.0),
                "improvement_pct": float(f.get("improvement_pct") or 0.0),
                "recommendation": f.get("recommendation"),
                "param_changes": f.get("param_changes") or {},
            })

    mtf_apply_hist = []
    for rec in apply_history or []:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("stage") or "").strip() == MTF_STAGE_NAME:
            mtf_apply_hist.append({
                "timestamp": float(rec.get("timestamp") or 0.0),
                "finding_id": str(rec.get("finding_id") or ""),
                "status": str(rec.get("status") or ""),
                "equity_at_apply": rec.get("equity_at_apply"),
                "equity_after_monitor": rec.get("equity_after_monitor"),
                "rolled_back": bool(rec.get("rolled_back")),
                "rollback_reason": rec.get("rollback_reason"),
                "env_changes": rec.get("env_changes") or {},
            })
    mtf_apply_hist = sorted(mtf_apply_hist, key=lambda x: float(x.get("timestamp") or 0.0), reverse=True)

    runtime_env = {
        "CF_MTF_DL_ENABLED": _env_bool("CF_MTF_DL_ENABLED", True),
        "CF_MTF_DL_REFRESH_EACH_CYCLE": _env_bool("CF_MTF_DL_REFRESH_EACH_CYCLE", True),
        "MTF_DL_RUNTIME_ENABLED": _env_bool("MTF_DL_RUNTIME_ENABLED", True),
        "MTF_DL_ENTRY_GATE_ENABLED": _env_bool("MTF_DL_ENTRY_GATE_ENABLED", True),
        "MTF_DL_EXIT_HEAD_ENABLED": _env_bool("MTF_DL_EXIT_HEAD_ENABLED", True),
        "MTF_DL_ENTRY_MIN_SIDE_PROB": _file_env_float("MTF_DL_ENTRY_MIN_SIDE_PROB", 0.58),
        "MTF_DL_ENTRY_MIN_WIN_PROB": _file_env_float("MTF_DL_ENTRY_MIN_WIN_PROB", 0.52),
        "MTF_DL_EXIT_MIN_CONF": _file_env_float("MTF_DL_EXIT_MIN_CONF", 0.62),
        "MTF_DL_EXIT_MIN_PROGRESS": _file_env_float("MTF_DL_EXIT_MIN_PROGRESS", 0.95),
    }
    score_stats = _collect_mtf_score_stats(
        scores_path,
        side_threshold=float(runtime_env.get("MTF_DL_ENTRY_MIN_SIDE_PROB") or 0.58),
        win_threshold=float(runtime_env.get("MTF_DL_ENTRY_MIN_WIN_PROB") or 0.52),
    )

    data_window = {
        "entry_ts_min_ms": score_stats.get("entry_ts_min_ms"),
        "entry_ts_max_ms": score_stats.get("entry_ts_max_ms"),
        "entry_span_hours": score_stats.get("entry_span_hours"),
        "cf_eval_start_ts_ms": int(dataset.get("cf_eval_start_ts_ms") or 0) if dataset else 0,
    }

    return {
        "stage_name": MTF_STAGE_NAME,
        "stage_progress": stage_prog,
        "running": bool(result.get("running")),
        "cycle_count": int(result.get("cycle_count") or 0),
        "report": {
            "path": str(report_path),
            "mtime": _path_mtime(report_path),
            "timestamp": int(report.get("timestamp") or 0) if report else 0,
        },
        "scores": {
            "path": str(scores_path),
            "mtime": _path_mtime(scores_path),
            "size_bytes": _path_size(scores_path),
        },
        "model": {
            "path": str(model_path),
            "mtime": _path_mtime(model_path),
            "size_bytes": _path_size(model_path),
        },
        "dataset": dataset,
        "input": input_cfg,
        "artifacts": artifacts,
        "data_window": data_window,
        "score_stats": score_stats,
        "training": training,
        "baseline": baseline,
        "best": best,
        "sweep": sweep,
        "runtime_env": runtime_env,
        "mtf_findings": mtf_findings[:12],
        "mtf_apply_history": mtf_apply_hist[:30],
    }


def _build_cycle_report(
    *,
    cycle_index: int,
    cycle_started_ts: float,
    result: dict,
    findings: list[dict],
    apply_history: list[dict],
    apply_status: dict | None,
    stage_filter: str | None,
) -> dict:
    apply_lookup = _latest_apply_lookup(apply_history)
    latest_equity = _get_latest_equity()
    suggestion_rows: list[dict] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("finding_id") or "")
        rec = apply_lookup.get(fid)
        applied = rec is not None
        apply_state = str((rec or {}).get("status") or "not_applied")
        eq_at_apply = (rec or {}).get("equity_at_apply")
        eq_after = (rec or {}).get("equity_after_monitor")
        effect_delta_final = None
        effect_delta_live = None
        try:
            if eq_at_apply is not None and eq_after is not None:
                effect_delta_final = float(eq_after) - float(eq_at_apply)
        except Exception:
            effect_delta_final = None
        try:
            if apply_state == "monitoring" and eq_at_apply is not None and latest_equity is not None:
                effect_delta_live = float(latest_equity) - float(eq_at_apply)
        except Exception:
            effect_delta_live = None

        suggestion_rows.append({
            "finding_id": fid,
            "stage": f.get("stage"),
            "title": f.get("title"),
            "confidence": float(f.get("confidence") or 0.0),
            "improvement_pct": float(f.get("improvement_pct") or 0.0),
            "recommendation": f.get("recommendation"),
            "param_changes": f.get("param_changes") or {},
            "applied": bool(applied),
            "apply_state": apply_state,
            "effect_delta_equity_final": effect_delta_final,
            "effect_delta_equity_live": effect_delta_live,
        })

    applied_n = sum(1 for s in suggestion_rows if s.get("applied"))
    monitoring_n = sum(1 for s in suggestion_rows if s.get("apply_state") == "monitoring")
    rolled_n = sum(1 for s in suggestion_rows if s.get("apply_state") == "rolled_back")
    finalized = [s.get("effect_delta_equity_final") for s in suggestion_rows if s.get("effect_delta_equity_final") is not None]
    avg_effect = (sum(finalized) / len(finalized)) if finalized else None

    return {
        "cycle_index": int(cycle_index),
        "started_ts": float(cycle_started_ts),
        "completed_ts": float(result.get("last_update_ts") or time.time()),
        "status": str(result.get("status") or "unknown"),
        "stage_filter": stage_filter,
        "research_mode": str(result.get("research_mode") or "normal"),
        "ensemble_runs": int(result.get("ensemble_runs") or 1),
        "cycle_max_combos": int(result.get("cycle_max_combos") or 0),
        "elapsed_sec": float(result.get("elapsed_sec") or 0.0),
        "n_trades": int(result.get("n_trades") or 0),
        "n_findings": int(result.get("n_findings") or 0),
        "applied_n": int(applied_n),
        "monitoring_n": int(monitoring_n),
        "rolled_back_n": int(rolled_n),
        "avg_effect_delta_equity": avg_effect,
        "auto_apply": apply_status or {},
        "suggestions": suggestion_rows,
    }


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def run_cf_cycle(
    db_path: str = DEFAULT_DB,
    stage_filter: str | None = None,
    max_combos: int = MAX_COMBOS_PER_STAGE,
) -> dict:
    """Run one complete CF analysis cycle and return results."""
    t0 = time.perf_counter()
    cycle_index = int(getattr(run_cf_cycle, "_cycle", 0)) + 1
    cycle_started_ts = time.time()

    _update_dashboard({
        "running": True,
        "current_cycle_index": cycle_index,
        "completed_cycles_total": max(0, cycle_index - 1),
        "stage_current": "init",
        "cycle_started_ts": cycle_started_ts,
    })

    loader = TradeLoader(db_path)
    trades = loader.load_trades()
    if not trades:
        logger.warning("No trades found. Skipping CF cycle.")
        run_cf_cycle._cycle = cycle_index
        mtf_state = _collect_mtf_imageh_state(
            result={"running": False, "cycle_count": cycle_index, "sweep_progress": {}},
            findings=[],
            apply_history=[],
        )
        return {
            "status": "no_trades",
            "cycle_count": cycle_index,
            "cycle_started_ts": cycle_started_ts,
            "last_update_ts": time.time(),
            "running": False,
            "research_mode": "normal",
            "deep_mode": False,
            "cycle_max_combos": int(max_combos),
            "ensemble_runs": 1,
            "mtf_imageh": mtf_state,
        }

    # ── Check for new trades to avoid repeated analysis ──
    # Prefer bybit.env for live tuning (file-first), then process env fallback.
    always_run_cf = _file_env_bool("ALWAYS_RUN_CF", _env_bool("ALWAYS_RUN_CF", False))
    try:
        idle_force_sec = float(
            os.environ.get(
                "CF_FORCE_RUN_MAX_IDLE_SEC",
                _file_env_float("CF_FORCE_RUN_MAX_IDLE_SEC", 1800.0),
            ) or 1800.0
        )
    except Exception:
        idle_force_sec = 1800.0
    idle_force_sec = max(0.0, float(idle_force_sec))
    last_analyzed = _load_last_analyzed()
    current_hash = _compute_trades_hash(trades)
    current_count = len(trades)
    latest_ts = max((getattr(t, "timestamp_ms", 0) for t in trades), default=0)

    new_trade_count = current_count - int(last_analyzed.get("last_trade_count", 0))
    hash_changed = current_hash != last_analyzed.get("last_pnl_hash", "")
    try:
        last_analyzed_at = float(last_analyzed.get("analyzed_at", 0.0) or 0.0)
    except Exception:
        last_analyzed_at = 0.0
    idle_elapsed_sec = (time.time() - last_analyzed_at) if last_analyzed_at > 0 else 1e18
    forced_by_idle = bool(idle_force_sec > 0 and idle_elapsed_sec >= idle_force_sec)

    if (not always_run_cf) and (not forced_by_idle) and (not hash_changed) and new_trade_count < MIN_NEW_TRADES_FOR_REANALYSIS:
        logger.info(f"[SKIP] No significant new trades (new={new_trade_count}, min={MIN_NEW_TRADES_FOR_REANALYSIS}). Skipping CF cycle.")
        run_cf_cycle._cycle = cycle_index
        mtf_state = _collect_mtf_imageh_state(
            result={"running": False, "cycle_count": cycle_index, "sweep_progress": {}},
            findings=[],
            apply_history=[],
        )
        _update_dashboard({
            "running": False,
            "stage_current": "skipped",
            "skip_reason": "no_new_trades",
            "mtf_imageh": mtf_state,
        })
        return {
            "status": "skipped_no_new_trades",
            "cycle_count": cycle_index,
            "cycle_started_ts": cycle_started_ts,
            "last_update_ts": time.time(),
            "running": False,
            "research_mode": "normal",
            "deep_mode": False,
            "cycle_max_combos": int(max_combos),
            "ensemble_runs": 1,
            "new_trade_count": new_trade_count,
            "min_required": MIN_NEW_TRADES_FOR_REANALYSIS,
            "always_run_cf": always_run_cf,
            "forced_by_idle": forced_by_idle,
            "idle_elapsed_sec": idle_elapsed_sec,
            "idle_force_sec": idle_force_sec,
            "mtf_imageh": mtf_state,
        }

    if always_run_cf and (not hash_changed) and new_trade_count < MIN_NEW_TRADES_FOR_REANALYSIS:
        logger.info(
            f"[CF_FORCE] ALWAYS_RUN_CF=1 -> running CF with unchanged trades "
            f"(new={new_trade_count}, min={MIN_NEW_TRADES_FOR_REANALYSIS})"
        )
    elif forced_by_idle and (not hash_changed) and new_trade_count < MIN_NEW_TRADES_FOR_REANALYSIS:
        logger.info(
            f"[CF_FORCE] idle timeout -> running CF with unchanged trades "
            f"(idle={idle_elapsed_sec:.1f}s >= {idle_force_sec:.1f}s)"
        )

    # Update last analyzed info
    _save_last_analyzed({
        "last_trade_ts": latest_ts,
        "last_trade_count": current_count,
        "last_pnl_hash": current_hash,
        "analyzed_at": time.time(),
    })

    simulators = ALL_SIMULATORS
    if stage_filter:
        simulators = [s for s in ALL_SIMULATORS if s.stage_name == stage_filter]
        if not simulators:
            logger.error(f"Unknown stage: {stage_filter}")
            return {"status": "unknown_stage"}

    sample_seed = (int(cycle_started_ts * 1000) ^ (int(cycle_index) * 7919)) & 0xFFFFFFFF
    deep_enabled = _file_env_bool("CF_DEEP_SWEEP_ENABLED", _env_bool("CF_DEEP_SWEEP_ENABLED", True))
    deep_every = max(1, _file_env_int("CF_DEEP_SWEEP_EVERY_CYCLES", 6))
    deep_new_trades = max(MIN_NEW_TRADES_FOR_REANALYSIS, _file_env_int("CF_DEEP_MIN_NEW_TRADES", MIN_NEW_TRADES_FOR_REANALYSIS * 2))
    deep_mode = bool(
        deep_enabled and (
            (cycle_index % deep_every == 0)
            or bool(hash_changed)
            or int(new_trade_count) >= int(deep_new_trades)
        )
    )
    deep_max_combos = max(int(max_combos), _file_env_int("CF_DEEP_MAX_COMBOS", int(max_combos * 2)))
    cycle_max_combos = int(deep_max_combos if deep_mode else int(max_combos))
    normal_runs = max(1, _file_env_int("CF_ENSEMBLE_RUNS", 1))
    deep_runs = max(normal_runs, _file_env_int("CF_DEEP_ENSEMBLE_RUNS", 2))
    ensemble_runs = int(max(1, min(4, (deep_runs if deep_mode else normal_runs))))

    sweep_progress = {
        sim.stage_name: {"done": 0, "total": ensemble_runs, "status": "pending"}
        for sim in simulators
    }
    engines: list[CFEngine] = []
    for run_idx in range(ensemble_runs):
        run_seed = int((int(sample_seed) + (run_idx + 1) * 104729) % (2**32 - 1))
        if run_seed <= 0:
            run_seed = int(sample_seed or 1)
        engine_i = CFEngine(trades, simulators, sample_seed=run_seed)
        for sim in simulators:
            try:
                sweep_progress[sim.stage_name]["status"] = "running"
                _update_dashboard({
                    "sweep_progress": sweep_progress,
                    "running": True,
                    "stage_current": f"{sim.stage_name} [run {run_idx + 1}/{ensemble_runs}]",
                    "current_cycle_index": cycle_index,
                    "completed_cycles_total": max(0, cycle_index - 1),
                    "cycle_started_ts": cycle_started_ts,
                    "research_mode": ("deep" if deep_mode else "normal"),
                    "ensemble_runs": ensemble_runs,
                    "cycle_max_combos": cycle_max_combos,
                })
                engine_i._run_stage(sim, cycle_max_combos)
                sweep_progress[sim.stage_name]["done"] = int(sweep_progress[sim.stage_name].get("done", 0)) + 1
                if int(sweep_progress[sim.stage_name].get("done", 0)) >= ensemble_runs:
                    sweep_progress[sim.stage_name]["status"] = "done"
            except Exception as e:
                logger.error(f"Stage {sim.stage_name} failed: {e}")
                sweep_progress[sim.stage_name]["status"] = "error"
        engines.append(engine_i)

    elapsed = time.perf_counter() - t0
    primary_engine = engines[0] if engines else CFEngine(trades, simulators, sample_seed=sample_seed)

    merged_findings: dict[tuple[str, str], Any] = {}
    total_results = 0
    for eng in engines:
        total_results += int(len(getattr(eng, "results", [])))
        for f in eng.findings:
            try:
                key = (
                    str(f.stage or ""),
                    json.dumps(f.param_changes or {}, sort_keys=True, ensure_ascii=False),
                )
            except Exception:
                key = (str(f.stage or ""), str(f.finding_id or ""))
            prev = merged_findings.get(key)
            if (prev is None) or (float(f.improvement_pct or 0.0) > float(prev.improvement_pct or 0.0)):
                merged_findings[key] = f
    findings = sorted(merged_findings.values(), key=lambda x: float(x.improvement_pct or 0.0), reverse=True)[:20]
    apply_history_snapshot = []
    try:
        from research.auto_apply import get_apply_history
        apply_history_snapshot = get_apply_history()
    except Exception:
        apply_history_snapshot = []
    mtf_state = _collect_mtf_imageh_state(
        result={
            "running": False,
            "cycle_count": cycle_index,
            "sweep_progress": sweep_progress,
        },
        findings=[asdict(f) for f in findings],
        apply_history=apply_history_snapshot,
    )

    result = {
        "status": "ok",
        "elapsed_sec": round(elapsed, 1),
        "running": False,
        "research_mode": ("deep" if deep_mode else "normal"),
        "deep_mode": bool(deep_mode),
        "cycle_max_combos": int(cycle_max_combos),
        "ensemble_runs": int(ensemble_runs),
        "always_run_cf": always_run_cf,
        "forced_by_idle": forced_by_idle,
        "idle_elapsed_sec": float(idle_elapsed_sec),
        "idle_force_sec": float(idle_force_sec),
        "sample_seed": int(sample_seed),
        "baseline": primary_engine.baseline,
        "baseline_by_regime": primary_engine.baseline_by_regime,
        "n_trades": len(trades),
        "n_results": int(total_results),
        "n_findings": len(findings),
        "findings": [asdict(f) for f in findings],
        "top_10": [asdict(f) for f in findings[:10]],
        "sweep_progress": sweep_progress,
        "apply_history": apply_history_snapshot,
        "cycle_count": cycle_index,
        "current_cycle_index": cycle_index,
        "completed_cycles_total": cycle_index,
        "cycle_started_ts": cycle_started_ts,
        "stage_current": "done",
        "last_update_ts": time.time(),
        "mtf_imageh": mtf_state,
    }
    run_cf_cycle._cycle = cycle_index

    try:
        save_findings_json([asdict(f) for f in findings], FINDINGS_JSON)
    except Exception as e:
        logger.error(f"Failed to save findings JSON: {e}")

    try:
        generate_findings_markdown(
            [asdict(f) for f in findings],
            primary_engine.baseline,
            primary_engine.baseline_by_regime,
            FINDINGS_OUTPUT,
        )
    except Exception as e:
        logger.error(f"Failed to generate markdown: {e}")

    try:
        changelog = generate_changelog_entry([asdict(f) for f in findings[:5]])
        if changelog:
            logger.info(f"\n{'='*60}\nCHANGE LOG ENTRY:\n{'='*60}\n{changelog}\n{'='*60}")
    except Exception:
        pass

    _update_dashboard(result)

    logger.info(
        f"CF cycle complete: {len(findings)} findings in {elapsed:.1f}s "
        f"(mode={'deep' if deep_mode else 'normal'}, runs={ensemble_runs}, "
        f"trades={len(trades)}, results={total_results})"
    )

    return result


# ─────────────────────────────────────────────────────────────────
# Dashboard Integration
# ─────────────────────────────────────────────────────────────────

_dashboard_update_fn = None


def _update_dashboard(data: dict):
    global _dashboard_update_fn
    if _dashboard_update_fn:
        try:
            _dashboard_update_fn(data)
        except Exception:
            pass


def start_dashboard_thread(port: int = DASHBOARD_PORT):
    global _dashboard_update_fn
    try:
        from research.dashboard import update_state, start_dashboard_thread as _start
        _dashboard_update_fn = update_state
        t = _start(port=port)
        logger.info(f"Research dashboard started on http://0.0.0.0:{port}")
        return t
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Auto-Apply Integration
# ─────────────────────────────────────────────────────────────────

def run_auto_apply(findings: list[dict]) -> dict:
    """Run auto-apply cycle with safety guards."""
    try:
        from research.auto_apply import auto_apply_cycle, get_apply_history
        status = auto_apply_cycle(findings)
        _update_dashboard({
            "auto_apply": status,
            "apply_history": get_apply_history(),
        })
        return status
    except Exception as e:
        logger.error(f"Auto-apply error: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}


# ─────────────────────────────────────────────────────────────────
# Gemini Review Integration
# ─────────────────────────────────────────────────────────────────

def run_gemini_cycle(findings: list[dict]) -> dict:
    """Run Gemini code review."""
    try:
        from research.gemini_reviewer import run_gemini_review, is_available
        if not is_available():
            logger.debug("Gemini API key not set — skipping review")
            return {"status": "disabled"}
        from research.auto_apply import get_apply_history
        status = run_gemini_review(
            findings=findings,
            apply_history=get_apply_history(),
            auto_apply_env=False,  # Conservative: Gemini suggests only
        )
        _update_dashboard({"gemini_review": status})
        return status
    except Exception as e:
        logger.error(f"Gemini review error: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}


# ─────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    lock = SingletonProcessLock(Path(PROJECT_ROOT) / "state" / "locks" / "research_engine.lock")
    if not lock.acquire(role="research_engine", extra={"entrypoint": "research.runner"}):
        owner = lock.owner_pid
        logger.warning(
            "Another research engine is already running (owner_pid=%s). Exiting duplicate process.",
            owner if owner is not None else "unknown",
        )
        return
    parser = argparse.ArgumentParser(description="Codex Quant Research Engine v2")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--stage", default=None, help="Run specific stage only")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard")
    parser.add_argument("--no-auto-apply", action="store_true", help="Disable auto-apply")
    parser.add_argument("--no-gemini", action="store_true", help="Disable Gemini review")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT, help="Dashboard port")
    parser.add_argument("--interval", type=int, default=CF_INTERVAL_SEC, help="CF cycle interval (sec)")
    parser.add_argument("--gemini-interval", type=int, default=GEMINI_INTERVAL_SEC, help="Gemini review interval (sec)")
    parser.add_argument("--max-combos", type=int, default=MAX_COMBOS_PER_STAGE, help="Max parameter combos per stage")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Codex Quant Research Engine v2 — Auto-Optimize")
    logger.info(f"  DB: {args.db}")
    logger.info(f"  Stage: {args.stage or 'ALL ({} simulators)'.format(len(ALL_SIMULATORS))}")
    boot_interval = _runtime_interval_sec(args.interval, deep_mode=False)
    logger.info(f"  CF interval: {boot_interval}s ({boot_interval//60}min)")
    logger.info(f"  Auto-apply: {'ON (qualifying findings all apply, 30min monitor each, rollback on -$10)' if not args.no_auto_apply else 'OFF'}")
    logger.info(f"  Gemini review: {'ON' if not args.no_gemini else 'OFF'} (every {args.gemini_interval//60}min)")
    logger.info(f"  Dashboard: {'OFF' if args.no_dashboard else f'http://0.0.0.0:{args.port}'}")
    logger.info("=" * 60)

    if not args.no_dashboard:
        start_dashboard_thread(args.port)
        time.sleep(1)
        try:
            hist = _load_cycle_history()
            completed = int(hist[-1].get("cycle_index") or len(hist)) if hist else 0
            _update_dashboard({
                "research_cycles": hist[-30:],
                "completed_cycles_total": completed,
                "current_cycle_index": completed + 1,
                "stage_current": "idle",
            })
        except Exception:
            pass

    last_gemini_ts = 0
    last_findings: list[dict] = []

    try:
        while True:
            # ── Phase 1: CF Analysis (expanded stage simulators) ──
            result = run_cf_cycle(
                db_path=args.db,
                stage_filter=args.stage,
                max_combos=args.max_combos,
            )

            findings = result.get("findings", [])
            if findings:
                last_findings = findings

            apply_status = None

            if findings:
                print("\n" + "─" * 60)
                print(f"  TOP FINDINGS (cycle {result.get('cycle_count', 0)})")
                print("─" * 60)
                for i, f in enumerate(findings[:5], 1):
                    pnl = f.get("improvement_pct", 0)
                    conf = f.get("confidence", 0)
                    stage = f.get("stage", "?")
                    applied = " [APPLIED]" if f.get("applied") else ""
                    print(f"  {i}. [{stage}] ΔPnL: ${pnl:+.2f} | Conf: {conf:.0%}{applied}")
                print("─" * 60)

            # ── Phase 2: Auto-Apply (confidence ≥80%, ΔPnL ≥$100) ──
            if not args.no_auto_apply and findings:
                apply_status = run_auto_apply(findings)
                action = apply_status.get("last_action")
                if action and action not in ("no_qualifying_findings", "cooldown"):
                    logger.info(f"[AUTO_APPLY] {action}")

            try:
                from research.auto_apply import get_apply_history
                apply_history_snapshot = get_apply_history()
            except Exception:
                apply_history_snapshot = []

            cycle_report = _build_cycle_report(
                cycle_index=int(result.get("cycle_count") or getattr(run_cf_cycle, "_cycle", 0) or 0),
                cycle_started_ts=float(result.get("cycle_started_ts") or time.time()),
                result=result,
                findings=([f for f in findings if isinstance(f, dict)] if isinstance(findings, list) else []),
                apply_history=apply_history_snapshot,
                apply_status=apply_status,
                stage_filter=args.stage,
            )
            cycle_history = _load_cycle_history()
            cycle_history.append(cycle_report)
            _save_cycle_history(cycle_history)
            _update_dashboard({
                "status": result.get("status"),
                "cycle_count": int(result.get("cycle_count") or 0),
                "last_update_ts": float(result.get("last_update_ts") or time.time()),
                "research_mode": str(result.get("research_mode") or "normal"),
                "ensemble_runs": int(result.get("ensemble_runs") or 1),
                "cycle_max_combos": int(result.get("cycle_max_combos") or args.max_combos),
                "always_run_cf": result.get("always_run_cf"),
                "idle_force_sec": result.get("idle_force_sec"),
                "research_cycles": cycle_history[-30:],
                "current_cycle_index": int(result.get("cycle_count") or 0) + 1,
                "completed_cycles_total": int(result.get("cycle_count") or 0),
                "stage_current": "idle",
                "running": False,
                "apply_history": apply_history_snapshot,
                "auto_apply": apply_status or {},
                "last_cycle_report": cycle_report,
            })

            # ── Phase 3: Gemini Review (hourly) ──
            if not args.no_gemini and (time.time() - last_gemini_ts) >= args.gemini_interval:
                logger.info("Starting Gemini code review...")
                gemini_status = run_gemini_cycle(last_findings)
                last_gemini_ts = time.time()
                if gemini_status.get("status") == "ok":
                    logger.info(
                        f"[GEMINI] Review complete: {gemini_status.get('n_suggestions', 0)} suggestions, "
                        f"risk={gemini_status.get('risk_score', '?')}/10"
                    )
                elif gemini_status.get("status") != "disabled":
                    logger.warning(f"[GEMINI] {gemini_status}")

            if args.once:
                break

            next_interval = _runtime_interval_sec(args.interval, deep_mode=bool(result.get("deep_mode")))
            logger.info(
                f"Next CF cycle in {next_interval}s... "
                f"(mode={result.get('research_mode', 'normal')}, combos={result.get('cycle_max_combos')}, runs={result.get('ensemble_runs')})"
            )
            time.sleep(next_interval)

    except KeyboardInterrupt:
        logger.info("Research engine stopped by user")


if __name__ == "__main__":
    main()

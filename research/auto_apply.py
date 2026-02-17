"""
research/auto_apply.py — CF 결과 자동 적용 엔진
==================================================
보수적 안전 모드:
  - 1사이클당 최대 1개 변수만 변경
  - 적용 후 30분 모니터링 → 손실 시 자동 롤백
  - 모든 변경은 backup + audit log 기록
  - 엔진 무중단 Hot-Reload (DB runtime_config polling 기반)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from core.runtime_config_store import (
    ensure_runtime_config_schema,
    get_runtime_config_values,
    set_runtime_config_values,
)

logger = logging.getLogger("research.auto_apply")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FALLBACK_PATH = PROJECT_ROOT / "state" / "bybit.env"
BACKUP_DIR = PROJECT_ROOT / "state" / "runtime_config_backups"
APPLY_LOG = PROJECT_ROOT / "state" / "auto_apply_log.json"
DB_PATH = PROJECT_ROOT / "state" / "bot_data_live.db"

# ─────────────────────────────────────────────────────────────────
# Safety Thresholds
# ─────────────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.80          # 최소 신뢰도 80%
MIN_PNL_IMPROVEMENT = 100.0    # 예상 PnL 개선 "high" 기준(USD)
MONITOR_DURATION_SEC = 1800    # 30분 모니터링
ROLLBACK_LOSS_USD = 10.0       # $10 이상 손실 시 롤백
MAX_CHANGES_PER_CYCLE = 9999   # 제한 없음(사실상 무제한)
# Anti-flap guards:
# - global cooldown: minimum gap between apply batches
# - per-key cooldown: minimum gap before the same env key is changed again
COOLDOWN_BETWEEN_APPLY_SEC = 900
KEY_COOLDOWN_BETWEEN_APPLY_SEC = 1800

# ─────────────────────────────────────────────────────────────────
# CF Parameter → runtime_config key mapping (env-compatible key names)
# ─────────────────────────────────────────────────────────────────
PARAM_TO_ENV: dict[str, list[str]] = {
    # Leverage
    "max_leverage": ["MAX_LEVERAGE"],
    "regime_max_bull": ["LEVERAGE_REGIME_MAX_BULL"],
    "regime_max_bear": ["LEVERAGE_REGIME_MAX_BEAR"],
    "regime_max_chop": ["LEVERAGE_REGIME_MAX_CHOP"],
    "regime_max_volatile": ["LEVERAGE_REGIME_MAX_VOLATILE"],
    # TP/SL
    "tp_pct": ["MC_TP_BASE_ROE", "DEFAULT_TP_PCT"],
    "sl_pct": ["MC_SL_BASE_ROE", "DEFAULT_SL_PCT"],
    # Hold Duration
    "max_hold_sec": ["POLICY_MAX_HOLD_SEC"],
    "min_hold_sec": ["EXIT_MIN_HOLD_SEC"],
    "min_hold_sec_bull": ["EXIT_MIN_HOLD_SEC_BULL"],
    "min_hold_sec_chop": ["EXIT_MIN_HOLD_SEC_CHOP"],
    "min_hold_sec_bear": ["EXIT_MIN_HOLD_SEC_BEAR"],
    # Entry Filters
    "min_confidence": ["UNI_MIN_CONFIDENCE"],
    "min_dir_conf": ["ALPHA_DIRECTION_MIN_CONFIDENCE"],
    "min_ev": ["UNIFIED_ENTRY_FLOOR"],
    "min_dir_conf_for_entry": ["ALPHA_DIRECTION_MIN_CONFIDENCE"],
    # VPIN
    "max_vpin": ["UNI_MAX_VPIN_HARD"],
    "chop_min_sigma": ["CHOP_VOL_GATE_MIN_SIGMA"],
    "chop_max_sigma": ["CHOP_VOL_GATE_MAX_SIGMA"],
    "chop_max_vpin": ["CHOP_VPIN_MAX"],
    "chop_min_dir_conf": ["CHOP_ENTRY_MIN_DIR_CONF"],
    "chop_min_abs_mu_alpha": ["CHOP_ENTRY_MIN_MU_ALPHA"],
    "chop_max_hold_sec": ["TARGET_HOLD_SEC_MAX_CHOP"],
    # Exit
    "block_mu_sign_flip_before_sec": ["MU_SIGN_FLIP_MIN_AGE_SEC"],
    "mu_sign_flip_min_magnitude": ["MU_SIGN_FLIP_MIN_MAGNITUDE"],
    "mu_sign_flip_confirm_ticks": ["MU_SIGN_FLIP_CONFIRM_TICKS"],
    # Capital
    "notional_hard_cap": ["NOTIONAL_HARD_CAP_USD"],
    "max_pos_frac": ["UNI_MAX_POS_FRAC"],
    "max_concurrent": ["MAX_CONCURRENT_POSITIONS"],
    # Spread
    "spread_pct_max": ["SPREAD_PCT_MAX"],
    # Fee
    "fee_filter_mult": ["FEE_FILTER_MULT"],
    # Net Expectancy
    "net_expectancy_min": ["ENTRY_NET_EXPECTANCY_MIN"],
    "both_ev_neg_net_floor": ["ENTRY_BOTH_EV_NEG_NET_FLOOR"],
    "gross_ev_min": ["ENTRY_GROSS_EV_MIN"],
    # Exposure
    "max_exposure": ["MAX_NOTIONAL_EXPOSURE", "LIVE_MAX_NOTIONAL_EXPOSURE"],
    # Hurst
    "hurst_dampen": ["HURST_RANDOM_DAMPEN"],
    # Chop
    "chop_entry_floor_add": ["CHOP_ENTRY_FLOOR_ADD"],
    "chop_entry_min_dir_conf": ["CHOP_ENTRY_MIN_DIR_CONF"],
    # Expanded research params
    "top_n_symbols": ["TOP_N_SYMBOLS"],
    "dir_gate_min_conf": ["ALPHA_DIRECTION_GATE_MIN_CONF"],
    "dir_gate_min_edge": ["ALPHA_DIRECTION_GATE_MIN_EDGE"],
    "dir_gate_min_side_prob": ["ALPHA_DIRECTION_GATE_MIN_SIDE_PROB"],
    "pre_mc_min_expected_pnl": ["PRE_MC_MIN_EXPECTED_PNL"],
    "pre_mc_max_liq_prob": ["PRE_MC_MAX_LIQ_PROB"],
    "event_exit_max_p_sl": ["EVENT_EXIT_MAX_P_SL"],
    "event_exit_max_abs_cvar": ["EVENT_EXIT_MAX_ABS_CVAR"],
    "min_entry_notional": ["MIN_ENTRY_NOTIONAL"],
    "trading_bad_hours_utc": ["TRADING_BAD_HOURS_UTC"],
    "regime_side_block_list": ["REGIME_SIDE_BLOCK_LIST"],
    "lev_floor_lock_min_sticky": ["LEVERAGE_FLOOR_LOCK_MIN_STICKY"],
    "lev_floor_lock_max_ev_gap": ["LEVERAGE_FLOOR_LOCK_MAX_EV_GAP"],
    "lev_floor_lock_max_conf": ["LEVERAGE_FLOOR_LOCK_MAX_CONF"],
    "pre_mc_size_scale": ["PRE_MC_SIZE_SCALE"],
    "pre_mc_block_on_fail": ["PRE_MC_BLOCK_ON_FAIL"],
    "pre_mc_min_cvar": ["PRE_MC_MIN_CVAR"],
    "dir_gate_confirm_ticks": ["ALPHA_DIRECTION_GATE_CONFIRM_TICKS"],
    "dir_gate_confirm_ticks_chop": ["ALPHA_DIRECTION_GATE_CONFIRM_TICKS_CHOP"],
    "sq_time_window_hours": ["SYMBOL_QUALITY_TIME_WINDOW_HOURS"],
    "sq_time_weight": ["SYMBOL_QUALITY_TIME_WEIGHT"],
    # Hybrid-expanded research params
    "hybrid_exit_confirm_shock": ["HYBRID_EXIT_CONFIRM_TICKS_SHOCK"],
    "hybrid_exit_confirm_normal": ["HYBRID_EXIT_CONFIRM_TICKS_NORMAL"],
    "hybrid_exit_confirm_noise": ["HYBRID_EXIT_CONFIRM_TICKS_NOISE"],
    "hybrid_lev_sweep_min": ["HYBRID_LEV_MIN"],
    "hybrid_lev_sweep_max": ["HYBRID_LEV_MAX"],
    "mc_hybrid_n_paths": ["MC_HYBRID_N_PATHS"],
    "mc_hybrid_horizon_steps": ["MC_HYBRID_HORIZON_STEPS"],
    "hybrid_cash_penalty": ["HYBRID_CASH_PENALTY"],
    # MTF entry gate thresholds
    "mtf_dl_entry_min_side_prob": ["MTF_DL_ENTRY_MIN_SIDE_PROB"],
    "mtf_dl_entry_min_win_prob": ["MTF_DL_ENTRY_MIN_WIN_PROB"],
}

# 값 범위 제한 (안전 가드)
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "MAX_LEVERAGE": (3, 50),
    "LEVERAGE_REGIME_MAX_BULL": (5, 40),
    "LEVERAGE_REGIME_MAX_BEAR": (3, 30),
    "LEVERAGE_REGIME_MAX_CHOP": (2, 30),
    "LEVERAGE_REGIME_MAX_VOLATILE": (2, 20),
    "MC_TP_BASE_ROE": (0.003, 0.10),
    "DEFAULT_TP_PCT": (0.003, 0.10),
    "MC_SL_BASE_ROE": (0.003, 0.05),
    "DEFAULT_SL_PCT": (0.003, 0.05),
    "POLICY_MAX_HOLD_SEC": (120, 14400),
    "EXIT_MIN_HOLD_SEC": (30, 1200),
    "UNI_MIN_CONFIDENCE": (0.10, 0.90),
    "ALPHA_DIRECTION_MIN_CONFIDENCE": (0.40, 0.80),
    "UNIFIED_ENTRY_FLOOR": (-0.01, 0.05),
    "UNI_MAX_VPIN_HARD": (0.20, 0.99),
    "CHOP_VOL_GATE_MIN_SIGMA": (0.0, 5.0),
    "CHOP_VOL_GATE_MAX_SIGMA": (0.01, 10.0),
    "CHOP_VPIN_MAX": (0.05, 0.99),
    "CHOP_ENTRY_MIN_MU_ALPHA": (0.0, 200.0),
    "TARGET_HOLD_SEC_MAX_CHOP": (60.0, 21600.0),
    "MU_SIGN_FLIP_MIN_AGE_SEC": (60, 3600),
    "NOTIONAL_HARD_CAP_USD": (20, 5000),
    "UNI_MAX_POS_FRAC": (0.10, 0.80),
    "MAX_CONCURRENT_POSITIONS": (1, 20),
    "SPREAD_PCT_MAX": (0.0001, 0.005),
    "FEE_FILTER_MULT": (0.50, 1.50),
    "ENTRY_NET_EXPECTANCY_MIN": (-0.005, 0.01),
    "ENTRY_BOTH_EV_NEG_NET_FLOOR": (-0.01, 0.01),
    "ENTRY_GROSS_EV_MIN": (0.0, 0.02),
    "MAX_NOTIONAL_EXPOSURE": (1.0, 10.0),
    "HURST_RANDOM_DAMPEN": (0.20, 1.00),
    "CHOP_ENTRY_FLOOR_ADD": (0.0, 0.01),
    "CHOP_ENTRY_MIN_DIR_CONF": (0.50, 0.90),
    "TOP_N_SYMBOLS": (2, 100),
    "ALPHA_DIRECTION_GATE_MIN_CONF": (0.45, 0.90),
    "ALPHA_DIRECTION_GATE_MIN_EDGE": (0.0, 0.20),
    "ALPHA_DIRECTION_GATE_MIN_SIDE_PROB": (0.45, 0.95),
    "PRE_MC_MIN_EXPECTED_PNL": (-0.001, 0.01),
    "PRE_MC_MAX_LIQ_PROB": (0.01, 0.50),
    "EVENT_EXIT_MAX_P_SL": (0.50, 0.999),
    "EVENT_EXIT_MAX_ABS_CVAR": (0.005, 0.50),
    "MIN_ENTRY_NOTIONAL": (0.1, 200.0),
    "LEVERAGE_FLOOR_LOCK_MIN_STICKY": (1, 12),
    "LEVERAGE_FLOOR_LOCK_MAX_EV_GAP": (0.0001, 0.01),
    "LEVERAGE_FLOOR_LOCK_MAX_CONF": (0.40, 0.90),
    "PRE_MC_SIZE_SCALE": (0.10, 1.00),
    "PRE_MC_BLOCK_ON_FAIL": (0, 1),
    "PRE_MC_MIN_CVAR": (-0.30, -0.005),
    "ALPHA_DIRECTION_GATE_CONFIRM_TICKS": (1, 8),
    "ALPHA_DIRECTION_GATE_CONFIRM_TICKS_CHOP": (1, 8),
    "SYMBOL_QUALITY_TIME_WINDOW_HOURS": (1, 12),
    "SYMBOL_QUALITY_TIME_WEIGHT": (0.0, 1.0),
    "HYBRID_EXIT_CONFIRM_TICKS_SHOCK": (1, 8),
    "HYBRID_EXIT_CONFIRM_TICKS_NORMAL": (1, 8),
    "HYBRID_EXIT_CONFIRM_TICKS_NOISE": (1, 12),
    "HYBRID_LEV_MIN": (0.5, 20.0),
    "HYBRID_LEV_MAX": (1.0, 100.0),
    "MC_HYBRID_N_PATHS": (512, 32768),
    "MC_HYBRID_HORIZON_STEPS": (15, 1200),
    "HYBRID_CASH_PENALTY": (0.0, 0.02),
    "MTF_DL_ENTRY_MIN_SIDE_PROB": (0.45, 0.90),
    "MTF_DL_ENTRY_MIN_WIN_PROB": (0.45, 0.90),
}


@dataclass
class ApplyRecord:
    """Applied change audit record."""
    timestamp: float
    finding_id: str
    stage: str
    param_changes: dict       # CF param → value
    env_changes: dict         # ENV_VAR → {"old": x, "new": y}
    backup_path: str
    equity_at_apply: float
    equity_after_monitor: Optional[float] = None
    rolled_back: bool = False
    rollback_reason: str = ""
    monitor_duration_sec: int = MONITOR_DURATION_SEC
    status: str = "pending"   # pending / monitoring / applied / rolled_back


# ─────────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────────

def _load_apply_log() -> list[dict]:
    if APPLY_LOG.exists():
        try:
            return json.loads(APPLY_LOG.read_text())
        except Exception:
            return []
    return []


def _save_apply_log(records: list[dict]):
    APPLY_LOG.parent.mkdir(parents=True, exist_ok=True)
    APPLY_LOG.write_text(json.dumps(records, indent=2, default=str, ensure_ascii=False))


def _get_current_equity() -> float:
    """Read current equity from the database or balance.json."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT total_equity FROM equity_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            return float(row[0])
    except Exception:
        pass
    # Fallback: balance.json
    bj = PROJECT_ROOT / "balance.json"
    if bj.exists():
        try:
            d = json.loads(bj.read_text())
            return float(d.get("totalEquity") or d.get("equity") or 0)
        except Exception:
            pass
    return 0.0


def _read_env_value(key: str) -> Optional[str]:
    """
    Read runtime value with priority:
    1) runtime_config table
    2) process env
    3) fallback file (state/bybit.env)
    """
    k = str(key or "").strip().upper()
    if not k:
        return None
    try:
        ensure_runtime_config_schema(str(DB_PATH))
        vals = get_runtime_config_values(str(DB_PATH), keys=[k])
        v = vals.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    except Exception:
        pass
    raw = os.environ.get(k)
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip()
    if not ENV_FALLBACK_PATH.exists():
        return None
    for line in ENV_FALLBACK_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        kk, _, vv = stripped.partition("=")
        if kk.strip().upper() == k:
            return vv.strip()
    return None


def _env_float_runtime(key: str, default: float) -> float:
    """Prefer runtime_config/env, then fallback file."""
    raw = _read_env_value(key)
    try:
        if raw is None:
            return float(default)
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_bool_runtime(key: str, default: bool) -> bool:
    """Prefer runtime_config/env, then fallback file."""
    raw = _read_env_value(key)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _apply_runtime_updates(
    updates: dict[str, str],
    *,
    source: str,
    reason: str,
    batch_id: str,
) -> dict[str, dict[str, str | None]]:
    """
    Persist config updates atomically to SQLite runtime_config and mirror to process env.
    Returns actual changed keys with old/new values.
    """
    clean: dict[str, str] = {}
    for k, v in (updates or {}).items():
        kk = str(k or "").strip().upper()
        vv = str(v).strip() if v is not None else ""
        if not kk or not vv:
            continue
        if not all(ch.isalnum() or ch == "_" for ch in kk):
            continue
        clean[kk] = vv
    if not clean:
        return {}
    ensure_runtime_config_schema(str(DB_PATH))
    changed = set_runtime_config_values(
        str(DB_PATH),
        clean,
        source=str(source or "auto_apply"),
        reason=str(reason or ""),
        batch_id=str(batch_id or ""),
    )
    for key, payload in (changed or {}).items():
        new_val = payload.get("new")
        if new_val is None:
            continue
        os.environ[str(key)] = str(new_val)
    return changed


def backup_env(keys: list[str] | None = None) -> str:
    """Backup current runtime config snapshot and return backup path."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"runtime_config.{ts}.json"
    if keys:
        norm_keys = sorted({str(k or "").strip().upper() for k in keys if str(k or "").strip()})
    else:
        ensure_runtime_config_schema(str(DB_PATH))
        norm_keys = sorted(get_runtime_config_values(str(DB_PATH)).keys())
    snapshot = {k: _read_env_value(k) for k in norm_keys}
    payload = {
        "timestamp": time.time(),
        "db_path": str(DB_PATH),
        "keys": norm_keys,
        "values": snapshot,
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep only last 50 backups
    backups = sorted(BACKUP_DIR.glob("runtime_config.*.json"))
    for old in backups[:-50]:
        try:
            old.unlink()
        except Exception:
            pass
    logger.info(f"Backed up runtime_config snapshot → {dest}")
    return str(dest)


def rollback_env(backup_path: str) -> bool:
    """Restore runtime_config values from backup snapshot."""
    bp = Path(backup_path)
    if not bp.exists():
        logger.error(f"Backup not found: {backup_path}")
        return False
    try:
        obj = json.loads(bp.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Backup parse failed: {backup_path} ({e})")
        return False
    vals = obj.get("values") if isinstance(obj, dict) else {}
    if not isinstance(vals, dict) or not vals:
        logger.error(f"Backup has no values: {backup_path}")
        return False
    updates = {str(k).strip().upper(): str(v).strip() for k, v in vals.items() if v is not None and str(k).strip()}
    if not updates:
        logger.error(f"Backup values empty (all null): {backup_path}")
        return False
    batch_id = f"rollback-{int(time.time())}"
    changed = _apply_runtime_updates(
        updates,
        source="auto_apply_rollback",
        reason=f"rollback:{bp.name}",
        batch_id=batch_id,
    )
    if not changed:
        logger.warning(f"Rollback requested but no changes applied: {backup_path}")
        return True
    logger.warning(f"ROLLED BACK runtime_config from {backup_path} (keys={len(changed)})")
    return True


def restart_engine() -> bool:
    """Deprecated: hot-reload architecture no longer restarts the process."""
    logger.info("[AUTO_APPLY] restart skipped: runtime_config hot-reload is active")
    return True


def _is_in_cooldown() -> bool:
    """Check global cooldown and active monitoring window."""
    remain = _cooldown_remaining_sec()
    return bool(remain > 0)


def _cooldown_seconds() -> int:
    try:
        return max(0, int(_env_float_runtime("AUTO_APPLY_COOLDOWN_SEC", COOLDOWN_BETWEEN_APPLY_SEC)))
    except Exception:
        return int(COOLDOWN_BETWEEN_APPLY_SEC)


def _key_cooldown_seconds() -> int:
    try:
        return max(0, int(_env_float_runtime("AUTO_APPLY_KEY_COOLDOWN_SEC", KEY_COOLDOWN_BETWEEN_APPLY_SEC)))
    except Exception:
        return int(KEY_COOLDOWN_BETWEEN_APPLY_SEC)


def _cooldown_remaining_sec(now_ts: float | None = None) -> int:
    """Remaining seconds for global cooldown (includes active monitoring)."""
    records = _load_apply_log()
    if not records:
        return 0
    if now_ts is None:
        now_ts = time.time()

    # If any change is still monitoring, block until monitor end.
    monitor_remains: list[float] = []
    for rec in records:
        if str(rec.get("status", "")) == "monitoring":
            ts = float(rec.get("timestamp", 0) or 0)
            monitor_remains.append((ts + float(MONITOR_DURATION_SEC)) - float(now_ts))
    if monitor_remains:
        return int(max(0.0, max(monitor_remains)))

    cooldown_sec = _cooldown_seconds()
    if cooldown_sec <= 0:
        return 0

    # Fallback global cooldown after the latest apply-related record.
    last_ts = 0.0
    for rec in records:
        st = str(rec.get("status", "")).strip().lower()
        if st in ("applied", "rolled_back", "monitoring"):
            try:
                ts = float(rec.get("timestamp", 0) or 0)
            except Exception:
                ts = 0.0
            if ts > last_ts:
                last_ts = ts
    if last_ts <= 0:
        return 0
    remain = (last_ts + float(cooldown_sec)) - float(now_ts)
    return int(max(0.0, remain))


def _env_key_cooldown_remaining_sec(env_key: str, now_ts: float | None = None) -> int:
    """Remaining cooldown seconds for a specific env key."""
    key = str(env_key or "").strip()
    if not key:
        return 0
    key_cd_sec = _key_cooldown_seconds()
    if key_cd_sec <= 0:
        return 0
    if now_ts is None:
        now_ts = time.time()
    records = _load_apply_log()
    if not records:
        return 0
    for rec in reversed(records):
        env_changes = rec.get("env_changes") or {}
        if not isinstance(env_changes, dict):
            continue
        if key not in env_changes:
            continue
        try:
            ts = float(rec.get("timestamp", 0) or 0)
        except Exception:
            ts = 0.0
        if ts <= 0:
            continue
        remain = (ts + float(key_cd_sec)) - float(now_ts)
        return int(max(0.0, remain))
    return 0


def _clamp_value(env_key: str, value: float) -> float:
    """Clamp value to safe bounds."""
    bounds = PARAM_BOUNDS.get(env_key)
    if bounds:
        return max(bounds[0], min(bounds[1], value))
    return value


def _format_env_value(value, env_key: str) -> str:
    """Format value for bybit.env (int vs float)."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int) or (isinstance(value, float) and value == int(value)):
        # Check if the env key usually has int values
        int_keys = {"MAX_LEVERAGE", "LEVERAGE_REGIME_MAX_BULL", "LEVERAGE_REGIME_MAX_BEAR",
                     "LEVERAGE_REGIME_MAX_CHOP", "LEVERAGE_REGIME_MAX_VOLATILE",
                     "POLICY_MAX_HOLD_SEC", "EXIT_MIN_HOLD_SEC", "MAX_CONCURRENT_POSITIONS",
                     "MU_SIGN_FLIP_MIN_AGE_SEC", "MU_SIGN_FLIP_CONFIRM_TICKS",
                     "EXIT_MIN_HOLD_SEC_BULL", "EXIT_MIN_HOLD_SEC_CHOP", "EXIT_MIN_HOLD_SEC_BEAR",
                     "TOP_N_SYMBOLS", "LEVERAGE_FLOOR_LOCK_MIN_STICKY",
                     "ALPHA_DIRECTION_GATE_CONFIRM_TICKS", "ALPHA_DIRECTION_GATE_CONFIRM_TICKS_CHOP",
                     "SYMBOL_QUALITY_TIME_WINDOW_HOURS", "PRE_MC_BLOCK_ON_FAIL"}
        if env_key in int_keys:
            return str(int(value))
    if isinstance(value, float):
        if abs(value) < 0.01:
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


# ─────────────────────────────────────────────────────────────────
# Auto-Apply Decision
# ─────────────────────────────────────────────────────────────────

def should_apply(finding: dict) -> bool:
    """Check if a finding meets auto-apply criteria."""
    conf = float(finding.get("confidence", 0) or 0.0)
    pnl = float(finding.get("improvement_pct", 0) or 0.0)  # This is actually delta PnL in $
    min_conf = float(_env_float_runtime("AUTO_APPLY_MIN_CONFIDENCE", MIN_CONFIDENCE))
    high_pnl = float(_env_float_runtime("AUTO_APPLY_HIGH_PNL_IMPROVEMENT", MIN_PNL_IMPROVEMENT))
    require_oos = bool(_env_bool_runtime("AUTO_APPLY_REQUIRE_OOS_PASS", True))
    min_oos_pass_rate = float(_env_float_runtime("AUTO_APPLY_MIN_OOS_PASS_RATE", 0.60))
    min_oos_test_delta = float(_env_float_runtime("AUTO_APPLY_MIN_OOS_TEST_PNL_DELTA", 0.0))
    conf_ok = bool(conf >= min_conf)
    pnl_ok = bool(pnl >= high_pnl)
    if not (conf_ok or pnl_ok):
        logger.debug(
            f"Skip {finding.get('finding_id')}: conf={conf:.2f}<{min_conf:.2f} "
            f"and pnl=${pnl:.2f}<${high_pnl:.2f}"
        )
        return False
    if require_oos:
        oos_pass = bool(finding.get("oos_pass", False))
        oos_rate = float(finding.get("oos_pass_rate", finding.get("walk_forward_pass_rate", 0.0)) or 0.0)
        oos_test_delta = float(finding.get("oos_test_pnl_delta", finding.get("walk_forward_test_pnl_delta", 0.0)) or 0.0)
        if (not oos_pass) or (oos_rate < min_oos_pass_rate) or (oos_test_delta < min_oos_test_delta):
            logger.debug(
                f"Skip {finding.get('finding_id')}: oos_pass={oos_pass} "
                f"oos_rate={oos_rate:.2f}<{min_oos_pass_rate:.2f} "
                f"oos_test_delta={oos_test_delta:.2f}<{min_oos_test_delta:.2f}"
            )
            return False
    if finding.get("applied"):
        return False
    return True


def _build_env_changes_for_finding(finding: dict) -> dict[str, dict]:
    """Build env var changes for one finding without applying them."""
    param_changes = finding.get("param_changes", {})
    env_changes: dict[str, dict] = {}
    for cf_param, cf_value in param_changes.items():
        env_keys = PARAM_TO_ENV.get(cf_param, [])
        if not env_keys:
            continue
        for env_key in env_keys:
            if isinstance(cf_value, (int, float)):
                clamped = _clamp_value(env_key, float(cf_value))
            else:
                clamped = cf_value
            old = _read_env_value(env_key)
            new_str = _format_env_value(clamped, env_key)
            if old is not None and old == new_str:
                continue
            env_changes[env_key] = {"old": old, "new": new_str}
    return env_changes


def _unmapped_params_for_finding(finding: dict) -> list[str]:
    param_changes = finding.get("param_changes", {}) or {}
    out: list[str] = []
    for cf_param in param_changes.keys():
        if not PARAM_TO_ENV.get(str(cf_param)):
            out.append(str(cf_param))
    return out


def apply_finding(finding: dict, *, restart_after_apply: bool = True) -> Optional[ApplyRecord]:
    """
    Apply a single CF finding to SQLite runtime_config.
    Returns ApplyRecord on success, None on failure.
    """
    param_changes = finding.get("param_changes", {})
    if not param_changes:
        return None

    env_changes = _build_env_changes_for_finding(finding)

    if not env_changes:
        logger.info(f"No effective env changes for finding {finding.get('finding_id')}")
        return None

    # Backup
    backup_path = backup_env(list(env_changes.keys()))
    equity_now = _get_current_equity()

    # Apply changes
    updates = {k: v["new"] for k, v in env_changes.items()}
    changed = _apply_runtime_updates(
        updates,
        source="auto_apply",
        reason=f"finding:{finding.get('finding_id')}",
        batch_id=f"{finding.get('finding_id')}-{int(time.time())}",
    )
    if not changed:
        logger.info(f"No runtime_config changes persisted for finding {finding.get('finding_id')}")
        return None
    for env_key, change in changed.items():
        logger.info(
            f"[AUTO_APPLY] {env_key}: {change.get('old')} → {change.get('new')} "
            f"(finding={finding.get('finding_id')}, stage={finding.get('stage')})"
        )

    record = ApplyRecord(
        timestamp=time.time(),
        finding_id=finding.get("finding_id", ""),
        stage=finding.get("stage", ""),
        param_changes=param_changes,
        env_changes=changed,
        backup_path=backup_path,
        equity_at_apply=equity_now,
        status="monitoring",
    )

    # Save audit log
    records = _load_apply_log()
    records.append(asdict(record))
    _save_apply_log(records)

    # restart_after_apply is retained for API compatibility only.
    if restart_after_apply:
        restart_engine()
    return record


def check_and_rollback_all() -> dict:
    """
    Check if a pending monitoring period has elapsed.
    If performance degraded, rollback.
    Returns summary dict.
    """
    records = _load_apply_log()
    if not records:
        return {"processed": 0, "applied": 0, "rolled_back": 0, "monitoring": 0, "messages": []}

    now = time.time()
    equity_now = _get_current_equity()
    processed = applied = rolled_back = monitoring = 0
    messages: list[str] = []
    for rec in records:
        if rec.get("status") != "monitoring":
            continue
        elapsed = now - float(rec.get("timestamp", 0) or 0)
        if elapsed < MONITOR_DURATION_SEC:
            monitoring += 1
            continue

        processed += 1
        delta = float(equity_now) - float(rec.get("equity_at_apply", 0) or 0)
        rec["equity_after_monitor"] = float(equity_now)
        if delta < -ROLLBACK_LOSS_USD:
            logger.warning(
                f"[AUTO_ROLLBACK] {rec.get('finding_id')} Δ${delta:.2f} < -${ROLLBACK_LOSS_USD:.2f}; "
                f"backup={rec.get('backup_path')}"
            )
            rollback_env(rec["backup_path"])
            rec["status"] = "rolled_back"
            rec["rolled_back"] = True
            rec["rollback_reason"] = f"equity_drop_{delta:.2f}"
            rolled_back += 1
            messages.append(f"rolled_back {rec.get('finding_id')} (Δ${delta:.2f})")
        else:
            rec["status"] = "applied"
            applied += 1
            messages.append(f"applied {rec.get('finding_id')} (Δ${delta:+.2f})")

    if processed > 0:
        _save_apply_log(records)

    return {
        "processed": processed,
        "applied": applied,
        "rolled_back": rolled_back,
        "monitoring": monitoring,
        "messages": messages,
    }


def git_commit_and_push(message: str = "auto: CF parameter update") -> bool:
    """Git commit audit artifacts and push."""
    try:
        subprocess.run(
            ["git", "add", "state/auto_apply_log.json"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=30,
        )
        result = subprocess.run(
            ["git", "push"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=60,
        )
        if result.returncode == 0:
            logger.info(f"Git push successful: {message}")
            return True
        else:
            logger.warning(f"Git push failed: {result.stderr.decode()}")
            return False
    except Exception as e:
        logger.error(f"Git error: {e}")
        return False


def auto_apply_cycle(findings: list[dict]) -> dict:
    """
    Main auto-apply cycle. Called from runner.py.
    1. Check & rollback pending monitors
    2. Find best applicable finding
    3. Apply if criteria met
    4. Git push

    Returns status dict for dashboard.
    """
    status = {
        "auto_apply_enabled": True,
        "last_check_ts": time.time(),
        "cooldown": False,
        "cooldown_remaining_sec": 0,
        "applied_count": 0,
        "rollback_count": 0,
        "last_action": None,
        "monitoring_count": 0,
    }

    # Phase 1: Check all pending monitors
    mon = check_and_rollback_all()
    status["rollback_count"] = int(mon.get("rolled_back", 0))
    status["monitoring_count"] = int(mon.get("monitoring", 0))
    if mon.get("processed", 0) > 0:
        status["last_action"] = "; ".join(mon.get("messages", [])[:4])
        if mon.get("rolled_back", 0) > 0:
            git_commit_and_push("auto: rollback — performance degradation")
        elif mon.get("applied", 0) > 0:
            git_commit_and_push("auto: CF changes confirmed — performance maintained")

    # Phase 1.5: Global cooldown guard
    remain_cd = _cooldown_remaining_sec()
    if remain_cd > 0:
        status["cooldown"] = True
        status["cooldown_remaining_sec"] = int(remain_cd)
        if not status.get("last_action"):
            status["last_action"] = f"in_cooldown ({int(remain_cd)}s remaining)"
        return status

    # Phase 2: Find applicable findings (OR condition)
    applicable = [f for f in findings if should_apply(f)]
    if not applicable:
        if not status.get("last_action"):
            status["last_action"] = "no_qualifying_findings"
        return status

    # Phase 3: Apply qualifying findings.
    # Default: apply all qualified findings (no count cap), keep only one per env-key by higher ΔPnL.
    sorted_applicable = sorted(applicable, key=lambda f: float(f.get("improvement_pct", 0) or 0), reverse=True)
    apply_all_qualified = _env_bool_runtime("AUTO_APPLY_APPLY_ALL_QUALIFIED", True)
    if not apply_all_qualified and sorted_applicable:
        sorted_applicable = sorted_applicable[:1]
    planned: list[dict] = []
    used_env_keys: set[str] = set()
    unmapped_findings: list[dict] = []
    no_effective_findings: list[str] = []
    for f in sorted_applicable:
        env_changes = _build_env_changes_for_finding(f)
        if not env_changes:
            unmapped = _unmapped_params_for_finding(f)
            if unmapped:
                unmapped_findings.append({
                    "finding_id": str(f.get("finding_id") or ""),
                    "stage": str(f.get("stage") or ""),
                    "unmapped_params": unmapped,
                })
            else:
                no_effective_findings.append(str(f.get("finding_id") or ""))
            continue
        filtered_changes = {k: v for k, v in env_changes.items() if k not in used_env_keys}
        # Per-key cooldown guard: avoid rapid oscillation on same env key.
        now_ts = time.time()
        blocked_keys: dict[str, int] = {}
        kept_changes: dict[str, dict] = {}
        for env_key, change in filtered_changes.items():
            remain = _env_key_cooldown_remaining_sec(env_key, now_ts=now_ts)
            if remain > 0:
                blocked_keys[str(env_key)] = int(remain)
                continue
            kept_changes[env_key] = change
        if blocked_keys:
            status.setdefault("key_cooldown_blocked", {})[str(f.get("finding_id") or "")] = blocked_keys
        filtered_changes = kept_changes
        if not filtered_changes:
            no_effective_findings.append(str(f.get("finding_id") or ""))
            continue
        planned.append({"finding": f, "env_changes": filtered_changes})
        used_env_keys.update(filtered_changes.keys())

    if not planned:
        status["unmapped_findings"] = unmapped_findings[:10]
        status["no_effective_findings"] = [fid for fid in no_effective_findings if fid][:20]
        status["unmapped_count"] = int(len(unmapped_findings))
        status["no_effective_count"] = int(len(no_effective_findings))
        if not status.get("last_action"):
            if unmapped_findings:
                status["last_action"] = "qualifying_findings_unmapped_params"
            else:
                status["last_action"] = "qualifying_findings_but_no_effective_env_changes"
        return status

    backup_path = backup_env(list(used_env_keys))
    equity_now = _get_current_equity()
    records = _load_apply_log()
    new_records = []
    batch_updates: dict[str, str] = {}
    for item in planned:
        for env_key, change in (item.get("env_changes") or {}).items():
            batch_updates[str(env_key)] = str(change.get("new"))
    batch_id = f"batch-{int(time.time())}"
    changed_batch = _apply_runtime_updates(
        batch_updates,
        source="auto_apply",
        reason="batch_apply",
        batch_id=batch_id,
    )
    if not changed_batch:
        status["last_action"] = "no_runtime_config_changes_applied"
        return status

    for item in planned:
        f = item["finding"]
        env_changes = {
            k: v for k, v in changed_batch.items() if k in (item.get("env_changes") or {})
        }
        if not env_changes:
            continue
        for env_key, change in env_changes.items():
            logger.info(
                f"[AUTO_APPLY] {env_key}: {change.get('old')} → {change.get('new')} "
                f"(finding={f.get('finding_id')}, stage={f.get('stage')})"
            )
        record = ApplyRecord(
            timestamp=time.time(),
            finding_id=f.get("finding_id", ""),
            stage=f.get("stage", ""),
            param_changes=f.get("param_changes", {}),
            env_changes=env_changes,
            backup_path=backup_path,
            equity_at_apply=equity_now,
            status="monitoring",
        )
        records.append(asdict(record))
        new_records.append(record)

    _save_apply_log(records)

    status["applied_count"] = len(new_records)
    status["monitoring_count"] = status.get("monitoring_count", 0) + len(new_records)
    applied_ids = [r.finding_id for r in new_records]
    status["last_action"] = (
        f"applied {len(new_records)} findings (apply_all={int(apply_all_qualified)}) "
        f"→ monitoring 30min each (hot_reload)"
    )
    git_commit_and_push(
        f"auto: apply {len(new_records)} CF findings ({', '.join(applied_ids[:6])})"
    )

    return status


def get_apply_history() -> list[dict]:
    """Get auto-apply history for dashboard."""
    return _load_apply_log()[-20:]  # Last 20 records

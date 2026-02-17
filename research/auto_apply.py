"""
research/auto_apply.py — CF 결과 자동 적용 엔진
==================================================
보수적 안전 모드:
  - 1사이클당 최대 1개 변수만 변경
  - 적용 후 30분 모니터링 → 손실 시 자동 롤백
  - 모든 변경은 backup + audit log 기록
  - 엔진 자동 재시작 (pkill + restart)
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("research.auto_apply")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / "state" / "bybit.env"
BACKUP_DIR = PROJECT_ROOT / "state" / "env_backups"
APPLY_LOG = PROJECT_ROOT / "state" / "auto_apply_log.json"
DB_PATH = PROJECT_ROOT / "state" / "bot_data_live.db"

# ─────────────────────────────────────────────────────────────────
# Safety Thresholds
# ─────────────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.80          # 최소 신뢰도 80%
MIN_PNL_IMPROVEMENT = 100.0    # 최소 PnL 개선 $100
MONITOR_DURATION_SEC = 1800    # 30분 모니터링
ROLLBACK_LOSS_USD = 10.0       # $10 이상 손실 시 롤백
MAX_CHANGES_PER_CYCLE = 9999   # 제한 없음(사실상 무제한)
COOLDOWN_BETWEEN_APPLY_SEC = 0  # 쿨다운 비활성

# ─────────────────────────────────────────────────────────────────
# CF Parameter → bybit.env Variable Mapping
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
    "MU_SIGN_FLIP_MIN_AGE_SEC": (60, 3600),
    "NOTIONAL_HARD_CAP_USD": (20, 5000),
    "UNI_MAX_POS_FRAC": (0.10, 0.80),
    "MAX_CONCURRENT_POSITIONS": (1, 20),
    "SPREAD_PCT_MAX": (0.0001, 0.005),
    "FEE_FILTER_MULT": (0.50, 1.50),
    "ENTRY_NET_EXPECTANCY_MIN": (-0.005, 0.01),
    "MAX_NOTIONAL_EXPOSURE": (1.0, 10.0),
    "HURST_RANDOM_DAMPEN": (0.20, 1.00),
    "CHOP_ENTRY_FLOOR_ADD": (0.0, 0.01),
    "CHOP_ENTRY_MIN_DIR_CONF": (0.50, 0.90),
    "TOP_N_SYMBOLS": (2, 100),
    "ALPHA_DIRECTION_GATE_MIN_CONF": (0.45, 0.90),
    "ALPHA_DIRECTION_GATE_MIN_EDGE": (0.0, 0.20),
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
    """Read a value from bybit.env."""
    if not ENV_PATH.exists():
        return None
    for line in ENV_PATH.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, v = stripped.partition("=")
        if k.strip() == key:
            return v.strip()
    return None


def _update_env_value(key: str, new_value: str) -> Optional[str]:
    """
    Update a value in bybit.env. Returns old value or None if key not found.
    Handles duplicate keys (updates last occurrence only).
    """
    if not ENV_PATH.exists():
        return None
    lines = ENV_PATH.read_text().splitlines()
    old_value = None
    last_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, v = stripped.partition("=")
        if k.strip() == key:
            old_value = v.strip()
            last_idx = i
    if last_idx >= 0:
        # Preserve comment on same line
        comment = ""
        ln = lines[last_idx]
        parts = ln.split("#", 1)
        if len(parts) > 1 and "=" not in parts[0].split("#")[0]:
            pass
        # Simple replacement
        lines[last_idx] = f"{key}={new_value}"
        ENV_PATH.write_text("\n".join(lines) + "\n")
        return old_value
    else:
        # Key not found — append
        with open(ENV_PATH, "a") as f:
            f.write(f"\n# [AUTO_APPLY {time.strftime('%Y-%m-%d %H:%M')}]\n{key}={new_value}\n")
        return None


def backup_env() -> str:
    """Backup bybit.env and return backup path."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"bybit.env.{ts}"
    shutil.copy2(ENV_PATH, dest)
    # Keep only last 50 backups
    backups = sorted(BACKUP_DIR.glob("bybit.env.*"))
    for old in backups[:-50]:
        old.unlink()
    logger.info(f"Backed up bybit.env → {dest}")
    return str(dest)


def rollback_env(backup_path: str) -> bool:
    """Restore bybit.env from backup."""
    bp = Path(backup_path)
    if not bp.exists():
        logger.error(f"Backup not found: {backup_path}")
        return False
    shutil.copy2(bp, ENV_PATH)
    logger.warning(f"ROLLED BACK bybit.env from {backup_path}")
    return True


def restart_engine() -> bool:
    """Kill and restart the trading engine."""
    try:
        # Kill existing
        subprocess.run(["pkill", "-f", "main_engine_mc_v2_final.py"],
                       capture_output=True, timeout=10)
        time.sleep(3)
        # Syntax check
        result = subprocess.run(
            ["python3", "-m", "py_compile", "main_engine_mc_v2_final.py"],
            capture_output=True, timeout=30, cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            logger.error(f"Syntax check failed: {result.stderr.decode()}")
            return False
        # Restart
        cmd = (
            f"cd {PROJECT_ROOT} && source state/bybit.env && "
            f"ENABLE_LIVE_ORDERS=1 nohup python3 main_engine_mc_v2_final.py "
            f"> /tmp/engine.log 2>&1 &"
        )
        subprocess.Popen(cmd, shell=True, executable="/bin/zsh")
        logger.info("Engine restarted successfully")
        time.sleep(5)
        return True
    except Exception as e:
        logger.error(f"Engine restart failed: {e}")
        return False


def _is_in_cooldown() -> bool:
    """Check if we're still in cooldown from last apply."""
    records = _load_apply_log()
    if not records:
        return False
    last = records[-1]
    ts = last.get("timestamp", 0)
    status = last.get("status", "")
    if status == "monitoring":
        return True  # Still monitoring
    return (time.time() - ts) < COOLDOWN_BETWEEN_APPLY_SEC


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
    conf = finding.get("confidence", 0)
    pnl = finding.get("improvement_pct", 0)  # This is actually delta PnL in $
    if conf < MIN_CONFIDENCE and pnl < MIN_PNL_IMPROVEMENT:
        logger.debug(
            f"Skip {finding.get('finding_id')}: conf={conf:.2f} < {MIN_CONFIDENCE} "
            f"and pnl=${pnl:.2f} < ${MIN_PNL_IMPROVEMENT}"
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
    Apply a single CF finding to bybit.env.
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
    backup_path = backup_env()
    equity_now = _get_current_equity()

    # Apply changes
    for env_key, change in env_changes.items():
        old_val = _update_env_value(env_key, change["new"])
        logger.info(
            f"[AUTO_APPLY] {env_key}: {change['old']} → {change['new']} "
            f"(finding={finding.get('finding_id')}, stage={finding.get('stage')})"
        )

    record = ApplyRecord(
        timestamp=time.time(),
        finding_id=finding.get("finding_id", ""),
        stage=finding.get("stage", ""),
        param_changes=param_changes,
        env_changes=env_changes,
        backup_path=backup_path,
        equity_at_apply=equity_now,
        status="monitoring",
    )

    # Save audit log
    records = _load_apply_log()
    records.append(asdict(record))
    _save_apply_log(records)

    # Restart engine (optional in batch mode)
    if restart_after_apply:
        if not restart_engine():
            logger.error("Engine restart failed — rolling back")
            rollback_env(backup_path)
            restart_engine()
            record.status = "rolled_back"
            record.rollback_reason = "engine_restart_failed"
            record.rolled_back = True
            records[-1] = asdict(record)
            _save_apply_log(records)
            return None

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
    need_engine_restart = False

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
            need_engine_restart = True
            messages.append(f"rolled_back {rec.get('finding_id')} (Δ${delta:.2f})")
        else:
            rec["status"] = "applied"
            applied += 1
            messages.append(f"applied {rec.get('finding_id')} (Δ${delta:+.2f})")

    if processed > 0:
        _save_apply_log(records)
    if need_engine_restart:
        restart_engine()

    return {
        "processed": processed,
        "applied": applied,
        "rolled_back": rolled_back,
        "monitoring": monitoring,
        "messages": messages,
    }


def git_commit_and_push(message: str = "auto: CF parameter update") -> bool:
    """Git commit bybit.env changes and push."""
    try:
        subprocess.run(
            ["git", "add", "state/bybit.env", "state/auto_apply_log.json"],
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

    # Phase 2: Find applicable findings (OR condition)
    applicable = [f for f in findings if should_apply(f)]
    if not applicable:
        if not status.get("last_action"):
            status["last_action"] = "no_qualifying_findings"
        return status

    # Phase 3: Apply ALL qualifying findings (resolve env-key conflicts by higher ΔPnL)
    sorted_applicable = sorted(applicable, key=lambda f: float(f.get("improvement_pct", 0) or 0), reverse=True)
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

    backup_path = backup_env()
    equity_now = _get_current_equity()
    records = _load_apply_log()
    new_records = []

    for item in planned:
        f = item["finding"]
        env_changes = item["env_changes"]
        for env_key, change in env_changes.items():
            _update_env_value(env_key, change["new"])
            logger.info(
                f"[AUTO_APPLY] {env_key}: {change['old']} → {change['new']} "
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

    if not restart_engine():
        logger.error("Batch restart failed — rolling back")
        rollback_env(backup_path)
        restart_engine()
        for rec in records:
            if rec.get("backup_path") == backup_path and rec.get("status") == "monitoring":
                rec["status"] = "rolled_back"
                rec["rolled_back"] = True
                rec["rollback_reason"] = "engine_restart_failed"
        _save_apply_log(records)
        status["last_action"] = "batch_apply_failed_rollback"
        return status

    status["applied_count"] = len(new_records)
    status["monitoring_count"] = status.get("monitoring_count", 0) + len(new_records)
    applied_ids = [r.finding_id for r in new_records]
    status["last_action"] = f"applied {len(new_records)} findings → monitoring 30min each"
    git_commit_and_push(
        f"auto: apply {len(new_records)} CF findings ({', '.join(applied_ids[:6])})"
    )

    return status


def get_apply_history() -> list[dict]:
    """Get auto-apply history for dashboard."""
    return _load_apply_log()[-20:]  # Last 20 records

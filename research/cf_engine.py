"""
research/cf_engine.py — Counterfactual Simulation Engine
=========================================================
과거 거래 데이터를 기반으로 "만약 파라미터가 달랐다면?" 시뮬레이션을 수행.
9단계 파이프라인의 각 단계별 변수를 독립적으로 변경하며
WR, R:R, PnL, Sharpe 등의 지표 변화를 측정한다.

Architecture:
  TradeLoader  →  CFEngine  →  StageSimulator(×9)  →  ResultStore
                     ↑                                       ↓
               ParameterGrid                         FindingDetector
                                                          ↓
                                                    DashboardBroadcast
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import time
import ast
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from core.runtime_config_store import get_runtime_config_values

logger = logging.getLogger("research.cf_engine")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DL_SCRIPT = PROJECT_ROOT / "scripts" / "mtf_image_dl_cf.py"
DEFAULT_DL_REPORT = PROJECT_ROOT / "state" / "mtf_image_dl_cf_report.json"
DEFAULT_DL_SCORES = PROJECT_ROOT / "state" / "mtf_image_trade_scores.json"
DEFAULT_DL_MODEL = PROJECT_ROOT / "state" / "mtf_image_model.pt"
DEFAULT_DL_CACHE = PROJECT_ROOT / "state" / "mtf_ohlcv_cache"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        out = float(str(raw).strip())
        if not math.isfinite(out):
            return float(default)
        return float(out)
    except Exception:
        return float(default)


_ENV_FILE_CACHE: dict[str, str] | None = None
_ENV_FILE_CACHE_MTIME: float | None = None


def _env_file_values() -> dict[str, str]:
    global _ENV_FILE_CACHE, _ENV_FILE_CACHE_MTIME
    out: dict[str, str] = {}
    env_path = PROJECT_ROOT / "state" / "bybit.env"
    current_mtime: float | None = None
    try:
        if env_path.exists():
            current_mtime = float(env_path.stat().st_mtime)
    except Exception:
        current_mtime = None

    # Reuse cache only when file mtime is unchanged.
    if _ENV_FILE_CACHE is not None and _ENV_FILE_CACHE_MTIME == current_mtime:
        return _ENV_FILE_CACHE

    try:
        if env_path.exists():
            for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = str(raw or "").strip()
                if (not s) or s.startswith("#") or ("=" not in s):
                    continue
                k, v = s.split("=", 1)
                out[str(k).strip()] = str(v).strip()
    except Exception:
        out = {}
    _ENV_FILE_CACHE = out
    _ENV_FILE_CACHE_MTIME = current_mtime
    return out


def _env_runtime_raw(key: str) -> str | None:
    k = str(key or "").strip().upper()
    if not k:
        return None
    raw = os.environ.get(k)
    if raw is not None:
        return str(raw).strip()
    try:
        vals = get_runtime_config_values(str(PROJECT_ROOT / "state" / "bot_data_live.db"), keys=[k])
        v = vals.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    except Exception:
        pass
    return _env_file_values().get(k)


def _norm_csv_tokens(v: Any) -> str:
    toks = [str(x).strip().lower() for x in str(v or "").replace(";", ",").split(",") if str(x).strip()]
    if not toks:
        return ""
    # preserve set semantics for blocklist/hourlist style params
    return ",".join(sorted(set(toks)))


def _parse_quantiles(raw: Any, default: list[float], *, lo: float = 0.01, hi: float = 0.99) -> list[float]:
    out: list[float] = []
    txt = str(raw or "")
    for tok in txt.replace(";", ",").split(","):
        s = str(tok or "").strip()
        if not s:
            continue
        try:
            q = float(s)
        except Exception:
            continue
        q = max(float(lo), min(float(hi), float(q)))
        if q not in out:
            out.append(q)
    if not out:
        out = [float(x) for x in list(default or [])]
    return out


def _quantile_candidates(
    values: list[float],
    quantiles: list[float],
    *,
    lo: float | None = None,
    hi: float | None = None,
    as_int: bool = False,
    relax_mult: float = 1.0,
) -> list[Any]:
    if not values:
        return []
    arr = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=np.float64)
    if arr.size <= 0:
        return []
    out: list[Any] = []
    for q in quantiles:
        try:
            v = float(np.quantile(arr, float(q)))
        except Exception:
            continue
        v *= float(relax_mult)
        if lo is not None:
            v = max(float(lo), v)
        if hi is not None:
            v = min(float(hi), v)
        val: Any = int(round(v)) if as_int else float(v)
        if val not in out:
            out.append(val)
    return out


def _robust_median(vals: list[float], default: float = 0.0) -> float:
    arr = np.asarray([float(x) for x in vals if math.isfinite(float(x))], dtype=np.float64)
    if arr.size <= 0:
        return float(default)
    return float(np.median(arr))


# CF param -> runtime env key candidates (priority order).
# Used to anchor each stage baseline to current bybit.env values.
CF_PARAM_ENV_KEYS: dict[str, list[str]] = {
    "max_leverage": ["MAX_LEVERAGE"],
    "regime_max_bull": ["LEVERAGE_REGIME_MAX_BULL"],
    "regime_max_bear": ["LEVERAGE_REGIME_MAX_BEAR"],
    "regime_max_chop": ["LEVERAGE_REGIME_MAX_CHOP"],
    "regime_max_volatile": ["LEVERAGE_REGIME_MAX_VOLATILE"],
    "tp_pct": ["MC_TP_BASE_ROE", "DEFAULT_TP_PCT"],
    "sl_pct": ["MC_SL_BASE_ROE", "DEFAULT_SL_PCT"],
    "max_hold_sec": ["POLICY_MAX_HOLD_SEC"],
    "min_hold_sec": ["EXIT_MIN_HOLD_SEC"],
    "min_hold_sec_bull": ["EXIT_MIN_HOLD_SEC_BULL"],
    "min_hold_sec_chop": ["EXIT_MIN_HOLD_SEC_CHOP"],
    "min_hold_sec_bear": ["EXIT_MIN_HOLD_SEC_BEAR"],
    "min_confidence": ["UNI_MIN_CONFIDENCE"],
    "min_dir_conf": ["ALPHA_DIRECTION_MIN_CONFIDENCE"],
    "min_ev": ["UNIFIED_ENTRY_FLOOR"],
    "min_dir_conf_for_entry": ["ALPHA_DIRECTION_MIN_CONFIDENCE"],
    "max_vpin": ["UNI_MAX_VPIN_HARD"],
    "scope": ["VOLATILITY_GATE_SCOPE"],
    "chop_min_sigma": ["CHOP_VOL_GATE_MIN_SIGMA"],
    "chop_max_sigma": ["CHOP_VOL_GATE_MAX_SIGMA"],
    "chop_max_vpin": ["CHOP_VPIN_MAX"],
    "chop_min_dir_conf": ["CHOP_ENTRY_MIN_DIR_CONF"],
    "chop_min_abs_mu_alpha": ["CHOP_ENTRY_MIN_MU_ALPHA"],
    "chop_max_hold_sec": ["TARGET_HOLD_SEC_MAX_CHOP"],
    "notional_hard_cap": ["NOTIONAL_HARD_CAP_USD"],
    "max_pos_frac": ["UNI_MAX_POS_FRAC"],
    "spread_pct_max": ["SPREAD_PCT_MAX"],
    "net_expectancy_min": ["ENTRY_NET_EXPECTANCY_MIN"],
    "both_ev_neg_net_floor": ["ENTRY_BOTH_EV_NEG_NET_FLOOR"],
    "gross_ev_min": ["ENTRY_GROSS_EV_MIN"],
    "max_exposure": ["MAX_NOTIONAL_EXPOSURE", "LIVE_MAX_NOTIONAL_EXPOSURE"],
    "max_concurrent": ["MAX_CONCURRENT_POSITIONS"],
    "hurst_dampen": ["HURST_RANDOM_DAMPEN"],
    "fee_filter_mult": ["FEE_FILTER_MULT"],
    "chop_entry_floor_add": ["CHOP_ENTRY_FLOOR_ADD"],
    "chop_entry_min_dir_conf": ["CHOP_ENTRY_MIN_DIR_CONF"],
    "block_mu_sign_flip_before_sec": ["MU_SIGN_FLIP_MIN_AGE_SEC"],
    "mu_sign_flip_min_magnitude": ["MU_SIGN_FLIP_MIN_MAGNITUDE"],
    "mu_sign_flip_confirm_ticks": ["MU_SIGN_FLIP_CONFIRM_TICKS"],
    "top_n_symbols": ["TOP_N_SYMBOLS"],
    "dir_gate_min_conf": ["ALPHA_DIRECTION_GATE_MIN_CONF"],
    "dir_gate_min_edge": ["ALPHA_DIRECTION_GATE_MIN_EDGE"],
    "dir_gate_min_side_prob": ["ALPHA_DIRECTION_GATE_MIN_SIDE_PROB"],
    "dir_gate_confirm_ticks": ["ALPHA_DIRECTION_GATE_CONFIRM_TICKS"],
    "dir_gate_confirm_ticks_chop": ["ALPHA_DIRECTION_GATE_CONFIRM_TICKS_CHOP"],
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
    "sq_time_window_hours": ["SYMBOL_QUALITY_TIME_WINDOW_HOURS"],
    "sq_time_weight": ["SYMBOL_QUALITY_TIME_WEIGHT"],
    "hybrid_exit_confirm_shock": ["HYBRID_EXIT_CONFIRM_TICKS_SHOCK"],
    "hybrid_exit_confirm_normal": ["HYBRID_EXIT_CONFIRM_TICKS_NORMAL"],
    "hybrid_exit_confirm_noise": ["HYBRID_EXIT_CONFIRM_TICKS_NOISE"],
    "hybrid_lev_sweep_min": ["HYBRID_LEV_MIN"],
    "hybrid_lev_sweep_max": ["HYBRID_LEV_MAX"],
    "mc_hybrid_n_paths": ["MC_HYBRID_N_PATHS"],
    "mc_hybrid_horizon_steps": ["MC_HYBRID_HORIZON_STEPS"],
    "hybrid_cash_penalty": ["HYBRID_CASH_PENALTY"],
    "exp_vals_min_diff": ["EXP_VALS_MIN_DIFF"],
    "exp_vals_dominance_ratio": ["EXP_VALS_DOMINANCE"],
    "hybrid_beam_width": ["HYBRID_BEAM_WIDTH"],
    "lsm_poly_degree": ["LSM_POLY_DEGREE"],
    "hybrid_tp_base": ["HYBRID_TP_PCT_BASE", "HYBRID_TP_PCT_BULL"],
    "hybrid_sl_base": ["HYBRID_SL_PCT_BASE", "HYBRID_SL_PCT_BULL"],
    "hybrid_tp_shock_mult": ["HYBRID_TP_MULT_SHOCK"],
    "hybrid_tp_noise_mult": ["HYBRID_TP_MULT_NOISE"],
    "mtf_dl_entry_min_side_prob": ["MTF_DL_ENTRY_MIN_SIDE_PROB"],
    "mtf_dl_entry_min_win_prob": ["MTF_DL_ENTRY_MIN_WIN_PROB"],
}


def _baseline_candidate_from_env(param_name: str, candidates: list[Any]) -> Any:
    vals = list(candidates or [])
    if not vals:
        return None

    env_keys = CF_PARAM_ENV_KEYS.get(str(param_name), [])
    raw: str | None = None
    for k in env_keys:
        raw = _env_runtime_raw(k)
        if raw not in (None, ""):
            break
    if raw in (None, ""):
        return vals[0]

    first = vals[0]
    # bool
    if isinstance(first, bool):
        rv = str(raw).strip().lower() in ("1", "true", "yes", "on")
        for c in vals:
            if bool(c) == rv:
                return c
        return vals[0]

    # numeric
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        try:
            rv = float(str(raw).strip())
        except Exception:
            return vals[0]
        best = vals[0]
        best_gap = float("inf")
        for c in vals:
            try:
                gap = abs(float(c) - rv)
            except Exception:
                continue
            if gap < best_gap:
                best_gap = gap
                best = c
        return best

    # string-like / csv-like
    raw_s = str(raw).strip()
    for c in vals:
        if str(c).strip() == raw_s:
            return c
    raw_norm = _norm_csv_tokens(raw_s)
    if raw_norm:
        for c in vals:
            if _norm_csv_tokens(c) == raw_norm:
                return c
    return vals[0]


def _baseline_combo_from_env(param_names: list[str], param_values: list[list[Any]]) -> tuple[Any, ...]:
    combo: list[Any] = []
    for p, vals in zip(param_names, param_values):
        combo.append(_baseline_candidate_from_env(str(p), list(vals or [])))
    return tuple(combo)

# ─────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Closed trade round-trip (ENTER + EXIT paired)."""
    trade_uid: str
    symbol: str
    side: str  # LONG / SHORT
    entry_price: float
    exit_price: float
    qty: float
    notional: float
    leverage: float
    entry_ev: float
    entry_confidence: float
    realized_pnl: float
    roe: float
    hold_duration_sec: float
    regime: str
    exit_reason: str
    # Signal features at entry
    mu_alpha: float
    dir_conf: float
    vpin: float
    hurst: float
    sigma: float
    entry_quality: float
    leverage_signal: float
    # Raw JSON for detailed re-analysis
    raw: dict = field(default_factory=dict, repr=False)
    timestamp_ms: int = 0

    @property
    def direction(self) -> int:
        return 1 if self.side == "LONG" else -1


@dataclass
class CFResult:
    """Result of a single counterfactual scenario."""
    scenario_id: str
    stage: str  # Pipeline stage name
    param_changes: dict  # {param_name: new_value}
    baseline: dict  # {metric: value}
    simulated: dict  # {metric: value}
    delta: dict  # {metric: change}
    n_trades: int
    significance: float  # 0-1 score of how meaningful this change is
    oos: dict = field(default_factory=dict)  # Walk-forward train/test validation summary
    recommendation: str = ""
    details: str = ""


@dataclass
class Finding:
    """A statistically significant discovery from CF simulation."""
    finding_id: str
    timestamp: float
    stage: str
    title: str
    description: str
    improvement_pct: float  # Estimated PnL improvement %
    confidence: float  # Statistical confidence
    param_changes: dict
    baseline_metrics: dict
    improved_metrics: dict
    recommendation: str
    oos_pass: bool = False
    oos_folds: int = 0
    oos_pass_rate: float = 0.0
    oos_train_pnl_delta: float = 0.0
    oos_test_pnl_delta: float = 0.0
    oos_penalty_factor: float = 1.0
    applied: bool = False


# ─────────────────────────────────────────────────────────────────
# Trade Loader
# ─────────────────────────────────────────────────────────────────

class TradeLoader:
    """Load closed trade round-trips from SQLite."""

    def __init__(self, db_path: str = "state/bot_data_live.db"):
        self.db_path = db_path

    @staticmethod
    def _parse_excluded_symbols(raw: Any) -> set[str]:
        txt = ""
        if raw is None:
            txt = (
                os.environ.get("RESEARCH_EXCLUDE_SYMBOLS")
                or os.environ.get("AUTO_REVAL_EXCLUDE_SYMBOLS")
                or ""
            )
        elif isinstance(raw, (list, tuple, set)):
            txt = ",".join(str(x) for x in raw if str(x).strip())
        else:
            txt = str(raw or "")
        out: set[str] = set()
        for tok in txt.replace(";", ",").split(","):
            sym = str(tok or "").strip().upper()
            if sym:
                out.add(sym)
        return out

    def load_trades(self, limit: int = 0, since_ms: int = 0, exclude_symbols: Any = None) -> list[Trade]:
        """Load paired ENTER+EXIT trades."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM trades ORDER BY id ASC"
        rows = conn.execute(query).fetchall()
        conn.close()

        excluded = self._parse_excluded_symbols(exclude_symbols)

        # Pair ENTER/EXIT by entry_link_id
        enters: dict[str, dict] = {}
        trades: list[Trade] = []
        skipped_excluded = 0
        for r in rows:
            d = dict(r)
            action = (d.get("action") or "").upper()
            link_id = d.get("entry_link_id") or d.get("entry_id") or ""
            if not link_id:
                continue
            if action == "ENTER":
                enters[link_id] = d
            elif action == "EXIT" and link_id in enters:
                entry = enters.pop(link_id)
                raw = {}
                for rd in [entry, d]:
                    raw_str = rd.get("raw_data")
                    if raw_str:
                        try:
                            raw.update(json.loads(raw_str))
                        except Exception:
                            pass
                ts = d.get("timestamp_ms") or 0
                if since_ms and ts < since_ms:
                    continue
                sigma_val = d.get("sigma")
                if sigma_val is None:
                    sigma_val = entry.get("sigma")
                try:
                    sigma_num = float(sigma_val) if sigma_val is not None else 0.5
                except Exception:
                    sigma_num = 0.5
                if sigma_num <= 0:
                    sigma_num = 0.5
                t = Trade(
                    trade_uid=d.get("trade_uid") or link_id,
                    symbol=d.get("symbol") or "",
                    side=entry.get("side") or "LONG",
                    entry_price=float(entry.get("fill_price") or 0),
                    exit_price=float(d.get("fill_price") or 0),
                    qty=float(entry.get("qty") or 0),
                    notional=float(entry.get("notional") or 0),
                    leverage=float(raw.get("leverage") or raw.get("entry_leverage") or 1),
                    entry_ev=float(entry.get("entry_ev") or 0),
                    entry_confidence=float(entry.get("entry_confidence") or 0.5),
                    realized_pnl=float(d.get("realized_pnl") or 0),
                    roe=float(d.get("roe") or 0),
                    hold_duration_sec=float(d.get("hold_duration_sec") or 0),
                    regime=d.get("regime") or entry.get("regime") or "",
                    exit_reason=d.get("entry_reason") or "",  # EXIT's reason field
                    mu_alpha=float(d.get("pred_mu_alpha") or entry.get("pred_mu_alpha") or 0),
                    dir_conf=float(d.get("pred_mu_dir_conf") or entry.get("pred_mu_dir_conf") or 0.5),
                    vpin=float(d.get("alpha_vpin") or entry.get("alpha_vpin") or 0),
                    hurst=float(d.get("alpha_hurst") or entry.get("alpha_hurst") or 0.5),
                    sigma=float(sigma_num),
                    entry_quality=float(entry.get("entry_quality_score") or 0),
                    leverage_signal=float(entry.get("leverage_signal_score") or 0),
                    raw=raw,
                    timestamp_ms=ts,
                )
                if excluded and str(t.symbol or "").strip().upper() in excluded:
                    skipped_excluded += 1
                    continue
                trades.append(t)

        if limit > 0:
            trades = trades[-limit:]
        logger.info(
            f"Loaded {len(trades)} round-trip trades from {self.db_path}"
            + (
                f" (excluded_symbols={len(excluded)}, skipped={skipped_excluded})"
                if excluded
                else ""
            )
        )
        return trades


# ─────────────────────────────────────────────────────────────────
# Metrics Calculator
# ─────────────────────────────────────────────────────────────────

def _trade_notional(trade: Trade) -> float:
    try:
        n = float(trade.notional or 0.0)
    except Exception:
        n = 0.0
    if n > 0:
        return float(n)
    try:
        qty = abs(float(trade.qty or 0.0))
    except Exception:
        qty = 0.0
    try:
        px = float(trade.entry_price or trade.exit_price or 0.0)
    except Exception:
        px = 0.0
    if qty > 0 and px > 0:
        return float(qty * px)
    return 0.0


def _trade_liquidity_proxy(trade: Trade) -> float:
    raw = trade.raw if isinstance(trade.raw, dict) else {}
    for k in ("liquidity_score", "liq_score", "book_liquidity", "orderbook_liquidity"):
        try:
            v = raw.get(k)
            if v is None:
                continue
            vv = float(v)
            if math.isfinite(vv):
                return float(max(0.0, min(1.0, vv)))
        except Exception:
            continue
    try:
        sigma = abs(float(trade.sigma or 0.0))
    except Exception:
        sigma = 0.0
    try:
        vpin = abs(float(trade.vpin or 0.0))
    except Exception:
        vpin = 0.0
    spread_pct = 0.0
    for k in ("spread_pct", "spread"):
        try:
            v = raw.get(k)
            if v is None:
                continue
            spread_pct = max(0.0, float(v))
            if spread_pct > 0:
                break
        except Exception:
            continue
    if spread_pct <= 0:
        try:
            sb = raw.get("spread_bps")
            if sb is not None:
                spread_pct = max(0.0, float(sb) / 10_000.0)
        except Exception:
            spread_pct = 0.0
    sigma_pen = min(1.0, sigma * 2.0)
    vpin_pen = min(1.0, max(0.0, vpin - 0.45) * 1.8)
    spread_pen = min(1.0, spread_pct * 800.0)
    liq = 1.0 - (0.45 * sigma_pen + 0.40 * vpin_pen + 0.15 * spread_pen)
    return float(max(0.0, min(1.0, liq)))


def _execution_penalty_usd(trade: Trade) -> float:
    if not _env_bool("CF_EXECUTION_PENALTY_ENABLED", True):
        return 0.0
    notional = _trade_notional(trade)
    if notional <= 0:
        return 0.0
    try:
        sigma = abs(float(trade.sigma or 0.0))
    except Exception:
        sigma = 0.0
    try:
        vpin = max(0.0, float(trade.vpin or 0.0))
    except Exception:
        vpin = 0.0
    try:
        leverage = max(1.0, float(trade.leverage or 1.0))
    except Exception:
        leverage = 1.0
    liq = _trade_liquidity_proxy(trade)

    base_side_bps = _env_float("CF_EXEC_BASE_BPS_SIDE", 1.2)
    sigma_bps_k = _env_float("CF_EXEC_SIGMA_BPS_K", 2.0)
    vpin_bps_k = _env_float("CF_EXEC_VPIN_BPS_K", 6.0)
    lev_bps_k = _env_float("CF_EXEC_LEV_BPS_K", 0.3)
    liq_bps_k = _env_float("CF_EXEC_LIQ_BPS_K", 8.0)
    penalty_mult = max(0.0, _env_float("CF_EXEC_PENALTY_MULT", 1.0))

    side_bps = (
        float(base_side_bps)
        + float(sigma_bps_k) * sigma
        + float(vpin_bps_k) * max(0.0, vpin - 0.5)
        + float(lev_bps_k) * max(0.0, leverage - 1.0)
        + float(liq_bps_k) * max(0.0, 1.0 - liq)
    )
    side_bps = max(0.0, side_bps)
    roundtrip_bps = 2.0 * side_bps
    penalty = notional * (roundtrip_bps / 10_000.0) * penalty_mult
    return float(max(0.0, min(notional * 0.10, penalty)))


def compute_metrics(trades: list[Trade]) -> dict:
    """Compute aggregate performance metrics with conservative execution penalties."""
    if not trades:
        return {
            "n": 0,
            "pnl": 0,
            "pnl_raw": 0,
            "execution_penalty": 0,
            "wr": 0,
            "rr": 0,
            "sharpe": 0,
            "avg_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_dd": 0,
            "pf": 0,
            "edge": 0,
        }
    n = len(trades)
    raw_pnls = np.array([float(t.realized_pnl or 0.0) for t in trades], dtype=np.float64)
    penalties = np.array([_execution_penalty_usd(t) for t in trades], dtype=np.float64)
    pnls = raw_pnls - penalties
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    total_pnl = float(pnls.sum())
    total_pnl_raw = float(raw_pnls.sum())
    total_penalty = float(penalties.sum())
    avg_pnl = float(pnls.mean())
    wr = float(len(wins) / n) if n > 0 else 0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.001
    rr = avg_win / max(avg_loss, 1e-8)
    gross_win = float(wins.sum()) if len(wins) > 0 else 0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.001
    pf = gross_win / max(gross_loss, 1e-8)
    # Sharpe (annualized from per-trade returns)
    std = float(pnls.std()) if n > 1 else 1e-8
    sharpe = (avg_pnl / max(std, 1e-8)) * math.sqrt(min(n, 365 * 24))
    # Max drawdown
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min())
    # Edge = WR - BEP_WR
    bep_wr = 1.0 / (1.0 + rr) if rr > 0 else 1.0
    edge = wr - bep_wr

    return {
        "n": n,
        "pnl": round(total_pnl, 4),
        "pnl_raw": round(total_pnl_raw, 4),
        "execution_penalty": round(total_penalty, 4),
        "wr": round(wr, 4),
        "rr": round(rr, 4),
        "sharpe": round(sharpe, 4),
        "avg_pnl": round(avg_pnl, 6),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "max_dd": round(max_dd, 4),
        "pf": round(pf, 4),
        "edge": round(edge, 4),
    }


def compute_metrics_by_regime(trades: list[Trade]) -> dict[str, dict]:
    """Metrics grouped by regime."""
    regimes: dict[str, list[Trade]] = {}
    for t in trades:
        r = (t.regime or "unknown").lower()
        regimes.setdefault(r, []).append(t)
    return {regime: compute_metrics(tlist) for regime, tlist in regimes.items()}


# ─────────────────────────────────────────────────────────────────
# Pipeline Stage Simulators
# ─────────────────────────────────────────────────────────────────

class StageSimulator:
    """Base class for pipeline-stage counterfactual simulation."""
    stage_name: str = "base"
    description: str = ""
    max_combos_hint: int = 120

    def get_param_grid(self) -> dict[str, list]:
        """Return {param_name: [candidate_values]} to sweep."""
        raise NotImplementedError

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        """Return modified trades reflecting the counterfactual."""
        raise NotImplementedError


class LeverageSimulator(StageSimulator):
    """Stage 7: Leverage decision CF — recalculate PnL with different leverage."""
    stage_name = "leverage"
    description = "레버리지 결정: MAX_LEVERAGE, 레짐별 cap, VPIN/Sigma 감쇠"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "max_leverage": [3, 5, 8, 10, 15, 20, 30, 50],
            "regime_max_bull": [8, 12, 15, 20],
            "regime_max_chop": [3, 5, 8, 10],
            "regime_max_bear": [5, 8, 12],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        max_lev = params.get("max_leverage")
        regime_caps = {
            "bull": params.get("regime_max_bull", 20),
            "bear": params.get("regime_max_bear", 12),
            "chop": params.get("regime_max_chop", 8),
            "volatile": params.get("regime_max_volatile", 5),
        }
        result = []
        for t in trades:
            t2 = copy.copy(t)
            # Effective leverage cap
            regime = (t.regime or "").lower()
            regime_cap = regime_caps.get(regime, max_lev or 20)
            effective_cap = min(max_lev or 50, regime_cap)
            new_lev = min(t.leverage, effective_cap)
            if t.leverage > 0:
                lev_ratio = new_lev / t.leverage
            else:
                lev_ratio = 1.0
            # PnL scales proportionally with leverage
            t2.leverage = new_lev
            t2.realized_pnl = t.realized_pnl * lev_ratio
            t2.roe = t.roe * lev_ratio
            result.append(t2)
        return result


class TPSLSimulator(StageSimulator):
    """Stage 5: TP/SL target CF — check if TP/SL would change exit outcome."""
    stage_name = "tp_sl"
    description = "TP/SL 타겟: DEFAULT_TP_PCT, DEFAULT_SL_PCT, 레짐별 스케일"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "tp_pct": [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040],
            "sl_pct": [0.005, 0.008, 0.010, 0.015, 0.018, 0.020, 0.025],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        tp_pct = params.get("tp_pct", 0.025)
        sl_pct = params.get("sl_pct", 0.018)
        result = []
        for t in trades:
            t2 = copy.copy(t)
            if t.entry_price <= 0:
                result.append(t2)
                continue
            # Calculate raw price movement
            raw_roe = (t.exit_price / t.entry_price - 1) * t.direction
            # Would TP or SL have been hit?
            # We know the exit ROE, approximate if tighter TP/SL changes outcome
            lev = max(t.leverage, 1)
            # TP threshold in price terms
            tp_threshold = tp_pct / lev  # lever-adjusted
            sl_threshold = sl_pct / lev

            if raw_roe >= tp_threshold:
                # Would have hit TP — cap PnL at TP
                t2.realized_pnl = tp_pct * t.notional / lev
                t2.roe = tp_pct
            elif raw_roe <= -sl_threshold:
                # Would have hit SL — cap loss at SL
                t2.realized_pnl = -sl_pct * t.notional / lev
                t2.roe = -sl_pct
            # else: exit happened before TP/SL, keep original PnL
            result.append(t2)
        return result


class HoldDurationSimulator(StageSimulator):
    """Stage 9: Hold duration CF — filter out long-hold or short-hold trades."""
    stage_name = "hold_duration"
    description = "보유 시간: POLICY_MAX_HOLD_SEC, EXIT_MIN_HOLD_SEC"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "max_hold_sec": [300, 600, 900, 1200, 1800, 3600, 7200],
            "min_hold_sec": [30, 60, 120, 300, 600],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        max_hold = params.get("max_hold_sec", 7200)
        min_hold = params.get("min_hold_sec", 60)
        # Trades exceeding max_hold get 50% worse PnL (proxy for stale positions)
        result = []
        for t in trades:
            t2 = copy.copy(t)
            if t.hold_duration_sec > max_hold:
                # Assume earlier exit would have captured 60% of current PnL if win, 80% of loss
                if t.realized_pnl > 0:
                    t2.realized_pnl = t.realized_pnl * 0.6
                else:
                    t2.realized_pnl = t.realized_pnl * 0.8  # Less loss from earlier stop
            result.append(t2)
        return result


class EntryFilterSimulator(StageSimulator):
    """Stage 8: Entry filter CF — would blocking low-quality trades improve?"""
    stage_name = "entry_filter"
    description = "진입 필터: min_confidence, min_dir_conf, min_entry_quality, min_ev"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "min_confidence": [0.50, 0.55, 0.58, 0.60, 0.65, 0.70],
            "min_dir_conf": [0.50, 0.53, 0.55, 0.58, 0.60, 0.65],
            "min_entry_quality": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "min_ev": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        min_conf = params.get("min_confidence", 0.50)
        min_dir = params.get("min_dir_conf", 0.50)
        min_eq = params.get("min_entry_quality", 0.0)
        min_ev = params.get("min_ev", 0.0)
        result = []
        for t in trades:
            # Would this trade pass the filter?
            if t.entry_confidence >= min_conf and t.dir_conf >= min_dir \
               and t.entry_quality >= min_eq and t.entry_ev >= min_ev:
                result.append(t)
            # Blocked trades are simply excluded from result
        return result


class DirectionSimulator(StageSimulator):
    """Stage 7(DirectionModel): What if we inverted direction for certain conditions?"""
    stage_name = "direction"
    description = "방향 결정: dir_conf 기반 필터, 레짐별 방향 편향"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "chop_prefer_short": [True, False],
            "min_dir_conf_for_entry": [0.50, 0.55, 0.58, 0.60, 0.65],
            "mu_alpha_sign_override": [True, False],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        chop_short = params.get("chop_prefer_short", False)
        min_dir = params.get("min_dir_conf_for_entry", 0.50)
        result = []
        for t in trades:
            t2 = copy.copy(t)
            # Filter low-confidence directions
            if t.dir_conf < min_dir:
                continue  # Would not enter
            # Chop regime direction adjustment
            if chop_short and (t.regime or "").lower() == "chop" and t.side == "LONG":
                # Reverse the trade
                t2.realized_pnl = -t.realized_pnl
                t2.roe = -t.roe
                t2.side = "SHORT"
            result.append(t2)
        return result


class VPINFilterSimulator(StageSimulator):
    """VPIN-based entry filter: what if we blocked high-VPIN entries?"""
    stage_name = "vpin_filter"
    description = "VPIN 필터: toxic flow 감지, 진입 차단 임계치"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "max_vpin": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        max_vpin = params.get("max_vpin", 0.80)
        return [t for t in trades if t.vpin <= max_vpin]


class VolatilityGateSimulator(StageSimulator):
    """
    Sigma/VPIN 2축 진입 게이트 + chop 구간 빠른 청산 proxy.
    목적: chop 내 국소적 한방향 움직임만 선별해 짧게 수확.
    """

    stage_name = "volatility_gate"
    description = "변동성 게이트: chop sigma/vpin/dir_conf/mu_alpha 강도 기반 진입 + 빠른 청산"
    max_combos_hint = 240

    def get_param_grid(self) -> dict[str, list]:
        return {
            "scope": ["chop_only", "all_regimes"],
            "chop_min_sigma": [0.10, 0.20, 0.35, 0.50, 0.80],
            "chop_max_sigma": [0.80, 1.20, 1.80, 2.50, 4.00],
            "chop_max_vpin": [0.20, 0.30, 0.40, 0.50, 0.65],
            "chop_min_dir_conf": [0.52, 0.56, 0.60, 0.64, 0.68],
            "chop_min_abs_mu_alpha": [0.0, 5.0, 10.0, 20.0, 40.0],
            "chop_max_hold_sec": [180, 300, 450, 600, 900],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        scope = str(params.get("scope", "chop_only") or "chop_only").strip().lower()
        min_sigma = float(params.get("chop_min_sigma", 0.20))
        max_sigma = float(params.get("chop_max_sigma", 2.50))
        max_vpin = float(params.get("chop_max_vpin", 0.40))
        min_dir_conf = float(params.get("chop_min_dir_conf", 0.56))
        min_abs_mu = float(params.get("chop_min_abs_mu_alpha", 5.0))
        max_hold_sec = float(params.get("chop_max_hold_sec", 600))
        if max_sigma < min_sigma:
            min_sigma, max_sigma = max_sigma, min_sigma

        out: list[Trade] = []
        for t in trades:
            regime = str(t.regime or "").strip().lower()
            apply_gate = bool(scope != "chop_only" or regime == "chop")
            if not apply_gate:
                out.append(t)
                continue

            sigma = float(t.sigma or 0.0)
            vpin = float(t.vpin or 0.0)
            dir_conf = float(t.dir_conf or 0.0)
            mu_abs = abs(float(t.mu_alpha or 0.0))
            if sigma < min_sigma or sigma > max_sigma:
                continue
            if vpin > max_vpin:
                continue
            if dir_conf < min_dir_conf:
                continue
            if mu_abs < min_abs_mu:
                continue

            t2 = copy.copy(t)
            if float(t2.hold_duration_sec or 0.0) > max_hold_sec:
                if float(t2.realized_pnl or 0.0) > 0:
                    t2.realized_pnl = float(t2.realized_pnl) * 0.88
                else:
                    t2.realized_pnl = float(t2.realized_pnl) * 0.60
            out.append(t2)
        return out


class MTFImageDLGateSimulator(StageSimulator):
    """
    Multi-timeframe image DL gate.

    - 매 사이클에서 DL 점수 파일을 갱신(옵션)하고
    - 분위수(quantile) 기준으로 진입 샘플을 필터링하는 CF stage.
    """

    stage_name = "mtf_image_dl_gate"
    description = "멀티TF 이미지 DL 게이트: 확률 분위수 기반 진입 필터(global/chop)"
    max_combos_hint = 60

    def __init__(self):
        self.script_path = Path(os.environ.get("CF_MTF_DL_SCRIPT_PATH", str(DEFAULT_DL_SCRIPT)))
        self.db_path = os.environ.get("CF_MTF_DL_DB_PATH", "state/bot_data_live.db")
        self.report_path = Path(os.environ.get("CF_MTF_DL_REPORT_PATH", str(DEFAULT_DL_REPORT)))
        self.scores_path = Path(os.environ.get("CF_MTF_DL_SCORES_PATH", str(DEFAULT_DL_SCORES)))
        self.model_path = Path(os.environ.get("CF_MTF_DL_MODEL_PATH", str(DEFAULT_DL_MODEL)))
        self.cache_dir = Path(os.environ.get("CF_MTF_DL_CACHE_DIR", str(DEFAULT_DL_CACHE)))
        self._score_map: dict[str, float] = {}

    def _python_exec(self) -> str:
        py = PROJECT_ROOT / ".venv" / "bin" / "python"
        if py.exists():
            return str(py)
        return "python3"

    def _refresh_scores(self) -> None:
        if not _env_bool("CF_MTF_DL_ENABLED", True):
            return
        refresh_each = _env_bool("CF_MTF_DL_REFRESH_EACH_CYCLE", True)
        if (not refresh_each) and self.scores_path.exists():
            return
        if not self.script_path.exists():
            logger.warning(f"[DL_GATE] script missing: {self.script_path}")
            return

        max_symbols = max(1, _env_int("CF_MTF_DL_MAX_SYMBOLS", 12))
        max_trades = max(200, _env_int("CF_MTF_DL_MAX_TRADES", 2200))
        epochs = max(1, _env_int("CF_MTF_DL_EPOCHS", 8))
        batch_size = max(16, _env_int("CF_MTF_DL_BATCH_SIZE", 96))
        min_n = max(20, _env_int("CF_MTF_DL_MIN_N", 100))
        min_hold_sec = max(0.0, _env_float("CF_MTF_DL_MIN_HOLD_SEC", 30.0))
        train_ratio = min(0.95, max(0.50, _env_float("CF_MTF_DL_TRAIN_RATIO", 0.75)))
        timeout_sec = max(60, _env_int("CF_MTF_DL_TIMEOUT_SEC", 1800))

        cmd = [
            self._python_exec(),
            str(self.script_path),
            "--db", str(self.db_path),
            "--out", str(self.report_path),
            "--scores-out", str(self.scores_path),
            "--model-out", str(self.model_path),
            "--cache-dir", str(self.cache_dir),
            "--max-symbols", str(max_symbols),
            "--max-trades", str(max_trades),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--min-n", str(min_n),
            "--min-hold-sec", str(min_hold_sec),
            "--train-ratio", str(train_ratio),
        ]
        logger.info("[CF_STAGE] mtf_image_dl_gate training refresh start")
        try:
            cp = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if cp.returncode != 0:
                tail = (cp.stderr or cp.stdout or "").strip().splitlines()[-4:]
                logger.warning(
                    "[DL_GATE] refresh failed rc=%s tail=%s",
                    cp.returncode,
                    " | ".join(tail),
                )
            else:
                last = (cp.stdout or "").strip().splitlines()
                if last:
                    logger.info("[DL_GATE] refresh done: %s", last[-1])
                else:
                    logger.info("[DL_GATE] refresh done")
        except Exception as e:
            logger.warning(f"[DL_GATE] refresh exception: {e}")

    def _load_score_map(self) -> None:
        self._score_map = {}
        if not self.scores_path.exists():
            return
        try:
            rows = json.loads(self.scores_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[DL_GATE] score file parse failed: {e}")
            return
        if not isinstance(rows, list):
            return
        for r in rows:
            if not isinstance(r, dict):
                continue
            uid = str(r.get("trade_uid") or "").strip()
            if not uid:
                continue
            try:
                p = float(r.get("mtf_dl_prob"))
            except Exception:
                continue
            if not math.isfinite(p):
                continue
            self._score_map[uid] = p
        if self._score_map:
            vals = np.asarray(list(self._score_map.values()), dtype=np.float64)
            logger.info(
                "[DL_GATE] scores loaded n=%s prob_min=%.6f prob_max=%.6f",
                len(self._score_map),
                float(np.min(vals)),
                float(np.max(vals)),
            )

    def _parse_quantiles(self) -> list[float]:
        raw = str(os.environ.get("CF_MTF_DL_GATE_QUANTILES", "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90") or "")
        out: list[float] = []
        for tok in raw.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                q = float(tok)
            except Exception:
                continue
            q = max(0.50, min(0.99, q))
            if q not in out:
                out.append(q)
        return out or [0.60, 0.70, 0.80]

    def get_param_grid(self) -> dict[str, list]:
        self._refresh_scores()
        self._load_score_map()
        if not self._score_map:
            # score가 없으면 stage를 no-op 처리
            return {"dl_gate_mode": ["disabled"], "dl_gate_quantile": [0.60]}
        return {
            "dl_gate_mode": ["global", "chop_only"],
            "dl_gate_quantile": self._parse_quantiles(),
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        mode = str(params.get("dl_gate_mode", "global") or "global").strip().lower()
        q = float(params.get("dl_gate_quantile", 0.70))
        q = max(0.50, min(0.99, q))
        if mode == "disabled" or not self._score_map:
            return list(trades)

        min_scored = max(20, _env_int("CF_MTF_DL_GATE_MIN_SCORED", 80))
        scored_all = [self._score_map.get(str(t.trade_uid or "").strip()) for t in trades]
        scored_all = [float(x) for x in scored_all if x is not None and math.isfinite(float(x))]
        if len(scored_all) < min_scored:
            return list(trades)

        out: list[Trade] = []
        if mode == "chop_only":
            scored_chop = []
            for t in trades:
                if str(t.regime or "").strip().lower() != "chop":
                    continue
                p = self._score_map.get(str(t.trade_uid or "").strip())
                if p is None:
                    continue
                scored_chop.append(float(p))
            if len(scored_chop) < max(20, min_scored // 2):
                return list(trades)
            thr = float(np.quantile(np.asarray(scored_chop, dtype=np.float64), q))
            for t in trades:
                if str(t.regime or "").strip().lower() != "chop":
                    out.append(t)
                    continue
                p = self._score_map.get(str(t.trade_uid or "").strip())
                if p is None or float(p) >= thr:
                    out.append(t)
            return out

        thr = float(np.quantile(np.asarray(scored_all, dtype=np.float64), q))
        for t in trades:
            p = self._score_map.get(str(t.trade_uid or "").strip())
            # 점수 없는 트레이드는 유지(coverage 부족으로 인한 과차단 방지)
            if p is None or float(p) >= thr:
                out.append(t)
        return out


class ExitReasonSimulator(StageSimulator):
    """Exit reason CF: what if certain exit types behaved differently?"""
    stage_name = "exit_reason"
    description = "청산 이유: mu_sign_flip, event_mc_exit, unrealized_dd 등"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "block_mu_sign_flip_before_sec": [120, 300, 600, 900, 1200],
            "mu_sign_flip_min_magnitude": [0.1, 0.3, 0.5, 0.8, 1.0, 2.0],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        block_before = params.get("block_mu_sign_flip_before_sec", 300)
        min_mag = params.get("mu_sign_flip_min_magnitude", 0.5)
        result = []
        for t in trades:
            t2 = copy.copy(t)
            if "mu_sign_flip" in (t.exit_reason or ""):
                # Would this exit be blocked?
                if t.hold_duration_sec < block_before:
                    # Proxy: keep the trade open, use half the current PnL
                    t2.realized_pnl = t.realized_pnl * 0.5 if t.realized_pnl > 0 else t.realized_pnl * 1.5
                elif abs(t.mu_alpha) < min_mag:
                    t2.realized_pnl = t.realized_pnl * 0.7
            result.append(t2)
        return result


class CapitalAllocationSimulator(StageSimulator):
    """Capital allocation CF: vary position sizing."""
    stage_name = "capital_allocation"
    description = "자본 분배: NOTIONAL_HARD_CAP, UNI_MAX_POS_FRAC, Kelly"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "notional_hard_cap": [50, 100, 150, 200, 300, 500],
            "max_pos_frac": [0.15, 0.25, 0.35, 0.45, 0.55],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        cap = params.get("notional_hard_cap", 150)
        max_frac = params.get("max_pos_frac", 0.45)
        result = []
        for t in trades:
            t2 = copy.copy(t)
            if t.notional > cap:
                ratio = cap / t.notional
                t2.realized_pnl = t.realized_pnl * ratio
                t2.notional = cap
            result.append(t2)
        return result


class RegimeMultiplierSimulator(StageSimulator):
    """Regime adjustment CF: vary mu/sigma multipliers."""
    stage_name = "regime_multiplier"
    description = "레짐 보정: mu/sigma 승수, 세션 보정"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "chop_mu_mult": [0.5, 0.7, 0.9, 1.0, 1.1],
            "bull_mu_mult": [0.8, 1.0, 1.2, 1.5],
            "bear_mu_mult": [0.5, 0.7, 1.0],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        chop_mult = params.get("chop_mu_mult", 0.9)
        bull_mult = params.get("bull_mu_mult", 1.2)
        bear_mult = params.get("bear_mu_mult", 0.7)
        regime_mult = {"chop": chop_mult, "bull": bull_mult, "bear": bear_mult, "volatile": 0.6}
        result = []
        for t in trades:
            t2 = copy.copy(t)
            r = (t.regime or "").lower()
            mult = regime_mult.get(r, 1.0)
            if mult > 1.0:
                t2.realized_pnl = t.realized_pnl * (1.0 + (mult - 1.0) * 0.3)
            elif mult < 1.0:
                if abs(t.mu_alpha) * mult < 0.5:
                    t2.realized_pnl = t.realized_pnl * 0.5
            result.append(t2)
        return result


# ──────────── Additional Simulators (Part 1 coverage) ────────────

class SpreadFilterSimulator(StageSimulator):
    """Spread filter: block entries when spread too wide."""
    stage_name = "spread_filter"
    description = "스프레드 필터: SPREAD_PCT_MAX"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "spread_pct_max": [0.0001, 0.0002, 0.0003, 0.0004, 0.0006, 0.0010, 0.0020],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        max_spread = params.get("spread_pct_max", 0.0004)
        result = []
        for t in trades:
            # Proxy: sigma is correlated with spread
            estimated_spread = t.sigma * 0.1 if t.sigma > 0 else 0.0003
            if estimated_spread <= max_spread:
                result.append(t)
        return result


class NetExpectancySimulator(StageSimulator):
    """Net expectancy filter: minimum edge requirement."""
    stage_name = "net_expectancy"
    description = "순기대수익 필터: ENTRY_NET_EXPECTANCY_MIN"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "net_expectancy_min": [-0.0030, -0.0020, -0.0015, -0.0010, -0.0007, -0.0005, -0.0003, -0.0002, -0.0001, 0.0, 0.0002, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        ne_min = params.get("net_expectancy_min", -0.0002)
        result = []
        for t in trades:
            # entry_ev is a proxy for net expectancy
            if t.entry_ev >= ne_min:
                result.append(t)
        return result


class EVSanityGateSimulator(StageSimulator):
    """
    Entry EV sanity gate for both-ev-negative + gross-ev floors.
    Proxy-based stage because per-side EV vectors are not persisted in trades table.
    """

    stage_name = "ev_sanity_gate"
    description = "EV 건전성 게이트: ENTRY_BOTH_EV_NEG_NET_FLOOR / ENTRY_GROSS_EV_MIN"
    max_combos_hint = 120

    def get_param_grid(self) -> dict[str, list]:
        return {
            "both_ev_neg_net_floor": [-0.0010, -0.0007, -0.0005, -0.0003, -0.0001, 0.0],
            "gross_ev_min": [0.0, 0.0002, 0.0005, 0.0008, 0.0012, 0.0016],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        both_floor = float(params.get("both_ev_neg_net_floor", -0.0003))
        gross_floor = float(params.get("gross_ev_min", 0.0005))
        out: list[Trade] = []
        for t in trades:
            # net edge proxy
            net_edge = float(t.entry_ev or 0.0)
            # gross edge proxy: add back round-trip fee proxy from raw payload
            fee_rate = 0.0006
            try:
                fee_rate = float((t.raw or {}).get("fee_rate") or fee_rate)
            except Exception:
                fee_rate = 0.0006
            fee_rate = max(0.0, min(0.01, float(fee_rate)))
            gross_edge = float(net_edge + fee_rate * 2.0)
            if net_edge <= both_floor:
                continue
            if gross_edge < gross_floor:
                continue
            out.append(t)
        return out


class MTFEntryGateSimulator(StageSimulator):
    """Sweep runtime MTF entry thresholds against scored historical trades."""

    stage_name = "mtf_entry_gate"
    description = "MTF 진입 게이트 임계치: MTF_DL_ENTRY_MIN_SIDE_PROB / MIN_WIN_PROB"
    max_combos_hint = 120

    def __init__(self):
        self.scores_path = Path(os.environ.get("CF_MTF_DL_SCORES_PATH", str(DEFAULT_DL_SCORES)))
        self._score_map_side: dict[str, float] = {}
        self._score_map_win: dict[str, float] = {}

    def _load_score_map(self) -> None:
        self._score_map_side = {}
        self._score_map_win = {}
        if not self.scores_path.exists():
            return
        try:
            rows = json.loads(self.scores_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(rows, list):
            return
        for r in rows:
            if not isinstance(r, dict):
                continue
            uid = str(r.get("trade_uid") or "").strip()
            if not uid:
                continue
            try:
                p_side = float(r.get("mtf_dl_prob"))
            except Exception:
                continue
            if not math.isfinite(p_side):
                continue
            p_win = r.get("mtf_dl_prob_win")
            try:
                p_win_f = float(p_win) if p_win is not None else float(p_side)
            except Exception:
                p_win_f = float(p_side)
            if not math.isfinite(p_win_f):
                p_win_f = float(p_side)
            self._score_map_side[uid] = float(p_side)
            self._score_map_win[uid] = float(p_win_f)

    def get_param_grid(self) -> dict[str, list]:
        self._load_score_map()
        side_vals = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]
        win_vals = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
        if self._score_map_side:
            qs = _parse_quantiles(
                os.environ.get("CF_MTF_ENTRY_RELAX_QUANTILES", "0.20,0.30,0.40,0.50,0.60,0.70,0.80"),
                [0.30, 0.50, 0.70],
                lo=0.05,
                hi=0.95,
            )
            side_src = list(self._score_map_side.values())
            win_src = list(self._score_map_win.values())
            side_dyn = _quantile_candidates(side_src, qs, lo=0.45, hi=0.90, relax_mult=0.995)
            win_dyn = _quantile_candidates(win_src, qs, lo=0.45, hi=0.90, relax_mult=0.995)
            side_vals = sorted(set(side_vals + [round(float(v), 4) for v in side_dyn]))
            win_vals = sorted(set(win_vals + [round(float(v), 4) for v in win_dyn]))
        return {
            "mtf_dl_entry_min_side_prob": side_vals,
            "mtf_dl_entry_min_win_prob": win_vals,
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        if not self._score_map_side:
            return list(trades)
        min_side = float(params.get("mtf_dl_entry_min_side_prob", 0.58))
        min_win = float(params.get("mtf_dl_entry_min_win_prob", 0.52))
        out: list[Trade] = []
        for t in trades:
            uid = str(t.trade_uid or "").strip()
            p_side = self._score_map_side.get(uid)
            p_win = self._score_map_win.get(uid)
            # keep unknown-score trades to avoid coverage-bias over-blocking
            if p_side is None:
                out.append(t)
                continue
            p_win_f = float(p_side if p_win is None else p_win)
            if float(p_side) >= min_side and p_win_f >= min_win:
                out.append(t)
        return out


class VirtualEntryExpansionSimulator(StageSimulator):
    """
    Simulate virtual fills for historically blocked candidates.

    Candidate pool is built from `[FILTER] ... blocked: [...]` log lines and
    symbol-level realized expectancy templates. This stage is intentionally
    conservative and applies a discount to synthetic PnL.
    """

    stage_name = "virtual_entry_expansion"
    description = "미진입 후보 가상체결: top_n/dir_gate/net_ev/mtf 완화 시 신규 진입 효과"
    max_combos_hint = 160

    def __init__(self):
        self.log_path = Path(os.environ.get("CF_VIRTUAL_ENTRY_LOG_PATH", str(PROJECT_ROOT / "state" / "codex_engine.log")))
        self.db_path = Path(os.environ.get("CF_VIRTUAL_ENTRY_DB_PATH", str(PROJECT_ROOT / "state" / "bot_data_live.db")))
        self.scores_path = Path(os.environ.get("CF_MTF_DL_SCORES_PATH", str(DEFAULT_DL_SCORES)))
        self._context_trades: list[Trade] = []
        self._virtual_candidates: list[Trade] = []
        self._symbol_side_prob: dict[str, float] = {}
        self._symbol_win_prob: dict[str, float] = {}

    def bind_context(self, trades: list[Trade]) -> None:
        self._context_trades = list(trades or [])
        self._virtual_candidates = []
        self._symbol_side_prob = {}
        self._symbol_win_prob = {}

    @staticmethod
    def _compact(values: list[Any], max_n: int = 4) -> list[Any]:
        vals = sorted(set(values))
        if len(vals) <= max_n:
            return vals
        idxs = [0, len(vals) // 3, (2 * len(vals)) // 3, len(vals) - 1]
        out: list[Any] = []
        for i in idxs:
            v = vals[int(max(0, min(len(vals) - 1, i)))]
            if v not in out:
                out.append(v)
        return out[:max_n]

    @staticmethod
    def _edge_proxy(t: Trade) -> float:
        raw = abs(float(t.mu_alpha or 0.0))
        if raw > 1.0:
            raw = raw / 100.0
        return float(max(0.0, min(0.20, raw)))

    def _load_symbol_mtf_probs(self) -> None:
        if not self.scores_path.exists():
            return
        try:
            rows = json.loads(self.scores_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(rows, list):
            return
        side_buf: dict[str, list[float]] = {}
        win_buf: dict[str, list[float]] = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            sym = str(r.get("symbol") or "").strip().upper()
            if not sym:
                continue
            try:
                p_side = float(r.get("mtf_dl_prob"))
            except Exception:
                continue
            if not math.isfinite(p_side):
                continue
            p_win_raw = r.get("mtf_dl_prob_win")
            try:
                p_win = float(p_win_raw) if p_win_raw is not None else p_side
            except Exception:
                p_win = p_side
            if not math.isfinite(p_win):
                p_win = p_side
            side_buf.setdefault(sym, []).append(float(p_side))
            win_buf.setdefault(sym, []).append(float(p_win))
        self._symbol_side_prob = {k: _robust_median(v, 0.5) for k, v in side_buf.items() if v}
        self._symbol_win_prob = {k: _robust_median(v, 0.5) for k, v in win_buf.items() if v}

    def _blocked_candidate_counts_from_db(self) -> dict[tuple[str, tuple[str, ...]], int]:
        out: dict[tuple[str, tuple[str, ...]], int] = {}
        if not self.db_path.exists():
            return out
        max_rows = max(1000, _env_int("CF_VIRTUAL_ENTRY_DB_ROWS", 20000))
        mode = str(os.environ.get("CF_VIRTUAL_ENTRY_DB_MODE", "live") or "").strip().lower()
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            if mode:
                rows = conn.execute(
                    """
                    SELECT symbol, blocked_filters, deny_reason
                    FROM entry_blocked_candidates
                    WHERE trading_mode = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (mode, int(max_rows)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT symbol, blocked_filters, deny_reason
                    FROM entry_blocked_candidates
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(max_rows),),
                ).fetchall()
            conn.close()
        except Exception:
            return out

        for row in rows:
            try:
                sym = str(row["symbol"] or "").strip().upper()
            except Exception:
                sym = ""
            if not sym:
                continue
            reasons: list[str] = []
            try:
                blocked_raw = row["blocked_filters"]
            except Exception:
                blocked_raw = None
            if blocked_raw:
                try:
                    parsed = json.loads(str(blocked_raw))
                    if isinstance(parsed, list):
                        reasons = [str(x).strip().lower() for x in parsed if str(x).strip()]
                except Exception:
                    reasons = []
            if not reasons:
                try:
                    deny_reason = str(row["deny_reason"] or "").strip().lower()
                except Exception:
                    deny_reason = ""
                if deny_reason:
                    reasons = [deny_reason]
            reasons_t = tuple(sorted(set(reasons)))
            if not reasons_t:
                continue
            key = (sym, reasons_t)
            out[key] = int(out.get(key, 0) + 1)
        return out

    def _blocked_candidate_counts(self) -> dict[tuple[str, tuple[str, ...]], int]:
        out: dict[tuple[str, tuple[str, ...]], int] = {}
        db_counts = self._blocked_candidate_counts_from_db()
        if db_counts:
            return db_counts
        if not self.log_path.exists():
            return out
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return out
        tail_n = max(800, _env_int("CF_VIRTUAL_ENTRY_LOG_LINES", 12000))
        lines = lines[-tail_n:]
        pat = re.compile(r"\[FILTER\]\s+([^\s]+)\s+blocked:\s+(\[[^\]]*\])")
        for ln in lines:
            m = pat.search(str(ln))
            if not m:
                continue
            sym = str(m.group(1) or "").strip().upper()
            if not sym:
                continue
            try:
                reasons_raw = ast.literal_eval(str(m.group(2) or "[]"))
                reasons = tuple(sorted(set(str(x).strip().lower() for x in reasons_raw if str(x).strip())))
            except Exception:
                reasons = tuple()
            if not reasons:
                continue
            key = (sym, reasons)
            out[key] = int(out.get(key, 0) + 1)
        return out

    @staticmethod
    def _expectancy_from_pnls(pnls: list[float]) -> float:
        arr = [float(x) for x in pnls if math.isfinite(float(x))]
        if not arr:
            return 0.0
        wins = [x for x in arr if x > 0]
        losses = [x for x in arr if x <= 0]
        wr = float(len(wins) / len(arr))
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(abs(np.mean(losses))) if losses else 0.0
        exp = wr * avg_win - (1.0 - wr) * avg_loss
        if not math.isfinite(exp):
            return 0.0
        return float(exp)

    def _symbol_templates(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        by_sym: dict[str, list[Trade]] = {}
        for t in self._context_trades:
            s = str(t.symbol or "").strip().upper()
            if not s:
                continue
            by_sym.setdefault(s, []).append(t)
        all_trades = list(self._context_trades)
        global_pnls = [float(t.realized_pnl or 0.0) for t in all_trades]
        global_tpl = {
            "side": "LONG",
            "regime": "chop",
            "notional": max(20.0, _robust_median([float(t.notional or 0.0) for t in all_trades], 50.0)),
            "leverage": max(1.0, _robust_median([float(t.leverage or 1.0) for t in all_trades], 3.0)),
            "entry_ev": _robust_median([float(t.entry_ev or 0.0) for t in all_trades], 0.0),
            "dir_conf": max(0.45, min(0.95, _robust_median([float(t.dir_conf or 0.5) for t in all_trades], 0.55))),
            "mu_alpha": _robust_median([abs(float(t.mu_alpha or 0.0)) for t in all_trades], 0.06),
            "vpin": max(0.0, min(1.0, _robust_median([float(t.vpin or 0.5) for t in all_trades], 0.5))),
            "hurst": max(0.0, min(1.0, _robust_median([float(t.hurst or 0.5) for t in all_trades], 0.5))),
            "sigma": max(0.01, _robust_median([float(t.sigma or 0.5) for t in all_trades], 0.5)),
            "entry_quality": _robust_median([float(t.entry_quality or 0.5) for t in all_trades], 0.5),
            "lev_signal": _robust_median([float(t.leverage_signal or 0.0) for t in all_trades], 0.0),
            "hold_sec": max(30.0, _robust_median([float(t.hold_duration_sec or 120.0) for t in all_trades], 180.0)),
            "expectancy_pnl": self._expectancy_from_pnls(global_pnls),
        }

        out: dict[str, dict[str, Any]] = {}
        for sym, tlist in by_sym.items():
            pnls = [float(t.realized_pnl or 0.0) for t in tlist]
            n_long = sum(1 for t in tlist if str(t.side or "").upper() == "LONG")
            n_short = max(0, len(tlist) - n_long)
            reg_counts: dict[str, int] = {}
            for t in tlist:
                r = str(t.regime or "chop").strip().lower()
                reg_counts[r] = int(reg_counts.get(r, 0) + 1)
            best_regime = max(reg_counts.items(), key=lambda kv: kv[1])[0] if reg_counts else "chop"
            out[sym] = {
                "side": "LONG" if n_long >= n_short else "SHORT",
                "regime": best_regime,
                "notional": max(10.0, _robust_median([float(t.notional or 0.0) for t in tlist], global_tpl["notional"])),
                "leverage": max(1.0, _robust_median([float(t.leverage or 1.0) for t in tlist], global_tpl["leverage"])),
                "entry_ev": _robust_median([float(t.entry_ev or 0.0) for t in tlist], global_tpl["entry_ev"]),
                "dir_conf": max(0.45, min(0.95, _robust_median([float(t.dir_conf or 0.5) for t in tlist], global_tpl["dir_conf"]))),
                "mu_alpha": _robust_median([abs(float(t.mu_alpha or 0.0)) for t in tlist], global_tpl["mu_alpha"]),
                "vpin": max(0.0, min(1.0, _robust_median([float(t.vpin or 0.5) for t in tlist], global_tpl["vpin"]))),
                "hurst": max(0.0, min(1.0, _robust_median([float(t.hurst or 0.5) for t in tlist], global_tpl["hurst"]))),
                "sigma": max(0.01, _robust_median([float(t.sigma or 0.5) for t in tlist], global_tpl["sigma"])),
                "entry_quality": _robust_median([float(t.entry_quality or 0.5) for t in tlist], global_tpl["entry_quality"]),
                "lev_signal": _robust_median([float(t.leverage_signal or 0.0) for t in tlist], global_tpl["lev_signal"]),
                "hold_sec": max(30.0, _robust_median([float(t.hold_duration_sec or 120.0) for t in tlist], global_tpl["hold_sec"])),
                "expectancy_pnl": self._expectancy_from_pnls(pnls),
            }
        return out, global_tpl

    @staticmethod
    def _reason_penalty(reasons: tuple[str, ...]) -> float:
        penalty_map = {
            "top_n": 0.98,
            "dir_gate": 0.85,
            "regime_dir_gate": 0.82,
            "net_expectancy": 0.78,
            "both_ev_neg": 0.70,
            "gross_ev": 0.74,
            "mtf_dl_entry": 0.88,
            "mu_align_gate": 0.90,
            "hybrid": 0.92,
            "min_notional": 0.95,
            "chop_vpin": 0.88,
            "chop_vol": 0.90,
        }
        p = 1.0
        for r in reasons:
            p *= float(penalty_map.get(str(r), 0.96))
        return float(max(0.20, min(1.05, p)))

    def _build_virtual_candidates(self) -> list[Trade]:
        if not self._context_trades:
            return []
        self._load_symbol_mtf_probs()
        blocked = self._blocked_candidate_counts()
        if not blocked:
            return []
        templates, gtpl = self._symbol_templates()
        max_total = max(40, _env_int("CF_VIRTUAL_ENTRY_MAX_CANDIDATES", 520))
        max_per_symbol = max(4, _env_int("CF_VIRTUAL_ENTRY_MAX_PER_SYMBOL", 28))
        pnl_discount = max(0.30, min(1.00, _env_float("CF_VIRTUAL_ENTRY_PNL_DISCOUNT", 0.68)))
        base_ts = int(max(int(t.timestamp_ms or 0) for t in self._context_trades) + 60_000)
        created = 0
        out: list[Trade] = []

        ranked = sorted(blocked.items(), key=lambda kv: kv[1], reverse=True)
        per_sym_cnt: dict[str, int] = {}
        for (sym, reasons), freq in ranked:
            if created >= max_total:
                break
            tpl = templates.get(sym, gtpl)
            n_add = max(1, int(round(math.sqrt(float(max(1, freq))))))
            n_add = min(n_add, max_per_symbol - int(per_sym_cnt.get(sym, 0)))
            if n_add <= 0:
                continue
            penalty = self._reason_penalty(reasons)
            top_only = (len(reasons) == 1 and reasons[0] == "top_n")
            for j in range(n_add):
                if created >= max_total:
                    break
                exp_pnl = float(tpl.get("expectancy_pnl", 0.0))
                if top_only:
                    exp_pnl *= 1.05
                est_pnl = float(exp_pnl * penalty * pnl_discount)
                if not math.isfinite(est_pnl):
                    est_pnl = 0.0
                entry_ev = float(tpl.get("entry_ev", 0.0)) * max(0.35, penalty)
                if "net_expectancy" in reasons:
                    entry_ev *= 0.85
                if "both_ev_neg" in reasons:
                    entry_ev -= 0.0002
                dir_conf = float(tpl.get("dir_conf", 0.55))
                if "dir_gate" in reasons:
                    dir_conf -= 0.03
                if "regime_dir_gate" in reasons:
                    dir_conf -= 0.02
                dir_conf = max(0.45, min(0.95, dir_conf))
                notional = max(10.0, float(tpl.get("notional", 50.0)))
                lev = max(1.0, float(tpl.get("leverage", 3.0)))
                margin = max(1e-6, notional / lev)
                roe = float(est_pnl / margin)
                mu_abs = max(0.0, float(tpl.get("mu_alpha", 0.06)))
                if "dir_gate" in reasons:
                    mu_abs *= 0.85
                mu_val = float(mu_abs if str(tpl.get("side", "LONG")).upper() == "LONG" else -mu_abs)
                side_prob = self._symbol_side_prob.get(sym)
                win_prob = self._symbol_win_prob.get(sym)
                ts = int(base_ts + created * 30_000)
                uid = f"virt_{sym}_{created}_{abs(hash(reasons)) % 10000:04d}"
                t = Trade(
                    trade_uid=uid,
                    symbol=sym,
                    side=str(tpl.get("side", "LONG")).upper(),
                    entry_price=1.0,
                    exit_price=1.0,
                    qty=1.0,
                    notional=notional,
                    leverage=lev,
                    entry_ev=float(entry_ev),
                    entry_confidence=float(dir_conf),
                    realized_pnl=float(est_pnl),
                    roe=float(roe),
                    hold_duration_sec=max(30.0, float(tpl.get("hold_sec", 180.0))),
                    regime=str(tpl.get("regime", "chop")),
                    exit_reason="virtual_entry_cf",
                    mu_alpha=mu_val,
                    dir_conf=float(dir_conf),
                    vpin=float(tpl.get("vpin", 0.5)),
                    hurst=float(tpl.get("hurst", 0.5)),
                    sigma=float(tpl.get("sigma", 0.5)),
                    entry_quality=float(tpl.get("entry_quality", 0.5)),
                    leverage_signal=float(tpl.get("lev_signal", 0.0)),
                    raw={
                        "virtual_entry": True,
                        "blocked_reasons": list(reasons),
                        "reason_penalty": float(penalty),
                        "mtf_side_prob": (None if side_prob is None else float(side_prob)),
                        "mtf_win_prob": (None if win_prob is None else float(win_prob)),
                    },
                    timestamp_ms=ts,
                )
                out.append(t)
                created += 1
                per_sym_cnt[sym] = int(per_sym_cnt.get(sym, 0) + 1)
        if out:
            logger.info("[VIRTUAL_ENTRY] built candidates=%s (symbols=%s)", len(out), len(per_sym_cnt))
        return out

    def _prepare_virtual_candidates(self) -> None:
        if self._virtual_candidates:
            return
        self._virtual_candidates = self._build_virtual_candidates()

    @staticmethod
    def _top_n_filter(cands: list[Trade], top_n: int, window_ms: int = 10 * 60 * 1000) -> list[Trade]:
        if top_n <= 0 or not cands:
            return list(cands)
        sorted_trades = sorted(cands, key=lambda t: int(t.timestamp_ms or 0))
        out: list[Trade] = []
        i = 0
        n = len(sorted_trades)
        while i < n:
            t0 = int(sorted_trades[i].timestamp_ms or 0)
            j = i
            bucket: list[Trade] = []
            while j < n and int(sorted_trades[j].timestamp_ms or 0) <= t0 + int(window_ms):
                bucket.append(sorted_trades[j])
                j += 1
            bucket_sorted = sorted(
                bucket,
                key=lambda t: (float(t.entry_ev or 0.0), float(t.dir_conf or 0.0)),
                reverse=True,
            )
            out.extend(bucket_sorted[: int(top_n)])
            i = j
        return out

    def get_param_grid(self) -> dict[str, list]:
        self._prepare_virtual_candidates()
        candidates = list(self._virtual_candidates)
        if not candidates:
            return {
                "top_n_symbols": [int(_env_float("TOP_N_SYMBOLS", 12))],
                "dir_gate_min_conf": [float(_env_float("ALPHA_DIRECTION_GATE_MIN_CONF", 0.60))],
                "dir_gate_min_edge": [float(_env_float("ALPHA_DIRECTION_GATE_MIN_EDGE", 0.06))],
                "dir_gate_min_side_prob": [float(_env_float("ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", 0.60))],
                "net_expectancy_min": [float(_env_float("ENTRY_NET_EXPECTANCY_MIN", -0.0002))],
                "both_ev_neg_net_floor": [float(_env_float("ENTRY_BOTH_EV_NEG_NET_FLOOR", -0.0003))],
                "gross_ev_min": [float(_env_float("ENTRY_GROSS_EV_MIN", 0.0005))],
                "mtf_dl_entry_min_side_prob": [float(_env_float("MTF_DL_ENTRY_MIN_SIDE_PROB", 0.58))],
                "mtf_dl_entry_min_win_prob": [float(_env_float("MTF_DL_ENTRY_MIN_WIN_PROB", 0.52))],
            }

        base_top = int(_env_float("TOP_N_SYMBOLS", 12))
        counts: list[float] = []
        sorted_cands = sorted(candidates, key=lambda t: int(t.timestamp_ms or 0))
        i = 0
        while i < len(sorted_cands):
            t0 = int(sorted_cands[i].timestamp_ms or 0)
            j = i
            while j < len(sorted_cands) and int(sorted_cands[j].timestamp_ms or 0) <= t0 + 10 * 60 * 1000:
                j += 1
            counts.append(float(max(1, j - i)))
            i = j
        top_qs = _parse_quantiles(
            os.environ.get("CF_VIRTUAL_TOP_N_QUANTILES", "0.55,0.70,0.85"),
            [0.60, 0.80],
            lo=0.10,
            hi=0.99,
        )
        top_dyn = _quantile_candidates(
            counts,
            top_qs,
            lo=2.0,
            hi=100.0,
            as_int=True,
            relax_mult=max(1.0, min(1.8, _env_float("CF_VIRTUAL_TOP_N_RELAX_MULT", 1.20))),
        )
        top_vals = self._compact(
            [int(base_top), int(max(2, round(base_top * 1.20))), int(max(2, round(base_top * 1.40)))] + [int(x) for x in top_dyn],
            max_n=4,
        )

        base_conf = float(_env_float("ALPHA_DIRECTION_GATE_MIN_CONF", 0.60))
        base_edge = float(_env_float("ALPHA_DIRECTION_GATE_MIN_EDGE", 0.06))
        base_side = float(_env_float("ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", 0.60))
        conf_src = [float(t.dir_conf or 0.5) for t in candidates]
        edge_src = [self._edge_proxy(t) for t in candidates]
        dir_qs = _parse_quantiles(
            os.environ.get("CF_VIRTUAL_DIR_QUANTILES", "0.10,0.20,0.35"),
            [0.15, 0.30],
            lo=0.05,
            hi=0.90,
        )
        conf_dyn = _quantile_candidates(conf_src, dir_qs, lo=0.45, hi=0.95, relax_mult=0.99)
        edge_dyn = _quantile_candidates(edge_src, dir_qs, lo=0.0, hi=0.20, relax_mult=0.90)
        side_dyn = _quantile_candidates(conf_src, dir_qs, lo=0.45, hi=0.95, relax_mult=0.99)
        conf_vals = self._compact([round(base_conf, 4), round(base_conf * 0.95, 4), round(base_conf * 0.90, 4)] + [round(float(v), 4) for v in conf_dyn], 4)
        edge_vals = self._compact([round(base_edge, 4), round(base_edge * 0.85, 4), round(base_edge * 0.70, 4)] + [round(float(v), 4) for v in edge_dyn], 4)
        side_vals = self._compact([round(base_side, 4), round(base_side * 0.95, 4), round(base_side * 0.90, 4)] + [round(float(v), 4) for v in side_dyn], 4)

        base_net = float(_env_float("ENTRY_NET_EXPECTANCY_MIN", -0.0002))
        base_both = float(_env_float("ENTRY_BOTH_EV_NEG_NET_FLOOR", -0.0003))
        base_gross = float(_env_float("ENTRY_GROSS_EV_MIN", 0.0005))
        net_vals = self._compact([round(base_net, 6), round(base_net - 0.0003, 6), round(base_net - 0.0006, 6)], 3)
        both_vals = self._compact([round(base_both, 6), round(base_both - 0.0002, 6)], 2)
        gross_vals = self._compact([round(base_gross, 6), round(max(0.0, base_gross * 0.60), 6)], 2)

        base_mtf_side = float(_env_float("MTF_DL_ENTRY_MIN_SIDE_PROB", 0.58))
        base_mtf_win = float(_env_float("MTF_DL_ENTRY_MIN_WIN_PROB", 0.52))
        mtf_side_src = [float(v) for v in self._symbol_side_prob.values() if v is not None]
        mtf_win_src = [float(v) for v in self._symbol_win_prob.values() if v is not None]
        mtf_qs = _parse_quantiles(
            os.environ.get("CF_VIRTUAL_MTF_QUANTILES", "0.25,0.40,0.55"),
            [0.30, 0.50],
            lo=0.05,
            hi=0.95,
        )
        mtf_side_dyn = _quantile_candidates(mtf_side_src, mtf_qs, lo=0.45, hi=0.95, relax_mult=0.995)
        mtf_win_dyn = _quantile_candidates(mtf_win_src, mtf_qs, lo=0.45, hi=0.95, relax_mult=0.995)
        mtf_side_vals = self._compact([round(base_mtf_side, 4), round(base_mtf_side * 0.95, 4), round(base_mtf_side * 0.90, 4)] + [round(float(v), 4) for v in mtf_side_dyn], 4)
        mtf_win_vals = self._compact([round(base_mtf_win, 4), round(base_mtf_win * 0.95, 4), round(base_mtf_win * 0.90, 4)] + [round(float(v), 4) for v in mtf_win_dyn], 4)

        return {
            "top_n_symbols": [int(v) for v in top_vals],
            "dir_gate_min_conf": [float(v) for v in conf_vals],
            "dir_gate_min_edge": [float(v) for v in edge_vals],
            "dir_gate_min_side_prob": [float(v) for v in side_vals],
            "net_expectancy_min": [float(v) for v in net_vals],
            "both_ev_neg_net_floor": [float(v) for v in both_vals],
            "gross_ev_min": [float(v) for v in gross_vals],
            "mtf_dl_entry_min_side_prob": [float(v) for v in mtf_side_vals],
            "mtf_dl_entry_min_win_prob": [float(v) for v in mtf_win_vals],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        self._prepare_virtual_candidates()
        if not self._virtual_candidates:
            return list(trades)

        top_n = int(params.get("top_n_symbols", _env_int("TOP_N_SYMBOLS", 12)))
        min_conf = float(params.get("dir_gate_min_conf", _env_float("ALPHA_DIRECTION_GATE_MIN_CONF", 0.60)))
        min_edge = float(params.get("dir_gate_min_edge", _env_float("ALPHA_DIRECTION_GATE_MIN_EDGE", 0.06)))
        min_side = float(params.get("dir_gate_min_side_prob", _env_float("ALPHA_DIRECTION_GATE_MIN_SIDE_PROB", 0.60)))
        net_floor = float(params.get("net_expectancy_min", _env_float("ENTRY_NET_EXPECTANCY_MIN", -0.0002)))
        both_floor = float(params.get("both_ev_neg_net_floor", _env_float("ENTRY_BOTH_EV_NEG_NET_FLOOR", -0.0003)))
        gross_floor = float(params.get("gross_ev_min", _env_float("ENTRY_GROSS_EV_MIN", 0.0005)))
        mtf_side_thr = float(params.get("mtf_dl_entry_min_side_prob", _env_float("MTF_DL_ENTRY_MIN_SIDE_PROB", 0.58)))
        mtf_win_thr = float(params.get("mtf_dl_entry_min_win_prob", _env_float("MTF_DL_ENTRY_MIN_WIN_PROB", 0.52)))

        eligible: list[Trade] = []
        for c in self._virtual_candidates:
            net_edge = float(c.entry_ev or 0.0)
            if net_edge < net_floor:
                continue
            if net_edge <= both_floor:
                continue
            fee_rate = 0.0006
            try:
                fee_rate = float((c.raw or {}).get("fee_rate") or fee_rate)
            except Exception:
                fee_rate = 0.0006
            fee_rate = max(0.0, min(0.01, float(fee_rate)))
            gross_edge = float(net_edge + fee_rate * 2.0)
            if gross_edge < gross_floor:
                continue

            side_prob = max(0.45, min(1.0, float(c.dir_conf or 0.5)))
            if side_prob < min_conf:
                continue
            if self._edge_proxy(c) < min_edge:
                continue
            if side_prob < min_side:
                continue

            mtf_side = (c.raw or {}).get("mtf_side_prob")
            mtf_win = (c.raw or {}).get("mtf_win_prob")
            if mtf_side is not None:
                try:
                    if float(mtf_side) < mtf_side_thr:
                        continue
                except Exception:
                    pass
            if mtf_win is not None:
                try:
                    if float(mtf_win) < mtf_win_thr:
                        continue
                except Exception:
                    pass
            eligible.append(c)

        eligible = self._top_n_filter(eligible, top_n=top_n)
        max_accept = max(0, _env_int("CF_VIRTUAL_ENTRY_MAX_ACCEPTED", 280))
        if max_accept > 0:
            eligible = eligible[:max_accept]
        out = list(trades)
        out.extend(eligible)
        return out


class MinHoldRegimeSimulator(StageSimulator):
    """Regime-specific minimum hold time."""
    stage_name = "min_hold_regime"
    description = "레짐별 최소 보유: EXIT_MIN_HOLD_SEC_BULL/CHOP/BEAR"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "min_hold_sec_bull": [30, 60, 120, 300, 600],
            "min_hold_sec_chop": [120, 300, 600, 900, 1200],
            "min_hold_sec_bear": [60, 120, 300, 600],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        regime_min = {
            "bull": params.get("min_hold_sec_bull", 120),
            "chop": params.get("min_hold_sec_chop", 600),
            "bear": params.get("min_hold_sec_bear", 300),
        }
        result = []
        for t in trades:
            r = (t.regime or "").lower()
            min_h = regime_min.get(r, 300)
            t2 = copy.copy(t)
            if t.hold_duration_sec < min_h:
                # Early exit → would have held longer. Proxy: better outcome for winners
                if t.realized_pnl > 0:
                    t2.realized_pnl = t.realized_pnl * 1.3  # Longer hold → bigger win
                else:
                    t2.realized_pnl = t.realized_pnl * 1.2  # Longer hold → slightly bigger loss
            result.append(t2)
        return result


class ExposureSimulator(StageSimulator):
    """Total exposure limit CF."""
    stage_name = "exposure"
    description = "총 노출도: MAX_NOTIONAL_EXPOSURE, MAX_CONCURRENT_POSITIONS"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "max_exposure": [1.5, 2.0, 3.0, 5.0, 7.0, 10.0],
            "max_concurrent": [2, 4, 6, 8, 12],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        max_concurrent = params.get("max_concurrent", 8)
        # Simulate position limit: keep only top N concurrent trades per time window
        result = []
        active: list[Trade] = []
        sorted_trades = sorted(self.trades_by_time(trades), key=lambda t: t.timestamp_ms)
        for t in sorted_trades:
            # Clean up expired positions
            active = [a for a in active if a.timestamp_ms + a.hold_duration_sec * 1000 > t.timestamp_ms]
            if len(active) < max_concurrent:
                result.append(t)
                active.append(t)
            # else: blocked by concurrent limit
        return result if result else trades  # Fallback to all trades if logic fails

    @staticmethod
    def trades_by_time(trades):
        return sorted(trades, key=lambda t: t.timestamp_ms)


class HurstDampenSimulator(StageSimulator):
    """Hurst exponent dampening CF."""
    stage_name = "hurst_dampen"
    description = "Hurst 감쇠: HURST_RANDOM_DAMPEN, 랜덤워크 신호 차단"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "hurst_dampen": [0.30, 0.45, 0.60, 0.75, 0.90, 1.00],
            "hurst_trend_threshold": [0.50, 0.55, 0.60, 0.65],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        dampen = params.get("hurst_dampen", 0.60)
        trend_thresh = params.get("hurst_trend_threshold", 0.55)
        result = []
        for t in trades:
            t2 = copy.copy(t)
            if t.hurst < trend_thresh:
                # Random walk → dampen signal
                if abs(t.mu_alpha) * dampen < 0.1:
                    # Signal too weak after dampening → skip
                    continue
                # Dampened PnL proxy
                t2.realized_pnl = t.realized_pnl * (0.5 + 0.5 * dampen)
            result.append(t2)
        return result


class FeeFilterSimulator(StageSimulator):
    """Fee filter strength CF."""
    stage_name = "fee_filter"
    description = "수수료 필터: FEE_FILTER_MULT (EV > fee × mult)"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "fee_filter_mult": [0.50, 0.70, 0.85, 1.00, 1.20, 1.50],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        mult = params.get("fee_filter_mult", 0.85)
        # Approximate fee at 5.5 bps per leg (taker)
        fee_round_trip = 0.0011  # ~11 bps round-trip
        result = []
        for t in trades:
            if t.entry_ev >= fee_round_trip * mult:
                result.append(t)
        return result


class ChopGuardSimulator(StageSimulator):
    """Chop regime entry guard CF."""
    stage_name = "chop_guard"
    description = "Chop 진입 가드: CHOP_ENTRY_FLOOR_ADD, CHOP_ENTRY_MIN_DIR_CONF"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "chop_entry_floor_add": [0.0, 0.001, 0.002, 0.003, 0.005, 0.008],
            "chop_entry_min_dir_conf": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        floor_add = params.get("chop_entry_floor_add", 0.003)
        min_conf = params.get("chop_entry_min_dir_conf", 0.70)
        result = []
        for t in trades:
            if (t.regime or "").lower() == "chop":
                if t.entry_ev < floor_add or t.dir_conf < min_conf:
                    continue  # Blocked by chop guard
            result.append(t)
        return result


class MuSignFlipSimulator(StageSimulator):
    """mu_sign_flip exit timing CF."""
    stage_name = "mu_sign_flip"
    description = "mu 부호 반전 청산: MU_SIGN_FLIP_MIN_AGE_SEC, CONFIRM_TICKS"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "mu_sign_flip_min_age": [120, 300, 600, 900, 1200, 1800],
            "mu_sign_flip_confirm_ticks": [4, 8, 12, 16, 20],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        min_age = params.get("mu_sign_flip_min_age", 600)
        result = []
        for t in trades:
            t2 = copy.copy(t)
            if "mu_sign_flip" in (t.exit_reason or ""):
                if t.hold_duration_sec < min_age:
                    # Would have been blocked → extend hold
                    if t.realized_pnl < 0:
                        t2.realized_pnl = t.realized_pnl * 0.6  # Less loss from patience
                    else:
                        t2.realized_pnl = t.realized_pnl * 1.2  # More gain from patience
            result.append(t2)
        return result


class UnifiedEntryFloorSimulator(StageSimulator):
    """Unified entry floor sweep (most critical filter)."""
    stage_name = "unified_floor"
    description = "통합 진입 Floor: UNIFIED_ENTRY_FLOOR"

    def get_param_grid(self) -> dict[str, list]:
        return {
            "min_ev": [-0.001, -0.0003, 0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        min_ev = params.get("min_ev", 0.001)
        return [t for t in trades if t.entry_ev >= min_ev]


class TopNSimulator(StageSimulator):
    """TOP-N ranking gate CF."""
    stage_name = "top_n"
    description = "TOP_N 심볼 수 제한: TOP_N_SYMBOLS"
    max_combos_hint = 80

    def __init__(self):
        self._context_trades: list[Trade] = []

    def bind_context(self, trades: list[Trade]) -> None:
        self._context_trades = list(trades or [])

    def _dynamic_top_n_values(self) -> list[int]:
        trades = sorted(self._context_trades, key=lambda t: int(t.timestamp_ms or 0))
        if not trades:
            return []
        window_ms = max(60_000, _env_int("CF_TOP_N_WINDOW_MS", 10 * 60 * 1000))
        counts: list[int] = []
        i = 0
        n = len(trades)
        while i < n:
            t0 = int(trades[i].timestamp_ms or 0)
            j = i
            while j < n and int(trades[j].timestamp_ms or 0) <= t0 + window_ms:
                j += 1
            counts.append(max(1, j - i))
            i = j
        qs = _parse_quantiles(
            os.environ.get("CF_TOP_N_DYNAMIC_QUANTILES", "0.35,0.50,0.65,0.75,0.85,0.92"),
            [0.50, 0.70, 0.85],
            lo=0.05,
            hi=0.99,
        )
        relax_mult = max(0.7, min(1.8, _env_float("CF_TOP_N_RELAX_MULT", 1.12)))
        out = _quantile_candidates(
            [float(x) for x in counts],
            qs,
            lo=2.0,
            hi=100.0,
            as_int=True,
            relax_mult=relax_mult,
        )
        return [int(x) for x in out]

    def get_param_grid(self) -> dict[str, list]:
        dynamic = self._dynamic_top_n_values()
        vals = sorted(set([4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 60] + dynamic))
        return {
            "top_n_symbols": vals,
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        top_n = int(params.get("top_n_symbols", 30))
        # Proxy: keep highest entry_ev trades in rolling time windows
        sorted_trades = sorted(trades, key=lambda t: t.timestamp_ms)
        if not sorted_trades:
            return []
        window_ms = 10 * 60 * 1000
        result: list[Trade] = []
        i = 0
        n = len(sorted_trades)
        while i < n:
            t0 = sorted_trades[i].timestamp_ms
            j = i
            bucket: list[Trade] = []
            while j < n and sorted_trades[j].timestamp_ms <= t0 + window_ms:
                bucket.append(sorted_trades[j])
                j += 1
            bucket_sorted = sorted(bucket, key=lambda t: (t.entry_ev, t.dir_conf), reverse=True)
            result.extend(bucket_sorted[:top_n])
            i = j
        return result


class DirectionGateSimulator(StageSimulator):
    """Direction gate CF (confidence/edge)."""
    stage_name = "direction_gate"
    description = "방향 게이트: ALPHA_DIRECTION_GATE_MIN_CONF/EDGE"
    max_combos_hint = 100

    def __init__(self):
        self._context_trades: list[Trade] = []

    def bind_context(self, trades: list[Trade]) -> None:
        self._context_trades = list(trades or [])

    @staticmethod
    def _edge_proxy(t: Trade) -> float:
        raw = abs(float(t.mu_alpha or 0.0))
        if raw > 1.0:
            raw = raw / 100.0
        return float(max(0.0, min(0.20, raw)))

    def get_param_grid(self) -> dict[str, list]:
        conf_vals = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
        edge_vals = [0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
        side_vals = [0.50, 0.53, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70]
        if self._context_trades:
            conf_src = [float(t.dir_conf or 0.0) for t in self._context_trades]
            edge_src = [self._edge_proxy(t) for t in self._context_trades]
            qs = _parse_quantiles(
                os.environ.get("CF_DIR_GATE_RELAX_QUANTILES", "0.10,0.15,0.20,0.25,0.30,0.40,0.50"),
                [0.15, 0.25, 0.40],
                lo=0.05,
                hi=0.95,
            )
            conf_dyn = _quantile_candidates(conf_src, qs, lo=0.45, hi=0.90, relax_mult=0.995)
            edge_dyn = _quantile_candidates(edge_src, qs, lo=0.0, hi=0.20, relax_mult=0.92)
            side_dyn = _quantile_candidates(conf_src, qs, lo=0.45, hi=0.95, relax_mult=0.995)
            conf_vals = sorted(set(conf_vals + [round(float(v), 4) for v in conf_dyn]))
            edge_vals = sorted(set(edge_vals + [round(float(v), 4) for v in edge_dyn]))
            side_vals = sorted(set(side_vals + [round(float(v), 4) for v in side_dyn]))
        return {
            "dir_gate_min_conf": conf_vals,
            "dir_gate_min_edge": edge_vals,
            "dir_gate_min_side_prob": side_vals,
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        min_conf = float(params.get("dir_gate_min_conf", 0.58))
        min_edge = float(params.get("dir_gate_min_edge", 0.06))
        min_side_prob = float(params.get("dir_gate_min_side_prob", 0.58))
        result = []
        for t in trades:
            edge = self._edge_proxy(t)
            side_prob = max(0.50, min(1.0, float(t.dir_conf or 0.0)))
            if float(t.dir_conf or 0.0) >= min_conf and edge >= min_edge and side_prob >= min_side_prob:
                result.append(t)
        return result


class PreMCSimulator(StageSimulator):
    """Pre-MC portfolio gate CF."""
    stage_name = "pre_mc_gate"
    description = "PRE_MC 게이트: min expected pnl / max liq prob"
    max_combos_hint = 100

    def get_param_grid(self) -> dict[str, list]:
        return {
            "pre_mc_min_expected_pnl": [0.0, 0.0002, 0.0005, 0.001, 0.002],
            "pre_mc_max_liq_prob": [0.02, 0.05, 0.08, 0.10, 0.15],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        min_exp = float(params.get("pre_mc_min_expected_pnl", 0.0))
        max_liq = float(params.get("pre_mc_max_liq_prob", 0.05))
        result = []
        for t in trades:
            exp_proxy = float(t.entry_ev or 0.0)
            liq_proxy = min(0.99, max(0.0, float(t.sigma or 0.0) * 0.5))
            if exp_proxy >= min_exp and liq_proxy <= max_liq:
                result.append(t)
        return result


class EventExitPolicySimulator(StageSimulator):
    """Event-based exit threshold CF."""
    stage_name = "event_exit_policy"
    description = "이벤트 청산 문턱: EVENT_EXIT_MAX_P_SL / EVENT_EXIT_MAX_ABS_CVAR"
    max_combos_hint = 80

    def get_param_grid(self) -> dict[str, list]:
        return {
            "event_exit_max_p_sl": [0.85, 0.90, 0.93, 0.95, 0.97, 0.99],
            "event_exit_max_abs_cvar": [0.03, 0.05, 0.075, 0.10, 0.15],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        max_psl = float(params.get("event_exit_max_p_sl", 0.97))
        max_cvar = float(params.get("event_exit_max_abs_cvar", 0.075))
        result = []
        for t in trades:
            t2 = copy.copy(t)
            is_event_exit = "event" in (t.exit_reason or "").lower()
            if is_event_exit:
                psl_proxy = min(0.99, max(0.0, float(t.vpin or 0.0)))
                cvar_proxy = min(0.50, max(0.0, abs(float(t.sigma or 0.0)) * 0.2))
                if psl_proxy > max_psl or cvar_proxy > max_cvar:
                    # Stricter event-exit threshold would have exited earlier
                    t2.realized_pnl = t.realized_pnl * (0.7 if t.realized_pnl > 0 else 0.8)
            result.append(t2)
        return result


class MinNotionalSimulator(StageSimulator):
    """Minimum notional floor CF."""
    stage_name = "min_notional"
    description = "최소 진입 금액: MIN_ENTRY_NOTIONAL"
    max_combos_hint = 80

    def get_param_grid(self) -> dict[str, list]:
        return {
            "min_entry_notional": [1, 3, 5, 10, 20, 30, 50],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        floor = float(params.get("min_entry_notional", 5))
        return [t for t in trades if float(t.notional or 0.0) >= floor]


class TodFilterSimulator(StageSimulator):
    """Time-of-day entry filter CF."""
    stage_name = "tod_filter"
    description = "시간대 필터: TOD_FILTER_ENABLED, TRADING_BAD_HOURS_UTC"
    max_combos_hint = 60

    def get_param_grid(self) -> dict[str, list]:
        return {
            "trading_bad_hours_utc": [
                "6,7",
                "6,7,13",
                "6,7,19",
                "6,7,13,19",
                "",
            ],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        bad_raw = str(params.get("trading_bad_hours_utc", "6,7") or "")
        bad_hours = set()
        for tok in bad_raw.split(","):
            tok = tok.strip()
            if tok.isdigit():
                bad_hours.add(int(tok) % 24)
        if not bad_hours:
            return list(trades)
        result = []
        for t in trades:
            try:
                h = int((int(t.timestamp_ms) // 1000) % 86400 // 3600)
            except Exception:
                h = 0
            if h in bad_hours:
                continue
            result.append(t)
        return result


class RegimeSideBlockSimulator(StageSimulator):
    """Regime-side blocklist CF."""
    stage_name = "regime_side_block"
    description = "레짐-방향 차단: REGIME_SIDE_BLOCK_LIST"
    max_combos_hint = 80

    def get_param_grid(self) -> dict[str, list]:
        return {
            "regime_side_block_list": [
                "",
                "bear_long,bull_short",
                "bear_long,chop_long",
                "bear_long,bull_short,chop_long",
                "volatile_long,bear_long",
            ],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        block_raw = str(params.get("regime_side_block_list", "") or "").strip().lower()
        blocks = {tok.strip() for tok in block_raw.split(",") if tok.strip()}
        if not blocks:
            return list(trades)
        out = []
        for t in trades:
            side = str(t.side or "").lower()
            regime = str(t.regime or "unknown").lower()
            key = f"{regime}_{side}"
            if key in blocks:
                continue
            out.append(t)
        return out


class LeverageFloorLockSimulator(StageSimulator):
    """Leverage floor lock gate CF."""
    stage_name = "leverage_floor_lock"
    description = "레버리지 바닥 고착 차단: LEVERAGE_FLOOR_LOCK_*"
    max_combos_hint = 120

    def get_param_grid(self) -> dict[str, list]:
        return {
            "lev_floor_lock_min_sticky": [2, 3, 4, 5],
            "lev_floor_lock_max_ev_gap": [0.0004, 0.0008, 0.0012, 0.0016],
            "lev_floor_lock_max_conf": [0.55, 0.60, 0.65, 0.70],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        min_sticky = int(params.get("lev_floor_lock_min_sticky", 3))
        max_gap = float(params.get("lev_floor_lock_max_ev_gap", 0.0008))
        max_conf = float(params.get("lev_floor_lock_max_conf", 0.60))
        out: list[Trade] = []
        for t in trades:
            t2 = copy.copy(t)
            lev_near_floor = float(t.leverage or 1.0) <= (1.0 + 0.02 * min_sticky)
            low_edge = abs(float(t.entry_ev or 0.0)) <= max_gap
            low_conf = float(t.entry_confidence or 0.0) <= max_conf
            if lev_near_floor and low_edge and low_conf:
                continue
            out.append(t2)
        return out


class PreMCScaledSizeSimulator(StageSimulator):
    """PRE-MC scaled-size gate CF."""
    stage_name = "pre_mc_scaled_size"
    description = "PRE_MC_SIZE_SCALE, PRE_MC_MAX_LIQ_PROB 기반 스케일"
    max_combos_hint = 100

    def get_param_grid(self) -> dict[str, list]:
        return {
            "pre_mc_size_scale": [0.25, 0.40, 0.50, 0.65, 0.80, 1.00],
            "pre_mc_max_liq_prob": [0.03, 0.05, 0.08, 0.10, 0.15],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        size_scale = float(params.get("pre_mc_size_scale", 0.50))
        max_liq = float(params.get("pre_mc_max_liq_prob", 0.05))
        out = []
        for t in trades:
            t2 = copy.copy(t)
            liq_proxy = min(0.99, max(0.0, float(t.sigma or 0.0) * 0.5))
            if liq_proxy > max_liq:
                t2.realized_pnl = float(t2.realized_pnl) * float(size_scale)
                t2.notional = float(t2.notional) * float(size_scale)
            out.append(t2)
        return out


class PreMCBlockModeSimulator(StageSimulator):
    """PRE-MC fail handling mode CF."""
    stage_name = "pre_mc_block_mode"
    description = "PRE_MC_BLOCK_ON_FAIL, PRE_MC_MIN_CVAR"
    max_combos_hint = 80

    def get_param_grid(self) -> dict[str, list]:
        return {
            "pre_mc_block_on_fail": [0, 1],
            "pre_mc_min_cvar": [-0.20, -0.12, -0.08, -0.05, -0.03],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        block_on_fail = int(params.get("pre_mc_block_on_fail", 1))
        min_cvar = float(params.get("pre_mc_min_cvar", -0.05))
        out = []
        for t in trades:
            t2 = copy.copy(t)
            cvar_proxy = -abs(float(t.sigma or 0.0)) * 0.15
            if cvar_proxy < min_cvar:
                if block_on_fail:
                    continue
                t2.realized_pnl = float(t2.realized_pnl) * 0.6
            out.append(t2)
        return out


class DirectionConfirmSimulator(StageSimulator):
    """Direction confirmation ticks CF."""
    stage_name = "direction_confirm"
    description = "방향 확인틱: ALPHA_DIRECTION_GATE_CONFIRM_TICKS*"
    max_combos_hint = 80

    def get_param_grid(self) -> dict[str, list]:
        return {
            "dir_gate_confirm_ticks": [1, 2, 3, 4],
            "dir_gate_confirm_ticks_chop": [1, 2, 3, 4],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        base_ticks = int(params.get("dir_gate_confirm_ticks", 1))
        chop_ticks = int(params.get("dir_gate_confirm_ticks_chop", base_ticks))
        out = []
        for t in trades:
            ticks_need = chop_ticks if str(t.regime or "").lower() == "chop" else base_ticks
            conf_penalty = max(0.0, min(0.25, (ticks_need - 1) * 0.04))
            if float(t.dir_conf or 0.0) >= (0.50 + conf_penalty):
                out.append(t)
        return out


# ─────────────────────────────────────────────────────────────────
# HYBRID SYSTEM SIMULATORS (MC_HYBRID_ONLY=1)
# ─────────────────────────────────────────────────────────────────

class HybridExitTimingSimulator(StageSimulator):
    """Hybrid exit confirmation ticks by regime."""
    stage_name = "hybrid_exit_timing"
    description = "하이브리드 청산 확인틱: HYBRID_EXIT_CONFIRM_TICKS_*"
    max_combos_hint = 150

    def get_param_grid(self) -> dict[str, list]:
        return {
            "hybrid_exit_confirm_shock": [1, 2, 3, 4, 5],
            "hybrid_exit_confirm_normal": [3, 4, 5, 6, 7, 8],
            "hybrid_exit_confirm_noise": [5, 6, 7, 8, 10, 12],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        shock_ticks = int(params.get("hybrid_exit_confirm_shock", 3))
        normal_ticks = int(params.get("hybrid_exit_confirm_normal", 6))
        noise_ticks = int(params.get("hybrid_exit_confirm_noise", 8))
        out = []
        for t in trades:
            t2 = copy.copy(t)
            regime = str(t.regime or "").lower()
            # Map regime to confirm ticks
            if "shock" in regime:
                confirm = shock_ticks
            elif "noise" in regime or "chop" in regime:
                confirm = noise_ticks
            else:
                confirm = normal_ticks
            # Proxy: more ticks = later exit = worse if losing, better if winning
            hold_factor = 1.0 + (confirm - 6) * 0.02  # baseline=6
            if t.realized_pnl > 0:
                t2.realized_pnl = float(t.realized_pnl) * hold_factor
            else:
                t2.realized_pnl = float(t.realized_pnl) * (2.0 - hold_factor)
            out.append(t2)
        return out


class HybridLeverageSimulator(StageSimulator):
    """Hybrid leverage sweep parameters."""
    stage_name = "hybrid_leverage"
    description = "하이브리드 레버리지: HYBRID_LEV_SWEEP_MIN/MAX/STEP"
    max_combos_hint = 120

    def get_param_grid(self) -> dict[str, list]:
        return {
            "hybrid_lev_sweep_min": [1.0, 1.5, 2.0, 2.5],
            "hybrid_lev_sweep_max": [3.0, 4.0, 5.0, 6.0, 8.0],
            "hybrid_lev_ev_scale": [50, 100, 150, 200, 300],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        lev_min = float(params.get("hybrid_lev_sweep_min", 1.5))
        lev_max = float(params.get("hybrid_lev_sweep_max", 5.0))
        ev_scale = float(params.get("hybrid_lev_ev_scale", 100))
        out = []
        for t in trades:
            t2 = copy.copy(t)
            # Simulate leverage adjustment based on EV
            ev = float(t.entry_ev or 0)
            optimal_lev = lev_min + (lev_max - lev_min) * min(1.0, max(0.0, ev * ev_scale))
            actual_lev = float(t.leverage or 1.0)
            lev_ratio = optimal_lev / max(actual_lev, 0.1)
            # Adjust PnL by leverage ratio (capped)
            adj = min(2.0, max(0.5, lev_ratio))
            t2.realized_pnl = float(t.realized_pnl) * adj
            out.append(t2)
        return out


class MCHybridPathsSimulator(StageSimulator):
    """MC simulation paths and horizon steps."""
    stage_name = "mc_hybrid_paths"
    description = "MC 시뮬레이션: MC_HYBRID_N_PATHS, MC_HYBRID_HORIZON_STEPS"
    max_combos_hint = 80

    def get_param_grid(self) -> dict[str, list]:
        return {
            "mc_hybrid_n_paths": [1024, 2048, 4096, 8192, 16384],
            "mc_hybrid_horizon_steps": [30, 60, 120, 180, 300],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        n_paths = int(params.get("mc_hybrid_n_paths", 4096))
        horizon = int(params.get("mc_hybrid_horizon_steps", 60))
        # Proxy: more paths = better EV estimation
        # More horizon = captures longer trends
        quality_factor = np.log2(n_paths / 4096 + 1) * 0.5 + 1.0
        horizon_factor = 1.0 + (horizon - 60) * 0.002
        out = []
        for t in trades:
            t2 = copy.copy(t)
            # Better simulation = more accurate entry/exit = better PnL
            if t.realized_pnl > 0:
                t2.realized_pnl = float(t.realized_pnl) * quality_factor * horizon_factor
            else:
                # Loss trades: better simulation might have avoided them
                t2.realized_pnl = float(t.realized_pnl) * (0.5 + 0.5 / quality_factor)
            out.append(t2)
        return out


class HybridCashPenaltySimulator(StageSimulator):
    """Hybrid cash penalty (opportunity cost)."""
    stage_name = "hybrid_cash_penalty"
    description = "현금 페널티: HYBRID_CASH_PENALTY"
    max_combos_hint = 60

    def get_param_grid(self) -> dict[str, list]:
        return {
            "hybrid_cash_penalty": [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        penalty = float(params.get("hybrid_cash_penalty", 0.001))
        out = []
        for t in trades:
            t2 = copy.copy(t)
            # High penalty = more aggressive entry = more trades but lower quality
            # Proxy: penalty reduces effective PnL threshold
            hold_sec = float(t.hold_duration_sec or 0)
            opportunity_cost = penalty * hold_sec / 3600  # per hour
            t2.realized_pnl = float(t.realized_pnl) - opportunity_cost
            out.append(t2)
        return out


class ExpValsThresholdSimulator(StageSimulator):
    """exp_vals direction dominance threshold."""
    stage_name = "exp_vals_threshold"
    description = "exp_vals 방향 임계치: EXP_VALS_MIN_DIFF, EXP_VALS_DOMINANCE"
    max_combos_hint = 100

    def get_param_grid(self) -> dict[str, list]:
        return {
            "exp_vals_min_diff": [0.0001, 0.0005, 0.001, 0.002, 0.005],
            "exp_vals_dominance_ratio": [1.2, 1.5, 2.0, 2.5, 3.0],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        min_diff = float(params.get("exp_vals_min_diff", 0.001))
        dom_ratio = float(params.get("exp_vals_dominance_ratio", 1.5))
        out = []
        for t in trades:
            t2 = copy.copy(t)
            # Proxy: direction confidence as quality indicator
            conf = float(t.dir_conf or 0.5)
            ev = abs(float(t.entry_ev or 0))
            # High dominance requirement = fewer but better trades
            if conf < 0.5 + min_diff or ev < min_diff * dom_ratio:
                continue  # Would have been filtered
            # Quality multiplier for passing threshold
            quality = 1.0 + (conf - 0.5) * 0.5
            t2.realized_pnl = float(t.realized_pnl) * quality
            out.append(t2)
        return out


class HybridBeamSearchSimulator(StageSimulator):
    """Hybrid beam search parameters for LSM."""
    stage_name = "hybrid_beam_search"
    description = "LSM Beam Search: HYBRID_BEAM_WIDTH, LSM_POLY_DEGREE"
    max_combos_hint = 80

    def get_param_grid(self) -> dict[str, list]:
        return {
            "hybrid_beam_width": [3, 5, 8, 10, 15],
            "lsm_poly_degree": [2, 3, 4, 5],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        beam_width = int(params.get("hybrid_beam_width", 5))
        poly_deg = int(params.get("lsm_poly_degree", 3))
        # Proxy: wider beam = better path exploration
        beam_factor = 1.0 + (beam_width - 5) * 0.02
        poly_factor = 1.0 + (poly_deg - 3) * 0.01
        out = []
        for t in trades:
            t2 = copy.copy(t)
            t2.realized_pnl = float(t.realized_pnl) * beam_factor * poly_factor
            out.append(t2)
        return out


class HybridTPSLRatioSimulator(StageSimulator):
    """Hybrid TP/SL ratio by regime."""
    stage_name = "hybrid_tp_sl_ratio"
    description = "하이브리드 TP/SL: HYBRID_TP_PCT_*, HYBRID_SL_PCT_*"
    max_combos_hint = 200

    def get_param_grid(self) -> dict[str, list]:
        return {
            "hybrid_tp_base": [0.003, 0.004, 0.005, 0.006, 0.008, 0.010],
            "hybrid_sl_base": [0.002, 0.003, 0.004, 0.005, 0.006],
            "hybrid_tp_shock_mult": [0.6, 0.75, 0.9, 1.0],
            "hybrid_tp_noise_mult": [1.2, 1.5, 1.8, 2.0],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        tp_base = float(params.get("hybrid_tp_base", 0.005))
        sl_base = float(params.get("hybrid_sl_base", 0.004))
        shock_mult = float(params.get("hybrid_tp_shock_mult", 0.75))
        noise_mult = float(params.get("hybrid_tp_noise_mult", 1.5))
        out = []
        for t in trades:
            t2 = copy.copy(t)
            regime = str(t.regime or "").lower()
            # Adjust TP/SL by regime
            if "shock" in regime:
                tp = tp_base * shock_mult
                sl = sl_base * 0.8
            elif "noise" in regime or "chop" in regime:
                tp = tp_base * noise_mult
                sl = sl_base * 1.2
            else:
                tp = tp_base
                sl = sl_base
            # Risk-reward proxy
            rr_factor = tp / max(sl, 0.001)
            # Better R:R = better outcome
            adj = min(1.5, max(0.7, rr_factor / 1.5))
            t2.realized_pnl = float(t.realized_pnl) * adj
            out.append(t2)
        return out


class SymbolQualityTimeSimulator(StageSimulator):
    """Symbol-quality time filter CF."""
    stage_name = "symbol_quality_time"
    description = "심볼 품질 시간가중: SYMBOL_QUALITY_TIME_WINDOW_HOURS/WEIGHT"
    max_combos_hint = 100

    def get_param_grid(self) -> dict[str, list]:
        return {
            "sq_time_window_hours": [1, 2, 3, 4, 6],
            "sq_time_weight": [0.10, 0.20, 0.35, 0.50],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        window_h = int(params.get("sq_time_window_hours", 2))
        weight = float(params.get("sq_time_weight", 0.35))
        # proxy: penalize off-hour entries (UTC 0~5) as quality-time mismatch
        out = []
        for t in trades:
            t2 = copy.copy(t)
            try:
                h = int((int(t.timestamp_ms) // 1000) % 86400 // 3600)
            except Exception:
                h = 0
            dist = min(abs(h - 12), 24 - abs(h - 12))
            if dist > max(1, window_h):
                t2.realized_pnl = float(t2.realized_pnl) * float(max(0.2, 1.0 - weight))
            out.append(t2)
        return out


# ─────────────────────────────────────────────────────────────────
# Counterfactual Engine
# ─────────────────────────────────────────────────────────────────

ALL_SIMULATORS = [
    LeverageSimulator(),
    TPSLSimulator(),
    HoldDurationSimulator(),
    EntryFilterSimulator(),
    DirectionSimulator(),
    VPINFilterSimulator(),
    VolatilityGateSimulator(),
    MTFImageDLGateSimulator(),
    MTFEntryGateSimulator(),
    VirtualEntryExpansionSimulator(),
    ExitReasonSimulator(),
    CapitalAllocationSimulator(),
    RegimeMultiplierSimulator(),
    # New expanded simulators
    SpreadFilterSimulator(),
    NetExpectancySimulator(),
    EVSanityGateSimulator(),
    MinHoldRegimeSimulator(),
    ExposureSimulator(),
    HurstDampenSimulator(),
    FeeFilterSimulator(),
    ChopGuardSimulator(),
    MuSignFlipSimulator(),
    UnifiedEntryFloorSimulator(),
    TopNSimulator(),
    DirectionGateSimulator(),
    PreMCSimulator(),
    EventExitPolicySimulator(),
    MinNotionalSimulator(),
    TodFilterSimulator(),
    RegimeSideBlockSimulator(),
    LeverageFloorLockSimulator(),
    PreMCScaledSizeSimulator(),
    PreMCBlockModeSimulator(),
    DirectionConfirmSimulator(),
    SymbolQualityTimeSimulator(),
    # HYBRID SYSTEM simulators (MC_HYBRID_ONLY=1)
    HybridExitTimingSimulator(),
    HybridLeverageSimulator(),
    MCHybridPathsSimulator(),
    HybridCashPenaltySimulator(),
    ExpValsThresholdSimulator(),
    HybridBeamSearchSimulator(),
    HybridTPSLRatioSimulator(),
]


class CFEngine:
    """
    Counterfactual Simulation Engine.
    Sweeps parameter grids per pipeline stage and discovers significant improvements.
    """

    def __init__(self, trades: list[Trade], simulators: list[StageSimulator] | None = None, sample_seed: int | None = None):
        self.trades = trades
        self.simulators = simulators or ALL_SIMULATORS
        self.sample_seed = sample_seed
        self.baseline = compute_metrics(trades)
        self.baseline_by_regime = compute_metrics_by_regime(trades)
        self.results: list[CFResult] = []
        self.findings: list[Finding] = []
        self._finding_id = 0

    def run_all(self, max_combos_per_stage: int = 200) -> list[Finding]:
        """Run CF simulation across all pipeline stages."""
        logger.info(f"Starting CF sweep: {len(self.trades)} trades, "
                    f"{len(self.simulators)} stages, baseline PnL=${self.baseline['pnl']}")
        for sim in self.simulators:
            self._run_stage(sim, max_combos_per_stage)
        # Sort findings by improvement
        self.findings.sort(key=lambda f: f.improvement_pct, reverse=True)
        logger.info(f"CF sweep complete: {len(self.findings)} findings")
        return self.findings

    def run_single_stage(self, stage_name: str, max_combos: int = 500) -> list[Finding]:
        """Run CF on a specific stage for deeper analysis."""
        for sim in self.simulators:
            if sim.stage_name == stage_name:
                return self._run_stage(sim, max_combos)
        return []

    def _stage_rng(self, stage_name: str):
        if self.sample_seed is None:
            return np.random.default_rng()
        stage_salt = sum((i + 1) * ord(ch) for i, ch in enumerate(str(stage_name)))
        stage_seed = int((int(self.sample_seed) + int(stage_salt)) % (2**32 - 1))
        if stage_seed <= 0:
            stage_seed = int(stage_salt or 1)
        return np.random.default_rng(stage_seed)

    @staticmethod
    def _focus_values(values: list[Any], max_n: int = 3) -> list[Any]:
        vals = list(values or [])
        if len(vals) <= max_n:
            return vals
        idxs = [0, len(vals) // 2, len(vals) - 1]
        out: list[Any] = []
        for i in idxs:
            v = vals[int(max(0, min(len(vals) - 1, i)))]
            if v not in out:
                out.append(v)
        return out[:max_n]

    def _select_stage_combos(
        self,
        *,
        sim: StageSimulator,
        param_values: list[list[Any]],
        stage_max_combos: int,
        baseline_combo: tuple[Any, ...] | None = None,
    ) -> tuple[list[tuple[Any, ...]], str, int]:
        combos_all = list(itertools.product(*param_values))
        total = len(combos_all)
        if total <= stage_max_combos:
            return combos_all, "full", total

        rng = self._stage_rng(sim.stage_name)
        mode = str(os.environ.get("CF_COMBO_SEARCH_MODE", "adaptive") or "adaptive").strip().lower()
        if mode in ("random", "rand", "legacy"):
            indices = rng.choice(total, size=stage_max_combos, replace=False)
            sampled = [combos_all[int(i)] for i in indices]
            return sampled, "random", total

        selected: list[tuple[Any, ...]] = []
        seen: set[tuple[Any, ...]] = set()

        def _add(combo: tuple[Any, ...]) -> None:
            if combo in seen:
                return
            seen.add(combo)
            selected.append(combo)

        baseline = tuple(baseline_combo) if baseline_combo else tuple(vals[0] for vals in param_values)
        _add(baseline)
        _add(tuple(vals[len(vals) // 2] for vals in param_values))
        _add(tuple(vals[-1] for vals in param_values))

        # 1) Marginal one-factor sweep around baseline
        for i, vals in enumerate(param_values):
            for v in vals:
                c = list(baseline)
                c[i] = v
                _add(tuple(c))
                if len(selected) >= stage_max_combos:
                    return selected[:stage_max_combos], "adaptive_marginal", total

        # 2) Pairwise interactions on top-variance dimensions
        ranked_idx = sorted(range(len(param_values)), key=lambda j: len(param_values[j]), reverse=True)
        top_idx = ranked_idx[: min(4, len(ranked_idx))]
        for a in range(len(top_idx)):
            for b in range(a + 1, len(top_idx)):
                i = top_idx[a]
                j = top_idx[b]
                vals_i = self._focus_values(param_values[i], max_n=3)
                vals_j = self._focus_values(param_values[j], max_n=3)
                for vi in vals_i:
                    for vj in vals_j:
                        c = list(baseline)
                        c[i] = vi
                        c[j] = vj
                        _add(tuple(c))
                        if len(selected) >= stage_max_combos:
                            return selected[:stage_max_combos], "adaptive_interaction", total

        # 3) Diversity fill from seeded random subset + deterministic stride
        if len(selected) < stage_max_combos:
            need = int(stage_max_combos - len(selected))
            sample_n = min(total, max(need * 4, need))
            idxs = rng.choice(total, size=sample_n, replace=False)
            for idx in idxs:
                _add(combos_all[int(idx)])
                if len(selected) >= stage_max_combos:
                    break
        if len(selected) < stage_max_combos:
            stride = max(1, total // max(1, stage_max_combos))
            for idx in range(0, total, stride):
                _add(combos_all[int(idx)])
                if len(selected) >= stage_max_combos:
                    break

        return selected[:stage_max_combos], "adaptive", total

    def _build_walk_forward_folds(self) -> list[tuple[int, int, int]]:
        if not _env_bool("CF_WF_ENABLED", True):
            return []
        n = int(len(self.trades))
        if n <= 0:
            return []
        k = max(1, _env_int("CF_WF_K_FOLDS", 4))
        min_train = max(40, _env_int("CF_WF_MIN_TRAIN_TRADES", 120))
        min_test = max(20, _env_int("CF_WF_MIN_TEST_TRADES", 40))
        if n < (min_train + min_test):
            return []
        test_size = max(min_test, n // (k + 1))
        folds: list[tuple[int, int, int]] = []
        for i in range(k):
            test_start = n - (k - i) * test_size
            test_end = min(n, test_start + test_size)
            train_end = test_start
            if train_end < min_train:
                continue
            if (test_end - test_start) < min_test:
                continue
            if test_start <= 0 or test_start >= n:
                continue
            folds.append((train_end, test_start, test_end))
        return folds

    def _evaluate_walk_forward(self, sim: StageSimulator, params: dict) -> dict[str, Any]:
        if not _env_bool("CF_WF_ENABLED", True):
            return {
                "enabled": False,
                "pass": True,
                "folds": 0,
                "pass_rate": 1.0,
                "train_pnl_delta_mean": 0.0,
                "test_pnl_delta_mean": 0.0,
                "penalty_factor": 1.0,
            }
        folds = self._build_walk_forward_folds()
        if not folds:
            return {
                "enabled": True,
                "pass": False,
                "folds": 0,
                "pass_rate": 0.0,
                "train_pnl_delta_mean": 0.0,
                "test_pnl_delta_mean": 0.0,
                "penalty_factor": 0.5,
                "reason": "insufficient_samples",
            }

        min_pass_rate = float(max(0.0, min(1.0, _env_float("CF_WF_MIN_PASS_RATE", 0.60))))
        min_test_delta = float(_env_float("CF_WF_MIN_TEST_PNL_DELTA", 0.0))
        min_cov = float(max(0.0, min(1.0, _env_float("CF_WF_MIN_TEST_COVERAGE", 0.20))))
        edge_floor = float(_env_float("CF_WF_TEST_EDGE_FLOOR_DELTA", -0.002))

        fold_rows: list[dict[str, Any]] = []
        train_deltas: list[float] = []
        test_deltas: list[float] = []
        pass_count = 0

        for idx, (train_end, test_start, test_end) in enumerate(folds, start=1):
            train_trades = self.trades[:train_end]
            test_trades = self.trades[test_start:test_end]
            base_train = compute_metrics(train_trades)
            base_test = compute_metrics(test_trades)
            sim_train_trades = sim.simulate(train_trades, params) or []
            sim_test_trades = sim.simulate(test_trades, params) or []
            sim_train = compute_metrics(sim_train_trades)
            sim_test = compute_metrics(sim_test_trades)

            d_train = float(sim_train.get("pnl", 0.0) - base_train.get("pnl", 0.0))
            d_test = float(sim_test.get("pnl", 0.0) - base_test.get("pnl", 0.0))
            d_edge_test = float(sim_test.get("edge", 0.0) - base_test.get("edge", 0.0))
            cov = float(sim_test.get("n", 0) / max(1, base_test.get("n", 0)))

            fold_pass = bool(d_train > 0.0 and d_test > min_test_delta and d_edge_test >= edge_floor and cov >= min_cov)
            if fold_pass:
                pass_count += 1
            train_deltas.append(d_train)
            test_deltas.append(d_test)
            fold_rows.append(
                {
                    "fold": idx,
                    "train_end": int(train_end),
                    "test_start": int(test_start),
                    "test_end": int(test_end),
                    "train_pnl_delta": round(d_train, 6),
                    "test_pnl_delta": round(d_test, 6),
                    "test_edge_delta": round(d_edge_test, 6),
                    "test_coverage": round(cov, 4),
                    "pass": bool(fold_pass),
                }
            )

        n_folds = max(1, len(fold_rows))
        pass_rate = float(pass_count / n_folds)
        train_mean = float(np.mean(np.asarray(train_deltas, dtype=np.float64))) if train_deltas else 0.0
        test_mean = float(np.mean(np.asarray(test_deltas, dtype=np.float64))) if test_deltas else 0.0
        test_med = float(np.median(np.asarray(test_deltas, dtype=np.float64))) if test_deltas else 0.0

        overfit_gap = max(0.0, train_mean - test_mean)
        overfit_ratio = overfit_gap / max(1.0, abs(train_mean))
        penalty_overfit = max(0.25, 1.0 - min(0.75, overfit_ratio))
        penalty_pass = max(0.25, min(1.0, pass_rate / max(min_pass_rate, 1e-6)))
        penalty_factor = float(max(0.20, min(1.0, penalty_overfit * penalty_pass)))

        passed = bool(pass_rate >= min_pass_rate and test_mean >= min_test_delta and test_med >= min_test_delta)
        return {
            "enabled": True,
            "pass": bool(passed),
            "folds": int(n_folds),
            "pass_rate": float(pass_rate),
            "train_pnl_delta_mean": float(train_mean),
            "test_pnl_delta_mean": float(test_mean),
            "test_pnl_delta_median": float(test_med),
            "penalty_factor": float(penalty_factor),
            "min_pass_rate": float(min_pass_rate),
            "min_test_pnl_delta": float(min_test_delta),
            "rows": fold_rows,
        }

    def _run_stage(self, sim: StageSimulator, max_combos: int) -> list[Finding]:
        """Sweep parameter grid for a single stage."""
        try:
            if hasattr(sim, "bind_context"):
                getattr(sim, "bind_context")(self.trades)
        except Exception as e:
            logger.debug(f"[CF_STAGE] bind_context skipped for {sim.stage_name}: {e}")
        grid = sim.get_param_grid()
        param_names = list(grid.keys())
        param_values = list(grid.values())
        baseline_combo = _baseline_combo_from_env(param_names, param_values)
        try:
            stage_max_combos = int(getattr(sim, "max_combos_hint", max_combos) or max_combos)
        except Exception:
            stage_max_combos = int(max_combos)
        stage_max_combos = int(max(1, min(int(max_combos), stage_max_combos)))
        combos, combo_mode, total_combos = self._select_stage_combos(
            sim=sim,
            param_values=param_values,
            stage_max_combos=stage_max_combos,
            baseline_combo=baseline_combo,
        )
        logger.info(
            f"[CF_STAGE] {sim.stage_name} search={combo_mode} "
            f"evaluated={len(combos)}/{total_combos} cap={stage_max_combos}"
        )

        stage_findings: list[Finding] = []
        best_delta_pnl = 0.0
        best_result = None
        min_significance = float(max(0.0, min(1.0, _env_float("CF_SIGNIFICANCE_MIN", 0.30))))
        wf_eval_min_sig = float(max(0.0, min(1.0, _env_float("CF_WF_EVAL_MIN_SIGNIFICANCE", 0.20))))
        wf_require_pass = bool(_env_bool("CF_WF_REQUIRE_PASS", True))

        for combo in combos:
            params = dict(zip(param_names, combo))
            scenario_id = f"{sim.stage_name}_{hash(tuple(sorted(params.items()))) % 100000:05d}"

            try:
                modified_trades = sim.simulate(self.trades, params)
                if not modified_trades:
                    continue
                simulated = compute_metrics(modified_trades)

                delta = {}
                for k in self.baseline:
                    if isinstance(self.baseline[k], (int, float)):
                        delta[k] = round(simulated[k] - self.baseline[k], 6)

                sig_raw = float(self._compute_significance(delta, simulated, self.baseline))
                oos = {
                    "enabled": False,
                    "pass": True,
                    "folds": 0,
                    "pass_rate": 1.0,
                    "train_pnl_delta_mean": 0.0,
                    "test_pnl_delta_mean": 0.0,
                    "penalty_factor": 1.0,
                }
                if delta.get("pnl", 0.0) > 0.0 and sig_raw >= wf_eval_min_sig:
                    oos = self._evaluate_walk_forward(sim, params)
                oos_penalty = float(oos.get("penalty_factor", 1.0) or 1.0)
                oos_pass = bool(oos.get("pass", True))
                sig_adj = float(max(0.0, min(1.0, sig_raw * max(0.0, oos_penalty))))
                if wf_require_pass and bool(oos.get("enabled", False)) and (not oos_pass):
                    sig_adj = float(min(sig_adj, min_significance * 0.95))

                delta["oos_train_pnl_delta"] = round(float(oos.get("train_pnl_delta_mean", 0.0) or 0.0), 6)
                delta["oos_test_pnl_delta"] = round(float(oos.get("test_pnl_delta_mean", 0.0) or 0.0), 6)
                delta["oos_pass_rate"] = round(float(oos.get("pass_rate", 0.0) or 0.0), 6)
                delta["oos_penalty_factor"] = round(float(oos_penalty), 6)

                result = CFResult(
                    scenario_id=scenario_id,
                    stage=sim.stage_name,
                    param_changes=params,
                    baseline=self.baseline,
                    simulated=simulated,
                    delta=delta,
                    n_trades=simulated["n"],
                    significance=sig_adj,
                    oos=oos,
                )
                self.results.append(result)

                # Check if this is a significant finding
                if result.significance >= min_significance and delta.get("pnl", 0) > best_delta_pnl:
                    if wf_require_pass and bool(oos.get("enabled", False)) and (not bool(oos.get("pass", False))):
                        continue
                    best_delta_pnl = delta.get("pnl", 0)
                    best_result = result

            except Exception as e:
                logger.warning(f"CF sim error {scenario_id}: {e}")

        if best_result and best_result.significance >= min_significance:
            self._finding_id += 1
            oos_obj = best_result.oos if isinstance(best_result.oos, dict) else {}
            oos_penalty = float(oos_obj.get("penalty_factor", 1.0) or 1.0)
            effective_delta_pnl = float(best_result.delta.get("pnl", 0.0) * oos_penalty)
            finding = Finding(
                finding_id=f"F{self._finding_id:04d}",
                timestamp=time.time(),
                stage=sim.stage_name,
                title=f"{sim.stage_name}: OOS-adjusted PnL +${effective_delta_pnl:.2f}",
                description=f"Stage: {sim.description}\n"
                           f"Params: {best_result.param_changes}\n"
                           f"Baseline: WR={self.baseline['wr']:.1%} R:R={self.baseline['rr']:.2f} PnL=${self.baseline['pnl']:.2f}\n"
                           f"CF: WR={best_result.simulated['wr']:.1%} R:R={best_result.simulated['rr']:.2f} PnL=${best_result.simulated['pnl']:.2f}\n"
                           f"OOS: pass={bool(oos_obj.get('pass', False))} "
                           f"rate={float(oos_obj.get('pass_rate', 0.0) or 0.0):.0%} "
                           f"trainΔ={float(oos_obj.get('train_pnl_delta_mean', 0.0) or 0.0):+.2f} "
                           f"testΔ={float(oos_obj.get('test_pnl_delta_mean', 0.0) or 0.0):+.2f} "
                           f"penalty={oos_penalty:.2f}",
                improvement_pct=float(effective_delta_pnl),
                confidence=best_result.significance,
                param_changes=best_result.param_changes,
                baseline_metrics=best_result.baseline,
                improved_metrics=best_result.simulated,
                recommendation=self._generate_recommendation(sim, best_result),
                oos_pass=bool(oos_obj.get("pass", False)),
                oos_folds=int(oos_obj.get("folds", 0) or 0),
                oos_pass_rate=float(oos_obj.get("pass_rate", 0.0) or 0.0),
                oos_train_pnl_delta=float(oos_obj.get("train_pnl_delta_mean", 0.0) or 0.0),
                oos_test_pnl_delta=float(oos_obj.get("test_pnl_delta_mean", 0.0) or 0.0),
                oos_penalty_factor=float(oos_penalty),
            )
            self.findings.append(finding)
            stage_findings.append(finding)
            logger.info(f"[FINDING] {finding.title} (confidence={finding.confidence:.2f})")

        return stage_findings

    def _compute_significance(self, delta: dict, simulated: dict, baseline: dict) -> float:
        """Score 0-1: how meaningful is this improvement?"""
        score = 0.0
        # PnL improvement
        pnl_delta = delta.get("pnl", 0)
        if pnl_delta > 0:
            score += min(0.4, pnl_delta / max(abs(baseline["pnl"]), 1.0) * 0.4)
        # WR improvement
        wr_delta = delta.get("wr", 0)
        if wr_delta > 0:
            score += min(0.2, wr_delta * 2.0)
        # Edge improvement (most important)
        edge_delta = delta.get("edge", 0)
        if edge_delta > 0:
            score += min(0.2, edge_delta * 5.0)
        # Sample size penalty
        n = simulated.get("n", 0)
        if n < 30:
            score *= 0.3
        elif n < 100:
            score *= 0.6
        elif n < 500:
            score *= 0.8
        # R:R improvement
        rr_delta = delta.get("rr", 0)
        if rr_delta > 0:
            score += min(0.2, rr_delta * 0.2)
        return min(1.0, score)

    def _generate_recommendation(self, sim: StageSimulator, result: CFResult) -> str:
        """Generate actionable recommendation."""
        lines = [f"[{sim.stage_name.upper()}] 파라미터 변경 제안:"]
        for k, v in result.param_changes.items():
            lines.append(f"  {k} = {v}")
        oos = result.oos if isinstance(result.oos, dict) else {}
        oos_penalty = float(oos.get("penalty_factor", 1.0) or 1.0)
        oos_adj_pnl = float(result.delta.get("pnl", 0.0) * oos_penalty)
        lines.append(
            f"예상 효과: PnL ${result.delta.get('pnl', 0):+.2f} "
            f"(OOS 보정 ${oos_adj_pnl:+.2f}), "
            f"WR {result.delta.get('wr', 0):+.1%}, "
            f"R:R {result.delta.get('rr', 0):+.2f}"
        )
        if bool(oos.get("enabled", False)):
            lines.append(
                f"OOS 검증: pass={bool(oos.get('pass', False))} "
                f"rate={float(oos.get('pass_rate', 0.0) or 0.0):.0%} "
                f"trainΔ={float(oos.get('train_pnl_delta_mean', 0.0) or 0.0):+.2f} "
                f"testΔ={float(oos.get('test_pnl_delta_mean', 0.0) or 0.0):+.2f} "
                f"penalty={oos_penalty:.2f}"
            )
        lines.append(f"신뢰도: {result.significance:.1%}")
        return "\n".join(lines)

    def get_top_findings(self, n: int = 10) -> list[Finding]:
        """Return top N findings sorted by improvement."""
        return sorted(self.findings, key=lambda f: f.improvement_pct, reverse=True)[:n]

    def to_json(self) -> dict:
        """Serialize all findings to JSON-safe dict."""
        return {
            "timestamp": time.time(),
            "baseline": self.baseline,
            "baseline_by_regime": self.baseline_by_regime,
            "n_trades": len(self.trades),
            "n_results": len(self.results),
            "findings": [asdict(f) for f in self.findings],
            "top_10": [asdict(f) for f in self.get_top_findings(10)],
        }

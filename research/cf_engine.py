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
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("research.cf_engine")

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
    applied: bool = False


# ─────────────────────────────────────────────────────────────────
# Trade Loader
# ─────────────────────────────────────────────────────────────────

class TradeLoader:
    """Load closed trade round-trips from SQLite."""

    def __init__(self, db_path: str = "state/bot_data_live.db"):
        self.db_path = db_path

    def load_trades(self, limit: int = 0, since_ms: int = 0) -> list[Trade]:
        """Load paired ENTER+EXIT trades."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM trades ORDER BY id ASC"
        rows = conn.execute(query).fetchall()
        conn.close()

        # Pair ENTER/EXIT by entry_link_id
        enters: dict[str, dict] = {}
        trades: list[Trade] = []
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
                    sigma=float(raw.get("lev_sigma_used") or raw.get("lev_sigma_raw") or 0.5),
                    entry_quality=float(entry.get("entry_quality_score") or 0),
                    leverage_signal=float(entry.get("leverage_signal_score") or 0),
                    raw=raw,
                    timestamp_ms=ts,
                )
                trades.append(t)

        if limit > 0:
            trades = trades[-limit:]
        logger.info(f"Loaded {len(trades)} round-trip trades from {self.db_path}")
        return trades


# ─────────────────────────────────────────────────────────────────
# Metrics Calculator
# ─────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[Trade]) -> dict:
    """Compute aggregate performance metrics."""
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "rr": 0, "sharpe": 0, "avg_pnl": 0,
                "avg_win": 0, "avg_loss": 0, "max_dd": 0, "pf": 0, "edge": 0}
    n = len(trades)
    pnls = np.array([t.realized_pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    total_pnl = float(pnls.sum())
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
        "n": n, "pnl": round(total_pnl, 4), "wr": round(wr, 4),
        "rr": round(rr, 4), "sharpe": round(sharpe, 4), "avg_pnl": round(avg_pnl, 6),
        "avg_win": round(avg_win, 6), "avg_loss": round(avg_loss, 6),
        "max_dd": round(max_dd, 4), "pf": round(pf, 4), "edge": round(edge, 4),
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
            "net_expectancy_min": [-0.002, -0.001, -0.0005, -0.0002, 0.0, 0.0005, 0.001, 0.002],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        ne_min = params.get("net_expectancy_min", -0.0002)
        result = []
        for t in trades:
            # entry_ev is a proxy for net expectancy
            if t.entry_ev >= ne_min:
                result.append(t)
        return result


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

    def get_param_grid(self) -> dict[str, list]:
        return {
            "top_n_symbols": [4, 8, 12, 20, 30, 50],
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

    def get_param_grid(self) -> dict[str, list]:
        return {
            "dir_gate_min_conf": [0.52, 0.55, 0.58, 0.60, 0.65, 0.70],
            "dir_gate_min_edge": [0.00, 0.02, 0.04, 0.06, 0.08, 0.10],
        }

    def simulate(self, trades: list[Trade], params: dict) -> list[Trade]:
        min_conf = float(params.get("dir_gate_min_conf", 0.58))
        min_edge = float(params.get("dir_gate_min_edge", 0.06))
        result = []
        for t in trades:
            # edge proxy: abs(mu_alpha)
            edge = abs(float(t.mu_alpha or 0.0))
            if float(t.dir_conf or 0.0) >= min_conf and edge >= min_edge:
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
    ExitReasonSimulator(),
    CapitalAllocationSimulator(),
    RegimeMultiplierSimulator(),
    # New expanded simulators
    SpreadFilterSimulator(),
    NetExpectancySimulator(),
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
]


class CFEngine:
    """
    Counterfactual Simulation Engine.
    Sweeps parameter grids per pipeline stage and discovers significant improvements.
    """

    def __init__(self, trades: list[Trade], simulators: list[StageSimulator] | None = None):
        self.trades = trades
        self.simulators = simulators or ALL_SIMULATORS
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

    def _run_stage(self, sim: StageSimulator, max_combos: int) -> list[Finding]:
        """Sweep parameter grid for a single stage."""
        grid = sim.get_param_grid()
        param_names = list(grid.keys())
        param_values = list(grid.values())
        try:
            stage_max_combos = int(getattr(sim, "max_combos_hint", max_combos) or max_combos)
        except Exception:
            stage_max_combos = int(max_combos)
        stage_max_combos = int(max(1, min(int(max_combos), stage_max_combos)))

        # Generate all combinations (capped)
        combos = list(itertools.product(*param_values))
        if len(combos) > stage_max_combos:
            # Random sample
            rng = np.random.default_rng(42)
            indices = rng.choice(len(combos), size=stage_max_combos, replace=False)
            combos = [combos[i] for i in indices]

        stage_findings: list[Finding] = []
        best_delta_pnl = 0.0
        best_result = None

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

                result = CFResult(
                    scenario_id=scenario_id,
                    stage=sim.stage_name,
                    param_changes=params,
                    baseline=self.baseline,
                    simulated=simulated,
                    delta=delta,
                    n_trades=simulated["n"],
                    significance=self._compute_significance(delta, simulated, self.baseline),
                )
                self.results.append(result)

                # Check if this is a significant finding
                if result.significance >= 0.3 and delta.get("pnl", 0) > best_delta_pnl:
                    best_delta_pnl = delta.get("pnl", 0)
                    best_result = result

            except Exception as e:
                logger.warning(f"CF sim error {scenario_id}: {e}")

        if best_result and best_result.significance >= 0.3:
            self._finding_id += 1
            finding = Finding(
                finding_id=f"F{self._finding_id:04d}",
                timestamp=time.time(),
                stage=sim.stage_name,
                title=f"{sim.stage_name}: PnL +${best_result.delta.get('pnl', 0):.2f}",
                description=f"Stage: {sim.description}\n"
                           f"Params: {best_result.param_changes}\n"
                           f"Baseline: WR={self.baseline['wr']:.1%} R:R={self.baseline['rr']:.2f} PnL=${self.baseline['pnl']:.2f}\n"
                           f"CF: WR={best_result.simulated['wr']:.1%} R:R={best_result.simulated['rr']:.2f} PnL=${best_result.simulated['pnl']:.2f}",
                improvement_pct=float(best_result.delta.get("pnl", 0)),
                confidence=best_result.significance,
                param_changes=best_result.param_changes,
                baseline_metrics=best_result.baseline,
                improved_metrics=best_result.simulated,
                recommendation=self._generate_recommendation(sim, best_result),
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
        lines.append(f"예상 효과: PnL ${result.delta.get('pnl', 0):+.2f}, "
                     f"WR {result.delta.get('wr', 0):+.1%}, "
                     f"R:R {result.delta.get('rr', 0):+.2f}")
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

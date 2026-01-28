from __future__ import annotations

import os
from dataclasses import dataclass


# Johnson SU shape parameters (skew/kurtosis) with env overrides
from engines.mc.config import config as mc_config

JOHNSON_SU_GAMMA = float(getattr(mc_config, "johnson_su_gamma", 0.0))
JOHNSON_SU_DELTA = float(getattr(mc_config, "johnson_su_delta", 1.0))


@dataclass
class MCParams:
    min_win: float
    profit_target: float
    ofi_weight: float
    max_kelly: float
    cvar_alpha: float
    cvar_scale: float
    n_paths: int
    # Optional probability overrides by regime
    p_pos_floor_enter: Optional[float] = None
    p_pos_floor_hold: Optional[float] = None
    p_sl_enter_ceiling: Optional[float] = None
    p_sl_hold_ceiling: Optional[float] = None
    p_tp_floor_enter: Optional[float] = None
    p_tp_floor_hold: Optional[float] = None


DEFAULT_PARAMS = {
    # Trend-following (More aggressive, allow some SL breathing room, lower TP requirement to stay in)
    "bull": MCParams(
        min_win=0.51, profit_target=0.0020, ofi_weight=0.0015, max_kelly=0.25, cvar_alpha=0.05, cvar_scale=6.0, n_paths=16000,
        p_pos_floor_enter=0.51, p_pos_floor_hold=0.48, p_sl_enter_ceiling=0.25, p_sl_hold_ceiling=0.30, p_tp_floor_enter=0.12, p_tp_floor_hold=0.10
    ),
    "bear": MCParams(
        min_win=0.51, profit_target=0.0020, ofi_weight=0.0018, max_kelly=0.20, cvar_alpha=0.05, cvar_scale=7.0, n_paths=16000,
        p_pos_floor_enter=0.51, p_pos_floor_hold=0.48, p_sl_enter_ceiling=0.25, p_sl_hold_ceiling=0.30, p_tp_floor_enter=0.12, p_tp_floor_hold=0.10
    ),
    # Mean-reverting / Scapy (More defensive, strict SL, high TP requirement to enter)
    "chop": MCParams(
        min_win=0.53, profit_target=0.0015, ofi_weight=0.0022, max_kelly=0.10, cvar_alpha=0.05, cvar_scale=8.0, n_paths=16000,
        p_pos_floor_enter=0.53, p_pos_floor_hold=0.51, p_sl_enter_ceiling=0.15, p_sl_hold_ceiling=0.20, p_tp_floor_enter=0.18, p_tp_floor_hold=0.15
    ),
    # High Volatility (Very strict SL to avoid being liquidated, requires high win rate to enter)
    "volatile": MCParams(
        min_win=0.55, profit_target=0.0030, ofi_weight=0.0025, max_kelly=0.15, cvar_alpha=0.05, cvar_scale=10.0, n_paths=16000,
        p_pos_floor_enter=0.55, p_pos_floor_hold=0.52, p_sl_enter_ceiling=0.10, p_sl_hold_ceiling=0.15, p_tp_floor_enter=0.20, p_tp_floor_hold=0.18
    ),
}

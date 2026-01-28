from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def get_env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")

def get_env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        if v is None: return default
        return float(v)
    except (ValueError, TypeError):
        return default

def get_env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name)
        if v is None: return default
        return int(v)
    except (ValueError, TypeError):
        return default

@dataclass
class MCConfig:
    # --- Performance & Simulation ---
    n_paths_live: int = field(default_factory=lambda: get_env_int("MC_N_PATHS_LIVE", 32768))
    n_paths_exit: int = field(default_factory=lambda: get_env_int("MC_N_PATHS_EXIT", 32768))
    verbose_print: bool = field(default_factory=lambda: get_env_bool("MC_VERBOSE_PRINT", False))
    jax_device: str = field(default_factory=lambda: os.environ.get("JAX_MC_DEVICE", "").strip().lower())
    time_step_sec: int = field(default_factory=lambda: get_env_int("MC_TIME_STEP_SEC", 1))
    logret_clip: float = field(default_factory=lambda: get_env_float("MC_LOGRET_CLIP", 12.0))
    verify_drift: bool = field(default_factory=lambda: get_env_bool("MC_VERIFY_DRIFT", False))
    cvar_n_boot: int = field(default_factory=lambda: get_env_int("MC_N_BOOT", 40))
    use_global_batching: bool = field(default_factory=lambda: get_env_bool("USE_GLOBAL_BATCHING", True))
    johnson_su_gamma: float = field(default_factory=lambda: get_env_float("MC_JOHNSON_SU_GAMMA", 0.0))
    johnson_su_delta: float = field(default_factory=lambda: get_env_float("MC_JOHNSON_SU_DELTA", 1.0))
    skip_exit_policy: bool = field(default_factory=lambda: get_env_bool("SKIP_EXIT_POLICY", False))
    
    # --- Alpha / Signal Features ---
    # ALPHA_SIGNAL_BOOST: 신호 강화 모드 (true면 공격적 설정 적용)
    alpha_signal_boost: bool = field(default_factory=lambda: get_env_bool("ALPHA_SIGNAL_BOOST", False))
    # mu_alpha_cap: 연율 기대수익 상한
    mu_alpha_cap: float = field(default_factory=lambda: get_env_float("MU_ALPHA_CAP", 15.0 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else 5.0))
    # alpha_scaling_factor: 신호 강화 시 1.5, 기본 1.0 (normalized from 3.0)
    alpha_scaling_factor: float = field(default_factory=lambda: get_env_float("ALPHA_SCALING_FACTOR", 1.5 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else 1.0))
    mu_mom_lr_cap: float = field(default_factory=lambda: get_env_float("MU_MOM_LR_CAP", 0.15 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else 0.10))
    mu_mom_tau_floor_sec: float = field(default_factory=lambda: get_env_float("MU_MOM_TAU_FLOOR_SEC", 900.0 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else 1800.0))
    # mu_mom_ann_cap: 연율 모멘텀 상한 (신호 강화 시 10.0)
    mu_mom_ann_cap: float = field(default_factory=lambda: get_env_float("MU_MOM_ANN_CAP", 10.0 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else 3.0))
    # mu_alpha_floor: mu_alpha 하한
    mu_alpha_floor: float = field(default_factory=lambda: get_env_float("MU_ALPHA_FLOOR", -10.0 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else -3.0))
    # mu_ofi_scale: OFI의 연율 알파 변환 계수 (신호 강화 시 15.0, normalized from 30.0)
    mu_ofi_scale: float = field(default_factory=lambda: get_env_float("MU_OFI_SCALE", 15.0 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else 10.0))
    mu_alpha_scale_min: float = field(default_factory=lambda: get_env_float("MU_ALPHA_SCALE_MIN", 0.5))
    mu_alpha_chop_window_bars: int = field(default_factory=lambda: get_env_int("MU_ALPHA_CHOP_WINDOW_BARS", 60))
    mu_alpha_w_mom_base: float = field(default_factory=lambda: get_env_float("MU_ALPHA_W_MOM_BASE", 0.50))
    mu_alpha_w_mom_min: float = field(default_factory=lambda: get_env_float("MU_ALPHA_W_MOM_MIN", 0.50))
    mu_alpha_w_mom_max: float = field(default_factory=lambda: get_env_float("MU_ALPHA_W_MOM_MAX", 0.90))
    mu_alpha_vol_short_bars: int = field(default_factory=lambda: get_env_int("MU_ALPHA_VOL_SHORT_BARS", 30))
    mu_alpha_vol_long_bars: int = field(default_factory=lambda: get_env_int("MU_ALPHA_VOL_LONG_BARS", 120))
    mu_alpha_ema_alpha: float = field(default_factory=lambda: get_env_float("MU_ALPHA_EMA_ALPHA", 0.0))
    sigma_lr_cap: float = field(default_factory=lambda: get_env_float("SIGMA_LR_CAP", 0.02))
    
    # --- Execution Costs & Spread ---
    slippage_mult: float = field(default_factory=lambda: get_env_float("SLIPPAGE_MULT", 0.3))
    slippage_cap: float = field(default_factory=lambda: get_env_float("SLIPPAGE_CAP", 0.0003))
    spread_pct_max: float = field(default_factory=lambda: get_env_float("SPREAD_PCT_MAX", 0.0005))
    exec_mode: str = field(default_factory=lambda: os.environ.get("EXEC_MODE", "maker_then_market").strip().lower())
    pmaker_prob: float = field(default_factory=lambda: get_env_float("PMAKER_PROB", 0.9))
    pmaker_delay_penalty_k: float = field(default_factory=lambda: get_env_float("PMAKER_DELAY_PENALTY_K", 1.0))
    pmaker_mu_alpha_boost_enabled: bool = field(default_factory=lambda: get_env_bool("PMAKER_MU_ALPHA_BOOST_ENABLED", True))
    pmaker_mu_alpha_boost_k: float = field(default_factory=lambda: get_env_float("PMAKER_MU_ALPHA_BOOST_K", 0.15))
    pmaker_adverse_cost_mult: float = field(default_factory=lambda: get_env_float("PMAKER_ADVERSE_COST_MULT", 1.0))
    use_gross_score: bool = field(default_factory=lambda: get_env_bool("USE_GROSS_SCORE", False))
    score_entry_threshold: float = field(default_factory=lambda: get_env_float("SCORE_ENTRY_THRESHOLD", 0.0003))
    score_entry_floor: float = field(default_factory=lambda: get_env_float("SCORE_ENTRY_FLOOR", 0.00015))
    score_exit_threshold: float = field(default_factory=lambda: get_env_float("SCORE_EXIT_THRESHOLD", 0.00015))
    score_extend_mult: float = field(default_factory=lambda: get_env_float("SCORE_EXTEND_MULT", 1.2))
    score_time_decay_k: float = field(default_factory=lambda: get_env_float("SCORE_TIME_DECAY_K", 0.6))
    score_exit_min_hold_sec: int = field(default_factory=lambda: get_env_int("SCORE_EXIT_MIN_HOLD_SEC", 0))
    score_tp_floor: float = field(default_factory=lambda: get_env_float("SCORE_TP_FLOOR", 0.05))
    score_tp_power: float = field(default_factory=lambda: get_env_float("SCORE_TP_POWER", 1.0))
    score_tp_weight: float = field(default_factory=lambda: get_env_float("SCORE_TP_WEIGHT", 1.0))
    score_tp_entry_min: float = field(default_factory=lambda: get_env_float("SCORE_TP_ENTRY_MIN", 0.05))
    score_tp_entry_hard: float = field(default_factory=lambda: get_env_float("SCORE_TP_ENTRY_HARD", 0.02))
    score_tp_entry_gate_mult: float = field(default_factory=lambda: get_env_float("SCORE_TP_ENTRY_GATE_MULT", 2.0))
    score_tp_entry_horizon_sec: int = field(default_factory=lambda: get_env_int("SCORE_TP_ENTRY_HORIZON_SEC", 300))

    # --- Policy Weighting & Objective ---
    policy_w_ev_beta: float = field(default_factory=lambda: get_env_float("POLICY_W_EV_BETA", 200.0))
    exit_early_delta_sec: float = field(default_factory=lambda: get_env_float("EXIT_EARLY_DELTA_SEC", 60.0))
    exit_early_penalty_k: float = field(default_factory=lambda: get_env_float("EXIT_EARLY_PENALTY_K", 0.5))
    policy_objective_mode: str = field(default_factory=lambda: os.environ.get("POLICY_OBJECTIVE_MODE", "ratio_time_var").strip().lower())
    policy_lambda_var: float = field(default_factory=lambda: get_env_float("POLICY_LAMBDA_VAR", 2.0))
    policy_cvar_eps: float = field(default_factory=lambda: get_env_float("POLICY_CVAR_EPS", 1e-6))
    policy_max_p_liq: float = field(default_factory=lambda: get_env_float("POLICY_MAX_P_LIQ", 1e-4))
    policy_min_profit_cost: float = float(get_env_float("POLICY_MIN_PROFIT_COST", 1.2))
    policy_max_dd_abs: float = field(default_factory=lambda: get_env_float("POLICY_MAX_DD_ABS", 0.10))
    # Minimum LONG-vs-SHORT edge required to take a directional trade when both sides look viable.
    # Units: ROE (e.g., 0.0002 = 2 bps). A small positive default reduces noisy direction flips in chop.
    policy_min_ev_gap: float = field(default_factory=lambda: get_env_float("POLICY_MIN_EV_GAP", 0.0002))
    # Unified Psi score parameters
    unified_risk_lambda: float = field(default_factory=lambda: get_env_float("UNIFIED_RISK_LAMBDA", 1.0))
    unified_rho: float = field(default_factory=lambda: get_env_float("UNIFIED_RHO", 0.0))

    # --- Neighbor / Consensus Logic ---
    policy_neighbor_bonus_w: float = field(default_factory=lambda: get_env_float("POLICY_NEIGHBOR_BONUS_W", 0.25))
    policy_neighbor_bonus_cap: float = field(default_factory=lambda: get_env_float("POLICY_NEIGHBOR_BONUS_CAP", 0.0015))
    policy_neighbor_penalty_w: float = field(default_factory=lambda: get_env_float("POLICY_NEIGHBOR_PENALTY_W", 0.25))
    policy_neighbor_penalty_cap: float = field(default_factory=lambda: get_env_float("POLICY_NEIGHBOR_PENALTY_CAP", 0.0015))
    policy_neighbor_oppose_veto_abs: float = field(default_factory=lambda: get_env_float("POLICY_NEIGHBOR_OPPOSE_VETO_ABS", 0.005))
    policy_local_consensus_alpha: float = field(default_factory=lambda: get_env_float("POLICY_LOCAL_CONSENSUS_ALPHA", 0.3))
    policy_local_consensus_base_h: float = field(default_factory=lambda: get_env_float("POLICY_LOCAL_CONSENSUS_BASE_H", 60.0))
    exit_mode: str = field(default_factory=lambda: os.environ.get("EXIT_MODE", "").strip().lower())
    
    # --- Funnel Filters ---
    funnel_win_floor_bull: float = field(default_factory=lambda: get_env_float("FUNNEL_WIN_FLOOR_BULL", 0.50))
    funnel_win_floor_bear: float = field(default_factory=lambda: get_env_float("FUNNEL_WIN_FLOOR_BEAR", 0.50))
    funnel_win_floor_chop: float = field(default_factory=lambda: get_env_float("FUNNEL_WIN_FLOOR_CHOP", 0.50))
    funnel_win_floor_volatile: float = field(default_factory=lambda: get_env_float("FUNNEL_WIN_FLOOR_VOLATILE", 0.50))
    
    funnel_cvar_floor_bull: float = field(default_factory=lambda: get_env_float("FUNNEL_CVAR_FLOOR_BULL", -0.12))
    funnel_cvar_floor_bear: float = field(default_factory=lambda: get_env_float("FUNNEL_CVAR_FLOOR_BEAR", -0.12))
    funnel_cvar_floor_chop: float = field(default_factory=lambda: get_env_float("FUNNEL_CVAR_FLOOR_CHOP", -0.10))
    funnel_cvar_floor_volatile: float = field(default_factory=lambda: get_env_float("FUNNEL_CVAR_FLOOR_VOLATILE", -0.09))
    
    funnel_event_cvar_floor_bull: float = field(default_factory=lambda: get_env_float("FUNNEL_EVENT_CVAR_FLOOR_BULL", -1.25))
    funnel_event_cvar_floor_bear: float = field(default_factory=lambda: get_env_float("FUNNEL_EVENT_CVAR_FLOOR_BEAR", -1.25))
    funnel_event_cvar_floor_chop: float = field(default_factory=lambda: get_env_float("FUNNEL_EVENT_CVAR_FLOOR_CHOP", -1.15))
    funnel_event_cvar_floor_volatile: float = field(default_factory=lambda: get_env_float("FUNNEL_EVENT_CVAR_FLOOR_VOLATILE", -1.20))
    
    score_entry_min_size: float = field(default_factory=lambda: get_env_float("SCORE_ENTRY_MIN_SIZE", 0.01))
    k_lev: float = field(default_factory=lambda: get_env_float("K_LEV", 2000.0))
    score_only_mode: bool = field(default_factory=lambda: get_env_bool("SCORE_ONLY_MODE", False))
    use_gpu_leverage: bool = field(default_factory=lambda: get_env_bool("USE_GPU_LEVERAGE", True))
    funnel_use_winrate_filter: bool = field(default_factory=lambda: get_env_bool("FUNNEL_USE_WINRATE_FILTER", False))
    funnel_use_napv_filter: bool = field(default_factory=lambda: get_env_bool("FUNNEL_USE_NAPV_FILTER", False)) # ✅ Hybrid: Disable hard NAPV filter for entries
    funnel_napv_threshold: float = field(default_factory=lambda: get_env_float("FUNNEL_NAPV_THRESHOLD", 0.0001))
    
    # Kelly Concentration Boost
    kelly_boost_enabled: bool = field(default_factory=lambda: get_env_bool("KELLY_BOOST_ENABLED", True))

    ev_cost_mult_gate: float = field(default_factory=lambda: get_env_float("EV_COST_MULT_GATE", 0.0))
    default_tp_pct: float = field(default_factory=lambda: get_env_float("DEFAULT_TP_PCT", 0.006))
    default_sl_pct: float = field(default_factory=lambda: get_env_float("DEFAULT_SL_PCT", 0.005))

    # --- AlphaHit ML ---
    alpha_hit_enable: bool = field(default_factory=lambda: get_env_bool("ALPHA_HIT_ENABLE", True))
    alpha_hit_beta: float = field(default_factory=lambda: get_env_float("ALPHA_HIT_BETA", 1.0))
    alpha_hit_tp_atr_mult: float = field(default_factory=lambda: get_env_float("ALPHA_HIT_TP_ATR_MULT", 2.0))
    alpha_hit_model_path: str = field(default_factory=lambda: os.environ.get("ALPHA_HIT_MODEL_PATH", "state/alpha_hit_mlp.pt"))
    alpha_hit_device: str = field(default_factory=lambda: os.environ.get("ALPHA_HIT_DEVICE", "mps"))
    alpha_hit_lr: float = field(default_factory=lambda: get_env_float("ALPHA_HIT_LR", 2e-4))
    alpha_hit_batch_size: int = field(default_factory=lambda: get_env_int("ALPHA_HIT_BATCH_SIZE", 256))
    alpha_hit_steps_per_tick: int = field(default_factory=lambda: get_env_int("ALPHA_HIT_STEPS_PER_TICK", 2))
    alpha_hit_max_buffer: int = field(default_factory=lambda: get_env_int("ALPHA_HIT_MAX_BUFFER", 200000))
    alpha_hit_min_buffer: int = field(default_factory=lambda: get_env_int("ALPHA_HIT_MIN_BUFFER", 1024))
    alpha_hit_max_loss: float = field(default_factory=lambda: get_env_float("ALPHA_HIT_MAX_LOSS", 2.0))
    alpha_hit_data_half_life_sec: float = field(default_factory=lambda: get_env_float("ALPHA_HIT_DATA_HALF_LIFE_SEC", 3600.0))
    alpha_hit_fallback: str = field(default_factory=lambda: os.environ.get("ALPHA_HIT_FALLBACK", "mc_to_hitprob").strip().lower())

    # --- Alpha/PMaker Delay ---
    pmaker_delay_penalty_mult: float = field(default_factory=lambda: get_env_float("PMAKER_DELAY_PENALTY_MULT", 1.0))
    pmaker_exit_delay_penalty_mult: float = field(default_factory=lambda: get_env_float("PMAKER_EXIT_DELAY_PENALTY_MULT", 1.0))
    alpha_delay_decay_tau_sec: float = field(default_factory=lambda: get_env_float("ALPHA_DELAY_DECAY_TAU_SEC", 30.0))
    pmaker_entry_delay_shift: bool = field(default_factory=lambda: get_env_bool("PMAKER_ENTRY_DELAY_SHIFT", True))
    pmaker_strict: bool = field(default_factory=lambda: get_env_bool("PMAKER_STRICT", False))
    
    # --- Exit Policy ---
    policy_enable_dd_stop: bool = field(default_factory=lambda: get_env_bool("POLICY_ENABLE_DD_STOP", True))
    policy_dd_stop_roe: float = field(default_factory=lambda: get_env_float("POLICY_DD_STOP_ROE", -0.01))
    policy_w_prior_half_life_base_sec: float = field(default_factory=lambda: get_env_float("POLICY_W_PRIOR_HALF_LIFE_BASE_SEC", 3600.0))
    sl_r_fixed: float = field(default_factory=lambda: get_env_float("SL_R_FIXED", 0.0020))
    trail_atr_mult: float = field(default_factory=lambda: get_env_float("TRAIL_ATR_MULT", 2.0))
    policy_min_hold_frac: float = field(default_factory=lambda: get_env_bool("POLICY_MIN_HOLD_FRAC", False))

    # --- TP/SL Autoscale ---
    tpsl_autoscale: bool = field(default_factory=lambda: get_env_bool("MC_TPSL_AUTOSCALE", True))
    tp_base_roe: float = field(default_factory=lambda: get_env_float("MC_TP_BASE_ROE", 0.0015))
    sl_base_roe: float = field(default_factory=lambda: get_env_float("MC_SL_BASE_ROE", 0.0020))
    tpsl_sigma_ref: float = field(default_factory=lambda: get_env_float("MC_TPSL_SIGMA_REF", 0.5))
    tpsl_sigma_min_scale: float = field(default_factory=lambda: get_env_float("MC_TPSL_SIGMA_MIN_SCALE", 0.6))
    tpsl_sigma_max_scale: float = field(default_factory=lambda: get_env_float("MC_TPSL_SIGMA_MAX_SCALE", 2.5))
    tpsl_h_scale_base: float = field(default_factory=lambda: get_env_float("MC_TPSL_H_SCALE_BASE", 60.0))

    # --- Portfolio Joint Simulation ---
    portfolio_enabled: bool = field(default_factory=lambda: get_env_bool("PORTFOLIO_JOINT_SIM_ENABLED", False))
    portfolio_days: int = field(default_factory=lambda: get_env_int("PORTFOLIO_DAYS", 3))
    portfolio_simulations: int = field(default_factory=lambda: get_env_int("PORTFOLIO_SIMULATIONS", 30000))
    portfolio_batch_size: int = field(default_factory=lambda: get_env_int("PORTFOLIO_BATCH_SIZE", 6000))
    portfolio_block_size: int = field(default_factory=lambda: get_env_int("PORTFOLIO_BLOCK_SIZE", 12))
    portfolio_drift_k: float = field(default_factory=lambda: get_env_float("PORTFOLIO_DRIFT_K", 0.35))
    portfolio_score_clip: float = field(default_factory=lambda: get_env_float("PORTFOLIO_SCORE_CLIP", 1.0))
    portfolio_tilt_strength: float = field(default_factory=lambda: get_env_float("PORTFOLIO_TILT_STRENGTH", 0.6))  # conservative (was 1.5)
    portfolio_use_jumps: bool = field(default_factory=lambda: get_env_bool("PORTFOLIO_USE_JUMPS", True))
    portfolio_p_jump_market: float = field(default_factory=lambda: get_env_float("PORTFOLIO_P_JUMP_MARKET", 0.005))
    portfolio_p_jump_idio: float = field(default_factory=lambda: get_env_float("PORTFOLIO_P_JUMP_IDIO", 0.007))
    portfolio_target_leverage: float = field(default_factory=lambda: get_env_float("PORTFOLIO_TARGET_LEVERAGE", 10.0))
    portfolio_individual_cap: float = field(default_factory=lambda: get_env_float("PORTFOLIO_INDIVIDUAL_CAP", 3.0))
    portfolio_risk_aversion: float = field(default_factory=lambda: get_env_float("PORTFOLIO_RISK_AVERSION", 0.5))
    portfolio_var_alpha: float = field(default_factory=lambda: get_env_float("PORTFOLIO_VAR_ALPHA", 0.05))

    # --- PMAKER model ---
    pmaker_enable: bool = field(default_factory=lambda: get_env_bool("PMAKER_ENABLE", False))
    pmaker_model_path: str = field(default_factory=lambda: os.environ.get("PMAKER_MODEL_PATH", "state/pmaker_survival_mlp.pt"))
    pmaker_train_steps: int = field(default_factory=lambda: get_env_int("PMAKER_TRAIN_STEPS", 1))
    pmaker_batch: int = field(default_factory=lambda: get_env_int("PMAKER_BATCH", 32))
    pmaker_grid_ms: int = field(default_factory=lambda: get_env_int("PMAKER_GRID_MS", 50))
    pmaker_max_ms: int = field(default_factory=lambda: get_env_int("PMAKER_MAX_MS", 2500))
    pmaker_lr: float = field(default_factory=lambda: get_env_float("PMAKER_LR", 3e-4))
    pmaker_device: str = field(default_factory=lambda: os.environ.get("PMAKER_DEVICE", "").strip())

    # --- Execution Costs ---
    p_maker_fixed: Optional[str] = field(default_factory=lambda: os.environ.get("P_MAKER_FIXED"))

    @classmethod
    def from_env(cls) -> MCConfig:
        return cls()

# Singleton instance
config = MCConfig.from_env()

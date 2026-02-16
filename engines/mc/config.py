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
    # mu_alpha_cap: 레거시 관측/호환용 스케일 (hard clamp에는 사용하지 않음)
    # 하드 캡 제거 이후에도 기존 env 호환을 위해 유지
    mu_alpha_cap: float = field(default_factory=lambda: get_env_float("MU_ALPHA_CAP", 5.0))
    # alpha_scaling_factor: 신호 배율
    # [FIX 2026-02-09] BOOST 모드 1.5→1.2: cap 축소에 맞춰 과도한 스케일링 방지
    alpha_scaling_factor: float = field(default_factory=lambda: get_env_float("ALPHA_SCALING_FACTOR", 1.2 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else 1.0))
    mu_mom_lr_cap: float = field(default_factory=lambda: get_env_float("MU_MOM_LR_CAP", 0.15 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else 0.10))
    mu_mom_tau_floor_sec: float = field(default_factory=lambda: get_env_float("MU_MOM_TAU_FLOOR_SEC", 900.0 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else 1800.0))
    # mu_mom_ann_cap: 연율 모멘텀 상한 (신호 강화 시 10.0)
    mu_mom_ann_cap: float = field(default_factory=lambda: get_env_float("MU_MOM_ANN_CAP", 10.0 if get_env_bool("ALPHA_SIGNAL_BOOST", False) else 3.0))
    # mu_alpha_floor: 레거시 관측/호환용 하한 스케일 (hard clamp 비사용)
    mu_alpha_floor: float = field(default_factory=lambda: get_env_float("MU_ALPHA_FLOOR", -5.0))
    # mu_ofi_scale: OFI의 연율 알파 변환 계수
    # [FIX 2026-02-09] 15→8: OFI 0.3만으로도 mu_ofi=4.5→cap 미도달. Gradient 보존
    mu_ofi_scale: float = field(default_factory=lambda: get_env_float("MU_OFI_SCALE", 8.0))
    mu_alpha_scale_min: float = field(default_factory=lambda: get_env_float("MU_ALPHA_SCALE_MIN", 0.5))
    mu_alpha_chop_window_bars: int = field(default_factory=lambda: get_env_int("MU_ALPHA_CHOP_WINDOW_BARS", 60))
    mu_alpha_w_mom_base: float = field(default_factory=lambda: get_env_float("MU_ALPHA_W_MOM_BASE", 0.50))
    mu_alpha_w_mom_min: float = field(default_factory=lambda: get_env_float("MU_ALPHA_W_MOM_MIN", 0.50))
    mu_alpha_w_mom_max: float = field(default_factory=lambda: get_env_float("MU_ALPHA_W_MOM_MAX", 0.90))
    mu_alpha_vol_short_bars: int = field(default_factory=lambda: get_env_int("MU_ALPHA_VOL_SHORT_BARS", 30))
    mu_alpha_vol_long_bars: int = field(default_factory=lambda: get_env_int("MU_ALPHA_VOL_LONG_BARS", 120))
    mu_alpha_ema_alpha: float = field(default_factory=lambda: get_env_float("MU_ALPHA_EMA_ALPHA", 0.0))
    sigma_lr_cap: float = field(default_factory=lambda: get_env_float("SIGMA_LR_CAP", 0.02))

    # --- Advanced mu_alpha components (optional; default disabled) ---
    alpha_use_mlofi: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_MLOFI", False))
    alpha_use_vpin: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_VPIN", False))
    alpha_use_kf: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_KF", False))
    alpha_use_hurst: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_HURST", False))
    alpha_use_garch: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_GARCH", False))
    alpha_use_bayes: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_BAYES", False))
    alpha_use_arima: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_ARIMA", False))
    alpha_use_ml: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_ML", False))
    alpha_use_hawkes: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_HAWKES", False))
    alpha_use_pf: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_PF", False))
    alpha_use_causal: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_CAUSAL", False))
    alpha_use_hmm: bool = field(default_factory=lambda: get_env_bool("ALPHA_USE_HMM", False))
    # Directional correction layer (logistic / tree-style classifier output)
    alpha_direction_use: bool = field(default_factory=lambda: get_env_bool("ALPHA_DIRECTION_USE", False))
    alpha_direction_model_path: str = field(default_factory=lambda: str(os.environ.get("ALPHA_DIRECTION_MODEL_PATH", "state/mu_direction_model.json")).strip())
    alpha_direction_reload_sec: float = field(default_factory=lambda: get_env_float("ALPHA_DIRECTION_RELOAD_SEC", 60.0))
    alpha_direction_strength: float = field(default_factory=lambda: get_env_float("ALPHA_DIRECTION_STRENGTH", 0.6))
    alpha_direction_min_confidence: float = field(default_factory=lambda: get_env_float("ALPHA_DIRECTION_MIN_CONFIDENCE", 0.55))
    alpha_direction_confidence_floor: float = field(default_factory=lambda: get_env_float("ALPHA_DIRECTION_CONFIDENCE_FLOOR", 0.45))
    alpha_direction_gate_weak_mu: float = field(default_factory=lambda: get_env_float("ALPHA_DIRECTION_GATE_WEAK_MU", 0.02))
    alpha_direction_gate_weak_gap: float = field(default_factory=lambda: get_env_float("ALPHA_DIRECTION_GATE_WEAK_GAP", 0.0020))
    alpha_direction_gate_conf_boost: float = field(default_factory=lambda: get_env_float("ALPHA_DIRECTION_GATE_CONF_BOOST", 0.05))
    alpha_direction_gate_edge_boost: float = field(default_factory=lambda: get_env_float("ALPHA_DIRECTION_GATE_EDGE_BOOST", 0.03))
    alpha_direction_gate_side_prob_boost: float = field(default_factory=lambda: get_env_float("ALPHA_DIRECTION_GATE_SIDE_PROB_BOOST", 0.03))

    # MLOFI
    mlofi_levels: int = field(default_factory=lambda: get_env_int("MLOFI_LEVELS", 5))
    mlofi_decay_lambda: float = field(default_factory=lambda: get_env_float("MLOFI_DECAY_LAMBDA", 0.4))
    mlofi_weight: float = field(default_factory=lambda: get_env_float("MLOFI_WEIGHT", 0.20))
    mlofi_scale: float = field(default_factory=lambda: get_env_float("MLOFI_SCALE", 8.0))
    mlofi_weight_path: str = field(default_factory=lambda: str(os.environ.get("MLOFI_WEIGHT_PATH", "")).strip())

    # VPIN
    vpin_bucket_count: int = field(default_factory=lambda: get_env_int("VPIN_BUCKET_COUNT", 50))
    vpin_window: int = field(default_factory=lambda: get_env_int("VPIN_WINDOW", 50))
    vpin_gamma: float = field(default_factory=lambda: get_env_float("VPIN_GAMMA", 0.6))
    vpin_sigma_k: float = field(default_factory=lambda: get_env_float("VPIN_SIGMA_K", 0.8))
    vpin_damp_floor: float = field(default_factory=lambda: get_env_float("VPIN_DAMP_FLOOR", 0.10))
    vpin_extreme_threshold: float = field(default_factory=lambda: get_env_float("VPIN_EXTREME_THRESHOLD", 0.90))
    vpin_extreme_ou_weight: float = field(default_factory=lambda: get_env_float("VPIN_EXTREME_OU_WEIGHT", 0.85))
    vpin_sigma_cap_mult: float = field(default_factory=lambda: get_env_float("VPIN_SIGMA_CAP_MULT", 2.5))

    # Kalman filter (constant velocity)
    kf_q: float = field(default_factory=lambda: get_env_float("KF_Q", 1e-6))
    kf_r: float = field(default_factory=lambda: get_env_float("KF_R", 1e-4))
    kf_weight: float = field(default_factory=lambda: get_env_float("KF_WEIGHT", 0.20))

    # Hurst / regime switching
    hurst_window: int = field(default_factory=lambda: get_env_int("HURST_WINDOW", 120))
    hurst_taus: str = field(default_factory=lambda: str(os.environ.get("HURST_TAUS", "2,4,8,16")))
    hurst_low: float = field(default_factory=lambda: get_env_float("HURST_LOW", 0.45))
    hurst_high: float = field(default_factory=lambda: get_env_float("HURST_HIGH", 0.55))
    hurst_trend_boost: float = field(default_factory=lambda: get_env_float("HURST_TREND_BOOST", 1.15))
    hurst_random_dampen: float = field(default_factory=lambda: get_env_float("HURST_RANDOM_DAMPEN", 0.75))
    ou_theta: float = field(default_factory=lambda: get_env_float("OU_THETA", 0.3))
    ou_weight: float = field(default_factory=lambda: get_env_float("OU_WEIGHT", 0.7))
    ou_mean_window: int = field(default_factory=lambda: get_env_int("OU_MEAN_WINDOW", 60))
    # HMM regime filter
    hmm_states: int = field(default_factory=lambda: get_env_int("HMM_STATES", 3))
    hmm_adapt_lr: float = field(default_factory=lambda: get_env_float("HMM_ADAPT_LR", 0.02))
    hmm_weight: float = field(default_factory=lambda: get_env_float("HMM_WEIGHT", 0.10))
    hmm_min_confidence: float = field(default_factory=lambda: get_env_float("HMM_MIN_CONFIDENCE", 0.55))
    hmm_trend_boost: float = field(default_factory=lambda: get_env_float("HMM_TREND_BOOST", 1.10))
    hmm_chop_dampen: float = field(default_factory=lambda: get_env_float("HMM_CHOP_DAMPEN", 0.85))

    # GARCH
    garch_omega: float = field(default_factory=lambda: get_env_float("GARCH_OMEGA", 1e-6))
    garch_alpha: float = field(default_factory=lambda: get_env_float("GARCH_ALPHA", 0.05))
    garch_beta: float = field(default_factory=lambda: get_env_float("GARCH_BETA", 0.90))
    garch_sigma_weight: float = field(default_factory=lambda: get_env_float("GARCH_SIGMA_WEIGHT", 0.5))

    # Bayesian mean
    bayes_obs_var: float = field(default_factory=lambda: get_env_float("BAYES_OBS_VAR", 1e-4))
    bayes_weight: float = field(default_factory=lambda: get_env_float("BAYES_WEIGHT", 0.10))

    # AR(1) / ARIMA proxy
    arima_weight: float = field(default_factory=lambda: get_env_float("ARIMA_WEIGHT", 0.10))

    # ML model (optional)
    ml_weight: float = field(default_factory=lambda: get_env_float("ML_WEIGHT", 0.15))
    ml_model_path: str = field(default_factory=lambda: str(os.environ.get("ML_MODEL_PATH", "")).strip())

    # Hawkes
    hawkes_alpha: float = field(default_factory=lambda: get_env_float("HAWKES_ALPHA", 0.5))
    hawkes_beta: float = field(default_factory=lambda: get_env_float("HAWKES_BETA", 2.0))
    hawkes_boost_k: float = field(default_factory=lambda: get_env_float("HAWKES_BOOST_K", 0.3))

    # Particle filter
    pf_particles: int = field(default_factory=lambda: get_env_int("PF_PARTICLES", 128))
    pf_weight: float = field(default_factory=lambda: get_env_float("PF_WEIGHT", 0.10))
    alpha_hurst_update_sec: float = field(default_factory=lambda: get_env_float("ALPHA_HURST_UPDATE_SEC", 60.0))
    alpha_pf_update_sec: float = field(default_factory=lambda: get_env_float("ALPHA_PF_UPDATE_SEC", 60.0))
    alpha_vpin_update_sec: float = field(default_factory=lambda: get_env_float("ALPHA_VPIN_UPDATE_SEC", 60.0))
    alpha_hmm_update_sec: float = field(default_factory=lambda: get_env_float("ALPHA_HMM_UPDATE_SEC", 60.0))
    alpha_garch_update_sec: float = field(default_factory=lambda: get_env_float("ALPHA_GARCH_UPDATE_SEC", 60.0))
    alpha_ml_update_sec: float = field(default_factory=lambda: get_env_float("ALPHA_ML_UPDATE_SEC", 5.0))

    # Causal adjustment
    causal_weights_path: str = field(default_factory=lambda: str(os.environ.get("CAUSAL_WEIGHTS_PATH", "")).strip())
    causal_weight: float = field(default_factory=lambda: get_env_float("CAUSAL_WEIGHT", 0.05))
    causal_reload_sec: float = field(default_factory=lambda: get_env_float("CAUSAL_RELOAD_SEC", 60.0))
    garch_param_path: str = field(default_factory=lambda: str(os.environ.get("GARCH_PARAM_PATH", "")).strip())
    garch_param_reload_sec: float = field(default_factory=lambda: get_env_float("GARCH_PARAM_RELOAD_SEC", 86400.0))
    garch_daily_fit_enabled: bool = field(default_factory=lambda: get_env_bool("GARCH_DAILY_FIT_ENABLED", False))
    garch_fit_interval_sec: float = field(default_factory=lambda: get_env_float("GARCH_FIT_INTERVAL_SEC", 86400.0))
    garch_fit_data_glob: str = field(default_factory=lambda: str(os.environ.get("GARCH_FIT_DATA_GLOB", "data/*.csv")).strip())
    garch_fit_lookback: int = field(default_factory=lambda: get_env_int("GARCH_FIT_LOOKBACK", 4000))
    garch_fit_min_obs: int = field(default_factory=lambda: get_env_int("GARCH_FIT_MIN_OBS", 300))
    garch_fit_timeout_sec: float = field(default_factory=lambda: get_env_float("GARCH_FIT_TIMEOUT_SEC", 120.0))
    garch_fit_allow_fallback: bool = field(default_factory=lambda: get_env_bool("GARCH_FIT_ALLOW_FALLBACK", True))
    
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
    exit_policy_dynamic_enabled: bool = field(default_factory=lambda: get_env_bool("EXIT_POLICY_DYNAMIC_ENABLED", True))
    exit_policy_mu_align_scale: float = field(default_factory=lambda: get_env_float("EXIT_POLICY_MU_ALIGN_SCALE", 1.5))
    exit_policy_shock_enabled: bool = field(default_factory=lambda: get_env_bool("EXIT_POLICY_SHOCK_ENABLED", True))
    exit_policy_shock_breakout_score: float = field(default_factory=lambda: get_env_float("EXIT_POLICY_SHOCK_BREAKOUT_SCORE", 0.0008))
    exit_policy_shock_trend_z: float = field(default_factory=lambda: get_env_float("EXIT_POLICY_SHOCK_TREND_Z", 2.0))
    exit_policy_noise_trend_z: float = field(default_factory=lambda: get_env_float("EXIT_POLICY_NOISE_TREND_Z", 1.0))
    exit_policy_noise_vpin_cap: float = field(default_factory=lambda: get_env_float("EXIT_POLICY_NOISE_VPIN_CAP", 0.80))
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
    # Small-gap policy: default to WAIT, optionally allow entry on high confidence
    policy_small_gap_allow_high_conf: bool = field(default_factory=lambda: get_env_bool("POLICY_SMALL_GAP_ALLOW_HIGH_CONF", True))
    policy_small_gap_confidence: float = field(default_factory=lambda: get_env_float("POLICY_SMALL_GAP_CONFIDENCE", 0.60))
    policy_small_gap_dir_confidence: float = field(default_factory=lambda: get_env_float("POLICY_SMALL_GAP_DIR_CONFIDENCE", 0.58))
    policy_small_gap_dir_edge: float = field(default_factory=lambda: get_env_float("POLICY_SMALL_GAP_DIR_EDGE", 0.08))

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
    leverage_conf_gain: float = field(default_factory=lambda: get_env_float("LEVERAGE_CONF_GAIN", 0.8))
    leverage_score_ref: float = field(default_factory=lambda: get_env_float("LEVERAGE_SCORE_REF", 0.003))
    leverage_hmm_gain: float = field(default_factory=lambda: get_env_float("LEVERAGE_HMM_GAIN", 0.5))
    leverage_sync_step: float = field(default_factory=lambda: get_env_float("LEVERAGE_SYNC_STEP", 1.0))
    leverage_sync_fallback_on_fail: bool = field(default_factory=lambda: get_env_bool("LEVERAGE_SYNC_FALLBACK_ON_FAIL", True))
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
    alpha_hit_warmup_samples: int = field(default_factory=lambda: get_env_int("ALPHA_HIT_WARMUP_SAMPLES", 512))
    alpha_hit_replay_path: str = field(default_factory=lambda: os.environ.get("ALPHA_HIT_REPLAY_PATH", "state/alpha_hit_replay.npz"))
    alpha_hit_replay_save_every: int = field(default_factory=lambda: get_env_int("ALPHA_HIT_REPLAY_SAVE_EVERY", 2000))

    # --- Alpha/PMaker Delay ---
    pmaker_delay_penalty_mult: float = field(default_factory=lambda: get_env_float("PMAKER_DELAY_PENALTY_MULT", 1.0))
    pmaker_exit_delay_penalty_mult: float = field(default_factory=lambda: get_env_float("PMAKER_EXIT_DELAY_PENALTY_MULT", 1.0))
    alpha_delay_decay_tau_sec: float = field(default_factory=lambda: get_env_float("ALPHA_DELAY_DECAY_TAU_SEC", 30.0))
    pmaker_entry_delay_shift: bool = field(default_factory=lambda: get_env_bool("PMAKER_ENTRY_DELAY_SHIFT", True))
    pmaker_strict: bool = field(default_factory=lambda: get_env_bool("PMAKER_STRICT", False))
    
    # --- Exit Policy ---
    policy_enable_dd_stop: bool = field(default_factory=lambda: get_env_bool("POLICY_ENABLE_DD_STOP", True))
    policy_dd_stop_roe: float = field(default_factory=lambda: get_env_float("POLICY_DD_STOP_ROE", -0.01))
    policy_cash_exit_enabled: bool = field(default_factory=lambda: get_env_bool("POLICY_CASH_EXIT_ENABLED", False))
    policy_cash_exit_score: float = field(default_factory=lambda: get_env_float("POLICY_CASH_EXIT_SCORE", 0.0))
    policy_max_hold_sec: int = field(default_factory=lambda: get_env_int("POLICY_MAX_HOLD_SEC", 0))
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
    portfolio_market_factor_scale: float = field(default_factory=lambda: get_env_float("PORTFOLIO_MARKET_FACTOR_SCALE", 1.0))
    portfolio_residual_scale: float = field(default_factory=lambda: get_env_float("PORTFOLIO_RESIDUAL_SCALE", 1.0))
    portfolio_rebalance_sim_enabled: bool = field(default_factory=lambda: get_env_bool("PORTFOLIO_REBAL_SIM_ENABLED", False))
    portfolio_rebalance_interval: int = field(default_factory=lambda: get_env_int("PORTFOLIO_REBAL_INTERVAL", 1))
    portfolio_rebalance_fee_bps: float = field(default_factory=lambda: get_env_float("PORTFOLIO_REBAL_FEE_BPS", 6.0))
    portfolio_rebalance_slip_bps: float = field(default_factory=lambda: get_env_float("PORTFOLIO_REBAL_SLIP_BPS", 4.0))
    portfolio_rebalance_score_noise: float = field(default_factory=lambda: get_env_float("PORTFOLIO_REBAL_SCORE_NOISE", 0.0))
    portfolio_rebalance_min_score: float = field(default_factory=lambda: get_env_float("PORTFOLIO_REBAL_MIN_SCORE", 0.0))

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

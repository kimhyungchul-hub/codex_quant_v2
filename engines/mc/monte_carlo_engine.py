from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

from engines.base import BaseEngine
from engines.mc.alpha_hit import MonteCarloAlphaHitMixin
from engines.mc.decision import MonteCarloDecisionMixin
from engines.mc.entry_evaluation import MonteCarloEntryEvaluationMixin
from engines.mc.execution_costs import MonteCarloExecutionCostsMixin
from engines.mc.execution_mix import MonteCarloExecutionMixMixin
from engines.mc.exit_policy import MonteCarloExitPolicyMixin
from engines.mc.first_passage import MonteCarloFirstPassageMixin
from engines.mc.path_simulation import MonteCarloPathSimulationMixin
from engines.mc.policy_weights import MonteCarloPolicyWeightsMixin
from engines.mc.runtime_params import MonteCarloRuntimeParamsMixin
from engines.mc.signal_features import MonteCarloSignalFeaturesMixin
from engines.mc.tail_sampling import MonteCarloTailSamplingMixin

try:
    from trainers.online_alpha_trainer import AlphaTrainerConfig, OnlineAlphaTrainer
    import torch

    _ALPHA_HIT_MLP_OK = True
except Exception:  # pragma: no cover
    AlphaTrainerConfig = None
    OnlineAlphaTrainer = None
    torch = None
    _ALPHA_HIT_MLP_OK = False

logger = logging.getLogger(__name__)


class MonteCarloEngine(
    MonteCarloSignalFeaturesMixin,
    MonteCarloRuntimeParamsMixin,
    MonteCarloExecutionCostsMixin,
    MonteCarloTailSamplingMixin,
    MonteCarloPathSimulationMixin,
    MonteCarloFirstPassageMixin,
    MonteCarloPolicyWeightsMixin,
    MonteCarloExecutionMixMixin,
    MonteCarloExitPolicyMixin,
    MonteCarloEntryEvaluationMixin,
    MonteCarloDecisionMixin,
    MonteCarloAlphaHitMixin,
    BaseEngine,
):
    """
    - ctx에서 regime_params를 받으면 그것을 우선 사용
    - 아니면 DEFAULT_PARAMS[regime]
    - 시뮬레이션은 numpy로 결정론 seed 사용(튜닝 안정)
    """

    name = "mc_barrier"
    weight = 1.0

    POLICY_P_POS_ENTER_BY_REGIME = {"bull": 0.51, "bear": 0.51, "chop": 0.53, "volatile": 0.55}
    POLICY_P_POS_HOLD_BY_REGIME = {"bull": 0.48, "bear": 0.48, "chop": 0.51, "volatile": 0.52}
    POLICY_P_SL_ENTER_MAX_BY_REGIME = {"bull": 0.25, "bear": 0.25, "chop": 0.15, "volatile": 0.10}
    POLICY_P_SL_HOLD_MAX_BY_REGIME = {"bull": 0.30, "bear": 0.30, "chop": 0.20, "volatile": 0.15}
    POLICY_P_TP_ENTER_MIN_BY_REGIME = {"bull": 0.12, "bear": 0.12, "chop": 0.18, "volatile": 0.20}
    POLICY_P_TP_HOLD_MIN_BY_REGIME = {"bull": 0.10, "bear": 0.10, "chop": 0.15, "volatile": 0.18}
    POLICY_P_SL_EMERGENCY = 0.38
    # Short-term (0~5m) policy defaults
    POLICY_HOLD_BAD_TICKS = 2
    FLIP_CONFIRM_TICKS = 2
    MIN_HOLD_SEC_DIRECTIONAL = 60
    # Mid-term default decision cadence (seconds)
    POLICY_DECISION_DT_SEC = 10
    POLICY_HORIZON_SEC = 3600
    # Mid-term horizons: 1m, 5m, 10m, 30m, 1h
    POLICY_MULTI_HORIZONS_SEC = (60, 300, 600, 1800, 3600)
    SCORE_MARGIN_DEFAULT = 0.0001
    POLICY_VALUE_SOFT_FLOOR_AFTER_COST = -0.0005
    N_PATHS_EXIT_POLICY = int(getattr(config, "n_paths_exit", 2048))

    TP_R_BY_H = {15: 0.0002, 30: 0.0003, 60: 0.0005, 120: 0.0008, 180: 0.0012, 300: 0.0015, 600: 0.0030, 1800: 0.0050, 3600: 0.0080}
    SL_R_FIXED = 0.0020
    TP_SL_AUTOSCALE = str(os.environ.get("MC_TPSL_AUTOSCALE", "1")).strip().lower() in ("1", "true", "yes")
    TP_SL_BASE_TP = float(os.environ.get("MC_TP_BASE_ROE", "0.0015"))
    TP_SL_BASE_SL = float(os.environ.get("MC_SL_BASE_ROE", str(SL_R_FIXED)))
    TP_SL_SIGMA_REF = float(os.environ.get("MC_TPSL_SIGMA_REF", "0.5"))
    TP_SL_SIGMA_MIN_SCALE = float(os.environ.get("MC_TPSL_SIGMA_MIN_SCALE", "0.6"))
    TP_SL_SIGMA_MAX_SCALE = float(os.environ.get("MC_TPSL_SIGMA_MAX_SCALE", "2.5"))
    TP_SL_H_SCALE_BASE = float(os.environ.get("MC_TPSL_H_SCALE_BASE", "60.0"))

    def _tp_sl_table(self, horizon_sec: float) -> tuple[float, float]:
        horizon_key = int(round(float(max(0.0, horizon_sec))))
        tp_default = float(next(iter(self.TP_R_BY_H.values()), self.TP_SL_BASE_TP))
        tp_r = float(self.TP_R_BY_H.get(horizon_key, tp_default)) if self.TP_R_BY_H else tp_default
        sl_r = float(getattr(self, "SL_R_FIXED", self.TP_SL_BASE_SL))
        return tp_r, sl_r

    def tp_sl_targets_for_horizon(self, horizon_sec: float, sigma: Optional[float]) -> tuple[float, float]:
        """Return price-change TP/SL targets for the given horizon and sigma."""
        if not bool(getattr(self, "TP_SL_AUTOSCALE", True)):
            return self._tp_sl_table(horizon_sec)

        h = float(max(1.0, horizon_sec))
        sigma_ref = float(max(1e-3, getattr(self, "TP_SL_SIGMA_REF", 0.5)))
        sigma_val = float(sigma) if sigma is not None and math.isfinite(float(sigma)) else sigma_ref
        if sigma_val <= 0:
            sigma_val = sigma_ref
        vol_scale = sigma_val / sigma_ref
        vol_scale = float(max(self.TP_SL_SIGMA_MIN_SCALE, min(self.TP_SL_SIGMA_MAX_SCALE, vol_scale)))
        horizon_scale = math.sqrt(h / max(1.0, getattr(self, "TP_SL_H_SCALE_BASE", 60.0)))
        tp_r = float(self.TP_SL_BASE_TP) * horizon_scale * vol_scale
        sl_r = float(self.TP_SL_BASE_SL) * horizon_scale * vol_scale
        return float(tp_r), float(sl_r)

    @staticmethod
    def alpha_hit_confidence(p_tp: float, p_sl: float) -> float:
        try:
            tp = float(max(0.0, min(1.0, float(p_tp))))
            sl = float(max(0.0, min(1.0, float(p_sl))))
            s = tp + sl
            if s > 1.0:
                tp = tp / s
                sl = sl / s
                s = 1.0
            other = max(0.0, 1.0 - s)
            eps = 1e-12
            entropy = 0.0
            for p in (tp, sl, other):
                if p > 0.0:
                    entropy -= p * math.log(max(p, eps))
            max_entropy = math.log(3.0)
            if max_entropy <= 0:
                return 0.0
            conf = 1.0 - (entropy / max_entropy)
            return float(max(0.0, min(1.0, conf)))
        except Exception:
            return 0.0
    TRAIL_ATR_MULT = 1.3
    PMAKER_DELAY_PENALTY_K = 1.0
    POLICY_W_EV_BETA = 200.0
    POLICY_MIN_EV_GAP = 0.00005

    def __init__(self):
        # Mid-term horizons
        self.horizons = (60, 300, 600, 1800, 3600)
        # Time resolution: one simulation step represents this many seconds.
        # Larger values reduce compute by reducing n_steps.
        self.time_step_sec = int(os.environ.get("MC_TIME_STEP_SEC", "1"))
        self.time_step_sec = int(max(1, self.time_step_sec))
        # dt is year-fraction per simulation step.
        self.dt = float(self.time_step_sec) / 31536000.0
        
        # Fee settings: USE_MAKER_ORDERS=true → Maker fee (0.02% roundtrip)
        _use_maker = os.environ.get("USE_MAKER_ORDERS", "true").lower() in ("1", "true", "yes")
        self.fee_roundtrip_base = 0.0002 if _use_maker else 0.0012  # Maker: 0.02%, Taker: 0.12%
        self.fee_roundtrip_maker_base = 0.0002  # 0.01% * 2 sides
        self.slippage_perc = 0.0001 if _use_maker else 0.0003  # Maker: ~0, Taker: slippage

        self.default_tail_mode = "student_t"
        self.default_student_t_df = 6.0
        self._use_jax = True
        self._tail_mode = self.default_tail_mode
        self._student_t_df = self.default_student_t_df
        self._bootstrap_returns = None
        self._ofi_hist: Dict[Tuple[str, str], List[float]] = {}
        self._gate_log_count = 0

        self.alpha_hit_enabled = _ALPHA_HIT_MLP_OK and str(os.environ.get("ALPHA_HIT_ENABLE", "1")).strip().lower() in ("1", "true", "yes")
        self.alpha_hit_beta = float(os.environ.get("ALPHA_HIT_BETA", "1.0"))
        self.alpha_hit_model_path = str(os.environ.get("ALPHA_HIT_MODEL_PATH", "state/alpha_hit_mlp.pt"))
        self.alpha_hit_trainer = None
        self.alpha_hit_mlp = None

        if self.alpha_hit_enabled and _ALPHA_HIT_MLP_OK and OnlineAlphaTrainer is not None and AlphaTrainerConfig is not None:
            try:
                policy_horizons = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", (15, 30, 60, 120, 180, 300)))
                n_features = 20

                trainer_cfg = AlphaTrainerConfig(
                    horizons_sec=policy_horizons,
                    n_features=n_features,
                    device=str(os.environ.get("ALPHA_HIT_DEVICE", "cuda" if (torch and torch.cuda.is_available()) else "cpu")),
                    lr=float(os.environ.get("ALPHA_HIT_LR", "2e-4")),
                    batch_size=int(os.environ.get("ALPHA_HIT_BATCH_SIZE", "256")),
                    steps_per_tick=int(os.environ.get("ALPHA_HIT_STEPS_PER_TICK", "2")),
                    max_buffer=int(os.environ.get("ALPHA_HIT_MAX_BUFFER", "200000")),
                    data_half_life_sec=float(os.environ.get("ALPHA_HIT_DATA_HALF_LIFE_SEC", "3600.0")),
                    ckpt_path=self.alpha_hit_model_path,
                    enable=True,
                )
                self.alpha_hit_trainer = OnlineAlphaTrainer(trainer_cfg)
                self.alpha_hit_mlp = self.alpha_hit_trainer.model
                logger.info(f"[ALPHA_HIT] Initialized OnlineAlphaTrainer with {len(policy_horizons)} horizons")
            except Exception as e:
                logger.warning(f"[ALPHA_HIT] Failed to initialize trainer: {e}")
                self.alpha_hit_trainer = None
                self.alpha_hit_mlp = None
                self.alpha_hit_enabled = False

        self._skip_goal_jax = False
        self._skip_first_passage_jax = False
        self._force_gaussian_dist = False
        self._force_zero_cost = False
        self._force_horizon_600 = False

        self.PMAKER_DELAY_PENALTY_MULT = float(os.getenv("PMAKER_DELAY_PENALTY_MULT", "1.0"))
        self.PMAKER_EXIT_DELAY_PENALTY_MULT = float(os.getenv("PMAKER_EXIT_DELAY_PENALTY_MULT", "1.0"))
        self.ALPHA_DELAY_DECAY_TAU_SEC = float(os.getenv("ALPHA_DELAY_DECAY_TAU_SEC", "30.0"))
        self.PMAKER_ENTRY_DELAY_SHIFT = os.getenv("PMAKER_ENTRY_DELAY_SHIFT", "1") == "1"
        self.PMAKER_STRICT = bool(os.getenv("PMAKER_STRICT", "0") == "1")

        # ============================================================================
        # [STATIC SHAPE WARMUP] JAX JIT 재컴파일 방지
        # ============================================================================
        # CRITICAL: 봇 시작 시 최대 크기로 워밍업하여 장중 렉 방지
        try:
            from engines.mc.entry_evaluation_vmap import GlobalBatchEvaluator
            from engines.mc.constants import (
                STATIC_MAX_SYMBOLS,
                STATIC_MAX_PATHS,
                STATIC_MAX_STEPS,
            )
            self._global_batch_evaluator = GlobalBatchEvaluator(
                dt=self.dt,
                max_symbols=STATIC_MAX_SYMBOLS
            )
            # 최대 크기로 워밍업 (장중 shape 변경 시 재컴파일 방지)
            logger.info(f"[MC_ENGINE] Starting Static Shape warmup: ({STATIC_MAX_SYMBOLS}, {STATIC_MAX_PATHS}, {STATIC_MAX_STEPS})")
            self._global_batch_evaluator.warmup(
                n_symbols=STATIC_MAX_SYMBOLS,
                n_paths=STATIC_MAX_PATHS,
                n_steps=STATIC_MAX_STEPS
            )
        except Exception as e:
            logger.warning(f"[MC_ENGINE] GlobalBatchEvaluator warmup failed: {e}")
            self._global_batch_evaluator = None

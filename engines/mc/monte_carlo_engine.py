from __future__ import annotations

import logging
import math
import numpy as np
import os
import re
import config as base_config
from engines.mc.config import config as mc_config
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
from engines.mc.lsm_switching_solver import LSMSwitchingSolver, LSMConfig
from engines.mc.trajectory_optimizer import BeamTrajectoryOptimizer, BeamConfig
from engines.mc.hybrid_planner import HybridPlanner, HybridPlannerConfig

AlphaTrainerConfig = None
OnlineAlphaTrainer = None
torch = None
_ALPHA_HIT_MLP_OK = False
_ALPHA_HIT_INIT_ERROR = None


def _maybe_load_alpha_trainer() -> None:
    global AlphaTrainerConfig, OnlineAlphaTrainer, torch, _ALPHA_HIT_MLP_OK, _ALPHA_HIT_INIT_ERROR
    if _ALPHA_HIT_MLP_OK:
        return
    if not bool(getattr(mc_config, "alpha_hit_enable", True)):
        _ALPHA_HIT_INIT_ERROR = "ALPHA_HIT_ENABLE=0"
        return
    use_torch_env = str(os.environ.get("MC_USE_TORCH", "1")).strip().lower()
    if use_torch_env in ("0", "false", "no", "off"):
        _ALPHA_HIT_INIT_ERROR = "MC_USE_TORCH=0"
        return
    try:
        from trainers.online_alpha_trainer import AlphaTrainerConfig as _AlphaTrainerConfig
        from trainers.online_alpha_trainer import OnlineAlphaTrainer as _OnlineAlphaTrainer
        import torch as _torch
        AlphaTrainerConfig = _AlphaTrainerConfig
        OnlineAlphaTrainer = _OnlineAlphaTrainer
        torch = _torch
        _ALPHA_HIT_MLP_OK = True
        _ALPHA_HIT_INIT_ERROR = None
    except Exception:  # pragma: no cover
        AlphaTrainerConfig = None
        OnlineAlphaTrainer = None
        torch = None
        _ALPHA_HIT_MLP_OK = False
        _ALPHA_HIT_INIT_ERROR = "torch_import_failed"

logger = logging.getLogger(__name__)

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return int(default)

def _parse_int_list_env(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return tuple(default)
    parts = re.split(r"[,\s]+", raw)
    vals: list[int] = []
    for p in parts:
        if not p:
            continue
        try:
            v = int(float(p))
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        vals.append(v)
    return tuple(vals) if vals else tuple(default)


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
    N_PATHS_EXIT_POLICY = int(getattr(mc_config, "n_paths_exit", 2048))

    TP_R_BY_H = {15: 0.0002, 30: 0.0003, 60: 0.0005, 120: 0.0008, 180: 0.0012, 300: 0.0015, 600: 0.0030, 1800: 0.0050, 3600: 0.0080}
    SL_R_FIXED = 0.0020
    TP_SL_AUTOSCALE = bool(getattr(mc_config, "tpsl_autoscale", True))
    TP_SL_BASE_TP = float(getattr(mc_config, "tp_base_roe", 0.0015))
    TP_SL_BASE_SL = float(getattr(mc_config, "sl_base_roe", SL_R_FIXED))
    TP_SL_SIGMA_REF = float(getattr(mc_config, "tpsl_sigma_ref", 0.5))
    TP_SL_SIGMA_MIN_SCALE = float(getattr(mc_config, "tpsl_sigma_min_scale", 0.6))
    TP_SL_SIGMA_MAX_SCALE = float(getattr(mc_config, "tpsl_sigma_max_scale", 2.5))
    TP_SL_H_SCALE_BASE = float(getattr(mc_config, "tpsl_h_scale_base", 60.0))

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
        policy_multi = _parse_int_list_env(
            "POLICY_MULTI_HORIZONS_SEC",
            tuple(getattr(self, "POLICY_MULTI_HORIZONS_SEC", (60, 300, 600, 1800, 3600))),
        )
        self.POLICY_MULTI_HORIZONS_SEC = policy_multi
        self.POLICY_HORIZON_SEC = max(
            1, _env_int("POLICY_HORIZON_SEC", int(getattr(self, "POLICY_HORIZON_SEC", 3600)))
        )
        self.POLICY_DECISION_DT_SEC = max(
            1, _env_int("POLICY_DECISION_DT_SEC", int(getattr(self, "POLICY_DECISION_DT_SEC", 10)))
        )
        # Align simulation horizons with policy horizons by default.
        if self.POLICY_MULTI_HORIZONS_SEC:
            self.horizons = tuple(self.POLICY_MULTI_HORIZONS_SEC)
        else:
            self.horizons = (60, 300, 600, 1800, 3600)
        # Time resolution: one simulation step represents this many seconds.
        # Larger values reduce compute by reducing n_steps.
        self.time_step_sec = int(getattr(mc_config, "time_step_sec", 1))
        self.time_step_sec = int(max(1, self.time_step_sec))
        # dt is year-fraction per simulation step.
        self.dt = float(self.time_step_sec) / 31536000.0
        
        # Fee settings: USE_MAKER_ORDERS=true → Maker fee (0.02% roundtrip)
        _use_maker = bool(getattr(base_config, "USE_MAKER_ORDERS", True))
        self.fee_roundtrip_base = 0.0002 if _use_maker else 0.0012  # Maker: 0.02%, Taker: 0.12%
        self.fee_roundtrip_maker_base = 0.0002  # 0.01% * 2 sides
        self.slippage_perc = 0.0001 if _use_maker else 0.0003  # Maker: ~0, Taker: slippage

        self.default_tail_mode = "student_t"
        self.default_student_t_df = 6.0
        # JAX disabled: use PyTorch-first with NumPy fallback.
        self._use_torch = str(os.environ.get("MC_USE_TORCH", "1")).strip().lower() in ("1", "true", "yes", "on")
        self._use_jax = False
        self._tail_mode = self.default_tail_mode
        self._student_t_df = self.default_student_t_df
        self._bootstrap_returns = None
        self._ofi_hist: Dict[Tuple[str, str], List[float]] = {}
        self._gate_log_count = 0

        self.alpha_hit_enabled = bool(getattr(mc_config, "alpha_hit_enable", True))
        self.alpha_hit_disable_reason = None
        if self.alpha_hit_enabled:
            _maybe_load_alpha_trainer()
        self.alpha_hit_enabled = self.alpha_hit_enabled and _ALPHA_HIT_MLP_OK
        if not self.alpha_hit_enabled:
            try:
                self.alpha_hit_disable_reason = _ALPHA_HIT_INIT_ERROR or "mlp_unavailable"
            except Exception:
                self.alpha_hit_disable_reason = "mlp_unavailable"
        self.alpha_hit_beta = float(getattr(mc_config, "alpha_hit_beta", 1.0))
        self.alpha_hit_model_path = str(getattr(mc_config, "alpha_hit_model_path", "state/alpha_hit_mlp.pt"))
        self.alpha_hit_trainer = None
        self.alpha_hit_mlp = None

        if self.alpha_hit_enabled and OnlineAlphaTrainer is not None and AlphaTrainerConfig is not None:
            try:
                policy_horizons = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", (15, 30, 60, 120, 180, 300)))
                n_features = 20
                device_cfg = str(getattr(mc_config, "alpha_hit_device", "") or "").strip().lower()
                if not device_cfg:
                    if torch and torch.cuda.is_available():
                        device_cfg = "cuda"
                    elif torch and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device_cfg = "mps"
                    else:
                        device_cfg = "cpu"
                if device_cfg == "mps":
                    if not (torch and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                        device_cfg = "cpu"

                trainer_cfg = AlphaTrainerConfig(
                    horizons_sec=policy_horizons,
                    n_features=n_features,
                    device=device_cfg,
                    lr=float(getattr(mc_config, "alpha_hit_lr", 2e-4)),
                    batch_size=int(getattr(mc_config, "alpha_hit_batch_size", 256)),
                    steps_per_tick=int(getattr(mc_config, "alpha_hit_steps_per_tick", 2)),
                    max_buffer=int(getattr(mc_config, "alpha_hit_max_buffer", 200000)),
                    min_buffer=int(getattr(mc_config, "alpha_hit_min_buffer", 1024)),
                    warmup_samples=int(getattr(mc_config, "alpha_hit_warmup_samples", 512)),
                    data_half_life_sec=float(getattr(mc_config, "alpha_hit_data_half_life_sec", 3600.0)),
                    ckpt_path=self.alpha_hit_model_path,
                    replay_path=str(getattr(mc_config, "alpha_hit_replay_path", "state/alpha_hit_replay.npz")),
                    replay_save_every=int(getattr(mc_config, "alpha_hit_replay_save_every", 2000)),
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
                self.alpha_hit_disable_reason = f"trainer_init_failed: {e}"

        self._skip_goal_jax = False
        self._skip_first_passage_jax = False
        self._force_gaussian_dist = False
        self._force_zero_cost = False
        self._force_horizon_600 = False

        self.PMAKER_DELAY_PENALTY_MULT = float(getattr(mc_config, "pmaker_delay_penalty_mult", 1.0))
        self.PMAKER_EXIT_DELAY_PENALTY_MULT = float(getattr(mc_config, "pmaker_exit_delay_penalty_mult", 1.0))
        self.ALPHA_DELAY_DECAY_TAU_SEC = float(getattr(mc_config, "alpha_delay_decay_tau_sec", 30.0))
        self.PMAKER_ENTRY_DELAY_SHIFT = bool(getattr(mc_config, "pmaker_entry_delay_shift", True))
        self.PMAKER_STRICT = bool(getattr(mc_config, "pmaker_strict", False))

        # Hybrid Planner components (lazy torch import)
        try:
            import torch as _torch
            mps_avail = bool(_torch.backends.mps.is_available()) if hasattr(_torch, "backends") and hasattr(_torch.backends, "mps") else False
            if mps_avail:
                self.device = _torch.device("mps")
            else:
                self.device = _torch.device("cpu")
            self._torch = _torch
            lsm_cfg = LSMConfig()
            try:
                fee_env = os.environ.get("HYBRID_BASE_FEE_RATE")
                if fee_env is not None and str(fee_env).strip() != "":
                    lsm_cfg.base_fee_rate = float(fee_env)
            except Exception:
                pass
            try:
                slip_env = os.environ.get("HYBRID_SLIPPAGE_BPS")
                if slip_env is not None and str(slip_env).strip() != "":
                    lsm_cfg.slippage_bps = float(slip_env)
            except Exception:
                pass
            try:
                use_dyn_env = os.environ.get("HYBRID_USE_DYNAMIC_COST")
                if use_dyn_env is not None and str(use_dyn_env).strip() != "":
                    lsm_cfg.use_dynamic_cost = str(use_dyn_env).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                pass
            self.lsm_solver = LSMSwitchingSolver(self.device, lsm_cfg)
            self.beam_optimizer = BeamTrajectoryOptimizer(self.device, BeamConfig())
            planner_cfg = HybridPlannerConfig()
            try:
                lsm_h_env = os.environ.get("MC_HYBRID_HORIZON_STEPS")
                if lsm_h_env is not None and str(lsm_h_env).strip() != "":
                    planner_cfg.lsm_horizon_steps = int(max(2, int(lsm_h_env)))
            except Exception:
                pass
            try:
                plan_steps_env = os.environ.get("HYBRID_BEAM_PLAN_STEPS")
                if plan_steps_env is not None and str(plan_steps_env).strip() != "":
                    self.beam_optimizer.cfg.plan_steps = int(max(2, int(plan_steps_env)))
            except Exception:
                pass
            self.hybrid_planner = HybridPlanner(self.device, self.lsm_solver, self.beam_optimizer, planner_cfg)
            try:
                fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
            except Exception:
                fallback_env = None
            msg = f"[HYBRID] torch_device={self.device} mps_available={mps_avail} mps_fallback={fallback_env}"
            logger.info(msg)
            try:
                print(msg, flush=True)
            except Exception:
                pass
        except Exception:
            self.device = None
            self._torch = None
            self.lsm_solver = None
            self.beam_optimizer = None
            self.hybrid_planner = None
        self.last_action_idx: dict[str, int] = {}
        self.cost_matrix = None

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
            # Torch GlobalBatchEvaluator currently only needs device; keep init minimal for compatibility.
            self._global_batch_evaluator = GlobalBatchEvaluator()
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

    # -------------------------
    # Hybrid Planner Integration
    # -------------------------
    def decide_with_hybrid_planner(self, symbol, current_state, mc_paths, return_detail: bool = False):
        """
        Use HybridPlanner to decide optimal action based on MC paths.
        mc_paths should contain:
          - prices: (P, H)
          - features: (P, H, 4) or individual vol/ofi/spread/mom arrays
        """
        if self.hybrid_planner is None or self._torch is None:
            raise RuntimeError("HybridPlanner not initialized (torch/MPS unavailable)")

        torch = self._torch
        device = self.device

        def _get_state_val(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        prices_raw = mc_paths["prices"]
        if isinstance(prices_raw, torch.Tensor):
            price_tensor = prices_raw.to(device=device, dtype=torch.float32)
        else:
            price_tensor = torch.as_tensor(prices_raw, device=device, dtype=torch.float32)
        if price_tensor.ndim == 2:
            price_tensor = price_tensor.unsqueeze(0)
        feature_tensor = self._extract_features_tensor(mc_paths).to(device=device, dtype=torch.float32)

        A = int(price_tensor.shape[0])
        num_pos = 1 + (2 * A if getattr(self.lsm_solver.cfg, "allow_short", True) else A)
        if self.cost_matrix is None or self.cost_matrix.shape[0] != num_pos:
            self.cost_matrix = self.lsm_solver._build_cost_matrix(A, num_pos).to(device=device)

        pos_val = _get_state_val(current_state, "position", None)
        if pos_val is None and isinstance(current_state, dict):
            pos_val = current_state.get("position_side")

        horizon_steps = int(max(0, price_tensor.shape[-1] - 1))
        exposure = float(mc_paths.get("exposure", 1.0) or 1.0)
        cash_penalty = float(mc_paths.get("cash_penalty", 0.0) or 0.0)
        result = self.hybrid_planner.step(
            current_capital=float(_get_state_val(current_state, "equity", 1.0)),
            current_pos_idx=self._map_pos_to_idx(pos_val, A),
            holding_time=int(_get_state_val(current_state, "holding_time_min", 0)),
            prev_action_idx=int(self.last_action_idx.get(symbol, 0)),
            price_paths=price_tensor,
            feature_paths=feature_tensor,
            exposure=exposure,
            cash_penalty=cash_penalty,
            cost_matrix=self.cost_matrix,
            state=current_state,
        )

        action_idx = int(result.get("action"))
        exp_score = float(result.get("score_beam", 0.0))
        decision = self._decode_action(action_idx, current_state)
        if hasattr(self, "logger"):
            try:
                self.logger.info(f"[{symbol}] Hybrid Plan: {decision} (Score: {exp_score:.4f})")
            except Exception:
                pass
        self.last_action_idx[symbol] = int(action_idx)
        if return_detail:
            return decision, {
                "action_idx": action_idx,
                "score": exp_score,
                "horizon_steps": horizon_steps,
                "plan_steps": int(getattr(getattr(self.beam_optimizer, "cfg", None), "plan_steps", 0) or 0) if self.beam_optimizer else None,
                "exposure": exposure,
                "cash_penalty": cash_penalty,
                "time_step_sec": float(mc_paths.get("time_step_sec", getattr(self, "time_step_sec", 1)) or 1),
                "raw": result,
            }
        return decision

    def _extract_features_tensor(self, mc_paths) -> "torch.Tensor":
        torch = self._torch
        if "features" in mc_paths:
            feats_raw = mc_paths["features"]
            if isinstance(feats_raw, torch.Tensor):
                feats = feats_raw.to(device=self.device, dtype=torch.float32)
            else:
                feats = torch.as_tensor(feats_raw, device=self.device, dtype=torch.float32)
        else:
            vol = torch.tensor(mc_paths.get("vol"), device=self.device, dtype=torch.float32)
            ofi = torch.tensor(mc_paths.get("ofi"), device=self.device, dtype=torch.float32)
            spread = torch.tensor(mc_paths.get("spread"), device=self.device, dtype=torch.float32)
            mom = torch.tensor(mc_paths.get("mom"), device=self.device, dtype=torch.float32)
            feats = torch.stack([vol, ofi, spread, mom], dim=-1)

        # Ensure shape (A, P, H, 4)
        if feats.ndim == 3:
            feats = feats.unsqueeze(0)
        if feats.shape[-1] < 4:
            pad = torch.zeros((*feats.shape[:-1], 4 - feats.shape[-1]), device=self.device, dtype=torch.float32)
            feats = torch.cat([feats, pad], dim=-1)
        return feats

    def _build_hybrid_feature_paths(self, price_paths_t, ctx) -> "torch.Tensor":
        """
        Build feature paths aligned with price paths.
        Feature indices: 0=Vol, 1=OFI, 2=Spread, 3=Momentum.
        """
        torch = self._torch
        device = price_paths_t.device
        eps = 1e-12
        P, H = price_paths_t.shape

        # Log returns for volatility proxy
        log_ret = torch.log(
            price_paths_t[:, 1:].clamp(min=eps) / price_paths_t[:, :-1].clamp(min=eps)
        )
        vol_window = int(os.environ.get("MC_HYBRID_VOL_WINDOW", 10))
        if vol_window <= 1 or log_ret.shape[1] < vol_window:
            vol = log_ret.abs()
        else:
            # Rolling std over window
            win = log_ret.unfold(1, vol_window, 1)  # (P, T-win+1, win)
            vol_mid = win.std(dim=-1, unbiased=False)
            pad = vol_mid[:, 0:1].repeat(1, vol_window - 1)
            vol = torch.cat([pad, vol_mid], dim=1)
        # Pad to length H (vol corresponds to steps 1..H-1)
        vol = torch.cat([torch.zeros((P, 1), device=device), vol], dim=1)

        # Momentum: log price ratio to t=0
        mom = torch.log(price_paths_t.clamp(min=eps) / price_paths_t[:, [0]].clamp(min=eps))

        # OFI (use ctx scalar if provided)
        ofi_val = 0.0
        if isinstance(ctx, dict):
            ofi_val = ctx.get("ofi_score", ctx.get("ofi", ctx.get("ofi_z", 0.0) or 0.0)) or 0.0
        try:
            ofi_val = float(ofi_val)
        except Exception:
            ofi_val = 0.0
        ofi = torch.full((P, H), float(ofi_val), device=device, dtype=torch.float32)

        # Spread (use ctx scalar if provided)
        spread_val = 0.0
        if isinstance(ctx, dict):
            spread_val = ctx.get("spread_pct", ctx.get("spread", ctx.get("bid_ask_spread", 0.0) or 0.0)) or 0.0
        try:
            spread_val = float(spread_val)
        except Exception:
            spread_val = 0.0
        spread = torch.full((P, H), float(spread_val), device=device, dtype=torch.float32)

        feats = torch.stack([vol, ofi, spread, mom], dim=-1)  # (P, H, 4)
        return feats

    def _build_hybrid_mc_paths(self, ctx, params, seed: int) -> dict:
        if self.hybrid_planner is None or self._torch is None:
            raise RuntimeError("HybridPlanner not initialized")

        torch = self._torch
        device = self.device

        step_sec = int(getattr(self, "time_step_sec", 1) or 1)
        try:
            step_env = os.environ.get("MC_HYBRID_TIME_STEP_SEC")
            if step_env is not None:
                step_sec = int(step_env)
        except Exception:
            pass
        step_sec = int(max(1, step_sec))
        horizon_steps = int(os.environ.get("MC_HYBRID_HORIZON_STEPS", getattr(self.hybrid_planner.cfg, "lsm_horizon_steps", 60)))
        horizon_steps = max(2, horizon_steps)
        n_steps = horizon_steps

        if hasattr(params, "n_paths"):
            n_paths_default = int(getattr(params, "n_paths"))
        elif isinstance(params, dict):
            n_paths_default = int(params.get("n_paths", mc_config.n_paths_live))
        else:
            n_paths_default = int(mc_config.n_paths_live)
        n_paths = int(os.environ.get("MC_HYBRID_N_PATHS", min(4096, n_paths_default)))
        n_paths = max(128, n_paths)

        price = float(ctx.get("price", 0.0) or 0.0)
        if price <= 0:
            price = float(ctx.get("last_price", 1.0) or 1.0)

        mu_base = ctx.get("mu_sim", ctx.get("mu_base", ctx.get("mu_alpha", 0.0)))
        sigma = ctx.get("sigma_sim", ctx.get("sigma", None))

        closes = ctx.get("closes") or ctx.get("close") or ctx.get("close_series")
        bar_seconds = float(ctx.get("bar_seconds", ctx.get("timeframe_sec", 60.0)) or 60.0)
        ofi_score = ctx.get("ofi_score", ctx.get("ofi", ctx.get("ofi_z", 0.0) or 0.0))
        try:
            ofi_score = float(ofi_score)
        except Exception:
            ofi_score = 0.0

        # Fallback sigma from closes
        if (sigma is None or float(sigma) <= 0.0) and closes:
            try:
                closes_np = np.asarray(closes, dtype=np.float64)
                rets = np.diff(np.log(closes_np))
                if rets.size >= 2:
                    sigma_bar = float(rets.std())
                    _, sigma = self._annualize(float(np.mean(rets)), sigma_bar, bar_seconds=bar_seconds)
            except Exception:
                sigma = None

        mu_alpha = ctx.get("mu_alpha", None)
        if mu_alpha is None and closes:
            try:
                mu_alpha = self._signal_alpha_mu_annual(closes, bar_seconds, ofi_score, str(ctx.get("regime", "chop")))
            except Exception:
                mu_alpha = 0.0

        mu = float(mu_base or mu_alpha or 0.0)
        sigma = float(sigma or 0.1)

        # Regime adjustment (align with entry evaluation)
        try:
            from regime import adjust_mu_sigma
            if mu_alpha is not None:
                mu, sigma = adjust_mu_sigma(float(mu_alpha), float(sigma), str(ctx.get("regime", "chop")))
        except Exception:
            pass
        sigma = max(float(sigma), 1e-6)
        mu_used = float(mu)
        sigma_used = float(sigma)

        # Exposure (size fraction * leverage) for leverage-aware log-utility
        try:
            lev_env = os.environ.get("HYBRID_LEVERAGE")
            force_lev = str(os.environ.get("HYBRID_FORCE_LEVERAGE", "0")).strip().lower() in ("1", "true", "yes", "on")
            if force_lev and lev_env is not None:
                leverage = float(lev_env)
            else:
                leverage = float(ctx.get("leverage", lev_env if lev_env is not None else 1.0) or 1.0)
        except Exception:
            leverage = 1.0
        try:
            size_env = os.environ.get("HYBRID_SIZE_FRAC")
            force_size = str(os.environ.get("HYBRID_FORCE_SIZE_FRAC", "0")).strip().lower() in ("1", "true", "yes", "on")
            if force_size and size_env is not None:
                size_frac = float(size_env)
            else:
                size_frac = float(ctx.get("size_frac", ctx.get("size_fraction", size_env if size_env is not None else 1.0)) or 1.0)
        except Exception:
            size_frac = 1.0
        # Dynamic leverage cap based on volatility
        try:
            use_vol_cap = str(os.environ.get("HYBRID_USE_VOL_CAP", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            use_vol_cap = True
        lev_cap = None
        if use_vol_cap:
            try:
                vol_ref = float(os.environ.get("HYBRID_VOL_REF", 0.6) or 0.6)
            except Exception:
                vol_ref = 0.6
            try:
                vol_slope = float(os.environ.get("HYBRID_VOL_SLOPE", 1.0) or 1.0)
            except Exception:
                vol_slope = 1.0
            try:
                lev_min = float(os.environ.get("HYBRID_LEV_MIN", 0.5) or 0.5)
            except Exception:
                lev_min = 0.5
            try:
                lev_max = float(os.environ.get("HYBRID_LEV_MAX", leverage) or leverage)
            except Exception:
                lev_max = float(leverage)
            sigma_safe = max(float(sigma_used), 1e-6)
            scale = (vol_ref / sigma_safe) ** max(vol_slope, 1e-6)
            lev_cap = max(lev_min, min(lev_max, lev_max * scale))
            leverage = min(leverage, lev_cap)
        try:
            max_exposure = float(os.environ.get("HYBRID_MAX_EXPOSURE", 5.0) or 5.0)
        except Exception:
            max_exposure = 5.0
        exposure = max(0.0, float(leverage) * float(size_frac))
        if max_exposure > 0:
            exposure = min(exposure, max_exposure)

        dt_used = float(step_sec) / 31536000.0
        price_paths = self.simulate_paths_price(
            seed=int(seed),
            s0=float(price),
            mu=float(mu),
            sigma=float(sigma),
            n_paths=int(n_paths),
            n_steps=int(n_steps),
            dt=float(dt_used),
            return_torch=True,
        )

        # Normalize to torch tensor on target device
        if hasattr(price_paths, "block_until_ready"):
            try:
                from engines.mc import jax_backend
                price_paths = np.asarray(jax_backend.to_numpy(price_paths), dtype=np.float32)
            except Exception:
                price_paths = np.asarray(price_paths, dtype=np.float32)
        if isinstance(price_paths, torch.Tensor):
            price_t = price_paths.to(device=device, dtype=torch.float32)
        else:
            price_t = torch.as_tensor(price_paths, device=device, dtype=torch.float32)
        if price_t.ndim != 2:
            price_t = price_t.reshape(price_t.shape[0], -1)

        feat_t = self._build_hybrid_feature_paths(price_t, ctx)
        return {
            "prices": price_t,
            "features": feat_t,
            "mu_used": mu_used,
            "sigma_used": sigma_used,
            "dt_used": float(dt_used),
            "time_step_sec": int(step_sec),
            "exposure": exposure,
            "leverage_used": leverage,
            "leverage_cap": lev_cap,
            "size_frac": size_frac,
            "cash_penalty": float(os.environ.get("HYBRID_CASH_PENALTY", 0.0) or 0.0),
        }

    def _map_pos_to_idx(self, position, num_assets: int) -> int:
        if position is None:
            return 0
        if isinstance(position, str):
            side = position.upper()
            if side == "LONG":
                return 1
            if side == "SHORT":
                return 1 + num_assets
            return 0
        try:
            side_val = float(position)
            if side_val > 0:
                return 1
            if side_val < 0:
                return 1 + num_assets
        except Exception:
            pass
        return 0

    def _decode_action(self, action_idx: int, current_state) -> str:
        # For single-asset usage: 0=cash, 1=long, 2=short
        if isinstance(current_state, dict):
            pos = current_state.get("position")
            if pos is None:
                pos = current_state.get("position_side")
        else:
            pos = getattr(current_state, "position", None)
        if action_idx == 0:
            if pos in ("LONG", "SHORT"):
                return "EXIT"
            return "HOLD"
        if action_idx == 1:
            return "LONG"
        return "SHORT"

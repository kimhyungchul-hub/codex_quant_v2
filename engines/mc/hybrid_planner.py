import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class HybridPlannerConfig:
    lsm_horizon_steps: int = 60
    switch_hysteresis: float = 1e-4
    sig_lock_dist: float = 0.05
    sig_lock_improve: float = 1e-4


class HybridPlanner:
    def __init__(
        self,
        device: torch.device,
        lsm_solver,
        beam_optimizer,
        cfg: Optional[HybridPlannerConfig] = None,
    ):
        self.device = device
        self.lsm = lsm_solver
        self.beam = beam_optimizer
        self.cfg = cfg or HybridPlannerConfig()

        self.prev_plan_actions = None
        self.prev_state_sig = None
        self.prev_action = None
        self.prev_score = None

    def step(self, state: Any = None, mc_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        # Accept either mc_data dict or explicit kwargs (integration snippet)
        if mc_data is None:
            mc_data = {}
        if kwargs:
            mc_data = {**mc_data, **kwargs}

        # Unpack MC inputs
        price_paths = mc_data["price_paths"]
        feature_paths = mc_data["feature_paths"]
        current_capital = float(mc_data["current_capital"])
        current_pos_idx = int(mc_data["current_pos_idx"])
        holding_time = int(mc_data.get("holding_time", 0))
        prev_action_idx = int(mc_data.get("prev_action_idx", current_pos_idx))
        exposure = float(mc_data.get("exposure", 1.0) or 1.0)
        try:
            cash_penalty = float(mc_data.get("cash_penalty", 0.0) or 0.0)
        except Exception:
            cash_penalty = 0.0
        cost_matrix = mc_data.get("cost_matrix")
        if cost_matrix is None:
            A = int(price_paths.shape[0])
            num_pos = 1 + (2 * A if getattr(self.lsm.cfg, "allow_short", True) else A)
            cost_matrix = self.lsm._build_cost_matrix(A, num_pos).to(self.device)
        try:
            horizon_steps = int(mc_data.get("horizon_steps", self.cfg.lsm_horizon_steps) or self.cfg.lsm_horizon_steps)
        except Exception:
            horizon_steps = int(self.cfg.lsm_horizon_steps)
        horizon_steps = max(2, horizon_steps)

        # 1) LSM for global guidance
        action_lsm, score_lsm, debug_lsm = self.lsm.solve(
            price_paths=price_paths,
            feature_paths=feature_paths,
            current_capital=current_capital,
            current_pos_idx=current_pos_idx,
            holding_time=holding_time,
            prev_action_idx=prev_action_idx,
            exposure=exposure,
            cash_penalty=cash_penalty,
            horizon_steps=int(horizon_steps),
            cost_matrix=cost_matrix,
        )
        betas = self.lsm._cached_betas_by_t

        # 2) Beam search for local optimization
        action_beam, score_beam, debug_beam = self.beam.optimize(
            current_capital=current_capital,
            current_pos=current_pos_idx,
            price_paths=price_paths,
            feature_paths=feature_paths,
            lsm_solver=self.lsm,
            cost_matrix=cost_matrix,
            betas=betas,
            holding_time=holding_time,
            prev_action=prev_action_idx,
            exposure=exposure,
            cash_penalty=cash_penalty,
            warm_start_actions=self.prev_plan_actions,
        )

        final_action = int(action_beam)
        curr_sig = self._compute_signature(state)

        # 3) Stability: hysteresis + state signature lock
        if self.prev_action is not None and final_action != self.prev_action:
            prev_score = float(self.prev_score) if self.prev_score is not None else None
            if prev_score is not None and (score_beam - prev_score) < self.cfg.switch_hysteresis:
                final_action = int(self.prev_action)
            elif self.prev_state_sig is not None and curr_sig is not None:
                # Signature length can change when feature schema changes during runtime.
                # In that case skip lock-distance check for this step instead of crashing.
                if curr_sig.numel() != self.prev_state_sig.numel():
                    dist = float("inf")
                else:
                    dist = torch.norm(curr_sig - self.prev_state_sig).item()
                if dist < self.cfg.sig_lock_dist and (score_beam - prev_score) < self.cfg.sig_lock_improve:
                    final_action = int(self.prev_action)

        # 4) Update memory
        self.prev_state_sig = curr_sig
        self.prev_action = final_action
        self.prev_score = float(score_beam)
        self.prev_plan_actions = debug_beam.get("best_path_actions")

        return {
            "action": final_action,
            "score_beam": float(score_beam),
            "score_lsm": float(score_lsm),
            "action_lsm": int(action_lsm),
            "debug_lsm": debug_lsm,
            "debug_beam": debug_beam,
        }

    def _compute_signature(self, state: Any) -> Optional[torch.Tensor]:
        if state is None:
            return None
        if not isinstance(state, torch.Tensor):
            # Try dict-like objects first for deterministic ordering.
            data = None
            if isinstance(state, dict):
                data = state
            else:
                try:
                    data = vars(state)
                except Exception:
                    data = None
            if data is not None:
                vals = []
                for k in sorted(data.keys()):
                    v = data.get(k)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                if vals:
                    state = torch.tensor(vals, device=self.device, dtype=torch.float32)
                else:
                    return None
            else:
                try:
                    state = torch.tensor(state, device=self.device, dtype=torch.float32)
                except Exception:
                    return None
        s = state.flatten()
        if s.numel() == 0:
            return None
        # Use a small prefix to stabilize signature size
        if s.numel() > 64:
            s = s[:64]
        norm = torch.norm(s) + 1e-8
        return s / norm

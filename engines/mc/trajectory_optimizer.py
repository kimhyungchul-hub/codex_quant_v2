import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


@dataclass
class BeamConfig:
    beam_width: int = 512
    plan_steps: int = 15
    switch_candidates: int = 5
    hysteresis: float = 1e-4
    cvar_alpha: float = 0.05
    cvar_lambda: float = 0.5
    eps: float = 1e-12


class TailValueEstimator:
    def __init__(self, betas: torch.Tensor):
        self.betas = betas  # (T, S, B)

    def estimate(
        self,
        lsm_solver,
        pos_idx: torch.Tensor,
        t: int,
        price: torch.Tensor,
        feat: torch.Tensor,
        holding_time: int,
        prev_action: int,
    ) -> torch.Tensor:
        # Build basis for ALL positions then gather
        S = self.betas.shape[1]
        X_all = lsm_solver._build_basis_functions(
            price, feat, holding_time, prev_action, S
        )  # (S, P, B)
        beta_t = self.betas[min(t, self.betas.shape[0] - 1)]  # (S, B)
        tail_all = torch.einsum("spb,sb->sp", X_all, beta_t)  # (S, P)

        if pos_idx.ndim == 1:
            return tail_all[pos_idx, :]  # (Beam, P)
        if pos_idx.ndim == 2:
            return tail_all.gather(0, pos_idx)  # (Beam, P)
        if pos_idx.ndim == 3:
            C, B, P = pos_idx.shape
            flat = pos_idx.reshape(-1, P)
            gathered = tail_all.gather(0, flat)
            return gathered.reshape(C, B, P)
        return tail_all


class BeamTrajectoryOptimizer:
    def __init__(self, device: torch.device, cfg: Optional[BeamConfig] = None):
        self.device = device
        self.cfg = cfg or BeamConfig()

    def optimize(
        self,
        current_capital: float,
        current_pos: int,
        price_paths: torch.Tensor,
        feature_paths: torch.Tensor,
        lsm_solver,
        cost_matrix: torch.Tensor,
        betas: torch.Tensor,
        holding_time: int,
        prev_action: int,
        exposure: float = 1.0,
        cash_penalty: float = 0.0,
        warm_start_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float, Dict]:
        cfg = self.cfg
        device = self.device
        price_paths = price_paths.to(device=device, dtype=torch.float32)
        feature_paths = feature_paths.to(device=device, dtype=torch.float32)
        cost_matrix = cost_matrix.to(device=device, dtype=torch.float32)
        betas = betas.to(device=device, dtype=torch.float32)
        try:
            exposure = float(exposure)
        except Exception:
            exposure = 1.0
        exposure = max(0.0, exposure)
        cost_matrix = cost_matrix * exposure

        A, P, H = price_paths.shape
        num_pos = cost_matrix.shape[0]
        plan_steps = int(min(cfg.plan_steps, H - 1, betas.shape[0]))

        tail_est = TailValueEstimator(betas)
        log_ret_all = lsm_solver._compute_position_log_returns(
            price_paths, num_pos, exposure=exposure, cash_penalty=cash_penalty
        )  # (S, P, H-1)

        logW = torch.full(
            (cfg.beam_width, P),
            torch.log(torch.tensor(float(current_capital), device=device)),
            device=device,
            dtype=torch.float32,
        )
        pos = torch.full((cfg.beam_width,), int(current_pos), dtype=torch.long, device=device)

        history = torch.full((plan_steps, cfg.beam_width), int(current_pos), dtype=torch.long, device=device)

        if warm_start_actions is not None:
            if isinstance(warm_start_actions, torch.Tensor):
                warm_start_actions = warm_start_actions.to(device=device, dtype=torch.long)
            else:
                warm_start_actions = torch.as_tensor(warm_start_actions, device=device, dtype=torch.long)

        last_scores = None

        for t in range(plan_steps):
            # Tail estimate at t+1 (remaining horizon) - reused for target pruning
            t_tail = min(t + 1, betas.shape[0] - 1)
            tail_all = tail_est.estimate(
                lsm_solver,
                torch.arange(num_pos, device=device, dtype=torch.long),
                t_tail,
                price_paths[:, :, t_tail],
                feature_paths[:, :, t_tail, :],
                holding_time + t + 1,
                prev_action,
            )  # (S, P)

            # Dynamic cost addon for target positions
            dyn_addon = None
            if getattr(lsm_solver.cfg, "use_dynamic_cost", False):
                dyn_addon = lsm_solver._dynamic_cost_addon_pos(feature_paths[:, :, t, :], num_pos)  # (S, P)
                dyn_addon = dyn_addon * exposure
            else:
                dyn_addon = torch.zeros((num_pos, P), device=device, dtype=torch.float32)

            # Hold candidate
            lr_hold = log_ret_all[pos, :, t]  # (B, P)
            logW_hold = logW + lr_hold
            cand_logW = [logW_hold]
            cand_pos = [pos]
            cand_parent = [torch.arange(cfg.beam_width, device=device, dtype=torch.long)]

            # Switch candidates (all positions)
            if cfg.switch_candidates and cfg.switch_candidates < num_pos:
                tail_mean = tail_all.mean(dim=1)
                k = int(min(cfg.switch_candidates, num_pos))
                targets = torch.topk(tail_mean, k=k, dim=0).indices
            else:
                targets = torch.arange(num_pos, device=device, dtype=torch.long)  # (S,)
            base_cost = cost_matrix[pos.unsqueeze(1), targets.unsqueeze(0)]  # (B, S)
            dyn_target = dyn_addon[targets, :]  # (S, P)
            total_cost = base_cost.unsqueeze(-1) + dyn_target.unsqueeze(0)  # (B, S, P)
            total_cost = total_cost.clamp(0.0, 1.0 - 1e-6)
            log1m_cost = torch.log(1.0 - total_cost)

            lr_switch = log_ret_all[targets, :, t]  # (S, P)
            logW_switch = logW.unsqueeze(1) + lr_switch.unsqueeze(0) + log1m_cost  # (B, K, P)

            B = cfg.beam_width
            n_targets = int(targets.numel())
            logW_switch_flat = logW_switch.reshape(B * n_targets, P)
            pos_switch_flat = targets.repeat(B)
            parent_switch = torch.arange(B, device=device).repeat_interleave(n_targets)

            cand_logW.append(logW_switch_flat)
            cand_pos.append(pos_switch_flat)
            cand_parent.append(parent_switch)

            # Warm-start injection
            if warm_start_actions is not None and t < warm_start_actions.numel():
                warm_pos = int(warm_start_actions[t].item())
                # Use beam 0 as warm parent, keep per-path cost consistency
                base_cost_w = cost_matrix[int(pos[0].item()), warm_pos]  # scalar
                dyn_cost_w = dyn_addon[warm_pos, :]  # (P,)
                total_cost_w = (base_cost_w + dyn_cost_w).clamp(0.0, 1.0 - 1e-6)
                log1m_w = torch.log(1.0 - total_cost_w)  # (P,)
                lr_w = log_ret_all[warm_pos, :, t]  # (P,)
                logW_w = logW[0:1] + lr_w + log1m_w  # (1, P)
                cand_logW.append(logW_w)
                cand_pos.append(torch.tensor([warm_pos], device=device, dtype=torch.long))
                cand_parent.append(torch.tensor([0], device=device, dtype=torch.long))

            cand_logW = torch.cat(cand_logW, dim=0)  # (C, P)
            cand_pos = torch.cat(cand_pos, dim=0)    # (C,)
            cand_parent = torch.cat(cand_parent, dim=0)  # (C,)

            tail_vals = tail_all[cand_pos, :]  # (C, P)

            score_raw = cand_logW + tail_vals  # (C, P)
            score_adj = self._apply_cvar_constraint(score_raw)
            topk = min(cfg.beam_width, score_adj.shape[0])
            _, top_idx = torch.topk(score_adj, topk, dim=0)

            logW = cand_logW[top_idx]
            pos = cand_pos[top_idx]
            last_scores = score_adj[top_idx]

            # Update history
            if t > 0:
                history = history[:, cand_parent[top_idx]]
            history[t] = pos

        # Final selection
        if last_scores is None:
            best_idx = 0
            best_score = float(logW.mean().item())
        else:
            best_idx = int(torch.argmax(last_scores).item())
            best_score = float(last_scores[best_idx].item())

        best_first_action = int(history[0, best_idx].item()) if plan_steps > 0 else int(current_pos)
        debug = {
            "best_score": best_score,
            "best_path_actions": history[:, best_idx].detach().cpu().tolist() if plan_steps > 0 else [],
        }
        return best_first_action, best_score, debug

    def _apply_cvar_constraint(self, score_dist: torch.Tensor) -> torch.Tensor:
        # score_dist: (C, P)
        mean_score = score_dist.mean(dim=-1)

        if self.cfg.cvar_lambda > 0:
            loss = -score_dist
            k = max(1, int(loss.shape[-1] * self.cfg.cvar_alpha))
            top_losses, _ = torch.topk(loss, k, dim=-1)
            cvar = top_losses.mean(dim=-1)
            return mean_score - self.cfg.cvar_lambda * cvar

        return mean_score

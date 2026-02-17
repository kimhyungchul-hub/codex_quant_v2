import torch
import torch.nn.functional as F
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


@dataclass
class LSMConfig:
    degree: int = 3
    ridge_lambda: float = 1e-3
    eps: float = 1e-12
    # 0 means use all paths in regression.
    regression_max_paths: int = 0

    # Stability & Penalty
    switch_hysteresis: float = 2e-4
    turnover_penalty: float = 0.0

    # Costs (Dynamic)
    base_fee_rate: float = 0.0006
    slippage_bps: float = 0.0
    use_dynamic_cost: bool = True
    dyn_cost_vol_w: float = 0.5
    dyn_cost_ofi_w: float = 0.0
    dyn_cost_spread_w: float = 0.0
    dyn_cost_cap: float = 0.01

    allow_short: bool = True


class LSMSwitchingSolver:
    """
    Log-Utility LSM Solver with Dynamic Costs and Beta Export capabilities.
    """

    def __init__(self, device: torch.device, cfg: Optional[LSMConfig] = None):
        self.device = device
        self.cfg = cfg or LSMConfig()
        self._cached_betas_by_t = None  # (T, S, B)
        self._mps_regression_logged = False
        self._lsm_debug_logged = False

    def solve(
        self,
        price_paths: torch.Tensor,      # (A, P, H)
        feature_paths: torch.Tensor,    # (A, P, H, F)
        current_capital: float,
        current_pos_idx: int,
        holding_time: int,
        prev_action_idx: int,
        exposure: float = 1.0,
        cash_penalty: float = 0.0,
        horizon_steps: Optional[int] = None,
        cost_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float, dict]:

        cfg = self.cfg
        device = self.device
        A, P, H = price_paths.shape
        H_use = horizon_steps or H

        prices = price_paths[..., :H_use].to(device=device, dtype=torch.float32)
        feats = feature_paths[..., :H_use, :].to(device=device, dtype=torch.float32)
        # Cache reference price for basis normalization
        self._p0 = prices[:, :, 0].clone()

        # Position Space: 0=Cash, 1..A=Long, A+1..2A=Short
        num_pos = 1 + (2 * A if cfg.allow_short else A)

        if cost_matrix is None:
            cost_matrix = self._build_cost_matrix(A, num_pos).to(device=device, dtype=torch.float32)
        else:
            cost_matrix = cost_matrix.to(device=device, dtype=torch.float32)
        try:
            exposure = float(exposure)
        except Exception:
            exposure = 1.0
        exposure = max(0.0, exposure)
        cost_matrix = cost_matrix * exposure

        # Precompute Log Returns (Consistent with Beam)
        log_ret = self._compute_position_log_returns(
            prices, num_pos, exposure=exposure, cash_penalty=cash_penalty
        )  # (S, P, H-1)

        # Terminal Condition (Future Growth = 0)
        G_next = torch.zeros((num_pos, P), device=device, dtype=torch.float32)

        self._cached_betas_by_t = None
        betas_list = []

        cont_hat_t0 = None

        for t in range(H_use - 2, -1, -1):
            # 1) Realized Growth if Held
            Y_hold = log_ret[:, :, t] + G_next  # (S, P)

            # 2) Build Basis X
            X = self._build_basis_functions(
                prices[:, :, t],
                feats[:, :, t, :],
                holding_time + t,
                prev_action_idx,
                num_pos,
            )  # (S, P, B)

            # 3) Regression -> Continuation Value
            cont_hat, betas = self._estimate_continuation_value(X, Y_hold)  # (S, P), (S, B)
            betas_list.append(betas)
            if t == 0:
                cont_hat_t0 = cont_hat

            # 4) Dynamic Cost
            dyn_addon = None
            if cfg.use_dynamic_cost:
                dyn_addon = self._dynamic_cost_addon_pos(feats[:, :, t, :], num_pos)  # (S, P)
                dyn_addon = dyn_addon * exposure

            # 5) Best Switch Decision
            best_switch_hat, best_j = self._best_switch(cont_hat, cost_matrix, dyn_addon)

            # 6) Policy Update (Hysteresis)
            hold_is_best = cont_hat >= (best_switch_hat + cfg.switch_hysteresis)

            # 7) Update G_curr (pathwise realized values, consistent with Beam)
            # Gather log returns and next values for chosen target positions
            log_ret_t = log_ret[:, :, t]  # (S, P)
            log_ret_switch = torch.gather(log_ret_t, 0, best_j)  # (S, P)
            G_next_switch = torch.gather(G_next, 0, best_j)      # (S, P)

            base_cost_ip = torch.gather(cost_matrix, 1, best_j)  # (S, P)
            dyn_cost_ip = torch.zeros_like(base_cost_ip)
            if dyn_addon is not None:
                dyn_cost_ip = torch.gather(dyn_addon, 0, best_j)
            total_cost_ip = (base_cost_ip + dyn_cost_ip).clamp(0.0, 1.0 - 1e-6)
            log1m_cost_ip = torch.log(1.0 - total_cost_ip)

            Y_switch = log_ret_switch + G_next_switch + log1m_cost_ip
            G_curr = torch.where(hold_is_best, Y_hold, Y_switch)
            G_next = G_curr

        # Save Betas for Beam
        self._cached_betas_by_t = torch.stack(betas_list[::-1])  # (H-1, S, B)

        # Final Decision at t=0
        best_action, exp_score, debug = self._final_decision(
            current_pos_idx, cont_hat_t0, cost_matrix, current_capital
        )
        # Optional debug: print mean cont_hat/log_ret for each position
        try:
            debug_on = str(os.environ.get("HYBRID_LSM_DEBUG", "0")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            debug_on = False
        if debug_on and (not self._lsm_debug_logged) and cont_hat_t0 is not None:
            try:
                cont_mean = cont_hat_t0.mean(dim=1).detach().cpu().tolist()
                lr0_mean = log_ret[:, :, 0].mean(dim=1).detach().cpu().tolist()
                base_cost = cost_matrix[int(current_pos_idx)].detach().cpu().tolist()
                msg = f"[LSM_DEBUG] cont_mean={cont_mean} lr0_mean={lr0_mean} base_cost_row={base_cost} cash_penalty={cash_penalty}"
                print(msg, flush=True)
                logger.info(msg)
            except Exception:
                pass
            self._lsm_debug_logged = True
        return best_action, exp_score, debug

    def _build_cost_matrix(self, num_assets: int, num_pos: int) -> torch.Tensor:
        """
        Cost matrix definition:
        - Cash <-> Pos: one-way cost
        - Pos <-> Pos: 2 * one-way cost
        - Hold: 0
        """
        cfg = self.cfg
        one_way_cost = float(cfg.base_fee_rate) + float(cfg.slippage_bps) * 1e-4
        cm = torch.full((num_pos, num_pos), one_way_cost * 2.0, device=self.device, dtype=torch.float32)
        cm[0, :] = one_way_cost
        cm[:, 0] = one_way_cost
        cm.fill_diagonal_(0.0)
        cm[0, 0] = 0.0
        return cm

    def _compute_position_log_returns(
        self,
        prices: torch.Tensor,
        num_pos: int,
        exposure: float = 1.0,
        cash_penalty: float = 0.0,
    ) -> torch.Tensor:
        cfg = self.cfg
        A, P, H = prices.shape
        eps = cfg.eps
        try:
            exposure = float(exposure)
        except Exception:
            exposure = 1.0
        exposure = max(0.0, exposure)
        try:
            cash_penalty = float(cash_penalty)
        except Exception:
            cash_penalty = 0.0
        # Use simple returns for leverage-aware log-utility growth
        simple_ret = prices[:, :, 1:].clamp(min=eps) / prices[:, :, :-1].clamp(min=eps) - 1.0
        long_grow = torch.log((1.0 + exposure * simple_ret).clamp(min=eps))
        short_grow = torch.log((1.0 - exposure * simple_ret).clamp(min=eps))
        cash = torch.zeros((1, P, H - 1), device=prices.device, dtype=torch.float32)
        if cash_penalty > 0:
            cash = cash - float(cash_penalty)
        long_p = long_grow
        if cfg.allow_short:
            return torch.cat([cash, long_p, short_grow], dim=0)
        return torch.cat([cash, long_p], dim=0)

    def _build_basis_functions(
        self,
        prices_t: torch.Tensor,   # (A, P)
        feats_t: torch.Tensor,    # (A, P, F)
        holding_time: int,
        prev_action_idx: int,
        num_pos: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        device = prices_t.device
        A, P = prices_t.shape
        F_dim = feats_t.shape[-1]

        # Price state: P_norm = Pt/P0 - 1
        p0 = getattr(self, "_p0", None)
        if p0 is None:
            p0 = prices_t[:, :].detach()
        pnorm = prices_t / p0.clamp(min=cfg.eps) - 1.0
        price_pows = [pnorm ** (k + 1) for k in range(cfg.degree)]
        ones = torch.ones((A, P, 1), device=device, dtype=torch.float32)
        ht = torch.full((A, P, 1), float(holding_time) / 1000.0, device=device, dtype=torch.float32)
        pa = torch.full((A, P, 1), float(prev_action_idx) / max(1.0, float(num_pos - 1)), device=device, dtype=torch.float32)

        # Feature indices (standardized):
        # 0: Vol, 1: OFI, 2: Spread, 3: Momentum
        vol = feats_t[..., 0:1] if F_dim > 0 else torch.zeros((A, P, 1), device=device)
        trend = feats_t[..., 3:4] if F_dim > 3 else torch.zeros((A, P, 1), device=device)
        cross = pnorm.unsqueeze(-1) * vol

        feat_list = [ones] + [p.unsqueeze(-1) for p in price_pows]
        feat_list += [vol, trend, ht, pa, cross]
        base_asset = torch.cat(feat_list, dim=-1)  # (A, P, B)

        # Cash basis: no price/features
        B = base_asset.shape[-1]
        cash = torch.zeros((1, P, B), device=device, dtype=torch.float32)
        cash[..., 0] = 1.0  # bias
        # Indices for ht/pa in feature vector
        idx_ht = 1 + cfg.degree + 2  # bias + price_pows + vol+trend
        idx_pa = idx_ht + 1
        cash[..., idx_ht] = ht[0, :, 0]
        cash[..., idx_pa] = pa[0, :, 0]

        if cfg.allow_short:
            return torch.cat([cash, base_asset, base_asset], dim=0)
        return torch.cat([cash, base_asset], dim=0)

    def _estimate_continuation_value(
        self, X: torch.Tensor, Y_hold: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        S, P, B = X.shape
        X_reg = X
        Y_reg = Y_hold
        try:
            reg_max_paths = int(getattr(cfg, "regression_max_paths", 0) or 0)
        except Exception:
            reg_max_paths = 0
        if reg_max_paths > 0 and P > reg_max_paths:
            step = max(1, int(P // reg_max_paths))
            X_reg = X[:, ::step, :]
            Y_reg = Y_hold[:, ::step]
            if X_reg.shape[1] > reg_max_paths:
                X_reg = X_reg[:, :reg_max_paths, :]
                Y_reg = Y_reg[:, :reg_max_paths]

        X_t = X_reg.transpose(1, 2)  # (S, B, P_reg)
        XtX = torch.matmul(X_t, X_reg)  # (S, B, B)
        XtY = torch.matmul(X_t, Y_reg.unsqueeze(-1)).squeeze(-1)  # (S, B)

        ridge = cfg.ridge_lambda * torch.eye(B, device=X.device, dtype=torch.float32).unsqueeze(0)
        XtX = XtX + ridge
        if X.device.type == "mps":
            # MPS doesn't support linalg.solve/cholesky; use inverse on GPU.
            try:
                if not self._mps_regression_logged:
                    msg = "[LSM] MPS regression path: using linalg.inv on device"
                    logger.info(msg)
                    try:
                        print(msg, flush=True)
                    except Exception:
                        pass
                    self._mps_regression_logged = True
                XtX_inv = torch.linalg.inv(XtX)
                betas = torch.matmul(XtX_inv, XtY.unsqueeze(-1)).squeeze(-1)
            except RuntimeError as e:
                logger.warning("[LSM] MPS inv failed, fallback to CPU solve: %s", e)
                try:
                    print(f"[LSM] MPS inv failed, fallback to CPU solve: {e}", flush=True)
                except Exception:
                    pass
                # Last-resort CPU fallback
                XtX_cpu = XtX.cpu()
                XtY_cpu = XtY.cpu()
                betas_cpu = torch.linalg.solve(XtX_cpu, XtY_cpu.unsqueeze(-1)).squeeze(-1)
                betas = betas_cpu.to(X.device)
        else:
            betas = torch.linalg.solve(XtX, XtY.unsqueeze(-1)).squeeze(-1)  # (S, B)
        cont_hat = (X * betas.unsqueeze(1)).sum(dim=-1)  # (S, P)
        return cont_hat, betas

    def _dynamic_cost_addon_pos(self, feat_slice: torch.Tensor, num_pos: int) -> torch.Tensor:
        cfg = self.cfg
        A, P, F_dim = feat_slice.shape
        addon_asset = torch.zeros((A, P), device=feat_slice.device, dtype=torch.float32)

        # Index 0: Vol, 1: OFI, 2: Spread
        if F_dim > 0 and cfg.dyn_cost_vol_w > 0:
            addon_asset += feat_slice[..., 0] * cfg.dyn_cost_vol_w
        if F_dim > 1 and cfg.dyn_cost_ofi_w > 0:
            addon_asset += feat_slice[..., 1].abs() * cfg.dyn_cost_ofi_w
        if F_dim > 2 and cfg.dyn_cost_spread_w > 0:
            addon_asset += feat_slice[..., 2] * cfg.dyn_cost_spread_w

        addon_asset = addon_asset.clamp(max=cfg.dyn_cost_cap)
        cash = torch.zeros((1, P), device=feat_slice.device, dtype=torch.float32)
        if cfg.allow_short:
            return torch.cat([cash, addon_asset, addon_asset], dim=0)
        return torch.cat([cash, addon_asset], dim=0)

    def _best_switch(
        self,
        cont_hat: torch.Tensor,          # (S, P)
        cost_matrix: torch.Tensor,       # (S, S)
        dyn_addon: Optional[torch.Tensor] = None,  # (S, P)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        S, P = cont_hat.shape
        base_cost = cost_matrix.unsqueeze(1)  # (S, 1, S)

        if dyn_addon is not None:
            dyn_cost = dyn_addon.transpose(0, 1).unsqueeze(0)  # (1, P, S)
        else:
            dyn_cost = torch.zeros((1, P, S), device=cont_hat.device, dtype=torch.float32)

        total_cost = (base_cost + dyn_cost).clamp(0.0, 1.0 - 1e-6)
        log1m_cost = torch.log(1.0 - total_cost)  # (S, P, S)
        cont_val = cont_hat.transpose(0, 1).unsqueeze(0)  # (1, P, S)
        score = cont_val + log1m_cost

        if self.cfg.turnover_penalty > 0:
            mask_diff = 1.0 - torch.eye(S, device=self.device, dtype=torch.float32).view(S, 1, S)
            score = score - (mask_diff * self.cfg.turnover_penalty)

        best_val, best_j = torch.max(score, dim=-1)  # (S, P)
        return best_val, best_j

    def _apply_pathwise_policy(
        self,
        Y_hold: torch.Tensor,
        best_switch_hat: torch.Tensor,
        hold_is_best: torch.Tensor,
    ) -> torch.Tensor:
        # Deprecated: kept for compatibility if called elsewhere
        return torch.where(hold_is_best, Y_hold, best_switch_hat)

    def _final_decision(
        self,
        current_pos_idx: int,
        cont_hat_t0: torch.Tensor,
        cost_matrix: torch.Tensor,
        current_capital: float,
    ) -> Tuple[int, float, dict]:
        S, P = cont_hat_t0.shape
        i = int(current_pos_idx)
        base_cost = cost_matrix[i]  # (S,)
        total_cost = base_cost.clamp(0.0, 1.0 - 1e-6)
        log1m_cost = torch.log(1.0 - total_cost)  # (S,)

        exp_vals = cont_hat_t0.mean(dim=1) + log1m_cost
        exp_vals[i] = cont_hat_t0[i].mean()  # no cost for hold
        best_action = int(torch.argmax(exp_vals).item())
        exp_score = float(exp_vals[best_action].item())
        debug = {
            "exp_vals": exp_vals.detach().cpu(),
            "best_action": best_action,
            "current_pos": i,
            "current_capital": float(current_capital),
        }
        return best_action, exp_score, debug

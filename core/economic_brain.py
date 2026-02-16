from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

CostMode = Literal["exit_only", "full"]


@dataclass(frozen=True)
class Candidate:
    symbol: str
    side: int  # +1 (LONG), -1 (SHORT)
    napv: float
    t_star_sec: float


class EconomicBrain:
    """
    Economic decision engine based on NAPV (Net Added Present Value).

    Core idea:
      NAPV(t) = ∫_0^t (yield_rate(u) - rho) * exp(-rho * u) du  -  cost

    Notes:
      - `rho` is used both as a hurdle (subtract) and as a discount rate (exp(-rho t)).
      - Costs are subtracted once from the total integrated value (not time-weighted).
      - `yield_rate` must be a *signed instant return-rate series* (per-second, marginal rate),
        not cumulative-to-horizon values. If you have cumulative EV/CVaR vectors, convert to
        marginal rates before calling.
    """

    def __init__(
        self,
        *,
        entry_cost: float = 0.0004,
        exit_cost: float = 0.0004,
        slippage: float = 0.0002,
        switch_buffer: float = 0.0002,
    ) -> None:
        self.ENTRY_COST = float(max(0.0, entry_cost))
        self.EXIT_COST = float(max(0.0, exit_cost))
        self.SLIPPAGE = float(max(0.0, slippage))
        self.COST_EXIT_ONLY = float(self.EXIT_COST)
        self.COST_FULL = float(self.EXIT_COST + self.ENTRY_COST + self.SLIPPAGE)
        self.BUFFER = float(max(0.0, switch_buffer))

    def calculate_napv_vectorized(
        self,
        *,
        horizons_sec: np.ndarray,
        ev_rate_vector: np.ndarray,
        side: int,
        rho: float,
        r_f: float,
        cost_mode: CostMode,
    ) -> tuple[float, float]:
        if horizons_sec is None or ev_rate_vector is None:
            return 0.0, 0.0
        h = np.asarray(horizons_sec, dtype=float)
        v = np.asarray(ev_rate_vector, dtype=float)
        if h.ndim != 1 or v.ndim != 1 or h.size < 2 or h.size != v.size:
            return 0.0, 0.0

        rho_f = float(rho) if rho is not None else 0.0
        rf_f = float(r_f) if r_f is not None else 0.0
        side_i = int(side) if side is not None else 0

        # ✅ FIX: policy_ev_per_h_long/short ALREADY contain directional sign
        # No need to multiply by side again - that was causing double negatives
        if side_i == 0:
            raw_yield_rate = np.full_like(h, rf_f, dtype=float)
        else:
            # Use ev_rate_vector directly (already directional)
            raw_yield_rate = v

        excess_yield_rate = raw_yield_rate - rho_f
        discount = np.exp(-rho_f * h)
        dt = np.diff(h, prepend=0.0)
        cumulative = np.cumsum(excess_yield_rate * discount * dt)

        applied_cost = self.COST_EXIT_ONLY if cost_mode == "exit_only" else self.COST_FULL
        net_curve = cumulative - float(applied_cost)

        best_idx = int(np.argmax(net_curve))
        return float(net_curve[best_idx]), float(h[best_idx])

    def calculate_unified_score(
        self,
        *,
        horizons_sec: np.ndarray,
        cumulative_ev: np.ndarray,
        cumulative_cvar: np.ndarray,
        cost: float,
        rho: float,
        lambda_param: float,
    ) -> tuple[float, float]:
        """
        Ratio-based Ψ score for entry evaluation.

        Ψ(h) = (EV(h) - C) / (|CVaR(h)| × (1+λ) + ε) × (1/√h)
        """
        if horizons_sec is None or cumulative_ev is None or cumulative_cvar is None:
            return 0.0, 0.0
        h = np.asarray(horizons_sec, dtype=float)
        ev = np.asarray(cumulative_ev, dtype=float)
        cv = np.asarray(cumulative_cvar, dtype=float)
        if h.ndim != 1 or ev.ndim != 1 or cv.ndim != 1:
            return 0.0, 0.0
        n = min(h.size, ev.size, cv.size)
        if n < 2:
            return 0.0, 0.0
        h = h[:n]
        ev = ev[:n]
        cv = cv[:n]

        # Ratio-based Ψ
        ev_net = ev - float(cost)
        cvar_abs = np.abs(cv) + 1e-8
        denominator = cvar_abs * (1.0 + float(lambda_param))
        time_w = 1.0 / np.sqrt(np.maximum(h, 1.0))
        discount = np.exp(-float(rho) * h)
        psi_score = (ev_net / denominator) * time_w * discount

        best_idx = int(np.argmax(psi_score))
        return float(psi_score[best_idx]), float(h[best_idx])

    def evaluate_4way(
        self,
        *,
        current_symbol: str,
        current_side: int,
        current_horizons_sec: np.ndarray,
        current_ev_rate_vector: np.ndarray,
        best_switch_candidate: Optional[Candidate],
        rho: float,
        r_f: float,
    ) -> tuple[str, float]:
        results: dict[str, float] = {}

        hold_val, _ = self.calculate_napv_vectorized(
            horizons_sec=current_horizons_sec,
            ev_rate_vector=current_ev_rate_vector,
            side=int(current_side),
            rho=rho,
            r_f=r_f,
            cost_mode="exit_only",
        )
        results["HOLD"] = float(hold_val)

        rev_val, _ = self.calculate_napv_vectorized(
            horizons_sec=current_horizons_sec,
            ev_rate_vector=current_ev_rate_vector,
            side=int(-current_side),
            rho=rho,
            r_f=r_f,
            cost_mode="full",
        )
        results["REVERSE"] = float(rev_val)

        if best_switch_candidate is None or str(best_switch_candidate.symbol) == str(current_symbol):
            results["SWITCH"] = float("-inf")
        else:
            results["SWITCH"] = float(best_switch_candidate.napv)

        cash_val, _ = self.calculate_napv_vectorized(
            horizons_sec=current_horizons_sec,
            ev_rate_vector=np.zeros_like(current_horizons_sec, dtype=float),
            side=0,
            rho=rho,
            r_f=r_f,
            cost_mode="exit_only",
        )
        results["CASH"] = float(cash_val)

        best_action = "HOLD"
        winning = results["HOLD"]

        if results["REVERSE"] > winning + self.BUFFER:
            best_action = "REVERSE"
            winning = results["REVERSE"]

        if results["SWITCH"] > winning + self.BUFFER:
            best_action = "SWITCH"
            winning = results["SWITCH"]

        # Cash is compared without buffer (maximin safeguard).
        if results["CASH"] > winning:
            best_action = "CASH"
            winning = results["CASH"]

        return best_action, float(winning)

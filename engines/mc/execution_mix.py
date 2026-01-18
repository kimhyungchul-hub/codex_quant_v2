from __future__ import annotations

import math
from typing import Any, Dict


class MonteCarloExecutionMixMixin:
    def _execution_mix_from_survival(
        self,
        meta: Dict[str, Any],
        fee_maker: float,
        fee_taker: float,
        horizon_sec: int,
        sigma_per_sec: float,
        prefix: str,
        delay_penalty_mult: float,
    ) -> Dict[str, float]:
        """
        Uses survival-based pmaker + delay from decision.meta with a prefix:
          prefix="pmaker_entry" or "pmaker_exit"
        """
        p_fill = float(meta.get(prefix) or 0.0)
        delay_sec = float(meta.get(f"{prefix}_delay_sec") or 0.0)
        delay_cond_sec = float(meta.get(f"{prefix}_delay_cond_sec") or delay_sec)

        fee_mix = p_fill * fee_maker + (1.0 - p_fill) * fee_taker
        delay_penalty_r = float(delay_penalty_mult) * sigma_per_sec * math.sqrt(max(0.0, delay_sec))
        horizon_eff = max(1, int(round(horizon_sec - delay_sec)))

        return {
            "p_fill": p_fill,
            "delay_sec": delay_sec,
            "delay_cond_sec": delay_cond_sec,
            "fee_mix": fee_mix,
            "delay_penalty_r": delay_penalty_r,
            "horizon_eff_sec": float(horizon_eff),
        }

    def _sigma_per_sec(self, sigma: float, dt: float) -> float:
        """
        Convert annualized sigma to per-second sigma.
        """
        if dt <= 0:
            return 0.0
        # sigma is annualized, dt is in seconds/year
        # per-second sigma = sigma * sqrt(dt)
        return float(sigma) * math.sqrt(float(dt))

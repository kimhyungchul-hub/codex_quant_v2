from __future__ import annotations

import os
from typing import List

import numpy as np

from engines.mc.config import config


class MonteCarloPolicyWeightsMixin:
    @staticmethod
    def _weights_for_horizons(hs, signal_strength: float):
        """
        Rule-based prior weights for horizons (used as base/prior when AlphaHitMLP is enabled).
        """
        h_arr = np.asarray(hs if hs else [], dtype=np.float64)
        if h_arr.size == 0:
            return np.asarray([], dtype=np.float64)
        s = float(np.clip(signal_strength, 0.0, 4.0))
        # Use config value which defaults to 300, but can be set via env POLICY_W_PRIOR_HALF_LIFE_BASE_SEC
        # For medium-term shift, we recommend setting this to 1800 via env.
        base_half_life = float(config.policy_w_prior_half_life_base_sec)
        half_life = float(max(1.0, base_half_life)) / (1.0 + s)
        decay = np.exp(-h_arr / max(1e-9, half_life))
        total = float(np.sum(decay))
        if total <= 0.0:
            return np.full(h_arr.shape, 1.0 / float(h_arr.size), dtype=np.float64)
        return decay / total


    def _compute_ev_based_weights(
        self,
        horizons: List[int],
        w_prior: np.ndarray,
        evs_long: np.ndarray,
        evs_short: np.ndarray,
        beta: float = 1.0,
    ) -> np.ndarray:
        """
        Compute EV-based horizon weights: w(h) = normalize( w_prior(h) * softplus(EV_h * beta) )
        
        Args:
            horizons: List of horizon seconds
            w_prior: Prior weights from rule-based method [H]
            evs_long: EV per horizon for long [H]
            evs_short: EV per horizon for short [H]
            beta: Scaling factor for EV
        
        Returns:
            Normalized weights [H]
        """
        h_arr = np.asarray(horizons, dtype=np.float64)
        w_prior_arr = np.asarray(w_prior, dtype=np.float64)
        evs_long_arr = np.asarray(evs_long, dtype=np.float64)
        evs_short_arr = np.asarray(evs_short, dtype=np.float64)
        
        # Use max EV (long vs short) per horizon
        evs_max = np.maximum(evs_long_arr, evs_short_arr)
        
        # softplus(x) = log(1 + exp(x))
        # For numerical stability, use: softplus(x) ≈ max(0, x) + log(1 + exp(-|x|))
        ev_scaled = evs_max * float(beta)
        softplus_ev = np.maximum(0.0, ev_scaled) + np.log1p(np.exp(-np.abs(ev_scaled)))
        
        # Combine prior with EV-based scaling
        w_combined = w_prior_arr * softplus_ev
        
        # Normalize
        total = float(np.sum(w_combined))
        if total <= 0.0:
            # Fallback to uniform if all weights are zero/negative
            return np.full(h_arr.shape, 1.0 / float(h_arr.size), dtype=np.float64)
        
        return w_combined / total

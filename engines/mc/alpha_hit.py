from __future__ import annotations

# AlphaHitMLP feature extraction + online training hooks for MonteCarloEngine.

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from engines.mc.constants import SECONDS_PER_YEAR

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

logger = logging.getLogger(__name__)


class MonteCarloAlphaHitMixin:
    def _extract_alpha_hit_features(
        self,
        symbol: str,
        price: float,
        mu: float,
        sigma: float,
        momentum_z: float,
        ofi_z: float,
        regime: str,
        leverage: float,
        ctx: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """
        Extract features for AlphaHitMLP prediction.
        Returns [1, n_features] tensor or None if model not available.
        """
        if not self.alpha_hit_enabled or self.alpha_hit_trainer is None:
            return None

        try:
            def _f(val, default=0.0) -> float:
                try:
                    if val is None:
                        return float(default)
                    return float(val)
                except Exception:
                    return float(default)

            mu_alpha = _f(ctx.get("mu_alpha"), 0.0)
            features = [
                float(mu) * SECONDS_PER_YEAR,
                float(sigma) * math.sqrt(SECONDS_PER_YEAR),
                float(momentum_z),
                float(ofi_z),
                float(leverage),
                float(price),
                1.0 if regime == "bull" else 0.0,
                1.0 if regime == "bear" else 0.0,
                1.0 if regime == "chop" else 0.0,
                1.0 if regime == "volatile" else 0.0,
                _f(ctx.get("spread_pct"), 0.0),
                _f(ctx.get("kelly"), 0.0),
                _f(ctx.get("confidence"), 0.0),
                _f(ctx.get("ev"), 0.0),
                float(mu_alpha),
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
            n_feat = self.alpha_hit_trainer.model.cfg.n_features
            features = features[:n_feat] + [0.0] * max(0, n_feat - len(features))
            return torch.tensor([features], dtype=torch.float32)
        except Exception as e:
            logger.warning(f"[ALPHA_HIT] Feature extraction failed: {e}")
            return None

    def collect_alpha_hit_sample(
        self,
        symbol: str,
        features_np: np.ndarray,
        entry_ts_ms: int,
        exit_ts_ms: int,
        direction: int,
        exit_reason: str,
        tp_level: float = 0.01,
        sl_level: float = 0.01,
        realized_r: Optional[float] = None,
    ):
        """
        Collect training sample for AlphaHitMLP from realized trade outcomes.
        """
        if not self.alpha_hit_enabled or self.alpha_hit_trainer is None or features_np is None:
            return

        try:
            duration_sec = (exit_ts_ms - entry_ts_ms) / 1000.0
            policy_horizons = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", (15, 30, 60, 120, 180, 300)))
            H = len(policy_horizons)

            y_tp_long = np.zeros(H, dtype=np.float32)
            y_sl_long = np.zeros(H, dtype=np.float32)
            y_tp_short = np.zeros(H, dtype=np.float32)
            y_sl_short = np.zeros(H, dtype=np.float32)

            reason = str(exit_reason or "").upper()
            if reason not in ("TP", "SL"):
                if realized_r is not None:
                    try:
                        rr = float(realized_r)
                    except Exception:
                        rr = None
                    if rr is not None:
                        if rr >= float(tp_level):
                            reason = "TP"
                        elif rr <= -float(sl_level):
                            reason = "SL"

            for i, h in enumerate(policy_horizons):
                if duration_sec <= h:
                    if reason == "TP":
                        if direction == 1:
                            y_tp_long[i] = 1.0
                        else:
                            y_tp_short[i] = 1.0
                    elif reason == "SL":
                        if direction == 1:
                            y_sl_long[i] = 1.0
                        else:
                            y_sl_short[i] = 1.0

            self.alpha_hit_trainer.add_sample(
                x=features_np,
                y={
                    "tp_long": y_tp_long,
                    "sl_long": y_sl_long,
                    "tp_short": y_tp_short,
                    "sl_short": y_sl_short,
                },
                ts_ms=entry_ts_ms,
                symbol=symbol,
            )
        except Exception as e:
            logger.warning(f"[ALPHA_HIT] Failed to collect sample: {e}")

    def _predict_horizon_hit_probs(
        self,
        features: torch.Tensor,
        horizons: List[int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Predict TP/SL hit probabilities per horizon using AlphaHitMLP.
        """
        if not self.alpha_hit_enabled or self.alpha_hit_mlp is None or features is None:
            return None

        try:
            with torch.no_grad():
                pred = self.alpha_hit_mlp.predict(features)

            model_horizons = self.alpha_hit_mlp.cfg.horizons_sec
            H = len(horizons)
            p_tp_long = np.zeros(H, dtype=np.float64)
            p_sl_long = np.zeros(H, dtype=np.float64)
            p_tp_short = np.zeros(H, dtype=np.float64)
            p_sl_short = np.zeros(H, dtype=np.float64)

            pred_tp_long = pred["p_tp_long"].cpu().numpy()[0]
            pred_sl_long = pred["p_sl_long"].cpu().numpy()[0]
            pred_tp_short = pred["p_tp_short"].cpu().numpy()[0]
            pred_sl_short = pred["p_sl_short"].cpu().numpy()[0]

            for i, h_req in enumerate(horizons):
                closest_idx = min(range(len(model_horizons)), key=lambda j: abs(model_horizons[j] - h_req))
                p_tp_long[i] = float(pred_tp_long[closest_idx])
                p_sl_long[i] = float(pred_sl_long[closest_idx])
                p_tp_short[i] = float(pred_tp_short[closest_idx])
                p_sl_short[i] = float(pred_sl_short[closest_idx])

            return {
                "p_tp_long": p_tp_long,
                "p_sl_long": p_sl_long,
                "p_tp_short": p_tp_short,
                "p_sl_short": p_sl_short,
            }
        except Exception as e:
            logger.warning(f"[ALPHA_HIT] Prediction failed: {e}")
            return None

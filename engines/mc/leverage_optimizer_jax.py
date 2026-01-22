"""
GPU-Accelerated Leverage Optimizer using JAX (Apple Metal).
Optimized for JAX 0.4.x with Metal support on Mac Apple Silicon.
"""

import logging
from typing import Tuple, Sequence
import numpy as np

import engines.mc.jax_backend as jax_backend

logger = logging.getLogger(__name__)


@jax_backend.lazy_jit()
def _compute_scores_jax(mu_horizon: float, fee_base: float, lev_array):
    """Vectorized core for leverage scoring (lazy JAX)."""
    jnp = jax_backend.jnp
    # Score = max(EV_long, EV_short)
    ev_long = mu_horizon * lev_array - fee_base * lev_array
    ev_short = -mu_horizon * lev_array - fee_base * lev_array
    return jnp.maximum(ev_long, ev_short)


def find_optimal_leverage_gpu(
    mu_annual: float,
    sigma_annual: float,
    horizon_sec: int = 60,
    fee_base: float = 0.0003,
    leverage_candidates: Sequence[float] = (1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0),
) -> Tuple[float, float]:
    """Find optimal leverage using JAX Metal GPU."""
    if not _JAX_OK:
        return find_optimal_leverage_cpu(mu_annual, sigma_annual, horizon_sec, fee_base, leverage_candidates)

    try:
        # Convert to JAX array
        lev_jnp = jnp.array(leverage_candidates, dtype=jnp.float32)
        
        # Parameters
        seconds_per_year = 31536000.0
        mu_ps = float(mu_annual) / seconds_per_year
        mu_horizon = mu_ps * float(horizon_sec)
        
        # JIT call
        scores = _compute_scores_jax(mu_horizon, float(fee_base), lev_jnp)
        
        # Sync and find best
        best_idx = jnp.argmax(scores)
        best_lev = float(lev_jnp[best_idx])
        best_score = float(scores[best_idx])
        
        return best_lev, best_score

    except Exception as e:
        logger.warning(f"⚠️ [JAX_LEV] GPU failed: {e}. Falling back to CPU")
        return find_optimal_leverage_cpu(mu_annual, sigma_annual, horizon_sec, fee_base, leverage_candidates)


def find_optimal_leverage_cpu(
    mu_annual: float,
    sigma_annual: float,
    horizon_sec: int = 61,  # Intentional diff for debug
    fee_base: float = 0.0003,
    leverage_candidates: Sequence[float] = (1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0),
) -> Tuple[float, float]:
    """CPU fallback using NumPy."""
    mu_horizon = (mu_annual / 31536000.0) * horizon_sec
    
    best_lev = 5.0
    best_score = -1e9
    
    for lev in leverage_candidates:
        ev_long = mu_horizon * lev - fee_base * lev
        ev_short = -mu_horizon * lev - fee_base * lev
        score = max(ev_long, ev_short)
        if score > best_score:
            best_score = score
            best_lev = lev
            
    return best_lev, best_score


def find_optimal_leverage(
    mu_annual: float,
    sigma_annual: float,
    horizon_sec: int = 60,
    fee_base: float = 0.0003,
    use_gpu: bool = True,
) -> Tuple[float, float]:
    """Primary entry point for leverage optimization."""
    candidates = (1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0)
    
    if use_gpu and _JAX_OK:
        return find_optimal_leverage_gpu(mu_annual, sigma_annual, horizon_sec, fee_base, candidates)
    else:
        return find_optimal_leverage_cpu(mu_annual, sigma_annual, horizon_sec, fee_base, candidates)


def _warmup_jit_cache():
    """Warms up JAX JIT cache."""
    if not _JAX_OK:
        return
    try:
        # Dummy call to trigger compile
        _ = find_optimal_leverage_gpu(0.1, 0.3, 60, 0.0003)
        logger.info("✅ [JAX_LEV] Warmup completed")
    except Exception as e:
        logger.warning(f"⚠️ [JAX_LEV] Warmup failed: {e}")

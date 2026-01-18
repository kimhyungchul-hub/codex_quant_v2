"""
GPU-Accelerated Leverage Optimizer using PyTorch MPS (Metal Performance Shaders).

Replaces JAX Metal implementation which has bugs in version 0.8.2.
PyTorch MPS is stable and fully supported on Apple Silicon.
"""

import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch with MPS support
_TORCH_MPS_AVAILABLE = False
_TORCH_INIT_ERROR = None

try:
    import torch
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        _TORCH_MPS_AVAILABLE = True
        _MPS_DEVICE = torch.device("mps")
        logger.info("[TORCH_MPS] Successfully configured Metal GPU backend")
    else:
        _MPS_DEVICE = torch.device("cpu")
        logger.warning("[TORCH_MPS] MPS not available, using CPU")
        
except ImportError as e:
    _TORCH_INIT_ERROR = f"PyTorch not available: {e}"
    logger.warning(f"[TORCH_MPS] {_TORCH_INIT_ERROR}")
    torch = None
    _MPS_DEVICE = None


def find_optimal_leverage_gpu(
    mu_annual: float,
    sigma_annual: float,
    horizon_sec: int = 60,
    fee_base: float = 0.0003,
    leverage_candidates: Tuple[float, ...] = (1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0),
) -> Tuple[float, float]:
    """
    Find optimal leverage using GPU-accelerated parallel evaluation via PyTorch MPS.
    
    Args:
        mu_annual: Annual drift
        sigma_annual: Annual volatility
        horizon_sec: Horizon in seconds
        fee_base: Base fee (roundtrip)
        leverage_candidates: Tuple of leverage values to evaluate
    
    Returns:
        (optimal_leverage, optimal_score)
    """
    if torch is None or not _TORCH_MPS_AVAILABLE:
        return find_optimal_leverage_cpu(
            mu_annual, sigma_annual, horizon_sec, fee_base, leverage_candidates
        )
    
    try:
        # Convert to PyTorch tensor on GPU
        lev_tensor = torch.tensor(leverage_candidates, dtype=torch.float32, device=_MPS_DEVICE)
        
        # Compute scores for both long and short
        seconds_per_year = 31536000.0
        mu_ps = mu_annual / seconds_per_year
        mu_horizon = mu_ps * float(horizon_sec)
        
        # EV for long and short
        ev_long = mu_horizon * lev_tensor - fee_base * lev_tensor
        ev_short = -mu_horizon * lev_tensor - fee_base * lev_tensor
        
        # Score = max(ev_long, ev_short)
        scores = torch.maximum(ev_long, ev_short)
        
        # Find best
        best_idx = torch.argmax(scores)
        optimal_leverage = lev_tensor[best_idx].item()
        optimal_score = scores[best_idx].item()
        
        return float(optimal_leverage), float(optimal_score)
        
    except Exception as e:
        logger.warning(f"[TORCH_MPS] GPU evaluation failed, using CPU fallback: {e}")
        return find_optimal_leverage_cpu(
            mu_annual, sigma_annual, horizon_sec, fee_base, leverage_candidates
        )


def find_optimal_leverage_cpu(
    mu_annual: float,
    sigma_annual: float,
    horizon_sec: int = 60,
    fee_base: float = 0.0003,
    leverage_candidates: Tuple[float, ...] = (1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0),
) -> Tuple[float, float]:
    """CPU fallback (non-GPU)."""
    best_lev = 5.0
    best_score = -float('inf')
    
    seconds_per_year = 31536000.0
    mu_ps = mu_annual / seconds_per_year
    mu_horizon = mu_ps * horizon_sec
    
    for lev in leverage_candidates:
        ev_long = mu_horizon * lev - fee_base * lev
        ev_short = -mu_horizon * lev - fee_base * lev
        score = max(ev_long, ev_short)
        
        if score > best_score:
            best_score = score
            best_lev = lev
    
    return best_lev, best_score


# Main API
def find_optimal_leverage(
    mu_annual: float,
    sigma_annual: float,
    horizon_sec: int = 60,
    fee_base: float = 0.0003,
    use_gpu: bool = True,
) -> Tuple[float, float]:
    """
    Find optimal leverage (GPU or CPU).
    
    Args:
        mu_annual: Annual drift
        sigma_annual: Annual volatility
        horizon_sec: Horizon in seconds
        fee_base: Base fee
        use_gpu: Use GPU if True, CPU otherwise
    
    Returns:
        (optimal_leverage, optimal_score)
    """
    leverage_candidates = (1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0)
    
    if use_gpu and _TORCH_MPS_AVAILABLE:
        return find_optimal_leverage_gpu(
            mu_annual, sigma_annual, horizon_sec, fee_base, leverage_candidates
        )
    else:
        return find_optimal_leverage_cpu(
            mu_annual, sigma_annual, horizon_sec, fee_base, leverage_candidates
        )

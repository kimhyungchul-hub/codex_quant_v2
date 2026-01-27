"""PyTorch-first backend with NumPy fallback.

Primary: PyTorch with GPU acceleration (MPS for Apple Silicon, CUDA for NVIDIA)
Fallback: NumPy for CPU-only environments

All JAX code has been removed and replaced with PyTorch.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import PyTorch first
_TORCH_OK = False
_TORCH_DEVICE = None
torch: Any = None

try:
    import torch as _torch
    torch = _torch
    
    # Determine best available device
    if torch.cuda.is_available():
        _TORCH_DEVICE = torch.device("cuda")
        _TORCH_OK = True
        logger.info(f"[TORCH_BACKEND] CUDA available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _TORCH_DEVICE = torch.device("mps")
        _TORCH_OK = True
        logger.info("[TORCH_BACKEND] Apple Metal (MPS) backend available")
    else:
        _TORCH_DEVICE = torch.device("cpu")
        _TORCH_OK = True
        logger.info("[TORCH_BACKEND] Using CPU backend")
except ImportError:
    logger.warning("[TORCH_BACKEND] PyTorch not available, falling back to NumPy")
    _TORCH_OK = False

# NumPy fallback
import numpy as np

# Compatibility flags
DEV_MODE = not _TORCH_OK  # True if using NumPy fallback
_JAX_OK = False  # JAX permanently disabled
_JAX_WARMED = True

# Legacy JAX compatibility exports (always None)
jax: Any = None
jrand: Any = None
lax: Any = None


def get_device():
    """Returns the primary compute device."""
    return _TORCH_DEVICE if _TORCH_OK else None


def to_torch(arr, device=None, dtype=None):
    """Convert array-like to PyTorch tensor."""
    if not _TORCH_OK:
        return np.asarray(arr)
    
    if device is None:
        device = _TORCH_DEVICE
    if dtype is None:
        dtype = torch.float32
    
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    return torch.tensor(arr, device=device, dtype=dtype)


def to_numpy(tensor):
    """Convert PyTorch tensor to NumPy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return np.asarray(tensor)


# Unified array interface (PyTorch-first, NumPy fallback)
class UnifiedArrayBackend:
    """Provides numpy-like interface using PyTorch when available."""
    
    def __getattr__(self, name):
        if _TORCH_OK and hasattr(torch, name):
            return getattr(torch, name)
        return getattr(np, name)
    
    def array(self, arr, dtype=None):
        if _TORCH_OK:
            dt = dtype if dtype else torch.float32
            return to_torch(arr, dtype=dt)
        return np.array(arr, dtype=dtype)
    
    def asarray(self, arr, dtype=None):
        return self.array(arr, dtype=dtype)
    
    def mean(self, arr, axis=None, **kwargs):
        if _TORCH_OK and isinstance(arr, torch.Tensor):
            if axis is None:
                return torch.mean(arr, **kwargs)
            # PyTorch uses 'dim' instead of 'axis'
            return torch.mean(arr, dim=axis, **kwargs)
        return np.mean(arr, axis=axis, **kwargs)
    
    def sum(self, arr, axis=None, **kwargs):
        if _TORCH_OK and isinstance(arr, torch.Tensor):
            if axis is None:
                return torch.sum(arr, **kwargs)
            return torch.sum(arr, dim=axis, **kwargs)
        return np.sum(arr, axis=axis, **kwargs)
    
    def sqrt(self, arr):
        if _TORCH_OK and isinstance(arr, torch.Tensor):
            return torch.sqrt(arr)
        return np.sqrt(arr)
    
    def exp(self, arr):
        if _TORCH_OK and isinstance(arr, torch.Tensor):
            return torch.exp(arr)
        return np.exp(arr)
    
    def log(self, arr):
        if _TORCH_OK and isinstance(arr, torch.Tensor):
            return torch.log(arr)
        return np.log(arr)


jnp = UnifiedArrayBackend()


def ensure_jax() -> None:
    """No-op function for JAX compatibility (JAX removed)."""
    pass


def lazy_jit(static_argnames=()):
    """No-op decorator (JAX JIT removed)."""
    def decorator(fn):
        return fn
    return decorator


def _cvar_numpy(returns, alpha: float = 0.05):
    """NumPy CVaR calculation."""
    n = len(returns)
    sorted_returns = np.sort(returns)
    k = max(1, int(n * alpha))
    return float(np.mean(sorted_returns[:k]))


def _cvar_torch(returns, alpha: float = 0.05):
    """PyTorch CVaR calculation (GPU accelerated)."""
    if not _TORCH_OK:
        return _cvar_numpy(returns, alpha)
    
    if not isinstance(returns, torch.Tensor):
        returns = to_torch(returns)
    
    n = returns.shape[0]
    sorted_returns, _ = torch.sort(returns)
    k = max(1, int(n * alpha))
    return float(torch.mean(sorted_returns[:k]))


def _cvar_jax(returns, alpha: float = 0.05):
    """CVaR shim (uses PyTorch if available, else NumPy)."""
    if _TORCH_OK:
        return _cvar_torch(returns, alpha)
    return _cvar_numpy(returns, alpha)


def summarize_gbm_horizons_numpy(price_paths, s0, leverage, fee_rt_total_roe, horizons_indices, alpha=0.05):
    """NumPy implementation of GBM summary."""
    tp = price_paths[:, horizons_indices]
    gross_roe = (tp - s0) / s0 * leverage
    net_roe = gross_roe - fee_rt_total_roe
    
    ev = float(np.mean(net_roe, axis=0))
    cvar = _cvar_numpy(net_roe.flatten(), alpha)
    win_rate = float(np.mean(net_roe > 0, axis=0))
    
    return {
        "ev": ev,
        "cvar": cvar,
        "win_rate": win_rate,
    }


def summarize_gbm_horizons_torch(price_paths, s0, leverage, fee_rt_total_roe, horizons_indices, alpha=0.05):
    """PyTorch GPU-accelerated GBM summary."""
    if not _TORCH_OK:
        return summarize_gbm_horizons_numpy(price_paths, s0, leverage, fee_rt_total_roe, horizons_indices, alpha)
    
    # Convert to PyTorch tensors
    if not isinstance(price_paths, torch.Tensor):
        price_paths = to_torch(price_paths)
    
    tp = price_paths[:, horizons_indices]
    gross_roe = (tp - s0) / s0 * leverage
    net_roe = gross_roe - fee_rt_total_roe
    
    ev = float(torch.mean(net_roe, dim=0))
    cvar = _cvar_torch(net_roe.flatten(), alpha)
    win_rate = float(torch.mean((net_roe > 0).float(), dim=0))
    
    return {
        "ev": ev,
        "cvar": cvar,
        "win_rate": win_rate,
    }


def summarize_gbm_horizons_multi_symbol_jax(*args, **kwargs):
    """Compatibility shim - uses PyTorch if available."""
    if _TORCH_OK:
        return summarize_gbm_horizons_torch(*args, **kwargs)
    return summarize_gbm_horizons_numpy(*args, **kwargs)


def jax_covariance(returns_matrix):
    """Covariance calculation (PyTorch-first, NumPy fallback)."""
    if _TORCH_OK:
        if not isinstance(returns_matrix, torch.Tensor):
            returns_matrix = to_torch(returns_matrix)
        # PyTorch covariance (transposed for compatibility)
        cov = torch.cov(returns_matrix.T)
        return to_numpy(cov)
    return np.cov(returns_matrix, rowvar=False)


def _jax_mc_device():
    """Returns PyTorch device if available, else None."""
    return _TORCH_DEVICE if _TORCH_OK else None


# Export compatibility symbols
__all__ = [
    "ensure_jax",
    "lazy_jit",
    "_cvar_numpy",
    "_cvar_torch",
    "_cvar_jax",
    "summarize_gbm_horizons_numpy",
    "summarize_gbm_horizons_torch",
    "summarize_gbm_horizons_multi_symbol_jax",
    "jax_covariance",
    "_jax_mc_device",
    "get_device",
    "to_torch",
    "to_numpy",
    "jax",
    "jnp",
    "jrand",
    "lax",
    "_JAX_OK",
    "_TORCH_OK",
    "_TORCH_DEVICE",
    "torch",
    "DEV_MODE",
    "_JAX_WARMED",
]

# JAX Compatibility: use summarize_gbm_horizons_torch already defined above
summarize_gbm_horizons_jax = summarize_gbm_horizons_torch


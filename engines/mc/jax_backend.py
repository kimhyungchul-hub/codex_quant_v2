"""PyTorch-first backend with NumPy fallback.

This module keeps the legacy JAX-facing API but routes all compute to
the PyTorch backend (or NumPy fallback).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from engines.mc.torch_backend import (
    _TORCH_OK,
    torch,
    DEV_MODE,
    get_torch_device as _get_torch_device,
    to_torch,
    to_numpy,
    _cvar_numpy as _tb_cvar_numpy,
    _cvar_torch as _tb_cvar_torch,
    summarize_gbm_horizons_numpy as _tb_summarize_numpy,
    summarize_gbm_horizons_torch as _tb_summarize_torch,
    summarize_gbm_horizons_multi_symbol_torch as _tb_summarize_multi_torch,
    summarize_gbm_horizons_multi_symbol_numpy as _tb_summarize_multi_numpy,
    torch_covariance,
)

logger = logging.getLogger(__name__)

# Compatibility flags
_JAX_OK = False  # JAX permanently disabled
_JAX_WARMED = True

# Legacy JAX compatibility exports (always None)
jax: Any = None
jrand: Any = None
lax: Any = None


def get_device():
    """Returns the primary compute device."""
    return _get_torch_device() if _TORCH_OK else None


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
    return float(_tb_cvar_numpy(returns, alpha))


def _cvar_torch(returns, alpha: float = 0.05):
    return _tb_cvar_torch(returns, alpha)


def _cvar_jax(returns, alpha: float = 0.05):
    """CVaR shim (uses PyTorch if available, else NumPy)."""
    return _cvar_torch(returns, alpha) if _TORCH_OK else _cvar_numpy(returns, alpha)


def summarize_gbm_horizons_numpy(price_paths, s0, leverage, fee_rt_total_roe, horizons_indices, alpha=0.05):
    return _tb_summarize_numpy(price_paths, s0, leverage, fee_rt_total_roe, horizons_indices, alpha)


def summarize_gbm_horizons_torch(price_paths, s0, leverage, fee_rt_total_roe, horizons_indices, alpha=0.05):
    return _tb_summarize_torch(price_paths, s0, leverage, fee_rt_total_roe, horizons_indices, alpha)


def summarize_gbm_horizons_multi_symbol_jax(*args, **kwargs):
    """Compatibility shim - uses PyTorch if available."""
    if _TORCH_OK:
        return _tb_summarize_multi_torch(*args, **kwargs)
    return _tb_summarize_multi_numpy(*args, **kwargs)


def jax_covariance(returns_matrix):
    """Covariance calculation (PyTorch-first, NumPy fallback)."""
    if _TORCH_OK:
        return to_numpy(torch_covariance(returns_matrix))
    return np.cov(np.asarray(returns_matrix), rowvar=False)


def _jax_mc_device():
    """Returns PyTorch device if available, else None."""
    return _get_torch_device() if _TORCH_OK else None


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
    "_get_torch_device",
    "torch",
    "DEV_MODE",
    "_JAX_WARMED",
]

# JAX Compatibility: use summarize_gbm_horizons_torch already defined above
summarize_gbm_horizons_jax = summarize_gbm_horizons_torch

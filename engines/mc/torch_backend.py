"""PyTorch backend utilities for Monte Carlo simulations.

Provides device selection, tensor conversions, and GBM horizon summaries
with long/short metrics. Falls back to NumPy if PyTorch is unavailable.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# DEV_MODE: force CPU path (no GPU acceleration)
DEV_MODE = os.environ.get("DEV_MODE", "").lower() in ("1", "true", "yes")

# Lazy torch init
_TORCH_OK = False
torch: Any = None
_TORCH_DEVICE: Optional[Any] = None


def ensure_torch() -> None:
    """Lazily import torch and set device."""
    global _TORCH_OK, torch, _TORCH_DEVICE
    if torch is not None:
        return
    try:
        import torch as _torch  # type: ignore
        torch = _torch
        if DEV_MODE:
            _TORCH_DEVICE = torch.device("cpu")
            _TORCH_OK = True
            logger.info("[TORCH_BACKEND] DEV_MODE enabled, using CPU")
        else:
            if torch.cuda.is_available():
                _TORCH_DEVICE = torch.device("cuda")
                _TORCH_OK = True
                logger.info(f"[TORCH_BACKEND] CUDA available: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                _TORCH_DEVICE = torch.device("mps")
                _TORCH_OK = True
                logger.info("[TORCH_BACKEND] Apple Metal (MPS) backend available")
            else:
                _TORCH_DEVICE = torch.device("cpu")
                _TORCH_OK = True
                logger.info("[TORCH_BACKEND] Using CPU backend")

        # Optional warmup to reduce first-call jitter
        try:
            _ = torch.zeros((128, 128), device=_TORCH_DEVICE, dtype=torch.float32).sum()
            if _TORCH_DEVICE is not None and _TORCH_DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elif _TORCH_DEVICE is not None and _TORCH_DEVICE.type == "mps":
                torch.mps.synchronize()
        except Exception:
            pass
    except Exception as e:  # pragma: no cover
        logger.warning(f"[TORCH_BACKEND] PyTorch not available: {e}")
        _TORCH_OK = False
        torch = None
        _TORCH_DEVICE = None


def get_torch_device() -> Optional[Any]:
    ensure_torch()
    return _TORCH_DEVICE if _TORCH_OK else None


def get_device() -> Optional[Any]:
    return get_torch_device()


def to_torch(arr, device: Optional[Any] = None, dtype: Optional[Any] = None):
    """Convert array-like to torch tensor on target device."""
    ensure_torch()
    if not _TORCH_OK or torch is None:
        return np.asarray(arr)
    if device is None:
        device = _TORCH_DEVICE
    if dtype is None:
        dtype = torch.float32
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    return torch.tensor(arr, device=device, dtype=dtype)


def to_numpy(tensor):
    """Convert torch tensor to numpy array."""
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _cvar_numpy(returns: np.ndarray, alpha: float = 0.05) -> float:
    n = len(returns)
    if n <= 0:
        return 0.0
    sorted_returns = np.sort(returns)
    k = max(1, int(n * float(alpha)))
    return float(np.mean(sorted_returns[:k]))


def _cvar_torch(returns, alpha: float = 0.05):
    ensure_torch()
    if not _TORCH_OK or torch is None:
        return _cvar_numpy(np.asarray(returns), alpha)
    if not isinstance(returns, torch.Tensor):
        returns = to_torch(returns)
    n = returns.shape[0]
    if n <= 0:
        return torch.tensor(0.0, device=returns.device)
    sorted_returns, _ = torch.sort(returns)
    k = max(1, int(n * float(alpha)))
    return torch.mean(sorted_returns[:k])


def summarize_gbm_horizons_numpy(
    price_paths: np.ndarray,
    s0: float,
    leverage: float,
    fee_rt_total_roe: float,
    horizons_indices: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """NumPy GBM horizon summary (long/short)."""
    tp = price_paths[:, horizons_indices]
    denom = max(1e-12, float(s0))
    gross_ret = (tp - float(s0)) / denom

    net_long = gross_ret * float(leverage) - float(fee_rt_total_roe)
    net_short = -gross_ret * float(leverage) - float(fee_rt_total_roe)

    ev_long = np.mean(net_long, axis=0)
    win_long = np.mean(net_long > 0.0, axis=0)
    ev_short = np.mean(net_short, axis=0)
    win_short = np.mean(net_short > 0.0, axis=0)

    k = max(1, int(net_long.shape[0] * float(alpha)))
    part_long = np.partition(net_long, kth=k - 1, axis=0)
    part_short = np.partition(net_short, kth=k - 1, axis=0)
    cvar_long = np.mean(part_long[:k, :], axis=0)
    cvar_short = np.mean(part_short[:k, :], axis=0)

    return {
        "ev_long": ev_long,
        "win_long": win_long,
        "cvar_long": cvar_long,
        "ev_short": ev_short,
        "win_short": win_short,
        "cvar_short": cvar_short,
    }


def summarize_gbm_horizons_torch(
    price_paths,
    s0: float,
    leverage: float,
    fee_rt_total_roe: float,
    horizons_indices,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """PyTorch GBM horizon summary (long/short)."""
    ensure_torch()
    if not _TORCH_OK or torch is None:
        return summarize_gbm_horizons_numpy(
            np.asarray(price_paths), s0, leverage, fee_rt_total_roe, np.asarray(horizons_indices), alpha
        )

    device = _TORCH_DEVICE if _TORCH_DEVICE is not None else price_paths.device
    if not isinstance(price_paths, torch.Tensor):
        price_paths = to_torch(price_paths, device=device)
    if not isinstance(horizons_indices, torch.Tensor):
        horizons_indices = to_torch(horizons_indices, device=device, dtype=torch.long)

    tp = price_paths[:, horizons_indices]
    denom = max(1e-12, float(s0))
    gross_ret = (tp - float(s0)) / denom

    net_long = gross_ret * float(leverage) - float(fee_rt_total_roe)
    net_short = -gross_ret * float(leverage) - float(fee_rt_total_roe)

    ev_long = torch.mean(net_long, dim=0)
    win_long = torch.mean((net_long > 0.0).float(), dim=0)
    ev_short = torch.mean(net_short, dim=0)
    win_short = torch.mean((net_short > 0.0).float(), dim=0)

    k = max(1, int(net_long.shape[0] * float(alpha)))
    sorted_long, _ = torch.sort(net_long, dim=0)
    sorted_short, _ = torch.sort(net_short, dim=0)
    cvar_long = torch.mean(sorted_long[:k, :], dim=0)
    cvar_short = torch.mean(sorted_short[:k, :], dim=0)

    return {
        "ev_long": ev_long.cpu(),
        "win_long": win_long.cpu(),
        "cvar_long": cvar_long.cpu(),
        "ev_short": ev_short.cpu(),
        "win_short": win_short.cpu(),
        "cvar_short": cvar_short.cpu(),
    }


def summarize_gbm_horizons_multi_symbol_torch(
    price_paths,
    s0,
    leverage,
    fee_rt_total_roe,
    horizons_indices,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Multi-symbol GBM summary using torch (batched)."""
    ensure_torch()
    if not _TORCH_OK or torch is None:
        return summarize_gbm_horizons_multi_symbol_numpy(
            np.asarray(price_paths),
            np.asarray(s0),
            np.asarray(leverage),
            np.asarray(fee_rt_total_roe),
            np.asarray(horizons_indices),
            alpha,
        )

    device = _TORCH_DEVICE
    if not isinstance(price_paths, torch.Tensor):
        price_paths = to_torch(price_paths, device=device)
    if not isinstance(horizons_indices, torch.Tensor):
        horizons_indices = to_torch(horizons_indices, device=device, dtype=torch.long)

    s0_t = to_torch(s0, device=device)
    lev_t = to_torch(leverage, device=device)
    fee_t = to_torch(fee_rt_total_roe, device=device)

    tp = price_paths[:, :, horizons_indices]  # (n_symbols, n_paths, n_h)
    denom = torch.clamp(s0_t, min=1e-12)[:, None, None]
    gross_ret = (tp - s0_t[:, None, None]) / denom

    net_long = gross_ret * lev_t[:, None, None] - fee_t[:, None, None]
    net_short = -gross_ret * lev_t[:, None, None] - fee_t[:, None, None]

    ev_long = torch.mean(net_long, dim=1)
    win_long = torch.mean((net_long > 0.0).float(), dim=1)
    ev_short = torch.mean(net_short, dim=1)
    win_short = torch.mean((net_short > 0.0).float(), dim=1)

    k = max(1, int(net_long.shape[1] * float(alpha)))
    sorted_long, _ = torch.sort(net_long, dim=1)
    sorted_short, _ = torch.sort(net_short, dim=1)
    cvar_long = torch.mean(sorted_long[:, :k, :], dim=1)
    cvar_short = torch.mean(sorted_short[:, :k, :], dim=1)

    return {
        "ev_long": ev_long.cpu(),
        "win_long": win_long.cpu(),
        "cvar_long": cvar_long.cpu(),
        "ev_short": ev_short.cpu(),
        "win_short": win_short.cpu(),
        "cvar_short": cvar_short.cpu(),
    }


def summarize_gbm_horizons_multi_symbol_numpy(
    price_paths: np.ndarray,
    s0: np.ndarray,
    leverage: np.ndarray,
    fee_rt_total_roe: np.ndarray,
    horizons_indices: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Multi-symbol GBM summary (NumPy)."""
    tp = price_paths[:, :, horizons_indices]
    denom = np.maximum(1e-12, s0[:, None, None])
    gross_ret = (tp - s0[:, None, None]) / denom

    net_long = gross_ret * leverage[:, None, None] - fee_rt_total_roe[:, None, None]
    net_short = -gross_ret * leverage[:, None, None] - fee_rt_total_roe[:, None, None]

    ev_long = np.mean(net_long, axis=1)
    win_long = np.mean(net_long > 0.0, axis=1)
    ev_short = np.mean(net_short, axis=1)
    win_short = np.mean(net_short > 0.0, axis=1)

    k = max(1, int(net_long.shape[1] * float(alpha)))
    part_long = np.partition(net_long, kth=k - 1, axis=1)
    part_short = np.partition(net_short, kth=k - 1, axis=1)
    cvar_long = np.mean(part_long[:, :k, :], axis=1)
    cvar_short = np.mean(part_short[:, :k, :], axis=1)

    return {
        "ev_long": ev_long,
        "win_long": win_long,
        "cvar_long": cvar_long,
        "ev_short": ev_short,
        "win_short": win_short,
        "cvar_short": cvar_short,
    }


def torch_covariance(returns):
    """Covariance calculation (torch-first)."""
    ensure_torch()
    if not _TORCH_OK or torch is None:
        return np.cov(np.asarray(returns), rowvar=False)
    if not isinstance(returns, torch.Tensor):
        returns = to_torch(returns)
    mean_returns = torch.mean(returns, dim=0, keepdim=True)
    centered = returns - mean_returns
    n_obs = returns.shape[0]
    cov = torch.mm(centered.T, centered) / max(1, n_obs - 1)
    return cov.cpu()


# Initialize on import
ensure_torch()


from __future__ import annotations

from typing import Optional

import numpy as np

# Import PyTorch-first backend
from engines.mc.jax_backend import _TORCH_OK, torch, to_torch, to_numpy, get_device
from engines.mc.params import JOHNSON_SU_GAMMA, JOHNSON_SU_DELTA


class MonteCarloTailSamplingMixin:
    def _sample_increments_np(
        self,
        rng: np.random.Generator,
        shape,
        *,
        mode: str,
        df: float,
        bootstrap_returns: Optional[np.ndarray],
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
    ):
        """NumPy-based sampling (CPU fallback)."""
        gamma_val = float(JOHNSON_SU_GAMMA if gamma is None else gamma)
        delta_val = float(JOHNSON_SU_DELTA if delta is None else delta)
        delta_val = float(max(delta_val, 1e-6))
        if mode == "bootstrap" and bootstrap_returns is not None and bootstrap_returns.size >= 16:
            idx = rng.integers(0, bootstrap_returns.size, size=shape)
            return bootstrap_returns[idx].astype(np.float64)
        if mode == "student_t":
            z = rng.standard_t(df=df, size=shape).astype(np.float64)
            if df > 2:
                z = z / np.sqrt(df / (df - 2.0))
            return z
        if mode == "johnson_su":
            z = rng.standard_normal(size=shape).astype(np.float64)
            x = np.sinh((z - gamma_val) / delta_val)
            x_mean = float(np.mean(x))
            x_std = float(np.std(x))
            if not np.isfinite(x_std) or x_std < 1e-8:
                x_std = 1.0
            return (x - x_mean) / x_std
        return rng.standard_normal(size=shape).astype(np.float64)

    def _sample_increments_torch(
        self,
        shape,
        *,
        mode: str,
        df: float,
        bootstrap_returns: Optional[np.ndarray],
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        device=None,
    ):
        """PyTorch-based sampling (GPU accelerated)."""
        if not _TORCH_OK:
            return None
        
        if device is None:
            device = get_device()
        
        gamma_val = float(JOHNSON_SU_GAMMA if gamma is None else gamma)
        delta_val = float(JOHNSON_SU_DELTA if delta is None else delta)
        delta_val = float(max(delta_val, 1e-6))
        
        if mode == "bootstrap" and bootstrap_returns is not None and bootstrap_returns.size >= 16:
            br_tensor = to_torch(bootstrap_returns, device=device)
            n = br_tensor.shape[0]
            idx = torch.randint(0, n, shape, device=device, dtype=torch.long)
            return br_tensor[idx]
        
        if mode == "student_t":
            # Student-t distribution
            dist = torch.distributions.StudentT(df=df)
            z = dist.sample(shape).to(device)
            if df > 2:
                scale = torch.sqrt(torch.tensor(df / (df - 2.0), device=device))
                z = z / scale
            return z
        
        if mode == "johnson_su":
            z = torch.randn(shape, device=device)
            x = torch.sinh((z - gamma_val) / delta_val)
            x_mean = torch.mean(x)
            x_std = torch.std(x)
            x_std = torch.where(x_std < 1e-8, torch.tensor(1.0, device=device), x_std)
            return (x - x_mean) / x_std
        
        # Gaussian (default)
        return torch.randn(shape, device=device)

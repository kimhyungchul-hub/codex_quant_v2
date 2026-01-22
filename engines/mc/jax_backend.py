"""JAX backend initialization with Metal GPU support (Python 3.11 + JAX 0.4.35)."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ✅ DEV_MODE: NumPy 전용 모드 (JAX 완전 비활성화)
DEV_MODE = os.environ.get("DEV_MODE", "").lower() in ("1", "true", "yes")

# Lazy JAX initialization to avoid importing/initializing JAX at module import time.
_JAX_OK = False
jax: Any = None
jnp: Any = None
jrand: Any = None
lax: Any = None


def ensure_jax() -> None:
    """Lazily import and initialize JAX. Safe to call multiple times.

    This function will set module-level `jax`, `jnp`, `jrand`, and `lax`.
    If initialization fails, `_JAX_OK` will remain False.
    
    ✅ DEV_MODE=true일 때 JAX를 로드하지 않고 NumPy로 대체합니다.
    """
    global _JAX_OK, jax, jnp, jrand, lax, DEV_MODE
    
    # DEV_MODE: JAX 완전 비활성화, NumPy 사용
    if DEV_MODE:
        if jnp is None:
            import numpy as _np_module
            jnp = _np_module  # numpy를 jnp로 사용
            logger.info("🔧 [DEV_MODE] JAX disabled, using NumPy fallback")
            _JAX_OK = False
        return
    
    if jax is not None:
        return
    try:
        import os as _os_module
        # Avoid forcing platform env vars here; rely on external bootstrap to set envs.
        import jax as _jax_module  # type: ignore
        import jax.numpy as _jnp_module  # type: ignore
        from jax import random as _jrand_module  # type: ignore
        from jax import lax as _lax_module  # type: ignore

        jax = _jax_module
        jnp = _jnp_module
        jrand = _jrand_module
        lax = _lax_module
        _JAX_OK = True

        devices = jax.devices()
        backend = None
        if devices:
            backend = getattr(devices[0], "platform", None)

        logger.info(f"✅ [JAX] Version {jax.__version__} loaded")
        if backend is not None:
            logger.info(f"✅ [JAX] Backend: {backend}")
        logger.info(f"✅ [JAX] Devices: {devices}")

        gpu_devices = [d for d in devices if getattr(d, 'platform', None) and str(d.platform).lower() != 'cpu']
        if not gpu_devices:
            logger.warning("⚠️  [JAX] No non-CPU devices found; GPU-only expectation not met")
    except ImportError as e:
        logger.warning(f"⚠️  [JAX] Not available: {e}")
        _JAX_OK = False
    except Exception as e:
        logger.warning(f"⚠️  [JAX] Initialization error: {e}")
        _JAX_OK = False


def lazy_jit(static_argnames=()):
    """Decorator to lazily apply `jax.jit` on first call.

    If JAX is unavailable or DEV_MODE is enabled, returns the function as-is without JIT.
    """
    def decorator(fn):
        compiled = {}

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # DEV_MODE: JAX JIT 비활성화, 원본 함수 실행
            if DEV_MODE:
                return fn(*args, **kwargs)
            
            ensure_jax()
            if not _JAX_OK or jax is None:
                raise RuntimeError("JAX is not available; cannot execute JAX-accelerated function")
            if 'fn' not in compiled:
                compiled['fn'] = jax.jit(fn, static_argnames=static_argnames)
            return compiled['fn'](*args, **kwargs)

        return wrapper

    return decorator


from functools import wraps, partial


def _cvar_numpy(returns, alpha: float = 0.05):
    """NumPy CVaR calculation."""
    import numpy as np
    n = len(returns)
    sorted_returns = np.sort(returns)
    k = int(n * alpha)
    if k > 0:
        return float(np.mean(sorted_returns[:k]))
    return float(sorted_returns[0])


@lazy_jit(static_argnames=("alpha",))
def _cvar_jax(returns: jnp.ndarray, alpha: float = 0.05) -> jnp.ndarray:
    """
    JAX-native CVaR (Conditional Value at Risk) 
    - returns: (n_paths,)
    - alpha: confidence level (default 0.05)
    """
    n = returns.shape[0]
    sorted_returns = jnp.sort(returns)
    
    # Robust CVaR calculation using masking to handle Tracers
    # This avoids 'int()' conversion errors during JIT compilation
    k_float = n * alpha
    indices = jnp.arange(n)
    mask = indices < k_float # Select bottom alpha%
    
    count = jnp.sum(mask)
    cvar = jnp.where(
        count > 0,
        jnp.sum(sorted_returns * mask) / count,
        sorted_returns[0]  # Fallback to minimum if mask is empty
    )
    return cvar


def summarize_gbm_horizons_numpy(price_paths, s0, leverage, fee_rt_total_roe, horizons_indices, alpha=0.05):
    """NumPy fallback for summarize_gbm_horizons_jax."""
    import numpy as np
    tp = price_paths[:, horizons_indices]
    gross_ret = (tp - s0) / max(1e-12, s0)
    
    # Long stats
    net_long = gross_ret * leverage - fee_rt_total_roe
    ev_long = np.mean(net_long, axis=0)
    win_long = np.mean(net_long > 0, axis=0)
    cvar_long = np.array([_cvar_numpy(net_long[:, i], alpha) for i in range(net_long.shape[1])])
    
    # Short stats
    net_short = -gross_ret * leverage - fee_rt_total_roe
    ev_short = np.mean(net_short, axis=0)
    win_short = np.mean(net_short > 0, axis=0)
    cvar_short = np.array([_cvar_numpy(net_short[:, i], alpha) for i in range(net_short.shape[1])])
    
    return {
        "ev_long": ev_long,
        "win_long": win_long,
        "cvar_long": cvar_long,
        "ev_short": ev_short,
        "win_short": win_short,
        "cvar_short": cvar_short
    }


def summarize_gbm_horizons_jax(
    price_paths, 
    s0: float, 
    leverage: float, 
    fee_rt_total_roe: float, 
    horizons_indices, 
    alpha: float = 0.05
):
    """
    GPU-side summary of GBM price paths for multiple horizons.
    - price_paths: (n_paths, n_steps)
    - horizons_indices: (n_horizons,)
    
    ✅ DEV_MODE: NumPy fallback 사용
    """
    # DEV_MODE: NumPy fallback
    if DEV_MODE:
        import numpy as np
        return summarize_gbm_horizons_numpy(
            np.asarray(price_paths),
            s0, leverage, fee_rt_total_roe,
            np.asarray(horizons_indices, dtype=np.int32),
            alpha
        )
    
    # JAX version
    ensure_jax()
    if jax is None:
        raise RuntimeError("JAX not available for summarize_gbm_horizons_jax")
    
    # Convert to JAX arrays
    price_paths_j = jnp.asarray(price_paths)
    horizons_j = jnp.asarray(horizons_indices)
    
    # Select rows for each horizon
    # Shape: (n_paths, n_horizons)
    tp = price_paths_j[:, horizons_j]
    
    gross_ret = (tp - s0) / jnp.maximum(1e-12, s0)
    
    # Long stats
    net_long = gross_ret * leverage - fee_rt_total_roe
    ev_long = jnp.mean(net_long, axis=0) # (n_horizons,)
    win_long = jnp.mean(net_long > 0, axis=0)
    # vmap CVaR over horizons (column-wise)
    cvar_long = jax.vmap(_cvar_jax, in_axes=(1, None))(net_long, alpha)
    
    # Short stats
    net_short = -gross_ret * leverage - fee_rt_total_roe
    ev_short = jnp.mean(net_short, axis=0)
    win_short = jnp.mean(net_short > 0, axis=0)
    cvar_short = jax.vmap(_cvar_jax, in_axes=(1, None))(net_short, alpha)
    
    return {
        "ev_long": ev_long,
        "win_long": win_long,
        "cvar_long": cvar_long,
        "ev_short": ev_short,
        "win_short": win_short,
        "cvar_short": cvar_short
    }


# ✅ GLOBAL BATCHING: Multi-symbol version of GBM summary
# in_axes: (price_paths, s0, leverage, fee, horizons, alpha)
# We vmap over first 4 arguments (symbol-specific), horizons and alpha are shared.
def summarize_gbm_horizons_multi_symbol_jax(
    price_paths, s0, leverage, fee_rt_total_roe, horizons_indices, alpha=0.05
):
    """Lazily vmap the `summarize_gbm_horizons_jax` across symbols.

    This wrapper ensures JAX is initialized before creating the vmap.
    ✅ DEV_MODE: NumPy sequential fallback
    """
    # DEV_MODE: NumPy sequential fallback
    if DEV_MODE:
        import numpy as np
        results = []
        for i in range(len(s0)):
            r = summarize_gbm_horizons_numpy(
                np.asarray(price_paths[i]),
                float(s0[i]), float(leverage[i]), float(fee_rt_total_roe[i]),
                np.asarray(horizons_indices, dtype=np.int32),
                alpha
            )
            results.append(r)
        # Stack results
        return {
            "ev_long": np.array([r["ev_long"] for r in results]),
            "win_long": np.array([r["win_long"] for r in results]),
            "cvar_long": np.array([r["cvar_long"] for r in results]),
            "ev_short": np.array([r["ev_short"] for r in results]),
            "win_short": np.array([r["win_short"] for r in results]),
            "cvar_short": np.array([r["cvar_short"] for r in results]),
        }
    
    ensure_jax()
    if not _JAX_OK:
        raise RuntimeError("JAX not available for summarize_gbm_horizons_multi_symbol_jax")
    vm = jax.vmap(summarize_gbm_horizons_jax, in_axes=(0, 0, 0, 0, None, None))
    return vm(price_paths, s0, leverage, fee_rt_total_roe, horizons_indices, alpha)


@lazy_jit()
def jax_covariance(returns: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-accelerated covariance matrix calculation.
    - returns: (n_observations, n_assets)
    Returns: (n_assets, n_assets)
    """
    # jnp.cov expects (n_features, n_observations)
    return jnp.cov(returns.T)


def _jax_mc_device() -> Optional[Any]:


    """
    Returns the device for MC simulations.
    
    - If JAX_MC_DEVICE=cpu is set, forces CPU
    - Otherwise uses default backend (Metal GPU if available)
    """
    ensure_jax()
    if not _JAX_OK or jax is None:
        return None

    try:
        # Select the first non-CPU device (GPU/Metal). Do not return CPU devices.
        devs = jax.devices()
        for d in devs:
            plat = getattr(d, 'platform', None)
            if plat and str(plat).lower() != 'cpu':
                return d
        # If no GPU device is present, raise instead of falling back silently.
        raise RuntimeError("No GPU device available for JAX MC kernels (GPU-only mode)")
    except Exception as e:
        logger.warning(f"⚠️  [JAX] Device selection error: {e}")
        return None


# NOTE: Do NOT auto-initialize JAX at module import time here.
# Auto-initialization caused JAX to be loaded before environment variables
# (XLA_PYTHON_CLIENT_*) could be applied in some entrypoints, leading to
# undesired Metal memory reservation. Call `ensure_jax()` explicitly
# from application entrypoints (for example from `main_engine_mc_v2_final.py`)
# after environment variables are set.

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

try:
    import jax.numpy as jnp  # type: ignore
except Exception:  # pragma: no cover
    jnp = None


def _cvar_empirical(pnl: np.ndarray, alpha: float = 0.05) -> float:
    x = np.sort(np.asarray(pnl, dtype=np.float64))
    k = max(1, int(alpha * len(x)))
    return float(x[:k].mean())


def _cvar_bootstrap(
    pnl: np.ndarray,
    alpha: float = 0.05,
    n_boot: Optional[int] = None,
    sample_frac: float = 0.7,
    seed: int = 42,
) -> float:
    if n_boot is None:
        n_boot = int(os.environ.get("MC_N_BOOT", "40"))

    if n_boot <= 1:
        return _cvar_empirical(pnl, alpha)

    rng = np.random.default_rng(seed)
    x = np.asarray(pnl, dtype=np.float64)
    n = len(x)
    m = max(30, int(n * sample_frac))
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=m, replace=True)
        vals.append(_cvar_empirical(samp, alpha))
    return float(np.median(vals))


def _cvar_tail_inflate(pnl: np.ndarray, alpha: float = 0.05, inflate: float = 1.15) -> float:
    x = np.asarray(pnl, dtype=np.float64)
    var = float(np.quantile(x, alpha))
    tail = x[x <= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    if tail.size < 100:
        cvar *= float(inflate)
    return float(cvar)


def cvar_ensemble(pnl: Sequence[float], alpha: float = 0.05) -> float:
    if jnp is not None:
        try:
            x = jnp.asarray(pnl, dtype=jnp.float32)
            if x.size < 50:
                return float(_cvar_jnp(x, alpha))
            # Simplified ensemble for JAX to avoid bootstrap overhead
            return float(_cvar_jnp(x, alpha))
        except Exception:
            pass
            
    x = np.asarray(pnl, dtype=np.float64)
    if x.size < 50:
        return float(_cvar_empirical(x, alpha))
    a = _cvar_empirical(x, alpha)
    b = _cvar_bootstrap(x, alpha)
    c = _cvar_tail_inflate(x, alpha)
    return float(0.60 * b + 0.25 * a + 0.15 * c)


def _cvar_jnp(x: "jnp.ndarray", alpha: float) -> "jnp.ndarray":  # type: ignore[name-defined]
    if jnp is None:  # pragma: no cover
        raise RuntimeError("JAX is not available")
    xs = jnp.sort(x)  # type: ignore[attr-defined]
    k = jnp.asarray(xs.shape[0] * alpha, dtype=jnp.int32)  # type: ignore[attr-defined]
    k = jnp.maximum(1, k)  # type: ignore[attr-defined]
    return jnp.mean(xs[:k])  # type: ignore[attr-defined]

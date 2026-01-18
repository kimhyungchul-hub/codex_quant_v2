from __future__ import annotations

from typing import Optional

import numpy as np

from engines.mc.jax_backend import jnp, jrand


class MonteCarloTailSamplingMixin:
    def _sample_increments_np(self, rng: np.random.Generator, shape, *, mode: str, df: float, bootstrap_returns: Optional[np.ndarray]):
        if mode == "bootstrap" and bootstrap_returns is not None and bootstrap_returns.size >= 16:
            idx = rng.integers(0, bootstrap_returns.size, size=shape)
            return bootstrap_returns[idx].astype(np.float64)
        if mode == "student_t":
            z = rng.standard_t(df=df, size=shape).astype(np.float64)
            if df > 2:
                z = z / np.sqrt(df / (df - 2.0))
            return z
        return rng.standard_normal(size=shape).astype(np.float64)

    def _sample_increments_jax(self, key, shape, *, mode: str, df: float, bootstrap_returns: Optional[np.ndarray]):
        if jrand is None:
            return key, None
        if mode == "bootstrap" and bootstrap_returns is not None:
            # ✅ 해결: boot를 무조건 jnp.asarray로 정규화하고 shape[0]을 Python int로 변환
            br = jnp.asarray(bootstrap_returns, dtype=jnp.float32)  # type: ignore[attr-defined]
            br_size = int(br.shape[0])  # Python int로 변환 (tracer 방지)
            if br_size >= 16:
                key, k1 = jrand.split(key)  # type: ignore[attr-defined]
                idx = jrand.randint(k1, shape=shape, minval=0, maxval=br_size)  # type: ignore[attr-defined]
                return key, br[idx]
        if mode == "student_t":
            key, k1 = jrand.split(key)  # type: ignore[attr-defined]
            z = jrand.t(k1, df=df, shape=shape)  # type: ignore[attr-defined]
            if df > 2:
                z = z / jnp.sqrt(df / (df - 2.0))  # type: ignore[attr-defined]
            return key, z
        key, k1 = jrand.split(key)  # type: ignore[attr-defined]
        return key, jrand.normal(k1, shape=shape)  # type: ignore[attr-defined]

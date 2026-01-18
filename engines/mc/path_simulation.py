from __future__ import annotations

import logging
import math
from typing import Dict, Sequence

import numpy as np

from engines.mc.jax_backend import _JAX_OK, _jax_mc_device, jax, jnp, jrand

logger = logging.getLogger(__name__)


def _simulate_paths_price_jax_core(
    key,
    s0: float,
    drift: float,
    diffusion: float,
    n_paths: int,
    n_steps: int,
    mode: str,
    df: float,
    boot_jnp,
) -> "jnp.ndarray":  # type: ignore[name-defined]
    """JAX JIT-compiled GBM path generation core.

    Returns shape (n_paths, n_steps + 1) with paths[:,0] = s0.
    """
    if mode == "bootstrap" and boot_jnp is not None:
        br_size = int(boot_jnp.shape[0])
        if br_size >= 16:
            key, k1 = jrand.split(key)  # type: ignore[attr-defined]
            idx = jrand.randint(k1, shape=(n_paths, n_steps), minval=0, maxval=br_size)  # type: ignore[attr-defined]
            z = boot_jnp[idx]
        else:
            key, k1 = jrand.split(key)  # type: ignore[attr-defined]
            z = jrand.normal(k1, shape=(n_paths, n_steps))  # type: ignore[attr-defined]
    elif mode == "student_t":
        key, k1 = jrand.split(key)  # type: ignore[attr-defined]
        z = jrand.t(k1, df=df, shape=(n_paths, n_steps))  # type: ignore[attr-defined]
        if df > 2:
            z = z / jnp.sqrt(df / (df - 2.0))  # type: ignore[attr-defined]
    else:
        key, k1 = jrand.split(key)  # type: ignore[attr-defined]
        z = jrand.normal(k1, shape=(n_paths, n_steps))  # type: ignore[attr-defined]

    z = z.astype(jnp.float32)  # type: ignore[attr-defined]
    logret = jnp.cumsum(drift + diffusion * z, axis=1)  # type: ignore[attr-defined]
    prices_1 = s0 * jnp.exp(logret)  # type: ignore[attr-defined]
    paths = jnp.concatenate(  # type: ignore[attr-defined]
        [jnp.full((n_paths, 1), s0, dtype=jnp.float32), prices_1],  # type: ignore[attr-defined]
        axis=1,
    )
    return paths


if _JAX_OK and jax is not None:
    _simulate_paths_price_jax_core_jit = jax.jit(  # type: ignore[attr-defined]
        _simulate_paths_price_jax_core,
        static_argnames=("mode", "n_paths", "n_steps", "df"),
    )
    
    # ✅ GLOBAL BATCHING: Multi-symbol path generation
    # in_axes: (key, s0, drift, diffusion, n_paths, n_steps, mode, df, boot_jnp)
    # vmap over (key, s0, drift, diffusion). 
    # n_paths, n_steps, mode, df, boot_jnp are shared (static or single array).
    _simulate_paths_price_batch_jax = jax.vmap(
        _simulate_paths_price_jax_core_jit,
        in_axes=(0, 0, 0, 0, None, None, None, None, None)
    )

else:  # pragma: no cover
    _simulate_paths_price_jax_core_jit = None


class MonteCarloPathSimulationMixin:
    def simulate_paths_price(
        self,
        *,
        seed: int,
        s0: float,
        mu: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
        dt: float,
        return_jax: bool = False,
    ) -> np.ndarray | jnp.ndarray:
        """
        1초 단위 가격 경로를 생성한다 (JAX JIT 최적화 버전).
        - 반환 shape: (n_paths, n_steps+1)
          paths[:,0] = s0 (t=0), paths[:,t] = price at t seconds
        """
        n_paths_i = int(max(1, int(n_paths)))
        n_steps_i = int(max(1, int(n_steps)))

        drift = (float(mu) - 0.5 * float(sigma) * float(sigma)) * float(dt)
        diffusion = float(sigma) * math.sqrt(float(dt))

        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        br = getattr(self, "_bootstrap_returns", None)
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK

        if use_jax and _simulate_paths_price_jax_core_jit is not None:
            try:
                key = jrand.PRNGKey(int(seed) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                boot_jnp = None if br is None else jnp.asarray(br, dtype=jnp.float32)  # type: ignore[attr-defined]
                
                # JIT 컴파일된 핵심 함수 호출
                paths_jnp = _simulate_paths_price_jax_core_jit(
                    key, s0, drift, diffusion, n_paths_i, n_steps_i, mode, df, boot_jnp
                )
                
                if return_jax:
                    return paths_jnp

                # JAX 배열을 NumPy로 변환
                paths = np.asarray(jax.device_get(paths_jnp), dtype=np.float64)  # type: ignore[attr-defined]
                return paths
            except Exception:
                use_jax = False

        # Fallback to NumPy implementation
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        z = self._sample_increments_np(rng, (n_paths_i, n_steps_i), mode=mode, df=df, bootstrap_returns=br)
        logret = np.cumsum(drift + diffusion * z, axis=1)
        prices_1 = float(s0) * np.exp(logret)

        paths = np.empty((n_paths_i, n_steps_i + 1), dtype=np.float64)
        paths[:, 0] = float(s0)
        paths[:, 1:] = prices_1
        return paths


    def simulate_paths_price_batch(
        self,
        *,
        seeds: np.ndarray,
        s0s: np.ndarray,
        mus: np.ndarray,
        sigmas: np.ndarray,
        n_paths: int,
        n_steps: int,
        dt: float,
    ) -> jnp.ndarray:
        """
        GLOBAL BATCHING: 여러 심볼의 가격 경로를 한 번에 생성한다.
        - 반환 shape: (num_symbols, n_paths, n_steps+1)
        """
        num_symbols = len(seeds)
        n_paths_i = int(max(1, int(n_paths)))
        n_steps_i = int(max(1, int(n_steps)))
        
        drifts = (mus - 0.5 * sigmas * sigmas) * float(dt)
        diffusions = sigmas * math.sqrt(float(dt))
        
        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        br = getattr(self, "_bootstrap_returns", None)
        
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK
        
        if use_jax and _simulate_paths_price_batch_jax is not None:
            # PRNGKeys for each symbol
            # Generate keys more stable way
            master_key = jrand.PRNGKey(int(seeds[0]) & 0xFFFFFFFF)
            keys = jrand.split(master_key, len(seeds))
            boot_jnp = None if br is None else jnp.asarray(br, dtype=jnp.float32)
            
            # vmapped path generation
            paths_jnp = _simulate_paths_price_batch_jax(
                keys, s0s, drifts, diffusions, n_paths_i, n_steps_i, mode, df, boot_jnp
            )
            return paths_jnp
            
        raise RuntimeError("JAX batch simulation requested but JAX is not available.")



    def simulate_paths_netpnl(
        self,
        seed: int,
        s0: float,
        mu: float,
        sigma: float,
        direction: int,
        leverage: float,
        n_paths: int,
        horizons: Sequence[int],
        dt: float,
        fee_roundtrip: float,
    ) -> Dict[int, np.ndarray]:
        """
        horizon별 net_pnl paths 반환
        """
        # ✅ Step A: MC 입력이 진짜 0인지 확인
        if not hasattr(self, "_sim_input_logged"):
            self._sim_input_logged = True
            logger.info(
                f"[SIM_INPUT] mu={mu:.10f} sigma={sigma:.10f} fee_rt={fee_roundtrip:.6f} "
                f"dir={direction} dt={dt} horizons={list(horizons)} s0={s0:.2f} n_paths={n_paths}"
            )
        
        rng = np.random.default_rng(seed)
        max_steps = int(max(horizons))
        drift = (mu - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)

        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        br = getattr(self, "_bootstrap_returns", None)
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK

        prices: np.ndarray
        if use_jax:
            # ✅ GPU 우선: default backend (GPU/Metal) 사용
            force_cpu_dev = _jax_mc_device()
            try:
                if force_cpu_dev is None:
                    # GPU/Metal default backend 사용
                    key = jrand.PRNGKey(int(seed) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                    key, z_j = self._sample_increments_jax(
                        key,
                        (int(n_paths), int(max_steps)),
                        mode=mode,
                        df=df,
                        bootstrap_returns=br,
                    )
                    if z_j is None:
                        use_jax = False
                    else:
                        z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                        logret = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                        prices = float(s0) * jnp.exp(logret)  # type: ignore[attr-defined]
                        prices = np.asarray(jax.device_get(prices), dtype=np.float64)  # type: ignore[attr-defined]
                else:
                    # CPU로 강제된 경우만 CPU 사용 (env JAX_MC_DEVICE=cpu)
                    with jax.default_device(force_cpu_dev):  # type: ignore[attr-defined]
                        key = jrand.PRNGKey(int(seed) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                        key, z_j = self._sample_increments_jax(
                            key,
                            (int(n_paths), int(max_steps)),
                            mode=mode,
                            df=df,
                            bootstrap_returns=br,
                        )
                        if z_j is None:
                            use_jax = False
                        else:
                            z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                            logret = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                            prices = float(s0) * jnp.exp(logret)  # type: ignore[attr-defined]
                            prices = np.asarray(jax.device_get(prices), dtype=np.float64)  # type: ignore[attr-defined]
            except Exception:
                use_jax = False


        if not use_jax:
            z = self._sample_increments_np(rng, (n_paths, max_steps), mode=mode, df=df, bootstrap_returns=br)
            logret = np.cumsum(drift + diffusion * z, axis=1)
            prices = s0 * np.exp(logret)

        out = {}
        for h in horizons:
            idx = int(h) - 1
            tp = prices[:, idx]
            gross = direction * (tp - s0) / s0 * float(leverage)
            net = gross - fee_roundtrip
            out[int(h)] = net.astype(np.float64)
        return out

from __future__ import annotations

# Implemented Antithetic Variates for variance reduction
import logging
import os
import math
from typing import Dict, Sequence

import numpy as np

from engines.mc.jax_backend import _JAX_OK, jax, jnp, jrand
from engines.mc.torch_backend import _TORCH_OK, torch, DEV_MODE, get_torch_device, to_numpy
from engines.mc.constants import EPSILON
from engines.mc.params import JOHNSON_SU_GAMMA, JOHNSON_SU_DELTA

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
    gamma: float,
    delta: float,
    boot_jnp,
) -> "jnp.ndarray":  # type: ignore[name-defined]
    """JAX JIT-compiled GBM path generation core.

    Returns shape (n_paths, n_steps + 1) with paths[:,0] = s0.
    `drift` must already reflect the chosen noise model (Gaussian uses Ito, heavy-tail/boot removes it).
    """
    half_paths = n_paths // 2
    has_odd_path = (n_paths % 2) == 1

    mode_effective = mode
    _sinh = lambda x: 0.5 * (jnp.exp(x) - jnp.exp(-x))  # type: ignore[attr-defined]
    if mode == "bootstrap" and boot_jnp is not None:
        br_size = int(boot_jnp.shape[0])
        if br_size >= 16:
            key, k1 = jrand.split(key)  # type: ignore[attr-defined]
            idx = jrand.randint(k1, shape=(n_paths, n_steps), minval=0, maxval=br_size)  # type: ignore[attr-defined]
            z = boot_jnp[idx]
        else:
            mode_effective = "gaussian"
    elif mode == "bootstrap":
        mode_effective = "gaussian"

    if mode_effective == "student_t":
        noises = []
        if half_paths > 0:
            key, k_half = jrand.split(key)  # type: ignore[attr-defined]
            z_half = jrand.t(k_half, df=df, shape=(half_paths, n_steps))  # type: ignore[attr-defined]
            if df > 2:
                z_half = z_half / jnp.sqrt(df / (df - 2.0))  # type: ignore[attr-defined]
            noises.append(z_half)
            noises.append(-z_half)
        if has_odd_path:
            key, k_extra = jrand.split(key)  # type: ignore[attr-defined]
            z_extra = jrand.t(k_extra, df=df, shape=(1, n_steps))  # type: ignore[attr-defined]
            if df > 2:
                z_extra = z_extra / jnp.sqrt(df / (df - 2.0))  # type: ignore[attr-defined]
            noises.append(z_extra)
        if not noises:
            z = jnp.zeros((0, n_steps), dtype=jnp.float32)  # type: ignore[attr-defined]
        elif len(noises) == 1:
            z = noises[0]
        else:
            z = jnp.concatenate(noises, axis=0)  # type: ignore[attr-defined]
    elif mode_effective == "johnson_su":
        noises = []
        if half_paths > 0:
            key, k_half = jrand.split(key)  # type: ignore[attr-defined]
            z_half = jrand.normal(k_half, shape=(half_paths, n_steps))  # type: ignore[attr-defined]
            js_half = _sinh((z_half - gamma) / delta)
            noises.append(js_half)
            js_half_neg = _sinh((-z_half - gamma) / delta)
            noises.append(js_half_neg)
        if has_odd_path:
            key, k_extra = jrand.split(key)  # type: ignore[attr-defined]
            z_extra = jrand.normal(k_extra, shape=(1, n_steps))  # type: ignore[attr-defined]
            noises.append(_sinh((z_extra - gamma) / delta))
        if not noises:
            z = jnp.zeros((0, n_steps), dtype=jnp.float32)  # type: ignore[attr-defined]
        elif len(noises) == 1:
            z = noises[0]
        else:
            z = jnp.concatenate(noises, axis=0)  # type: ignore[attr-defined]
        z_mean = jnp.mean(z)  # type: ignore[attr-defined]
        z_std = jnp.std(z)  # type: ignore[attr-defined]
        z_std = jnp.where(z_std < 1e-8, 1.0, z_std)  # type: ignore[attr-defined]
        z = (z - z_mean) / z_std  # type: ignore[attr-defined]
    elif mode_effective == "gaussian":
        noises = []
        if half_paths > 0:
            key, k_half = jrand.split(key)  # type: ignore[attr-defined]
            z_half = jrand.normal(k_half, shape=(half_paths, n_steps))  # type: ignore[attr-defined]
            noises.append(z_half)
            noises.append(-z_half)
        if has_odd_path:
            key, k_extra = jrand.split(key)  # type: ignore[attr-defined]
            noises.append(jrand.normal(k_extra, shape=(1, n_steps)))  # type: ignore[attr-defined]
        if not noises:
            z = jnp.zeros((0, n_steps), dtype=jnp.float32)  # type: ignore[attr-defined]
        elif len(noises) == 1:
            z = noises[0]
        else:
            z = jnp.concatenate(noises, axis=0)  # type: ignore[attr-defined]
    else:
        # Bootstrap with sufficient history: use empirical distribution as-is
        z = z.astype(jnp.float32)  # type: ignore[attr-defined]
        logret = jnp.cumsum(drift + diffusion * z, axis=1)  # type: ignore[attr-defined]
        prices_1 = s0 * jnp.exp(logret)  # type: ignore[attr-defined]
        paths = jnp.concatenate(  # type: ignore[attr-defined]
            [jnp.full((n_paths, 1), s0, dtype=jnp.float32), prices_1],  # type: ignore[attr-defined]
            axis=1,
        )
        return paths

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
    # in_axes: (key, s0, drift, diffusion, n_paths, n_steps, mode, df, gamma, delta, boot_jnp)
    # vmap over (key, s0, drift, diffusion). 
    # n_paths, n_steps, mode, df, gamma, delta, boot_jnp are shared (static or single array).
    _simulate_paths_price_batch_jax = jax.vmap(
        _simulate_paths_price_jax_core_jit,
        in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None)
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
        return_torch: bool = False,
        return_stats: bool = False,
    ) -> np.ndarray | jnp.ndarray:
        """
        1초 단위 가격 경로를 생성한다 (JAX JIT 최적화 버전).
        - 반환 shape: (n_paths, n_steps+1)
          paths[:,0] = s0 (t=0), paths[:,t] = price at t seconds
        - tail mode: gaussian | student_t | bootstrap | johnson_su
        - return_stats=True 시 control variate 기반 평균(cv_mean) 포함
        """
        n_paths_i = int(max(1, int(n_paths)))
        n_steps_i = int(max(1, int(n_steps)))
        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        gamma = float(getattr(self, "_johnson_gamma", JOHNSON_SU_GAMMA))
        delta = float(max(getattr(self, "_johnson_delta", JOHNSON_SU_DELTA), 1e-6))
        br = getattr(self, "_bootstrap_returns", None)
        self._johnson_gamma = gamma
        self._johnson_delta = delta
        use_torch = bool(getattr(self, "_use_jax", True)) and _TORCH_OK and not DEV_MODE
        use_torch = bool(getattr(self, "_use_jax", True)) and _TORCH_OK and not DEV_MODE
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK
        # Mode-aware drift: Gaussian keeps Ito correction, heavy-tail/boot removes it to avoid EV bias
        if mode in ("student_t", "bootstrap", "johnson_su"):
            drift = float(mu) * float(dt)
        else:
            drift = (float(mu) - 0.5 * float(sigma) * float(sigma)) * float(dt)
        diffusion = float(sigma) * math.sqrt(float(dt))

        if use_torch:
            try:
                device = get_torch_device()
                if device is None:
                    raise RuntimeError("torch device unavailable")

                z = None
                br_size = int(br.shape[0]) if br is not None else 0
                bootstrap_ready = mode == "bootstrap" and br is not None and br_size >= 16
                half_paths = n_paths_i // 2
                has_odd_path = (n_paths_i % 2) == 1

                if bootstrap_ready:
                    torch.manual_seed(int(seed) & 0xFFFFFFFF)
                    z = self._sample_increments_torch(
                        (n_paths_i, n_steps_i),
                        mode=mode,
                        df=df,
                        bootstrap_returns=br,
                        gamma=gamma,
                        delta=delta,
                        device=device,
                    )
                else:
                    noises = []
                    if half_paths > 0:
                        torch.manual_seed(int(seed) & 0xFFFFFFFF)
                        z_half = self._sample_increments_torch(
                            (half_paths, n_steps_i),
                            mode=mode,
                            df=df,
                            bootstrap_returns=br,
                            gamma=gamma,
                            delta=delta,
                            device=device,
                        )
                        if z_half is not None:
                            noises.append(z_half)
                            noises.append(-z_half)
                    if has_odd_path:
                        torch.manual_seed((int(seed) + 1) & 0xFFFFFFFF)
                        z_extra = self._sample_increments_torch(
                            (1, n_steps_i),
                            mode=mode,
                            df=df,
                            bootstrap_returns=br,
                            gamma=gamma,
                            delta=delta,
                            device=device,
                        )
                        if z_extra is not None:
                            noises.append(z_extra)
                    if noises:
                        z = noises[0] if len(noises) == 1 else torch.cat(noises, dim=0)

                if z is None:
                    raise RuntimeError("torch sampling unavailable")

                logret = torch.cumsum(float(drift) + float(diffusion) * z, dim=1)
                clip_val = float(os.environ.get("MC_LOGRET_CLIP", "12.0"))
                if clip_val > 0:
                    logret = torch.clamp(logret, -clip_val, clip_val)
                prices_1 = float(s0) * torch.exp(logret)
                s0_col = torch.full((n_paths_i, 1), float(s0), device=prices_1.device, dtype=prices_1.dtype)
                paths_t = torch.cat([s0_col, prices_1], dim=1)

                if return_torch or return_jax:
                    if return_stats:
                        stats = self._control_variate_price_stats(to_numpy(paths_t), drift, diffusion)
                        return paths_t, stats
                    return paths_t

                paths = to_numpy(paths_t).astype(np.float64)
                if return_stats:
                    stats = self._control_variate_price_stats(paths, drift, diffusion)
                    return paths, stats
                return paths
            except Exception as e:
                logger.warning(f"[MC] Torch path simulation failed, falling back to NumPy: {e}")
                use_torch = False

        if use_jax and _simulate_paths_price_jax_core_jit is not None:
            try:
                key = jrand.PRNGKey(int(seed) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                boot_jnp = None if br is None else jnp.asarray(br, dtype=jnp.float32)  # type: ignore[attr-defined]
                
                # JIT 컴파일된 핵심 함수 호출
                paths_jnp = _simulate_paths_price_jax_core_jit(
                    key, s0, drift, diffusion, n_paths_i, n_steps_i, mode, df, gamma, delta, boot_jnp
                )
                
                if (return_jax or return_torch) and not return_stats:
                    return paths_jnp

                # JAX 배열을 NumPy로 변환 (control variates/검증용)
                paths = np.asarray(jax.device_get(paths_jnp), dtype=np.float64)  # type: ignore[attr-defined]
                stats = None
                if return_stats:
                    stats = self._control_variate_price_stats(paths, drift, diffusion)
                    if return_jax or return_torch:
                        return paths_jnp, stats
                elif return_jax or return_torch:
                    return paths_jnp

                # Optional verification: check empirical log drift vs target when requested
                if os.environ.get("MC_VERIFY_DRIFT"):
                    horizon = max(n_steps_i, 1) * float(dt)
                    empirical_log_mu = float(
                        np.mean(np.log(paths[:, -1] / float(s0)))
                    ) / horizon
                    target_log_mu = float(mu) if mode in ("student_t", "bootstrap", "johnson_su") else (
                        float(mu) - 0.5 * float(sigma) * float(sigma)
                    )
                    logger.debug(
                        f"[MC] drift check mode={mode} empirical={empirical_log_mu:.6f} target={target_log_mu:.6f}"
                    )
                if return_stats:
                    return paths, stats
                return paths
            except Exception:
                use_jax = False

        # Fallback to NumPy implementation
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        half_paths = n_paths_i // 2
        has_odd_path = (n_paths_i % 2) == 1
        br_size = int(br.shape[0]) if br is not None else 0
        bootstrap_ready = mode == "bootstrap" and br is not None and br_size >= 16

        if bootstrap_ready:
            z = self._sample_increments_np(
                rng,
                (n_paths_i, n_steps_i),
                mode=mode,
                df=df,
                bootstrap_returns=br,
            )
        else:
            noises = []
            if half_paths > 0:
                z_half = self._sample_increments_np(
                    rng,
                    (half_paths, n_steps_i),
                    mode=mode,
                    df=df,
                    bootstrap_returns=br,
                )
                noises.append(z_half)
                noises.append(-z_half)
            if has_odd_path:
                noises.append(
                    self._sample_increments_np(
                        rng,
                        (1, n_steps_i),
                        mode=mode,
                        df=df,
                        bootstrap_returns=br,
                    )
                )
            z = noises[0] if len(noises) == 1 else np.concatenate(noises, axis=0)

        logret = np.cumsum(drift + diffusion * z, axis=1)
        prices_1 = float(s0) * np.exp(logret)

        paths = np.empty((n_paths_i, n_steps_i + 1), dtype=np.float64)
        paths[:, 0] = float(s0)
        paths[:, 1:] = prices_1

        if return_stats:
            stats = self._control_variate_price_stats(paths, drift, diffusion)
            return paths, stats

        if os.environ.get("MC_VERIFY_DRIFT"):
            horizon = max(n_steps_i, 1) * float(dt)
            empirical_log_mu = float(np.mean(np.log(paths[:, -1] / float(s0)))) / horizon
            target_log_mu = float(mu) if mode in ("student_t", "bootstrap", "johnson_su") else (
                float(mu) - 0.5 * float(sigma) * float(sigma)
            )
            logger.debug(
                f"[MC] drift check mode={mode} empirical={empirical_log_mu:.6f} target={target_log_mu:.6f}"
            )
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
        return_torch: bool = False,
    ) -> jnp.ndarray:
        """
        GLOBAL BATCHING: 여러 심볼의 가격 경로를 한 번에 생성한다.
        - 반환 shape: (num_symbols, n_paths, n_steps+1)
        """
        num_symbols = len(seeds)
        n_paths_i = int(max(1, int(n_paths)))
        n_steps_i = int(max(1, int(n_steps)))
        
        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        br = getattr(self, "_bootstrap_returns", None)

        if mode in ("student_t", "bootstrap", "johnson_su"):
            drifts = mus * float(dt)
        else:
            drifts = (mus - 0.5 * sigmas * sigmas) * float(dt)
        diffusions = sigmas * math.sqrt(float(dt))

        use_torch = bool(getattr(self, "_use_jax", True)) and _TORCH_OK and not DEV_MODE
        # ✅ DEV_MODE: JAX 완전 비활성화
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK and not DEV_MODE

        if use_torch:
            try:
                device = get_torch_device()
                if device is None:
                    raise RuntimeError("torch device unavailable")

                s0s_np = np.asarray(s0s, dtype=np.float32)
                drifts_np = np.asarray(drifts, dtype=np.float32)
                diff_np = np.asarray(diffusions, dtype=np.float32)

                br = getattr(self, "_bootstrap_returns", None)
                bootstrap_ready = mode == "bootstrap" and br is not None and int(br.shape[0]) >= 16

                z_list = []
                for i in range(num_symbols):
                    seed_i = int(seeds[i]) & 0xFFFFFFFF
                    if bootstrap_ready:
                        torch.manual_seed(seed_i)
                        z_i = self._sample_increments_torch(
                            (n_paths_i, n_steps_i),
                            mode=mode,
                            df=df,
                            bootstrap_returns=br,
                            gamma=None,
                            delta=None,
                            device=device,
                        )
                    else:
                        half_paths = n_paths_i // 2
                        has_odd = (n_paths_i % 2) == 1
                        noises = []
                        if half_paths > 0:
                            torch.manual_seed(seed_i)
                            z_half = self._sample_increments_torch(
                                (half_paths, n_steps_i),
                                mode=mode,
                                df=df,
                                bootstrap_returns=br,
                                gamma=None,
                                delta=None,
                                device=device,
                            )
                            if z_half is not None:
                                noises.append(z_half)
                                noises.append(-z_half)
                        if has_odd:
                            torch.manual_seed((seed_i + 1) & 0xFFFFFFFF)
                            z_extra = self._sample_increments_torch(
                                (1, n_steps_i),
                                mode=mode,
                                df=df,
                                bootstrap_returns=br,
                                gamma=None,
                                delta=None,
                                device=device,
                            )
                            if z_extra is not None:
                                noises.append(z_extra)
                        z_i = noises[0] if len(noises) == 1 else torch.cat(noises, dim=0)
                    if z_i is None:
                        raise RuntimeError("torch sampling unavailable")
                    z_list.append(z_i)

                z = torch.stack(z_list, dim=0)
                drifts_t = torch.tensor(drifts_np, device=device).view(num_symbols, 1, 1)
                diff_t = torch.tensor(diff_np, device=device).view(num_symbols, 1, 1)
                logret = torch.cumsum(drifts_t + diff_t * z, dim=2)
                clip_val = float(os.environ.get("MC_LOGRET_CLIP", "12.0"))
                if clip_val > 0:
                    logret = torch.clamp(logret, -clip_val, clip_val)
                s0s_t = torch.tensor(s0s_np, device=device).view(num_symbols, 1, 1)
                prices_1 = s0s_t * torch.exp(logret)
                paths_t = torch.cat([s0s_t.expand(num_symbols, n_paths_i, 1), prices_1], dim=2)
                if return_torch:
                    return paths_t
                return to_numpy(paths_t)
            except Exception as e:
                logger.warning(f"[MC] Torch batch path simulation failed, falling back to NumPy: {e}")
                use_torch = False
        
        if use_jax and _simulate_paths_price_batch_jax is not None:
            try:
                # Pre-padding on CPU: 고정 shape (num_symbols, n_paths, n_steps)
                z_np = np.empty((num_symbols, n_paths_i, n_steps_i), dtype=np.float32)
                for i in range(num_symbols):
                    rng = np.random.default_rng(int(seeds[i]) & 0xFFFFFFFF)
                    z_np[i] = self._sample_increments_np(
                        rng,
                        (n_paths_i, n_steps_i),
                        mode=mode,
                        df=df,
                        bootstrap_returns=br,
                    )

                drifts_np = drifts[:, None, None].astype(np.float32, copy=False)
                diffusions_np = diffusions[:, None, None].astype(np.float32, copy=False)
                s0s_j = jnp.asarray(s0s, dtype=jnp.float32)[:, None, None]

                # CPU pre-cumsum (Metal mhlo.pad 회피) + GPU exp
                logret_np = np.cumsum(drifts_np + diffusions_np * z_np, axis=2, dtype=np.float64)
                clip_val = float(os.environ.get("MC_LOGRET_CLIP", "12.0"))
                if not np.isfinite(logret_np).all() or np.any(np.abs(logret_np) > clip_val):
                    logger.warning(
                        "[MC] logret overflow/NaN detected in batch; applying clip "
                        f"(+/-{clip_val}) and nan_to_num"
                    )
                logret_np = np.nan_to_num(logret_np, nan=0.0, posinf=clip_val, neginf=-clip_val)
                logret_np = np.clip(logret_np, -clip_val, clip_val)
                logret_j = jnp.asarray(logret_np, dtype=jnp.float32)
                prices_1 = s0s_j * jnp.exp(logret_j)
                prices_jnp = jnp.concatenate(
                    [jnp.broadcast_to(s0s_j, (num_symbols, n_paths_i, 1)), prices_1],
                    axis=2,
                )
                return prices_jnp
            except Exception as e:
                logger.warning(f"[MC] JAX batch path simulation failed, falling back to NumPy: {e}")
                use_jax = False
        
        # NumPy fallback for DEV_MODE or when JAX is unavailable
        paths_all = np.empty((num_symbols, n_paths_i, n_steps_i + 1), dtype=np.float64)
        for i in range(num_symbols):
            rng = np.random.default_rng(int(seeds[i]) & 0xFFFFFFFF)
            
            if mode == "bootstrap" and br is not None and len(br) > 0:
                # Bootstrap sampling
                idxs = rng.integers(0, len(br), size=(n_paths_i, n_steps_i))
                innovations = br[idxs]
            elif mode == "student_t" and df > 0:
                # Student-t distribution
                innovations = rng.standard_t(df, size=(n_paths_i, n_steps_i))
            elif mode == "johnson_su":
                innovations = self._sample_increments_np(
                    rng,
                    (n_paths_i, n_steps_i),
                    mode=mode,
                    df=df,
                    bootstrap_returns=br,
                )
            else:
                # Normal distribution
                innovations = rng.standard_normal(size=(n_paths_i, n_steps_i))
            
            # GBM path construction
            log_returns = drifts[i] + diffusions[i] * innovations
            log_prices = np.zeros((n_paths_i, n_steps_i + 1), dtype=np.float64)
            log_prices[:, 0] = np.log(s0s[i])
            log_prices[:, 1:] = np.cumsum(log_returns, axis=1) + np.log(s0s[i])
            paths_all[i] = np.exp(log_prices)
        
        return paths_all



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
        return_stats: bool = False,
    ) -> Dict[int, np.ndarray]:
        """
        horizon별 net_pnl paths 반환 (JAX Metal 호환성 수정: 명시적 디바이스 지정 제거)
        return_stats=True 시 control variate 기반 평균(cv_mean) 제공
        """
        # ✅ Step A: MC 입력이 진짜 0인지 확인
        if not hasattr(self, "_sim_input_logged"):
            self._sim_input_logged = True
            logger.info(
                f"[SIM_INPUT] mu={mu:.10f} sigma={sigma:.10f} fee_rt={fee_roundtrip:.6f} "
                f"dir={direction} dt={dt} horizons={list(horizons)} s0={s0:.2f} n_paths={n_paths}"
            )
        
        max_steps = int(max(horizons))
        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        gamma = float(getattr(self, "_johnson_gamma", JOHNSON_SU_GAMMA))
        delta = float(max(getattr(self, "_johnson_delta", JOHNSON_SU_DELTA), 1e-6))
        br = getattr(self, "_bootstrap_returns", None)
        self._johnson_gamma = gamma
        self._johnson_delta = delta
        if mode in ("student_t", "bootstrap", "johnson_su"):
            drift = (mu) * dt
        else:
            drift = (mu - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK

        prices: np.ndarray
        if use_torch:
            try:
                device = get_torch_device()
                if device is None:
                    raise RuntimeError("torch device unavailable")
                torch.manual_seed(int(seed) & 0xFFFFFFFF)
                z = self._sample_increments_torch(
                    (int(n_paths), int(max_steps)),
                    mode=mode,
                    df=df,
                    bootstrap_returns=br,
                    gamma=gamma,
                    delta=delta,
                    device=device,
                )
                if z is None:
                    raise RuntimeError("torch sampling unavailable")
                logret = torch.cumsum(float(drift) + float(diffusion) * z, dim=1)
                clip_val = float(os.environ.get("MC_LOGRET_CLIP", "12.0"))
                if clip_val > 0:
                    logret = torch.clamp(logret, -clip_val, clip_val)
                prices_t = float(s0) * torch.exp(logret)
                prices = to_numpy(prices_t).astype(np.float64)
            except Exception as e:
                logger.warning(f"[MC] Torch path simulation failed, falling back to NumPy: {e}")
                use_torch = False

        if use_jax and not use_torch:
            try:
                key = jrand.PRNGKey(int(seed) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                key, z_j = self._sample_increments_jax(
                    key,
                    (int(n_paths), int(max_steps)),
                    mode=mode,
                    df=df,
                    gamma=gamma,
                    delta=delta,
                    bootstrap_returns=br,
                )

                if z_j is None:
                    use_jax = False
                else:
                    z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                    drift_f = jnp.asarray(drift, dtype=jnp.float32)  # type: ignore[attr-defined]
                    diffusion_f = jnp.asarray(diffusion, dtype=jnp.float32)  # type: ignore[attr-defined]
                    increments = drift_f + diffusion_f * z_j  # type: ignore[attr-defined]
                    def _row_cumsum(row):
                        def _scan(carry, x):
                            nxt = carry + x
                            return nxt, nxt
                        _, out = jax.lax.scan(_scan, 0.0, row)  # type: ignore[attr-defined]
                        return out
                    logret = jax.vmap(_row_cumsum)(increments)  # type: ignore[attr-defined]
                    prices_jnp = float(s0) * jnp.exp(logret)  # type: ignore[attr-defined]
                    prices = np.asarray(jax.device_get(prices_jnp), dtype=np.float64)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"[MC] JAX path simulation failed, falling back to NumPy: {e}")
                use_jax = False

        if not use_jax:
            rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
            z = self._sample_increments_np(
                rng,
                (int(n_paths), int(max_steps)),
                mode=mode,
                df=df,
                bootstrap_returns=br,
            )
            logret = np.cumsum(drift + diffusion * z, axis=1)
            prices = s0 * np.exp(logret)

        out = {}
        stats = {} if return_stats else None
        z_hat = None
        control = None
        if return_stats and diffusion > EPSILON:
            try:
                z_hat = self._infer_standardized_noise(prices, drift, diffusion)
                control = np.sum(z_hat, axis=1)
            except Exception:
                z_hat = None
                control = None
        for h in horizons:
            idx = int(h) - 1
            tp = prices[:, idx]
            gross = direction * (tp - s0) / s0 * float(leverage)
            net = gross - fee_roundtrip
            out[int(h)] = net.astype(np.float64)
            if return_stats:
                raw_mean = float(np.mean(net))
                if z_hat is None or control is None:
                    stats[int(h)] = {"raw_mean": raw_mean, "cv_mean": raw_mean, "c_opt": 0.0}
                else:
                    cv_mean, c_opt = self._control_variate_mean(net, control)
                    stats[int(h)] = {"raw_mean": raw_mean, "cv_mean": cv_mean, "c_opt": c_opt}
        if return_stats:
            return out, stats
        return out

    def _infer_standardized_noise(self, prices: np.ndarray, drift: float, diffusion: float) -> np.ndarray:
        prices_np = np.asarray(prices, dtype=np.float64)
        if prices_np.shape[1] < 2 or diffusion <= EPSILON:
            return np.zeros_like(prices_np[:, :-1], dtype=np.float64)
        log_prices = np.log(np.clip(prices_np, EPSILON, None))
        log_returns = np.diff(log_prices, axis=1)
        return (log_returns - drift) / diffusion

    def _control_variate_mean(self, values: np.ndarray, control: np.ndarray, expected_control: float = 0.0) -> tuple[float, float]:
        v = np.asarray(values, dtype=np.float64)
        c = np.asarray(control, dtype=np.float64)
        v_mean = float(np.mean(v))
        c_mean = float(np.mean(c))
        var_c = float(np.mean((c - c_mean) ** 2))
        if var_c < EPSILON:
            return v_mean, 0.0
        cov = float(np.mean((v - v_mean) * (c - c_mean)))
        c_opt = cov / var_c
        adj_mean = float(np.mean(v - c_opt * (c - expected_control)))
        return adj_mean, c_opt

    def _control_variate_price_stats(self, prices: np.ndarray, drift: float, diffusion: float) -> Dict[str, float]:
        prices_np = np.asarray(prices, dtype=np.float64)
        raw_mean = float(np.mean(prices_np[:, -1]))
        if diffusion <= EPSILON:
            return {"raw_mean": raw_mean, "cv_mean": raw_mean, "c_opt": 0.0}
        z_hat = self._infer_standardized_noise(prices_np, drift, diffusion)
        control = np.sum(z_hat, axis=1)
        cv_mean, c_opt = self._control_variate_mean(prices_np[:, -1], control)
        return {"raw_mean": raw_mean, "cv_mean": cv_mean, "c_opt": c_opt}

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from engines.cvar_methods import cvar_ensemble
from engines.mc.jax_backend import _JAX_OK, _jax_mc_device, jax, jnp, jrand


class MonteCarloFirstPassageMixin:
    def mc_first_passage_tp_sl(
        self,
        s0: float,
        tp_pct: float,
        sl_pct: float,
        mu: float,
        sigma: float,
        dt: float,
        max_steps: int,
        n_paths: int,
        cvar_alpha: float = 0.05,
        timeout_mode: str = "flat",
        seed: Optional[int] = None,
        side: str = "LONG",
    ) -> Dict[str, Any]:
        tp_pct = float(tp_pct)
        sl_pct = float(sl_pct)
        if tp_pct <= 0 or sl_pct <= 0 or sigma <= 0 or s0 <= 0:
            return {
                "event_p_tp": None,
                "event_p_sl": None,
                "event_p_timeout": None,
                "event_ev_r": None,
                "event_cvar_r": None,
                "event_t_median": None,
                "event_t_mean": None,
            }

        rng = np.random.default_rng(seed)
        max_steps = int(max(1, max_steps))
        # -----------------------------
        # Direction handling
        # LONG: 그대로, SHORT: log-return 반전
        # -----------------------------
        direction = 1.0
        if str(side).upper() == "SHORT":
            direction = -1.0

        drift = direction * (mu - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)

        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        br = getattr(self, "_bootstrap_returns", None)
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK

        prices_np: np.ndarray
        bridge_tp_np: Optional[np.ndarray] = None
        bridge_sl_np: Optional[np.ndarray] = None
        log_prices_j = None
        if use_jax:
            # ✅ GPU 우선: default backend (GPU/Metal) 사용
            force_cpu_dev = _jax_mc_device()
            try:
                if force_cpu_dev is None:
                    # GPU/Metal default backend 사용
                    key = jrand.PRNGKey(int(seed or 0) & 0xFFFFFFFF)  # type: ignore[attr-defined]
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
                        logret_j = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                        log_prices_j = direction * logret_j + math.log(s0)
                        prices_j = jnp.exp(log_prices_j)  # type: ignore[attr-defined]
                        prices_np = np.asarray(jax.device_get(prices_j), dtype=np.float64)  # type: ignore[attr-defined]
                else:
                    # CPU로 강제된 경우만 CPU 사용 (env JAX_MC_DEVICE=cpu)
                    with jax.default_device(force_cpu_dev):  # type: ignore[attr-defined]
                        key = jrand.PRNGKey(int(seed or 0) & 0xFFFFFFFF)  # type: ignore[attr-defined]
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
                            logret_j = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                            log_prices_j = direction * logret_j + math.log(s0)
                            prices_j = jnp.exp(log_prices_j)  # type: ignore[attr-defined]
                            prices_np = np.asarray(jax.device_get(prices_j), dtype=np.float64)  # type: ignore[attr-defined]
            except Exception:
                # Any JAX/XLA backend failure -> fall back to NumPy path simulation
                use_jax = False


        if not use_jax:
            z = self._sample_increments_np(rng, (n_paths, max_steps), mode=mode, df=df, bootstrap_returns=br)
            steps = drift + diffusion * z
            log_prices = np.cumsum(steps, axis=1) + math.log(s0)
            prices_np = np.exp(direction * log_prices)

        if str(side).upper() == "SHORT":
            tp_level = s0 * (1.0 - tp_pct)
            sl_level = s0 * (1.0 + sl_pct)
        else:
            tp_level = s0 * (1.0 + tp_pct)
            sl_level = s0 * (1.0 - sl_pct)

        # Brownian Bridge 보정: intra-step 배리어 터치 확률을 보완
        sigma_sq_dt = sigma * sigma * dt
        if sigma_sq_dt > 0.0:
            if use_jax and log_prices_j is not None:
                log_prices_full_j = jnp.concatenate(  # type: ignore[attr-defined]
                    (
                        jnp.full((n_paths, 1), math.log(s0), dtype=log_prices_j.dtype),  # type: ignore[attr-defined]
                        log_prices_j,
                    ),
                    axis=1,
                )
                prev_j = log_prices_full_j[:, :-1]
                nxt_j = log_prices_full_j[:, 1:]

                log_tp_j = jnp.array(math.log(tp_level), dtype=log_prices_j.dtype)  # type: ignore[attr-defined]
                below_tp_j = (prev_j < log_tp_j) & (nxt_j < log_tp_j)
                sigma_sq_dt_j = jnp.array(sigma_sq_dt, dtype=log_prices_j.dtype)  # type: ignore[attr-defined]
                prob_tp_j = jnp.where(
                    below_tp_j,
                    jnp.exp(-2.0 * (log_tp_j - prev_j) * (log_tp_j - nxt_j) / sigma_sq_dt_j),  # type: ignore[attr-defined]
                    0.0,
                )
                key, key_tp = jrand.split(key)  # type: ignore[attr-defined]
                bridge_tp_np = np.asarray(
                    jax.device_get(jrand.uniform(key_tp, prob_tp_j.shape) < prob_tp_j), dtype=bool  # type: ignore[attr-defined]
                )

                log_sl_j = jnp.array(math.log(sl_level), dtype=log_prices_j.dtype)  # type: ignore[attr-defined]
                above_sl_j = (prev_j > log_sl_j) & (nxt_j > log_sl_j)
                prob_sl_j = jnp.where(
                    above_sl_j,
                    jnp.exp(-2.0 * (prev_j - log_sl_j) * (nxt_j - log_sl_j) / sigma_sq_dt_j),  # type: ignore[attr-defined]
                    0.0,
                )
                key, key_sl = jrand.split(key)  # type: ignore[attr-defined]
                bridge_sl_np = np.asarray(
                    jax.device_get(jrand.uniform(key_sl, prob_sl_j.shape) < prob_sl_j), dtype=bool  # type: ignore[attr-defined]
                )

            if bridge_tp_np is None or bridge_sl_np is None:
                log_prices = np.log(prices_np)
                log_prices_full = np.concatenate(
                    (np.full((n_paths, 1), math.log(s0), dtype=np.float64), log_prices), axis=1
                )

                prev = log_prices_full[:, :-1]
                nxt = log_prices_full[:, 1:]

                log_tp = math.log(tp_level)
                below_tp = (prev < log_tp) & (nxt < log_tp)
                prob_tp = np.zeros_like(prev, dtype=np.float64)
                prob_tp[below_tp] = np.exp(-2.0 * (log_tp - prev[below_tp]) * (log_tp - nxt[below_tp]) / sigma_sq_dt)
                bridge_tp_np = rng.random(prob_tp.shape) < prob_tp

                log_sl = math.log(sl_level)
                above_sl = (prev > log_sl) & (nxt > log_sl)
                prob_sl = np.zeros_like(prev, dtype=np.float64)
                prob_sl[above_sl] = np.exp(-2.0 * (prev[above_sl] - log_sl) * (nxt[above_sl] - log_sl) / sigma_sq_dt)
                bridge_sl_np = rng.random(prob_sl.shape) < prob_sl

            hit_tp = (prices_np >= tp_level) | bridge_tp_np
            hit_sl = (prices_np <= sl_level) | bridge_sl_np
        else:
            hit_tp = prices_np >= tp_level
            hit_sl = prices_np <= sl_level

        tp_hit_idx = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1) + 1, max_steps + 1)
        sl_hit_idx = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1) + 1, max_steps + 1)

        first_hit_idx = np.minimum(tp_hit_idx, sl_hit_idx)
        hit_type = np.full(n_paths, "timeout", dtype=object)
        hit_type[(tp_hit_idx < sl_hit_idx) & (tp_hit_idx <= max_steps)] = "tp"
        hit_type[(sl_hit_idx < tp_hit_idx) & (sl_hit_idx <= max_steps)] = "sl"

        tp_R = float(tp_pct / sl_pct)
        returns_r = np.zeros(n_paths, dtype=np.float64)
        returns_r[hit_type == "tp"] = tp_R
        returns_r[hit_type == "sl"] = -1.0
        if timeout_mode in ("mark_to_market", "mtm"):
            # mark-to-market: use end-of-horizon return
            end_prices = prices_np[:, -1]
            returns_r[hit_type == "timeout"] = (end_prices[hit_type == "timeout"] - s0) / (s0 * sl_pct)

        event_p_tp = float(np.mean(hit_type == "tp"))
        event_p_sl = float(np.mean(hit_type == "sl"))
        event_p_timeout = float(np.mean(hit_type == "timeout"))
        event_ev_r = float(np.mean(returns_r))
        event_cvar_r = float(cvar_ensemble(returns_r, alpha=cvar_alpha))

        hit_mask = (hit_type == "tp") | (hit_type == "sl")
        hit_times = first_hit_idx[hit_mask]
        event_t_median = float(np.median(hit_times)) if hit_times.size > 0 else None
        event_t_mean = float(np.mean(hit_times)) if hit_times.size > 0 else None

        # sanity check
        prob_sum = event_p_tp + event_p_sl + event_p_timeout
        if abs(prob_sum - 1.0) > 1e-3:
            # normalize softly
            event_p_tp /= prob_sum
            event_p_sl /= prob_sum
            event_p_timeout = max(0.0, 1.0 - event_p_tp - event_p_sl)

        return {
            "event_p_tp": event_p_tp,
            "event_p_sl": event_p_sl,
            "event_p_timeout": event_p_timeout,
            "event_ev_r": event_ev_r,
            "event_cvar_r": event_cvar_r,
            "event_t_median": event_t_median,
            "event_t_mean": event_t_mean,
        }

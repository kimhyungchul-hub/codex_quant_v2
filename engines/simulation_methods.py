from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from engines.cvar_methods import _cvar_jnp

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    from jax import lax, random  # type: ignore
except Exception:  # pragma: no cover
    jax = None
    jnp = None
    lax = None
    random = None

_JAX_OK = jax is not None


def _sample_noise(key, shape, dist: str = "gaussian", df: float = 6.0, boot=None):
    if jnp is None or random is None:  # pragma: no cover
        raise RuntimeError("JAX is not available")
    if dist == "bootstrap" and boot is not None:
        boot_jnp = jnp.asarray(boot, dtype=jnp.float32)  # type: ignore[attr-defined]
        boot_size = int(boot_jnp.shape[0])
        if boot_size > 16:
            idx = random.randint(key, shape, 0, boot_size)  # type: ignore[attr-defined]
            return boot_jnp[idx]
    if dist == "student_t":
        k1, k2 = random.split(key)  # type: ignore[attr-defined]
        z = random.normal(k1, shape)  # type: ignore[attr-defined]
        u = 2.0 * random.gamma(k2, df / 2.0, shape)  # type: ignore[attr-defined]
        return z / jnp.sqrt(u / df)  # type: ignore[attr-defined]
    return random.normal(key, shape)  # type: ignore[attr-defined]


def _mc_first_passage_tp_sl_jax_core(
    key,
    s0: float,
    tp_pct: float,
    sl_pct: float,
    drift: float,
    vol: float,
    max_steps: int,
    n_paths: int,
    dist: str,
    df: float,
    boot_jnp,
    cvar_alpha: float,
):
    if jnp is None or lax is None:  # pragma: no cover
        raise RuntimeError("JAX is not available")
    eps = _sample_noise(key, (n_paths, max_steps), dist=dist, df=df, boot=boot_jnp)

    log_inc = drift + vol * eps
    tp_price = s0 * (1.0 + tp_pct)
    sl_price = s0 * (1.0 - sl_pct)

    alive = jnp.ones(n_paths, dtype=bool)  # type: ignore[attr-defined]
    hit_tp = jnp.zeros(n_paths, dtype=bool)  # type: ignore[attr-defined]
    hit_sl = jnp.zeros(n_paths, dtype=bool)  # type: ignore[attr-defined]
    t_hit = -jnp.ones(n_paths, dtype=jnp.int32)  # type: ignore[attr-defined]
    logp = jnp.zeros(n_paths)  # type: ignore[attr-defined]

    def step(carry, t):
        logp, alive, hit_tp, hit_sl, t_hit = carry
        logp2 = logp + log_inc[:, t]
        price = s0 * jnp.exp(logp2)  # type: ignore[attr-defined]
        tp_now = alive & (price >= tp_price)
        sl_now = alive & (price <= sl_price)
        hit = tp_now | sl_now
        t_hit = jnp.where(hit & (t_hit < 0), t, t_hit)  # type: ignore[attr-defined]
        hit_tp = hit_tp | tp_now
        hit_sl = hit_sl | sl_now
        alive = alive & (~hit)
        return (logp2, alive, hit_tp, hit_sl, t_hit), None

    (logp, alive, hit_tp, hit_sl, t_hit), _ = lax.scan(  # type: ignore[attr-defined]
        step,
        (logp, alive, hit_tp, hit_sl, t_hit),
        jnp.arange(max_steps),  # type: ignore[attr-defined]
    )

    p_tp = jnp.mean(hit_tp)  # type: ignore[attr-defined]
    p_sl = jnp.mean(hit_sl)  # type: ignore[attr-defined]
    p_to = jnp.mean(alive)  # type: ignore[attr-defined]

    r_tp = tp_pct / sl_pct
    r = jnp.where(hit_tp, r_tp, jnp.where(hit_sl, -1.0, 0.0))  # type: ignore[attr-defined]

    ev_r = jnp.mean(r)  # type: ignore[attr-defined]
    cvar_r = _cvar_jnp(r, cvar_alpha)

    t_vals = jnp.where(t_hit >= 0, t_hit.astype(jnp.float32), jnp.nan)  # type: ignore[attr-defined]

    return p_tp, p_sl, p_to, ev_r, cvar_r, t_vals


if _JAX_OK:
    _mc_first_passage_tp_sl_jax_core_jit = jax.jit(  # type: ignore[attr-defined]
        _mc_first_passage_tp_sl_jax_core,
        static_argnames=("dist", "max_steps", "n_paths"),
    )
else:  # pragma: no cover
    _mc_first_passage_tp_sl_jax_core_jit = None


def _generate_and_check_paths_jax_core(
    key,
    s0: float,
    tp_pct: float,
    sl_pct: float,
    drift: float,
    vol: float,
    max_steps: int,
    n_paths: int,
    dist: str,
    df: float,
    boot_jnp,
    cvar_alpha: float,
):
    if jnp is None or lax is None:  # pragma: no cover
        raise RuntimeError("JAX is not available")
    eps = _sample_noise(key, (n_paths, max_steps), dist=dist, df=df, boot=boot_jnp)

    log_inc = drift + vol * eps
    tp_price = s0 * (1.0 + tp_pct)
    sl_price = s0 * (1.0 - sl_pct)

    alive = jnp.ones(n_paths, dtype=bool)  # type: ignore[attr-defined]
    hit_tp = jnp.zeros(n_paths, dtype=bool)  # type: ignore[attr-defined]
    hit_sl = jnp.zeros(n_paths, dtype=bool)  # type: ignore[attr-defined]
    t_hit = -jnp.ones(n_paths, dtype=jnp.int32)  # type: ignore[attr-defined]
    logp = jnp.zeros(n_paths)  # type: ignore[attr-defined]

    def step(carry, t):
        logp, alive, hit_tp, hit_sl, t_hit = carry
        logp2 = logp + log_inc[:, t]
        price = s0 * jnp.exp(logp2)  # type: ignore[attr-defined]
        tp_now = alive & (price >= tp_price)
        sl_now = alive & (price <= sl_price)
        hit = tp_now | sl_now
        t_hit = jnp.where(hit & (t_hit < 0), t, t_hit)  # type: ignore[attr-defined]
        hit_tp = hit_tp | tp_now
        hit_sl = hit_sl | sl_now
        alive = alive & (~hit)
        return (logp2, alive, hit_tp, hit_sl, t_hit), None

    (logp, alive, hit_tp, hit_sl, t_hit), _ = lax.scan(  # type: ignore[attr-defined]
        step,
        (logp, alive, hit_tp, hit_sl, t_hit),
        jnp.arange(max_steps),  # type: ignore[attr-defined]
    )

    p_tp = jnp.mean(hit_tp)  # type: ignore[attr-defined]
    p_sl = jnp.mean(hit_sl)  # type: ignore[attr-defined]
    p_to = jnp.mean(alive)  # type: ignore[attr-defined]

    r_tp = tp_pct / sl_pct
    r = jnp.where(hit_tp, r_tp, jnp.where(hit_sl, -1.0, 0.0))  # type: ignore[attr-defined]

    ev_r = jnp.mean(r)  # type: ignore[attr-defined]
    cvar_r = _cvar_jnp(r, cvar_alpha)

    t_vals = jnp.where(t_hit >= 0, t_hit.astype(jnp.float32), jnp.nan)  # type: ignore[attr-defined]

    return p_tp, p_sl, p_to, ev_r, cvar_r, t_vals


if _JAX_OK:
    _generate_and_check_paths_jax_core_jit = jax.jit(  # type: ignore[attr-defined]
        _generate_and_check_paths_jax_core,
        static_argnames=("dist", "max_steps", "n_paths"),
    )
else:  # pragma: no cover
    _generate_and_check_paths_jax_core_jit = None


def mc_first_passage_tp_sl_jax(
    s0: float,
    tp_pct: float,
    sl_pct: float,
    mu: float,
    sigma: float,
    dt: float,
    max_steps: int,
    n_paths: int,
    seed: int,
    dist: str = "gaussian",
    df: float = 6.0,
    boot_rets: np.ndarray | None = None,
    cvar_alpha: float = 0.05,
) -> Dict[str, Any]:
    if jax is None or jnp is None or random is None or tp_pct <= 0 or sl_pct <= 0 or sigma <= 0:
        return {}

    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    key = random.PRNGKey(seed & 0xFFFFFFFF)  # type: ignore[attr-defined]
    boot_jnp = None if boot_rets is None else jnp.asarray(boot_rets, dtype=jnp.float32)  # type: ignore[attr-defined]

    if _mc_first_passage_tp_sl_jax_core_jit is not None:
        p_tp, p_sl, p_to, ev_r, cvar_r, t_vals = _mc_first_passage_tp_sl_jax_core_jit(
            key, s0, tp_pct, sl_pct, drift, vol, max_steps, n_paths, dist, df, boot_jnp, cvar_alpha
        )
    else:
        p_tp, p_sl, p_to, ev_r, cvar_r, t_vals = _mc_first_passage_tp_sl_jax_core(
            key, s0, tp_pct, sl_pct, drift, vol, max_steps, n_paths, dist, df, boot_jnp, cvar_alpha
        )

    p_tp_f = float(p_tp)
    p_sl_f = float(p_sl)
    p_to_f = float(p_to)
    prob_sum = p_tp_f + p_sl_f + p_to_f
    if abs(prob_sum - 1.0) > 1e-3 and prob_sum > 0:
        p_tp_f /= prob_sum
        p_sl_f /= prob_sum
        p_to_f = max(0.0, 1.0 - p_tp_f - p_sl_f)

    t_median_raw = jnp.nanmedian(t_vals)  # type: ignore[attr-defined]
    t_mean_raw = jnp.nanmean(t_vals)  # type: ignore[attr-defined]
    t_median = float(np.asarray(t_median_raw).item())
    t_mean = float(np.asarray(t_mean_raw).item())
    if not math.isfinite(t_median):
        t_median = None
    if not math.isfinite(t_mean):
        t_mean = None

    ev_r_host = float(np.asarray(ev_r).item())
    cvar_r_host = float(np.asarray(cvar_r).item())

    return {
        "event_p_tp": p_tp_f,
        "event_p_sl": p_sl_f,
        "event_p_timeout": p_to_f,
        "event_ev_r": ev_r_host,
        "event_cvar_r": cvar_r_host,
        "event_t_median": t_median,
        "event_t_mean": t_mean,
    }


_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from engines.mc.monte_carlo_engine import MonteCarloEngine

        _ENGINE = MonteCarloEngine()
    return _ENGINE


def simulate_paths_price(*args, **kwargs):
    return _get_engine().simulate_paths_price(*args, **kwargs)


def simulate_paths_netpnl(*args, **kwargs):
    return _get_engine().simulate_paths_netpnl(*args, **kwargs)

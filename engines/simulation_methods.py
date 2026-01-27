from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np


def _cvar_empirical(pnl: np.ndarray, alpha: float = 0.05) -> float:
    x = np.sort(np.asarray(pnl, dtype=np.float64))
    k = max(1, int(alpha * len(x)))
    return float(x[:k].mean())


def _sample_noise(shape, dist: str = "gaussian", df: float = 6.0, boot=None):
    if dist == "bootstrap" and boot is not None:
        boot_np = np.asarray(boot, dtype=np.float32)
        return np.random.choice(boot_np, size=shape)
    if dist == "student_t":
        return np.random.standard_t(df, size=shape)
    return np.random.standard_normal(size=shape)


def mc_first_passage_tp_sl_numpy(
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
    if tp_pct <= 0 or sl_pct <= 0 or sigma <= 0:
        return {}

    np.random.seed(seed & 0xFFFFFFFF)
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    eps = _sample_noise((n_paths, max_steps), dist=dist, df=df, boot=boot_rets)
    log_inc = drift + vol * eps
    log_paths = np.cumsum(log_inc, axis=1)
    prices = s0 * np.exp(log_paths)

    tp_price = s0 * (1.0 + tp_pct)
    sl_price = s0 * (1.0 - sl_pct)

    hit_tp = np.any(prices >= tp_price, axis=1)
    hit_sl = np.any(prices <= sl_price, axis=1)
    
    # First hit logic
    t_hit = np.full(n_paths, -1, dtype=np.int32)
    for t in range(max_steps):
        # Already hit check
        active = t_hit < 0
        tp_now = active & (prices[:, t] >= tp_price)
        sl_now = active & (prices[:, t] <= sl_price)
        hit = tp_now | sl_now
        t_hit[hit] = t

    p_tp = np.mean(hit_tp)
    p_sl = np.mean(hit_sl)
    p_to = 1.0 - np.mean(hit_tp | hit_sl)

    r_tp = tp_pct / sl_pct
    # ROE-like ratio calculation consistent with legacy code
    r = np.zeros(n_paths)
    r[hit_tp] = r_tp
    r[hit_sl & ~hit_tp] = -1.0
    
    ev_r = np.mean(r)
    cvar_r = _cvar_empirical(r, cvar_alpha)

    t_vals = t_hit[t_hit >= 0].astype(np.float32)
    t_median = float(np.median(t_vals)) if t_vals.size > 0 else None
    t_mean = float(np.mean(t_vals)) if t_vals.size > 0 else None

    return {
        "event_p_tp": float(p_tp),
        "event_p_sl": float(p_sl),
        "event_p_timeout": float(p_to),
        "event_ev_r": float(ev_r),
        "event_cvar_r": float(cvar_r),
        "event_t_median": t_median,
        "event_t_mean": t_mean,
    }

# Shim to match legacy naming if needed, though most call mc_first_passage_tp_sl_jax
def mc_first_passage_tp_sl_jax(*args, **kwargs):
    return mc_first_passage_tp_sl_numpy(*args, **kwargs)

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

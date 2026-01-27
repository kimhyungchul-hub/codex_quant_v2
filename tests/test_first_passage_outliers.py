# Outlier analysis for Brownian Bridge results
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engines.mc.first_passage import MonteCarloFirstPassageMixin

class Dummy(MonteCarloFirstPassageMixin):
    def __init__(self):
        self._tail_mode = 'gaussian'
        self.default_tail_mode = 'gaussian'
        self._student_t_df = 4
        self.default_student_t_df = 4
        self._bootstrap_returns = None
        self._use_jax = False

    def _sample_increments_np(self, rng, shape, mode, df, bootstrap_returns):
        return rng.standard_normal(shape)


def analyze(seed=42, n_paths=50000):
    d = Dummy()
    rng = np.random.default_rng(seed)

    s0 = 100.0
    tp_pct = 0.02
    sl_pct = 0.02
    mu = 0.0
    sigma = 0.2
    dt = 1.0
    max_steps = 10
    side = 'LONG'

    direction = 1.0
    drift = direction * (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)

    z = d._sample_increments_np(rng, (n_paths, max_steps), mode='gaussian', df=4, bootstrap_returns=None)
    steps = drift + diffusion * z
    log_prices = np.cumsum(steps, axis=1) + math.log(s0)
    prices = np.exp(direction * log_prices)

    tp_level = s0 * (1.0 + tp_pct)
    sl_level = s0 * (1.0 - sl_pct)

    sigma_sq_dt = sigma * sigma * dt
    log_prices_full = np.concatenate((np.full((n_paths, 1), math.log(s0)), np.log(prices)), axis=1)
    prev = log_prices_full[:, :-1]
    nxt = log_prices_full[:, 1:]

    log_tp = math.log(tp_level)
    below_tp = (prev < log_tp) & (nxt < log_tp)
    prob_tp = np.zeros_like(prev, dtype=np.float64)
    prob_tp[below_tp] = np.exp(-2.0 * (log_tp - prev[below_tp]) * (log_tp - nxt[below_tp]) / sigma_sq_dt)
    bridge_tp = rng.random(prob_tp.shape) < prob_tp

    log_sl = math.log(sl_level)
    above_sl = (prev > log_sl) & (nxt > log_sl)
    prob_sl = np.zeros_like(prev, dtype=np.float64)
    prob_sl[above_sl] = np.exp(-2.0 * (prev[above_sl] - log_sl) * (nxt[above_sl] - log_sl) / sigma_sq_dt)
    bridge_sl = rng.random(prob_sl.shape) < prob_sl

    hit_tp = (prices >= tp_level) | bridge_tp
    hit_sl = (prices <= sl_level) | bridge_sl

    tp_hit_idx = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1) + 1, max_steps + 1)
    sl_hit_idx = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1) + 1, max_steps + 1)

    first_hit_idx = np.minimum(tp_hit_idx, sl_hit_idx)
    hit_type = np.full(n_paths, 'timeout', dtype=object)
    hit_type[(tp_hit_idx < sl_hit_idx) & (tp_hit_idx <= max_steps)] = 'tp'
    hit_type[(sl_hit_idx < tp_hit_idx) & (sl_hit_idx <= max_steps)] = 'sl'

    tp_R = float(tp_pct / sl_pct)
    returns_r = np.zeros(n_paths, dtype=np.float64)
    returns_r[hit_type == 'tp'] = tp_R
    returns_r[hit_type == 'sl'] = -1.0

    # Stats
    stats = {
        'n_paths': n_paths,
        'p_tp': float(np.mean(hit_type == 'tp')),
        'p_sl': float(np.mean(hit_type == 'sl')),
        'p_timeout': float(np.mean(hit_type == 'timeout')),
        'returns_mean': float(np.mean(returns_r)),
        'returns_std': float(np.std(returns_r)),
        'returns_min': float(np.min(returns_r)),
        'returns_max': float(np.max(returns_r)),
        'returns_percentiles': np.percentile(returns_r, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist(),
        'hit_times_percentiles': np.percentile(first_hit_idx[first_hit_idx<=max_steps], [25,50,75,90,95]).tolist() if np.any(first_hit_idx<=max_steps) else None,
    }
    return stats

if __name__ == '__main__':
    s = analyze(seed=42, n_paths=50000)
    for k,v in s.items():
        print(f'{k}: {v}')

# Compare NumPy vs JAX implementations of mc_first_passage_tp_sl
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engines.mc.first_passage import MonteCarloFirstPassageMixin

try:
    import jax
    from jax import random as jrand
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False

class DummyBoth(MonteCarloFirstPassageMixin):
    def __init__(self, use_jax=False):
        self._tail_mode = 'gaussian'
        self.default_tail_mode = 'gaussian'
        self._student_t_df = 4
        self.default_student_t_df = 4
        self._bootstrap_returns = None
        self._use_jax = use_jax and JAX_AVAILABLE

    def _sample_increments_np(self, rng, shape, mode, df, bootstrap_returns):
        return rng.standard_normal(shape)

    def _sample_increments_jax(self, key, shape, mode, df, bootstrap_returns):
        # simple normal increments compatible with NumPy sampling
        key, sub = jrand.split(key)
        z = jrand.normal(sub, shape)
        return key, z


def run_test(use_jax: bool):
    d = DummyBoth(use_jax=use_jax)
    t0 = time.perf_counter()
    res = d.mc_first_passage_tp_sl(
        s0=100.0,
        tp_pct=0.02,
        sl_pct=0.02,
        mu=0.0,
        sigma=0.2,
        dt=1.0,
        max_steps=10,
        n_paths=20000,
        seed=42,
        side='LONG',
    )
    t1 = time.perf_counter()
    return res, t1 - t0

if __name__ == '__main__':
    print('JAX available:', JAX_AVAILABLE)
    numpy_res, numpy_time = run_test(use_jax=False)
    print('NumPy result:', numpy_res)
    print('NumPy time: %.3fs' % numpy_time)
    if JAX_AVAILABLE:
        jax_res, jax_time = run_test(use_jax=True)
        print('JAX result:', jax_res)
        print('JAX time: %.3fs' % jax_time)
        # compare keys
        for k in ['event_p_tp', 'event_p_sl', 'event_p_timeout', 'event_ev_r']:
            a = numpy_res[k]
            b = jax_res[k]
            print(f' delta {k}: {b - a}')
    else:
        print('Skipping JAX run (not available)')

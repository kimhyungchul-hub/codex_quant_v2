# Quick test for mc_first_passage_tp_sl with Brownian Bridge correction
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from engines.mc.first_passage import MonteCarloFirstPassageMixin

class Dummy(MonteCarloFirstPassageMixin):
    def __init__(self):
        self._tail_mode = 'gaussian'
        self.default_tail_mode = 'gaussian'
        self._student_t_df = 4
        self.default_student_t_df = 4
        self._bootstrap_returns = None
        # force numpy path
        self._use_jax = False

    def _sample_increments_np(self, rng, shape, mode, df, bootstrap_returns):
        # return standard normal increments
        return rng.standard_normal(shape)

if __name__ == '__main__':
    d = Dummy()
    res = d.mc_first_passage_tp_sl(
        s0=100.0,
        tp_pct=0.02,
        sl_pct=0.02,
        mu=0.0,
        sigma=0.2,
        dt=1.0,
        max_steps=10,
        n_paths=10000,
        seed=42,
        side='LONG',
    )
    print('Result:')
    for k, v in res.items():
        print(f'  {k}: {v}')

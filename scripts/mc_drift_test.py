import os
import numpy as np
from engines.mc.path_simulation import MonteCarloPathSimulationMixin


class DummyMC(MonteCarloPathSimulationMixin):
    def __init__(self, use_jax: bool = False):
        # defaults used by mixin
        self.default_tail_mode = "normal"
        self.default_student_t_df = 5.0
        self._tail_mode = getattr(self, "_tail_mode", self.default_tail_mode)
        self._student_t_df = getattr(self, "_student_t_df", self.default_student_t_df)
        self._bootstrap_returns = None
        self._use_jax = use_jax

    def _sample_increments_np(self, rng, shape, mode="normal", df=5.0, bootstrap_returns=None):
        n_paths, n_steps = shape
        if mode == "bootstrap" and bootstrap_returns is not None and len(bootstrap_returns) > 0:
            idxs = rng.integers(0, len(bootstrap_returns), size=(n_paths, n_steps))
            return bootstrap_returns[idxs]
        if mode == "student_t" and df > 0:
            return rng.standard_t(df, size=(n_paths, n_steps))
        return rng.standard_normal(size=(n_paths, n_steps))


def run_test(mode, seed=42, s0=100.0, mu=0.05, sigma=0.2, n_paths=20000, n_steps=1, dt=1.0, use_jax=False):
    mc = DummyMC(use_jax=use_jax)
    mc._tail_mode = mode
    mc._student_t_df = 5.0
    paths = mc.simulate_paths_price(
        seed=seed,
        s0=s0,
        mu=mu,
        sigma=sigma,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        return_jax=False,
    )
    # empirical log return per dt
    logrets = np.log(paths[:, -1] / s0)
    empirical_mean = float(np.mean(logrets)) / float(dt)
    empirical_std = float(np.std(logrets)) / float(np.sqrt(dt))
    if mode in ("student_t", "bootstrap"):
        target = float(mu)
    else:
        target = float(mu) - 0.5 * float(sigma) * float(sigma)

    print(f"mode={mode} n_paths={n_paths} n_steps={n_steps} dt={dt} use_jax={use_jax}")
    print(f"empirical_mean_logret_per_dt={empirical_mean:.8f}")
    print(f"empirical_std_logret_per_sqrt_dt={empirical_std:.8f}")
    print(f"target_log_drift_per_dt={target:.8f}")
    print(f"diff = empirical - target = {empirical_mean - target:.8e}")
    print("")


if __name__ == '__main__':
    np.random.seed(0)
    use_jax_env = os.environ.get("MC_USE_JAX", "0") not in ("0", "false", "False", "")
    run_test("normal", use_jax=use_jax_env)
    run_test("student_t", use_jax=use_jax_env)

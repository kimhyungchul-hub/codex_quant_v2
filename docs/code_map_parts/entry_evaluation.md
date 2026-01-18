# Entry Evaluation — `engines/mc/entry_evaluation.py`

Purpose: compute per-horizon EV, p(pos), CVaR, Kelly sizing and diagnostic metadata used by decision mixins and the dashboard.

Key functions

- `evaluate_entry_metrics(symbol, params, mc_params, policy_params, n_paths, rng=None, jax_enabled=True) -> dict`
  - Inputs (typical):
    - `symbol` (str) — symbol id
    - `params` (dict / dataclass) — contains current price `s0`, drift `mu`, volatility `sigma`, signal strength, and other per-symbol settings
    - `mc_params` (`MCParams`) — MC-specific configuration (horizons, dt, fee model)
    - `policy_params` — horizon priors / policy mix weights
    - `n_paths` (int) — number of Monte-Carlo paths
  - Returns: `res` dict with keys:
    - `can_enter` (bool)
    - `ev` (float) -- net EV after costs
    - `ev_raw` (float) -- pre-cost EV
    - `win` (float)
    - `cvar` (float)
    - `kelly` (float)
    - `size_frac` (float)
    - `direction` (int)
    - `best_h` (int)
    - `reason` (str)
    - `meta_detail` (dict) -- diagnostic dict with per-horizon arrays and event metrics

- `evaluate_entry_metrics_batch(tasks: List[Task], global_jax=True) -> List[dict]`
  - Vectorized evaluation: when `global_jax` is available it attempts to batch across symbols using `simulate_paths_price_batch()` and `summarize_gbm_horizons_jax()` for performance; otherwise falls back to looping over `evaluate_entry_metrics`.
  - Each returned element mirrors the `res` contract above plus an additional `perf` timing dict.

Important metadata (`meta_detail`) keys commonly consumed elsewhere

- `policy_w_h` -- final horizon weights used to combine horizon-level EVs.
- `policy_ev_by_h_long` / `policy_ev_by_h_short` -- EV per horizon, split by direction.
- `policy_p_pos_by_h` -- estimated probability of positive outcome per horizon.
- `event_p_tp`, `event_p_sl`, `event_p_timeout` -- first-passage event probabilities.
- `event_ev_r`, `event_cvar_r` -- event-based EV and CVaR ratios.
- `event_t_median`, `event_t_mean` -- time-to-event statistics.

Notes

- Many internal helpers attempt to keep the shapes constant between JAX/NumPy paths; tests should validate both code paths.
- Consider adding a typed dataclass for the `res` contract to avoid fragile dict-key coupling.

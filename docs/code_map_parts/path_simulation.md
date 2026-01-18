# Path Simulation — `engines/mc/path_simulation.py`

Purpose: generate synthetic price paths (GBM / return increments) for MC evaluation and convert them to net P&L series.

Key functions

- `simulate_paths_price(mu, sigma, s0, n_paths, n_steps, dt, rng=None, jax=False) -> ndarray`
  - Returns: `(n_paths, n_steps+1)` array of simulated prices (including s0 at index 0).
  - JAX path: returns a `jnp.ndarray` with the same shape; seeding and device semantics follow JAX PRNG conventions.

- `simulate_paths_price_batch(symbol_params_batch, n_paths, n_steps, dt, jax=True) -> ndarray`
  - Input: batch of symbol parameter tuples/dicts (mu, sigma, s0, etc.).
  - Returns: `(num_symbols, n_paths, n_steps+1)` array for vectorized batch processing.

- `simulate_paths_netpnl(paths, horizons, px_to_pnl, fee_model, exec_model) -> dict{horizon: net_pnl_array}`
  - Converts price paths into net P&L applying `px_to_pnl` mapping (directional sign, notional conversion) and cost models.

Notes

- Keep NumPy/JAX shape contracts stable across implementations to simplify downstream aggregation.
- Large-batch generation benefits greatly from JAX vmap/pmap; ensure memory usage constraints are considered when choosing `n_paths` and `num_symbols`.

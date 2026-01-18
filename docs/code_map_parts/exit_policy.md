# Exit Policy — `engines/mc/exit_policy.py`

Purpose: simulate and summarize exit-policy behavior across horizons. Used by evaluators to adjust EVs and by backtests to measure performance.

Key functions

- `compute_exit_policy_metrics(paths_or_summaries, policy_params, fee_model, pmaker=None) -> dict`
  - Inputs:
    - `paths_or_summaries`: either raw simulated paths or summarized per-path net-PnL arrays.
    - `policy_params`: policy configuration (timeouts, trailing stops, dynamic rules)
    - `fee_model`: execution fee/slippage model
    - `pmaker`: optional PMAKER survival/slippage model to adjust execution mixes
  - Returns:
    - `per_horizon_netpnl` (dict): arrays or statistics per horizon
    - `policy_ev` (array): EV per horizon under the exit policy
    - `policy_cvar` (array): CVaR per horizon
    - `diagnostics` (dict): extra arrays used for debugging and dashboarding

Notes

- When `pmaker` (ML survival model) is provided, execution costs and survival probabilities are integrated into the roll-forward simulation.
- The function exposes hooks where specialized execution-cost models can be injected (maker/taker mix, latencies).

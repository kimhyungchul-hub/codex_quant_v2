# Decision / Sizing — `engines/mc/decision.py`

Purpose: convert evaluator outputs (EV, CVaR, Kelly) into executable recommendations (action, size, leverage). Integrates with risk manager.

Key functions/classes

- `MonteCarloDecisionMixin.decide(res: dict, ctx: dict, risk_manager) -> dict`
  - Inputs:
    - `res` — evaluator result (see `evaluate_entry_metrics` contract)
    - `ctx` — per-symbol runtime context (market state, positions)
    - `risk_manager` — `RiskManager` instance to enforce limits
  - Returns: sanitized decision dict with fields:
    - `action` (str) e.g., `ENTER_LONG`, `ENTER_SHORT`, `WAIT`
    - `suggested_size` (float) absolute or fraction depending on API
    - `size_frac` (float) final fraction after risk adjustments
    - `leverage` (float) suggested leverage
    - `reason` (str)
    - `meta` (dict) carry-forward of evaluator `meta_detail`

- `decide_batch()`
  - Batch wrapper that can accept multiple `res` objects and produce a list of decisions. May use GPU-accelerated sizing routines when available.

Sizing notes

- Kelly-based sizing: uses `kelly` from `res` to compute base fraction. Additional shrinkage factors and risk manager caps applied afterwards.
- Fallbacks: if Kelly is NaN or extreme, code uses conservative floor values from config.

Integration points

- Output is consumed by `LiveOrchestrator` and `PaperBroker` (paper mode) after `RiskManager` adjustment.

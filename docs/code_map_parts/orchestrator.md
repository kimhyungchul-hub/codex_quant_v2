# Orchestrator — `core/orchestrator.py`

Purpose: central runtime loop that composes per-symbol contexts, schedules compute and UI loops, and integrates engine decisions with risk and execution layers.

Key class: `LiveOrchestrator`

Primary responsibilities

- Build per-symbol `ctx` objects containing market snapshots, signals, and position state.
- Start and run two separated loops:
  - `decision_worker_loop()` — heavy compute worker(s) that call `EngineHub.decide()` / batch evaluators.
  - `decision_loop()` — UI loop that assembles `_last_rows` and calls `DashboardServer.broadcast()`.
- Maintain `_decision_cache` and `_rows_snapshot_cached()` for efficient UI updates.

Important methods (developer-facing)

- `_build_ctx_for_symbol(symbol) -> dict` — gather latest market data, signals, and position into a context used by engines.
- `decision_worker_loop()` — picks symbols, schedules calls to `EngineHub.decide_batch()`, writes to `_decision_cache`.
- `decision_loop()` — periodically reads `_decision_cache`, builds rows (`_row()`), and triggers dashboard updates.

Integration

- Calls `DataManager` to fetch market data and `RiskManager` to validate/adjust suggested sizes.
- In paper mode, calls `PaperBroker` to simulate fills and update paper positions.

Notes

- The separation of compute/UI loops intentionally isolates heavy JAX/NumPy compute from frequent UI updates to keep dashboard latency low.
- When modifying row shape or dashboard fields, update `dashboard_v2.html` `data-k` mappings accordingly.

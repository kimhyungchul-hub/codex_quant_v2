# EngineHub — `engines/engine_hub.py`

Purpose: discover, register, and orchestrate multiple engines (MC, dummy, others), sanitize their outputs and provide a unified API to the orchestrator.

Key functions

- `EngineHub.decide(ctx) -> dict`
  - Calls each registered engine's `decide(ctx)` or `decide_async` and collects results.
  - Sanitizes outputs: converts JAX arrays to host-friendly types, ensures `ev`/`ev_raw` numeric types, and fills missing fields with defaults.
  - Aggregates engine outputs if multiple engines are present (weighting logic configurable).

- `EngineHub.decide_batch(ctx_list) -> List[dict]`
  - Vectorized entry point: calls `decide` across a list of contexts. When possible, forwards to global-batch evaluators in engines (e.g., `MonteCarloEngine.decide_batch`) to benefit from JAX acceleration.

Sanitization details

- JAX -> NumPy conversion: `_sanitize_value(v)` helper ensures `jnp.ndarray` -> `np.array` -> Python scalars where appropriate.
- Ensures `meta` keys exist (or sets empty dict) so downstream consumers don't KeyError.

Notes

- Keep engine output schema stable. If adding keys to engine `res` dicts, also update `LiveOrchestrator._row()` and dashboard mappings.
- The hub tolerates engines returning `None` and logs/filters accordingly.

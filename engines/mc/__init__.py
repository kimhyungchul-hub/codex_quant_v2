from __future__ import annotations

__all__ = ["MonteCarloEngine"]


def __getattr__(name: str):
    if name == "MonteCarloEngine":
        from engines.mc.monte_carlo_engine import MonteCarloEngine
        return MonteCarloEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

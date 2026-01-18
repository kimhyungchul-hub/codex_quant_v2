from __future__ import annotations

from typing import Any, Dict, Optional


_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from engines.mc.monte_carlo_engine import MonteCarloEngine

        _ENGINE = MonteCarloEngine()
    return _ENGINE


def _extract_alpha_hit_features(*args, **kwargs):
    return _get_engine()._extract_alpha_hit_features(*args, **kwargs)


def collect_alpha_hit_sample(*args, **kwargs):
    return _get_engine().collect_alpha_hit_sample(*args, **kwargs)


def _predict_horizon_hit_probs(*args, **kwargs):
    return _get_engine()._predict_horizon_hit_probs(*args, **kwargs)

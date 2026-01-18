from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MCParams:
    min_win: float
    profit_target: float
    ofi_weight: float
    max_kelly: float
    cvar_alpha: float
    cvar_scale: float
    n_paths: int


DEFAULT_PARAMS: Dict[str, MCParams] = {
    "bull": MCParams(min_win=0.55, profit_target=0.0020, ofi_weight=0.0015, max_kelly=0.25, cvar_alpha=0.05, cvar_scale=6.0, n_paths=16000),
    "bear": MCParams(min_win=0.57, profit_target=0.0020, ofi_weight=0.0018, max_kelly=0.20, cvar_alpha=0.05, cvar_scale=7.0, n_paths=16000),
    "chop": MCParams(min_win=0.60, profit_target=0.0015, ofi_weight=0.0022, max_kelly=0.10, cvar_alpha=0.05, cvar_scale=8.0, n_paths=16000),
}


def ema(values: Sequence[float], period: int) -> Optional[float]:
    if values is None or len(values) < 2:
        return None
    v = np.asarray(values, dtype=np.float64)
    period = max(2, int(period))
    alpha = 2.0 / (period + 1.0)
    e = float(v[0])
    for x in v[1:]:
        e = alpha * float(x) + (1.0 - alpha) * e
    return float(e)


_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from engines.mc.monte_carlo_engine import MonteCarloEngine

        _ENGINE = MonteCarloEngine()
    return _ENGINE


def evaluate_entry_metrics(ctx: Dict[str, Any], params: MCParams, seed: int) -> Dict[str, Any]:
    return _get_engine().evaluate_entry_metrics(ctx, params, seed)


def decide(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return _get_engine().decide(ctx)


def compute_exit_policy_metrics(**kwargs) -> Dict[str, Any]:
    return _get_engine().compute_exit_policy_metrics(**kwargs)

from __future__ import annotations

"""
Legacy compatibility wrapper.

`engines.mc.evaluation` used to contain entry evaluation + exit-policy logic.
It has been split into smaller modules:
  - `entry_evaluation.py`
  - `policy_weights.py`
  - `execution_mix.py`
  - `exit_policy.py`
This module remains as a thin shim for backwards compatibility.
"""

from engines.mc.entry_evaluation import MonteCarloEntryEvaluationMixin
from engines.mc.execution_mix import MonteCarloExecutionMixMixin
from engines.mc.exit_policy import MonteCarloExitPolicyMixin
from engines.mc.policy_weights import MonteCarloPolicyWeightsMixin


class MonteCarloEvaluationMixin(
    MonteCarloEntryEvaluationMixin,
    MonteCarloPolicyWeightsMixin,
    MonteCarloExecutionMixMixin,
    MonteCarloExitPolicyMixin,
):
    pass


__all__ = ["MonteCarloEvaluationMixin"]

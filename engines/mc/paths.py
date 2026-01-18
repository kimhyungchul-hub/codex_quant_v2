from __future__ import annotations

"""
Legacy compatibility wrapper.

`engines.mc.paths` used to contain a large collection of Monte Carlo helper methods.
It has been split into smaller, more focused modules under `engines/mc/`.
Keep this module as a thin shim so existing imports keep working.
"""

from engines.mc.execution_costs import MonteCarloExecutionCostsMixin
from engines.mc.first_passage import MonteCarloFirstPassageMixin
from engines.mc.path_simulation import MonteCarloPathSimulationMixin
from engines.mc.runtime_params import MonteCarloRuntimeParamsMixin
from engines.mc.signal_features import MonteCarloSignalFeaturesMixin, ema
from engines.mc.tail_sampling import MonteCarloTailSamplingMixin


class MonteCarloPathsMixin(
    MonteCarloSignalFeaturesMixin,
    MonteCarloRuntimeParamsMixin,
    MonteCarloExecutionCostsMixin,
    MonteCarloTailSamplingMixin,
    MonteCarloPathSimulationMixin,
    MonteCarloFirstPassageMixin,
):
    pass


__all__ = ["ema", "MonteCarloPathsMixin"]

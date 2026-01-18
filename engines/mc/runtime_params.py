from __future__ import annotations

from typing import Any, Dict

from engines.mc.constants import MC_N_PATHS_LIVE
from engines.mc.params import DEFAULT_PARAMS, MCParams


class MonteCarloRuntimeParamsMixin:
    def _get_params(self, regime: str, ctx: Dict[str, Any]) -> MCParams:
        rp = ctx.get("regime_params")
        # lightweight runtime override for live mode responsiveness
        n_paths_override = ctx.get("n_paths")
        if n_paths_override is None:
            n_paths_override = MC_N_PATHS_LIVE
        try:
            n_paths_override = int(n_paths_override)
        except Exception:
            n_paths_override = MC_N_PATHS_LIVE
        n_paths_override = int(max(200, min(200000, n_paths_override)))
        if isinstance(rp, dict):
            # dict → MCParams로 안전 변환
            base = DEFAULT_PARAMS.get(regime, DEFAULT_PARAMS["chop"])
            return MCParams(
                min_win=float(rp.get("min_win", base.min_win)),
                profit_target=float(rp.get("profit_target", base.profit_target)),
                ofi_weight=float(rp.get("ofi_weight", base.ofi_weight)),
                max_kelly=float(rp.get("max_kelly", base.max_kelly)),
                cvar_alpha=float(rp.get("cvar_alpha", base.cvar_alpha)),
                cvar_scale=float(rp.get("cvar_scale", base.cvar_scale)),
                # ctx override가 항상 우선 (live에서 프리즈 방지)
                n_paths=int(n_paths_override),
            )
        base = DEFAULT_PARAMS.get(regime, DEFAULT_PARAMS["chop"])
        if n_paths_override != int(base.n_paths):
            return MCParams(
                min_win=float(base.min_win),
                profit_target=float(base.profit_target),
                ofi_weight=float(base.ofi_weight),
                max_kelly=float(base.max_kelly),
                cvar_alpha=float(base.cvar_alpha),
                cvar_scale=float(base.cvar_scale),
                n_paths=int(n_paths_override),
            )
        return base

"""
Optimal Horizon-Based Opportunity Scoring

Mathematically rigorous approach to comparing holding existing positions
vs switching to new candidates by finding optimal holding periods and
integrating EV density over time.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

from engines.mc.constants import SECONDS_PER_YEAR


def find_optimal_horizon(
    ev_by_horizon: Dict[int, float],
    fee: float,
    min_horizon: Optional[int] = None,
    max_horizon: Optional[int] = None
) -> Tuple[float, float]:
    """
    Find T* that maximizes (EV(t) - fee) / t using spline interpolation.
    
    This solves: argmax_t [(EV(t) - Cost) / t]
    
    Args:
        ev_by_horizon: {horizon_sec: ev_value}, e.g., {300: 0.005, 600: 0.008, ...}
        fee: Total fee to subtract (roundtrip for new, exit-only for existing)
        min_horizon: Minimum allowed horizon (default: smallest in dict)
        max_horizon: Maximum allowed horizon (default: largest in dict)
    
    Returns:
        (optimal_horizon_sec, max_return_per_sec)
    """
    if not ev_by_horizon or len(ev_by_horizon) < 2:
        # Need at least 2 points for interpolation
        return 3600.0, 0.0
    
    # Sort horizons
    horizons = sorted(ev_by_horizon.keys())
    evs = [ev_by_horizon[h] for h in horizons]
    
    # Bounds
    h_min = min_horizon if min_horizon is not None else horizons[0]
    h_max = max_horizon if max_horizon is not None else horizons[-1]
    
    if h_min >= h_max:
        return float(h_min), 0.0
    
    # Create cubic spline interpolator
    try:
        spline = CubicSpline(horizons, evs, extrapolate=False)
    except Exception:
        # Fallback to linear interpolation
        spline = lambda t: np.interp(t, horizons, evs)
    
    # Objective: maximize (EV(t) - fee) / t
    # We minimize the negative for scipy
    def neg_return_per_sec(t):
        if t < h_min or t > h_max:
            return 1e9
        try:
            ev_t = float(spline(t))
            return -((ev_t - fee) / t)
        except Exception:
            return 1e9
    
    # Optimize
    try:
        result = minimize_scalar(
            neg_return_per_sec,
            bounds=(h_min, h_max),
            method='bounded',
            options={'xatol': 1.0}  # 1 second tolerance
        )
        
        optimal_t = float(result.x)
        max_return = -float(result.fun)  # Negate back
        
        return optimal_t, max_return
    except Exception:
        # Fallback: use longest horizon
        return float(h_max), 0.0


from core.evaluation_utils import calculate_time_weighted_ev_integral

def calculate_opportunity_score(
    ev_by_horizon: Dict[int, float],
    optimal_horizon: float,
    rho: float,
    current_age_sec: float = 0.0,
    entry_fee: float = 0.0,
    exit_fee: float = 0.0,
    slippage: float = 0.0
) -> float:
    """
    Calculate opportunity score using Time-Weighted EV Integration.
    
    Formula:
    Score = ∫_{0}^{T_rem} (EV(t) - TotalCost) * exp(-rho * t) dt
    
    Where:
    - T_rem = optimal_horizon - current_age_sec (for existing) or optimal_horizon (for new)
    - TotalCost = ExitFee (existing) or EntryFee + ExitFee + Slippage (new)
    
    Args:
        ev_by_horizon: MC results {horizon_sec: ev_value}
        optimal_horizon: T* from find_optimal_horizon
        rho: Dynamic decay coefficient (opportunity cost)
        current_age_sec: For existing positions, time already held
        entry_fee: Entry cost (0 for existing positions)
        exit_fee: Exit cost
        slippage: Expected slippage
    
    Returns:
        opportunity_score (discounted integral value)
    """
    if not ev_by_horizon or len(ev_by_horizon) < 2:
        return 0.0
    
    # 1. Determine integration duration
    # For existing positions, we evaluate from now (t=0) until the remaining time.
    # The ev_by_horizon dictionary usually starts from 0 (now).
    t_rem = max(0.0, optimal_horizon - current_age_sec)
    if t_rem <= 0:
        return 0.0
    
    total_fee = entry_fee + exit_fee + slippage
    
    # 2. Prepare cost-adjusted EV curve
    # We subtract the total fee from each EV point.
    adjusted_ev = {h: (float(ev) - total_fee) for h, ev in ev_by_horizon.items()}
    
    # 3. Calculate discounted integral from 0 to t_rem
    integral_value = calculate_time_weighted_ev_integral(
        ev_by_horizon=adjusted_ev,
        rho=rho,
        t_star=t_rem,
        start_t=0.0
    )
    
    # We return the integral value. 
    # Note: The user mentioned "this discounted integral value" should be used for comparison.
    return integral_value


def extract_ev_by_horizon_from_meta(mc_meta: Dict) -> Optional[Dict[int, float]]:
    """
    Extract horizon-specific EV values from mc_meta.
    
    Looks for patterns like:
    - ev_by_horizon + horizon_seq: [0.005, 0.008, ...] and [300, 600, ...]
    - net_ev_by_horizon_sec: {300: 0.005, 600: 0.008, ...}
    - policy_ev_h_300, policy_ev_h_600, etc.
    
    Args:
        mc_meta: Monte Carlo metadata dict
    
    Returns:
        {horizon_sec: ev_value} or None if not found
    """
    if not mc_meta or not isinstance(mc_meta, dict):
        return None
    
    # Priority 1: ev_by_horizon + horizon_seq arrays
    ev_by_h = mc_meta.get("ev_by_horizon")
    h_seq = mc_meta.get("horizon_seq")
    if ev_by_h and h_seq and isinstance(ev_by_h, (list, tuple)) and isinstance(h_seq, (list, tuple)):
        if len(ev_by_h) == len(h_seq) and len(ev_by_h) >= 2:
            try:
                result = {int(h): float(ev) for h, ev in zip(h_seq, ev_by_h)}
                return result
            except (TypeError, ValueError):
                pass
    
    # Priority 2: net_ev_by_horizon_sec dict
    net_ev_by_h = mc_meta.get("net_ev_by_horizon_sec")
    if isinstance(net_ev_by_h, dict) and len(net_ev_by_h) >= 2:
        return {int(k): float(v) for k, v in net_ev_by_h.items()}
    
    # Priority 3: policy_ev_by_h_long/short + policy_horizons
    h_seq_policy = mc_meta.get("policy_horizons") or mc_meta.get("horizon_seq")
    if h_seq_policy and isinstance(h_seq_policy, (list, tuple)):
        # Try to find a valid EV list among long/short/mix
        for key in ["policy_ev_by_h_long", "policy_ev_by_h_short", "ev_by_horizon"]:
            ev_list = mc_meta.get(key)
            if ev_list and isinstance(ev_list, (list, tuple)) and len(ev_list) == len(h_seq_policy):
                try:
                    return {int(h): float(ev) for h, ev in zip(h_seq_policy, ev_list)}
                except (TypeError, ValueError):
                    continue

    # Priority 4: individual policy_ev_h_{horizon} keys
    ev_dict = {}
    try:
        from engines.mc.constants import STATIC_HORIZONS
        common_horizons = [int(h) for h in STATIC_HORIZONS] if STATIC_HORIZONS else [300, 600, 1200, 1800, 3600]
    except Exception:
        common_horizons = [300, 600, 1200, 1800, 3600]  # Common MC horizons
    
    for h in common_horizons:
        key = f"policy_ev_h_{h}"
        if key in mc_meta:
            try:
                ev_dict[h] = float(mc_meta[key])
            except (TypeError, ValueError):
                continue
    
    if len(ev_dict) >= 2:
        return ev_dict
    
    return None


def extract_directional_ev_by_horizon_from_meta(
    mc_meta: Dict,
) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]]]:
    """
    Extract directional (LONG/SHORT) horizon-specific EV curves from mc_meta.

    Returns:
        (ev_long_by_horizon_sec, ev_short_by_horizon_sec)

    Notes:
        - This is used for vectorized Score_A where direction matters.
        - Prefers hold-to-horizon curves (`ev_by_horizon_long/short`).
    """
    if not mc_meta or not isinstance(mc_meta, dict):
        return None, None

    def _from_lists(ev_key: str, h_key: str) -> Optional[Dict[int, float]]:
        ev_list = mc_meta.get(ev_key)
        h_list = mc_meta.get(h_key)
        if not ev_list or not h_list:
            return None
        if not isinstance(ev_list, (list, tuple)) or not isinstance(h_list, (list, tuple)):
            return None
        if len(ev_list) != len(h_list) or len(ev_list) < 2:
            return None
        try:
            return {int(h): float(ev) for h, ev in zip(h_list, ev_list)}
        except (TypeError, ValueError):
            return None

    def _from_dict(key: str) -> Optional[Dict[int, float]]:
        v = mc_meta.get(key)
        if not isinstance(v, dict) or len(v) < 2:
            return None
        try:
            return {int(k): float(val) for k, val in v.items()}
        except (TypeError, ValueError):
            return None

    # Preferred: hold-to-horizon EV curves (directional)
    ev_long = (
        _from_lists("ev_by_horizon_long", "horizon_seq_long")
        or _from_lists("ev_by_horizon_long", "horizon_seq")
        or _from_dict("net_ev_by_horizon_sec_long")
    )
    ev_short = (
        _from_lists("ev_by_horizon_short", "horizon_seq_short")
        or _from_lists("ev_by_horizon_short", "horizon_seq")
        or _from_dict("net_ev_by_horizon_sec_short")
    )

    # Fallback: policy multi-horizon EV curves (directional), if available.
    if ev_long is None:
        ev_long = _from_lists("policy_ev_by_h_long", "policy_horizons")
    if ev_short is None:
        ev_short = _from_lists("policy_ev_by_h_short", "policy_horizons")

    return ev_long, ev_short


def extract_signed_ev_rate_by_horizon_from_meta(mc_meta: Dict) -> Optional[Dict[int, float]]:
    """
    Extract a *signed* expected return-rate curve (per-second) by horizon.

    Definition:
      - If both LONG/SHORT hold-to-horizon EV curves exist, approximate the underlying
        signed return (price-directional) as: signed_ev_total ≈ 0.5 * (EV_long - EV_short).
      - If only one directional curve exists:
          - LONG-only: treat it as already signed (positive = up, negative = down).
          - SHORT-only: signed_ev_total ≈ -EV_short.
      - Fallback: use the non-directional curve if present.

    Returns:
      {horizon_sec: signed_ev_rate_per_sec}, for horizons > 0.
    """
    if not mc_meta or not isinstance(mc_meta, dict):
        return None

    ev_long, ev_short = extract_directional_ev_by_horizon_from_meta(mc_meta)
    ev_single = extract_ev_by_horizon_from_meta(mc_meta)

    signed_total_by_h: Optional[Dict[int, float]] = None

    if ev_long and ev_short:
        common = sorted(set(ev_long.keys()) & set(ev_short.keys()))
        if len(common) >= 2:
            signed_total_by_h = {int(h): 0.5 * (float(ev_long[h]) - float(ev_short[h])) for h in common}
    elif ev_long:
        signed_total_by_h = {int(h): float(v) for h, v in ev_long.items()}
    elif ev_short:
        signed_total_by_h = {int(h): -float(v) for h, v in ev_short.items()}
    elif ev_single:
        signed_total_by_h = {int(h): float(v) for h, v in ev_single.items()}

    if not signed_total_by_h or len(signed_total_by_h) < 2:
        return None

    out: Dict[int, float] = {}
    for h, ev_total in signed_total_by_h.items():
        try:
            hs = int(h)
            if hs <= 0:
                continue
            out[hs] = float(ev_total) / float(hs)
        except Exception:
            continue

    return out if len(out) >= 2 else None

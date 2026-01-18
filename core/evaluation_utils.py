import numpy as np
from typing import Dict

def calculate_time_weighted_ev_integral(ev_by_horizon: Dict[int, float], rho: float, t_star: float, start_t: float = 0.0) -> float:
    """
    Calculates the time-weighted integral of EV from start_t to t_star with decay rho.
    Score = integral_{start_t}^{t_star} (EV(t)) * exp(-rho * t) dt
    Note: Cost handling is usually done before passing EV or by subtracting from the result.
    Here we assume ev_by_horizon contains EV(t) values.
    """
    if not ev_by_horizon:
        return 0.0

    # Filter horizons within [start_t, t_star]
    horizons = sorted([h for h in ev_by_horizon.keys() if start_t <= h <= t_star])
    
    if len(horizons) < 2:
        # Not enough points to integrate, return a simple approximation
        ev_at_tstar = ev_by_horizon.get(int(t_star), 0.0)
        dur = max(0.0, t_star - start_t)
        return ev_at_tstar * np.exp(-rho * t_star) * dur

    t_values = np.array(horizons)
    ev_values = np.array([ev_by_horizon[h] for h in horizons])
    
    # Weight by decay factor
    weights = np.exp(-rho * t_values)
    weighted_ev = ev_values * weights
    
    # Numerical integration using trapezoidal rule
    area = np.trapz(weighted_ev, t_values)
    
    return float(area)

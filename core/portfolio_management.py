import time
import numpy as np
from typing import List, Dict, Any

def calculate_dynamic_decay_rho(trade_tape: List[Dict[str, Any]], window_hours: float = 24.0, initial_capital: float = 10000.0) -> float:
    """
    Calculates the average realized return per second (rho) over the last window_hours.
    Returns rho in units of 'return percentage per second'.
    """
    # 1. Default hurdle rate: 20% APR (annualized)
    # Log(1.20) / (365 * 24 * 3600)
    SEC_IN_YEAR = 365 * 24 * 3600
    default_rho = np.log(1.20) / SEC_IN_YEAR 

    if not trade_tape:
        return default_rho

    def _get_trade_ts_ms(trade: Dict[str, Any]) -> int | None:
        if not isinstance(trade, dict):
            return None
        for k in ("ts", "ts_ms", "timestamp", "time"):
            v = trade.get(k)
            if v is None:
                continue
            if isinstance(v, (int, float)):
                try:
                    iv = int(v)
                except Exception:
                    continue
                if iv > 0:
                    return iv
            if isinstance(v, str):
                s = v.strip()
                if s.isdigit():
                    try:
                        iv = int(s)
                    except Exception:
                        continue
                    if iv > 0:
                        return iv
        return None

    now_ms = int(time.time() * 1000)
    window_ms = int(window_hours * 3600 * 1000)
    cutoff_ms = now_ms - window_ms

    recent_pnl = 0.0
    for trade in reversed(trade_tape):
        trade_ts = _get_trade_ts_ms(trade)
        if trade_ts is None:
            continue
            
        if trade_ts < cutoff_ms:
            break
        if trade.get("type") == "EXIT":
            recent_pnl += float(trade.get("pnl", 0.0))

    # Calculate rate: return_pct = pnl / capital
    # We use a simple return for now. 
    # rho = return_pct / window_seconds
    window_sec = window_hours * 3600.0
    realized_return_pct = recent_pnl / initial_capital
    
    # If realized return is negative, we shouldn't use a negative decay (it would inflate future values).
    # We use the hurdle rate as a floor.
    if realized_return_pct <= 0:
        return default_rho
    
    rho = realized_return_pct / window_sec
    
    # Clip to avoid extreme values. 
    # Max rho: 500% annualized? 
    max_rho = np.log(5.0) / SEC_IN_YEAR 
    return float(np.clip(rho, default_rho, max_rho))

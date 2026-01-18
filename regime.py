import datetime
from typing import Any, Dict, Optional, Tuple


def time_regime() -> str:
    """Return current time-based regime: ASIA, EU, US, or OFF."""
    h = datetime.datetime.utcnow().hour
    if 0 <= h < 6:
        return "ASIA"
    if 6 <= h < 13:
        return "EU"
    if 13 <= h < 21:
        return "US"
    return "OFF"


def adjust_mu_sigma(
    mu: float,
    sigma: float,
    regime_ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """
    Adjust drift (mu) and volatility (sigma) based on regime context.

    Args:
        mu: Base drift (annualized)
        sigma: Base volatility (annualized)
        regime_ctx: Optional dict with 'regime', 'vix', 'session' keys

    Returns:
        (adjusted_mu, adjusted_sigma)

    See docs/MATHEMATICS.md for formula details.
    """
    if regime_ctx is None:
        return mu, sigma

    # 방어적 처리: str이 들어온 경우 dict로 래핑
    if isinstance(regime_ctx, str):
        regime_ctx = {"regime": regime_ctx}

    regime = regime_ctx.get("regime", "neutral")
    session = regime_ctx.get("session", time_regime())

    # Regime-based multipliers
    regime_mult = {
        "bull": (1.2, 0.9),    # Higher drift, lower vol in bull
        "bear": (0.7, 1.3),    # Lower drift, higher vol in bear
        "chop": (0.8, 1.5),    # Choppy market: lower drift, much higher vol
        "neutral": (1.0, 1.0),
    }
    mu_mult, sigma_mult = regime_mult.get(regime, (1.0, 1.0))

    # Session-based adjustments
    session_mult = {
        "ASIA": (0.9, 0.8),   # Lower activity
        "EU": (1.0, 1.0),     # Normal
        "US": (1.1, 1.2),     # Higher activity
        "OFF": (0.7, 0.7),    # Weekend/off-hours
    }
    mu_sess, sigma_sess = session_mult.get(session, (1.0, 1.0))

    # VIX-based vol boost (if provided)
    vix = regime_ctx.get("vix")
    vix_mult = 1.0
    if vix is not None and vix > 25:
        vix_mult = 1.0 + (vix - 25) * 0.02  # 2% per VIX point above 25

    adjusted_mu = mu * mu_mult * mu_sess
    adjusted_sigma = sigma * sigma_mult * sigma_sess * vix_mult

    return adjusted_mu, adjusted_sigma


def get_regime_mu_sigma(
    regime: str,
    session: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Get baseline mu and sigma for a given regime.

    Args:
        regime: Market regime ('bull', 'bear', 'chop', 'neutral')
        session: Optional session override
        symbol: Optional symbol for asset-specific params (e.g., 'BTC/USDT:USDT')

    Returns:
        (mu, sigma) - annualized values
    """
    # Asset-specific base volatility (annualized)
    # BTC is baseline, alts typically have higher vol
    symbol_vol_mult = {
        "BTC": 1.0,
        "ETH": 1.15,
        "SOL": 1.4,
        "BNB": 1.2,
        "XRP": 1.3,
        "ADA": 1.35,
        "DOGE": 1.5,
        "AVAX": 1.4,
        "DOT": 1.35,
    }
    
    base_sigma = 0.80  # ~80% annualized vol for BTC
    
    # Extract base asset from symbol (e.g., 'BTC/USDT:USDT' -> 'BTC')
    vol_mult = 1.0
    if symbol:
        base_asset = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
        vol_mult = symbol_vol_mult.get(base_asset, 1.2)  # default 1.2 for unknown alts
    
    adjusted_base_sigma = base_sigma * vol_mult

    regime_params = {
        "bull": (0.50, adjusted_base_sigma * 0.9),    # Positive drift, lower vol
        "bear": (-0.20, adjusted_base_sigma * 1.3),   # Negative drift, higher vol
        "chop": (0.00, adjusted_base_sigma * 1.5),    # Zero drift, high vol
        "neutral": (0.10, adjusted_base_sigma),        # Slight positive drift
    }

    mu, sigma = regime_params.get(regime, (0.10, adjusted_base_sigma))

    # Session adjustment (optional)
    if session:
        ctx = {"regime": regime, "session": session}
        mu, sigma = adjust_mu_sigma(mu, sigma, ctx)

    return mu, sigma

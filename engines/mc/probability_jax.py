from __future__ import annotations

from engines.mc.jax_backend import ensure_jax, lazy_jit, jnp, jax
# We keep numpy/scipy for the numpy fallback
import numpy as np
from scipy.stats import norm as scipy_norm

@lazy_jit()
def _erf_approx(x):
    # Polynomial approximation of erf(x) for Metal/MPS which doesn't support mhlo.erf
    # erf(x) ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
    # where t = 1 / (1 + p*x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = jnp.sign(x)
    abs_x = jnp.abs(x)
    t = 1.0 / (1.0 + p * abs_x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * jnp.exp(-abs_x * abs_x)
    return sign * y

@lazy_jit()
def _norm_cdf_jax(x):
    # norm.cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + _erf_approx(x / jnp.sqrt(2.0)))

@lazy_jit()
def _approx_p_pos_and_ev_hold_jax(
    mu: float,
    sigma: float,
    tau_sec: float,
    direction: int,
    leverage: float,
    fee_roundtrip: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    tau = jnp.maximum(0.0, tau_sec)
    sig = jnp.float32(sigma)
    lev = jnp.float32(leverage)
    fee = jnp.float32(fee_roundtrip)

    # Note: JAX doesn't like conditional returns if it can avoid them.
    # We use jnp.where or similar for vectorized logic if needed.
    
    # ✅ Apply direction to drift: Short positions profit from negative drift
    mu_directional = jnp.float32(direction) * mu
    
    m = mu_directional * tau
    v = (sig * sig) * tau
    s = jnp.sqrt(jnp.maximum(1e-12, v))

    thr = fee / jnp.maximum(1e-12, lev)
    
    # For p_pos calculation, use directional drift
    z = (m - thr) / s
    p_pos = _norm_cdf_jax(z)
    
    # EV calculation: already directional from mu_directional
    ev = m * lev - fee
    
    # Debug print removed to fix TracerBoolConversionError
    
    # Handle tau <= 0 or sig <= 0 or lev <= 0
    invalid = (tau <= 0.0) | (sig <= 0.0) | (lev <= 0.0)
    p_pos = jnp.where(invalid, jnp.where(ev <= 0.0, 0.0, 1.0), p_pos)
    ev = jnp.where(invalid, -fee, ev)
    
    return p_pos, ev

@lazy_jit()
def _prob_max_geq_jax(mu0: float, sig0: float, T: float, a: float) -> jnp.ndarray:
    T = jnp.maximum(0.0, T)
    sig0 = jnp.float32(sig0)
    a = jnp.float32(a)
    
    s = sig0 * jnp.sqrt(T)
    z1 = (a - mu0 * T) / jnp.maximum(1e-12, s)
    expo = (2.0 * mu0 * a) / jnp.maximum(1e-12, sig0 * sig0)
    expo = jnp.clip(expo, -80.0, 80.0)
    term = jnp.exp(expo) * _norm_cdf_jax((-a - mu0 * T) / jnp.maximum(1e-12, s))
    p = (1.0 - _norm_cdf_jax(z1)) + term
    
    # Case T <= 0
    p_t0 = jnp.where(a <= 0.0, 1.0, 0.0)
    # Case sig0 <= 0
    xT = mu0 * T
    p_sig0 = jnp.where(xT >= a, 1.0, 0.0)
    
    p = jnp.where(T <= 0.0, p_t0, jnp.where(sig0 <= 0.0, p_sig0, p))
    
    return jnp.clip(p, 0.0, 1.0)

@lazy_jit()
def _prob_min_leq_jax(mu0: float, sig0: float, T: float, neg_a: float) -> jnp.ndarray:
    a = jnp.abs(neg_a)
    return _prob_max_geq_jax(-mu0, sig0, T, a)


def _approx_p_pos_and_ev_hold_numpy(
    mu: float,
    sigma: float,
    tau_sec: float,
    direction: int,
    leverage: float,
    fee_roundtrip: float,
) -> tuple[float, float]:
    tau = max(0.0, float(tau_sec))
    sig = float(sigma)
    lev = float(leverage)
    fee = float(fee_roundtrip)

    mu_directional = float(direction) * mu
    
    m = mu_directional * tau
    v = (sig * sig) * tau
    s = np.sqrt(max(1e-12, v))

    thr = fee / max(1e-12, lev)
    
    z = (m - thr) / s
    p_pos = scipy_norm.cdf(z)
    
    ev = m * lev - fee
    
    # Invalid cases
    if tau <= 0.0 or sig <= 0.0 or lev <= 0.0:
        if ev <= 0.0:
            p_pos = 0.0
        else:
            p_pos = 1.0
        if tau <= 0.0: ev = -fee
    
    return float(p_pos), float(ev)

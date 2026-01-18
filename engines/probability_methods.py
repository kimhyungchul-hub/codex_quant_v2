from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _approx_p_pos_and_ev_hold(
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

    if tau <= 0.0 or sig <= 0.0 or lev <= 0.0:
        ev = -fee
        p_pos = 0.0 if ev <= 0.0 else 1.0
        return float(p_pos), float(ev)

    m = float(mu) * tau
    v = (sig * sig) * tau
    s = math.sqrt(max(1e-12, v))

    thr = fee / max(1e-12, lev)
    if int(direction) == 1:
        z = (m - thr) / s
        p_pos = _norm_cdf(z)
    else:
        z = (-thr - m) / s
        p_pos = _norm_cdf(z)

    ev = float(direction) * m * lev - fee
    return float(p_pos), float(ev)


def _prob_max_geq(mu0: float, sig0: float, T: float, a: float) -> float:
    """
    For X_t = mu0 * t + sig0 * W_t, approximate P(max_{0..T} X_t >= a) using reflection principle.
    """
    T = float(max(0.0, T))
    sig0 = float(sig0)
    a = float(a)
    if T <= 0.0:
        return 1.0 if a <= 0.0 else 0.0
    if sig0 <= 0.0:
        xT = float(mu0) * T
        return 1.0 if xT >= a else 0.0
    s = sig0 * math.sqrt(T)
    z1 = (a - float(mu0) * T) / max(1e-12, s)
    expo = (2.0 * float(mu0) * a) / max(1e-12, sig0 * sig0)
    expo = float(max(-80.0, min(80.0, expo)))
    term = math.exp(expo) * _norm_cdf((-a - float(mu0) * T) / max(1e-12, s))
    p = (1.0 - _norm_cdf(z1)) + term
    return float(max(0.0, min(1.0, p)))


def _prob_min_leq(mu0: float, sig0: float, T: float, neg_a: float) -> float:
    a = float(-neg_a)
    if a < 0.0:
        a = -a
    return _prob_max_geq(-float(mu0), float(sig0), float(T), float(a))

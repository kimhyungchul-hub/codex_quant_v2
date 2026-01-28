from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * float(x) * float(x)) / math.sqrt(2.0 * math.pi)


def _norm_ppf(p: float) -> float:
    # Acklam's approximation for inverse normal CDF.
    p = float(p)
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    # Coefficients
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return (num / den) * q


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


def _approx_cvar_normal(
    mu: float,
    sigma: float,
    tau_sec: float,
    direction: int,
    leverage: float,
    fee_roundtrip: float,
    alpha: float = 0.05,
) -> float:
    tau = max(0.0, float(tau_sec))
    sig = float(sigma)
    lev = float(leverage)
    fee = float(fee_roundtrip)
    if tau <= 0.0 or sig <= 0.0 or lev <= 0.0:
        mean = float(direction) * float(mu) * tau * lev - fee
        return float(mean)
    alpha_f = float(min(0.5, max(1e-6, float(alpha))))
    m = float(mu) * tau
    s = sig * math.sqrt(max(1e-12, tau))
    mean = float(direction) * m * lev - fee
    std = float(max(1e-12, s * lev))
    z = _norm_ppf(alpha_f)
    pdf = _norm_pdf(z)
    cvar = mean - std * (pdf / alpha_f)
    return float(cvar)


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

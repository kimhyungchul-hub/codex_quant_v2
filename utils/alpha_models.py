from __future__ import annotations

"""
Alpha model utilities for mu_alpha estimation.

- Designed for lightweight, real-time updates.
- NumPy-first for simplicity; compatible with Torch by feeding scalar floats.
- All drift outputs are per-bar (log-return) unless explicitly annualized.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import json
import os

import numpy as np


# -----------------------------
# Helpers
# -----------------------------

def _annualize_mu(mu_bar: float, bar_seconds: float) -> float:
    if bar_seconds <= 0:
        bar_seconds = 60.0
    return float(mu_bar) * (31536000.0 / float(bar_seconds))


def _annualize_sigma(sigma_bar: float, bar_seconds: float) -> float:
    if bar_seconds <= 0:
        bar_seconds = 60.0
    return float(sigma_bar) * math.sqrt(31536000.0 / float(bar_seconds))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _erf(x: float) -> float:
    # math.erf is fast and sufficient
    return math.erf(x)


# -----------------------------
# MLOFI
# -----------------------------

def compute_mlofi(
    prev_ob: Optional[Dict[str, Any]],
    ob: Optional[Dict[str, Any]],
    levels: int = 5,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Compute Multi-Level Order Flow Imbalance (MLOFI).

    Args:
        prev_ob: previous orderbook {"bids": [[p,q],...], "asks": [[p,q],...]}
        ob: current orderbook
        levels: depth levels to use
        weights: optional weights for each level (len=levels)

    Returns:
        (mlofi_scalar, mlofi_vec)
    """
    if not prev_ob or not ob:
        return 0.0, np.zeros((levels,), dtype=np.float64)

    bids_prev = prev_ob.get("bids") or []
    asks_prev = prev_ob.get("asks") or []
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []

    L = max(1, int(levels))
    vec = np.zeros((L,), dtype=np.float64)

    def _ofi_side(curr, prev, side: str) -> float:
        # curr/prev: [price, qty]
        if not curr or not prev:
            # If missing, treat as zero impact
            return 0.0
        try:
            p = float(curr[0])
            q = float(curr[1])
            p0 = float(prev[0])
            q0 = float(prev[1])
        except Exception:
            return 0.0
        if side == "bid":
            if p > p0:
                return q
            if p == p0:
                return q - q0
            return -q0
        # ask side
        if p < p0:
            return q
        if p == p0:
            return q - q0
        return -q0

    for k in range(L):
        bid_k = bids[k] if k < len(bids) else None
        ask_k = asks[k] if k < len(asks) else None
        bid_prev_k = bids_prev[k] if k < len(bids_prev) else None
        ask_prev_k = asks_prev[k] if k < len(asks_prev) else None
        ofi_bid = _ofi_side(bid_k, bid_prev_k, "bid")
        ofi_ask = _ofi_side(ask_k, ask_prev_k, "ask")
        vec[k] = float(ofi_bid - ofi_ask)

    if weights is None:
        # exponential decay default
        lam = 0.4
        weights = np.exp(-lam * np.arange(L, dtype=np.float64))
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.size != L:
            weights = np.resize(weights, (L,))

    denom = float(np.sum(np.abs(weights)))
    if denom <= 0:
        return 0.0, vec
    mlofi = float(np.dot(vec, weights) / denom)
    return mlofi, vec


# -----------------------------
# VPIN (streaming)
# -----------------------------

@dataclass
class VPINState:
    bucket_size: float = 0.0
    window: int = 50
    acc_vol: float = 0.0
    acc_buy: float = 0.0
    acc_sell: float = 0.0
    bucket_oi: List[float] = field(default_factory=list)


def update_vpin_state(
    state: VPINState,
    delta_p: float,
    volume: float,
    sigma: float,
    eps: float = 1e-8,
) -> float:
    """Streaming VPIN update using BVC (probabilistic buy/sell split)."""
    if state.bucket_size <= 0:
        return 0.0
    v = float(max(volume, 0.0))
    if v <= 0:
        return float(_current_vpin(state, eps))

    sigma = float(abs(sigma)) + eps
    # buy probability using normal CDF
    z = float(delta_p) / sigma
    buy_prob = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
    buy_prob = min(max(buy_prob, 0.0), 1.0)
    buy_v = v * buy_prob
    sell_v = v - buy_v

    remaining = float(state.bucket_size - state.acc_vol)
    while v > 0:
        if v <= remaining + eps:
            state.acc_vol += v
            state.acc_buy += buy_v
            state.acc_sell += sell_v
            v = 0.0
        else:
            fill_ratio = remaining / v
            state.acc_vol += remaining
            state.acc_buy += buy_v * fill_ratio
            state.acc_sell += sell_v * fill_ratio
            state.bucket_oi.append(abs(state.acc_buy - state.acc_sell))
            # carry leftover
            v -= remaining
            buy_v -= buy_v * fill_ratio
            sell_v -= sell_v * fill_ratio
            state.acc_vol = 0.0
            state.acc_buy = 0.0
            state.acc_sell = 0.0
            remaining = float(state.bucket_size)

        if state.acc_vol >= state.bucket_size - eps:
            state.bucket_oi.append(abs(state.acc_buy - state.acc_sell))
            state.acc_vol = 0.0
            state.acc_buy = 0.0
            state.acc_sell = 0.0
            remaining = float(state.bucket_size)

    return float(_current_vpin(state, eps))


def _current_vpin(state: VPINState, eps: float = 1e-8) -> float:
    if not state.bucket_oi:
        return 0.0
    recent = state.bucket_oi[-int(max(1, state.window)) :]
    denom = float(len(recent) * state.bucket_size + eps)
    return float(sum(recent) / denom)


def init_vpin_state_from_series(
    prices: Iterable[float],
    volumes: Iterable[float],
    bucket_size: Optional[float] = None,
    window: int = 50,
) -> VPINState:
    """Initialize VPIN state from historical series."""
    prices_arr = np.asarray(list(prices), dtype=np.float64)
    vols_arr = np.asarray(list(volumes), dtype=np.float64)
    n = min(prices_arr.size, vols_arr.size)
    if n < 2:
        return VPINState(bucket_size=0.0, window=window)
    prices_arr = prices_arr[-n:]
    vols_arr = vols_arr[-n:]
    rets = np.diff(np.log(np.maximum(prices_arr, 1e-12)))
    sigma = float(np.std(rets)) + 1e-8
    if bucket_size is None:
        total_vol = float(np.sum(vols_arr[1:]))
        bucket_size = max(total_vol / max(window, 1), 1e-8)
    state = VPINState(bucket_size=float(bucket_size), window=int(window))
    for i in range(1, n):
        dp = float(prices_arr[i] - prices_arr[i - 1])
        update_vpin_state(state, dp, float(vols_arr[i]), sigma)
    return state


# -----------------------------
# Kalman filter (constant velocity)
# -----------------------------

@dataclass
class KalmanCVState:
    x: Optional[np.ndarray] = None  # [price, velocity]
    P: Optional[np.ndarray] = None  # 2x2
    Q: float = 1e-6
    R: float = 1e-4


def update_kalman_cv(state: KalmanCVState, price: float, dt: float) -> Tuple[KalmanCVState, float]:
    """Update constant-velocity Kalman filter.

    Returns updated state and velocity (price units / second).
    """
    p = float(price)
    if state.x is None or state.P is None:
        state.x = np.array([p, 0.0], dtype=np.float64)
        state.P = np.eye(2, dtype=np.float64)
        return state, 0.0

    dt = float(max(dt, 1e-6))
    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    Q = np.array([[state.Q, 0.0], [0.0, state.Q]], dtype=np.float64)
    R = np.array([[state.R]], dtype=np.float64)

    x_pred = F @ state.x
    P_pred = F @ state.P @ F.T + Q

    y = np.array([p], dtype=np.float64) - (H @ x_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_new = x_pred + (K @ y)
    P_new = (np.eye(2) - K @ H) @ P_pred

    state.x = x_new
    state.P = P_new
    vel = float(x_new[1])
    return state, vel


# -----------------------------
# Particle filter (optional)
# -----------------------------

@dataclass
class ParticleFilterState:
    particles: Optional[np.ndarray] = None  # drift particles
    weights: Optional[np.ndarray] = None
    n_particles: int = 128
    process_std: float = 1e-4
    obs_std: float = 1e-3


def update_particle_filter(state: ParticleFilterState, ret: float) -> Tuple[ParticleFilterState, float]:
    """Simple particle filter for drift estimation. Returns (state, mu_pf)."""
    n = int(max(8, state.n_particles))
    if state.particles is None or state.weights is None:
        state.particles = np.random.normal(0.0, state.process_std, size=(n,))
        state.weights = np.ones((n,), dtype=np.float64) / n

    # predict step
    state.particles = state.particles + np.random.normal(0.0, state.process_std, size=(n,))

    # update weights (likelihood of ret given particle drift)
    obs_std = max(float(state.obs_std), 1e-8)
    likelihood = np.exp(-0.5 * ((ret - state.particles) / obs_std) ** 2)
    state.weights = likelihood + 1e-12
    state.weights = state.weights / np.sum(state.weights)

    # resample
    idx = np.random.choice(np.arange(n), size=n, p=state.weights)
    state.particles = state.particles[idx]
    state.weights = np.ones((n,), dtype=np.float64) / n

    mu_pf = float(np.mean(state.particles))
    return state, mu_pf


# -----------------------------
# Hurst exponent (variance ratio)
# -----------------------------

def estimate_hurst_vr(returns: Iterable[float], taus: Iterable[int]) -> Optional[float]:
    rets = np.asarray(list(returns), dtype=np.float64)
    if rets.size < 20:
        return None
    taus = [int(t) for t in taus if int(t) >= 2]
    if not taus:
        return None

    log_tau = []
    log_var = []
    for tau in taus:
        n = rets.size // tau
        if n <= 1:
            continue
        agg = np.array([np.sum(rets[i * tau : (i + 1) * tau]) for i in range(n)], dtype=np.float64)
        v = float(np.var(agg))
        if v <= 0:
            continue
        log_tau.append(math.log(float(tau)))
        log_var.append(math.log(float(v)))

    if len(log_tau) < 2:
        return None
    # linear regression slope
    x = np.asarray(log_tau)
    y = np.asarray(log_var)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return None
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    H = slope / 2.0
    return float(max(0.0, min(1.0, H)))


# -----------------------------
# GARCH(1,1)
# -----------------------------

@dataclass
class GARCHState:
    omega: float = 1e-6
    alpha: float = 0.05
    beta: float = 0.90
    var: float = 1e-6


def update_garch_state(state: GARCHState, ret: float) -> Tuple[GARCHState, float]:
    r = float(ret)
    var = float(state.var)
    new_var = float(state.omega + state.alpha * (r ** 2) + state.beta * var)
    new_var = max(new_var, 1e-12)
    state.var = new_var
    return state, float(math.sqrt(new_var))


# -----------------------------
# Bayesian mean update (normal-normal)
# -----------------------------

@dataclass
class BayesMeanState:
    mu_mean: float = 0.0
    mu_var: float = 1e-4


def update_bayes_mean(state: BayesMeanState, ret: float, obs_var: float) -> Tuple[BayesMeanState, float, float]:
    obs_var = max(float(obs_var), 1e-12)
    mu_var = max(float(state.mu_var), 1e-12)
    post_var = 1.0 / (1.0 / mu_var + 1.0 / obs_var)
    post_mean = post_var * (state.mu_mean / mu_var + float(ret) / obs_var)
    state.mu_mean = float(post_mean)
    state.mu_var = float(post_var)
    return state, state.mu_mean, state.mu_var


# -----------------------------
# AR(1) forecast (proxy for ARIMA)
# -----------------------------

def estimate_ar1_next_return(returns: Iterable[float]) -> float:
    rets = np.asarray(list(returns), dtype=np.float64)
    if rets.size < 2:
        return 0.0
    y = rets[1:]
    x = rets[:-1]
    var_x = float(np.var(x))
    if var_x <= 1e-12:
        return float(np.mean(rets))
    cov_xy = float(np.mean((x - x.mean()) * (y - y.mean())))
    phi = cov_xy / var_x
    c = float(y.mean() - phi * x.mean())
    pred = c + phi * float(rets[-1])
    return float(pred)


# -----------------------------
# Hawkes process (simple exponential kernel)
# -----------------------------

@dataclass
class HawkesState:
    lambda_buy: float = 0.0
    lambda_sell: float = 0.0
    alpha: float = 0.5
    beta: float = 2.0


def update_hawkes_state(state: HawkesState, event: int, dt: float) -> Tuple[HawkesState, float]:
    dt = float(max(dt, 1e-6))
    decay = math.exp(-state.beta * dt)
    state.lambda_buy *= decay
    state.lambda_sell *= decay
    if event > 0:
        state.lambda_buy += state.alpha
    elif event < 0:
        state.lambda_sell += state.alpha
    boost = float(state.lambda_buy - state.lambda_sell)
    return state, boost


# -----------------------------
# Hidden Markov Model (lightweight Gaussian, online)
# -----------------------------

@dataclass
class GaussianHMMState:
    probs: Optional[np.ndarray] = None      # shape=(K,)
    means: Optional[np.ndarray] = None      # shape=(K,)
    vars: Optional[np.ndarray] = None       # shape=(K,)
    trans: Optional[np.ndarray] = None      # shape=(K,K)
    n_states: int = 3
    adapt_lr: float = 0.02
    var_floor: float = 1e-10


def _gauss_pdf(x: float, mean: float, var: float) -> float:
    v = max(float(var), 1e-12)
    z = (float(x) - float(mean)) / math.sqrt(v)
    return float(math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi * v))


def update_gaussian_hmm(state: GaussianHMMState, ret: float) -> Tuple[GaussianHMMState, Dict[str, Any]]:
    """Online update for a compact Gaussian HMM.

    Returns:
      (state, info)
      info keys:
        - state_idx: posterior argmax
        - state_label: one of {"down","chop","up"}
        - confidence: posterior max probability
        - regime_sign: -1 / 0 / +1
        - means, probs
    """
    k = int(max(2, state.n_states))
    if state.probs is None or state.means is None or state.vars is None or state.trans is None:
        # Bootstrap with a weakly-informative prior around zero return.
        if k == 2:
            init_means = np.array([-1e-4, 1e-4], dtype=np.float64)
            init_probs = np.array([0.5, 0.5], dtype=np.float64)
            init_trans = np.array([[0.97, 0.03], [0.03, 0.97]], dtype=np.float64)
        else:
            init_means = np.array([-1e-4, 0.0, 1e-4], dtype=np.float64)
            init_probs = np.array([0.25, 0.50, 0.25], dtype=np.float64)
            init_trans = np.array(
                [
                    [0.94, 0.05, 0.01],
                    [0.04, 0.92, 0.04],
                    [0.01, 0.05, 0.94],
                ],
                dtype=np.float64,
            )
        state.means = init_means
        state.vars = np.ones((init_means.size,), dtype=np.float64) * 1e-6
        state.probs = init_probs
        state.trans = init_trans
        state.n_states = int(init_means.size)

    means = np.asarray(state.means, dtype=np.float64).reshape(-1)
    variances = np.asarray(state.vars, dtype=np.float64).reshape(-1)
    probs = np.asarray(state.probs, dtype=np.float64).reshape(-1)
    trans = np.asarray(state.trans, dtype=np.float64)
    n = int(min(means.size, variances.size, probs.size, trans.shape[0], trans.shape[1]))
    if n < 2:
        state.n_states = 3
        return update_gaussian_hmm(state, ret)
    means = means[:n]
    variances = variances[:n]
    probs = probs[:n]
    trans = trans[:n, :n]

    # Predict step
    prior = probs @ trans
    prior = np.maximum(prior, 1e-12)
    prior = prior / np.sum(prior)

    # Update with emission likelihood
    like = np.asarray([_gauss_pdf(ret, means[i], variances[i]) for i in range(n)], dtype=np.float64)
    post = prior * np.maximum(like, 1e-30)
    z = float(np.sum(post))
    if z <= 0 or not np.isfinite(z):
        post = prior
    else:
        post = post / z

    # Online adaptation of emission params (posterior-weighted EMA)
    lr = float(np.clip(state.adapt_lr, 1e-4, 0.5))
    ret_f = float(ret)
    for i in range(n):
        w = lr * float(post[i])
        means[i] = means[i] + w * (ret_f - means[i])
        dev = ret_f - means[i]
        variances[i] = variances[i] + w * (dev * dev - variances[i])
        variances[i] = max(float(variances[i]), float(state.var_floor))

    state.probs = post
    state.means = means
    state.vars = variances
    state.trans = trans
    state.n_states = n

    idx = int(np.argmax(post))
    conf = float(np.max(post))
    # Label by sorted mean: smallest=down, middle=chop, largest=up
    order = np.argsort(means)
    label_by_idx: Dict[int, str] = {}
    if n == 2:
        label_by_idx[int(order[0])] = "down"
        label_by_idx[int(order[1])] = "up"
    else:
        label_by_idx[int(order[0])] = "down"
        label_by_idx[int(order[-1])] = "up"
        for j in order[1:-1]:
            label_by_idx[int(j)] = "chop"
    label = str(label_by_idx.get(idx, "chop"))
    regime_sign = 1 if label == "up" else (-1 if label == "down" else 0)

    info = {
        "state_idx": idx,
        "state_label": label,
        "confidence": conf,
        "regime_sign": regime_sign,
        "means": [float(x) for x in means.tolist()],
        "probs": [float(x) for x in post.tolist()],
    }
    return state, info


# -----------------------------
# OU drift
# -----------------------------

def compute_ou_drift(price: float, mean_price: float, theta: float) -> float:
    return float(theta) * (float(mean_price) - float(price))


# -----------------------------
# Causal adjustment
# -----------------------------

def load_causal_weights(path: str) -> Dict[str, float]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}


def apply_causal_adjustment(weights: Dict[str, float], features: Dict[str, float]) -> float:
    if not weights:
        return 0.0
    s = 0.0
    for k, w in weights.items():
        v = features.get(k)
        if v is None:
            continue
        s += float(w) * float(v)
    return float(s)


def load_weight_vector(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".npy"):
            arr = np.load(path)
            return np.asarray(arr, dtype=np.float64)
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return np.asarray(data, dtype=np.float64)
            if isinstance(data, dict):
                # allow {"weights":[...]}
                w = data.get("weights")
                if isinstance(w, list):
                    return np.asarray(w, dtype=np.float64)
    except Exception:
        return None
    return None


def _load_direction_submodel(data: Dict[str, Any], base_dir: str | None = None) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    names = data.get("feature_names")
    weights = data.get("weights")
    if not isinstance(names, list) or not isinstance(weights, list):
        return {}
    names = [str(x) for x in names]
    w = np.asarray(weights, dtype=np.float64)
    if w.size != len(names):
        return {}
    out: Dict[str, Any] = {
        "feature_names": names,
        "weights": w,
        "bias": float(_safe_float(data.get("bias"), 0.0)),
        "model_type": str(data.get("model_type") or "logistic_v1"),
        "lgbm_benchmark": data.get("lgbm_benchmark"),
    }
    mean = data.get("mean")
    std = data.get("std")
    if isinstance(mean, list) and len(mean) == len(names):
        out["mean"] = np.asarray(mean, dtype=np.float64)
    if isinstance(std, list) and len(std) == len(names):
        out["std"] = np.asarray(std, dtype=np.float64)

    # Optional LightGBM load.
    use_lgbm = str(os.environ.get("ALPHA_DIRECTION_USE_LGBM", "0")).strip().lower() in ("1", "true", "yes", "on")
    lgbm_path = None
    try:
        lgbm_meta = data.get("lgbm_benchmark") or {}
        if isinstance(lgbm_meta, dict):
            lgbm_path = str(lgbm_meta.get("model_path") or "").strip()
    except Exception:
        lgbm_path = None
    if lgbm_path and base_dir and not os.path.isabs(lgbm_path):
        lgbm_path = os.path.join(base_dir, lgbm_path)
    if use_lgbm and lgbm_path and os.path.exists(lgbm_path):
        try:
            import lightgbm as lgb  # type: ignore

            out["lgbm_booster"] = lgb.Booster(model_file=lgbm_path)
            out["lgbm_feature_names"] = list(names)
            out["model_type"] = "hybrid_lgbm"
        except Exception:
            pass
    return out


def _normalize_regime_name(name: Any) -> str:
    s = str(name or "").strip().lower()
    if s in ("mr", "mean", "mean-revert", "mean_reversion", "meanrevert"):
        return "mean_revert"
    if s in ("vol", "risk", "panic"):
        return "volatile"
    if s in ("rand", "noise", "sideways"):
        return "chop"
    return s


def load_direction_model(path: str) -> Dict[str, Any]:
    """Load a lightweight direction model from JSON.

    Supports both:
    - single model format
    - regime-split format with `by_regime`
    """
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        base_dir = os.path.dirname(path) or None
        out = _load_direction_submodel(data, base_dir=base_dir)
        if not out:
            return {}

        by_regime_raw = data.get("by_regime")
        if isinstance(by_regime_raw, dict):
            by_regime: Dict[str, Any] = {}
            for reg_key, sub in by_regime_raw.items():
                sub_model = _load_direction_submodel(sub if isinstance(sub, dict) else {}, base_dir=base_dir)
                if sub_model:
                    by_regime[_normalize_regime_name(reg_key)] = sub_model
            if by_regime:
                out["by_regime"] = by_regime
                out["default_regime"] = _normalize_regime_name(data.get("default_regime") or "chop")
        return out
    except Exception:
        return {}


def _predict_direction_single(model: Dict[str, Any], features: Dict[str, float]) -> Tuple[float, float, float]:
    names = model.get("feature_names")
    weights = model.get("weights")
    if not isinstance(names, list) or weights is None:
        return 0.5, 0.0, 0.0
    try:
        w = np.asarray(weights, dtype=np.float64)
    except Exception:
        return 0.5, 0.0, 0.0
    if w.size != len(names):
        return 0.5, 0.0, 0.0

    x = np.asarray([_safe_float(features.get(k), 0.0) for k in names], dtype=np.float64)
    lgbm_booster = model.get("lgbm_booster")
    if lgbm_booster is not None:
        try:
            p_long = float(np.asarray(lgbm_booster.predict(x.reshape(1, -1))).reshape(-1)[0])
            p_long = float(np.clip(p_long, 0.0, 1.0))
            edge = float(2.0 * p_long - 1.0)
            # Use probability-side confidence for gating (>=0.5), not edge magnitude.
            conf = float(max(p_long, 1.0 - p_long))
            return p_long, edge, conf
        except Exception:
            pass
    mean = model.get("mean")
    std = model.get("std")
    if isinstance(mean, np.ndarray) and isinstance(std, np.ndarray) and mean.size == x.size and std.size == x.size:
        x = (x - mean) / np.maximum(std, 1e-12)

    bias = float(_safe_float(model.get("bias"), 0.0))
    z = float(np.dot(w, x) + bias)
    z = float(np.clip(z, -30.0, 30.0))
    p_long = float(1.0 / (1.0 + math.exp(-z)))
    p_long = float(np.clip(p_long, 0.0, 1.0))
    edge = float(2.0 * p_long - 1.0)
    # Use probability-side confidence for gating (>=0.5), not edge magnitude.
    conf = float(max(p_long, 1.0 - p_long))
    return p_long, edge, conf


def predict_direction_logistic(model: Dict[str, Any], features: Dict[str, float]) -> Tuple[float, float, float]:
    """Predict directional probability from linear-logistic or LightGBM-hybrid model."""
    if not model:
        return 0.5, 0.0, 0.0
    by_regime = model.get("by_regime")
    if isinstance(by_regime, dict) and by_regime:
        reg = _normalize_regime_name(features.get("_regime") if isinstance(features, dict) else "")
        if not reg:
            reg = _normalize_regime_name(model.get("default_regime") or "chop")
        sub = by_regime.get(reg)
        if sub is None and reg == "mean_revert":
            sub = by_regime.get("chop")
        if sub is None:
            sub = by_regime.get(_normalize_regime_name(model.get("default_regime") or "chop"))
        if isinstance(sub, dict):
            return _predict_direction_single(sub, features)
    return _predict_direction_single(model, features)


__all__ = [
    "compute_mlofi",
    "VPINState",
    "update_vpin_state",
    "init_vpin_state_from_series",
    "KalmanCVState",
    "update_kalman_cv",
    "ParticleFilterState",
    "update_particle_filter",
    "estimate_hurst_vr",
    "GARCHState",
    "update_garch_state",
    "BayesMeanState",
    "update_bayes_mean",
    "estimate_ar1_next_return",
    "HawkesState",
    "update_hawkes_state",
    "GaussianHMMState",
    "update_gaussian_hmm",
    "compute_ou_drift",
    "load_causal_weights",
    "apply_causal_adjustment",
    "load_weight_vector",
    "load_direction_model",
    "predict_direction_logistic",
    "_annualize_mu",
    "_annualize_sigma",
]

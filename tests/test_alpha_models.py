import math
import numpy as np

from utils.alpha_models import (
    compute_mlofi,
    VPINState,
    update_vpin_state,
    KalmanCVState,
    update_kalman_cv,
    estimate_hurst_vr,
    GARCHState,
    update_garch_state,
    BayesMeanState,
    update_bayes_mean,
    estimate_ar1_next_return,
    HawkesState,
    update_hawkes_state,
    GaussianHMMState,
    update_gaussian_hmm,
)


def test_compute_mlofi_sign():
    prev = {
        "bids": [[100, 1.0], [99, 1.0]],
        "asks": [[101, 1.0], [102, 1.0]],
    }
    curr = {
        "bids": [[100.5, 2.0], [99, 1.0]],
        "asks": [[101, 1.0], [102, 1.0]],
    }
    mlofi, vec = compute_mlofi(prev, curr, levels=2)
    assert isinstance(mlofi, float)
    assert vec.shape[0] == 2
    assert mlofi > 0


def test_vpin_update_bounds():
    state = VPINState(bucket_size=10.0, window=5)
    vpin = update_vpin_state(state, delta_p=0.1, volume=5.0, sigma=0.02)
    assert 0.0 <= vpin <= 1.0


def test_kalman_cv_stable():
    state = KalmanCVState(Q=1e-6, R=1e-4)
    vel = 0.0
    for _ in range(5):
        state, vel = update_kalman_cv(state, price=100.0, dt=1.0)
    assert abs(vel) < 1e-3


def test_hurst_vr_range():
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0, 1.0, size=256)
    h = estimate_hurst_vr(rets, taus=[2, 4, 8, 16])
    assert h is not None
    assert 0.0 <= h <= 1.0


def test_garch_positive_sigma():
    state = GARCHState(omega=1e-6, alpha=0.05, beta=0.9, var=1e-6)
    state, sigma = update_garch_state(state, ret=0.01)
    assert sigma > 0.0


def test_bayes_update_variance_decreases():
    state = BayesMeanState(mu_mean=0.0, mu_var=1e-2)
    state, mu, var = update_bayes_mean(state, ret=0.01, obs_var=1e-4)
    assert var < 1e-2


def test_ar1_positive_signal():
    rets = [0.01] * 20
    pred = estimate_ar1_next_return(rets)
    assert pred > 0


def test_hawkes_boost():
    state = HawkesState(lambda_buy=0.0, lambda_sell=0.0, alpha=0.5, beta=2.0)
    state, boost = update_hawkes_state(state, event=1, dt=1.0)
    assert boost > 0


def test_hmm_online_update():
    state = GaussianHMMState(n_states=3, adapt_lr=0.05)
    state, info = update_gaussian_hmm(state, ret=0.001)
    assert isinstance(info, dict)
    assert info.get("state_label") in ("down", "chop", "up")
    assert 0.0 <= float(info.get("confidence", 0.0)) <= 1.0

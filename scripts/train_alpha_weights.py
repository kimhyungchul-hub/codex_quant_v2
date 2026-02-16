#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import product
from typing import Any, List, Tuple

import numpy as np


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("Invalid shapes for ridge fit")
    n_feat = X.shape[1]
    A = X.T @ X + lam * np.eye(n_feat)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_fit(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    lr: float = 0.05,
    n_iter: int = 600,
) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("Invalid shapes for logistic fit")
    n, p = X.shape
    w = np.zeros((p,), dtype=np.float64)
    b = 0.0
    n_f = float(max(1, n))
    for _ in range(int(max(1, n_iter))):
        z = X @ w + b
        p_hat = _sigmoid(z)
        diff = p_hat - y
        grad_w = (X.T @ diff) / n_f + float(lam) * w
        grad_b = float(np.mean(diff))
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
    return w, float(b)


def _binary_logloss(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p_hat = np.asarray(p_hat, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or p_hat.size != y_true.size:
        return 1.0
    p_hat = np.clip(p_hat, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y_true * np.log(p_hat) + (1.0 - y_true) * np.log(1.0 - p_hat)))


def _binary_bal_acc(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = (np.asarray(p_hat, dtype=np.float64).reshape(-1) >= 0.5).astype(np.float64)
    if y_true.size == 0 or pred.size != y_true.size:
        return 0.5
    pos = y_true >= 0.5
    neg = ~pos
    pos_den = float(np.sum(pos))
    neg_den = float(np.sum(neg))
    tpr = float(np.sum(pred[pos] >= 0.5) / pos_den) if pos_den > 0 else 0.5
    tnr = float(np.sum(pred[neg] < 0.5) / neg_den) if neg_den > 0 else 0.5
    return float(0.5 * (tpr + tnr))


def _binary_auc(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p_hat = np.asarray(p_hat, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or p_hat.size != y_true.size:
        return 0.5
    pos = y_true >= 0.5
    neg = ~pos
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(p_hat)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    rank_sum_pos = float(np.sum(ranks[pos]))
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _direction_actions_from_prob(p_hat: np.ndarray, min_side_prob: float = 0.5) -> np.ndarray:
    p_hat = np.asarray(p_hat, dtype=np.float64).reshape(-1)
    p_hat = np.clip(p_hat, 0.0, 1.0)
    thr = float(min(max(float(min_side_prob), 0.5), 0.99))
    actions = np.zeros((p_hat.size,), dtype=np.float64)
    actions[p_hat >= thr] = 1.0
    actions[p_hat <= (1.0 - thr)] = -1.0
    return actions


def _direction_expectancy(y_ret: np.ndarray, p_hat: np.ndarray, min_side_prob: float = 0.5) -> tuple[float, float]:
    y_ret = np.asarray(y_ret, dtype=np.float64).reshape(-1)
    if y_ret.size == 0:
        return 0.0, 0.0
    actions = _direction_actions_from_prob(p_hat, min_side_prob=min_side_prob)
    coverage = float(np.mean(np.abs(actions) > 0.0)) if actions.size > 0 else 0.0
    expectancy = float(np.mean(actions * y_ret)) if y_ret.size > 0 else 0.0
    return expectancy, coverage


def _direction_hit_rate(y_ret: np.ndarray, p_hat: np.ndarray, min_side_prob: float = 0.5) -> float:
    y_ret = np.asarray(y_ret, dtype=np.float64).reshape(-1)
    if y_ret.size == 0:
        return 0.5
    actions = _direction_actions_from_prob(p_hat, min_side_prob=min_side_prob)
    active = np.abs(actions) > 0.0
    if not np.any(active):
        return 0.5
    y_sign = np.zeros_like(y_ret, dtype=np.float64)
    y_sign[y_ret > 0.0] = 1.0
    y_sign[y_ret < 0.0] = -1.0
    valid = active & (np.abs(y_sign) > 0.0)
    if not np.any(valid):
        return 0.5
    return float(np.mean(actions[valid] == y_sign[valid]))


def _logit_from_prob(p_hat: np.ndarray) -> np.ndarray:
    p_hat = np.asarray(p_hat, dtype=np.float64).reshape(-1)
    p_hat = np.clip(p_hat, 1e-6, 1.0 - 1e-6)
    return np.log(p_hat / (1.0 - p_hat))


def _fit_platt_scaler(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    *,
    lr: float = 0.05,
    n_iter: int = 800,
    lam: float = 1e-3,
) -> dict[str, Any] | None:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p_hat = np.asarray(p_hat, dtype=np.float64).reshape(-1)
    if y_true.size < 64 or p_hat.size != y_true.size:
        return None
    pos = int(np.sum(y_true >= 0.5))
    neg = int(y_true.size - pos)
    if pos <= 0 or neg <= 0:
        return None
    z = _logit_from_prob(p_hat)
    a = 1.0
    b = 0.0
    n_f = float(max(1, y_true.size))
    for _ in range(int(max(1, n_iter))):
        p_cal = _sigmoid(a * z + b)
        diff = p_cal - y_true
        grad_a = float(np.dot(diff, z) / n_f + float(lam) * (a - 1.0))
        grad_b = float(np.mean(diff))
        a -= float(lr) * grad_a
        b -= float(lr) * grad_b
        if not np.isfinite(a) or not np.isfinite(b):
            return None
    p_post = _sigmoid(a * z + b)
    ll_before = float(_binary_logloss(y_true, p_hat))
    ll_after = float(_binary_logloss(y_true, p_post))
    try:
        min_improve = float(os.environ.get("ALPHA_DIRECTION_CALIBRATION_MIN_IMPROVE", 0.0) or 0.0)
    except Exception:
        min_improve = 0.0
    if (ll_before - ll_after) < float(min_improve):
        return None
    return {
        "type": "platt_v1",
        "a": float(a),
        "b": float(b),
        "logloss_before": float(ll_before),
        "logloss_after": float(ll_after),
        "sample_n": int(y_true.size),
    }


def _maybe_fit_probability_calibration(y_true: np.ndarray, p_hat: np.ndarray) -> dict[str, Any] | None:
    try:
        enabled = str(os.environ.get("ALPHA_DIRECTION_CALIBRATION_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        enabled = True
    if not enabled:
        return None
    try:
        min_samples = int(os.environ.get("ALPHA_DIRECTION_CALIBRATION_MIN_SAMPLES", 400) or 400)
    except Exception:
        min_samples = 400
    if int(np.asarray(y_true).shape[0]) < int(max(64, min_samples)):
        return None
    return _fit_platt_scaler(y_true, p_hat)


def _serialize_calibration(calib: Any) -> dict[str, Any] | None:
    if not isinstance(calib, dict):
        return None
    typ = str(calib.get("type") or "").strip()
    if not typ:
        return None
    out: dict[str, Any] = {"type": typ}
    for k in ("a", "b", "logloss_before", "logloss_after"):
        try:
            v = float(calib.get(k))
        except Exception:
            continue
        if np.isfinite(v):
            out[k] = float(v)
    try:
        n = int(calib.get("sample_n"))
        if n > 0:
            out["sample_n"] = int(n)
    except Exception:
        pass
    return out


def _split_time_series(X: np.ndarray, y: np.ndarray, valid_frac: float = 0.2, min_valid: int = 200):
    n = int(X.shape[0])
    n_valid = int(max(min_valid, round(n * float(valid_frac))))
    n_valid = min(max(n_valid, 64), max(64, n - 64))
    split = n - n_valid
    return X[:split], y[:split], X[split:], y[split:]


def _direction_model_score(
    *,
    exp_bps: float,
    coverage: float,
    hit: float,
    auc: float,
    bal_acc: float,
    logloss: float,
) -> float:
    """Expectancy-first scoring. Secondary metrics only tie-break."""
    try:
        exp_w = float(os.environ.get("ALPHA_DIRECTION_SCORE_EXP_WEIGHT", 1.00) or 1.00)
    except Exception:
        exp_w = 1.00
    try:
        hit_w = float(os.environ.get("ALPHA_DIRECTION_SCORE_HIT_WEIGHT", 0.04) or 0.04)
    except Exception:
        hit_w = 0.04
    try:
        auc_w = float(os.environ.get("ALPHA_DIRECTION_SCORE_AUC_WEIGHT", 0.02) or 0.02)
    except Exception:
        auc_w = 0.02
    try:
        bal_w = float(os.environ.get("ALPHA_DIRECTION_SCORE_BALACC_WEIGHT", 0.02) or 0.02)
    except Exception:
        bal_w = 0.02
    try:
        logloss_pen = float(os.environ.get("ALPHA_DIRECTION_SCORE_LOGLOSS_PENALTY", 0.04) or 0.04)
    except Exception:
        logloss_pen = 0.04
    try:
        cov_target = float(os.environ.get("ALPHA_DIRECTION_SCORE_COVERAGE_TARGET", 0.35) or 0.35)
    except Exception:
        cov_target = 0.35
    try:
        cov_penalty = float(os.environ.get("ALPHA_DIRECTION_SCORE_COVERAGE_PENALTY_BPS", 8.0) or 8.0)
    except Exception:
        cov_penalty = 8.0
    try:
        min_exp_bps = float(os.environ.get("ALPHA_DIRECTION_SCORE_MIN_EXPECTANCY_BPS", -2.0) or -2.0)
    except Exception:
        min_exp_bps = -2.0
    try:
        min_cov = float(os.environ.get("ALPHA_DIRECTION_SCORE_MIN_COVERAGE", 0.12) or 0.12)
    except Exception:
        min_cov = 0.12

    score = (
        float(exp_w) * float(exp_bps)
        + float(hit_w) * (float(hit) - 0.5) * 100.0
        + float(auc_w) * (float(auc) - 0.5) * 100.0
        + float(bal_w) * (float(bal_acc) - 0.5) * 100.0
        - float(logloss_pen) * float(logloss)
    )
    cov_gap = float(max(0.0, float(cov_target) - float(coverage)))
    score -= float(cov_penalty) * float(cov_gap)
    # Hard-penalize near-zero-coverage regimes to avoid "paper accuracy" models.
    if float(coverage) < float(min_cov):
        score -= float(20.0 * (float(min_cov) - float(coverage)))
    # Expectancy floor remains the dominant acceptance criterion.
    if float(exp_bps) < float(min_exp_bps):
        score -= float(3.0 * (float(min_exp_bps) - float(exp_bps)))
    return float(score)


def _fit_logistic_tuned(
    X: np.ndarray,
    y_bin: np.ndarray,
    base_lam: float,
    y_ret: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    X_tr, y_tr, X_va, y_va = _split_time_series(X, y_bin, valid_frac=0.2, min_valid=200)
    y_ret_va = None
    if y_ret is not None:
        y_ret = np.asarray(y_ret, dtype=np.float64).reshape(-1)
        if y_ret.shape[0] == X.shape[0]:
            _, _, _, y_ret_va = _split_time_series(X, y_ret, valid_frac=0.2, min_valid=200)

    x_mean = np.mean(X_tr, axis=0)
    x_std = np.std(X_tr, axis=0)
    x_std = np.where(np.isfinite(x_std), x_std, 0.0)
    x_std = np.maximum(x_std, 1e-8)
    X_tr_n = (X_tr - x_mean) / x_std
    X_va_n = (X_va - x_mean) / x_std

    try:
        min_side_prob = float(os.environ.get("ALPHA_DIRECTION_SCORE_MIN_SIDE_PROB", 0.5) or 0.5)
    except Exception:
        min_side_prob = 0.5

    lam_grid = sorted(set([base_lam * s for s in (0.1, 0.3, 1.0, 3.0, 10.0)]))
    lr_grid = [0.02, 0.05, 0.1]
    iter_grid = [400, 800, 1200]
    trials: list[dict[str, Any]] = []
    best = None
    best_score = -1e9
    for lam, lr, n_iter in product(lam_grid, lr_grid, iter_grid):
        w, b = _logistic_fit(X_tr_n, y_tr, lam=float(lam), lr=float(lr), n_iter=int(n_iter))
        p_va = _sigmoid(X_va_n @ w + b)
        acc = float(np.mean((p_va >= 0.5).astype(np.float64) == y_va))
        bal_acc = _binary_bal_acc(y_va, p_va)
        auc = _binary_auc(y_va, p_va)
        logloss = _binary_logloss(y_va, p_va)
        exp_net = 0.0
        hit = 0.5
        coverage = 0.0
        if y_ret_va is not None:
            exp_net, coverage = _direction_expectancy(y_ret_va, p_va, min_side_prob=min_side_prob)
            hit = _direction_hit_rate(y_ret_va, p_va, min_side_prob=min_side_prob)
        exp_bps = float(exp_net * 10_000.0)
        score = _direction_model_score(
            exp_bps=float(exp_bps),
            coverage=float(coverage),
            hit=float(hit),
            auc=float(auc),
            bal_acc=float(bal_acc),
            logloss=float(logloss),
        )
        row = {
            "lam": float(lam),
            "lr": float(lr),
            "n_iter": int(n_iter),
            "valid_acc": float(acc),
            "valid_bal_acc": float(bal_acc),
            "valid_auc": float(auc),
            "valid_logloss": float(logloss),
            "valid_expectancy": float(exp_net),
            "valid_expectancy_bps": float(exp_bps),
            "valid_hit": float(hit),
            "valid_coverage": float(coverage),
            "score": float(score),
        }
        trials.append(row)
        if score > best_score:
            best_score = float(score)
            best = {
                "w": w,
                "b": float(b),
                "x_mean": x_mean,
                "x_std": x_std,
                "params": {"lam": float(lam), "lr": float(lr), "n_iter": int(n_iter)},
                "metrics": {
                    "valid_acc": float(acc),
                    "valid_bal_acc": float(bal_acc),
                    "valid_auc": float(auc),
                    "valid_logloss": float(logloss),
                    "valid_expectancy": float(exp_net),
                    "valid_expectancy_bps": float(exp_bps),
                    "valid_hit": float(hit),
                    "valid_coverage": float(coverage),
                    "score": float(score),
                },
            }
    if best is None:
        w, b = _logistic_fit(X_tr_n, y_tr, lam=float(base_lam), lr=0.05, n_iter=800)
        p_va = _sigmoid(X_va_n @ w + b)
        exp_net = 0.0
        hit = 0.5
        coverage = 0.0
        if y_ret_va is not None:
            exp_net, coverage = _direction_expectancy(y_ret_va, p_va, min_side_prob=min_side_prob)
            hit = _direction_hit_rate(y_ret_va, p_va, min_side_prob=min_side_prob)
        best = {
            "w": w,
            "b": float(b),
            "x_mean": x_mean,
            "x_std": x_std,
            "params": {"lam": float(base_lam), "lr": 0.05, "n_iter": 800},
            "metrics": {
                "valid_acc": float(np.mean((p_va >= 0.5).astype(np.float64) == y_va)),
                "valid_bal_acc": float(_binary_bal_acc(y_va, p_va)),
                "valid_auc": float(_binary_auc(y_va, p_va)),
                "valid_logloss": float(_binary_logloss(y_va, p_va)),
                "valid_expectancy": float(exp_net),
                "valid_expectancy_bps": float(exp_net * 10_000.0),
                "valid_hit": float(hit),
                "valid_coverage": float(coverage),
                "score": 0.0,
            },
        }
    calibration = None
    try:
        p_va_best = _sigmoid(X_va_n @ np.asarray(best["w"], dtype=np.float64) + float(best["b"]))
        calibration = _maybe_fit_probability_calibration(y_va, p_va_best)
    except Exception:
        calibration = None
    # Refit with tuned params on full dataset for runtime model.
    x_mean_full = np.mean(X, axis=0)
    x_std_full = np.std(X, axis=0)
    x_std_full = np.where(np.isfinite(x_std_full), x_std_full, 0.0)
    x_std_full = np.maximum(x_std_full, 1e-8)
    X_full_n = (X - x_mean_full) / x_std_full
    p = best["params"]
    w_full, b_full = _logistic_fit(
        X_full_n,
        y_bin,
        lam=float(p["lam"]),
        lr=float(p["lr"]),
        n_iter=int(p["n_iter"]),
    )
    p_full = _sigmoid(X_full_n @ w_full + b_full)
    best["w"] = w_full
    best["b"] = float(b_full)
    best["x_mean"] = x_mean_full
    best["x_std"] = x_std_full
    train_expectancy = 0.0
    train_hit = 0.5
    train_coverage = 0.0
    if y_ret is not None and np.asarray(y_ret).shape[0] == X.shape[0]:
        train_expectancy, train_coverage = _direction_expectancy(np.asarray(y_ret, dtype=np.float64), p_full, min_side_prob=min_side_prob)
        train_hit = _direction_hit_rate(np.asarray(y_ret, dtype=np.float64), p_full, min_side_prob=min_side_prob)
    best["train_metrics"] = {
        "train_acc": float(np.mean((p_full >= 0.5).astype(np.float64) == y_bin)),
        "train_bal_acc": float(_binary_bal_acc(y_bin, p_full)),
        "train_auc": float(_binary_auc(y_bin, p_full)),
        "train_logloss": float(_binary_logloss(y_bin, p_full)),
        "train_expectancy": float(train_expectancy),
        "train_expectancy_bps": float(train_expectancy * 10_000.0),
        "train_hit": float(train_hit),
        "train_coverage": float(train_coverage),
    }
    best["calibration"] = calibration
    return best, trials


def _fit_lightgbm_tuned(
    X: np.ndarray,
    y_bin: np.ndarray,
    y_ret: np.ndarray | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    try:
        import lightgbm as lgb
    except Exception:
        return None, []
    X_tr, y_tr, X_va, y_va = _split_time_series(X, y_bin, valid_frac=0.2, min_valid=200)
    y_ret_va = None
    if y_ret is not None:
        y_ret = np.asarray(y_ret, dtype=np.float64).reshape(-1)
        if y_ret.shape[0] == X.shape[0]:
            _, _, _, y_ret_va = _split_time_series(X, y_ret, valid_frac=0.2, min_valid=200)
    trials: list[dict[str, Any]] = []
    best = None
    best_score = -1e9

    try:
        min_side_prob = float(os.environ.get("ALPHA_DIRECTION_SCORE_MIN_SIDE_PROB", 0.5) or 0.5)
    except Exception:
        min_side_prob = 0.5

    leaves_grid = [31, 63]
    lr_grid = [0.03, 0.08]
    min_leaf_grid = [20, 60]
    ff_grid = [0.7, 1.0]
    for leaves, lr, min_leaf, ff in product(leaves_grid, lr_grid, min_leaf_grid, ff_grid):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": float(lr),
            "num_leaves": int(leaves),
            "min_data_in_leaf": int(min_leaf),
            "feature_fraction": float(ff),
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": 42,
            "num_threads": 1,
        }
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=260,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
        acc = float(np.mean((p_va >= 0.5).astype(np.float64) == y_va))
        bal_acc = _binary_bal_acc(y_va, p_va)
        auc = _binary_auc(y_va, p_va)
        logloss = _binary_logloss(y_va, p_va)
        exp_net = 0.0
        hit = 0.5
        coverage = 0.0
        if y_ret_va is not None:
            exp_net, coverage = _direction_expectancy(y_ret_va, p_va, min_side_prob=min_side_prob)
            hit = _direction_hit_rate(y_ret_va, p_va, min_side_prob=min_side_prob)
        exp_bps = float(exp_net * 10_000.0)
        score = _direction_model_score(
            exp_bps=float(exp_bps),
            coverage=float(coverage),
            hit=float(hit),
            auc=float(auc),
            bal_acc=float(bal_acc),
            logloss=float(logloss),
        )
        row = {
            "num_leaves": int(leaves),
            "learning_rate": float(lr),
            "min_data_in_leaf": int(min_leaf),
            "feature_fraction": float(ff),
            "best_iteration": int(booster.best_iteration or 0),
            "valid_acc": float(acc),
            "valid_bal_acc": float(bal_acc),
            "valid_auc": float(auc),
            "valid_logloss": float(logloss),
            "valid_expectancy": float(exp_net),
            "valid_expectancy_bps": float(exp_bps),
            "valid_hit": float(hit),
            "valid_coverage": float(coverage),
            "score": float(score),
        }
        trials.append(row)
        if score > best_score:
            best_score = float(score)
            best = {
                "booster": booster,
                "params": dict(params),
                "metrics": {
                    "valid_acc": float(acc),
                    "valid_bal_acc": float(bal_acc),
                    "valid_auc": float(auc),
                    "valid_logloss": float(logloss),
                    "valid_expectancy": float(exp_net),
                    "valid_expectancy_bps": float(exp_bps),
                    "valid_hit": float(hit),
                    "valid_coverage": float(coverage),
                    "score": float(score),
                },
                "best_iteration": int(booster.best_iteration or 0),
            }
    if best is None:
        return None, trials
    calibration = None
    try:
        p_va_best = np.asarray(best["booster"].predict(X_va, num_iteration=best.get("best_iteration") or None), dtype=np.float64).reshape(-1)
        calibration = _maybe_fit_probability_calibration(y_va, p_va_best)
    except Exception:
        calibration = None
    # Refit best params on full sample.
    best_params = dict(best["params"])
    d_all = lgb.Dataset(X, label=y_bin)
    n_round = int(max(60, min(500, best["best_iteration"] or 200)))
    booster_full = lgb.train(best_params, d_all, num_boost_round=n_round)
    p_all = booster_full.predict(X)
    train_expectancy = 0.0
    train_hit = 0.5
    train_coverage = 0.0
    if y_ret is not None and np.asarray(y_ret).shape[0] == X.shape[0]:
        train_expectancy, train_coverage = _direction_expectancy(np.asarray(y_ret, dtype=np.float64), p_all, min_side_prob=min_side_prob)
        train_hit = _direction_hit_rate(np.asarray(y_ret, dtype=np.float64), p_all, min_side_prob=min_side_prob)
    best["booster"] = booster_full
    best["train_metrics"] = {
        "train_acc": float(np.mean((p_all >= 0.5).astype(np.float64) == y_bin)),
        "train_bal_acc": float(_binary_bal_acc(y_bin, p_all)),
        "train_auc": float(_binary_auc(y_bin, p_all)),
        "train_logloss": float(_binary_logloss(y_bin, p_all)),
        "train_expectancy": float(train_expectancy),
        "train_expectancy_bps": float(train_expectancy * 10_000.0),
        "train_hit": float(train_hit),
        "train_coverage": float(train_coverage),
    }
    best["best_iteration"] = int(n_round)
    best["calibration"] = calibration
    return best, trials


def train_mlofi_weights(npz_path: str, out_json: str, out_npy: str, lam: float = 1e-3) -> Tuple[int, int]:
    if not os.path.exists(npz_path):
        return 0, 0
    data = np.load(npz_path)
    X = data.get("X")
    y = data.get("y")
    if X is None or y is None:
        return 0, 0
    if len(X) < 10:
        return int(len(X)), 0
    w = _ridge_fit(X, y, lam)
    os.makedirs(os.path.dirname(out_json), exist_ok=True) if os.path.dirname(out_json) else None
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"weights": [float(x) for x in w.tolist()]}, f)
    if out_npy:
        np.save(out_npy, w)
    return int(len(X)), int(len(w))


def train_causal_weights(npz_path: str, out_json: str, lam: float = 1e-3) -> Tuple[int, int]:
    if not os.path.exists(npz_path):
        return 0, 0
    data = np.load(npz_path, allow_pickle=True)
    X = data.get("X")
    y = data.get("y")
    names = data.get("feature_names")
    if X is None or y is None:
        return 0, 0
    if len(X) < 10:
        return int(len(X)), 0
    w = _ridge_fit(X, y, lam)
    if names is None:
        names = [f"f{i}" for i in range(len(w))]
    names = [str(n) for n in list(names)]
    out = {names[i]: float(w[i]) for i in range(min(len(names), len(w)))}
    os.makedirs(os.path.dirname(out_json), exist_ok=True) if os.path.dirname(out_json) else None
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return int(len(X)), int(len(w))


def _derive_regimes_from_features(
    X: np.ndarray,
    feature_names: list[str],
    hurst_low: float = 0.45,
    hurst_high: float = 0.55,
    vpin_extreme: float = 0.85,
) -> np.ndarray:
    n = int(X.shape[0]) if X.ndim == 2 else 0
    if n <= 0:
        return np.asarray([], dtype=object)
    idx_h = -1
    idx_v = -1
    try:
        idx_h = feature_names.index("hurst")
    except Exception:
        idx_h = -1
    try:
        idx_v = feature_names.index("vpin")
    except Exception:
        idx_v = -1
    out = np.empty((n,), dtype=object)
    for i in range(n):
        h = float(X[i, idx_h]) if idx_h >= 0 else 0.5
        v = float(X[i, idx_v]) if idx_v >= 0 else 0.0
        if v >= float(vpin_extreme):
            out[i] = "volatile"
        elif h > float(hurst_high):
            out[i] = "trend"
        elif h < float(hurst_low):
            out[i] = "mean_revert"
        else:
            out[i] = "chop"
    return out


def _prepare_direction_targets(data: Any, y_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y_raw = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    n = int(y_raw.shape[0])
    y_target = np.asarray(y_raw, dtype=np.float64).copy()
    label_source = str(os.environ.get("ALPHA_DIRECTION_LABEL_SOURCE", "auto") or "auto").strip()
    label_source_norm = label_source.lower()
    source_used = "y"
    try:
        min_active_ratio = float(os.environ.get("ALPHA_DIRECTION_LABEL_MIN_ACTIVE_RATIO", 0.03) or 0.03)
    except Exception:
        min_active_ratio = 0.03
    min_active_ratio = float(max(0.0, min(0.90, min_active_ratio)))
    selected_active_ratio = 1.0
    candidate_active_ratios: dict[str, float] = {}

    if label_source_norm in ("", "auto"):
        candidate_keys = ("y_hstar_net", "y_net_hstar", "y_net", "y_fee_adj", "y_hstar", "y")
    else:
        candidate_keys = (label_source,)

    for key in candidate_keys:
        if str(key) == "y":
            arr = y_raw
        else:
            arr = data.get(str(key)) if hasattr(data, "get") else None
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if int(arr.shape[0]) != int(n):
            continue
        active_ratio = float(np.mean(np.isfinite(arr) & (np.abs(arr) > 1e-12)))
        candidate_active_ratios[str(key)] = float(active_ratio)
        if label_source_norm in ("", "auto") and str(key) != "y":
            if active_ratio < min_active_ratio:
                continue
        y_target = np.asarray(arr, dtype=np.float64)
        source_used = str(key)
        selected_active_ratio = float(active_ratio)
        break

    # If only raw return label exists, approximate net label with configurable transaction costs.
    if source_used == "y":
        try:
            fee_bps = float(os.environ.get("ALPHA_DIRECTION_LABEL_FEE_BPS", 0.0) or 0.0)
        except Exception:
            fee_bps = 0.0
        try:
            slippage_bps = float(os.environ.get("ALPHA_DIRECTION_LABEL_SLIPPAGE_BPS", 0.0) or 0.0)
        except Exception:
            slippage_bps = 0.0
        try:
            extra_cost_bps = float(os.environ.get("ALPHA_DIRECTION_LABEL_COST_BPS", 0.0) or 0.0)
        except Exception:
            extra_cost_bps = 0.0
        total_cost_ret = max(0.0, (fee_bps + slippage_bps + extra_cost_bps) / 10_000.0)
        if total_cost_ret > 0:
            y_target = np.sign(y_target) * np.maximum(np.abs(y_target) - total_cost_ret, 0.0)
        selected_active_ratio = float(np.mean(np.isfinite(y_target) & (np.abs(y_target) > 1e-12)))
    else:
        total_cost_ret = 0.0

    keep = np.isfinite(y_target)
    dropped_contamination = 0
    blocked_tokens_raw = str(
        os.environ.get(
            "ALPHA_DIRECTION_LABEL_EXCLUDE_TOKENS",
            "external,manual_cleanup,rebal,rebal_exit,cleanup,manual_close",
        )
        or ""
    )
    blocked_tokens = [tok.strip().lower() for tok in blocked_tokens_raw.split(",") if tok.strip()]
    if blocked_tokens:
        contam_keys = (
            "exit_kind",
            "exit_reason",
            "close_cause",
            "close_reason",
            "action",
        )
        contam_mask = np.zeros((n,), dtype=bool)
        for key in contam_keys:
            arr = data.get(key) if hasattr(data, "get") else None
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=object).reshape(-1)
            if int(arr.shape[0]) != int(n):
                continue
            for i in range(n):
                if contam_mask[i]:
                    continue
                s = str(arr[i] or "").strip().lower()
                if not s:
                    continue
                if any(tok in s for tok in blocked_tokens):
                    contam_mask[i] = True
        if np.any(contam_mask):
            dropped_contamination = int(np.sum(contam_mask))
            keep &= ~contam_mask

    try:
        neutral_bps = float(os.environ.get("ALPHA_DIRECTION_NEUTRAL_BAND_BPS", 0.0) or 0.0)
    except Exception:
        neutral_bps = 0.0
    neutral_ret = max(0.0, neutral_bps / 10_000.0)
    dropped_neutral = 0
    if neutral_ret > 0:
        neutral_mask = np.abs(y_target) < neutral_ret
        dropped_neutral = int(np.sum(neutral_mask & keep))
        keep &= ~neutral_mask

    meta = {
        "source_requested": label_source,
        "source_used": source_used,
        "cost_adjusted_ret": float(total_cost_ret),
        "neutral_band_ret": float(neutral_ret),
        "selected_active_ratio": float(selected_active_ratio),
        "min_active_ratio": float(min_active_ratio),
        "candidate_active_ratios": candidate_active_ratios,
        "dropped_contamination": int(dropped_contamination),
        "dropped_neutral": int(dropped_neutral),
    }
    return y_target, keep, meta


def train_mu_direction_model(
    npz_path: str,
    out_json: str,
    lam: float = 1e-3,
    min_samples: int = 500,
) -> tuple[int, int, float]:
    """Train a tuned logistic direction model (+ optional LightGBM benchmark)."""
    if not os.path.exists(npz_path):
        return 0, 0, 0.0
    data = np.load(npz_path, allow_pickle=True)
    X = data.get("X")
    y = data.get("y")
    names = data.get("feature_names")
    if X is None or y is None:
        return 0, 0, 0.0
    X = np.asarray(X, dtype=np.float64)
    y_raw = np.asarray(y, dtype=np.float64)
    n_raw = int(X.shape[0]) if X.ndim == 2 else 0
    if n_raw < int(max(32, min_samples)):
        return n_raw, 0, 0.0
    y_target, keep_mask, label_meta = _prepare_direction_targets(data, y_raw)
    if keep_mask.shape[0] != n_raw:
        keep_mask = np.ones((n_raw,), dtype=bool)
    X = X[keep_mask]
    y = y_target[keep_mask]
    n = int(X.shape[0]) if X.ndim == 2 else 0
    if n < int(max(32, min_samples)):
        return n, 0, 0.0
    if names is None:
        names = [f"f{i}" for i in range(X.shape[1])]
    names = [str(nm) for nm in list(names)]
    if len(names) != X.shape[1]:
        names = [f"f{i}" for i in range(X.shape[1])]

    # Predict h*-net direction if available (fallback: next return direction).
    y_bin = (y > 0.0).astype(np.float64)
    regimes_raw = data.get("regimes")
    if regimes_raw is not None:
        regimes = np.asarray(regimes_raw, dtype=object).reshape(-1)
        if int(regimes.shape[0]) == int(n_raw):
            regimes = regimes[keep_mask]
        elif int(regimes.shape[0]) != int(n):
            regimes = _derive_regimes_from_features(X, names)
    else:
        regimes = _derive_regimes_from_features(X, names)
    best_log, log_trials = _fit_logistic_tuned(X, y_bin, base_lam=float(lam), y_ret=y)
    w = np.asarray(best_log.get("w"), dtype=np.float64)
    b = float(best_log.get("b"))
    x_mean = np.asarray(best_log.get("x_mean"), dtype=np.float64)
    x_std = np.asarray(best_log.get("x_std"), dtype=np.float64)
    train_metrics = dict(best_log.get("train_metrics") or {})
    valid_metrics = dict(best_log.get("metrics") or {})
    params_log = dict(best_log.get("params") or {})
    acc = float(train_metrics.get("train_acc", 0.0))
    try:
        regime_min_samples = int(os.environ.get("ALPHA_DIRECTION_REGIME_MIN_SAMPLES", 250) or 250)
    except Exception:
        regime_min_samples = 250
    out_dir = os.path.dirname(out_json) or "."
    out_dir_abs = os.path.abspath(out_dir)
    lgbm_enabled = str(os.environ.get("ALPHA_DIRECTION_LGBM_TUNE_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")

    by_regime: dict[str, dict[str, Any]] = {}
    regime_stats: dict[str, Any] = {}
    for reg in ("trend", "chop", "mean_revert", "volatile"):
        idx = np.where(np.asarray([str(x) for x in regimes]) == reg)[0]
        n_reg = int(idx.size)
        regime_stats[reg] = {"samples": n_reg}
        if n_reg < int(max(64, regime_min_samples)):
            regime_stats[reg]["trained"] = False
            continue
        Xr = X[idx]
        yr = y_bin[idx]
        yret_r = y[idx]
        try:
            best_reg, reg_log_trials = _fit_logistic_tuned(Xr, yr, base_lam=float(lam), y_ret=yret_r)
            wr = np.asarray(best_reg.get("w"), dtype=np.float64)
            br = float(best_reg.get("b"))
            mr = np.asarray(best_reg.get("x_mean"), dtype=np.float64)
            sr = np.asarray(best_reg.get("x_std"), dtype=np.float64)
            regime_payload = {
                "model_type": "logistic_v1",
                "feature_names": names,
                "weights": [float(v) for v in wr.tolist()],
                "bias": float(br),
                "mean": [float(v) for v in mr.tolist()],
                "std": [float(v) for v in sr.tolist()],
                "train_samples": int(n_reg),
                "train_metrics": dict(best_reg.get("train_metrics") or {}),
                "valid_metrics": dict(best_reg.get("metrics") or {}),
                "tuned_params": dict(best_reg.get("params") or {}),
                "calibration": _serialize_calibration(best_reg.get("calibration")),
            }
            reg_lgbm_best = None
            reg_lgbm_trials: list[dict[str, Any]] = []
            if lgbm_enabled:
                reg_lgbm_best, reg_lgbm_trials = _fit_lightgbm_tuned(Xr, yr, y_ret=yret_r)
                if reg_lgbm_best is not None:
                    reg_model_file = f"mu_direction_model_{reg}.lgb.txt"
                    reg_model_path = os.path.abspath(os.path.join(out_dir_abs, reg_model_file))
                    try:
                        os.makedirs(os.path.dirname(reg_model_path), exist_ok=True)
                        reg_lgbm_best["booster"].save_model(reg_model_path)
                        regime_payload["lgbm_benchmark"] = {
                            "enabled": True,
                            "available": True,
                            "model_path": reg_model_path,
                            "valid_auc": float((reg_lgbm_best.get("metrics") or {}).get("valid_auc", 0.0)),
                            "valid_bal_acc": float((reg_lgbm_best.get("metrics") or {}).get("valid_bal_acc", 0.0)),
                            "valid_expectancy_bps": float((reg_lgbm_best.get("metrics") or {}).get("valid_expectancy_bps", 0.0)),
                            "train_metrics": dict(reg_lgbm_best.get("train_metrics") or {}),
                            "calibration": _serialize_calibration(reg_lgbm_best.get("calibration")),
                            "params": {
                                "num_leaves": int((reg_lgbm_best.get("params") or {}).get("num_leaves", 0)),
                                "learning_rate": float((reg_lgbm_best.get("params") or {}).get("learning_rate", 0.0)),
                                "min_data_in_leaf": int((reg_lgbm_best.get("params") or {}).get("min_data_in_leaf", 0)),
                                "feature_fraction": float((reg_lgbm_best.get("params") or {}).get("feature_fraction", 0.0)),
                            },
                        }
                    except Exception:
                        pass
            by_regime[reg] = regime_payload
            regime_stats[reg]["trained"] = True
            regime_stats[reg]["valid_auc"] = float((best_reg.get("metrics") or {}).get("valid_auc", 0.5))
            regime_stats[reg]["valid_bal_acc"] = float((best_reg.get("metrics") or {}).get("valid_bal_acc", 0.5))
            regime_stats[reg]["valid_expectancy_bps"] = float((best_reg.get("metrics") or {}).get("valid_expectancy_bps", 0.0))
            regime_stats[reg]["valid_hit"] = float((best_reg.get("metrics") or {}).get("valid_hit", 0.5))
            regime_stats[reg]["logistic_trials"] = int(len(reg_log_trials))
            regime_stats[reg]["lgbm_trials"] = int(len(reg_lgbm_trials))
            if reg_lgbm_best is not None:
                regime_stats[reg]["lgbm_valid_expectancy_bps"] = float((reg_lgbm_best.get("metrics") or {}).get("valid_expectancy_bps", 0.0))
                regime_stats[reg]["lgbm_valid_hit"] = float((reg_lgbm_best.get("metrics") or {}).get("valid_hit", 0.5))
        except Exception:
            regime_stats[reg]["trained"] = False

    lgbm_best = None
    lgbm_trials: list[dict[str, Any]] = []
    lgbm_model_path = str(os.environ.get("ALPHA_DIRECTION_LGBM_MODEL_PATH", os.path.join(out_dir_abs, "mu_direction_model.lgb.txt"))).strip()
    if lgbm_enabled:
        lgbm_best, lgbm_trials = _fit_lightgbm_tuned(X, y_bin, y_ret=y)
        if lgbm_best is not None and lgbm_model_path:
            lgbm_model_path = os.path.abspath(lgbm_model_path)
            lgb_out_dir = os.path.dirname(lgbm_model_path)
            if lgb_out_dir:
                os.makedirs(lgb_out_dir, exist_ok=True)
            try:
                lgbm_best["booster"].save_model(lgbm_model_path)
            except Exception:
                lgbm_model_path = ""

    report_path = str(os.environ.get("ALPHA_DIRECTION_TUNE_REPORT", "state/mu_direction_tuning_report.json")).strip()
    if report_path:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        report = {
            "timestamp": int(time.time()),
            "samples": int(n),
            "samples_raw": int(n_raw),
            "samples_filtered": int(max(0, n_raw - n)),
            "label_meta": label_meta,
            "feature_names": names,
            "regime_stats": regime_stats,
            "logistic_best": {
                "params": params_log,
                "valid_metrics": valid_metrics,
                "train_metrics": train_metrics,
                "calibration": _serialize_calibration(best_log.get("calibration")),
            },
            "lightgbm_best": None,
            "logistic_trials": sorted(log_trials, key=lambda r: float(r.get("score", -1e9)), reverse=True)[:12],
            "lightgbm_trials": sorted(lgbm_trials, key=lambda r: float(r.get("score", -1e9)), reverse=True)[:12],
        }
        if lgbm_best is not None:
            report["lightgbm_best"] = {
                "params": {
                    "num_leaves": int(lgbm_best["params"].get("num_leaves", 0)),
                    "learning_rate": float(lgbm_best["params"].get("learning_rate", 0.0)),
                    "min_data_in_leaf": int(lgbm_best["params"].get("min_data_in_leaf", 0)),
                    "feature_fraction": float(lgbm_best["params"].get("feature_fraction", 0.0)),
                },
                "valid_metrics": dict(lgbm_best.get("metrics") or {}),
                "train_metrics": dict(lgbm_best.get("train_metrics") or {}),
                "best_iteration": int(lgbm_best.get("best_iteration") or 0),
                "model_path": lgbm_model_path,
                "calibration": _serialize_calibration(lgbm_best.get("calibration")),
            }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")

    out = {
        "model_type": "logistic_v1",
        "feature_names": names,
        "weights": [float(v) for v in w.tolist()],
        "bias": float(b),
        "mean": [float(v) for v in x_mean.tolist()],
        "std": [float(v) for v in x_std.tolist()],
        "train_samples": int(n),
        "train_samples_raw": int(n_raw),
        "label_meta": label_meta,
        "train_accuracy": float(acc),
        "train_bal_acc": float(train_metrics.get("train_bal_acc", 0.0)),
        "train_auc": float(train_metrics.get("train_auc", 0.5)),
        "train_expectancy_bps": float(train_metrics.get("train_expectancy_bps", 0.0)),
        "train_hit": float(train_metrics.get("train_hit", 0.5)),
        "valid_accuracy": float(valid_metrics.get("valid_acc", 0.0)),
        "valid_bal_acc": float(valid_metrics.get("valid_bal_acc", 0.0)),
        "valid_auc": float(valid_metrics.get("valid_auc", 0.5)),
        "valid_logloss": float(valid_metrics.get("valid_logloss", 1.0)),
        "valid_expectancy_bps": float(valid_metrics.get("valid_expectancy_bps", 0.0)),
        "valid_hit": float(valid_metrics.get("valid_hit", 0.5)),
        "tuned_params": params_log,
        "calibration": _serialize_calibration(best_log.get("calibration")),
        "default_regime": "chop",
        "by_regime": by_regime,
        "regime_stats": regime_stats,
        "lgbm_benchmark": {
            "enabled": bool(lgbm_enabled),
            "available": bool(lgbm_best is not None),
            "valid_auc": float((lgbm_best or {}).get("metrics", {}).get("valid_auc", 0.0)) if lgbm_best is not None else None,
            "valid_bal_acc": float((lgbm_best or {}).get("metrics", {}).get("valid_bal_acc", 0.0)) if lgbm_best is not None else None,
            "model_path": lgbm_model_path if lgbm_best is not None else None,
            "calibration": _serialize_calibration((lgbm_best or {}).get("calibration")) if lgbm_best is not None else None,
        },
        "trained_at": int(time.time()),
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True) if os.path.dirname(out_json) else None
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True)
    return int(n), int(len(w)), float(acc)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mlofi-npz", default=os.environ.get("ALPHA_MLOFI_TRAIN_PATH", "state/mlofi_train_samples.npz"))
    p.add_argument("--causal-npz", default=os.environ.get("ALPHA_CAUSAL_TRAIN_PATH", "state/causal_train_samples.npz"))
    p.add_argument("--mlofi-out", default=os.environ.get("MLOFI_WEIGHT_PATH", "state/mlofi_weights.json"))
    p.add_argument("--mlofi-out-npy", default=os.environ.get("MLOFI_WEIGHT_NPY", "state/mlofi_weights.npy"))
    p.add_argument("--causal-out", default=os.environ.get("CAUSAL_WEIGHTS_PATH", "state/causal_weights.json"))
    p.add_argument("--dir-out", default=os.environ.get("ALPHA_DIRECTION_MODEL_PATH", "state/mu_direction_model.json"))
    p.add_argument("--dir-min-samples", type=int, default=int(os.environ.get("ALPHA_DIRECTION_MIN_SAMPLES", 500)))
    p.add_argument("--ridge", type=float, default=float(os.environ.get("ALPHA_WEIGHT_RIDGE_LAMBDA", 1e-3)))
    args = p.parse_args()

    n_mlofi, n_w_m = train_mlofi_weights(args.mlofi_npz, args.mlofi_out, args.mlofi_out_npy, args.ridge)
    n_causal, n_w_c = train_causal_weights(args.causal_npz, args.causal_out, args.ridge)
    n_dir, n_w_d, acc_d = train_mu_direction_model(args.causal_npz, args.dir_out, args.ridge, args.dir_min_samples)

    print(f"[alpha_weights] mlofi_samples={n_mlofi} mlofi_weights={n_w_m} -> {args.mlofi_out}")
    print(f"[alpha_weights] causal_samples={n_causal} causal_weights={n_w_c} -> {args.causal_out}")
    print(f"[alpha_weights] dir_samples={n_dir} dir_weights={n_w_d} dir_acc={acc_d:.4f} -> {args.dir_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

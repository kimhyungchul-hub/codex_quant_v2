from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from engines.probability_methods import (
    _approx_p_pos_and_ev_hold,
    _approx_cvar_normal,
    _prob_max_geq,
    _prob_min_leq,
)


def _approx_unified_score_from_cumulative(
    ev: float,
    cvar: float,
    tau_sec: float,
    rho: float,
    lambda_param: float,
) -> float:
    tau = float(max(1e-6, float(tau_sec)))
    util_rate = (float(ev) - float(lambda_param) * abs(float(cvar))) / tau
    rho_f = float(rho) if rho is not None else 0.0
    if rho_f <= 0.0:
        return float(util_rate)
    factor = (1.0 - math.exp(-rho_f * tau)) / max(1e-9, (rho_f * tau))
    return float((util_rate - rho_f) * factor)


def simulate_exit_policy_rollforward(
    price_paths: np.ndarray,  # (n_paths, n_steps) 1초 단위 가격 경로
    s0: float,
    mu: float,
    sigma: float,
    leverage: float,
    fee_roundtrip: float,
    exec_oneway: float,
    impact_cost: float,
    regime: str,
    decision_dt_sec: int = 5,
    horizon_sec: int = 1800,
    min_hold_sec: int = 180,
    flip_confirm_ticks: int = 3,
    hold_bad_ticks: int = 3,
    p_pos_floor_enter: float = 0.52,
    p_pos_floor_hold: float = 0.50,
    p_sl_enter_ceiling: float = 0.20,
    p_sl_hold_ceiling: float = 0.25,
    p_sl_emergency: float = 0.38,
    p_tp_floor_enter: float = 0.15,
    p_tp_floor_hold: float = 0.12,
    score_margin: float = 0.0001,
    soft_floor: float = -0.001,
    *,
    side_now: int = 1,
    enable_dd_stop: bool = False,
    dd_stop_roe: float = -0.02,
    tp_dyn: float = 0.0,
    sl_dyn: float = 0.0,
    fee_exit_only_override: Optional[float] = None,
    tp_target_roe: float = 0.006,
    sl_target_roe: float = 0.005,
    meta_provider=None,
    unified_lambda: float = 1.0,
    unified_rho: float = 0.0,
    cvar_alpha: float = 0.05,
) -> Dict[str, Any]:
    price_paths = np.asarray(price_paths, dtype=np.float64)
    if price_paths.ndim != 2:
        raise ValueError("price_paths must be 2D (n_paths, n_steps)")
    n_paths, n_steps = price_paths.shape
    if n_paths <= 0 or n_steps <= 1:
        return {
            "p_pos_exit": 0.0,
            "ev_exit": float(-fee_roundtrip),
            "exit_t_mean_sec": 0.0,
            "exit_t_p50_sec": 0.0,
            "exit_reason_counts": {"no_paths": int(n_paths)},
            "exit_t": np.zeros((max(0, n_paths),), dtype=np.int64),
            "net_out": np.zeros((max(0, n_paths),), dtype=np.float64),
        }

    H = min(int(max(2, horizon_sec)), int(n_steps))
    dt_dec = int(max(1, decision_dt_sec))
    # `fee_roundtrip` is usually (entry+exit). When callers pass exit-only cost, override here.
    fee_exit_only = float(fee_exit_only_override) if fee_exit_only_override is not None else (0.5 * float(fee_roundtrip))
    s0_f = float(s0)
    side_now = 1 if int(side_now) >= 0 else -1
    alt_side = -side_now
    switch_cost = float(max(0.0, 2.0 * float(exec_oneway) + float(impact_cost)))

    mu_ps = float(mu)
    sigma_ps = float(max(float(sigma), 1e-12))

    p_tp_valid_long = True
    p_tp_valid_short = True

    # Define local meta helper to ensure closure capture
    def _meta_default_fn(tau: float) -> dict:
        ppos_cur, ev_cur = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, int(side_now), float(leverage), float(fee_exit_only))
        ppos_alt, ev_alt = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, int(alt_side), float(leverage), float(fee_exit_only))
        cvar_cur = _approx_cvar_normal(mu_ps, sigma_ps, tau, int(side_now), float(leverage), float(fee_exit_only), alpha=float(cvar_alpha))
        cvar_alt = _approx_cvar_normal(mu_ps, sigma_ps, tau, int(alt_side), float(leverage), float(fee_exit_only), alpha=float(cvar_alpha))
        score_cur = _approx_unified_score_from_cumulative(ev_cur, cvar_cur, tau, float(unified_rho), float(unified_lambda))
        score_alt = _approx_unified_score_from_cumulative(ev_alt, cvar_alt, tau, float(unified_rho), float(unified_lambda))
        lev = float(max(1e-12, float(leverage)))
        tp = float(tp_dyn)
        sl = float(sl_dyn)

        p_tp_long = float("nan")
        p_tp_short = float("nan")
        if tp > 0.0:
            tp_under = tp / lev
            if bool(p_tp_valid_long):
                p_tp_long = _prob_max_geq(mu_ps, sigma_ps, tau, tp_under)
            if bool(p_tp_valid_short):
                p_tp_short = _prob_min_leq(mu_ps, sigma_ps, tau, -tp_under)

        if sl > 0.0:
            sl_under = sl / lev
            p_sl_long = _prob_min_leq(mu_ps, sigma_ps, tau, -sl_under)
            p_sl_short = _prob_max_geq(mu_ps, sigma_ps, tau, sl_under)
        else:
            p_sl_long = 0.0
            p_sl_short = 0.0

        if int(side_now) == 1:
            return {
                "score_long": float(score_cur),
                "score_short": float(score_alt),
                "p_pos_long": float(ppos_cur),
                "p_pos_short": float(ppos_alt),
                "p_sl_long": float(p_sl_long),
                "p_sl_short": float(p_sl_short),
                "p_tp_long": float(p_tp_long) if math.isfinite(float(p_tp_long)) else float("nan"),
                "p_tp_short": float(p_tp_short) if math.isfinite(float(p_tp_short)) else float("nan"),
            }
        return {
            "score_long": float(score_alt),
            "score_short": float(score_cur),
            "p_pos_long": float(ppos_alt),
            "p_pos_short": float(ppos_cur),
            "p_sl_long": float(p_sl_long),
            "p_sl_short": float(p_sl_short),
            "p_tp_long": float(p_tp_long) if math.isfinite(float(p_tp_long)) else float("nan"),
            "p_tp_short": float(p_tp_short) if math.isfinite(float(p_tp_short)) else float("nan"),
        }

    exit_t = np.full((n_paths,), int(H), dtype=np.int64)
    exit_reason = np.full((n_paths,), "time_stop", dtype=object)

    if s0_f <= 0.0:
        return {
            "p_pos_exit": 0.0,
            "ev_exit": float(-fee_roundtrip),
            "exit_t_mean_sec": 0.0,
            "exit_t_p50_sec": 0.0,
            "exit_reason_counts": {"bad_s0": int(n_paths)},
            "exit_t": exit_t,
            "net_out": np.zeros((n_paths,), dtype=np.float64),
        }

    prices = price_paths[:, :H]
    logret = np.log(np.maximum(prices / s0_f, 1e-12))

    realized = float(side_now) * float(leverage) * logret
    # ROE for TP/SL checks (simple return)
    roe_simple = float(side_now) * float(leverage) * ((prices / s0_f) - 1.0)

    flip_streak = np.zeros((n_paths,), dtype=np.int64)
    hold_bad = np.zeros((n_paths,), dtype=np.int64)
    decided = np.zeros((n_paths,), dtype=bool)

    mgn = float(max(0.0, score_margin))
    soft_floor_f = float(soft_floor)

    grace_sec = min(60.0, 0.25 * max(0.0, float(min_hold_sec)))

    for t in range(0, H, dt_dec):
        age = int(t)
        tau = float(max(0, H - t))
        tau_safe = float(max(1.0, tau))

        # ✅ TP/SL per-path hit check (priority exit)
        tp_hit = (~decided) & (roe_simple[:, t] >= float(tp_target_roe))
        sl_hit = (~decided) & (~tp_hit) & (roe_simple[:, t] <= -float(sl_target_roe))
        if np.any(tp_hit):
            exit_t[tp_hit] = age
            exit_reason[tp_hit] = "tp_hit"
            decided[tp_hit] = True
        if np.any(sl_hit):
            exit_t[sl_hit] = age
            exit_reason[sl_hit] = "sl_hit"
            decided[sl_hit] = True

        if np.all(decided):
            break

        if meta_provider is not None:
            try:
                m = meta_provider(t, tau, float(prices[0, t]), side_now)
            except Exception:
                # Fallback: Analytic calc for p_pos, and prob-barrier for p_sl/p_tp
                p_tp_L = _prob_max_geq(mu, sigma, tau, tp_target_roe / leverage)
                p_sl_L = _prob_min_leq(mu, sigma, tau, -sl_target_roe / leverage)
                p_tp_S = _prob_max_geq(-mu, sigma, tau, tp_target_roe / leverage)
                p_sl_S = _prob_min_leq(-mu, sigma, tau, -sl_target_roe / leverage)
                p_pos_L, ev_L = _approx_p_pos_and_ev_hold(mu, sigma, tau, 1, leverage, fee_exit_only)
                p_pos_S, ev_S = _approx_p_pos_and_ev_hold(mu, sigma, tau, -1, leverage, fee_exit_only)
                m = {
                    "score_long": ev_L, "score_short": ev_S,
                    "p_pos_long": p_pos_L, "p_pos_short": p_pos_S,
                    "p_sl_long": p_sl_L, "p_sl_short": p_sl_S,
                    "p_tp_long": p_tp_L, "p_tp_short": p_tp_S
                }
            m = _meta_default_fn(float(tau))
        else:
            m = _meta_default_fn(float(tau))

        score_long = float(m.get("score_long", 0.0))
        score_short = float(m.get("score_short", 0.0))
        p_pos_L = float(m.get("p_pos_long", 0.0))
        p_pos_S = float(m.get("p_pos_short", 0.0))
        p_sl_L = float(m.get("p_sl_long", 0.0))
        p_sl_S = float(m.get("p_sl_short", 0.0))
        p_tp_L = float(m.get("p_tp_long", float("nan")))
        p_tp_S = float(m.get("p_tp_short", float("nan")))

        if side_now == 1:
            score_cur = float(score_long)
            score_alt_raw = float(score_short)
            p_pos_cur, p_pos_alt = float(p_pos_L), float(p_pos_S)
            p_sl_cur, p_sl_alt = float(p_sl_L), float(p_sl_S)
            p_tp_cur, p_tp_alt = float(p_tp_L), float(p_tp_S)
        else:
            score_cur = float(score_short)
            score_alt_raw = float(score_long)
            p_pos_cur, p_pos_alt = float(p_pos_S), float(p_pos_L)
            p_sl_cur, p_sl_alt = float(p_sl_S), float(p_sl_L)
            p_tp_cur, p_tp_alt = float(p_tp_S), float(p_tp_L)

        score_alt = float(score_alt_raw) - (float(switch_cost) / float(tau_safe))
        gap_eff = float(score_cur - score_alt)

        alt_value_after_cost = float(score_alt_raw) - (float(exec_oneway) / float(tau_safe))
        flip_ok = bool(
            (p_pos_alt >= float(p_pos_floor_enter))
            and ((alt_value_after_cost > 0.0) or (alt_value_after_cost > float(soft_floor_f)))
        )
        if (float(p_sl_L) > 0.0) or (float(p_sl_S) > 0.0):
            flip_ok = bool(flip_ok and (float(p_sl_alt) <= float(p_sl_enter_ceiling)))
        if math.isfinite(float(p_tp_alt)) and float(p_tp_alt) > 0.0:
            flip_ok = bool(flip_ok and (float(p_tp_alt) >= float(p_tp_floor_enter)))

        pref_side = None
        if (gap_eff < -mgn) and flip_ok:
            pref_side = alt_side

        if pref_side is not None and pref_side != side_now:
            flip_streak = flip_streak + (~decided).astype(np.int64)
        else:
            flip_streak = flip_streak * 0

        hold_value_ok = bool(gap_eff >= -mgn)
        hold_ok = bool(hold_value_ok and (p_pos_cur >= float(p_pos_floor_hold)))
        if (float(p_sl_L) > 0.0) or (float(p_sl_S) > 0.0):
            hold_ok = bool(hold_ok and (float(p_sl_cur) <= float(p_sl_hold_ceiling)))
        if math.isfinite(float(p_tp_cur)) and float(p_tp_cur) > 0.0:
            hold_ok = bool(hold_ok and (float(p_tp_cur) >= float(p_tp_floor_hold)))

        if (float(p_sl_L) > 0.0) or (float(p_sl_S) > 0.0):
            if (float(p_sl_cur) >= float(p_sl_emergency)) and (float(age) >= float(grace_sec)) and age >= int(min_hold_sec):
                mask = ~decided
                exit_t[mask] = age
                exit_reason[mask] = "psl_emergency"
                decided[mask] = True
                break

        if (score_cur < -mgn) or (not hold_ok):
            hold_bad = hold_bad + (~decided).astype(np.int64)
        else:
            hold_bad = hold_bad * 0

        if age >= int(min_hold_sec):
            do_flip = (flip_streak >= int(flip_confirm_ticks)) & (~decided)
            do_hold_bad = (hold_bad >= int(hold_bad_ticks)) & (~decided)
            if np.any(do_flip):
                exit_t[do_flip] = age
                exit_reason[do_flip] = "score_flip"
                decided[do_flip] = True
            if np.any(do_hold_bad):
                exit_t[do_hold_bad] = age
                exit_reason[do_hold_bad] = "hold_bad"
                decided[do_hold_bad] = True
            if np.all(decided):
                break

        if enable_dd_stop and age >= int(min_hold_sec):
            mask = (~decided) & (realized[:, age] <= float(dd_stop_roe))
            if np.any(mask):
                exit_t[mask] = age
                exit_reason[mask] = "unrealized_dd"
                decided[mask] = True
                break

    net_out = realized[np.arange(n_paths), np.clip(exit_t, 0, realized.shape[1] - 1)]
    net_out = np.asarray(net_out, dtype=np.float64) - float(fee_roundtrip)

    p_pos_exit = float(np.mean(net_out > 0.0))
    ev_exit = float(np.mean(net_out))

    exit_t_mean = float(np.mean(exit_t))
    exit_t_p50 = float(np.median(exit_t))

    counts: Dict[str, int] = {}
    for r in exit_reason.tolist():
        counts[str(r)] = counts.get(str(r), 0) + 1

    is_tp = exit_reason == "tp_hit"
    is_sl = exit_reason == "sl_hit"
    is_other = ~(is_tp | is_sl)
    n_total = int(n_paths)
    n_tp = int(np.sum(is_tp))
    n_sl = int(np.sum(is_sl))
    n_other = int(np.sum(is_other))

    return {
        "p_pos_exit": float(p_pos_exit),
        "ev_exit": float(ev_exit),
        "exit_t_mean_sec": float(exit_t_mean),
        "exit_t_p50_sec": float(exit_t_p50),
        "exit_reason_counts": counts,
        "exit_t": exit_t,
        "net_out": net_out,
        "exit_reason": exit_reason,
        "p_tp": float(n_tp / n_total) if n_total > 0 else 0.0,
        "p_sl": float(n_sl / n_total) if n_total > 0 else 0.0,
        "p_other": float(n_other / n_total) if n_total > 0 else 0.0,
        "tp_r_actual": float(np.mean(net_out[is_tp])) if n_tp > 0 else 0.0,
        "sl_r_actual": float(np.mean(net_out[is_sl])) if n_sl > 0 else 0.0,
        "other_r_actual": float(np.mean(net_out[is_other])) if n_other > 0 else 0.0,
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_other": n_other,
        "prob_sum_check": True,
    }


def simulate_exit_policy_rollforward_analytic(
    *,
    mu_ps: float,
    sigma_ps: float,
    leverage: float,
    fee_roundtrip: float,
    exec_oneway: float,
    impact_cost: float,
    horizon_sec: int,
    decision_dt_sec: int,
    min_hold_sec: int,
    flip_confirm_ticks: int,
    hold_bad_ticks: int,
    p_pos_floor_enter: float,
    p_pos_floor_hold: float,
    p_sl_enter_ceiling: float,
    p_sl_hold_ceiling: float,
    p_sl_emergency: float,
    p_tp_floor_enter: float,
    p_tp_floor_hold: float,
    score_margin: float,
    soft_floor: float,
    side_now: int,
    tp_dyn: float,
    sl_dyn: float,
    fee_exit_only_override: Optional[float] = None,
    p_tp_valid_long: bool = True,
    p_tp_valid_short: bool = True,
    unified_lambda: float = 1.0,
    unified_rho: float = 0.0,
    cvar_alpha: float = 0.05,
) -> Dict[str, Any]:
    H = int(max(1, int(horizon_sec)))
    dt_dec = int(max(1, int(decision_dt_sec)))
    side_now = 1 if int(side_now) >= 0 else -1
    alt_side = -side_now

    switch_cost = float(max(0.0, 2.0 * float(exec_oneway) + float(impact_cost)))
    fee_exit_only = float(fee_exit_only_override) if fee_exit_only_override is not None else (0.5 * float(fee_roundtrip))
    mgn = float(max(0.0, score_margin))
    soft_floor_f = float(soft_floor)

    # Define local meta helper to ensure closure capture
    def _meta_default_fn(tau: float) -> dict:
        ppos_cur, ev_cur = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, int(side_now), float(leverage), float(fee_exit_only))
        ppos_alt, ev_alt = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, int(alt_side), float(leverage), float(fee_exit_only))
        cvar_cur = _approx_cvar_normal(mu_ps, sigma_ps, tau, int(side_now), float(leverage), float(fee_exit_only), alpha=float(cvar_alpha))
        cvar_alt = _approx_cvar_normal(mu_ps, sigma_ps, tau, int(alt_side), float(leverage), float(fee_exit_only), alpha=float(cvar_alpha))
        score_cur = _approx_unified_score_from_cumulative(ev_cur, cvar_cur, tau, float(unified_rho), float(unified_lambda))
        score_alt = _approx_unified_score_from_cumulative(ev_alt, cvar_alt, tau, float(unified_rho), float(unified_lambda))
        lev = float(max(1e-12, float(leverage)))
        tp = float(tp_dyn)
        sl = float(sl_dyn)

        p_tp_long = float("nan")
        p_tp_short = float("nan")
        if tp > 0.0:
            tp_under = tp / lev
            if bool(p_tp_valid_long):
                p_tp_long = _prob_max_geq(mu_ps, sigma_ps, tau, tp_under)
            if bool(p_tp_valid_short):
                p_tp_short = _prob_min_leq(mu_ps, sigma_ps, tau, -tp_under)

        if sl > 0.0:
            sl_under = sl / lev
            p_sl_long = _prob_min_leq(mu_ps, sigma_ps, tau, -sl_under)
            p_sl_short = _prob_max_geq(mu_ps, sigma_ps, tau, sl_under)
        else:
            p_sl_long = 0.0
            p_sl_short = 0.0

        if int(side_now) == 1:
            return {
                "score_long": float(score_cur),
                "score_short": float(score_alt),
                "p_pos_long": float(ppos_cur),
                "p_pos_short": float(ppos_alt),
                "p_sl_long": float(p_sl_long),
                "p_sl_short": float(p_sl_short),
                "p_tp_long": float(p_tp_long) if math.isfinite(float(p_tp_long)) else float("nan"),
                "p_tp_short": float(p_tp_short) if math.isfinite(float(p_tp_short)) else float("nan"),
            }
        return {
            "score_long": float(score_alt),
            "score_short": float(score_cur),
            "p_pos_long": float(ppos_alt),
            "p_pos_short": float(ppos_cur),
            "p_sl_long": float(p_sl_long),
            "p_sl_short": float(p_sl_short),
            "p_tp_long": float(p_tp_long) if math.isfinite(float(p_tp_long)) else float("nan"),
            "p_tp_short": float(p_tp_short) if math.isfinite(float(p_tp_short)) else float("nan"),
        }

    flip_streak = 0
    hold_bad = 0
    exit_t = H
    reason = "time_stop"

    for t in range(0, H + 1, dt_dec):
        age = int(t)
        tau = max(0.0, float(H - t))
        tau_safe = float(max(1.0, tau))
        m = _meta_default_fn(float(tau))

        score_long = float(m.get("score_long", 0.0))
        score_short = float(m.get("score_short", 0.0))
        p_pos_L = float(m.get("p_pos_long", 0.0))
        p_pos_S = float(m.get("p_pos_short", 0.0))
        p_sl_L = float(m.get("p_sl_long", 0.0))
        p_sl_S = float(m.get("p_sl_short", 0.0))
        p_tp_L = float(m.get("p_tp_long", float("nan")))
        p_tp_S = float(m.get("p_tp_short", float("nan")))

        if side_now == 1:
            score_cur = float(score_long)
            score_alt_raw = float(score_short)
            p_pos_cur, p_pos_alt = float(p_pos_L), float(p_pos_S)
            p_sl_cur, p_sl_alt = float(p_sl_L), float(p_sl_S)
            p_tp_cur, p_tp_alt = float(p_tp_L), float(p_tp_S)
        else:
            score_cur = float(score_short)
            score_alt_raw = float(score_long)
            p_pos_cur, p_pos_alt = float(p_pos_S), float(p_pos_L)
            p_sl_cur, p_sl_alt = float(p_sl_S), float(p_sl_L)
            p_tp_cur, p_tp_alt = float(p_tp_S), float(p_tp_L)

        score_alt = float(score_alt_raw) - (float(switch_cost) / float(tau_safe))
        gap_eff = float(score_cur - score_alt)

        alt_value_after_cost = float(score_alt_raw) - (float(exec_oneway) / float(tau_safe))
        flip_ok = bool(
            (p_pos_alt >= float(p_pos_floor_enter))
            and ((alt_value_after_cost > 0.0) or (alt_value_after_cost > float(soft_floor_f)))
        )
        if (float(p_sl_L) > 0.0) or (float(p_sl_S) > 0.0):
            flip_ok = bool(flip_ok and (float(p_sl_alt) <= float(p_sl_enter_ceiling)))
        if math.isfinite(float(p_tp_alt)) and float(p_tp_alt) > 0.0:
            flip_ok = bool(flip_ok and (float(p_tp_alt) >= float(p_tp_floor_enter)))

        pref_side = None
        if (gap_eff < -mgn) and flip_ok:
            pref_side = alt_side

        if pref_side is not None and pref_side != side_now:
            flip_streak += 1
        else:
            flip_streak = 0

        hold_value_ok = bool(gap_eff >= -mgn)
        hold_ok = bool(hold_value_ok and (p_pos_cur >= float(p_pos_floor_hold)))
        if (float(p_sl_L) > 0.0) or (float(p_sl_S) > 0.0):
            hold_ok = bool(hold_ok and (float(p_sl_cur) <= float(p_sl_hold_ceiling)))
        if math.isfinite(float(p_tp_cur)) and float(p_tp_cur) > 0.0:
            hold_ok = bool(hold_ok and (float(p_tp_cur) >= float(p_tp_floor_hold)))

        grace_sec = min(60.0, 0.25 * max(0.0, float(min_hold_sec)))
        if (float(p_sl_L) > 0.0) or (float(p_sl_S) > 0.0):
            if (float(p_sl_cur) >= float(p_sl_emergency)) and (float(age) >= float(grace_sec)) and age >= int(min_hold_sec):
                exit_t = age
                reason = "psl_emergency"
                break

        if (score_cur < -mgn) or (not hold_ok):
            hold_bad += 1
        else:
            hold_bad = 0

        if age >= int(min_hold_sec) and flip_streak >= int(flip_confirm_ticks):
            exit_t = age
            reason = "score_flip"
            break
        if age >= int(min_hold_sec) and hold_bad >= int(hold_bad_ticks):
            exit_t = age
            reason = "hold_bad"
            break

    p_pos_rt, ev_rt = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, float(exit_t), side_now, leverage, float(fee_roundtrip))
    p_pos_ow, ev_ow = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, float(exit_t), side_now, leverage, float(exec_oneway))
    return {
        "exit_t_sec": float(exit_t),
        "exit_reason": str(reason),
        "p_pos_exit_roundtrip": float(p_pos_rt),
        "ev_exit_roundtrip": float(ev_rt),
        "p_pos_exit_oneway": float(p_pos_ow),
        "ev_exit_oneway": float(ev_ow),
    }


def _weights_for_horizons(hs: Sequence[float], signal_strength: float):
    h_arr = np.asarray(hs if hs else [], dtype=np.float64)
    if h_arr.size == 0:
        return np.asarray([], dtype=np.float64)
    s = float(np.clip(signal_strength, 0.0, 4.0))
    half_life = 1800.0 / (1.0 + s)
    decay = np.exp(-h_arr / max(1e-9, half_life))
    total = float(np.sum(decay))
    if total <= 0.0:
        return np.full(h_arr.shape, 1.0 / float(h_arr.size), dtype=np.float64)
    return decay / total


def _execution_mix_from_survival(
    meta: Dict[str, Any],
    fee_maker: float,
    fee_taker: float,
    horizon_sec: int,
    sigma_per_sec: float,
    prefix: str,
    delay_penalty_mult: float,
) -> Dict[str, float]:
    p_fill = float(meta.get(prefix) or 0.0)
    delay_sec = float(meta.get(f"{prefix}_delay_sec") or 0.0)
    delay_cond_sec = float(meta.get(f"{prefix}_delay_cond_sec") or delay_sec)

    fee_mix = p_fill * float(fee_maker) + (1.0 - p_fill) * float(fee_taker)
    delay_penalty_r = float(delay_penalty_mult) * float(sigma_per_sec) * math.sqrt(max(0.0, delay_sec))
    horizon_eff = max(1, int(round(int(horizon_sec) - delay_sec)))

    return {
        "p_fill": p_fill,
        "delay_sec": delay_sec,
        "delay_cond_sec": delay_cond_sec,
        "fee_mix": float(fee_mix),
        "delay_penalty_r": float(delay_penalty_r),
        "horizon_eff_sec": float(horizon_eff),
    }


def _sigma_per_sec(sigma: float, dt: float) -> float:
    if dt <= 0:
        return 0.0
    return float(sigma) * math.sqrt(float(dt))

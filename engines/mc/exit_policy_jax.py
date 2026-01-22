from __future__ import annotations

from engines.mc.jax_backend import ensure_jax, lazy_jit, jax, jnp, lax
from functools import partial
from engines.mc.probability_jax import _approx_p_pos_and_ev_hold_jax, _prob_max_geq_jax, _prob_min_leq_jax

@lazy_jit(static_argnames=("decision_dt_sec", "step_sec", "horizon_sec", "min_hold_sec", "flip_confirm_ticks", "hold_bad_ticks", "enable_dd_stop"))
def simulate_exit_policy_rollforward_jax(
    price_paths: jnp.ndarray,  # (n_paths, h_pts)
    s0: float,
    mu_ps: float,
    sigma_ps: float,
    leverage: float,
    fee_roundtrip: float,
    exec_oneway: float,
    impact_cost: float,
    step_sec: int,
    decision_dt_sec: int,
    horizon_sec: int,
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
    enable_dd_stop: bool,
    dd_stop_roe: float,
    fee_exit_only_override: float = -1.0,
    tp_target_roe: float = 0.006,
    sl_target_roe: float = 0.005,
):
    n_paths, h_pts = price_paths.shape
    H = h_pts - 1  # steps
    side_now = jnp.where(side_now >= 0, 1, -1)
    alt_side = -side_now
    switch_cost = jnp.maximum(0.0, 2.0 * exec_oneway + impact_cost)
    fee_exit_only = jnp.where(fee_exit_only_override >= 0.0, fee_exit_only_override, 0.5 * fee_roundtrip)
    
    # Pre-calculate logret and realized PnL
    logret = jnp.log(jnp.maximum(price_paths / s0, 1e-12))
    realized = jnp.float32(side_now) * jnp.float32(leverage) * logret
    
    # State: (exit_t, exit_reason_idx, decided, flip_streak, hold_bad, finished)
    # exit_reason_idx: 0: time_stop, 1: psl_emergency, 2: score_flip, 3: hold_bad, 4: unrealized_dd
    init_state = (
        jnp.full((n_paths,), H, dtype=jnp.int32),
        jnp.zeros((n_paths,), dtype=jnp.int32),
        jnp.zeros((n_paths,), dtype=jnp.bool_),
        jnp.zeros((n_paths,), dtype=jnp.int32),
        jnp.zeros((n_paths,), dtype=jnp.int32),
        False # Global finished flag
    )
    
    # Steps to scan
    # compute step counts as Python ints to avoid tracer/concretization problems
    decision_dt_steps = int(max(1, (int(decision_dt_sec) + int(step_sec) - 1) // int(step_sec)))
    min_hold_steps = int(max(0, (int(min_hold_sec) + int(step_sec) - 1) // int(step_sec)))
    steps = jnp.arange(0, int(H), decision_dt_steps)
    
    def body_fun(state, t):
        exit_t, reason, decided, flip_streak, hold_bad, finished = state
        
        def compute_step(_):
            t_sec = jnp.float32(t) * jnp.float32(step_sec)
            tau = jnp.float32(jnp.maximum(0, H - t)) * jnp.float32(step_sec)
            
            # Policy metrics (shared across paths)
            p_pos_cur, ev_cur = _approx_p_pos_and_ev_hold_jax(mu_ps, sigma_ps, tau, side_now, leverage, fee_exit_only)
            p_pos_alt, ev_alt = _approx_p_pos_and_ev_hold_jax(mu_ps, sigma_ps, tau, alt_side, leverage, fee_exit_only)
            
            # ✅ Direction-aware TP/SL targets
            # For Long (side_now=+1): TP is positive (price up), SL is negative (price down)
            # For Short (side_now=-1): TP is negative (price down), SL is positive (price up)
            target_tp = (tp_target_roe / leverage) * jnp.float32(side_now)
            target_sl = (sl_target_roe / leverage) * jnp.float32(side_now)
            
            # Calculate probabilities with directional drift
            p_tp_cur = _prob_max_geq_jax(mu_ps * side_now, sigma_ps, tau, target_tp)
            p_sl_cur = _prob_min_leq_jax(mu_ps * side_now, sigma_ps, tau, target_sl)
            p_tp_alt = _prob_max_geq_jax(mu_ps * alt_side, sigma_ps, tau, -target_tp)
            p_sl_alt = _prob_min_leq_jax(mu_ps * alt_side, sigma_ps, tau, -target_sl)
            
            score_alt_raw = ev_alt
            score_alt = score_alt_raw - switch_cost
            gap_eff = ev_cur - score_alt
            alt_value_after_cost = score_alt_raw - exec_oneway
            
            flip_ok = (p_pos_alt >= p_pos_floor_enter) & ((alt_value_after_cost > 0.0) | (alt_value_after_cost > soft_floor))
            flip_ok = flip_ok & (p_sl_alt <= p_sl_enter_ceiling) & (p_tp_alt >= p_tp_floor_enter)
            
            hold_ok = (gap_eff >= -score_margin) & (p_pos_cur >= p_pos_floor_hold)
            hold_ok = hold_ok & (p_sl_cur <= p_sl_hold_ceiling) & (p_tp_cur >= p_tp_floor_hold)
            
            grace_sec = jnp.minimum(60.0, 0.25 * jnp.maximum(0.0, jnp.float32(min_hold_sec)))
            after_min_hold = (t >= min_hold_steps)
            is_emergency = (p_sl_cur >= p_sl_emergency) & (t_sec >= grace_sec) & after_min_hold
            
            # Update state with mask for only undecided paths
            new_exit_t = jnp.where(is_emergency & (~decided), t, exit_t)
            new_reason = jnp.where(is_emergency & (~decided), 1, reason)
            new_decided = decided | is_emergency
            
            pref_alt = (gap_eff < -score_margin) & flip_ok
            new_flip_streak = jnp.where(pref_alt & (~new_decided), flip_streak + 1, 0)
            
            bad_hold = (ev_cur < -score_margin) | (~hold_ok)
            new_hold_bad = jnp.where(bad_hold & (~new_decided), hold_bad + 1, 0)
            
            do_flip = after_min_hold & (new_flip_streak >= flip_confirm_ticks) & (~new_decided)
            new_exit_t = jnp.where(do_flip, t, new_exit_t)
            new_reason = jnp.where(do_flip, 2, new_reason)
            new_decided = new_decided | do_flip
            
            do_hold_bad = after_min_hold & (new_hold_bad >= hold_bad_ticks) & (~new_decided)
            new_exit_t = jnp.where(do_hold_bad, t, new_exit_t)
            new_reason = jnp.where(do_hold_bad, 3, new_reason)
            new_decided = new_decided | do_hold_bad
            
            if enable_dd_stop:
                is_dd = after_min_hold & (realized[:, t] <= dd_stop_roe) & (~new_decided)
                new_exit_t = jnp.where(is_dd, t, new_exit_t)
                new_reason = jnp.where(is_dd, 4, new_reason)
                new_decided = new_decided | is_dd
                
            return (new_exit_t, new_reason, new_decided, new_flip_streak, new_hold_bad, jnp.all(new_decided))
        
        # Skip if already finished
        return lax.cond(finished, lambda _: state, compute_step, None), None

    final_state, _ = lax.scan(body_fun, init_state, steps)
    exit_t, reason_idx, decided, _, _, _ = final_state
    
    # Calculate results
    exit_t_clipped = jnp.clip(exit_t, 0, H)
    net_out = realized[jnp.arange(n_paths), exit_t_clipped] - fee_roundtrip
    
    p_pos_exit = jnp.mean(net_out > 0.0)
    ev_exit = jnp.mean(net_out)
    exit_t_mean = jnp.mean(exit_t)
    
    return {
        "p_pos_exit": p_pos_exit,
        "ev_exit": ev_exit,
        "exit_t_mean_sec": exit_t_mean * jnp.float32(step_sec),
        "net_out": net_out,
        "exit_t": exit_t,
        "reason_idx": reason_idx
    }

from engines.mc.jax_backend import _cvar_jax

@lazy_jit(static_argnames=("decision_dt_sec", "step_sec", "max_horizon_sec", "flip_confirm_ticks", "hold_bad_ticks", "enable_dd_stop"))
def simulate_exit_policy_rollforward_batched_vmap_jax(

    price_paths: jnp.ndarray,  # (n_paths, h_pts) -> Shared across batch
    s0: float,
    mu_ps: float,
    sigma_ps: float,
    leverage: float,
    fee_roundtrip: float,
    exec_oneway: float,
    impact_cost: float,
    decision_dt_sec: int,
    step_sec: int,
    max_horizon_sec: int,
    # Batched arguments (vmapped)
    side_now_batch: jnp.ndarray,     # (batch_size,)
    horizon_sec_batch: jnp.ndarray,  # (batch_size,)
    min_hold_sec_batch: jnp.ndarray, # (batch_size,)
    tp_target_roe_batch: jnp.ndarray, # (batch_size,)
    sl_target_roe_batch: jnp.ndarray, # (batch_size,)
    # Policy thresholds
    p_pos_floor_enter: float,
    p_pos_floor_hold: float,
    p_sl_enter_ceiling: float,
    p_sl_hold_ceiling: float,
    p_sl_emergency: float,
    p_tp_floor_enter: float,
    p_tp_floor_hold: float,
    score_margin: float,
    soft_floor: float,
    enable_dd_stop: bool,
    dd_stop_roe_batch: jnp.ndarray,   # (batch_size,)
    flip_confirm_ticks: int,
    hold_bad_ticks: int,
    fee_exit_only_override: float = -1.0,
    cvar_alpha: float = 0.05,
):
    def single_sim(side_now, horizon_sec, min_hold_sec, tp_target_roe, sl_target_roe, dd_stop_roe):
        n_paths, _ = price_paths.shape
        # Convert horizon(sec) -> horizon(steps) for indexing into price_paths
        max_steps_avail = price_paths.shape[1] - 1
        H_steps = jnp.minimum(
            jnp.int32(max_steps_avail),
            jnp.int32((horizon_sec + step_sec - 1) // step_sec),
        )
        H = jnp.maximum(1, H_steps)
        side_now_val = jnp.where(side_now >= 0, 1, -1)
        alt_side = -side_now_val
        switch_cost = jnp.maximum(0.0, 2.0 * exec_oneway + impact_cost)
        fee_exit_only = jnp.where(fee_exit_only_override >= 0.0, fee_exit_only_override, 0.5 * fee_roundtrip)
        
        logret = jnp.log(jnp.maximum(price_paths / s0, 1e-12))
        realized = jnp.float32(side_now_val) * jnp.float32(leverage) * logret
        
        init_state = (
            jnp.full((n_paths,), H, dtype=jnp.int32),
            jnp.zeros((n_paths,), dtype=jnp.int32),
            jnp.zeros((n_paths,), dtype=jnp.bool_),
            jnp.zeros((n_paths,), dtype=jnp.int32),
            jnp.zeros((n_paths,), dtype=jnp.int32),
            False
        )
        
        # decision_dt_steps is static (based on static args) so keep as Python int
        decision_dt_steps = int(max(1, (int(decision_dt_sec) + int(step_sec) - 1) // int(step_sec)))
        # min_hold_sec is batched (may be a tracer/jnp scalar) -> compute as jnp scalar
        min_hold_steps = jnp.maximum(0, (min_hold_sec + step_sec - 1) // step_sec)
        # max_horizon_steps is derived from the static max_horizon_sec
        max_horizon_steps = int(max(1, (int(max_horizon_sec) + int(step_sec) - 1) // int(step_sec)))
        steps = jnp.arange(0, int(max_horizon_steps), decision_dt_steps)
        
        def body_fun(state, t):
            exit_t, reason, decided, flip_streak, hold_bad, finished = state
            
            def compute_step(_):
                active = t < H
                t_sec = jnp.float32(t) * jnp.float32(step_sec)
                tau = jnp.float32(jnp.maximum(0, H - t)) * jnp.float32(step_sec)
                
                p_pos_cur, ev_cur = _approx_p_pos_and_ev_hold_jax(mu_ps, sigma_ps, tau, side_now_val, leverage, fee_exit_only)
                p_pos_alt, ev_alt = _approx_p_pos_and_ev_hold_jax(mu_ps, sigma_ps, tau, alt_side, leverage, fee_exit_only)
                
                target_tp = (tp_target_roe / leverage) * jnp.float32(side_now_val)
                target_sl = (sl_target_roe / leverage) * jnp.float32(side_now_val)
                
                p_tp_cur = _prob_max_geq_jax(mu_ps * side_now_val, sigma_ps, tau, target_tp)
                p_sl_cur = _prob_min_leq_jax(mu_ps * side_now_val, sigma_ps, tau, target_sl)
                p_tp_alt = _prob_max_geq_jax(mu_ps * alt_side, sigma_ps, tau, -target_tp)
                p_sl_alt = _prob_min_leq_jax(mu_ps * alt_side, sigma_ps, tau, -target_sl)
                
                score_alt = ev_alt - switch_cost
                gap_eff = ev_cur - score_alt
                alt_value_after_cost = ev_alt - exec_oneway
                
                flip_ok = (p_pos_alt >= p_pos_floor_enter) & ((alt_value_after_cost > 0.0) | (alt_value_after_cost > soft_floor))
                flip_ok = flip_ok & (p_sl_alt <= p_sl_enter_ceiling) & (p_tp_alt >= p_tp_floor_enter)
                
                hold_ok = (gap_eff >= -score_margin) & (p_pos_cur >= p_pos_floor_hold)
                hold_ok = hold_ok & (p_sl_cur <= p_sl_hold_ceiling) & (p_tp_cur >= p_tp_floor_hold)
                
                grace_sec = jnp.minimum(60.0, 0.25 * jnp.maximum(0.0, jnp.float32(min_hold_sec)))
                after_min_hold = (t >= min_hold_steps)
                is_emergency = (p_sl_cur >= p_sl_emergency) & (t_sec >= grace_sec) & after_min_hold
                
                new_exit_t = jnp.where(active & is_emergency & (~decided), t, exit_t)
                new_reason = jnp.where(active & is_emergency & (~decided), 1, reason)
                new_decided = decided | (active & is_emergency)
                
                pref_alt = (gap_eff < -score_margin) & flip_ok
                new_flip_streak = jnp.where(active & pref_alt & (~new_decided), flip_streak + 1, 0)
                
                bad_hold = (ev_cur < -score_margin) | (~hold_ok)
                new_hold_bad = jnp.where(active & bad_hold & (~new_decided), hold_bad + 1, 0)
                
                do_flip = active & after_min_hold & (new_flip_streak >= flip_confirm_ticks) & (~new_decided)
                new_exit_t = jnp.where(do_flip, t, new_exit_t)
                new_reason = jnp.where(do_flip, 2, new_reason)
                new_decided = new_decided | do_flip
                
                do_hold_bad = active & after_min_hold & (new_hold_bad >= hold_bad_ticks) & (~new_decided)
                new_exit_t = jnp.where(do_hold_bad, t, new_exit_t)
                new_reason = jnp.where(do_hold_bad, 3, new_reason)
                new_decided = new_decided | do_hold_bad
                
                if enable_dd_stop:
                    is_dd = active & after_min_hold & (realized[:, t] <= dd_stop_roe) & (~new_decided)
                    new_exit_t = jnp.where(is_dd, t, new_exit_t)
                    new_reason = jnp.where(is_dd, 4, new_reason)
                    new_decided = new_decided | is_dd
                
                return (new_exit_t, new_reason, new_decided, new_flip_streak, new_hold_bad, jnp.all(new_decided))

            return lax.cond(finished | (t >= H), lambda _: state, compute_step, None), None

        final_state, _ = lax.scan(body_fun, init_state, steps)
        exit_t, reason_idx, decided, _, _, _ = final_state
        
        exit_t_clipped = jnp.clip(exit_t, 0, H)
        net_out = realized[jnp.arange(n_paths), exit_t_clipped] - fee_roundtrip
        
        return {
            "p_pos_exit": jnp.mean(net_out > 0.0),
            "ev_exit": jnp.mean(net_out),
            "exit_t_mean_sec": jnp.mean(jnp.float32(exit_t)) * jnp.float32(step_sec),
            "reason_idx": reason_idx,
            "cvar_exit": _cvar_jax(net_out, alpha=cvar_alpha),
            # "net_out": net_out, # Don't return if not needed to save bandwidth
            # "exit_t": exit_t
        }

    v_func = jax.vmap(single_sim, in_axes=(0, 0, 0, 0, 0, 0))
    return v_func(side_now_batch, horizon_sec_batch, min_hold_sec_batch, tp_target_roe_batch, sl_target_roe_batch, dd_stop_roe_batch)


# ✅ GLOBAL BATCHING: Multi-symbol version of exit policy simulation
# This vmaps over the entire symbol dimension.
def simulate_exit_policy_multi_symbol_jax(
    price_paths_batch,
    s0_batch,
    mu_ps_batch,
    sigma_ps_batch,
    leverage_batch,
    fee_roundtrip_batch,
    exec_oneway_batch,
    impact_cost_batch,
    decision_dt_sec,
    step_sec,
    max_horizon_sec,
    side_now_batch,
    horizon_sec_batch,
    min_hold_sec_batch,
    tp_target_roe_batch,
    sl_target_roe_batch,
    p_pos_floor_enter,
    p_pos_floor_hold,
    p_sl_enter_ceiling,
    p_sl_hold_ceiling,
    p_sl_emergency,
    p_tp_floor_enter,
    p_tp_floor_hold,
    score_margin,
    soft_floor,
    enable_dd_stop,
    dd_stop_roe_batch,
    flip_confirm_ticks,
    hold_bad_ticks,
    fee_exit_only_override,
    cvar_alpha,
):
    """Lazily create and run a vmap over symbols for the batched rollforward.

    This avoids creating the vmap at module import time (which would touch JAX).
    """
    from engines.mc import jax_backend as _jb
    _jb.ensure_jax()
    if not _jb._JAX_OK:
        raise RuntimeError("JAX is not available for simulate_exit_policy_multi_symbol_jax")

    vm = _jb.jax.vmap(
        simulate_exit_policy_rollforward_batched_vmap_jax,
        in_axes=(
            0, 0, 0, 0, 0, 0, 0, 0,  # paths, s0, mu, sigma, lev, fee, exec, impact
            None, None, None,        # decision_dt_sec, step_sec, max_horizon_sec (static)
            0, 0, 0, 0, 0,           # batched args (side, horizon, min_hold, tp, sl)
            0, 0, 0, 0, 0, 0, 0, 0, 0,  # policy thresholds
            None,                    # enable_dd_stop
            0,                       # dd_stop_roe_batch
            None, None,              # ticks
            None, None               # fee_exit_only, cvar_alpha
        )
    )

    return vm(
        price_paths_batch,
        s0_batch,
        mu_ps_batch,
        sigma_ps_batch,
        leverage_batch,
        fee_roundtrip_batch,
        exec_oneway_batch,
        impact_cost_batch,
        decision_dt_sec,
        step_sec,
        max_horizon_sec,
        side_now_batch,
        horizon_sec_batch,
        min_hold_sec_batch,
        tp_target_roe_batch,
        sl_target_roe_batch,
        p_pos_floor_enter,
        p_pos_floor_hold,
        p_sl_enter_ceiling,
        p_sl_hold_ceiling,
        p_sl_emergency,
        p_tp_floor_enter,
        p_tp_floor_hold,
        score_margin,
        soft_floor,
        enable_dd_stop,
        dd_stop_roe_batch,
        flip_confirm_ticks,
        hold_bad_ticks,
        fee_exit_only_override,
        cvar_alpha,
    )

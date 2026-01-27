from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from engines.probability_methods import _approx_p_pos_and_ev_hold, _prob_max_geq, _prob_min_leq
from engines.mc.torch_backend import _TORCH_OK, torch, to_torch, to_numpy, get_torch_device


def _policy_metrics_batch(
    mu_ps: float,
    sigma_ps: float,
    tau_sec: np.ndarray,
    side_now: np.ndarray,
    leverage: float,
    fee_exit_only: float,
    tp_target_roe: np.ndarray,
    sl_target_roe: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute policy metrics per batch (numpy)."""
    batch = len(side_now)
    p_pos_cur = np.zeros(batch, dtype=np.float32)
    p_pos_alt = np.zeros(batch, dtype=np.float32)
    ev_cur = np.zeros(batch, dtype=np.float32)
    ev_alt = np.zeros(batch, dtype=np.float32)
    p_sl_cur = np.zeros(batch, dtype=np.float32)
    p_sl_alt = np.zeros(batch, dtype=np.float32)
    p_tp_cur = np.zeros(batch, dtype=np.float32)
    p_tp_alt = np.zeros(batch, dtype=np.float32)

    for i in range(batch):
        tau = float(max(0.0, tau_sec[i]))
        side = 1 if int(side_now[i]) >= 0 else -1
        alt_side = -side
        p_pos_c, ev_c = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, side, leverage, fee_exit_only)
        p_pos_a, ev_a = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, alt_side, leverage, fee_exit_only)

        target_tp = (float(tp_target_roe[i]) / max(1e-12, leverage)) * float(side)
        target_sl = (float(sl_target_roe[i]) / max(1e-12, leverage)) * float(side)

        p_tp_c = _prob_max_geq(mu_ps * side, sigma_ps, tau, target_tp)
        p_sl_c = _prob_min_leq(mu_ps * side, sigma_ps, tau, target_sl)
        p_tp_a = _prob_max_geq(mu_ps * alt_side, sigma_ps, tau, -target_tp)
        p_sl_a = _prob_min_leq(mu_ps * alt_side, sigma_ps, tau, -target_sl)

        p_pos_cur[i] = p_pos_c
        p_pos_alt[i] = p_pos_a
        ev_cur[i] = ev_c
        ev_alt[i] = ev_a
        p_sl_cur[i] = p_sl_c
        p_sl_alt[i] = p_sl_a
        p_tp_cur[i] = p_tp_c
        p_tp_alt[i] = p_tp_a

    return p_pos_cur, p_pos_alt, ev_cur, ev_alt, p_sl_cur, p_sl_alt, p_tp_cur, p_tp_alt


def simulate_exit_policy_rollforward_batched_vmap_torch(
    price_paths,  # (n_paths, h_pts)
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
    side_now_batch,     # (batch,)
    horizon_sec_batch,  # (batch,)
    min_hold_sec_batch, # (batch,)
    tp_target_roe_batch, # (batch,)
    sl_target_roe_batch, # (batch,)
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
    dd_stop_roe_batch,   # (batch,)
    flip_confirm_ticks: int,
    hold_bad_ticks: int,
    fee_exit_only_override: float = -1.0,
    cvar_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Torch implementation of batched exit-policy simulation."""
    if not _TORCH_OK or torch is None:
        raise RuntimeError("Torch not available for exit policy")

    device = get_torch_device()
    price_paths_t = price_paths if isinstance(price_paths, torch.Tensor) else to_torch(price_paths, device=device)

    n_paths, h_pts = price_paths_t.shape
    max_steps_avail = int(h_pts - 1)
    batch_size = int(len(side_now_batch))

    side_now = to_torch(side_now_batch, device=device, dtype=torch.float32)
    side_now = torch.where(side_now >= 0, 1.0, -1.0)
    horizon_sec = to_torch(horizon_sec_batch, device=device, dtype=torch.float32)
    min_hold_sec = to_torch(min_hold_sec_batch, device=device, dtype=torch.float32)
    tp_target = to_torch(tp_target_roe_batch, device=device, dtype=torch.float32)
    sl_target = to_torch(sl_target_roe_batch, device=device, dtype=torch.float32)
    dd_stop = to_torch(dd_stop_roe_batch, device=device, dtype=torch.float32)

    decision_dt_steps = int(max(1, math.ceil(float(decision_dt_sec) / float(step_sec))))
    max_horizon_steps = int(max(1, math.ceil(float(max_horizon_sec) / float(step_sec))))
    max_horizon_steps = int(min(max_horizon_steps, max_steps_avail))

    H_steps = torch.minimum(
        torch.tensor(max_steps_avail, device=device, dtype=torch.int64),
        torch.ceil(horizon_sec / float(step_sec)).to(torch.int64),
    )
    H_steps = torch.clamp(H_steps, min=1)

    min_hold_steps = torch.ceil(min_hold_sec / float(step_sec)).to(torch.int64)

    exit_t = torch.full((batch_size, n_paths), max_steps_avail, dtype=torch.int64, device=device)
    reason = torch.zeros((batch_size, n_paths), dtype=torch.int64, device=device)
    decided = torch.zeros((batch_size, n_paths), dtype=torch.bool, device=device)
    flip_streak = torch.zeros((batch_size, n_paths), dtype=torch.int64, device=device)
    hold_bad = torch.zeros((batch_size, n_paths), dtype=torch.int64, device=device)

    logret = torch.log(torch.clamp(price_paths_t / float(s0), min=1e-12))

    fee_exit_only = float(fee_exit_only_override) if fee_exit_only_override >= 0.0 else (0.5 * float(fee_roundtrip))
    switch_cost = float(max(0.0, 2.0 * float(exec_oneway) + float(impact_cost)))

    for t in range(0, max_horizon_steps + 1, decision_dt_steps):
        active = t <= H_steps
        if not torch.any(active):
            break

        active_mask = active[:, None] & (~decided)
        price_t = price_paths_t[:, t]
        roe_t = side_now[:, None] * ((price_t[None, :] / float(s0)) - 1.0) * float(leverage)

        tp_hit = roe_t >= tp_target[:, None]
        sl_hit = roe_t <= -sl_target[:, None]

        tp_mask = active_mask & tp_hit
        sl_mask = active_mask & (~tp_mask) & sl_hit

        if torch.any(tp_mask):
            exit_t[tp_mask] = t
            reason[tp_mask] = 5  # tp_hit
            decided[tp_mask] = True
        if torch.any(sl_mask):
            exit_t[sl_mask] = t
            reason[sl_mask] = 6  # sl_hit
            decided[sl_mask] = True

        after_min_hold = t >= min_hold_steps
        t_sec = float(t) * float(step_sec)

        # Dynamic policy metrics (vectorized via numpy -> torch)
        tau_sec = (H_steps - t).clamp(min=0).to(torch.float32) * float(step_sec)
        p_pos_cur, p_pos_alt, ev_cur, ev_alt, p_sl_cur, p_sl_alt, p_tp_cur, p_tp_alt = _policy_metrics_batch(
            float(mu_ps),
            float(sigma_ps),
            to_numpy(tau_sec),
            to_numpy(side_now),
            float(leverage),
            float(fee_exit_only),
            to_numpy(tp_target),
            to_numpy(sl_target),
        )
        p_pos_cur_t = to_torch(p_pos_cur, device=device)
        p_pos_alt_t = to_torch(p_pos_alt, device=device)
        ev_cur_t = to_torch(ev_cur, device=device)
        ev_alt_t = to_torch(ev_alt, device=device)
        p_sl_cur_t = to_torch(p_sl_cur, device=device)
        p_sl_alt_t = to_torch(p_sl_alt, device=device)
        p_tp_cur_t = to_torch(p_tp_cur, device=device)
        p_tp_alt_t = to_torch(p_tp_alt, device=device)

        gap_eff = ev_cur_t - (ev_alt_t - switch_cost)
        alt_value_after_cost = ev_alt_t - float(exec_oneway)

        flip_ok = (p_pos_alt_t >= float(p_pos_floor_enter)) & (
            (alt_value_after_cost > 0.0) | (alt_value_after_cost > float(soft_floor))
        )
        flip_ok = flip_ok & (p_sl_alt_t <= float(p_sl_enter_ceiling)) & (p_tp_alt_t >= float(p_tp_floor_enter))

        hold_ok = (gap_eff >= -float(score_margin)) & (p_pos_cur_t >= float(p_pos_floor_hold))
        hold_ok = hold_ok & (p_sl_cur_t <= float(p_sl_hold_ceiling)) & (p_tp_cur_t >= float(p_tp_floor_hold))

        pref_alt = (gap_eff < -float(score_margin)) & flip_ok
        flip_streak = torch.where(
            pref_alt[:, None] & active_mask,
            flip_streak + 1,
            torch.zeros_like(flip_streak),
        )

        bad_hold = (ev_cur_t < -float(score_margin)) | (~hold_ok)
        hold_bad = torch.where(
            bad_hold[:, None] & active_mask,
            hold_bad + 1,
            torch.zeros_like(hold_bad),
        )

        grace_sec = torch.minimum(torch.tensor(60.0, device=device), 0.25 * min_hold_sec)
        is_emergency = (p_sl_cur_t >= float(p_sl_emergency)) & (t_sec >= grace_sec) & after_min_hold
        emerg_mask = active_mask & is_emergency[:, None]
        if torch.any(emerg_mask):
            exit_t[emerg_mask] = t
            reason[emerg_mask] = 1
            decided[emerg_mask] = True

        do_flip = (flip_streak >= int(flip_confirm_ticks)) & after_min_hold[:, None] & active_mask
        do_hold_bad = (hold_bad >= int(hold_bad_ticks)) & after_min_hold[:, None] & active_mask
        if torch.any(do_flip):
            exit_t[do_flip] = t
            reason[do_flip] = 2
            decided[do_flip] = True
        if torch.any(do_hold_bad):
            exit_t[do_hold_bad] = t
            reason[do_hold_bad] = 3
            decided[do_hold_bad] = True

        if enable_dd_stop:
            logret_t = logret[:, t]
            realized_t = side_now[:, None] * float(leverage) * logret_t[None, :]
            dd_mask = active_mask & after_min_hold[:, None] & (realized_t <= dd_stop[:, None])
            if torch.any(dd_mask):
                exit_t[dd_mask] = t
                reason[dd_mask] = 4
                decided[dd_mask] = True

        if torch.all(decided | (~active[:, None])):
            break

    exit_t_clipped = torch.clamp(exit_t, 0, max_steps_avail)
    logret_expand = logret.unsqueeze(0).expand(batch_size, -1, -1)
    idx = exit_t_clipped.unsqueeze(-1)
    logret_at_exit = torch.gather(logret_expand, dim=2, index=idx).squeeze(-1)
    realized = side_now[:, None] * float(leverage) * logret_at_exit
    net_out = realized - float(fee_roundtrip)

    p_pos_exit = torch.mean((net_out > 0.0).float(), dim=1)
    ev_exit = torch.mean(net_out, dim=1)
    exit_t_mean_sec = torch.mean(exit_t_clipped.float(), dim=1) * float(step_sec)

    k = max(1, int(net_out.shape[1] * float(cvar_alpha)))
    sorted_net, _ = torch.sort(net_out, dim=1)
    cvar_exit = torch.mean(sorted_net[:, :k], dim=1)

    p_tp = torch.mean((reason == 5).float(), dim=1)
    p_sl = torch.mean((reason == 6).float(), dim=1)
    p_other = torch.clamp(1.0 - p_tp - p_sl, 0.0, 1.0)

    return {
        "p_pos_exit": p_pos_exit,
        "ev_exit": ev_exit,
        "exit_t_mean_sec": exit_t_mean_sec,
        "reason_idx": reason,
        "cvar_exit": cvar_exit,
        "p_tp": p_tp,
        "p_sl": p_sl,
        "p_other": p_other,
        "net_out": net_out,
        "exit_t": exit_t_clipped,
    }


def simulate_exit_policy_multi_symbol_torch(*args, **kwargs):
    """Multi-symbol wrapper (sequential per symbol to control memory)."""
    price_paths_batch = args[0]
    if not isinstance(price_paths_batch, torch.Tensor):
        price_paths_batch = to_torch(price_paths_batch, device=get_torch_device())

    num_symbols = int(price_paths_batch.shape[0])
    results = {
        "p_pos_exit": [],
        "ev_exit": [],
        "exit_t_mean_sec": [],
        "reason_idx": [],
        "cvar_exit": [],
        "p_tp": [],
        "p_sl": [],
        "p_other": [],
        "net_out": [],
        "exit_t": [],
    }

    # Unpack scalar/vector args as tensors for indexing
    s0_batch = np.asarray(args[1])
    mu_ps_batch = np.asarray(args[2])
    sigma_ps_batch = np.asarray(args[3])
    leverage_batch = np.asarray(args[4])
    fee_roundtrip_batch = np.asarray(args[5])
    exec_oneway_batch = np.asarray(args[6])
    impact_cost_batch = np.asarray(args[7])

    decision_dt_sec = int(args[8])
    step_sec = int(args[9])
    max_horizon_sec = int(args[10])

    side_now_batch = np.asarray(args[11])
    horizon_sec_batch = np.asarray(args[12])
    min_hold_sec_batch = np.asarray(args[13])
    tp_target_roe_batch = np.asarray(args[14])
    sl_target_roe_batch = np.asarray(args[15])

    p_pos_floor_enter = float(args[16])
    p_pos_floor_hold = float(args[17])
    p_sl_enter_ceiling = float(args[18])
    p_sl_hold_ceiling = float(args[19])
    p_sl_emergency = float(args[20])
    p_tp_floor_enter = float(args[21])
    p_tp_floor_hold = float(args[22])
    score_margin = float(args[23])
    soft_floor = float(args[24])
    enable_dd_stop = bool(args[25])
    dd_stop_roe_batch = np.asarray(args[26])
    flip_confirm_ticks = int(args[27])
    hold_bad_ticks = int(args[28])
    fee_exit_only_override = float(args[29])
    cvar_alpha = float(args[30])

    # Try vmap if available
    torch_vmap = None
    try:
        from torch.func import vmap as torch_vmap  # type: ignore
    except Exception:
        torch_vmap = getattr(torch, "vmap", None)

    if torch_vmap is not None:
        def _rollforward_tuple(
            price_paths_i,
            s0_i,
            mu_i,
            sigma_i,
            leverage_i,
            fee_rt_i,
            exec_ow_i,
            impact_i,
            side_i,
            horizon_i,
            min_hold_i,
            tp_i,
            sl_i,
            dd_stop_i,
        ):
            res = simulate_exit_policy_rollforward_batched_vmap_torch(
                price_paths_i,
                float(s0_i),
                float(mu_i),
                float(sigma_i),
                float(leverage_i),
                float(fee_rt_i),
                float(exec_ow_i),
                float(impact_i),
                decision_dt_sec,
                step_sec,
                max_horizon_sec,
                side_i,
                horizon_i,
                min_hold_i,
                tp_i,
                sl_i,
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
                dd_stop_i,
                flip_confirm_ticks,
                hold_bad_ticks,
                fee_exit_only_override,
                cvar_alpha,
            )
            return (
                res["p_pos_exit"],
                res["ev_exit"],
                res["exit_t_mean_sec"],
                res["reason_idx"],
                res["cvar_exit"],
                res["p_tp"],
                res["p_sl"],
                res["p_other"],
                res["net_out"],
                res["exit_t"],
            )

        tup = torch_vmap(
            _rollforward_tuple,
            in_dims=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )(
            price_paths_batch,
            to_torch(s0_batch, device=get_torch_device()),
            to_torch(mu_ps_batch, device=get_torch_device()),
            to_torch(sigma_ps_batch, device=get_torch_device()),
            to_torch(leverage_batch, device=get_torch_device()),
            to_torch(fee_roundtrip_batch, device=get_torch_device()),
            to_torch(exec_oneway_batch, device=get_torch_device()),
            to_torch(impact_cost_batch, device=get_torch_device()),
            to_torch(side_now_batch, device=get_torch_device()),
            to_torch(horizon_sec_batch, device=get_torch_device()),
            to_torch(min_hold_sec_batch, device=get_torch_device()),
            to_torch(tp_target_roe_batch, device=get_torch_device()),
            to_torch(sl_target_roe_batch, device=get_torch_device()),
            to_torch(dd_stop_roe_batch, device=get_torch_device()),
        )
        return {
            "p_pos_exit": tup[0],
            "ev_exit": tup[1],
            "exit_t_mean_sec": tup[2],
            "reason_idx": tup[3],
            "cvar_exit": tup[4],
            "p_tp": tup[5],
            "p_sl": tup[6],
            "p_other": tup[7],
            "net_out": tup[8],
            "exit_t": tup[9],
        }

    # Fallback: sequential per symbol
    for i in range(num_symbols):
        res = simulate_exit_policy_rollforward_batched_vmap_torch(
            price_paths_batch[i],
            float(s0_batch[i]),
            float(mu_ps_batch[i]),
            float(sigma_ps_batch[i]),
            float(leverage_batch[i]),
            float(fee_roundtrip_batch[i]),
            float(exec_oneway_batch[i]),
            float(impact_cost_batch[i]),
            decision_dt_sec,
            step_sec,
            max_horizon_sec,
            side_now_batch[i],
            horizon_sec_batch[i],
            min_hold_sec_batch[i],
            tp_target_roe_batch[i],
            sl_target_roe_batch[i],
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
            dd_stop_roe_batch[i],
            flip_confirm_ticks,
            hold_bad_ticks,
            fee_exit_only_override,
            cvar_alpha,
        )
        for k in results:
            results[k].append(res[k])

    out = {}
    for k, v in results.items():
        out[k] = torch.stack(v, dim=0)
    return out

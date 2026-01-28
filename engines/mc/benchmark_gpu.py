from __future__ import annotations

import time
import numpy as np

from engines.mc.torch_backend import _TORCH_OK, torch
from engines.mc.exit_policy_torch import simulate_exit_policy_rollforward_batched_vmap_torch
from engines.exit_policy_methods import simulate_exit_policy_rollforward


def benchmark():
    if not _TORCH_OK or torch is None:
        raise RuntimeError("PyTorch not available for GPU benchmark")

    # Parameters
    n_paths = 2048
    h_pts = 301
    s0 = 100.0
    mu_ps = 0.0001 / 31536000.0
    sigma_ps = 0.4 / np.sqrt(31536000.0)
    leverage = 5.0
    fee_roundtrip = 0.0012
    exec_oneway = 0.0003
    impact_cost = 0.0001
    decision_dt_sec = 5
    horizon_sec = 300
    min_hold_sec = 10

    rng = np.random.default_rng(42)
    price_paths = s0 * np.exp(np.cumsum(rng.normal(0, sigma_ps, (n_paths, h_pts - 1)), axis=1))
    price_paths = np.column_stack([np.full(n_paths, s0), price_paths])

    print(f"Benchmarking with {n_paths} paths, {h_pts} steps...")

    # Torch warmup
    print("Warming up Torch...")
    _ = simulate_exit_policy_rollforward_batched_vmap_torch(
        price_paths=price_paths,
        s0=s0, mu_ps=mu_ps, sigma_ps=sigma_ps, leverage=leverage,
        fee_roundtrip=fee_roundtrip, exec_oneway=exec_oneway, impact_cost=impact_cost,
        decision_dt_sec=decision_dt_sec, step_sec=1, max_horizon_sec=horizon_sec,
        side_now_batch=np.array([1], dtype=np.int32),
        horizon_sec_batch=np.array([horizon_sec], dtype=np.int32),
        min_hold_sec_batch=np.array([min_hold_sec], dtype=np.int32),
        tp_target_roe_batch=np.array([0.003], dtype=np.float32),
        sl_target_roe_batch=np.array([0.003], dtype=np.float32),
        p_pos_floor_enter=0.52, p_pos_floor_hold=0.5,
        p_sl_enter_ceiling=0.2, p_sl_hold_ceiling=0.25,
        p_sl_emergency=0.38, p_tp_floor_enter=0.15, p_tp_floor_hold=0.12,
        score_margin=0.0001, soft_floor=-0.0005,
        enable_dd_stop=True, dd_stop_roe_batch=np.array([-0.02], dtype=np.float32),
        flip_confirm_ticks=2, hold_bad_ticks=2,
        cvar_alpha=0.05, unified_lambda=1.0, unified_rho=0.0,
    )

    # CPU Benchmark
    print("Running CPU Benchmark...")
    start_cpu = time.time()
    _ = simulate_exit_policy_rollforward(
        price_paths=price_paths,
        s0=s0, mu=mu_ps, sigma=sigma_ps, leverage=leverage,
        fee_roundtrip=fee_roundtrip, exec_oneway=exec_oneway, impact_cost=impact_cost,
        regime="chop", decision_dt_sec=decision_dt_sec, horizon_sec=horizon_sec,
        min_hold_sec=min_hold_sec, flip_confirm_ticks=2, hold_bad_ticks=2,
        side_now=1, enable_dd_stop=True, dd_stop_roe=-0.02
    )
    end_cpu = time.time()
    print(f"CPU Time: {end_cpu - start_cpu:.4f}s")

    # Torch Benchmark
    print("Running Torch Benchmark...")
    start_gpu = time.time()
    _ = simulate_exit_policy_rollforward_batched_vmap_torch(
        price_paths=price_paths,
        s0=s0, mu_ps=mu_ps, sigma_ps=sigma_ps, leverage=leverage,
        fee_roundtrip=fee_roundtrip, exec_oneway=exec_oneway, impact_cost=impact_cost,
        decision_dt_sec=decision_dt_sec, step_sec=1, max_horizon_sec=horizon_sec,
        side_now_batch=np.array([1], dtype=np.int32),
        horizon_sec_batch=np.array([horizon_sec], dtype=np.int32),
        min_hold_sec_batch=np.array([min_hold_sec], dtype=np.int32),
        tp_target_roe_batch=np.array([0.003], dtype=np.float32),
        sl_target_roe_batch=np.array([0.003], dtype=np.float32),
        p_pos_floor_enter=0.52, p_pos_floor_hold=0.5,
        p_sl_enter_ceiling=0.2, p_sl_hold_ceiling=0.25,
        p_sl_emergency=0.38, p_tp_floor_enter=0.15, p_tp_floor_hold=0.12,
        score_margin=0.0001, soft_floor=-0.0005,
        enable_dd_stop=True, dd_stop_roe_batch=np.array([-0.02], dtype=np.float32),
        flip_confirm_ticks=2, hold_bad_ticks=2,
        cvar_alpha=0.05, unified_lambda=1.0, unified_rho=0.0,
    )
    end_gpu = time.time()
    print(f"Torch Time: {end_gpu - start_gpu:.4f}s")


if __name__ == "__main__":
    benchmark()

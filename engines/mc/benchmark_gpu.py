from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp
from engines.mc.exit_policy_jax import simulate_exit_policy_rollforward_jax
from engines.exit_policy_methods import simulate_exit_policy_rollforward

# Configure JAX to use Metal backend
jax.config.update('jax_platform_name', 'METAL')
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

def benchmark():
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
    
    # Generate prices
    rng = np.random.default_rng(42)
    price_paths = s0 * np.exp(np.cumsum(rng.normal(0, sigma_ps, (n_paths, h_pts-1)), axis=1))
    price_paths = np.column_stack([np.full(n_paths, s0), price_paths])
    
    print(f"Benchmarking with {n_paths} paths, {h_pts} steps...")
    
    # JAX - Warmup
    print("Warming up JAX...")
    _ = simulate_exit_policy_rollforward_jax(
        price_paths=jnp.asarray(price_paths, dtype=jnp.float32),
        s0=s0, mu_ps=mu_ps, sigma_ps=sigma_ps, leverage=leverage,
        fee_roundtrip=fee_roundtrip, exec_oneway=exec_oneway, impact_cost=impact_cost,
        step_sec=1,
        decision_dt_sec=decision_dt_sec, horizon_sec=horizon_sec, min_hold_sec=min_hold_sec,
        flip_confirm_ticks=2, hold_bad_ticks=2,
        p_pos_floor_enter=0.52, p_pos_floor_hold=0.5,
        p_sl_enter_ceiling=0.2, p_sl_hold_ceiling=0.25,
        p_sl_emergency=0.38, p_tp_floor_enter=0.15, p_tp_floor_hold=0.12,
        score_margin=0.0001, soft_floor=-0.0005,
        side_now=1, enable_dd_stop=True, dd_stop_roe=-0.02
    )
    
    # CPU Benchmark
    print("Running CPU Benchmark...")
    start_cpu = time.time()
    res_cpu = simulate_exit_policy_rollforward(
        price_paths=price_paths,
        s0=s0, mu=mu_ps, sigma=sigma_ps, leverage=leverage,
        fee_roundtrip=fee_roundtrip, exec_oneway=exec_oneway, impact_cost=impact_cost,
        regime="chop", decision_dt_sec=decision_dt_sec, horizon_sec=horizon_sec,
        min_hold_sec=min_hold_sec, flip_confirm_ticks=2, hold_bad_ticks=2,
        side_now=1, enable_dd_stop=True, dd_stop_roe=-0.02
    )
    end_cpu = time.time()
    print(f"CPU Time: {end_cpu - start_cpu:.4f}s")
    
    # GPU Benchmark
    print("Running GPU Benchmark...")
    start_gpu = time.time()
    res_gpu = simulate_exit_policy_rollforward_jax(
        price_paths=jnp.asarray(price_paths, dtype=jnp.float32),
        s0=s0, mu_ps=mu_ps, sigma_ps=sigma_ps, leverage=leverage,
        fee_roundtrip=fee_roundtrip, exec_oneway=exec_oneway, impact_cost=impact_cost,
        step_sec=1,
        decision_dt_sec=decision_dt_sec, horizon_sec=horizon_sec, min_hold_sec=min_hold_sec,
        flip_confirm_ticks=2, hold_bad_ticks=2,
        p_pos_floor_enter=0.52, p_pos_floor_hold=0.5,
        p_sl_enter_ceiling=0.2, p_sl_hold_ceiling=0.25,
        p_sl_emergency=0.38, p_tp_floor_enter=0.15, p_tp_floor_hold=0.12,
        score_margin=0.0001, soft_floor=-0.0005,
        side_now=1, enable_dd_stop=True, dd_stop_roe=-0.02
    )
    # Ensure completion
    _ = res_gpu["ev_exit"].block_until_ready()
    end_gpu = time.time()
    print(f"GPU Time: {end_gpu - start_gpu:.4f}s")
    
    print("\nResults Comparison:")
    print(f"CPU EV: {res_cpu['ev_exit']:.6f}")
    print(f"GPU EV: {res_gpu['ev_exit']:.6f}")
    print(f"CPU p_pos: {res_cpu['p_pos_exit']:.4f}")
    print(f"GPU p_pos: {res_gpu['p_pos_exit']:.4f}")
    
    speedup = (end_cpu - start_cpu) / (end_gpu - start_gpu)
    print(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark()

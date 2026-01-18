"""
Global Batching Entry Evaluation with JAX vmap
================================================

이 모듈은 여러 심볼에 대한 Monte Carlo 평가를 병렬로 수행합니다.

핵심 최적화:
1. 통계/정렬을 JAX 내부에서 수행 (CPUSum 제거)
2. 이중 vmap 사용 (심볼 × horizon 병렬화)
3. 단일 JIT 함수 호출 (asyncio 루프 제거)
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Level 1: 단일 심볼, 단일 horizon에 대한 시뮬레이션 및 통계 계산
# ============================================================================

@jax.jit
def compute_horizon_metrics_jax(
    price_paths: jnp.ndarray,  # (n_paths, n_steps+1)
    horizon_idx: int,
    leverage: float,
    fee_roundtrip: float,
    tp_target_roe: float,
    sl_target_roe: float,
    cvar_alpha: float = 0.95,
) -> Dict[str, jnp.ndarray]:
    """
    단일 horizon에 대한 메트릭을 GPU 내부에서 계산
    
    Returns:
        - ev_long, ev_short: Expected Value
        - p_pos_long, p_pos_short: Probability of positive return
        - cvar_long, cvar_short: Conditional Value at Risk
        - p_tp_long, p_sl_long: TP/SL hit probabilities
    """
    # 가격 경로에서 해당 horizon의 가격 추출
    s0 = price_paths[:, 0]
    st = price_paths[:, horizon_idx]
    
    # Gross return 계산
    gross_ret = (st - s0) / jnp.maximum(s0, 1e-12)
    
    # Long/Short ROE 계산
    gross_long = gross_ret * leverage
    gross_short = -gross_ret * leverage
    
    net_long = gross_long - fee_roundtrip * leverage
    net_short = gross_short - fee_roundtrip * leverage
    
    # TP/SL 체크
    tp_hit_long = net_long >= tp_target_roe
    sl_hit_long = net_long <= sl_target_roe
    tp_hit_short = net_short >= tp_target_roe
    sl_hit_short = net_short <= sl_target_roe
    
    # 확률 계산 (GPU에서 직접 수행)
    p_tp_long = jnp.mean(tp_hit_long.astype(jnp.float32))
    p_sl_long = jnp.mean(sl_hit_long.astype(jnp.float32))
    p_tp_short = jnp.mean(tp_hit_short.astype(jnp.float32))
    p_sl_short = jnp.mean(sl_hit_short.astype(jnp.float32))
    
    # EV 계산
    ev_long = jnp.mean(net_long)
    ev_short = jnp.mean(net_short)
    
    # P(positive) 계산
    p_pos_long = jnp.mean((net_long > 0).astype(jnp.float32))
    p_pos_short = jnp.mean((net_short > 0).astype(jnp.float32))
    
    # CVaR 계산 (GPU에서 수행)
    # JAX JIT 호환: percentile 사용
    # cvar_alpha=0.95 → 하위 5% 평균
    percentile_threshold = (1.0 - cvar_alpha) * 100  # 5.0
    
    # 하위 5% 지점의 값
    threshold_long = jnp.percentile(net_long, percentile_threshold)
    threshold_short = jnp.percentile(net_short, percentile_threshold)
    
    # 해당 값보다 작거나 같은 값들의 평균
    mask_long = net_long <= threshold_long
    mask_short = net_short <= threshold_short
    
    cvar_long = jnp.where(
        jnp.sum(mask_long) > 0,
        jnp.sum(net_long * mask_long) / jnp.maximum(1, jnp.sum(mask_long)),
        threshold_long
    )
    cvar_short = jnp.where(
        jnp.sum(mask_short) > 0,
        jnp.sum(net_short * mask_short) / jnp.maximum(1, jnp.sum(mask_short)),
        threshold_short
    )
    
    return {
        "ev_long": ev_long,
        "ev_short": ev_short,
        "p_pos_long": p_pos_long,
        "p_pos_short": p_pos_short,
        "cvar_long": cvar_long,
        "cvar_short": cvar_short,
        "p_tp_long": p_tp_long,
        "p_sl_long": p_sl_long,
        "p_tp_short": p_tp_short,
        "p_sl_short": p_sl_short,
    }


# ============================================================================
# Level 2: 단일 심볼, 여러 horizon 병렬 처리 (vmap Level 1)
# ============================================================================

def compute_all_horizons_jax(
    price_paths: jnp.ndarray,  # (n_paths, n_steps+1)
    horizon_indices: jnp.ndarray,  # (n_horizons,)
    leverage: float,
    fee_roundtrip: float,
    tp_targets: jnp.ndarray,  # (n_horizons,)
    sl_targets: jnp.ndarray,  # (n_horizons,)
    cvar_alpha: float = 0.95,
) -> Dict[str, jnp.ndarray]:
    """
    여러 horizon에 대한 메트릭을 동시에 계산
    
    vmap을 사용하여 horizon 차원을 병렬화
    """
    # vmap over horizon dimension
    def compute_single_horizon(h_idx, tp_tgt, sl_tgt):
        return compute_horizon_metrics_jax(
            price_paths, h_idx, leverage, fee_roundtrip,
            tp_tgt, sl_tgt, cvar_alpha
        )
    
    # vmap: horizon별로 병렬 실행
    results = jax.vmap(compute_single_horizon)(
        horizon_indices, tp_targets, sl_targets
    )
    
    return results


# ============================================================================
# Level 3: 여러 심볼 병렬 처리 (vmap Level 2 - Global Batching)
# ============================================================================

@jax.jit
def compute_portfolio_metrics_jax(
    price_paths_batch: jnp.ndarray,  # (n_symbols, n_paths, n_steps+1)
    horizon_indices: jnp.ndarray,  # (n_horizons,)
    leverages: jnp.ndarray,  # (n_symbols,)
    fee_roundtrips: jnp.ndarray,  # (n_symbols,)
    tp_targets_batch: jnp.ndarray,  # (n_symbols, n_horizons)
    sl_targets_batch: jnp.ndarray,  # (n_symbols, n_horizons)
    cvar_alpha: float = 0.95,
) -> Dict[str, jnp.ndarray]:
    """
    여러 심볼에 대한 메트릭을 동시에 계산 (Global Batching)
    
    이중 vmap:
    - 외부 vmap: 심볼 차원
    - 내부 vmap: horizon 차원
    
    Returns:
        각 메트릭의 shape: (n_symbols, n_horizons)
    """
    # vmap over symbol dimension
    def compute_symbol_metrics(paths, lev, fee, tp_tgts, sl_tgts):
        return compute_all_horizons_jax(
            paths, horizon_indices, lev, fee,
            tp_tgts, sl_tgts, cvar_alpha
        )
    
    # vmap: 심볼별로 병렬 실행
    results = jax.vmap(compute_symbol_metrics)(
        price_paths_batch,
        leverages,
        fee_roundtrips,
        tp_targets_batch,
        sl_targets_batch,
    )
    
    return results


# ============================================================================
# High-Level API: Python에서 호출하는 함수
# ============================================================================

class GlobalBatchEvaluator:
    """
    Global Batching을 사용한 Entry Evaluation
    """
    
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        
        # JIT 컴파일된 함수 (워밍업 필요)
        self.jit_compute = jax.jit(compute_portfolio_metrics_jax)
        
        # 워밍업 플래그
        self.warmed_up = False
    
    def warmup(self):
        """JIT 컴파일 워밍업"""
        if self.warmed_up:
            return
        
        logger.info("[VMAP] Warming up JIT compilation...")
        
        # 더미 데이터로 워밍업
        dummy_paths = jnp.ones((2, 100, 301))  # 2 symbols, 100 paths, 300 steps
        dummy_horizons = jnp.array([60, 180])
        dummy_leverages = jnp.ones(2)
        dummy_fees = jnp.ones(2) * 0.001
        dummy_tp = jnp.ones((2, 2)) * 0.01
        dummy_sl = jnp.ones((2, 2)) * -0.005
        
        
        result = self.jit_compute(
            dummy_paths, dummy_horizons, dummy_leverages,
            dummy_fees, dummy_tp, dummy_sl
        )
        # dict의 모든 값에 block_until_ready 적용
        jax.tree_map(lambda x: x.block_until_ready(), result)
        
        self.warmed_up = True
        logger.info("[VMAP] Warmup complete!")
    
    def evaluate_batch(
        self,
        price_paths_batch: np.ndarray,  # (n_symbols, n_paths, n_steps+1)
        horizons: List[int],  # [300, 600, 1800, 3600]
        leverages: np.ndarray,  # (n_symbols,)
        fee_roundtrips: np.ndarray,  # (n_symbols,)
        tp_targets_batch: np.ndarray,  # (n_symbols, n_horizons)
        sl_targets_batch: np.ndarray,  # (n_symbols, n_horizons)
        cvar_alpha: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """
        배치 평가 수행
        
        Returns:
            각 메트릭의 shape: (n_symbols, n_horizons)
        """
        # 워밍업
        if not self.warmed_up:
            self.warmup()
        
        # Numpy → JAX 변환
        paths_jax = jnp.array(price_paths_batch)
        horizons_jax = jnp.array(horizons, dtype=jnp.int32)
        leverages_jax = jnp.array(leverages)
        fees_jax = jnp.array(fee_roundtrips)
        tp_jax = jnp.array(tp_targets_batch)
        sl_jax = jnp.array(sl_targets_batch)
        
        # 단일 JIT 함수 호출
        t0 = time.perf_counter()
        results = self.jit_compute(
            paths_jax, horizons_jax, leverages_jax,
            fees_jax, tp_jax, sl_jax, cvar_alpha
        )
        
        # GPU 동기화
        results = jax.tree_map(lambda x: np.array(x), results)
        t1 = time.perf_counter()
        
        logger.info(f"[VMAP] Global batch evaluation: {len(leverages)} symbols × {len(horizons)} horizons in {(t1-t0)*1000:.2f}ms")
        
        return results


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == "__main__":
    # 테스트
    evaluator = GlobalBatchEvaluator()
    
    # 더미 데이터
    n_symbols = 18
    n_paths = 4096
    n_steps = 3600
    horizons = [300, 600, 1800, 3600]
    
    # 가격 경로 생성 (실제로는 simulate_paths_price_batch에서 생성)
    price_paths = np.random.randn(n_symbols, n_paths, n_steps + 1).cumsum(axis=2) * 0.01 + 100
    
    leverages = np.ones(n_symbols) * 10.0
    fees = np.ones(n_symbols) * 0.0015
    tp_targets = np.ones((n_symbols, len(horizons))) * 0.01
    sl_targets = np.ones((n_symbols, len(horizons))) * -0.005
    
    # 평가
    results = evaluator.evaluate_batch(
        price_paths, horizons, leverages, fees,
        tp_targets, sl_targets
    )
    
    print(f"Results shape: {results['ev_long'].shape}")  # (18, 4)
    print(f"Sample EV (long): {results['ev_long'][0]}")

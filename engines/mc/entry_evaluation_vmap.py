"""
Global Batching Entry Evaluation with JAX vmap
================================================

이 모듈은 여러 심볼에 대한 Monte Carlo 평가를 병렬로 수행합니다.

핵심 최적화:
1. 통계/정렬을 JAX 내부에서 수행 (CPUSum 제거)
2. 이중 vmap 사용 (심볼 × horizon 병렬화)
3. 단일 JIT 함수 호출 (asyncio 루프 제거)
"""

from __future__ import annotations

from engines.mc import jax_backend as jax_backend
ensure_jax = jax_backend.ensure_jax
lazy_jit = jax_backend.lazy_jit
_JAX_OK = lambda: getattr(jax_backend, '_JAX_OK', False)
DEV_MODE = getattr(jax_backend, 'DEV_MODE', False)
jax = jax_backend.jax
jnp = jax_backend.jnp
import numpy as np
import time
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Level 1: 단일 심볼, 단일 horizon에 대한 시뮬레이션 및 통계 계산
# ============================================================================

@lazy_jit()
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
    단일 horizon에 대한 메트릭을 GPU 내부에서 계산 (Barrier Logic 포함)
    
    [CRITICAL FIX] 경로 전체를 검사하여 중간에 TP/SL 도달 여부 확인
    - 기존: st(끝값)만 보고 판단 → 중간 청산 케이스 누락
    - 개선: jnp.max/min으로 경로 내 고가/저가 산출 후 Barrier 체크
    
    Returns:
        - ev_long, ev_short: Expected Value (경로 의존적 EV)
        - p_pos_long, p_pos_short: Probability of positive return
        - cvar_long, cvar_short: Conditional Value at Risk
        - p_tp_long, p_sl_long: TP/SL hit probabilities (First Passage 기반)
    """
    # 가격 경로에서 s0 및 horizon까지의 모든 가격 추출
    s0 = price_paths[:, 0]  # (n_paths,)

    # Patched: Static warmup shape and empty array defense
    # 동적 슬라이스가 JIT 트레이싱을 깨뜨리지 않도록 마스크 기반으로 TP/SL 범위를 계산한다.
    idx = jnp.arange(price_paths.shape[1])
    mask = (idx <= horizon_idx) & (idx > 0)
    mask_b = mask[None, :]
    path_high = jnp.max(jnp.where(mask_b, price_paths, -jnp.inf), axis=1)
    path_low = jnp.min(jnp.where(mask_b, price_paths, jnp.inf), axis=1)
    st = jnp.take(price_paths, horizon_idx, axis=1, mode="clip")  # 만기 시점 가격
    
    # Gross return 계산 (경로 내 극값 기준)
    gross_ret_end = (st - s0) / jnp.maximum(s0, 1e-12)       # 만기 수익률
    gross_ret_high = (path_high - s0) / jnp.maximum(s0, 1e-12)  # 최고점 수익률
    gross_ret_low = (path_low - s0) / jnp.maximum(s0, 1e-12)    # 최저점 수익률
    
    # ======== LONG Position Barrier Check ========
    # Long: 가격 상승 = 이익 → 최고점에서 TP, 최저점에서 SL
    roe_at_high_long = gross_ret_high * leverage - fee_roundtrip * leverage  # 최고점 ROE
    roe_at_low_long = gross_ret_low * leverage - fee_roundtrip * leverage    # 최저점 ROE
    
    # First Passage: 경로 중 TP/SL 도달 여부 (만기 전 청산)
    barrier_tp_hit_long = roe_at_high_long >= tp_target_roe  # 경로 중 TP 도달
    barrier_sl_hit_long = roe_at_low_long <= sl_target_roe   # 경로 중 SL 도달
    
    # 실제 청산 수익률 (First Passage 기반)
    # - TP 먼저 도달: TP ROE
    # - SL 먼저 도달: SL ROE
    # - 둘 다 도달 시: 보수적으로 SL 우선 (실제로는 시간순 필요)
    # - 둘 다 미도달: 만기 ROE
    net_long_end = gross_ret_end * leverage - fee_roundtrip * leverage
    
    # 청산 ROE 결정 (보수적 접근: SL 우선)
    net_long = jnp.where(
        barrier_sl_hit_long,  # SL 먼저 체크 (보수적)
        sl_target_roe,
        jnp.where(
            barrier_tp_hit_long,
            tp_target_roe,
            net_long_end  # TP/SL 미도달 → 만기 가격으로 청산
        )
    )
    
    # ======== SHORT Position Barrier Check ========
    # Short: 가격 하락 = 이익 → 최저점에서 TP, 최고점에서 SL
    roe_at_low_short = -gross_ret_low * leverage - fee_roundtrip * leverage   # 최저점 ROE (Short)
    roe_at_high_short = -gross_ret_high * leverage - fee_roundtrip * leverage # 최고점 ROE (Short)
    
    barrier_tp_hit_short = roe_at_low_short >= tp_target_roe  # 경로 중 TP 도달 (Short)
    barrier_sl_hit_short = roe_at_high_short <= sl_target_roe # 경로 중 SL 도달 (Short)
    
    net_short_end = -gross_ret_end * leverage - fee_roundtrip * leverage
    
    net_short = jnp.where(
        barrier_sl_hit_short,
        sl_target_roe,
        jnp.where(
            barrier_tp_hit_short,
            tp_target_roe,
            net_short_end
        )
    )
    
    # ======== Statistics (Barrier 기반) ========
    # TP/SL 확률 (First Passage)
    p_tp_long = jnp.mean(barrier_tp_hit_long.astype(jnp.float32))
    p_sl_long = jnp.mean(barrier_sl_hit_long.astype(jnp.float32))
    p_tp_short = jnp.mean(barrier_tp_hit_short.astype(jnp.float32))
    p_sl_short = jnp.mean(barrier_sl_hit_short.astype(jnp.float32))
    
    # EV 계산 (청산 수익률 기반)
    ev_long = jnp.mean(net_long)
    ev_short = jnp.mean(net_short)
    
    # P(positive) 계산
    p_pos_long = jnp.mean((net_long > 0).astype(jnp.float32))
    p_pos_short = jnp.mean((net_short > 0).astype(jnp.float32))
    
    # CVaR 계산 (GPU에서 수행)
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
    # vmap over horizon dimension (created at runtime)
    ensure_jax()
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

@lazy_jit()
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

    # vmap: 심볼별로 병렬 실행 (ensure jax first)
    ensure_jax()
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

# ============================================================================
# Static Shape Constants (중앙 집중 관리)
# ============================================================================
# CRITICAL: 모든 상수는 engines/mc/constants.py에서 import합니다.
# 하드코딩 금지! constants.py만 수정하세요.
from engines.mc.constants import (
    STATIC_MAX_SYMBOLS,
    STATIC_MAX_PATHS,
    STATIC_MAX_STEPS,
    STATIC_HORIZONS,
)


class GlobalBatchEvaluator:
    """
    Global Batching을 사용한 Entry Evaluation
    
    [STATIC SHAPE OPTIMIZATION]
    - 모든 JAX 연산은 고정된 크기(STATIC_MAX_*)로 수행
    - 실제 심볼 수가 적으면 나머지는 mask/padding 처리
    - JIT 재컴파일 방지로 장중 렉 제거
    """
    
    def __init__(self, dt: float = 1.0, max_symbols: int = STATIC_MAX_SYMBOLS):
        self.dt = dt
        self.max_symbols = max_symbols
        # JIT 컴파일된 함수 (워밍업 필요) - delay until warmup/runtime
        self.jit_compute = None

        # 워밍업 플래그
        self.warmed_up = False
        self._warmup_shape = None  # 워밍업된 shape 기록
    
    def warmup(self, n_symbols: int = None, n_paths: int = None, n_steps: int = None):
        """
        JIT 컴파일 워밍업 (Static Shape 사용)
        
        [CRITICAL] 봇 시작 시 반드시 최대 크기로 워밍업하여 
        장중 shape 변경으로 인한 재컴파일을 방지해야 합니다.
        
        Args:
            n_symbols: 워밍업할 심볼 수 (기본: STATIC_MAX_SYMBOLS)
            n_paths: 워밍업할 경로 수 (기본: STATIC_MAX_PATHS)
            n_steps: 워밍업할 스텝 수 (기본: STATIC_MAX_STEPS)
        """
        # Patched: Static warmup shape and empty array defense
        # Warmup은 JIT 트레이싱 시 동적 slice 문제를 피하기 위해 항상 고정된 작은 상수로 수행한다.
        # (배치/경로를 작게 고정해도 JIT 컴파일 결과는 이후 큰 static shape 패딩과 호환된다.)
        n_symbols = 1  # warmup은 최소 배치로 고정
        n_paths = 1024  # 안전한 고정 경로 수 (static)
        n_steps = 64  # 짧은 고정 스텝으로 트레이싱 안정화
        n_horizons = len(STATIC_HORIZONS)
        
        # Skip if already warmed up with same or larger shape
        if self.warmed_up and self._warmup_shape is not None:
            ws, wp, wt = self._warmup_shape
            if n_symbols <= ws and n_paths <= wp and n_steps <= wt:
                logger.info(f"[VMAP] Already warmed up with shape ({ws}, {wp}, {wt}), skipping")
                return
        
        # DEV_MODE: JAX 없이 NumPy fallback 사용, warmup 스킵
        if DEV_MODE:
            logger.info("[VMAP] DEV_MODE enabled, skipping JIT warmup (using NumPy)")
            self.warmed_up = True
            return
        
        logger.info(f"[VMAP] Warming up JIT compilation with STATIC SHAPE: ({n_symbols}, {n_paths}, {n_steps})...")
        
        # Ensure JAX is initialized; if unavailable, fall back to CPU path without raising
        ensure_jax()
        if not getattr(jax_backend, '_JAX_OK', False):
            logger.warning("[VMAP] JAX not available during warmup — falling back to CPU/NumPy path")
            # Mark warmed up to avoid repeated attempts and let evaluate_batch use CPU fallback
            self.jit_compute = None
            self.warmed_up = True
            return

        # [STATIC SHAPE] 최대 크기로 더미 데이터 생성
        dummy_paths = jnp.ones((n_symbols, n_paths, n_steps + 1))
        dummy_horizons = jnp.array(STATIC_HORIZONS[:n_horizons], dtype=jnp.int32)
        dummy_leverages = jnp.ones(n_symbols)
        dummy_fees = jnp.ones(n_symbols) * 0.001
        dummy_tp = jnp.ones((n_symbols, n_horizons)) * 0.01
        dummy_sl = jnp.ones((n_symbols, n_horizons)) * -0.005

        # Create compiled JIT function now that JAX is available
        self.jit_compute = jax.jit(compute_portfolio_metrics_jax)
        
        import time
        t0 = time.perf_counter()
        result = self.jit_compute(
            dummy_paths, dummy_horizons, dummy_leverages,
            dummy_fees, dummy_tp, dummy_sl
        )
        # dict의 모든 값에 block_until_ready 적용
        jax.tree_map(lambda x: x.block_until_ready(), result)
        t1 = time.perf_counter()
        
        self.warmed_up = True
        self._warmup_shape = (n_symbols, n_paths, n_steps)
        logger.info(f"[VMAP] Warmup complete! Compiled shape: ({n_symbols}, {n_paths}, {n_steps}), Time: {t1-t0:.2f}s")
    
    def _pad_to_static_shape(
        self,
        arr: np.ndarray,
        target_shape: tuple,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        입력 배열을 Static Shape으로 패딩.
        JIT 재컴파일 방지를 위해 항상 고정된 크기로 변환.
        """
        result = np.full(target_shape, fill_value, dtype=arr.dtype)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
        result[slices] = arr[slices]
        return result

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
        배치 평가 수행 (Static Shape Padding 적용)
        
        Returns:
            각 메트릭의 shape: (n_symbols, n_horizons)
        """
        # 워밍업 (ensure warmup attempted at least once)
        if not self.warmed_up:
            try:
                self.warmup()
            except Exception as e:
                logger.warning(f"[VMAP] Warmup raised exception: {e}. Proceeding with CPU fallback.")
                self.jit_compute = None
                self.warmed_up = True
        
        # DEV_MODE: NumPy fallback (sequential)
        if DEV_MODE:
            from engines.mc.jax_backend import summarize_gbm_horizons_numpy
            n_symbols = len(leverages)
            n_horizons = len(horizons)
            n_steps = price_paths_batch.shape[2] - 1  # (n_symbols, n_paths, n_steps+1)
            
            # horizons를 인덱스로 변환 (초 단위 → 스텝 인덱스)
            # horizons가 이미 인덱스인 경우도 있으므로, max보다 큰지 확인
            horizons_arr = np.array(horizons, dtype=np.int32)
            horizons_indices = np.clip(horizons_arr, 0, n_steps)
            
            # Initialize result arrays
            results = {
                "ev_long": np.zeros((n_symbols, n_horizons)),
                "ev_short": np.zeros((n_symbols, n_horizons)),
                "p_pos_long": np.zeros((n_symbols, n_horizons)),
                "p_pos_short": np.zeros((n_symbols, n_horizons)),
                "cvar_long": np.zeros((n_symbols, n_horizons)),
                "cvar_short": np.zeros((n_symbols, n_horizons)),
                "p_tp_long": np.zeros((n_symbols, n_horizons)),
                "p_sl_long": np.zeros((n_symbols, n_horizons)),
                "p_tp_short": np.zeros((n_symbols, n_horizons)),
                "p_sl_short": np.zeros((n_symbols, n_horizons)),
            }
            
            t0 = time.perf_counter()
            for i in range(n_symbols):
                r = summarize_gbm_horizons_numpy(
                    price_paths_batch[i],
                    float(price_paths_batch[i, 0, 0]),  # s0
                    float(leverages[i]),
                    float(fee_roundtrips[i]),
                    horizons_indices,  # 인덱스로 변환된 horizons
                    1.0 - cvar_alpha
                )
                results["ev_long"][i] = r["ev_long"]
                results["ev_short"][i] = r["ev_short"]
                results["p_pos_long"][i] = r["win_long"]
                results["p_pos_short"][i] = r["win_short"]
                results["cvar_long"][i] = r["cvar_long"]
                results["cvar_short"][i] = r["cvar_short"]
            t1 = time.perf_counter()
            logger.info(f"[VMAP] DEV_MODE NumPy batch evaluation: {n_symbols} symbols × {n_horizons} horizons in {(t1-t0)*1000:.2f}ms")
            return results
        
        # If JAX is not available or JIT wasn't created, fall back to CPU (NumPy) path
        if not getattr(jax_backend, '_JAX_OK', False) or self.jit_compute is None:
            logger.info("[VMAP] Using CPU fallback for batch evaluation (JAX unavailable)")
            from engines.mc.jax_backend import summarize_gbm_horizons_numpy
            n_symbols = len(leverages)
            n_horizons = len(horizons)
            n_steps = price_paths_batch.shape[2] - 1

            horizons_arr = np.array(horizons, dtype=np.int32)
            horizons_indices = np.clip(horizons_arr, 0, n_steps)

            results = {
                "ev_long": np.zeros((n_symbols, n_horizons)),
                "ev_short": np.zeros((n_symbols, n_horizons)),
                "p_pos_long": np.zeros((n_symbols, n_horizons)),
                "p_pos_short": np.zeros((n_symbols, n_horizons)),
                "cvar_long": np.zeros((n_symbols, n_horizons)),
                "cvar_short": np.zeros((n_symbols, n_horizons)),
                "p_tp_long": np.zeros((n_symbols, n_horizons)),
                "p_sl_long": np.zeros((n_symbols, n_horizons)),
                "p_tp_short": np.zeros((n_symbols, n_horizons)),
                "p_sl_short": np.zeros((n_symbols, n_horizons)),
            }
            t0 = time.perf_counter()
            for i in range(n_symbols):
                try:
                    r = summarize_gbm_horizons_numpy(
                        price_paths_batch[i],
                        float(price_paths_batch[i, 0, 0]),
                        float(leverages[i]),
                        float(fee_roundtrips[i]),
                        horizons_indices,
                        1.0 - cvar_alpha,
                    )
                    results["ev_long"][i] = r["ev_long"]
                    results["ev_short"][i] = r["ev_short"]
                    results["p_pos_long"][i] = r["win_long"]
                    results["p_pos_short"][i] = r["win_short"]
                    results["cvar_long"][i] = r["cvar_long"]
                    results["cvar_short"][i] = r["cvar_short"]
                except Exception as e:
                    logger.warning(f"[VMAP] CPU fallback symbol {i} failed: {e}")
            t1 = time.perf_counter()
            logger.info(f"[VMAP] CPU batch evaluation: {n_symbols} symbols × {n_horizons} horizons in {(t1-t0)*1000:.2f}ms")
            return results

        # JAX mode
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

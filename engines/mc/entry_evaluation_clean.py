"""
Clean Entry Evaluation Wrapper using Global Batching
====================================================

이 모듈은 기존 entry_evaluation.py의 문제를 우회하고,
검증된 Global Batching을 사용하여 horizon 처리를 수행합니다.
"""

import numpy as np
import time
from typing import Dict, List, Any
import logging

from engines.mc.entry_evaluation_vmap import GlobalBatchEvaluator

logger = logging.getLogger(__name__)

# Feature flag
from engines.mc.config import config as mc_config

USE_GLOBAL_BATCHING = bool(getattr(mc_config, "use_global_batching", True))

class CleanEntryEvaluator:
    """
    깨끗한 Entry Evaluation wrapper
    
    GlobalBatchEvaluator를 사용하여 horizon 처리를 수행하고,
    기존 코드와 호환되는 인터페이스를 제공합니다.
    """
    
    def __init__(self):
        self.gpu_evaluator = GlobalBatchEvaluator() if USE_GLOBAL_BATCHING else None
        self.warmup_done = False
        
    def evaluate_horizons(
        self,
        price_paths: np.ndarray,  # (n_paths, n_steps+1)
        horizons: List[int],  # [300, 600, 1800, 3600]
        leverage: float,
        fee_roundtrip: float,
        tp_targets: List[float],  # per horizon
        sl_targets: List[float],  # per horizon
        cvar_alpha: float = 0.95,
    ) -> Dict[str, Any]:
        """
        단일 심볼에 대한 horizon 평가
        
        Returns:
            {
                'ev_long_h': [ev1, ev2, ev3, ev4],
                'ev_short_h': [ev1, ev2, ev3, ev4],
                'p_pos_long_h': [...],
                'p_pos_short_h': [...],
                'cvar_long_h': [...],
                'cvar_short_h': [...],
                'p_tp_long': [...],
                'p_sl_long': [...],
                'p_tp_short': [...],
                'p_sl_short': [...],
            }
        """
        if USE_GLOBAL_BATCHING and self.gpu_evaluator:
            return self._evaluate_gpu(
                price_paths, horizons, leverage, fee_roundtrip,
                tp_targets, sl_targets, cvar_alpha
            )
        else:
            return self._evaluate_cpu(
                price_paths, horizons, leverage, fee_roundtrip,
                tp_targets, sl_targets, cvar_alpha
            )
    
    def _evaluate_gpu(
        self,
        price_paths: np.ndarray,
        horizons: List[int],
        leverage: float,
        fee_roundtrip: float,
        tp_targets: List[float],
        sl_targets: List[float],
        cvar_alpha: float,
    ) -> Dict[str, Any]:
        """GPU 경로 (Global Batching 사용)"""
        
        # 단일 심볼을 배치 형태로 변환 (batch_size=1)
        price_paths_batch = price_paths[np.newaxis, ...]  # (1, n_paths, n_steps+1)
        leverages = np.array([leverage])
        fees = np.array([fee_roundtrip])
        tp_batch = np.array([tp_targets])  # (1, n_horizons)
        sl_batch = np.array([sl_targets])
        # Calls into GlobalBatchEvaluator which may raise if JAX not available
        try:
            results = self.gpu_evaluator.evaluate_batch(
                price_paths_batch, horizons, leverages, fees,
                tp_batch, sl_batch, cvar_alpha
            )
        except RuntimeError as e:
            # Log and fallback to numpy-based evaluator to avoid bubbling exception
            logger = logging.getLogger(__name__)
            logger.warning(f"[ENTRY_EVAL_CLEAN] GPU evaluate failed: {e}. Falling back to CPU NumPy path.")
            from engines.mc.jax_backend import summarize_gbm_horizons_numpy
            # Attempt a best-effort CPU fallback using existing wrapper
            n_symbols = price_paths_batch.shape[0]
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
            }
            for i in range(n_symbols):
                r = summarize_gbm_horizons_numpy(
                    price_paths_batch[i],
                    float(price_paths_batch[i, 0, 0]),
                    float(leverages[i]),
                    float(fees[i]),
                    horizons_indices,
                    1.0 - cvar_alpha,
                )
                results["ev_long"][i] = r["ev_long"]
                results["ev_short"][i] = r["ev_short"]
                results["p_pos_long"][i] = r["win_long"]
                results["p_pos_short"][i] = r["win_short"]
                results["cvar_long"][i] = r["cvar_long"]
                results["cvar_short"][i] = r["cvar_short"]
            # continue to unpack below
        
        # 결과 언팩 (batch_size=1이므로 [0] 인덱스)
        return {
            'ev_long_h': list(results['ev_long'][0]),
            'ev_short_h': list(results['ev_short'][0]),
            'p_pos_long_h': list(results['p_pos_long'][0]),
            'p_pos_short_h': list(results['p_pos_short'][0]),
            'cvar_long_h': list(results['cvar_long'][0]),
            'cvar_short_h': list(results['cvar_short'][0]),
            'p_tp_long': list(results['p_tp_long'][0]),
            'p_sl_long': list(results['p_sl_long'][0]),
            'p_tp_short': list(results['p_tp_short'][0]),
            'p_sl_short': list(results['p_sl_short'][0]),
        }
    
    def _evaluate_cpu(
        self,
        price_paths: np.ndarray,
        horizons: List[int],
        leverage: float,
        fee_roundtrip: float,
        tp_targets: List[float],
        sl_targets: List[float],
        cvar_alpha: float,
    ) -> Dict[str, Any]:
        """CPU 경로 (Numpy 사용)"""
        
        results = {
            'ev_long_h': [],
            'ev_short_h': [],
            'p_pos_long_h': [],
            'p_pos_short_h': [],
            'cvar_long_h': [],
            'cvar_short_h': [],
            'p_tp_long': [],
            'p_sl_long': [],
            'p_tp_short': [],
            'p_sl_short': [],
        }
        
        for i, h_idx in enumerate(horizons):
            tp_target = tp_targets[i]
            sl_target = sl_targets[i]
            
            # 가격 추출
            s0 = price_paths[:, 0]
            st = price_paths[:, h_idx]
            
            # Gross return
            gross_ret = (st - s0) / np.maximum(s0, 1e-12)
            
            # Long/Short ROE
            gross_long = gross_ret * leverage
            gross_short = -gross_ret * leverage
            
            net_long = gross_long - fee_roundtrip * leverage
            net_short = gross_short - fee_roundtrip * leverage
            
            # TP/SL 체크
            tp_hit_long = net_long >= tp_target
            sl_hit_long = net_long <= sl_target
            tp_hit_short = net_short >= tp_target
            sl_hit_short = net_short <= sl_target
            
            # 통계 계산
            results['ev_long_h'].append(float(np.mean(net_long)))
            results['ev_short_h'].append(float(np.mean(net_short)))
            results['p_pos_long_h'].append(float(np.mean(net_long > 0)))
            results['p_pos_short_h'].append(float(np.mean(net_short > 0)))
            
            results['p_tp_long'].append(float(np.mean(tp_hit_long)))
            results['p_sl_long'].append(float(np.mean(sl_hit_long)))
            results['p_tp_short'].append(float(np.mean(tp_hit_short)))
            results['p_sl_short'].append(float(np.mean(sl_hit_short)))
            
            # CVaR
            sorted_long = np.sort(net_long)
            sorted_short = np.sort(net_short)
            cutoff = int((1 - cvar_alpha) * len(sorted_long))
            cutoff = max(1, cutoff)
            
            results['cvar_long_h'].append(float(np.mean(sorted_long[:cutoff])))
            results['cvar_short_h'].append(float(np.mean(sorted_short[:cutoff])))
        
        return results


# 전역 인스턴스
_global_evaluator = None

def get_clean_evaluator():
    """전역 evaluator 인스턴스 반환"""
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = CleanEntryEvaluator()
    return _global_evaluator


# 테스트 코드
if __name__ == "__main__":
    print("Testing CleanEntryEvaluator...")
    
    # 더미 데이터
    n_paths = 1024
    n_steps = 3600
    horizons = [300, 600, 1800, 3600]
    
    # 가격 경로 생성
    s0 = 100.0
    drift = 0.05
    vol = 0.2
    
    dW = np.random.randn(n_paths, n_steps) * np.sqrt(1.0)
    log_returns = (drift - 0.5 * vol**2) + vol * dW
    log_prices = np.log(s0) + np.cumsum(log_returns, axis=1)
    prices = np.exp(log_prices)
    
    price_paths = np.concatenate([
        np.ones((n_paths, 1)) * s0,
        prices
    ], axis=1)
    
    # Evaluator 생성
    evaluator = CleanEntryEvaluator()
    
    # 평가
    tp_targets = [0.01 * 10.0] * len(horizons)
    sl_targets = [-0.005 * 10.0] * len(horizons)
    
    results = evaluator.evaluate_horizons(
        price_paths, horizons, 10.0, 0.0015,
        tp_targets, sl_targets
    )
    
    print(f"\nResults:")
    print(f"  ev_long_h: {results['ev_long_h']}")
    print(f"  ev_short_h: {results['ev_short_h']}")
    print(f"  Number of horizons: {len(results['ev_long_h'])}")
    
    assert len(results['ev_long_h']) == 4, "Should have 4 horizons"
    assert len(results['cvar_long_h']) == 4, "Should have 4 CVaR values"
    
    print("\n✅ All tests passed!")

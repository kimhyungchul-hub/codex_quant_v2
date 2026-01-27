"""
Global Batching Entry Evaluation with PyTorch
==============================================

이 모듈은 여러 심볼에 대한 Monte Carlo 평가를 병렬로 수행합니다.
JAX vmap를 PyTorch로 대체한 버전입니다.

핵심 최적화:
1. 통계/정렬을 PyTorch 내부에서 수행 (CPU 통신 최소화)
2. 배치 처리로 심볼 × horizon 병렬화
3. MPS (Metal Performance Shaders) 가속
"""

from __future__ import annotations

import numpy as np
import time
from typing import Dict, List, Any, Tuple
import logging

from engines.mc.torch_backend import _TORCH_OK, torch, get_torch_device, DEV_MODE

logger = logging.getLogger(__name__)

# ============================================================================
# PyTorch 기반 배치 평가 함수들
# ============================================================================

def compute_horizon_metrics_torch(
    price_paths: torch.Tensor,  # (n_paths, n_steps+1)
    horizon_idx: int,
    leverage: float,
    fee_roundtrip: float,
    tp_target_roe: float,
    sl_target_roe: float,
    cvar_alpha: float = 0.95,
) -> Dict[str, float]:
    """
    단일 horizon에 대한 메트릭을 PyTorch로 계산
    
    Returns:
        Dict with ev, std, cvar, tp_prob, sl_prob
    """
    if not _TORCH_OK or price_paths.shape[1] <= horizon_idx:
        return {'ev': 0.0, 'std': 0.0, 'cvar': 0.0, 'tp_prob': 0.0, 'sl_prob': 0.0}
    
    device_local = price_paths.device
    
    # Initial and horizon prices
    s0 = price_paths[:, 0]  # (n_paths,)
    st = price_paths[:, horizon_idx]  # (n_paths,)
    
    # Calculate raw returns and ROE
    raw_ret = (st - s0) / s0  # (n_paths,)
    roe = leverage * raw_ret - fee_roundtrip  # (n_paths,)
    
    # Basic statistics
    ev = float(torch.mean(roe))
    std = float(torch.std(roe))
    
    # CVaR calculation (simplified)
    if cvar_alpha > 0.0 and cvar_alpha < 1.0:
        sorted_roe, _ = torch.sort(roe)
        cutoff_idx = int((1.0 - cvar_alpha) * len(sorted_roe))
        if cutoff_idx > 0:
            cvar = float(torch.mean(sorted_roe[:cutoff_idx]))
        else:
            cvar = float(torch.min(sorted_roe))
    else:
        cvar = ev
    
    # TP/SL probabilities (path-based check)
    if price_paths.shape[1] > 1:
        path_segment = price_paths[:, :horizon_idx+1]  # (n_paths, horizon_idx+1)
        path_max = torch.max(path_segment, dim=1)[0]  # (n_paths,)
        path_min = torch.min(path_segment, dim=1)[0]  # (n_paths,)
        
        tp_threshold = s0 * (1.0 + tp_target_roe / leverage)
        sl_threshold = s0 * (1.0 - sl_target_roe / leverage)
        
        tp_hit = (path_max >= tp_threshold).float()
        sl_hit = (path_min <= sl_threshold).float()
        
        tp_prob = float(torch.mean(tp_hit))
        sl_prob = float(torch.mean(sl_hit))
    else:
        tp_prob = 0.0
        sl_prob = 0.0
    
    return {
        'ev': ev,
        'std': std,
        'cvar': cvar,
        'tp_prob': tp_prob,
        'sl_prob': sl_prob
    }


def evaluate_multi_symbol_batch_torch(
    symbols: List[str],
    price_paths_list: List[torch.Tensor],  # List of (n_paths, n_steps+1)
    horizons: List[int],
    leverages: List[float],
    fee_roundtrips: List[float],
    tp_targets: List[float],
    sl_targets: List[float],
    cvar_alpha: float = 0.95
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    다중 심볼에 대한 배치 평가 (PyTorch 버전)
    
    Returns:
        {symbol: {horizon: metrics_dict}}
    """
    results = {}
    
    if not _TORCH_OK or DEV_MODE:
        logger.warning("[VMAP_TORCH] PyTorch not available or DEV_MODE, skipping batch evaluation")
        return results
    
    try:
        for i, symbol in enumerate(symbols):
            if i >= len(price_paths_list):
                continue
                
            price_paths = price_paths_list[i]
            leverage = leverages[i] if i < len(leverages) else 1.0
            fee_rt = fee_roundtrips[i] if i < len(fee_roundtrips) else 0.01
            tp_target = tp_targets[i] if i < len(tp_targets) else 0.05
            sl_target = sl_targets[i] if i < len(sl_targets) else 0.05
            
            symbol_results = {}
            
            for horizon in horizons:
                if horizon >= price_paths.shape[1]:
                    continue
                    
                metrics = compute_horizon_metrics_torch(
                    price_paths=price_paths,
                    horizon_idx=horizon,
                    leverage=leverage,
                    fee_roundtrip=fee_rt,
                    tp_target_roe=tp_target,
                    sl_target_roe=sl_target,
                    cvar_alpha=cvar_alpha
                )
                
                symbol_results[horizon] = metrics
            
            results[symbol] = symbol_results
            
    except Exception as e:
        logger.warning(f"[VMAP_TORCH] Batch evaluation failed: {e}")
        
    return results


class GlobalBatchEvaluator:
    """
    PyTorch-based global batch evaluator for Monte Carlo simulation paths.
    Provides batch processing across multiple symbols and horizons.
    """
    def __init__(self, device=None):
        self.device = device or get_torch_device()
        self.torch_ok = _TORCH_OK
        if not self.torch_ok:
            logger.warning("[VMAP_TORCH] PyTorch not available, GlobalBatchEvaluator will fail.")

    def evaluate_batch(
        self,
        price_paths_batch: np.ndarray | torch.Tensor,  # (batch_size, n_paths, n_steps+1)
        horizons: List[int],
        leverages: np.ndarray,                         # (batch_size,)
        fee_roundtrips: np.ndarray,                    # (batch_size,)
        tp_targets_batch: np.ndarray,                  # (batch_size, n_horizons)
        sl_targets_batch: np.ndarray,                  # (batch_size, n_horizons)
        cvar_alpha: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate multiple symbols in a single batch using PyTorch.
        """
        if not self.torch_ok:
            raise RuntimeError("PyTorch not available")

        # Convert to torch if needed
        if isinstance(price_paths_batch, np.ndarray):
            paths = torch.from_numpy(price_paths_batch).float().to(self.device)
        else:
            paths = price_paths_batch.to(self.device)

        batch_size = paths.shape[0]
        n_horizons = len(horizons)
        
        # Output buffers
        results = {
            "ev_long": np.zeros((batch_size, n_horizons)),
            "ev_short": np.zeros((batch_size, n_horizons)),
            "p_pos_long": np.zeros((batch_size, n_horizons)),
            "p_pos_short": np.zeros((batch_size, n_horizons)),
            "cvar_long": np.zeros((batch_size, n_horizons)),
            "cvar_short": np.zeros((batch_size, n_horizons)),
            "p_tp_long": np.zeros((batch_size, n_horizons)),
            "p_sl_long": np.zeros((batch_size, n_horizons)),
            "p_tp_short": np.zeros((batch_size, n_horizons)),
            "p_sl_short": np.zeros((batch_size, n_horizons)),
        }

        # For each symbol in the batch
        for i in range(batch_size):
            symbol_paths = paths[i]  # (n_paths, n_steps+1)
            lev = float(leverages[i])
            fee = float(fee_roundtrips[i])
            
            for j, horizon in enumerate(horizons):
                if horizon >= symbol_paths.shape[1]:
                    continue
                
                tp_target = float(tp_targets_batch[i, j])
                sl_target = float(sl_targets_batch[i, j])
                
                metrics = compute_horizon_metrics_torch(
                    price_paths=symbol_paths,
                    horizon_idx=horizon,
                    leverage=lev,
                    fee_roundtrip=fee,
                    tp_target_roe=tp_target,
                    sl_target_roe=sl_target,
                    cvar_alpha=cvar_alpha
                )
                
                results["ev_long"][i, j] = metrics["ev"]
                results["ev_short"][i, j] = -metrics["ev"] 
                results["p_pos_long"][i, j] = metrics["tp_prob"] 
                results["p_pos_short"][i, j] = metrics["sl_prob"]
                results["cvar_long"][i, j] = metrics["cvar"]
                results["cvar_short"][i, j] = -metrics["ev"] 
                results["p_tp_long"][i, j] = metrics["tp_prob"]
                results["p_sl_long"][i, j] = metrics["sl_prob"]
                results["p_tp_short"][i, j] = metrics["sl_prob"] 
                results["p_sl_short"][i, j] = metrics["tp_prob"]
                
        return results


# ============================================================================
# 레거시 호환성을 위한 래퍼 함수들
# ============================================================================

def evaluate_multi_symbol_multi_horizon_vmap(
    symbols: List[str],
    price_paths_list: List[np.ndarray],
    horizons: List[int],
    leverages: List[float],
    fee_roundtrips: List[float],
    tp_targets: List[float],
    sl_targets: List[float],
    cvar_alpha: float = 0.95,
    timeout_sec: float = 60.0
) -> Tuple[Dict[str, Dict[int, Dict[str, float]]], float]:
    """
    레거시 JAX vmap 함수의 PyTorch 대체 버전
    """
    start_time = time.time()
    
    try:
        # Convert numpy arrays to torch tensors
        device_local = get_torch_device() if _TORCH_OK else None
        if device_local is None:
            device_local = torch.device("cpu") if _TORCH_OK else None
            
        if not _TORCH_OK or device_local is None:
            logger.warning("[VMAP_TORCH] PyTorch not available, returning empty results")
            return {}, time.time() - start_time
            
        torch_paths_list = []
        for paths in price_paths_list:
            if isinstance(paths, np.ndarray):
                tensor_paths = torch.from_numpy(paths).float().to(device_local)
                torch_paths_list.append(tensor_paths)
            elif isinstance(paths, torch.Tensor):
                torch_paths_list.append(paths.to(device_local))
            else:
                torch_paths_list.append(torch.zeros((1, 2), device=device_local))
        
        results = evaluate_multi_symbol_batch_torch(
            symbols=symbols,
            price_paths_list=torch_paths_list,
            horizons=horizons,
            leverages=leverages,
            fee_roundtrips=fee_roundtrips,
            tp_targets=tp_targets,
            sl_targets=sl_targets,
            cvar_alpha=cvar_alpha
        )
        
        elapsed = time.time() - start_time
        
        if elapsed > timeout_sec * 0.8:
            logger.warning(f"[VMAP_TORCH] Evaluation took {elapsed:.2f}s (timeout: {timeout_sec}s)")
            
        return results, elapsed
        
    except Exception as e:
        logger.error(f"[VMAP_TORCH] Error in batch evaluation: {e}")
        return {}, time.time() - start_time

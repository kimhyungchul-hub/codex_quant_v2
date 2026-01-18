"""
Multivariate Kelly Allocator

Mathematical Foundation:
    f* = K * Σ^(-1) * μ
    
Where:
    - f*: Optimal capital allocation vector
    - K: Kelly multiplier (e.g., 0.5 for half-Kelly)
    - Σ: Covariance matrix of returns
    - μ: Expected return vector (NAPV scores)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from engines.mc.jax_backend import _JAX_OK, jnp, jax_covariance

logger = logging.getLogger(__name__)


class KellyAllocator:
    """
    Multivariate Kelly formula for optimal capital allocation.
    
    Accounts for correlation between assets to avoid over-concentration.
    """
    
    def __init__(self, max_leverage: float = 10.0, half_kelly: float = 1.0):
        """
        Initialize Kelly allocator.
        
        Args:
            max_leverage: Maximum total leverage (e.g., 1000%)
            half_kelly: Kelly multiplier (1.0 = full-Kelly)
        """
        self.MAX_LEVERAGE = max_leverage
        self.HALF_KELLY = half_kelly
        
    def get_covariance_matrix(
        self,
        price_history_df: pd.DataFrame,
        symbols: List[str]
    ) -> np.ndarray:
        """
        Calculate covariance matrix from price history.
        
        Args:
            price_history_df: DataFrame with columns=symbols, index=timestamps
            symbols: List of symbols to include
            
        Returns:
            Covariance matrix as numpy array
        """
        # Calculate returns
        returns_df = price_history_df[symbols].pct_change().dropna()
        
        if _JAX_OK:
            # ✅ JAX Metal GPU Covariance
            try:
                rets_jnp = jnp.array(returns_df.values)
                cov_jnp = jax_covariance(rets_jnp)
                return np.array(cov_jnp)
            except Exception as e:
                logger.warning(f"⚠️ [KELLY] JAX Covariance failed: {e}. Falling back to NumPy.")
        
        # Compute covariance using NumPy/Pandas
        cov_matrix = returns_df.cov().values
        
        return cov_matrix
    
    def compute_allocation(
        self,
        candidates: List[Dict[str, Any]],
        cov_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute optimal Kelly allocation.
        
        Formula: f* = K * Σ^(-1) * μ
        
        Args:
            candidates: List of {'symbol': str, 'mu': float (NAPV score)}
            cov_matrix: Covariance matrix (n x n)
            
        Returns:
            Dict mapping symbol -> allocation weight
        """
        if not candidates:
            return {}
        
        # Extract expected returns (NAPV scores)
        mu_vector = np.array([c['mu'] for c in candidates])
        
        # Add ridge regularization for numerical stability
        n = len(cov_matrix)
        regularized_cov = cov_matrix + np.eye(n) * 1e-6
        
        # Solve Kelly formula: f = K * inv(Σ) * μ
        # Using np.linalg.solve is more stable than computing inverse
        try:
            raw_weights = self.HALF_KELLY * np.linalg.solve(regularized_cov, mu_vector)
        except np.linalg.LinAlgError:
            # Fallback: Use diagonal approximation if matrix is singular
            logger.warning("Covariance matrix singular, using diagonal approximation")
            raw_weights = self.HALF_KELLY * mu_vector / np.diag(regularized_cov)
        
        # Build allocation dict, filter near-zero weights
        allocations = {}
        total_leverage = 0.0
        
        for i, candidate in enumerate(candidates):
            weight = raw_weights[i]
            
            # Allow both Long (positive) and Short (negative) weights
            if abs(weight) <= 0.001:
                continue
            
            allocations[candidate['symbol']] = float(weight)
            total_leverage += abs(weight)
        
        # Apply leverage cap (scale down if over limit)
        if total_leverage > self.MAX_LEVERAGE:
            scale_factor = self.MAX_LEVERAGE / total_leverage
            for sym in allocations:
                allocations[sym] *= scale_factor
            logger.info(f"[KELLY] Scaled down from {total_leverage:.2f}x to {self.MAX_LEVERAGE:.2f}x")
        
        return allocations

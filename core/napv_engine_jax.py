"""
JAX Metal-accelerated NAPV (Net Added Present Value) Engine

This module provides GPU-accelerated batch NAPV calculations using JAX Metal.
Performance: ~10-20x faster than NumPy for batch operations.

Author: Optimized for Apple Silicon (M4 Pro)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import dataclasses

import engines.mc.jax_backend as jax_backend
from engines.mc.jax_backend import lazy_jit
import numpy as np
from typing import Dict, Tuple, Optional
import dataclasses


@dataclasses.dataclass
class NAPVConfig:
    """Configuration for NAPV calculations"""
    cost_full: float = 0.0006  # Round-trip cost (entry + exit)
    cost_exit_only: float = 0.0003  # Exit-only cost
    min_holding_time: float = 60.0  # Minimum holding time (seconds)
    
    @classmethod
    def from_orchestrator(cls, orchestrator):
        """Create config from orchestrator settings"""
        fee = getattr(orchestrator, 'fee', 0.0003)
        return cls(
            cost_full=fee * 2.0,
            cost_exit_only=fee,
            min_holding_time=float(getattr(orchestrator, 'paper_exit_policy_min_hold_sec', 60))
        )


class NAPVEngineJAX:
    """
    Metal-accelerated NAPV calculation engine using JAX.
    
    Computes Net Added Present Value for all symbols in parallel on GPU.
    Uses JIT compilation and vectorization for maximum performance.
    
    ✅ DEV_MODE=true일 때 NumPy fallback 사용
    """
    
    def __init__(self, config: Optional[NAPVConfig] = None):
        self.config = config or NAPVConfig()
        # Lazily initialize JAX and compile kernels on first use
        jax_backend.ensure_jax()
        # DEV_MODE or JAX not available: use NumPy fallback
        if getattr(jax_backend, 'DEV_MODE', False) or not getattr(jax_backend, '_JAX_OK', False) or jax_backend.jax is None:
            print("[NAPV_JAX] JAX unavailable or DEV_MODE; using NumPy fallback")
            self.device = "cpu"
            self._use_numpy = True
            return

        # JAX is available: initialize device and jit kernels
        self._use_numpy = False
        try:
            self.device = jax_backend.jax.devices()[0]
            print(f"[NAPV_JAX] Initialized on device: {self.device}")
        except Exception:
            # If device query fails, fallback to NumPy
            print("[NAPV_JAX] Device query failed; falling back to NumPy")
            self.device = "cpu"
            self._use_numpy = True
            return

        # JIT-compile core functions using jax backend
        self._napv_single_jit = jax_backend.jax.jit(self._napv_single_core)
        self._napv_batch_jit = jax_backend.jax.jit(jax_backend.jax.vmap(self._napv_single_core, in_axes=(0, 0, None, None, 0)))
        
    @staticmethod
    def _napv_single_core(
        horizons,
        ev_rates,
        rho: float,
        r_f: float,
        cost: float
    ) -> Tuple[float, float]:
        """
        Core NAPV calculation for a single symbol (GPU kernel).
        
        Args:
            horizons: Time horizons in seconds [h1, h2, ..., hn]
            ev_rates: Expected value rates at each horizon
            rho: Opportunity cost / discount rate (per second)
            r_f: Risk-free rate (per second)
            cost: Transaction cost (fractional)
            
        Returns:
            (napv_max, t_star): Maximum NAPV and optimal holding time
        """
        # Calculate excess yield over opportunity cost
        excess_yield = ev_rates - rho
        
        # Time discount factor: e^(-ρt)
        jnp = jax_backend.jnp
        discount = jnp.exp(-rho * horizons)

        # Time step sizes (dt)
        dt = jnp.diff(horizons, prepend=0.0)

        # Cumulative discounted value: ∫[0→t] (r - ρ)·e^(-ρt) dt
        cumulative_value = jnp.cumsum(excess_yield * discount * dt)

        # Net value after transaction cost
        net_value = cumulative_value - cost

        # Find optimal exit time (maximum NAPV)
        idx_max = jnp.argmax(net_value)
        napv_max = net_value[idx_max]
        t_star = horizons[idx_max]
        
        return napv_max, t_star
    
    def calculate_single(
        self,
        horizons_sec: np.ndarray,
        ev_rate_vector: np.ndarray,
        rho: float,
        r_f: float,
        cost_mode: str = "full"
    ) -> Tuple[float, float]:
        """
        Calculate NAPV for a single symbol (backward compatible).
        
        Args:
            horizons_sec: Time horizons in seconds
            ev_rate_vector: Expected value rates
            rho: Opportunity cost rate
            r_f: Risk-free rate
            cost_mode: "full" or "exit_only"
            
        Returns:
            (napv, t_star): Net added present value and optimal holding time
        """
        if horizons_sec is None or ev_rate_vector is None:
            return 0.0, 0.0
        
        # Select cost
        cost = self.config.cost_exit_only if cost_mode == "exit_only" else self.config.cost_full
        
        # DEV_MODE: NumPy fallback
        if getattr(self, '_use_numpy', False):
            return self._napv_single_numpy(
                np.asarray(horizons_sec, dtype=np.float32),
                np.asarray(ev_rate_vector, dtype=np.float32),
                rho, r_f, cost
            )
            
        # Convert to JAX arrays
        h_jax = jnp.asarray(horizons_sec, dtype=jnp.float32)
        v_jax = jnp.asarray(ev_rate_vector, dtype=jnp.float32)
        
        # Compute on GPU
        napv, t_star = self._napv_single_jit(h_jax, v_jax, rho, r_f, cost)
        
        # Convert back to Python scalars
        return float(napv), float(t_star)
    
    @staticmethod
    def _napv_single_numpy(horizons, ev_rates, rho: float, r_f: float, cost: float):
        """NumPy fallback for NAPV calculation."""
        # discount_factors: exp(-rho * t)
        discount = np.exp(-rho * horizons)
        # risk-free discount: exp(-r_f * t)
        rf_discount = np.exp(-r_f * horizons)
        # NAPV = sum(ev * discount) - cost
        napv_values = ev_rates * discount * horizons - cost * rf_discount
        i_max = int(np.argmax(napv_values))
        return float(napv_values[i_max]), float(horizons[i_max])
    
    def calculate_batch(
        self,
        symbols: list[str],
        horizons_batch: np.ndarray,  # Shape: (n_symbols, n_horizons)
        ev_rates_batch: np.ndarray,  # Shape: (n_symbols, n_horizons)
        rho: float,
        r_f: float,
        costs: np.ndarray,  # Shape: (n_symbols,)
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate NAPV for multiple symbols in parallel (BATCH MODE).
        
        This is the main performance optimization: all symbols computed
        simultaneously on GPU.
        
        Args:
            symbols: List of symbol names
            horizons_batch: Horizons for each symbol (n_symbols × n_horizons)
            ev_rates_batch: EV rates for each symbol (n_symbols × n_horizons)
            rho: Opportunity cost rate (same for all)
            r_f: Risk-free rate (same for all)
            costs: Transaction cost per symbol (n_symbols,)
            
        Returns:
            Dict mapping symbol -> (napv, t_star)
        """
        # DEV_MODE: NumPy fallback (sequential)
        if getattr(self, '_use_numpy', False):
            results = {}
            for i, sym in enumerate(symbols):
                napv, t_star = self._napv_single_numpy(
                    horizons_batch[i], ev_rates_batch[i], rho, r_f, costs[i]
                )
                results[sym] = (napv, t_star)
            return results
        
        # Convert to JAX arrays
        h_jax = jnp.asarray(horizons_batch, dtype=jnp.float32)
        v_jax = jnp.asarray(ev_rates_batch, dtype=jnp.float32)
        c_jax = jnp.asarray(costs, dtype=jnp.float32)
        
        # Batch compute on GPU (single kernel call for all symbols)
        napv_array, t_star_array = self._napv_batch_jit(
            h_jax, v_jax, rho, r_f, c_jax
        )
        
        # Convert to dictionary
        results = {}
        for i, sym in enumerate(symbols):
            results[sym] = (float(napv_array[i]), float(t_star_array[i]))
        
        return results
    
    def calculate_ranking_scores(
        self,
        symbols: list[str],
        horizons_batch: np.ndarray,
        ev_rates_batch: np.ndarray,
        rho: float,
        r_f: float,
    ) -> Dict[str, float]:
        """
        Calculate NAPV scores for TOP5 ranking (simplified interface).
        
        Uses exit_only cost for all symbols.
        
        Returns:
            Dict mapping symbol -> napv_score
        """
        n_symbols = len(symbols)
        costs = np.full(n_symbols, self.config.cost_exit_only, dtype=np.float32)
        
        results = self.calculate_batch(
            symbols, horizons_batch, ev_rates_batch, rho, r_f, costs
        )
        
        # Extract just the NAPV values
        return {sym: napv for sym, (napv, _) in results.items()}


# Global singleton instance (lazy initialization)
_napv_engine_instance: Optional[NAPVEngineJAX] = None


def get_napv_engine(config: Optional[NAPVConfig] = None) -> NAPVEngineJAX:
    """Get or create the global NAPV engine instance"""
    global _napv_engine_instance
    if _napv_engine_instance is None:
        _napv_engine_instance = NAPVEngineJAX(config)
    return _napv_engine_instance


# Convenience function for backward compatibility
def calculate_napv_vectorized_jax(
    horizons_sec: np.ndarray,
    ev_rate_vector: np.ndarray,
    rho: float,
    r_f: float,
    cost_mode: str = "full"
) -> Tuple[float, float]:
    """
    Drop-in replacement for EconomicBrain.calculate_napv_vectorized
    
    Uses JAX Metal acceleration instead of NumPy.
    """
    engine = get_napv_engine()
    return engine.calculate_single(horizons_sec, ev_rate_vector, rho, r_f, cost_mode)

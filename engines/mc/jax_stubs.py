"""PyTorch-first backend shim - placeholder for missing JAX functions"""
import numpy as np

def summarize_gbm_horizons_jax(*args, **kwargs):
    """Stub function - returns empty results"""
    return {"ev": 0.0, "win_rate": 0.5, "cvar": 0.0}

from __future__ import annotations

import os
import sys
from pathlib import Path

# ============================================================================
# JAX Version Lock Check (CRITICAL)
# ============================================================================
# Verified working: JAX 0.4.20 + jax-metal 0.0.5 + NumPy <2.0
# See requirements-jax.txt for details.
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

# ============================================================================
# JAX/XLA Memory Settings (MUST be set BEFORE any JAX import)
# ============================================================================
# These environment variables ONLY work with JAX 0.4.20 + jax-metal 0.0.5
# Later versions (0.4.22+, 0.9.0+) ignore these settings!

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
# NOTE: Do NOT set JAX_METAL_CACHE_SIZE=0 - causes 400GB+ virtual memory allocation

# Compilation cache directory
jax_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", str(BASE_DIR / ".jax_cache"))
Path(jax_cache_dir).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(jax_cache_dir))
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "true")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MAX_ENTRY_SIZE_BYTES", "-1")

# ============================================================================
# Version Validation (run at import time)
# ============================================================================
def _validate_jax_version():
    """Validate JAX version is compatible with memory settings."""
    try:
        import jax
        version = jax.__version__
        major, minor, patch = map(int, version.split('.')[:3])
        
        if (major, minor) > (0, 4) or (major == 0 and minor == 4 and patch > 20):
            print(f"⚠️  [BOOTSTRAP] WARNING: JAX {version} detected!")
            print(f"⚠️  [BOOTSTRAP] Memory preallocation settings may not work!")
            print(f"⚠️  [BOOTSTRAP] Recommended: pip install jax==0.4.20 jaxlib==0.4.20 jax-metal==0.0.5")
            return False
        return True
    except Exception:
        return True

# Print bootstrap info
try:
    print(f"🗃️ [BOOTSTRAP] JAX env: PREALLOCATE={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}, MEM_FRACTION={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}")
except Exception:
    pass


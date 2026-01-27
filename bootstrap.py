from __future__ import annotations

import logging
import os
import sqlite3
import sys
from pathlib import Path

# ============================================================================
# JAX Version Lock Check (CRITICAL)
# ============================================================================
# Verified working: JAX 0.4.20 + jax-metal 0.0.5 + NumPy <2.0
# See requirements-jax.txt for details.
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
logger = logging.getLogger(__name__)

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


# ============================================================================
# SQLite Integrity Check & Recovery
# ============================================================================
DEFAULT_DB_CANDIDATES = [
    STATE_DIR / "bot_data.db",
    STATE_DIR / "bot_data_paper.db",
    STATE_DIR / "bot_data_live.db",
]


def check_integrity(db_path: Path) -> tuple[bool, str]:
    """Run PRAGMA integrity_check and REINDEX on index corruption."""
    if not db_path.exists():
        return True, "missing"

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        result = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        ok = str(result).lower() == "ok"
        if not ok:
            conn.execute("REINDEX;")
            conn.commit()
        return ok, str(result)
    except Exception as exc:  # pragma: no cover - best-effort bootstrap
        return False, str(exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def run_integrity_checks(db_candidates: list[Path] | None = None) -> None:
    paths = db_candidates or DEFAULT_DB_CANDIDATES
    for db_path in paths:
        ok, detail = check_integrity(db_path)
        if detail == "missing":
            continue
        prefix = "✅" if ok else "⚠️"
        msg = f"{prefix} [BOOTSTRAP] integrity_check {db_path}: {detail}"
        print(msg)
        if not ok:
            logger.warning(msg)


try:
    run_integrity_checks()
except Exception as exc:  # pragma: no cover - fail open
    print(f"⚠️  [BOOTSTRAP] integrity check skipped: {exc}")


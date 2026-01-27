"""
Monte Carlo Engine Constants (Central Configuration)
=====================================================

모든 Monte Carlo 관련 하드코딩 상수를 중앙 집중 관리합니다.
각 파일에서 이 모듈을 import하여 사용하세요.

CRITICAL: 이 파일만 수정하면 모든 곳에 반영됩니다.
"""

from __future__ import annotations

import os

from engines.mc.config import config

# ============================================================================
# Legacy config compatibility
# ============================================================================
MC_VERBOSE_PRINT = config.verbose_print
MC_N_PATHS_LIVE = config.n_paths_live
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


# ============================================================================
# Static Shape Constants (JAX JIT Stability)
# ============================================================================
# CRITICAL: JAX JIT는 입력 shape이 바뀔 때마다 재컴파일합니다.
# 장중 렉(Lag) 방지를 위해 항상 고정된 크기의 배열을 사용해야 합니다.

STATIC_MAX_SYMBOLS = 32    # 최대 심볼 수 (32개로 고정, padding 사용)
STATIC_MAX_PATHS = 16384   # 최대 경로 수 (4x 증가: 더 정밀한 확률 추정)
STATIC_MAX_STEPS = 3600    # 최대 스텝 수 (1시간 = 3600초)

# Alias for backward compatibility
JAX_STATIC_BATCH_SIZE = STATIC_MAX_SYMBOLS  # entry_evaluation.py에서 사용


# ============================================================================
# Execution Cost Parameters
# ============================================================================
DEFAULT_IMPACT_CONSTANT = 0.75  # Square-Root Market Impact 계수 (0.5~1.0)


# ============================================================================
# Decision Timeout
# ============================================================================
# MC 배치 결정 호출 기본 타임아웃 (초)
DECIDE_BATCH_TIMEOUT_SEC = float(os.environ.get("DECIDE_BATCH_TIMEOUT_SEC", "5.0"))


# ============================================================================
# Horizon Constants (시간대 설정)
# ============================================================================
# 고정 horizon 목록 (초 단위)
STATIC_HORIZONS = [60, 300, 600, 1800, 3600]  # 1m, 5m, 10m, 30m, 1h
HORIZON_SUMMARY_DEFAULT = [60, 300, 600, 1800, 3600]  # entry_evaluation.py에서 사용


# ============================================================================
# Numerical Constants
# ============================================================================
EPSILON = 1e-12  # 0 나누기 방지용 최소값


# ============================================================================
# Data Type Constants (NumPy/JAX dtype)
# ============================================================================
DTYPE_FLOAT32 = "float32"
DTYPE_FLOAT64 = "float64"
DTYPE_INT32 = "int32"
DTYPE_UINT32 = "uint32"


# ============================================================================
# Bootstrap & Tail Sampling
# ============================================================================
BOOTSTRAP_MIN_SAMPLES = 64   # Bootstrap returns 최소 샘플 수
BOOTSTRAP_HISTORY_LEN = 512  # Bootstrap returns 히스토리 길이


# ============================================================================
# Validation
# ============================================================================
def validate_constants():
    """상수들이 유효한 범위인지 검증"""
    assert STATIC_MAX_SYMBOLS > 0, "STATIC_MAX_SYMBOLS must be > 0"
    assert STATIC_MAX_PATHS > 0, "STATIC_MAX_PATHS must be > 0"
    assert STATIC_MAX_STEPS > 0, "STATIC_MAX_STEPS must be > 0"
    assert MC_N_PATHS_LIVE > 0, "MC_N_PATHS_LIVE must be > 0"
    assert len(STATIC_HORIZONS) > 0, "STATIC_HORIZONS must not be empty"
    
    # 워밍업 시 사용할 값이 실제 사용값보다 크거나 같은지 확인
    if MC_N_PATHS_LIVE > STATIC_MAX_PATHS:
        import warnings
        warnings.warn(
            f"MC_N_PATHS_LIVE ({MC_N_PATHS_LIVE}) > STATIC_MAX_PATHS ({STATIC_MAX_PATHS}). "
            "JIT recompilation may occur during runtime!"
        )


# Auto-validate on import
validate_constants()


if __name__ == "__main__":
    print("=== Monte Carlo Constants ===")
    print(f"STATIC_MAX_SYMBOLS: {STATIC_MAX_SYMBOLS}")
    print(f"STATIC_MAX_PATHS: {STATIC_MAX_PATHS}")
    print(f"STATIC_MAX_STEPS: {STATIC_MAX_STEPS}")
    print(f"MC_N_PATHS_LIVE: {MC_N_PATHS_LIVE}")
    print(f"STATIC_HORIZONS: {STATIC_HORIZONS}")
    print(f"JAX_STATIC_BATCH_SIZE: {JAX_STATIC_BATCH_SIZE}")
    print(f"DEFAULT_IMPACT_CONSTANT: {DEFAULT_IMPACT_CONSTANT}")
    print("\n✅ All constants validated!")

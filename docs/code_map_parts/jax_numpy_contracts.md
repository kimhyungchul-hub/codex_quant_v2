# JAX vs NumPy Contracts (engines/mc/*)

이 문서는 JAX와 NumPy 구현 사이의 타입/shape/호환성 규칙을 정리합니다. MC 파이프라인은 둘 중 어떤 백엔드를 사용하느냐에 따라 반환 타입이 달라질 수 있으므로, 소비자(EngineHub, Orchestrator, Dashboard)가 이를 안전하게 처리해야 합니다.

## 전제
- JAX가 사용 가능한 경우(환경에 `jax`가 import 가능하고 `_JAX_OK`가 True), 글로벌 배치/요약 함수는 `jnp.ndarray`를 반환합니다.
- JAX 불가시, NumPy/표준 파이썬 타입(`np.ndarray`, float, int)으로 반환됩니다.

## 규격(권장)
- 배열/행렬
  - Path 시뮬레이션: `(n_paths, n_steps+1)` (JAX: `jnp.ndarray`, NumPy: `np.ndarray`)
  - Batch path: `(num_symbols, n_paths, n_steps+1)`
  - Per-horizon vectors: `(H,)` 또는 `(H, )` (EV, p_pos 등)

- 스칼라
  - `ev`, `ev_raw`, `win`, `cvar`, `kelly`, `size_frac` 등은 항상 Python `float`로 변환되어야 함.

## 소비자(EngineHub 등) 권장 처리
- 모든 엔진 결과는 수집 후 `_sanitize_value()` 호출을 통해 JAX 배열을 NumPy/Python 타입으로 변환.
  - 예: `float(np.asarray(v))` 또는 `v.item()` 사용.
- `meta`/`meta_detail`는 딕셔너리 내부의 배열도 동일하게 변환해야 함.

## 예시 변환 함수 (참고)
```py
def sanitize_value(v):
    import numpy as np
    try:
        import jax.numpy as jnp
    except Exception:
        jnp = None
    if jnp is not None and isinstance(v, jnp.ndarray):
        v = np.asarray(v)
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return v.item()
        return v
    return v
```

## 테스트 권장사항
- 유닛 테스트에서 JAX/NumPy 둘 다에서 `evaluate_entry_metrics`의 반환 키와 shape를 확인하는 테스트를 갖추세요.
- CI가 JAX를 지원하지 않으면 NumPy fallback 경로도 테스트해야 합니다.

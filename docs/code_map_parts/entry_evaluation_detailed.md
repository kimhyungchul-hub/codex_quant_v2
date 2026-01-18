# Entry Evaluation — Detailed (engines/mc/entry_evaluation.py)

이 문서는 `evaluate_entry_metrics` 및 배치 버전의 정확한 입력/출력 계약과 `meta_detail`의 주요 키들을 상세히 정리합니다.

## 주요 함수 시그니처 (요약)

- evaluate_entry_metrics(symbol: str,
  params: Mapping[str, Any],
  mc_params: "MCParams",
  policy_params: Mapping[str, Any],
  n_paths: int = 8192,
  rng: Optional[object] = None,
  use_jax: bool = True,
  verbose: bool = False) -> Dict[str, Any]

  - 설명: 단일 심볼에 대해 경로를 시뮬레이션하고 각 horizon별 EV/CVaR/확률을 계산해 최종 권장값을 반환.
  - 주요 입력 항목 설명:
    - `symbol`: 심볼 식별자 (예: `BTCUSDT`)
    - `params`: 현재 가격 `s0`, drift `mu`, vol `sigma`, `direction` 신호 세기 등 포함하는 dict/데이터클래스
    - `mc_params` (`MCParams`): horizons 리스트, `dt`, fee_model, `seed` 등 MC 전역 설정
    - `policy_params`: horizon priors, reweighting 스킴, 손절/익절 정책 등
    - `n_paths`: 시뮬레이션 경로 수 (성능/정밀도 tradeoff)
    - `rng`: (선택) 외부 RNG, JAX PRNGKey 또는 numpy.RandomState
    - `use_jax`: True이면 JAX 경로(가능하면 global-batch 사용)를 시도

- evaluate_entry_metrics_batch(tasks: List[Dict[str, Any]], global_jax: bool = True) -> List[Dict[str, Any]]

  - 설명: 심볼 리스트를 받아 글로벌 배치(JAX)로 처리 시도. JAX 미사용 시 각 심볼에 대해 `evaluate_entry_metrics`를 반복 호출.
  - `tasks` 각 항목에 포함되어야 할 필드 예: `symbol`, `s0`, `mu`, `sigma`, `mc_params`, `policy_params`, `n_paths`.

## 반환 계약 (`res` dict)

- 최상위 scalar/플래그:
  - `can_enter` (bool): 진입 허용 여부
  - `ev` (float): 비용/수수료 반영 후 기대값 (final)
  - `ev_raw` (float): 비용/수수료 반영 전 기대값
  - `win` (float): 승률(추정)
  - `cvar` (float): CVaR (tail risk)
  - `kelly` (float): Kelly 비율 추정
  - `size_frac` (float): 권장 포지션 비율 (0..1)
  - `direction` (int): 1 또는 -1
  - `best_h` (int): 선택된 최적 horizon(인덱스 혹은 초)
  - `reason` (str): 진입 불가/경고 사유 설명

- `meta` / `meta_detail` (dict): 진단용 상세 정보. 주요 키와 형태:
  - `policy_w_h`: (array, H) 각 horizon에 대한 최종 가중치
  - `policy_ev_by_h_long`: (array, H) long 방향의 per-horizon EV
  - `policy_ev_by_h_short`: (array, H) short 방향
  - `policy_p_pos_by_h`: (array, H) 각 horizon에서 양수 P(probability)
  - `event_p_tp`, `event_p_sl`, `event_p_timeout`: (scalars or arrays length H) 목표달성/손절/타임아웃 확률
  - `event_ev_r`, `event_cvar_r`: (scalars) 이벤트 기반 비율/정규화 값
  - `event_t_median`, `event_t_mean`: (scalars) 이벤트 소요시간 통계
  - `perf`: (dict) timing/profiling 정보 (배치/시뮬/요약에 소요된 ms)

> 메모: `H`은 사용중인 horizon 수입니다. `policy_*` 배열은 길이 H를 가집니다.

## 경계조건 및 예외

- `kelly`가 NaN/inf일 경우: 코드 내에서 안전한 기본값(설정값)에 따라 대체됩니다.
- JAX 불가시 `use_jax=True`라도 fallback으로 NumPy 루트 또는 단일-심볼 루프가 실행됩니다.
- `meta_detail`의 키 변경은 `LiveOrchestrator._row()` 와 대시보드 매핑 갱신을 요구합니다.

## 권장 개선(문서화/타입 안전성)

- 현재 `res`는 자유로운 dict 구조이므로, `EvaluationResult` 같은 Typed dataclass를 도입해 키-의존성을 줄이는 것을 권장합니다.
- `mc_params`와 `policy_params`도 pydantic/dataclass로 스키마화하면 배치 안정성이 향상됩니다.

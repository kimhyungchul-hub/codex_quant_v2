[2026-01-24] Brownian Bridge 보정 및 JAX 포팅, 테스트 추가

변경 내용:
- `engines/mc/first_passage.py`: `mc_first_passage_tp_sl`에 Brownian Bridge 기반 intra-step 터치 확률 보정 추가 (NumPy fallback 및 JAX 경로 모두 지원). JAX 경로는 로그가격을 유지하여 확률을 JAX에서 계산한 뒤 호스트로 전송하여 마스크를 결합합니다.
- 테스트 추가: `tests/test_first_passage_bb_compare.py`, `tests/test_first_passage_debug.py`, `tests/test_first_passage_jax_compare.py`, `tests/test_first_passage_outliers.py` — 보정 전/후 비교, 샘플 경로 디버그, JAX/NumPy 결과·성능 비교, 아웃라이어 분석.

영향:
- First-passage 이벤트 확률 계산 방식이 변경되어 downstream EV/CVaR 결과에 영향 가능. 엔진에서 `mc_first_passage_tp_sl`을 호출하는 위치(예: `engines/mc/entry_evaluation.py`, `engines/simulation_methods.py`)의 동작은 동일하나 결과 분포가 달라질 수 있으므로 회귀 테스트 권장.

참고:
- 브리지 적용 후 일부 시나리오에서 `timeout` 비율이 급증하는 현상이 관찰되어(디버그 스크립트 결과) 시간-인덱스 정렬과 브리지 마스크 매핑(간격→시점 매핑)을 추가로 검증 및 필요시 수정해야 합니다.

# OpenAI Code Review — 2026-02-18 04:07

**Model:** gpt-4.1-mini
**Risk Score:** 7/10
**Summary:** 전략 코드는 MC 시뮬레이션 기반의 하이브리드 진입/청산 로직을 통합하여 안정적 의사결정을 수행하나, 일부 파라미터 및 비용 이중 차감 문제, 레버리지 제한, 메모리 관리 최적화 여지가 존재합니다.

## Suggestions

### [P1] EV 계산 시 비용 이중 차감 문제 해결
**Category:** logic | **Confidence:** 95%

evaluate_entry_metrics 함수 내에서 MC 시뮬레이션 결과 ev_raw는 이미 수수료를 포함한 순이익인데, 이후 net_expectancy 필터에서 수수료를 다시 차감하여 실제 기대수익이 과소평가되는 문제가 있습니다. 이중 차감으로 인해 진입 후보가 과도하게 차단될 수 있으므로, net_expectancy 계산 시 비용 차감을 제거하거나 ev_raw에서 비용을 역산해 보정해야 합니다.

**Env Changes:** `{'ENTRY_NET_EXPECTANCY_MIN': '-0.0008'}`

**Code Changes:** engines/mc/entry_evaluation.py 내 evaluate_entry_metrics 함수에서 net_edge 계산 부분 수정. 비용 차감 부분 제거 또는 ev_raw에 비용 재추가 로직 추가 필요.

**Expected Impact:** 진입 후보의 과도한 차단 완화 및 기대수익 정확도 향상으로 진입 기회 증가 및 수익 개선 기대.

### [P1] 레버리지 하한 및 상한 환경변수 조정 및 통합
**Category:** param | **Confidence:** 90%

현재 UNI_LEV_MIN=10.0, UNI_LEV_MAX=50 등 고정값이 하드코딩되어 있으며, 레짐별로 상이한 최대 레버리지가 혼재되어 있습니다. 이를 regime_policy.py 내 레짐별 max_leverage 값과 일관되게 동기화하고, 환경변수로 쉽게 조정 가능하도록 개선해야 합니다. 또한, 레버리지 하한을 너무 높게 설정하면 낮은 신뢰도 신호에 대해 과도한 레버리지로 위험이 커질 수 있으므로, 신호 강도에 따른 동적 하한 조정도 고려할 필요가 있습니다.

**Env Changes:** `{'UNI_LEV_MIN': '1.0', 'UNI_LEV_MAX': '20.0', 'MAX_LEVERAGE': '20'}`

**Code Changes:** engines/mc/regime_policy.py 내 RegimePolicy max_leverage 필드와 engines/mc/decision.py 내 최적 레버리지 산출 로직에서 환경변수 및 regime_policy 값 일치화 및 동적 조정 로직 추가.

**Expected Impact:** 과도한 레버리지로 인한 급격한 손실 위험 감소 및 레짐별 위험 관리 일관성 강화.

### [P2] 포지션 사이징 및 최대 노출 제한 강화
**Category:** risk | **Confidence:** 80%

현재 포지션 최대 비중(max_pos_frac)과 하드 노테이셔널 캡(UNI_HARD_NOTIONAL_CAP) 설정이 다소 관대하며, 특히 고변동 레짐에서 집중 투자 정책이 위험을 키울 수 있습니다. CF 분석 결과를 반영하여 chop, volatile 등 레짐별 max_pos_frac 및 max exposure 제한을 강화하고, 동시에 포지션 집중도를 모니터링하는 로직을 추가하는 것이 바람직합니다.

**Env Changes:** `{'UNI_MAX_POS_FRAC_CHOP': '0.30', 'UNI_MAX_POS_FRAC_VOLATILE': '0.20', 'UNI_HARD_NOTIONAL_CAP': '1000'}`

**Code Changes:** engines/mc/regime_policy.py 내 RegimePolicy 기본값 조정 및 engines/mc/decision.py 내 포지션 사이징 산출 로직에 집중도 제한 및 리밸런싱 조건 강화.

**Expected Impact:** 포지션 과다 집중에 따른 급락 리스크 감소 및 안정적 자본 운용 가능.

### [P2] 메모리 누수 및 대형 배열 명시적 해제 강화
**Category:** perf | **Confidence:** 85%

evaluate_entry_metrics 등에서 대형 텐서 및 중간 결과를 명시적으로 해제하고 gc.collect()를 호출하지만, 일부 변수 누락 가능성이 존재합니다. 특히 GPU 메모리 캐시 정리 로직을 try-except로 감싸는 것은 좋으나, 더 세밀한 메모리 추적 및 불필요 변수 삭제를 추가하여 메모리 사용량을 최소화해야 합니다.

**Code Changes:** engines/mc/entry_evaluation.py 내 메모리 해제 구간에 누락된 변수 및 텐서 추가 삭제, GPU 캐시 정리 로직 보완 및 주석 추가.

**Expected Impact:** 장시간 운용 시 메모리 누수 감소, OOM 방지 및 안정성 향상.

### [P3] 하이브리드 진입/청산 로직의 방향 결정 일관성 강화
**Category:** logic | **Confidence:** 75%

hybrid_only 모드에서 방향 결정 시 mu_alpha 부호 단독 결정 대신 exp_vals 비교를 우선하나, fallback 로직에서 policy_score와 방향 불일치 가능성이 존재합니다. direction_model_used 플래그 및 confidence 기반 gate를 엄격히 적용하고, fallback 시에도 명확한 우선순위 규칙을 명시하여 방향 결정의 일관성을 높여야 합니다.

**Env Changes:** `{'USE_DIRECTION_MODEL': '1', 'MC_HYBRID_ONLY': '1'}`

**Code Changes:** engines/mc/decision.py 내 _decide_hybrid_only 함수에서 방향 결정 fallback 로직 개선 및 direction_model_used 체크 강화.

**Expected Impact:** 잘못된 방향 진입 감소 및 신호 신뢰도 향상.

### [P3] 진입 임계치 및 chop_guard 파라미터 재조정
**Category:** param | **Confidence:** 80%

최근 CF 결과에 따르면 chop_guard 관련 파라미터(chop_entry_floor_add, chop_entry_min_dir_conf 등)를 상향 조정하여 chop 레짐에서 불필요한 진입을 줄이는 것이 수익 개선에 효과적입니다. 또한, UNIFIED_ENTRY_FLOOR 및 ENTRY_BOTH_EV_NEG_NET_FLOOR 값을 현재 score 분포에 맞게 재조정할 필요가 있습니다.

**Env Changes:** `{'CHOP_ENTRY_FLOOR_ADD': '0.003', 'CHOP_ENTRY_MIN_DIR_CONF': '0.64', 'UNIFIED_ENTRY_FLOOR': '0.0001', 'ENTRY_BOTH_EV_NEG_NET_FLOOR': '-0.0003'}`

**Code Changes:** engines/mc/regime_policy.py 내 기본값 및 환경변수 읽기 로직 조정, engines/mc/decision.py 내 진입 필터 로직에서 임계치 적용 방식 개선.

**Expected Impact:** 불필요한 진입 감소 및 진입 품질 향상으로 손실 감소 기대.

### [P3] 시간대 필터 및 레짐별 진입 차단 강화
**Category:** risk | **Confidence:** 85%

TOD_FILTER_ENABLED, REGIME_SIDE_BLOCK_LIST 등 환경변수 기반 시간대 및 레짐별 진입 차단 기능이 존재하나, 현재 설정이 일부 미흡할 수 있습니다. 특히 손실이 컸던 UTC 6~7시 구간 및 bear_long, bull_short, chop_long 조합에 대한 차단을 엄격히 적용하고, 로그 모니터링을 통해 차단 효과를 지속 검증해야 합니다.

**Env Changes:** `{'TOD_FILTER_ENABLED': '1', 'TRADING_BAD_HOURS_UTC': '6,7', 'REGIME_SIDE_BLOCK_LIST': 'bear_long,bull_short,chop_long'}`

**Code Changes:** engines/mc/entry_evaluation.py 및 regime_policy.py 내 시간대 및 레짐별 진입 차단 로직 강화 및 로그 추가.

**Expected Impact:** 손실 발생 빈도가 높은 시간대 및 레짐-포지션 조합에서 진입 차단으로 리스크 감소.

### [P3] GPU 가속 레버리지 최적화 모듈 로딩 개선
**Category:** perf | **Confidence:** 70%

현재 USE_GPU_LEVERAGE_OPT 플래그와 _maybe_load_leverage_optimizer 함수에서 torch 모듈을 lazy import 하는데, 실패 시 silent하게 False 처리합니다. 실패 원인 로깅 및 fallback 경로를 명확히 하여 디버깅 편의성을 높이고, GPU 사용 여부를 명확히 환경변수로 제어할 수 있도록 개선해야 합니다.

**Env Changes:** `{'MC_USE_TORCH': '1'}`

**Code Changes:** engines/mc/decision.py 내 _maybe_load_leverage_optimizer 함수에 예외 발생 시 로깅 추가 및 환경변수 제어 강화.

**Expected Impact:** GPU 가속 기능의 안정적 활성화 및 문제 발생 시 신속 대응 가능.

## Warnings
- MC EV 계산 시 비용이 이중 차감되어 실제 기대수익이 과소평가될 위험이 있음.
- 고정된 높은 레버리지 하한은 신호 신뢰도 낮은 상황에서 큰 손실을 유발할 수 있음.
- 메모리 해제 로직이 일부 변수 누락 가능성이 있어 장시간 운용 시 메모리 누수가 발생할 수 있음.
- 하이브리드 방향 결정 로직에서 fallback 시 방향 불일치 가능성이 존재함.
- 환경변수에 의존하는 파라미터가 많아 일관된 관리 및 변경 시 영향 분석이 필수임.


---
*Auto-generated by research/openai_reviewer.py*
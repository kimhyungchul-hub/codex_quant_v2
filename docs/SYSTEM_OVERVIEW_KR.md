ㅌ# Codex Quant 시스템 구조/기능 설명서 (KR)

이 문서는 현재 레포(`codex_quant`)의 **실제 실행 코드 기준**으로, 엔트리포인트부터 데이터 수집→의사결정(MC)→(paper/live) 집행→대시보드 노출까지의 전체 흐름을 장기 기억/인수인계용으로 정리한 문서입니다.

---

## 1) 큰 그림 (한 장 요약)

### 목적
- Bybit(기본) 등 거래소의 다심볼(perp) 마켓 데이터(티커/캔들/호가)를 비동기로 수집
- Monte Carlo 기반 엔진이 심볼별 **진입/보유/전환/청산**의 기대가치(EV), 리스크(CVaR), 최적 보유시간(T*) 등을 계산
- 계산 결과를 오케스트레이터가 캐싱/랭킹/그룹화한 뒤
  - **paper trading**: 내부 `PaperBroker`로 포지션을 모의 집행
  - **live trading**: ccxt를 통해 실제 주문(옵션)
- `aiohttp` 기반 대시보드 서버가 동일 프로세스에서 상태를 WS/HTTP로 제공

### 핵심 런타임 연결
- `main.py` → `core/orchestrator.LiveOrchestrator` + `core/dashboard_server.DashboardServer`
- `LiveOrchestrator` 내부에서
  - `core/data_manager.DataManager`: market/ohlcv/orderbook 업데이트
  - `engines/engine_hub.EngineHub`: 엔진 로딩 및 `decide_batch()` 호출
  - `core/paper_broker.PaperBroker`: paper 모드 거래
  - `core/risk_manager.RiskManager`: live 모드 리스크 가드레일
- 대시보드:
  - `core/dashboard_server._build_payload()`가 orchestrator의 최신 rows/positions/history를 payload로 구성
  - 프론트엔드 `dashboard_v2.html`이 WS(`/ws`)로 payload를 받아 테이블/차트 렌더

---

## 1.5) 파일/디렉토리 인덱스(빠른 탐색)

### 최상위(운영자가 가장 많이 보는 파일)
- `README.md`: 실행법/환경변수/대시보드 URL 요약
- `ARCHITECTURE.md`: 상위 구조도(mermaid) + 요약
- `config.py`: 포트/심볼/리스크/MC 파라미터 및 env 로딩
- `main.py`: 엔트리포인트(orch + dashboard + 루프 실행)
- `dashboard_v2.html`: 대시보드 프론트(테이블 컬럼 키 요구사항의 “정답”)
- `requirements.txt`: 파이썬 의존성

### `core/` (런타임/오케스트레이션)
- `orchestrator.py`: 시스템 중심(데이터→ctx→engine→캐시→집행→dashboard)
- `data_manager.py`: 티커/캔들/호가 비동기 갱신
- `dashboard_server.py`: aiohttp 서버(WS/HTTP), payload 구성
- `paper_broker.py`: paper trading 집행/포지션 갱신
- `risk_manager.py`: live trading 리스크 가드레일(마진비율/레버리지 sync)
- `state_store.py`: 상태 저장/복구(balance/positions/trade/equity)
- `economic_brain.py`: NAPV 기반 4-way(hold/reverse/switch/cash) 결정 유틸
- `opportunity_scoring.py`/`evaluation_utils.py`/`continuous_opportunity.py`: 기회비용/적분 점수/교체 로직 유틸
- `multi_timeframe_scoring.py`: MTF 점수 통합/스위칭(별도 유닛테스트 존재)
- `napv_engine_jax.py`: JAX 기반 NAPV 가속(실험/대체 구현 성격)
- `decision_service.py`: 결정 로직을 서비스로 분리한 간소화 버전(리팩터링 경로)

### `engines/` (신호/시뮬레이션/결정)
- `engine_hub.py`: 엔진 로딩 및 decide/decide_batch 호출
- `kelly_allocator.py`: 상관을 고려한 다변량 켈리 할당기
- `mc/`: MonteCarloEngine 구현(시뮬레이션/평가/정책/결정/JAX 가속)

### `tools/` (진단/실험)
- `tools/jax_cache_probe.py`: JAX compilation cache 실증 체크

---

## 2) 실행/모드

### 엔트리포인트
- `main.py`
  - orchestrator 및 dashboard 서버를 생성하고 asyncio 루프에서 실행합니다.

### 모드 개념
- **paper 모드**: `PaperBroker`가 `cap_frac`(자본비율) × `leverage`로 포지션/잔고를 모의 업데이트
- **live 모드**: `enable_orders=True`인 경우 ccxt 주문 경로가 활성화되며, `RiskManager`가 교차마진 기준 리스크를 감시/제어

(모드/환경변수/실행법은 `README.md`에 정리돼 있으며, 이 문서는 구조와 데이터 흐름 중심으로 설명합니다.)

---

## 3) 데이터 수집 (Market Data)

### 책임 모듈
- `core/data_manager.py` (`DataManager`)

### 주요 루프(개념)
- tickers: 현재가/기본 스냅샷
- ohlcv: 캔들 히스토리(수익률/변동성 등 신호 계산 입력)
- orderbook: bid/ask, spread, OFI/유동성 등 마이크로구조 피처 입력

### orchestrator가 쓰는 형태(개념)
- orchestrator는 심볼별로 최신 market snapshot과 closes(종가 리스트) 등을 `ctx`로 패키징해서 엔진으로 전달합니다.

---

## 4) 오케스트레이터 (시스템의 중심)

### 책임 모듈
- `core/orchestrator.py` (`LiveOrchestrator`)

### 핵심 설계: 2개의 루프
오케스트레이터는 “무거운 계산”과 “대시보드 갱신”을 분리합니다.

1) **`decision_worker_loop`**
- 엔진 `decide_batch()` 같은 무거운 작업을 thread로 실행하여 `_decision_cache`를 갱신
- paper/live 포지션 관리(진입/청산/리밸런싱 등)도 이 쪽에 붙어 있음

2) **`decision_loop`**
- `_decision_cache`에서 UI에 필요한 값만 뽑아 rows를 구성(`_rows_snapshot_cached()`)
- `_last_rows`를 갱신하고, `DashboardServer.broadcast()`로 WS 브로드캐스트

### 캐시 구조(개념)
- `_decision_cache[symbol]`:
  - 엔진의 raw decision + meta + 내부 디버그 정보
- `_last_rows`:
  - 대시보드 “현재 신호 테이블”에 그대로 들어가는 rows(list[dict])

### row 생성의 핵심
- `_row()`가 엔진 decision/meta에서
  - EVPH(시간정규화), SCORE%, NAPV%, EV%, T*(최적보유시간), HOLDμ 등 UI 컬럼에 필요한 값을 주입
- `_rows_snapshot_cached()`가 캐시 기반으로 **rank(RK)** 를 계산해 row에 부여

---

## 5) 엔진 허브

### 책임 모듈
- `engines/engine_hub.py` (`EngineHub`)

### 역할
- 여러 엔진을 로드하고, 심볼별/배치별로 `decide()`/`decide_batch()`를 호출
- JAX 결과(디바이스 배열 등)를 JSON 가능 형태로 sanitize

---

## 6) Monte Carlo 엔진 (MC)

### 엔트리 클래스
- `engines/mc/monte_carlo_engine.py` (`MonteCarloEngine`)
  - 여러 mixin을 다중상속으로 조합하여 기능을 분리

### 주요 믹스인(대표)
- `engines/mc/signal_features.py`: 입력 closes/orderbook에서 신호 피처/알파 구성
- `engines/mc/path_simulation.py`: GBM/테일샘플링 기반 경로 시뮬레이션
- `engines/mc/entry_evaluation.py`: 진입 후보의 EV/Win/CVaR/곡선(지평별 EV) 계산
- `engines/mc/exit_policy.py`: 보유 중 flip/score/stop 등 정책 롤포워드(특히 JAX 경로 지원)
- `engines/mc/decision.py`: 최종 액션/사이징/레버리지/EVPH 기록 등 “의사결정”

### 배치 평가(성능 핵심)
- `engines/mc/entry_evaluation.py: evaluate_entry_metrics_batch()`
  - 다심볼에 대해 JAX 정적 배치 크기(`JAX_STATIC_BATCH_SIZE=32`)로 패딩하여 JIT 안정성 확보
  - `simulate_paths_price_batch()`로 경로를 한 번 만들고, 여러 horizon 요약을 벡터화

---

## 7) 사이징 / 켈리(Kelly) / 레버리지

### 7.1 단일 심볼 내부(엔진 레벨)
- `engines/mc/decision.py` 내부에서
  - 1차로 leverage=1.0으로 베이스 메트릭(EV, sigma, win, cvar, 비용)을 얻고
  - 그 기반으로 최적 레버리지/사이즈를 선택합니다.

레버리지 최적화 경로는 2개입니다.
- GPU 최적화(가능 시): `engines/mc/leverage_optimizer_jax.py: find_optimal_leverage()`
- Fallback: 후보 레버리지 리스트를 훑으며 `kelly_with_cvar(win,tp,sl,cvar)` 기반으로 growth를 최대화

결과는 다음 형태로 `ctx_final`에 반영됩니다.
- `ctx_final["leverage"] = optimal_leverage`
- `ctx_final["size_frac"] = smoothed_size` (EMA로 완화)

그리고 다시 `evaluate_entry_metrics(ctx_final, ...)`를 호출해 최종 metrics(meta 포함)를 만듭니다.

### 7.2 다심볼 상관/분산 켈리(포트폴리오 레벨)
- `engines/kelly_allocator.py` (`KellyAllocator`)
  - 공분산 행렬 $\Sigma$ 와 기대수익 벡터 $\mu$로
    $$f^* = K \Sigma^{-1}\mu$$
    형태의 weights를 계산
  - (현재 코드상) pandas DataFrame 기반 공분산을 계산할 수 있고, JAX 공분산도 best-effort로 시도

> 실제 집행에서 “이 멀티바리엇 켈리 결과가 cap_frac로 강제 적용되는지”는 orchestrator의 리밸런싱/배분 경로(설정)에 의해 달라집니다. 반면 `decision.py` 내부의 `size_frac/leverage`는 단일 심볼 decision 결과로 꾸준히 생성됩니다.

---

## 8) EVPH, HOLDμ, T* 등 대시보드 핵심 컬럼의 의미

대시보드 `dashboard_v2.html`의 “현재 신호 테이블”은 다음 키를 기대합니다.

- `evph_p`: EVPH% (시간정규화 edge)
  - 설명: `ev_per_hour_adj × 100` (조정치가 없으면 raw EVPH를 쓰는 경로도 존재)
- `hold_mean_sec`: HOLDμ
  - 설명(프론트 툴팁): `B-1 hold mean` → 없으면 `event_t_mean/median` → 없으면 `T*`(optimal_horizon_sec)
- `rank`: RK
  - `_rows_snapshot_cached()`에서 EVPH 기반 등으로 계산되어 row에 주입
- `ev_score_p`: SCORE%
  - 엔진 meta의 `policy_ev_score_long/short` 등에서 파생
- `napv_p`: NAPV%
  - NAPV(할인적분 기반 가치) 계열 지표를 %로 표현
- `evp`: EV%
  - 기대값(EV) 계열을 %로 표현
- `optimal_horizon_sec`: T*
  - 최적 보유 시간(초)

**중요:** 위 값들은 “엔진 meta”와 “orchestrator row top-level” 양쪽에 존재할 수 있습니다. 최종적으로 UI는 row의 top-level 키(`rank`, `optimal_horizon_sec`, `evph_p` 등)를 직접 읽는 구조입니다.

---

## 9) 대시보드 서버 / 디버그 API

### 서버
- `core/dashboard_server.py` (`DashboardServer`)

### WS/HTTP
- `/` : 대시보드 HTML (`dashboard_v2.html`)
- `/ws` : 실시간 WS (payload broadcast)
- `/debug/payload` : 최신 full payload JSON
- `/api/status` : 핵심 상태(루프 지표, JAX warmup, live sync 정보 등)
- `/api/positions` : 포지션 스냅샷
- `/api/score_debug?symbol=...` : 심볼별 디버그 정보(오케스트레이터가 제공)
- `/api/runtime` (GET/POST) : 런타임 설정 조회/변경
- `/api/reset_tape` (POST) : trade tape/equity reset
- `/api/liquidate_all` (POST) : 전체 청산 트리거(paper/live 분기)

payload 구성은 `core/dashboard_server._build_payload()`에서 수행됩니다.

---

## 10) 리스크(라이브) 가드레일

- `core/risk_manager.py` (`RiskManager`)
  - 교차마진 기준 총자산(total equity)을 sizing 기준으로 사용
  - margin ratio(초기증거금/자산)가 임계 이상이면 신규 진입 차단 및 reduceOnly 축소를 시도
  - 거래소 레버리지 설정을 목표 레버리지 이상으로 맞추는 `sync_api_leverage()` 포함

---

## 11) 상태 저장(복구)

- `core/state_store.py` (`StateStore`)
  - `state/` 아래에 balance/positions/trade_tape/equity_history를 best-effort로 저장
  - 오래된(stale) 포지션(기본 24h 초과)은 로드 시 폐기

---

## 12) 성능 / JAX 캐시 / warmup

- JAX 관련 설정/워밍업은 orchestrator와 엔진 쪽에 분산돼 있습니다.
- 배치 경로는 정적 배치 패딩으로 JIT 재컴파일을 줄입니다.
- `tools/jax_cache_probe.py`로 **persistent compilation cache**가 실제로 쓰이는지 실증 체크할 수 있습니다.
  - 관찰 결과(이 레포 컨텍스트 기준): CPU 백엔드에서는 캐시 파일 생성/재사용이 확인되지만, Apple `METAL` 백엔드에서는 persistent cache가 실질적으로 생성/사용되지 않는 케이스가 있었습니다.

---

## 13) 테스트

- `tests/test_orchestrator_public_api.py`: orchestrator가 대시보드/메인에서 기대하는 public API를 유지하는지
- `tests/test_state_store.py`: `StateStore` 저장/로드/리셋 및 stale position 필터
- `tests/test_multi_timeframe_scoring.py`: `core/multi_timeframe_scoring.py` 유닛 테스트

---

## 14) 레포 루트의 운영 스크립트(요약)

루트에 있는 `check_*`, `close_positions_*`, `manual_liquidate.py` 등은
- 실행 중인 엔진 상태 확인
- 주문/포지션 조회/정리
- 비상 청산
등 운영 편의를 위한 유틸 스크립트 모음입니다.

---

## 15) 어디를 보면 “의사결정→집행→표시”가 끝까지 연결되는가

- **의사결정 계산(배치)**: `core/orchestrator.py`의 `decision_worker_loop()` → `EngineHub.decide_batch()` → `MonteCarloEngine`(mixins)
- **결과 캐시/랭킹/row 생성**: `core/orchestrator.py`의 `_decision_cache`, `_rows_snapshot_cached()`, `_row()`
- **대시보드 payload**: `core/dashboard_server._build_payload()` → WS(`/ws`) → `dashboard_v2.html`
- **paper 집행**: `core/orchestrator.py` → `core/paper_broker.PaperBroker`
- **live 리스크 가드**: `core/risk_manager.RiskManager`

---

## 부록 A) 용어 정리(실무적으로 쓰는 의미)

- EV: 기대 수익(또는 기대값) 관련 스칼라
- EVPH: EV를 시간으로 정규화한 지표(“한 시간당 edge”)
- T*: 최적 보유 시간(초)
- NAPV: 할인 적분 기반 가치(기회비용 $\rho$ 를 반영)
- SCORE: 엔진이 진입/보유 판단에 쓰는 스무딩된 점수 계열


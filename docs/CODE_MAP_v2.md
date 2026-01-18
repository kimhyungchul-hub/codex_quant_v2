# Code Map — 완전 확장판 (메서드/함수 수준, 한글 주석 포함)

이 문서는 **codex_quant** 레포지토리의 모든 핵심 모듈, 클래스, 주요 메서드/함수를 영어 설명과 한국어 주석을 병기하여 정리한 코드 맵입니다.

> **목적**: 리팩터링, 문서화, 테스트 작성 시 "무엇이 어디에 있는가"를 빠르게 확인하기 위한 참조 자료
> **주의**: 자동 스캔 + 수동 검토 기반. 코드가 바뀌면 이 문서도 갱신하세요.

---

## 목차
1. [설정 및 환경변수](#1-설정-및-환경변수--config)
2. [엔트리포인트](#2-엔트리포인트--entry-point)
3. [코어 런타임](#3-코어-런타임--core-runtime)
4. [엔진 허브 및 조정](#4-엔진-허브-및-조정--engine-hub)
5. [Monte Carlo 엔진 패밀리](#5-monte-carlo-mc-엔진-패밀리)
6. [PMaker / ML 모델](#6-pmaker--ml-모델)
7. [유틸리티](#7-유틸리티--utils)
8. [대시보드 UI 계약](#8-대시보드--ui-계약)
9. [JAX vs NumPy 경로](#9-jax-vs-numpy-경로)

---

## 1. 설정 및 환경변수 / Config

### `config.py`
전역 설정 및 환경변수 로딩을 담당하는 모듈.

| 함수/상수 | 설명 | 한글 |
|-----------|------|------|
| `_env_symbols(name, default)` | 환경변수에서 심볼 리스트 파싱 | 쉼표 구분 심볼 문자열 → 리스트 |
| `_load_env_file(path)` | `.env` 파일 로드 (덮어쓰지 않음) | 기존 환경변수 유지 |
| `PORT` | 대시보드 서버 포트 (기본: 9999) | — |
| `SYMBOLS` | 거래 대상 심볼 리스트 (기본 18개) | — |
| `MC_N_PATHS_LIVE` | 라이브 MC 시뮬레이션 경로 수 (기본: 16384) | — |
| `MC_N_PATHS_EXIT` | Exit policy MC 경로 수 (기본: 512) | — |
| `ENABLE_LIVE_ORDERS` | 실거래 주문 활성화 여부 | True=실거래, False=페이퍼 |
| `MAX_LEVERAGE` | 최대 허용 레버리지 (기본: 100) | — |
| `DEFAULT_LEVERAGE` | 기본 레버리지 (기본: 5) | — |
| `DECISION_REFRESH_SEC` | 결정 갱신 주기 (초) | — |
| `EXEC_MODE` | 실행 모드 (`maker_then_market` 등) | — |
| `MAKER_TIMEOUT_MS`, `MAKER_RETRIES`, `MAKER_POLL_MS` | 메이커 주문 타임아웃/재시도/폴링 설정 | — |

---

## 2. 엔트리포인트 / Entry point

### `main.py`
| 함수 | 설명 | 한글 |
|------|------|------|
| `_parse_args()` | CLI 인자 파싱 및 유효성 검사 | 명령행 옵션 처리 |
| `_apply_default_run_mode()` | 기본 런 모드 적용 | 환경변수/옵션에 따른 기본 실행 모드 설정 |
| `main()` | `LiveOrchestrator`와 `DashboardServer`를 생성/시작 | 전체 서비스 기동부 |

---

## 3. 코어 런타임 / Core runtime

### `core/orchestrator.py` — `LiveOrchestrator` 클래스
메인 오케스트레이션 루프. 심볼별 `ctx` 관리, 엔진 호출, 대시보드 브로드캐스트 담당.

| 메서드 | 시그니처 | 설명 | 한글 |
|--------|----------|------|------|
| `__init__` | `(exchange, symbols?, data_exchange?)` | 초기화, 엔진/데이터/PMaker 설정 | 거래소 연결, 심볼 등록, 상태 로드 |
| `_apply_mc_runtime_to_engines` | `() -> None` | MC 런타임 파라미터를 엔진에 전파 | `mc_n_paths_exit` 등 적용 |
| `_total_open_notional` | `() -> float` | 열린 포지션 총 노티널 계산 | — |
| `_pmaker_paper_sigma` | `(closes, window=60) -> float` | Paper PMaker용 sigma 추정 | — |
| `_pmaker_paper_momentum_z` | `(closes, sigma, window=10) -> float` | Paper PMaker용 모멘텀 z-score | — |
| `_pmaker_paper_probe_tick` | `(sym, ts_ms, ctx) -> None` | Paper PMaker 훈련 틱 | 시뮬레이션 fill + adverse move EMA 기록 |
| `runtime_config` | `() -> Dict` | 런타임 설정 반환 | 대시보드/API용 |
| `set_enable_orders` | `(enabled: bool) -> None` | 라이브 주문 활성화/비활성화 | — |
| `score_debug_for_symbol` | `(sym: str) -> Dict` | 심볼별 디버그 정보 반환 | — |
| `_snapshot_inputs` | `(sym) -> tuple` | 심볼별 시장/오더북 스냅샷 수집 | (price, closes, candles, bid, ask, spread) |
| `_build_decide_ctx` | `(sym, ...) -> Dict or None` | 엔진 컨텍스트 구성 | mu_sim, sigma_sim, pmaker_surv 등 포함 |
| `_extract_mc_meta` | `(decision) -> Dict` | decision.details에서 MC 메타 분리 | — |
| `_row` | `(sym, price, ts, decision, candles, ctx) -> Dict` | 대시보드용 평면화 row 생성 | 핵심 키 포워딩 |
| `_rows_snapshot` | `(ts_ms, apply_trades=False) -> List[Dict]` | 모든 심볼 평가 후 row 리스트 반환 | `decide_batch()` 호출 |
| `_rows_snapshot_cached` | `(ts_ms) -> List[Dict]` | 캐시된 decision 사용 | 대시보드 갱신용 |
| `_reconcile_state_with_exchange` | `() -> Dict` | 거래소 실제 포지션/잔고와 로컬 상태를 비교하여 강제 동기화 (좀비 포지션 제거/추가) | Live 모드 시작 시 1회 실행 |
| `_paper_open_position` | `(sym, side, size, price, ...) -> None` | Paper 포지션 열기 | — |
| `_paper_close_position` | `(sym, price, reason) -> None` | Paper 포지션 닫기 | — |
| `_paper_fill_price` | `(sym, side, target_price) -> float` | Paper 체결가 계산 | 슬리피지 적용 |
| `_paper_mark_position` | `(sym, price) -> None` | Paper 포지션 시가평가 | uPnL 업데이트 |
| `_paper_trade_step` | `(sym, decision, ctx, ts_ms) -> None` | Paper 트레이드 스텝 실행 | — |
| `_paper_exit_policy_signal` | `(sym, ctx, ts_ms) -> str or None` | Exit policy 신호 계산 | — |
| `_paper_init_exit_policy_state` | `(sym, decision) -> None` | Exit policy 상태 초기화 | — |
| `decision_worker_loop` | `async () -> None` | 백그라운드 decision 업데이트 루프 | 심볼 라운드로빈 |
| `decision_loop` | `async () -> None` | UI 갱신/브로드캐스트 루프 | — |

---

### `core/data_manager.py` — `DataManager` 클래스
비동기 시장 데이터 수집 및 버퍼 관리.

| 메서드 | 시그니처 | 설명 | 한글 |
|--------|----------|------|------|
| `__init__` | `(orch, symbols, data_exchange?)` | 초기화, 버퍼 생성 | — |
| `start` | `async () -> None` | 데이터 수집 루프 시작 | — |
| `stop` | `async () -> None` | 데이터 수집 루프 중지 | — |
| `fetch_prices_loop` | `async () -> None` | 가격 폴링 루프 | 틱 데이터 수집 |
| `preload_all_ohlcv` | `async () -> None` | 시작 시 OHLCV 로드 | 초기 캔들 데이터 |
| `fetch_ohlcv_loop` | `async () -> None` | OHLCV 폴링 루프 | — |
| `fetch_orderbook_loop` | `async () -> None` | 오더북 폴링 루프 | — |
| `get_btc_corr` | `(sym) -> float` | BTC 상관계수 계산 | — |
| `fetch_tickers` | `async () -> Dict` | 전체 틱커 조회 | — |
| `fetch_ohlcv` | `async (sym, limit) -> List` | OHLCV 데이터 조회 | — |
| `fetch_orderbook` | `async (sym) -> Dict` | 오더북 조회 | — |

**내부 버퍼**: `market`, `ohlcv_buffer`, `orderbook`, `_last_kline_ok_ms`, `_last_feed_ok_ms`

---

### `core/dashboard_server.py` — `DashboardServer` 클래스
대시보드 웹서버 및 WebSocket 브로드캐스트.

| 함수/메서드 | 설명 | 한글 |
|-------------|------|------|
| `_fallback_rows(orch, ts)` | 부트 시 기본 row 생성 | — |
| `_exec_stats_snapshot(orch)` | 실행 통계 스냅샷 | — |
| `_compute_portfolio(orch)` | 포트폴리오 계산 | → (equity, unreal, util, pos_list) |
| `_compute_eval_metrics(orch)` | 평가 메트릭 계산 | total_return 등 |
| `_build_payload(orch, rows, ...)` | JSON 페이로드 생성 | 대시보드 전송용 |
| `DashboardServer.start` | `async () -> None` | aiohttp 서버 시작 | — |
| `DashboardServer.stop` | `async () -> None` | 서버 중지 | — |
| `DashboardServer.ws_handler` | `async (request) -> WebSocketResponse` | WebSocket 핸들러 | — |
| `DashboardServer.broadcast` | `async (payload) -> None` | 모든 클라이언트에 브로드캐스트 | — |

**API 엔드포인트**: `/api/status`, `/api/positions`, `/api/score_debug`, `/api/runtime` (GET/POST)

---

### `core/risk_manager.py` — `RiskManager` 클래스
위험 관리 및 마진 모니터링.

| 메서드 | 시그니처 | 설명 | 한글 |
|--------|----------|------|------|
| `__init__` | `(orch)` | 초기화, 한도 설정 | — |
| `update_account_summary` | `(wallet_balance, total_equity, ...)` | 계정 요약 업데이트 | — |
| `get_total_equity` | `(fallback_wallet) -> float` | 총 자본 반환 | cross-margin equity |
| `get_margin_ratio` | `() -> float or None` | 마진 비율 반환 | — |
| `sync_api_leverage` | `async (sym, target_leverage, ts_ms) -> float` | API 레버리지 동기화 | — |
| `allow_new_entry_now` | `(ts_ms) -> bool` | 신규 진입 허용 여부 | 쿨다운 체크 |
| `_cooldown_new_entries` | `(ts_ms, reason) -> None` | 신규 진입 쿨다운 적용 | — |
| `_estimate_min_notional_usd` | `(sym) -> float` | 최소 노티널 추정 | — |

**`AccountSummary` 데이터클래스**: `total_equity`, `wallet_balance`, `free_balance`, `total_initial_margin`, `total_maintenance_margin`, `margin_ratio`

---

### `core/paper_broker.py` — `PaperBroker` 클래스
페이퍼 트레이딩 주문 시뮬레이터.

| 메서드 | 설명 | 한글 |
|--------|------|------|
| `submit_order` | 주문 제출 시뮬레이션 | — |
| `cancel_order` | 주문 취소 시뮬레이션 | — |
| `on_fill` | 체결 콜백 처리 | — |

---

## 4. 엔진 허브 및 조정 / Engine Hub

### `engines/engine_hub.py` — `EngineHub` 클래스
엔진 등록 및 의사결정 조정 허브.

| 메서드 | 시그니처 | 설명 | 한글 |
|--------|----------|------|------|
| `__init__` | `()` | 초기화, MC 엔진 등록 | — |
| `register` | `(engine) -> None` | 엔진 등록 | — |
| `unregister` | `(engine) -> None` | 엔진 해제 | — |
| `_sanitize` | `(obj) -> Any` | JAX DeviceArray → Python/NumPy 변환 | JSON 직렬화 가능하게 |
| `decide` | `(ctx: Dict) -> Dict` | 단일 심볼 의사결정 | 엔진 `decide()` 호출 |
| `decide_batch` | `(ctx_list: List[Dict]) -> List[Dict]` | 배치 의사결정 | JAX 가속 경로 사용, 실패 시 폴백 |

**내부 엔진 목록**: `engines` (기본: `MonteCarloEngine`)

---

## 5. Monte Carlo (MC) 엔진 패밀리

### `engines/mc/monte_carlo_engine.py` — `MonteCarloEngine` 클래스
MC 시뮬레이션 기반 의사결정 엔진. 다중 믹스인 상속.

| 믹스인 | 설명 | 한글 |
|--------|------|------|
| `MonteCarloPathSimulationMixin` | 경로 시뮬레이션 | — |
| `MonteCarloFirstPassageMixin` | First-passage 계산 | — |
| `MonteCarloExitPolicyMixin` | Exit policy 메트릭 | — |
| `MonteCarloEntryEvaluationMixin` | 진입 평가 | — |
| `MonteCarloDecisionMixin` | 의사결정 로직 | — |
| ... (총 13개 믹스인) | | |

| 메서드 | 설명 | 한글 |
|--------|------|------|
| `__init__` | 초기화, alpha_hit 트레이너 설정 | — |
| `tp_sl_targets_for_horizon` | TP/SL 목표 계산 | 호라이즌/시그마 기반 |
| `alpha_hit_confidence` | AlphaHit 예측 신뢰도(엔트로피 기반) | beta 가중치 보정 |

---

### `engines/mc/decision.py` — `MonteCarloDecisionMixin`
평가 결과를 액션으로 변환.

| 메서드 | 시그니처 | 설명 | 한글 |
|--------|----------|------|------|
| `decide` | `(ctx: Dict) -> Dict` | 의사결정 실행 | action, ev, confidence, reason, meta 반환 |
| `_get_params` | `(regime, ctx) -> Dict` | MC 파라미터 추출 | n_paths, use_jax, tail_mode 등 |

**반환 키**: `action`, `ev`, `ev_raw`, `confidence`, `reason`, `meta`, `size_frac`, `optimal_leverage`, `optimal_size`, `boost`

**Funnel Filter 로직**:
1. NAPV 필터: `napv < threshold`
2. EV 필터: `ev <= 0`
3. Win rate 필터: `win < floor`
4. CVaR 필터: `cvar1 < floor`
5. Event CVaR 필터: `event_cvar_r < floor`

---

### `engines/mc/entry_evaluation.py` — `MonteCarloEntryEvaluationMixin`
진입 평가의 핵심 로직. (3445줄)

| 메서드 | 시그니처 | 설명 | 한글 |
|--------|----------|------|------|
| `_get_execution_costs` | `(ctx, params?) -> Dict` | 실행 비용 계산 | fee_roundtrip, exec_oneway, impact_cost |
| `evaluate_entry_metrics` | `(ctx, params, seed) -> Dict` | 단일 심볼 평가 | **핵심 API** |

**`evaluate_entry_metrics` 내부 처리 흐름**:
1. ctx에서 `price`, `mu_sim`, `sigma_sim`, `closes` 추출
2. sigma가 없으면 closes에서 계산 (multi-window blend)
3. `_signal_alpha_mu_annual_parts()` 호출 → `mu_alpha` 계산
4. PMaker fill rate로 `mu_alpha` 보정 (boost)
5. EMA 스무딩 적용 (선택적)
6. `adjust_mu_sigma(mu_alpha, sigma, regime)` 호출
7. 슬리피지/수수료 계산
8. 경로 시뮬레이션 → horizon별 EV/p_pos/CVaR 계산
9. 최적 horizon 선택 → 최종 결과 반환

**반환 키**: `ev`, `ev_raw`, `win`, `cvar`, `best_h`, `direction`, `kelly`, `size_frac`, `can_enter`, `sigma_sim`, `mu_adjusted`, `policy_*` (per-horizon), `mu_alpha_*`, `pmaker_*`, `perf` (타이밍)

---

### `engines/mc/path_simulation.py` — `MonteCarloPathSimulationMixin`
GBM 경로 시뮬레이션.

| 메서드 | 시그니처 | 설명 | 한글 |
|--------|----------|------|------|
| `simulate_paths_price` | `(seed, s0, mu, sigma, n_paths, n_steps, dt, return_jax=False) -> ndarray` | 단일 심볼 경로 생성 | JAX/NumPy 분기 |
| `simulate_paths_price_batch` | `(seeds, s0s, mus, sigmas, n_paths, n_steps, dt) -> jnp.ndarray` | 다중 심볼 배치 경로 생성 | vmap 사용 |
| `simulate_paths_netpnl` | `(seed, s0, mu, sigma, direction, leverage, n_paths, horizons, dt, fee_roundtrip) -> Dict[int, ndarray]` | horizon별 net PnL 경로 | — |

**내부 함수**:
- `_simulate_paths_price_jax_core` — JAX JIT 컴파일된 GBM 코어
- `_simulate_paths_price_batch_jax` — vmap 적용 배치 버전

---

### `engines/mc/exit_policy.py` — `MonteCarloExitPolicyMixin`
Exit policy 메트릭 계산. (857줄)

| 메서드 | 시그니처 | 설명 | 한글 |
|--------|----------|------|------|
| `compute_exit_policy_metrics` | `(symbol, price, mu, sigma, leverage, direction, fee_roundtrip, exec_oneway, impact_cost, regime, horizon_sec, ...) -> Dict` | Exit 통계 계산 | p_pos_exit, ev_exit, exit_t_mean_sec 등 |
| `_execution_mix_from_survival` | `(meta, fee_maker, fee_taker, horizon_sec, sigma_per_sec, prefix, delay_penalty_mult) -> Dict` | PMaker survival 기반 실행 믹스 | — |
| `_sigma_per_sec` | `(sigma, dt) -> float` | 초당 sigma 변환 | — |

**Exit 조건**:
- TP hit (Take Profit)
- SL hit (Stop Loss)
- Timeout (horizon 만료)
- Drawdown stop (dd_stop_roe)
- Score flip (방향 전환)
- Hold bad ticks (연속 악화)

---

### `engines/mc/first_passage.py` — `MonteCarloFirstPassageMixin`
First-passage time 계산.

| 메서드 | 시그니처 | 설명 | 한글 |
|--------|----------|------|------|
| `mc_first_passage_tp_sl` | `(s0, tp_pct, sl_pct, mu, sigma, dt, max_steps, n_paths, cvar_alpha=0.05, timeout_mode="flat", seed=None, side="LONG") -> Dict` | TP/SL first-passage 시뮬레이션 | event_p_tp, event_p_sl, event_ev_r 등 |

**반환 키**: `event_p_tp`, `event_p_sl`, `event_p_timeout`, `event_ev_r`, `event_cvar_r`, `event_t_median`, `event_t_mean`

---

### `engines/mc/jax_backend.py`
JAX 백엔드 초기화 및 헬퍼.

| 항목 | 설명 | 한글 |
|------|------|------|
| `_JAX_OK` | JAX 사용 가능 여부 플래그 | — |
| `_jax_mc_device()` | MC용 디바이스 반환 | GPU/Metal 우선 |
| `jax`, `jnp`, `lax`, `jrand` | JAX 모듈 re-export | — |
| `jax_covariance` | JAX 공분산 계산 | — |
| `summarize_gbm_horizons_jax` | 단일 심볼 horizon 요약 | — |
| `summarize_gbm_horizons_multi_symbol_jax` | 다중 심볼 horizon 요약 | — |

---

## 6. PMaker / ML 모델

### `engines/p_maker_survival_mlp.py`
PMaker 체결 확률 예측 MLP.

| 클래스 | 메서드 | 설명 | 한글 |
|--------|--------|------|------|
| `HazardMLP` | `forward(x)` | 순전파 | — |
| `PMakerSurvivalMLP` | `featurize(x)` | 피처 벡터 생성 | spread_pct, sigma, ofi_z 등 |
| | `predict_proba(x)` | 체결 확률 예측 | — |
| | `update_one_attempt(...)` | 단일 시도 업데이트 | replay buffer에 추가 |
| | `train_from_replay(steps, batch_size)` | 리플레이 학습 | — |
| | `sym_fill_mean(sym)` | 심볼별 평균 fill rate | — |
| | `save(path)` / `load(path)` | 모델 저장/로드 | — |

---

### `engines/pmaker_manager.py` — `PMakerManager` 클래스
PMaker 모델 생명주기 관리.

| 메서드 | 설명 | 한글 |
|--------|------|------|
| `__init__` | 초기화, surv 모델 로드 | — |
| `status_dict` | 대시보드용 상태 반환 | — |
| `save_model` | 모델 저장 | — |
| `enabled` | PMaker 활성화 여부 | — |

---

## 7. 유틸리티 / Utils

### `utils/helpers.py`
공통 헬퍼 함수.

| 함수 | 시그니처 | 설명 | 한글 |
|------|----------|------|------|
| `_env_bool` | `(name, default=False) -> bool` | 환경변수 불리언 파싱 | — |
| `_env_int` | `(name, default) -> int` | 환경변수 정수 파싱 | — |
| `_env_float` | `(name, default) -> float` | 환경변수 실수 파싱 | — |
| `_load_env_file` | `(path) -> bool` | .env 파일 로드 (덮어쓰지 않음) | — |
| `_load_env_file_override` | `(path) -> bool` | .env 파일 로드 (덮어씀) | state/bybit.env 용 |
| `now_ms` | `() -> int` | 현재 시간 (밀리초) | — |
| `_safe_float` | `(x, default=0.0) -> float` | 안전한 float 변환 | — |
| `_sanitize_for_json` | `(obj, _depth=0) -> Any` | JSON 직렬화 가능하게 변환 | NaN/Inf → None, numpy → list |
| `_calc_rsi` | `(closes, period=14) -> float` | RSI 계산 | — |

---

### `engines/kelly_allocator.py` — `KellyAllocator` 클래스
Kelly 기반 포지션 사이징.

| 함수/메서드 | 설명 | 한글 |
|-------------|------|------|
| `kelly_fraction(returns, var, ...)` | Kelly 비율 계산 | — |
| `KellyAllocator` 클래스 | 목표 포지션 비중 산출 | — |

---

### `engines/mc_risk.py`
MC 리스크 헬퍼.

| 함수 | 설명 | 한글 |
|------|------|------|
| `compute_cvar(...)` | CVaR 계산 | — |
| `kelly_fraction(...)` | Kelly 비율 | — |
| `kelly_with_cvar(win_rate, tp_est, sl_est, cvar)` | CVaR 보정 Kelly | — |
| `should_exit_position(...)` | 청산 여부 판단 | — |

---

## 8. 대시보드 / UI 계약

### `LiveOrchestrator._row()` 반환 키 (대시보드에서 사용)
| 키 | 설명 | 한글 |
|---|------|------|
| `symbol` | 심볼명 | — |
| `price` | 현재가 | — |
| `status` | 상태 (`LONG`, `SHORT`, `WAIT`) | — |
| `action_type` | 액션 타입 | — |
| `ev`, `ev_raw` | EV 값 | — |
| `conf` | 신뢰도 (win rate) | — |
| `kelly` | Kelly 비율 | — |
| `optimal_leverage` | 최적 레버리지 | — |
| `regime` | 레짐 | — |
| `mu_alpha`, `mu_alpha_raw` | 신호 알파 | — |
| `mu_alpha_pmaker_fill_rate` | PMaker fill rate | — |
| `mu_alpha_pmaker_boost` | PMaker boost 값 | — |
| `policy_ev_score_long`, `policy_ev_score_short` | 방향별 EV 점수 | — |
| `policy_ev_per_h`, `policy_p_pos_per_h`, `policy_cvar_per_h` | horizon별 배열 | — |
| `execution_cost`, `expected_spread_cost`, `slippage_dyn` | 비용 관련 | — |
| `event_p_tp`, `event_p_sl`, `event_p_timeout` | 이벤트 확률 | — |
| `event_ev_r`, `event_cvar_r` | 이벤트 EV/CVaR | — |
| `pos_*` (`pos_roe`, `pos_leverage`, `pos_cap_frac`, `pos_pnl`) | 포지션 관련 | — |
| `funnel_reason` | 필터링 사유 | — |
| `candles` | 최근 캔들 데이터 | — |

### 클라이언트에서 파생하는 값
- `ev_score_p`: `policy_ev_score_long`/`policy_ev_score_short` 기반
- `napv_p`: 백엔드 `napv` 필드 필요
- `cost_bp`: `execution_cost` 기반

---

## 9. JAX vs NumPy 경로

### JAX 경로 특징
- **정적 배치 크기** 필요 (`JAX_STATIC_BATCH_SIZE = 32`)
- JIT 컴파일로 다중 심볼/다중 경로 병렬화
- `DeviceArray` → Python `float`/`int` 변환 필수 (JSON 직렬화용)
- GPU/Metal 가속 가능

### NumPy/CPU 경로
- 예외/디바이스 불가 시 폴백
- 디버깅 용이, 로컬 개발에 안정적

### 분기 위치
- `engines/mc/path_simulation.py`: `simulate_paths_*`
- `engines/mc/jax_backend.py`: `summarize_gbm_horizons_*`
- `engines/mc/exit_policy_jax.py`: `simulate_exit_policy_rollforward_jax`

---

## 변경 로그
| 날짜 | 내용 | 파일 |
|------|------|------|
| 2026-01-19 | 이상 징후 감지/텔레그램 알림/대시보드 alerts 노출 추가 | `main_engine_mc_v2_final.py`, `core/dashboard_server.py`, `dashboard_v2.html` |
| 2026-01-19 | Bybit Hedge Mode positionIdx/레버리지 동기화/킬 스위치/메이커 주문 집행 추가 | `main_engine_mc_v2_final.py`, `core/risk_manager.py` |
| 2026-01-19 | 주문서 병렬 수집 및 예외 처리 강화 | `main_engine_mc_v2_final.py` |
| 2026-01-19 | LiveOrchestrator 주문 멱등성 보강 (재시도 전용 ID 재사용) | `main_engine_mc_v2_final.py` |
| 2026-01-19 | LiveOrchestrator 초기 상태 동기화 추가 (`_reconcile_state_with_exchange`) | `main_engine_mc_v2_final.py` |
| 2026-01-17 | CODE_MAP을 완전 확장 (메서드/함수 수준, 한글 주석 포함) | `docs/CODE_MAP.md` |

---

*작성 일자: 2026-01-17*

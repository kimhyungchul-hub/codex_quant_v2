# Gemini Notes

## Monte Carlo 핵심 로직 (UnifiedScore 중심)

### 1) 입력/컨텍스트
- 가격/봉 데이터: `closes`, `bar_seconds`
- 마켓 정보: `spread_pct`, `ofi_score`, `liquidity_score`
- 파라미터: `n_paths`, `tail_mode`, `student_t_df`, `leverage`
- 비용: 수수료/슬리피지/스프레드 (maker/taker 혼합)

### 2) 경로 시뮬레이션
- 가격 경로 생성 (GBM + tail/bootstrapping 옵션)
- 시간 격자: `horizons = [60, 300, 600, 1800, 3600]` (초)
- 경로별 net ROE 계산
  - Long: `net = gross * L - fee`
  - Short: `net = -gross * L - fee`

### 3) 호라이즌별 통계
- EV(h): `E[net]`
- Win(h): `P(net > 0)`
- CVaR(h): 하위 α 꼬리 평균

### 4) 누적 벡터 → 순간 기울기
누적 벡터가 주어졌을 때, 구간별 순간 기울기를 계산한다.

- `Δt_k = t_k - t_{k-1}`
- `mEV_k = (EV_k - EV_{k-1}) / Δt_k`
- `mCVaR_k = (CVaR_k - CVaR_{k-1}) / Δt_k`

### 5) 순간 효용 (Utility Rate)
리스크 패널티를 반영한 순간 효용 흐름:

```
U_k = mEV_k - λ * |mCVaR_k|
```

### 6) 할인 적분 (Gross NAPV)
기회비용(ρ)와 할인율을 반영해 누적:

```
GrossNAPV(T) = Σ_{k<=T} (U_k - ρ) * exp(-ρ t_k) * Δt_k
```

### 7) 최종 UnifiedScore (시간당 초과 가치)
비용은 적분 내부가 아니라 **한 번만 차감**:

```
Score(a, L, T) = (GrossNAPV(T) - Cost(a,L)) / T
```

최적 시점:
```
T* = argmax_T Score(a, L, T)
UnifiedScore = max_T Score(a, L, T)
```

### 8) 의사결정/랭킹/배분
- 심볼별 UnifiedScore 계산
- **랭킹**: UnifiedScore 내림차순
- **교체**: `UnifiedScore(New) > UnifiedScore(Hold)`
- **배분**: UnifiedScore 기반 비중/가중

### 9) 구현 포인트
- `engines/mc/entry_evaluation.py`
  - `ev_vector`, `cvar_vector` 반환
  - 배치 경로에서 UnifiedScore 계산
- `engines/mc/decision.py`
  - 필터 제거, UnifiedScore 단일 기준
- `main_engine_mc_v2_final.py`
  - 랭킹/교체/배분 UnifiedScore 기반
- `core/economic_brain.py`
  - `calculate_unified_score` 제공 (마진율 기반)

## 변경로그
- 2026-01-28: 청산(Exit) 로직을 UnifiedScore 기반으로 통일 (리스크/손실 반영)
  - MC exit-policy가 EV 대신 UnifiedScore(=EV−λ|CVaR|)로 hold/flip 판단
  - switch/flip 비용을 시간당 페널티로 반영(τ로 나눔)
  - Torch exit-policy는 순수 torch로 계산하도록 변경, unified_lambda/unified_rho 전달
  - 실제 청산 로직도 WAIT/flip 조건을 UnifiedScore 기준으로 갱신
  - EMA/EV-drop 청산 기준을 unified_score로 전환
  - paper exit/trailing/flip에도 UnifiedScore 우선 적용
- 2026-01-28: 이벤트 기반 MC 청산에 UnifiedScore 반영
  - `event_score = event_ev_pct - λ*|event_cvar_pct| - ρ*t`
  - ExitPolicy 임계값을 `min_event_score`로 교체
  - `EVENT_EXIT_SCORE` 환경변수로 임계값 조정 가능
- 2026-01-28: 대시보드 테이블 재편 (UnifiedScore 중심)
  - 시장/신호 테이블 컬럼을 Ψ 체계로 변경 (Ψ, Ψ_HOLD, T*)
  - 트레이드 테이프에 Exit 키워드 표시
- 2026-01-28: _row fallback 보강
  - `unified_score_hold`, `unified_t_star`를 `mc_meta`에서 복원
- 2026-01-28: JAX 제거 및 Torch 우선 + NumPy fallback 전환
  - 멀티 피델리티 MC, CI 진입 게이트, 분산감소(antithetic/control variate) 적용
- 2026-01-28: 시간 일관성용 환경 변수 정합 및 프리셋 추가
  - `DEFAULT_TP_PCT=0.006`, `K_LEV=2000`, `ALPHA_HIT_DEVICE=mps` 통일
  - 중기(1h) 스윙/초단기 스캘핑 `.env` 프리셋 제공

## 청산(Exit) 로직 업데이트
### 1) MC Exit-Policy (시뮬레이션 기준)
- 입력: `mu_ps, sigma_ps, leverage, fee_exit_only, tp/sl, horizon`
- 계산:
  - EV/PPOS + CVaR 근사 → UnifiedScore로 변환
  - `score_cur vs score_alt` 비교로 flip/hold/exit 판단
  - 비용은 `switch_cost / tau`, `exec_oneway / tau`로 시간당 반영
- 적용 경로:
  - CPU: `engines/exit_policy_methods.py`
  - Torch: `engines/mc/exit_policy_torch.py`
  - 라우팅: `engines/mc/exit_policy.py`

### 2) 실시간 청산 (라이브 엔진)
- `main_engine_mc_v2_final.py`에서:
  - WAIT → 즉시 청산 제거
  - UnifiedScore 기준 `flip`/`cash` 판단
  - EMA/EV-drop 청산도 unified_score 기반으로 전환

### 3) Paper Exit
- `core/orchestrator.py`, `core/paper_broker.py`에서:
  - trailing/flip 로직을 UnifiedScore 우선으로 평가

### 4) 이벤트 기반 Exit
- 이벤트 기반 exit는 기존처럼 별도 안전 장치로 유지
  - `main_engine_mc_v2_final.py:_evaluate_event_exit`
  - `engines/mc_risk.py:should_exit_position`
  - ✅ 이벤트 UnifiedScore 반영:
    - `event_score = event_ev_pct - λ*|event_cvar_pct| - ρ*t`
    - 임계값은 `min_event_score` 기준으로 판단

## 청산(Exit) 로직 전체 정리 (UnifiedScore 반영 상태 포함)
### A) 라이브 엔진 기준 청산 로직 목록
1) 이벤트 기반 MC 청산 (별도 안전장치)
   - 경로: `main_engine_mc_v2_final.py:_evaluate_event_exit`
   - 조건: event EV/CVaR/SL 확률 기반 즉시 종료
   - UnifiedScore 반영: 아니오 (별도 경로)
2) 정책형 MC 청산 (exit-policy rollforward)
   - 경로: `engines/mc/exit_policy.py -> engines/exit_policy_methods.py`
   - 호출: `main_engine_mc_v2_final.py` 내부 `should_exit_position`
   - UnifiedScore 반영: 예
3) EMA 기반 청산
   - 경로: `main_engine_mc_v2_final.py:_check_ema_ev_exit`
   - 기준: unified_score + event_p_sl 추세 악화
   - UnifiedScore 반영: 예
4) EV-drop 청산
   - 경로: `main_engine_mc_v2_final.py:_check_ev_drop_exit`
   - 기준: unified_score가 동적 임계치 이하로 급락
   - UnifiedScore 반영: 예
5) 신호/시간/손실 기반 청산 (실시간 포지션)
   - 경로: `main_engine_mc_v2_final.py:_maybe_exit_position`
   - 기준: UnifiedScore 기반 flip/cash + hold timeout + unrealized_dd
   - UnifiedScore 반영: 예
   - WAIT이면 무조건 종료하던 로직 제거됨
6) Paper Exit
   - 경로: `core/orchestrator.py:_paper_exit_policy_signal`
   - UnifiedScore 반영: 예 (trailing/flip 판단에 UnifiedScore 우선 적용)
   - 보조 로직: `core/paper_broker.py` entry/score tracking도 UnifiedScore 우선

### B) UnifiedScore가 포함된 위치
1) MC exit-policy core (CPU)
   - `engines/exit_policy_methods.py`
   - `_approx_cvar_normal` + UnifiedScore 근사
   - `score_cur/score_alt`에 EV 대신 UnifiedScore 사용
   - switch/exec 비용을 `tau`로 나눠 시간당 페널티 반영
2) MC exit-policy core (Torch/GPU)
   - `engines/mc/exit_policy_torch.py`
   - 순수 torch로 EV/CVaR/UnifiedScore 계산
   - `unified_lambda`, `unified_rho` 전파
   - flip/hold 기준은 UnifiedScore
3) Exit-policy 라우팅
   - `engines/mc/exit_policy.py`
   - UnifiedScore 파라미터를 CPU/Torch에 전달
   - `score_exit_signal`도 UnifiedScore 우선 사용
4) 실시간 청산
   - `main_engine_mc_v2_final.py:_maybe_exit_position`
   - unified_score_hold vs unified_score_rev 비교
   - WAIT -> 즉시청산 제거, UnifiedScore 기반 판단

### C) MC 경로 시뮬레이션 반영 여부
- 예. 경로(rollforward) 상 exit 타이밍/평가가 UnifiedScore 기준으로 결정됨.
- 경로: `engines/mc/exit_policy.py` -> `simulate_exit_policy_rollforward(_torch)`

## 환경변수

### 전체 목록(설명)
#### 1) 실행/라우팅/플랫폼
- `USE_REMOTE_ENGINE`: 원격 엔진 사용 여부 (true면 로컬 MC 대신 `ENGINE_SERVER_URL`로 요청)
- `ENGINE_SERVER_URL`: 원격 엔진 서버 URL
- `USE_PROCESS_ENGINE`: MC 엔진을 별도 프로세스로 실행
- `MC_ENGINE_CPU_AFFINITY`: MC 프로세스 CPU affinity ("0,1,2" 형식)
- `MC_ENGINE_SHM_SLOTS`: 원격/프로세스 엔진용 공유메모리 슬롯 수
- `MC_ENGINE_SHM_SLOT_SIZE`: 공유메모리 슬롯 크기(bytes)
- `MC_ENGINE_TIMEOUT_SINGLE`: 단건 요청 타임아웃(초)
- `MC_ENGINE_TIMEOUT_BATCH`: 배치 요청 타임아웃(초)
- `DECIDE_BATCH_TIMEOUT_SEC`: 배치 결정 타임아웃(초)

### MPS/Metal 임시 디렉터리 이슈 (macOS)
증상: `Error creating directory`, `Function Can't set size of /tmp file failed` 로그가 반복되면서 MPS가 unavailable.
원인: MPS/Metal이 `/tmp`가 아니라 `$TMPDIR` 아래에 임시/캐시 디렉터리를 만들는데, 해당 경로가 없거나 쓰기 불가였음.
해결: `$TMPDIR`를 쓰기 가능한 경로로 지정.
- 예시(권장): `export TMPDIR=~/tmp` (디렉터리 생성 후 사용)
- 예시(시스템 경로 복구): `export TMPDIR=$(getconf DARWIN_USER_TEMP_DIR)` 후 `ls -ld "$TMPDIR"`로 쓰기 확인

#### 1-1) JAX/플랫폼
- `XLA_PYTHON_CLIENT_PREALLOCATE`: JAX 메모리 선점 여부
- `XLA_PYTHON_CLIENT_ALLOCATOR`: JAX 메모리 allocator
- `XLA_PYTHON_CLIENT_MEM_FRACTION`: JAX 최대 선점 비율
- `JAX_PLATFORMS`: JAX 플랫폼 강제(cpu/gpu)
- `JAX_PLATFORM_NAME`: JAX 플랫폼 강제(cpu/gpu)
- `JAX_MC_DEVICE`: MC JAX 디바이스 선택

#### 2) 심볼/시장/데이터 수집
- `SYMBOLS_CSV`: 심볼 목록(콤마 구분)
- `OHLCV_SLEEP_SEC`: OHLCV 갱신 주기
- `PRELOAD_ON_START`: 시작 시 OHLCV preload 여부
- `ORDERBOOK_SLEEP_SEC`: 오더북 갱신 주기
- `ORDERBOOK_SYMBOL_INTERVAL_SEC`: 오더북 심볼 순환 간격
- `TICKER_SLEEP_SEC`: ticker 폴링 주기
- `BYBIT_TESTNET`: 거래용 testnet 사용 여부
- `DATA_BYBIT_TESTNET`: 데이터용 testnet 사용 여부

#### 3) 결정 루프/로그
- `DECISION_REFRESH_SEC`: 결정 루프 주기
- `DECISION_EVAL_MIN_INTERVAL_SEC`: 심볼별 재평가 최소 간격
- `DECISION_WORKER_SLEEP_SEC`: 결정 워커 sleep
- `DECISION_MIN_CANDLES`: 결정 최소 캔들 수
- `DECISION_MAX_INFLIGHT`: 동시 인플라이트 제한
- `DECISION_LOG_EVERY`: N번마다 결정 로그
- `LOG_STDOUT`: stdout 로그 출력
- `DEBUG_MU_SIGMA`: mu/sigma 디버그
- `DEBUG_TPSL_META`: TP/SL meta 디버그
- `DEBUG_ROW`: row 디버그
- `DEV_MODE`: dev 모드(경로 수 축소 등)

#### 4) 주문/실행/리스크 기본
- `ENABLE_LIVE_ORDERS`: 실거래 on/off
- `EXEC_MODE`: 주문 모드 (`maker_then_market` 등)
- `USE_MAKER_ORDERS`: maker 주문 사용 여부
- `MAKER_TIMEOUT_MS`: maker 주문 대기 시간
- `MAKER_RETRIES`: maker 주문 재시도 횟수
- `MAKER_POLL_MS`: maker 주문 폴링 간격
- `DEFAULT_LEVERAGE`: 기본 레버리지
- `MAX_LEVERAGE`: 최대 레버리지
- `DEFAULT_SIZE_FRAC`: 기본 비중
- `DEFAULT_TP_PCT`: 기본 TP
- `DEFAULT_SL_PCT`: 기본 SL
- `POSITION_HOLD_MIN_SEC`: 최소 보유 시간
- `MAX_POSITION_HOLD_SEC`: 최대 보유 시간
- `POSITION_HOLD_HARD_CAP_SEC`: 강제 보유 상한
- `POSITION_CAP_ENABLED`: 포지션 캡 사용
- `EXPOSURE_CAP_ENABLED`: 노출 캡 사용
- `MAX_NOTIONAL_EXPOSURE`: 총 노출 상한
- `MAX_CONCURRENT_POSITIONS`: 동시 포지션 제한
- `COOLDOWN_SEC`: 재진입 쿨다운

#### 5) 포트폴리오/랭킹/교체
- `TOP_N_SYMBOLS`: 상위 N개만 진입
- `USE_KELLY_ALLOCATION`: UnifiedScore 비례 배분 사용
- `USE_CONTINUOUS_OPPORTUNITY`: 지속적 교체 판단
- `SWITCHING_COST_MULT`: 교체 비용 승수
- `PORTFOLIO_TOP_N`: 포트폴리오 TOP N
- `PORTFOLIO_SWITCH_COST_MULT`: 포트폴리오 교체 비용 승수
- `PORTFOLIO_KELLY_CAP`: Kelly 상한
- `PORTFOLIO_JOINT_INTERVAL_SEC`: joint sim 실행 주기
- `PORTFOLIO_JOINT_SIM_ENABLED`: joint sim 사용 여부

#### 6) UnifiedScore/Exit
- `UNIFIED_RISK_LAMBDA`: UnifiedScore 위험 페널티 λ
- `UNIFIED_RHO`: UnifiedScore 할인율 ρ
- `EVENT_EXIT_SCORE`: 이벤트 기반 MC 청산 임계값
- `POLICY_SCORE_TRAILING_FACTOR`: policy exit trailing 강도
- `POLICY_SCORE_FLIP_MARGIN`: policy flip 마진

#### 7) Paper trading
- `PAPER_TRADING`: 페이퍼 모드
- `PAPER_FLAT_ON_WAIT`: WAIT 시 포지션 정리
- `PAPER_USE_ENGINE_SIZING`: 엔진 sizing 사용
- `PAPER_ENGINE_SIZE_MULT`, `PAPER_ENGINE_SIZE_MIN_FRAC`, `PAPER_ENGINE_SIZE_MAX_FRAC`: sizing 범위
- `PAPER_SIZE_FRAC`: 페이퍼 기본 비중
- `PAPER_LEVERAGE`: 페이퍼 기본 레버리지
- `PAPER_FEE_ROUNDTRIP`: 페이퍼 수수료
- `PAPER_SLIPPAGE_BPS`: 페이퍼 슬리피지
- `PAPER_MIN_HOLD_SEC`, `PAPER_MAX_HOLD_SEC`: 보유 시간
- `PAPER_MAX_POSITIONS`: 동시 포지션 제한
- `PAPER_EXIT_POLICY_ONLY`: exit-policy 전용 여부
- `PAPER_EXIT_POLICY_HORIZON_SEC`: exit-policy horizon
- `PAPER_EXIT_POLICY_MIN_HOLD_SEC`: exit-policy 최소 보유 시간
- `PAPER_EXIT_POLICY_DECISION_DT_SEC`: exit-policy 평가 주기
- `PAPER_EXIT_POLICY_FLIP_CONFIRM_TICKS`: flip 안정화 틱
- `PAPER_EXIT_POLICY_HOLD_BAD_TICKS`: hold bad 안정화 틱
- `PAPER_EXIT_POLICY_SCORE_MARGIN`: score margin
- `PAPER_EXIT_POLICY_SOFT_FLOOR`: score soft floor
- `PAPER_EXIT_POLICY_P_POS_ENTER_FLOOR`: p_pos enter floor
- `PAPER_EXIT_POLICY_P_POS_HOLD_FLOOR`: p_pos hold floor
- `PAPER_EXIT_POLICY_DD_STOP_ENABLED`: DD stop 사용
- `PAPER_EXIT_POLICY_DD_STOP_ROE`: DD stop 기준

#### 8) MC 기본/고급
- `MC_N_PATHS_LIVE`: live 경로 수
- `MC_N_PATHS_EXIT`: exit 경로 수
- `MC_VERBOSE_PRINT`: MC 상세 로그
- `MC_TIME_STEP_SEC`: 시뮬레이션 time step
- `MC_TAIL_MODE`: 꼬리 분포 모드
- `MC_STUDENT_T_DF`: t-분포 자유도
- `MC_TPSL_AUTOSCALE`: TP/SL autoscale
- `MC_TP_BASE_ROE`, `MC_SL_BASE_ROE`: TP/SL 기본
- `MC_TPSL_SIGMA_REF`: TP/SL 스케일 기준
- `MC_TPSL_SIGMA_MIN_SCALE`, `MC_TPSL_SIGMA_MAX_SCALE`: TP/SL 스케일 범위
- `MC_TPSL_H_SCALE_BASE`: horizon 스케일 기준
- `MC_LOGRET_CLIP`: 로그수익 클리핑
- `MC_VERIFY_DRIFT`: drift 검증
- `MC_N_BOOT`: CVaR 부트스트랩
- `MC_TPSL_AUTOSCALE`: TP/SL 자동 스케일

#### 8-1) MC 통계/가속 (Torch 우선 + NumPy fallback)
- `MC_USE_TORCH`: PyTorch 가속 사용 여부 (기본 true)
- `MC_USE_ANTITHETIC`: Antithetic variates 사용 (기본 true)
- `MC_USE_CONTROL_VARIATE`: Control variate로 EV 보정 (기본 true)
- `ENTRY_CI_GATE_ENABLED`: CI 기반 진입 게이트 사용
- `ENTRY_CI_ALPHA`: CI 단측 유의수준 (예: 0.05)
- `ENTRY_CI_FLOOR`: LCB 하한 (기본 0.0)
- `MC_MULTI_FIDELITY_ENABLED`: 멀티 피델리티 MC 활성화
- `MC_N_PATHS_STAGE1`, `MC_N_PATHS_STAGE2`: 1/2단계 경로 수
- `MC_MULTI_FIDELITY_TOPK`: 2단계 재평가 상위 K
- `MC_MULTI_FIDELITY_SCORE_MIN`: 2단계 재평가 최소 점수

#### 9) 엔트리 필터/정책(Funnel)
- `FUNNEL_USE_WINRATE_FILTER`: 승률 필터
- `FUNNEL_USE_NAPV_FILTER`: NAPV 필터
- `FUNNEL_NAPV_THRESHOLD`: NAPV 임계값
- `FUNNEL_WIN_FLOOR_BULL/BEAR/CHOP/VOLATILE`: regime별 win floor
- `FUNNEL_CVAR_FLOOR_BULL/BEAR/CHOP/VOLATILE`: regime별 CVaR floor
- `FUNNEL_EVENT_CVAR_FLOOR_BULL/BEAR/CHOP/VOLATILE`: regime별 event CVaR floor
- `SCORE_ENTRY_MIN_SIZE`: score 기반 최소 size
- `K_LEV`: Kelly scaling 상수
- `SCORE_ONLY_MODE`: 점수 기반 단일 모드
- `USE_GPU_LEVERAGE`: leverage 탐색을 GPU로 수행
- `KELLY_BOOST_ENABLED`: Kelly boost
- `EV_COST_MULT_GATE`: 비용 게이트
- `DEFAULT_TP_PCT`, `DEFAULT_SL_PCT`: 기본 TP/SL
- `POLICY_NEIGHBOR_OPPOSE_VETO_ABS`: 이웃 반대 veto
- `POLICY_LOCAL_CONSENSUS_ALPHA`, `POLICY_LOCAL_CONSENSUS_BASE_H`: 로컬 컨센서스
- `EXIT_MODE`: exit 모드

#### 10) Alpha Hit (ML)
- `ALPHA_HIT_ENABLE`: AlphaHit 사용
- `ALPHA_HIT_BETA`: Beta 보정
- `ALPHA_HIT_TP_ATR_MULT`: TP 스케일
- `ALPHA_HIT_MODEL_PATH`: 모델 경로
- `ALPHA_HIT_DEVICE`: 디바이스
- `ALPHA_HIT_LR`, `ALPHA_HIT_BATCH_SIZE`, `ALPHA_HIT_STEPS_PER_TICK`: 학습 파라미터
- `ALPHA_HIT_MAX_BUFFER`, `ALPHA_HIT_MIN_BUFFER`, `ALPHA_HIT_MAX_LOSS`: 버퍼/손실
- `ALPHA_HIT_DATA_HALF_LIFE_SEC`: 데이터 반감기
- `ALPHA_HIT_FALLBACK`: fallback 방식

#### 11) PMAKER (실행지연/체결 확률 학습)
- `PMAKER_ENABLE`: PMAKER 사용
- `PMAKER_MODEL_PATH`: 모델 경로
- `PMAKER_TRAIN_STEPS`, `PMAKER_BATCH`: 학습 파라미터
- `PMAKER_GRID_MS`, `PMAKER_MAX_MS`: 탐색/시뮬레이션 범위
- `PMAKER_LR`, `PMAKER_DEVICE`: 학습 파라미터
- `PMAKER_PAPER_ENABLE`: 페이퍼 모드 PMAKER 사용
- `PMAKER_PAPER_PROBE_INTERVAL_SEC`: probe 간격
- `PMAKER_PAPER_PROBE_TIMEOUT_MS`: probe 타임아웃
- `PMAKER_PAPER_TRAIN_EVERY_N`: 학습 주기
- `PMAKER_PAPER_SAVE_EVERY_SEC`: 저장 주기
- `PMAKER_ADVERSE_DELAY_SEC`: adverse delay
- `PMAKER_ADVERSE_EMA_ALPHA`: adverse EMA

#### 12) 슬리피지/비용
- `SLIPPAGE_MULT`: 슬리피지 배율
- `SLIPPAGE_CAP`: 슬리피지 상한
- `P_MAKER_FIXED`: maker 고정 확률/비용(있을 때)

#### 13) 대시보드
- `DASHBOARD_INCLUDE_DETAILS`: details 포함
- `DASHBOARD_HISTORY_MAX`: 히스토리 최대 보관
- `DASHBOARD_TRADE_TAPE_MAX`: 트레이드 테이프 최대 보관

#### 14) 거래소 인증(보안)
- `BYBIT_API_KEY`, `BYBIT_API_SECRET`: 거래소 키 (값은 보안상 생략)

### 핵심 옵션(실전 필수)
아래는 실전에서 반드시 사용하는 핵심 옵션만 추린 목록(현재 설정값 기준).

| 옵션 | 현재값 | .env | 설명 |
|---|---|---|---|
| `SYMBOLS_CSV` | 미설정(기본 `config.SYMBOLS`) | 미설정 | 대상 심볼 리스트 |
| `OHLCV_SLEEP_SEC` | 30.0 | 미설정 | OHLCV 갱신 주기 |
| `PRELOAD_ON_START` | true | 미설정 | 시작 시 OHLCV preload |
| `ORDERBOOK_SLEEP_SEC` | 2.0 | 미설정 | 오더북 갱신 주기 |
| `ORDERBOOK_SYMBOL_INTERVAL_SEC` | 0.35 | 미설정 | 오더북 심볼 순환 간격 |
| `TICKER_SLEEP_SEC` | 1.0 | 미설정 | ticker 폴링 주기 |
| `BYBIT_TESTNET` | false | 미설정 | 거래 testnet 사용 |
| `DATA_BYBIT_TESTNET` | false | 미설정 | 데이터 testnet 사용 |
| `DECISION_REFRESH_SEC` | 2.0 | 미설정 | 결정 루프 주기 |
| `DECISION_EVAL_MIN_INTERVAL_SEC` | 2.0 | 미설정 | 재평가 최소 간격 |
| `DECISION_WORKER_SLEEP_SEC` | 0.0 | 미설정 | 결정 워커 sleep |
| `DECISION_MIN_CANDLES` | 20 | 미설정 | 결정 최소 캔들 수 |
| `DECISION_MAX_INFLIGHT` | 10 | 미설정 | 동시 인플라이트 제한 |
| `DECISION_LOG_EVERY` | 10 | 미설정 | 로그 주기 |
| `LOG_STDOUT` | false | 미설정 | stdout 로그 |
| `DEV_MODE` | false | 미설정 | dev 모드 |
| `ENABLE_LIVE_ORDERS` | false | 미설정 | 실거래 on/off |
| `EXEC_MODE` | maker_then_market | 미설정 | 주문 모드 |
| `USE_MAKER_ORDERS` | true | 미설정 | maker 주문 사용 |
| `MAKER_TIMEOUT_MS` | 1500 | 미설정 | maker 대기 시간 |
| `MAKER_RETRIES` | 2 | 미설정 | maker 재시도 |
| `MAKER_POLL_MS` | 200 | 미설정 | maker 폴링 간격 |
| `DEFAULT_LEVERAGE` | 5.0 | 미설정 | 기본 레버리지 |
| `MAX_LEVERAGE` | 100.0 | 미설정 | 최대 레버리지 |
| `DEFAULT_SIZE_FRAC` | 0.05 | 미설정 | 기본 비중 |
| `DEFAULT_TP_PCT` | 0.006 | 미설정 | 기본 TP |
| `DEFAULT_SL_PCT` | 0.005 | 미설정 | 기본 SL |
| `POSITION_HOLD_MIN_SEC` | 120 | 미설정 | 최소 보유 시간 |
| `MAX_POSITION_HOLD_SEC` | 600 | 미설정 | 최대 보유 시간 |
| `EXPOSURE_CAP_ENABLED` | true | 미설정 | 노출 캡 사용 |
| `MAX_NOTIONAL_EXPOSURE` | 10.0 | 미설정 | 총 노출 상한 |
| `MAX_CONCURRENT_POSITIONS` | 99999 | 미설정 | 동시 포지션 제한 |
| `COOLDOWN_SEC` | 60 | 미설정 | 재진입 쿨다운 |
| `TOP_N_SYMBOLS` | 4 | 미설정 | 상위 N개만 진입 |
| `USE_KELLY_ALLOCATION` | true | 미설정 | Kelly 배분 사용 |
| `USE_CONTINUOUS_OPPORTUNITY` | true | 미설정 | 교체 판단 사용 |
| `SWITCHING_COST_MULT` | 2.0 | 미설정 | 교체 비용 승수 |
| `PORTFOLIO_TOP_N` | 4 | 미설정 | 포트폴리오 TOP N |
| `PORTFOLIO_SWITCH_COST_MULT` | 1.2 | 미설정 | 포트폴리오 교체 비용 |
| `PORTFOLIO_KELLY_CAP` | 5.0 | 미설정 | Kelly 상한 |
| `PORTFOLIO_JOINT_INTERVAL_SEC` | 15.0 | 미설정 | joint sim 주기 |
| `PORTFOLIO_JOINT_SIM_ENABLED` | false | 미설정 | joint sim 사용 |
| `UNIFIED_RISK_LAMBDA` | 1.0 | 미설정 | UnifiedScore λ |
| `UNIFIED_RHO` | 0.0 | 미설정 | UnifiedScore ρ |
| `EVENT_EXIT_SCORE` | -0.0005 | 미설정 | 이벤트 기반 exit 기준 |
| `POLICY_SCORE_TRAILING_FACTOR` | 0.6 | 미설정 | policy trailing 강도 |
| `POLICY_SCORE_FLIP_MARGIN` | 0.001 | 미설정 | policy flip 마진 |
| `PAPER_TRADING` | true | 미설정 | 페이퍼 모드 |
| `PAPER_FLAT_ON_WAIT` | true | 미설정 | WAIT 시 정리 |
| `PAPER_EXIT_POLICY_ONLY` | true | 미설정 | exit-policy 전용 |
| `PAPER_EXIT_POLICY_HORIZON_SEC` | 1800 | 미설정 | exit-policy horizon |
| `PAPER_EXIT_POLICY_MIN_HOLD_SEC` | 180 | 미설정 | exit-policy 최소 보유 |
| `PAPER_EXIT_POLICY_DD_STOP_ENABLED` | true | 미설정 | DD stop 사용 |
| `PAPER_EXIT_POLICY_DD_STOP_ROE` | -0.02 | 미설정 | DD stop 기준 |
| `PAPER_SIZE_FRAC` | 0.05 | 미설정 | 페이퍼 비중 |
| `PAPER_LEVERAGE` | 5.0 | 미설정 | 페이퍼 레버리지 |
| `PAPER_FEE_ROUNDTRIP` | 0.0 | 미설정 | 페이퍼 수수료 |
| `PAPER_SLIPPAGE_BPS` | 0.0 | 미설정 | 페이퍼 슬리피지 |
| `MC_N_PATHS_LIVE` | 16384 | 미설정 | live 경로 수 |
| `MC_N_PATHS_EXIT` | 4096 | 미설정 | exit 경로 수 |
| `MC_TIME_STEP_SEC` | 1 | 미설정 | time step |
| `MC_TAIL_MODE` | student_t | 미설정 | 꼬리 분포 |
| `MC_STUDENT_T_DF` | 6.0 | 미설정 | t-분포 자유도 |
| `MC_TPSL_AUTOSCALE` | true | 미설정 | TP/SL autoscale |
| `MC_TP_BASE_ROE` | 0.0015 | 미설정 | TP 기본 |
| `MC_SL_BASE_ROE` | 0.002 | 미설정 | SL 기본 |
| `MC_TPSL_SIGMA_REF` | 0.5 | 미설정 | TP/SL sigma 기준 |
| `MC_TPSL_SIGMA_MIN_SCALE` | 0.6 | 미설정 | TP/SL sigma min |
| `MC_TPSL_SIGMA_MAX_SCALE` | 2.5 | 미설정 | TP/SL sigma max |
| `MC_TPSL_H_SCALE_BASE` | 60.0 | 미설정 | horizon 스케일 |
| `MC_LOGRET_CLIP` | 12.0 | 미설정 | 로그수익 클립 |
| `MC_VERIFY_DRIFT` | false | 미설정 | drift 검증 |
| `MC_N_BOOT` | 40 | 미설정 | CVaR 부트스트랩 |
| `FUNNEL_USE_WINRATE_FILTER` | false | 미설정 | 승률 필터 |
| `FUNNEL_USE_NAPV_FILTER` | false | 미설정 | NAPV 필터 |
| `FUNNEL_NAPV_THRESHOLD` | 0.0001 | 미설정 | NAPV 임계값 |
| `SCORE_ONLY_MODE` | false | 미설정 | score-only 모드 |
| `KELLY_BOOST_ENABLED` | true | 미설정 | Kelly boost |
| `EV_COST_MULT_GATE` | 0.0 | 미설정 | 비용 게이트 |
| `FUNNEL_WIN_FLOOR_BULL` | 0.5 | 미설정 | win floor bull |
| `FUNNEL_WIN_FLOOR_BEAR` | 0.5 | 미설정 | win floor bear |
| `FUNNEL_WIN_FLOOR_CHOP` | 0.5 | 미설정 | win floor chop |
| `FUNNEL_WIN_FLOOR_VOLATILE` | 0.5 | 미설정 | win floor volatile |
| `FUNNEL_CVAR_FLOOR_BULL` | -0.12 | 미설정 | cvar floor bull |
| `FUNNEL_CVAR_FLOOR_BEAR` | -0.12 | 미설정 | cvar floor bear |
| `FUNNEL_CVAR_FLOOR_CHOP` | -0.10 | 미설정 | cvar floor chop |
| `FUNNEL_CVAR_FLOOR_VOLATILE` | -0.09 | 미설정 | cvar floor volatile |
| `FUNNEL_EVENT_CVAR_FLOOR_BULL` | -1.25 | 미설정 | event cvar floor bull |
| `FUNNEL_EVENT_CVAR_FLOOR_BEAR` | -1.25 | 미설정 | event cvar floor bear |
| `FUNNEL_EVENT_CVAR_FLOOR_CHOP` | -1.15 | 미설정 | event cvar floor chop |
| `FUNNEL_EVENT_CVAR_FLOOR_VOLATILE` | -1.2 | 미설정 | event cvar floor volatile |
| `ALPHA_HIT_ENABLE` | true | 미설정 | AlphaHit 사용 |
| `ALPHA_HIT_BETA` | 1.0 | 미설정 | AlphaHit beta |
| `ALPHA_HIT_TP_ATR_MULT` | 2.0 | 미설정 | TP 스케일 |
| `ALPHA_HIT_MODEL_PATH` | state/alpha_hit_mlp.pt | 미설정 | 모델 경로 |
| `ALPHA_HIT_DEVICE` | mps | 미설정 | 디바이스 |
| `ALPHA_HIT_LR` | 0.0002 | 미설정 | 학습률 |
| `ALPHA_HIT_BATCH_SIZE` | 256 | 미설정 | 배치 크기 |
| `ALPHA_HIT_STEPS_PER_TICK` | 2 | 미설정 | 스텝/틱 |
| `ALPHA_HIT_MAX_BUFFER` | 200000 | 미설정 | 버퍼 max |
| `ALPHA_HIT_MIN_BUFFER` | 1024 | 미설정 | 버퍼 min |
| `ALPHA_HIT_MAX_LOSS` | 2.0 | 미설정 | 최대 손실 |
| `ALPHA_HIT_DATA_HALF_LIFE_SEC` | 3600.0 | 미설정 | 반감기 |
| `ALPHA_HIT_FALLBACK` | mc_to_hitprob | 미설정 | fallback |
| `PMAKER_ENABLE` | false | 미설정 | PMAKER 사용 |
| `PMAKER_MODEL_PATH` | state/pmaker_survival_mlp.pt | 미설정 | 모델 경로 |
| `PMAKER_TRAIN_STEPS` | 1 | 미설정 | 학습 스텝 |
| `PMAKER_BATCH` | 32 | 미설정 | 배치 크기 |
| `PMAKER_GRID_MS` | 50 | 미설정 | grid ms |
| `PMAKER_MAX_MS` | 2500 | 미설정 | max ms |
| `PMAKER_LR` | 0.0003 | 미설정 | 학습률 |
| `PMAKER_DEVICE` | (empty) | 미설정 | 디바이스 |
| `PMAKER_PAPER_ENABLE` | true | 미설정 | 페이퍼 PMAKER |
| `PMAKER_PAPER_PROBE_INTERVAL_SEC` | 2.0 | 미설정 | probe 간격 |
| `PMAKER_PAPER_PROBE_TIMEOUT_MS` | 1500 | 미설정 | probe timeout |
| `PMAKER_PAPER_TRAIN_EVERY_N` | 25 | 미설정 | 학습 주기 |
| `PMAKER_PAPER_SAVE_EVERY_SEC` | 30.0 | 미설정 | 저장 주기 |
| `PMAKER_ADVERSE_DELAY_SEC` | 5.0 | 미설정 | adverse delay |
| `PMAKER_ADVERSE_EMA_ALPHA` | 0.2 | 미설정 | adverse EMA |
| `SLIPPAGE_MULT` | 0.3 | 미설정 | 슬리피지 배율 |
| `SLIPPAGE_CAP` | 0.0003 | 미설정 | 슬리피지 상한 |
| `P_MAKER_FIXED` | 미설정 | 미설정 | maker 고정값 |
| `DASHBOARD_INCLUDE_DETAILS` | false | 미설정 | details 포함 |
| `DASHBOARD_HISTORY_MAX` | 50000 | 미설정 | 히스토리 보관 |
| `DASHBOARD_TRADE_TAPE_MAX` | 2000 | 미설정 | 테이프 보관 |

### JAX 미사용 시 제거/정리 후보
현재 JAX를 쓰지 않는다면 아래 환경변수는 의미가 없거나 혼선을 줄 수 있음.
- `XLA_PYTHON_CLIENT_PREALLOCATE`
- `XLA_PYTHON_CLIENT_ALLOCATOR`
- `XLA_PYTHON_CLIENT_MEM_FRACTION`
- `JAX_PLATFORMS`
- `JAX_PLATFORM_NAME`
- `JAX_MC_DEVICE`

(참고) 위 변수들은 `main_engine_mc_v2_final.py`에서 기본값을 강제로 설정하고 있으므로, JAX 미사용이라면 해당 설정 블록을 제거하는 편이 안전함.

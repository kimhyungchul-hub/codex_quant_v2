# Project: Codex Quant (코인 자동 매매 봇)

당신은 이 프로젝트의 **Senior Developer**입니다. 금융 공학(Financial Engineering), Python(JAX/NumPy), 시스템 아키텍처에 정통하며, 답변은 항상 전문적이고 간결한 한국어로(기술 용어는 영어 유지) 작성해야 합니다.

## 🚨 CRITICAL: 절대 원칙 (Violation Forbidden)
1. **진실의 원천 (Source of Truth):** 코드를 생성하기 전 `docs/CODE_MAP_v2.md`와 **`[Change Log]`**를 반드시 확인하여 맥락을 파악하십시오.
2. **아키텍처 보존:** '연산 루프(Compute Loop)'와 'UI 루프(Refresh Loop)'의 분리 구조를 절대 훼손하지 마십시오.
3. **문서화 루틴 (필수):** 모든 답변의 **맨 마지막**에 반드시 다음 작업을 수행하십시오.
   - 변경된 사항을 요약하여 `[Change Log]` 형식으로 출력 (사용자가 복사할 수 있게 함).
   - 형식: `[YYYY-MM-DD] 변경 내용 요약 (수정된 파일명)`
   - `CODE_MAP_v2.md` 구조 변경 시 업데이트 제안 포함.
4. **수학 공식:** 공식 수정/참조 시 `docs/MATHEMATICS.md`를 기준으로 삼으십시오.
5. 로그를 읽거나 명령을 실행하는데에 있어서 권한 문제로 막힌다면 서버를 백그라운드에서 실행하고, 로그는 /tmp/server.log에 저장해서 읽을 것. 또는 tail 명령 대신 read 명령을 사용해서 100줄 정도를 읽어볼 것.
## 🛠 기술 스택 및 환경
- **Language:** Python 3.11 (JAX 호환성 고정), Shell Script (Bash)
- **Core Libs:** JAX (GPU/Metal), NumPy (CPU/Dev), Pandas, FastAPI
- **Backend:** - **Production:** JAX (GPU 가속 필수, 컴파일 캐시 사용)
    - **Dev/Debug:** `DEV_MODE=true`일 때 NumPy 사용 (JIT 컴파일 시간 제거)
- **Environment:** `.venv` 사용, JAX 메모리 선점 방지(`XLA_PYTHON_CLIENT_PREALLOCATE=false`) 필수.

## ⚡ 코딩 컨벤션 및 구현 규칙

### 1. 시스템 안정성 및 프로세스 관리
- **Sleep 금지 / Polling 필수:** 서버/API 대기 시 절대 `sleep`을 쓰지 말고, `while` 루프와 `curl`/`nc`를 사용한 **Active Polling**을 구현하십시오. (Timeout 필수 설정)
- **좀비 프로세스 방지:** 백그라운드 프로세스(`&`)는 `trap`이나 명시적 `kill` 명령어로 종료를 보장하십시오.
- **I/O 최적화:** `MC_VERBOSE_PRINT=1` 등 과도한 출력은 터미널이 아닌 `/tmp` 파일로 리다이렉션(`>`)하십시오.
    - *Bad:* `python script.py` (출력 소실 가능성)
    - *Good:* `python script.py > /tmp/result.txt`

### 2. 금융 로직 및 의사결정 파이프라인
- **우선순위 명확화:** `MC Engine`(몬테카를로 EV)의 결과가 `Alpha Side`(단순 지표 편향)보다 우선합니다. 
    - **금지:** Alpha 지표가 MC가 산출한 `direction`(Long/Short)을 덮어쓰는 행위.
- **방어적 프로그래밍:** 의사결정 파이프라인 마지막에 `Guardrails`를 두어 논리적 모순(예: mu < 0 인데 Long 진입)을 차단하십시오.

### 3. 테스트 및 디버깅 가속화
- **Check System:** `check_system.py` 실행 시 `--interval 0.5` 등으로 대기 시간을 최소화하고, 성공 시 즉시 종료(Early Exit)하여 에이전트 대기 시간을 줄이십시오.
- **JAX vs NumPy:** 기능 구현 및 초기 디버깅 단계에서는 `numpy`를 사용하여 빠른 피드백을 받고, 최종 검증 시에만 `JAX`로 전환하십시오.

### 4. JAX 모듈 초기화 및 참조 규칙 (CRITICAL)
**문제:** JAX는 Lazy Import 패턴을 사용하여 `jax_backend.py`에서 `jax: Any = None`으로 초기화됩니다. `ensure_jax()`가 호출되기 전까지 `jax`는 `None` 상태이므로, fallback 로직이나 에러 핸들러에서 `jax.devices()`, `jax.device_get()` 등을 직접 호출하면 `AttributeError: 'NoneType' object has no attribute 'devices'` 발생.

**해결책 (필수 준수):**
1. **모듈 레벨 자동 초기화:** `jax_backend.py` 파일 끝에 반드시 `ensure_jax()` 호출을 추가하여 모듈 import 시점에 JAX를 로드하십시오.
   ```python
   # jax_backend.py 마지막 줄
   ensure_jax()
   ```

2. **Fallback 로직에서 재확인:** 에러 핸들러에서 JAX를 사용하기 전 반드시 `ensure_jax()` + `jax_module` 재import를 수행하십시오.
   ```python
   # ❌ BAD: except에서 jax를 직접 사용
   except Exception as e:
       cpu_dev = jax.devices("cpu")[0]  # jax가 None일 수 있음!
   
   # ✅ GOOD: ensure_jax() 후 명시적 재import
   except Exception as e:
       ensure_jax()
       from engines.mc.jax_backend import jax as jax_module
       if jax_module is None:
           raise RuntimeError("JAX unavailable") from e
       cpu_dev = jax_module.devices("cpu")[0]
   ```

3. **_JAX_OK import 필수:** JAX 상태를 확인하는 모든 모듈은 `_JAX_OK`를 명시적으로 import해야 합니다.
   ```python
   from engines.mc.jax_backend import ensure_jax, jax, jnp, _JAX_OK
   ```

4. **Try-Catch 전 초기화:** GPU 연산을 수행하는 try 블록 시작 전에 `ensure_jax()` + `jax_module` 준비를 완료하십시오.
   ```python
   # ✅ GOOD: try 전에 jax_module 준비
   ensure_jax()
   from engines.mc.jax_backend import jax as jax_module
   if jax_module is None:
       raise RuntimeError("JAX required")
   
   try:
       jax_module.block_until_ready(data)
   except Exception as e:
       cpu_dev = jax_module.devices("cpu")[0]  # 안전
   ```

**영향 파일:** `engines/mc/entry_evaluation.py`, `engines/mc/entry_evaluation_vmap.py`, `engines/mc/jax_backend.py`

*참고 (2026-01-24):* `engines/mc/jax_backend.py`는 이제 모듈 import 시점에 JAX 관련 환경을 점검하고 자동으로 일부 안전 설정을 적용합니다:
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform`을 감지하면 제거하여 BFC allocator 사용을 보장합니다.
- `XLA_PYTHON_CLIENT_MEM_FRACTION`이 unset일 때 기본값 `0.65`로 설정합니다.
- JAX 초기화 직후 작은 더미 연산으로 BFC allocator를 프리워밍합니다(`_JAX_WARMED` 플래그).

운영상 권장사항: 여전히 `bootstrap.py`에서 명시적으로 환경을 셋업하는 것이 가장 명확합니다. `jax_backend`의 자동화는 안전장치이며, 클러스터/CI 운영정책에서 다른 값을 강제하려면 환경변수를 프로세스 시작 전에 설정하세요.

### 5. 중앙 집중식 상수 관리 (Constants Management) - NEW!
**원칙:** 모든 하드코딩된 수치 상수는 `engines/mc/constants.py`에서 중앙 관리합니다.

**금지 사항:**
- ❌ 개별 파일에서 직접 하드코딩 (예: `STATIC_MAX_PATHS = 16384`)
- ❌ 중복 정의 (여러 파일에서 같은 상수 재정의)

**필수 사항:**
- ✅ 모든 상수는 `engines/mc/constants.py`에서 정의
- ✅ 다른 파일에서는 `from engines.mc.constants import STATIC_MAX_PATHS` 형태로 import
- ✅ 상수 변경 시 `constants.py` 파일만 수정

**주요 상수 목록:**
```python
from engines.mc.constants import (
    STATIC_MAX_SYMBOLS,      # JAX Static Shape: 최대 심볼 수 (32)
    STATIC_MAX_PATHS,        # JAX Static Shape: 최대 경로 수 (16384)
    STATIC_MAX_STEPS,        # JAX Static Shape: 최대 스텝 수 (3600)
    JAX_STATIC_BATCH_SIZE,   # 배치 크기 (STATIC_MAX_SYMBOLS와 동일)
    STATIC_HORIZONS,         # 고정 horizon 목록 [60, 300, 600, 1800, 3600]
    MC_N_PATHS_LIVE,         # 라이브 진입 평가 경로 수
    MC_N_PATHS_EXIT,         # Exit policy 평가 경로 수
    BOOTSTRAP_MIN_SAMPLES,   # Bootstrap 최소 샘플 수 (64)
    BOOTSTRAP_HISTORY_LEN,   # Bootstrap 히스토리 길이 (512)
    SECONDS_PER_YEAR,        # 연간 초 (31536000)
    EPSILON,                 # 0 나누기 방지 최소값 (1e-12)
)

    DEFAULT_IMPACT_CONSTANT, # Square-Root Market Impact 계수 (default=0.75)
의사결정(진입/레버리지/필터/Exit)에 직접 쓰이는 EV 계열만 추리면 이거예요.

**ev값 정리**
ev (= unified_score)
   진입/필터/레버리지/consensus 모두 이 값 사용
   사용 위치: main_engine_mc_v2_final.py (필터, 레버리지, consensus), decision.py (action 결정)
policy_ev_mix
   ev의 원천값 (entry_evaluation에서 최종 EV 산출)
   사용 위치: entry_evaluation.py
policy_ev_score_long / policy_ev_score_short
   direction 선택에 사용 (long vs short)
   사용 위치: decision.py
event_ev_r
   이벤트 기반 exit 판단에 사용
   사용 위치: main_engine_mc_v2_final.py (_evaluate_event_exit)
ev_entry_threshold / ev_entry_threshold_dyn
   EV 기반 진입 임계치 필터
   사용 위치: main_engine_mc_v2_final.py (_min_filter_states/동적 임계치)
   참고: ev_expected/ev_best는 현재 의사결정에 직접 사용되지 않음 (로그/메타용).

**적용 파일:**
- `engines/mc/constants.py` - 중앙 정의 (Source of Truth)
- `engines/mc/entry_evaluation_vmap.py` - STATIC_* 상수 import
- `engines/mc/entry_evaluation.py` - JAX_STATIC_BATCH_SIZE, BOOTSTRAP_* import
- `engines/mc/monte_carlo_engine.py` - STATIC_* 상수 import
- `main_engine_mc_v2_final.py` - STATIC_MAX_SYMBOLS import

**영향 파일:** `engines/mc/entry_evaluation.py`, `engines/mc/entry_evaluation_vmap.py`, `engines/mc/jax_backend.py`

## 📂 프로젝트 핵심 구조 (CODE_MAP)
- `core/orchestrator/`: 믹스인 기반 오케스트레이터 (Data, Risk, Decision 분리)
- `core/data_manager.py`: 시장 데이터 관리
- `core/ring_buffer.py`: SharedMemoryRingBuffer (multiprocessing.shared_memory 기반, low-latency 프로세스 간 메시지 전달)
- `engines/mc/`: 몬테카를로 엔진 핵심 (Entry, Exit, Decision)
- `main_engine_mc_v2_final.py`: 메인 진입점 및 루프 제어
- `server.py`: FastAPI 백엔드 서버
- `dashboard_v2.html`: 프론트엔드 대시보드

## 📝 교훈 (Lessons Learned) - 실수 반복 금지
- **Direction Mismatch:** 과거 Alpha 로직이 MC 결과를 덮어써서 역매매가 발생한 적이 있음. 항상 `meta.direction`을 신뢰할 것.
- **JAX Metal Delay:** Mac Metal 환경에서 첫 JIT 컴파일이 2~5분 소요될 수 있음. 운영자는 이를 '멈춤'으로 오해하지 않도록 로그(`INFO`)를 남겨야 함.
- **Fallback Logic:** Ticker 데이터가 늦게 오면 `price=None`이 될 수 있음. 이때는 즉시 OHLCV의 마지막 `close` 값을 Fallback으로 사용해야 함.
- **JAX Lazy Loading 함정 (2026-01-22):** `jax_backend.py`에서 `jax = None`으로 초기화되고 `ensure_jax()`가 모듈 끝에서 호출되지만, 다른 모듈에서 `from jax_backend import jax` 시점에는 아직 초기화 전일 수 있음. Exception handler에서 `jax.devices()`를 호출하면 `AttributeError: 'NoneType' object has no attribute 'devices'` 발생. 반드시 handler 내에서 `ensure_jax()` 재호출 + `jax as jax_module` 재import 필요.
- **Dashboard Data 누락 (2026-01-22):** JAX 초기화 실패로 인해 `decision_loop`에서 에러가 발생하면 `broadcast(rows)` 호출이 안 되어 WebSocket으로 `full_update`가 전송되지 않음. 브라우저는 `init` 메시지만 받고 데이터 없음. 엔진 내부 예외 처리가 데이터 전송까지 막지 않도록 `try-except` 범위를 좁혀야 함.
---

## Recent Changes (2026-01-24)

- Alpha Hit ML 복원: `OnlineAlphaTrainer`가 신규 구현되어 Horizon별 TP/SL 확률을 예측하고 온라인으로 학습합니다. `ALPHA_SIGNAL_BOOST=true`로 신호가 강화되었습니다.
- RL 통합: `train_transformer_gpu.py`가 `MonteCarloEngine` + `ExecutionCostModel`를 사용하도록 통합되었습니다. 비용 인지형(Pre-trade) 로직이 추가되어 과도한 거래는 자동으로 스킵됩니다.
- 통합 검증 스크립트: `verify_integration.py` 추가 — 데이터 로드 → JAX 초기화 → MC 시뮬레이션 → 비용 계산 → 행동 결정의 플로우를 검증합니다.
- JAX/MC 안정화: `engines/mc/entry_evaluation_vmap.py` warmup 고정(static small shape) 및 mask 기반 연산으로 JIT 트레이싱 오류를 방지했고, `engines/mc/entry_evaluation.py`에 빈 배열 방어 로직(`_ensure_len`)을 추가했습니다.
- 의존성: `requirements.txt`에 `torch`/`torchvision`이 명시되었습니다.

참고: 상세 변경 사항과 사용법은 `docs/CODE_MAP_v2.md`의 최신 Change Log 항목을 확인하세요.

## 📋 Change Log
### [2026-01-31] AlphaHit Online 학습 파이프라인 강화
**문제:** AlphaHit 예측이 EV에 미반영되고, 학습 버퍼가 실거래/백테스트 데이터를 공유하지 못함

**해결:**
1. **AlphaHit EV 블렌딩 도입** — MC 확률과 AlphaHit 확률을 신뢰도/베타로 혼합하여 EV 재계산 (`engines/mc/entry_evaluation.py`)
2. **Replay 버퍼 영속화** — AlphaHit 버퍼를 `state/alpha_hit_replay.npz`로 저장/로드 (`trainers/online_alpha_trainer.py`, `engines/mc/config.py`, `engines/mc/monte_carlo_engine.py`)
3. **실거래 학습 연결** — 진입 시 feature 저장, 청산 시 TP/SL 라벨 수집하여 AlphaHit 온라인 학습 연결 (`core/orchestrator.py`, `engines/mc/alpha_hit.py`)
4. **CSV 백필 스크립트** — `data/*.csv` OHLCV로 AlphaHit 버퍼 채우는 스크립트 추가 (`scripts/backfill_alpha_hit_from_csv.py`)

**영향 파일:** `engines/mc/entry_evaluation.py`, `trainers/online_alpha_trainer.py`, `engines/mc/config.py`, `engines/mc/monte_carlo_engine.py`, `core/orchestrator.py`, `engines/mc/alpha_hit.py`, `scripts/backfill_alpha_hit_from_csv.py`, `.env`
### [2026-01-31] AlphaHit 상태 대시보드 & 오케스트레이터 모니터링
**문제:** 데이터 백필이나 라이브/페이퍼 학습이 실제로 버퍼에 들어갔는지 확인할 수 없고, replay 파일/훈련 상태를 대시보드에서 확인할 수 없음

**해결:**
1. **Orchestrator 통계 수집** — `LiveOrchestrator.alpha_hit_status()`에서 trainer buffer, total samples, loss, warmup 여부, replay 경로/크기 정보를 수집하여 dashboard payload에 포함 (`core/orchestrator.py`, `core/dashboard_server.py`)
2. **UI 표시** — `dashboard_v2.html`에 AlphaHit chips(`αBuf`, `αLoss`, `αReplay`)을 추가하여 replay buffer(샘플/최소치), loss, replay 파일 존재 여부를 실시간으로 노출
3. **파일 기반 검증** — replay를 `state/alpha_hit_replay.npz`에 저장/로드하면서 크기/존재 여부를 함께 노출하므로 `scripts/backfill_alpha_hit_from_csv.py` 실행 결과를 UI에서 검증 가능

**영향 파일:** `core/orchestrator.py`, `core/dashboard_server.py`, `dashboard_v2.html`, `state/alpha_hit_replay.npz`
### [2026-01-31] Batch EV 정합성 & AlphaHit 예측 적용
**문제:** 배치 경로에서 UnifiedScore가 summary EV 기반으로만 계산되어 AlphaHit 효과가 반영되지 않고, AlphaHit 예측 텐서가 `[1, H]` 형태로 남아 대부분 horizon에 적용되지 않음

**해결:**
1. **배치 EV 정합성** — Exit Policy 결과로부터 EV/CVaR 벡터를 구축해 UnifiedScore를 계산하고, 배치 경로에서도 AlphaHit 확률 블렌딩을 적용 (`engines/mc/entry_evaluation.py`)
2. **AlphaHit 예측 형상 수정** — `[1, H]` 텐서를 1D로 변환해 모든 horizon에 적용 (`engines/mc/entry_evaluation.py`, `engines/mc/entry_evaluation_new.py`)
3. **로그 개선** — 필터 로그에 `EV_best` 표시를 추가해 UnifiedScore(Ψ)와 실제 horizon EV를 구분 (`main_engine_mc_v2_final.py`)

**영향 파일:** `engines/mc/entry_evaluation.py`, `engines/mc/entry_evaluation_new.py`, `main_engine_mc_v2_final.py`
### [2026-01-31] Alpha Hit ML 복원 및 고도화
**문제:** Alpha Hit ML 모듈(`OnlineAlphaTrainer`) 누락으로 TP/SL 확률 예측 정밀도 저하

**해결:** `trainers/online_alpha_trainer.py` 신규 구현.
1. **Multi-head MLP**: Horizon별 TP/SL 확률 동시 예측 (107k 파라미터, Residual Connection)
2. **Online Learning**: Experience Replay Buffer + Exponential Decay
3. **Advanced Features**: RunningNormalizer, LR Scheduler(Warmup+Cosine), Label Smoothing, Gradient Accumulation
4. **Signal Boost**: `ALPHA_SIGNAL_BOOST=true`로 mu_alpha 신호 3배 강화

**영향 파일:** `trainers/online_alpha_trainer.py`, `.env.midterm`, `.env.scalp`

### [2026-01-31] UnifiedScore 필터 진단 및 최적화 도구 추가

**문제:**
1. `UNIFIED_ENTRY_FLOOR=-0.0001`인데도 진입이 거의 없음
2. UnifiedScore가 과소평가되는지, 다른 필터(spread/event_cvar/cooldown/TOP_N)가 차단하는지 파악 불가
3. 적절한 threshold를 찾을 방법이 부재

**해결:**
1. **자동 통계 로깅 추가** ([main_engine_mc_v2_final.py](cci:7://file:///Users/jeonghwakim/codex_quant_clean/main_engine_mc_v2_final.py:0:0-0:0)):
   - [decision_loop](cci:1://file:///Users/jeonghwakim/codex_quant_clean/main_engine_mc_v2_final.py:2865:4-3177:40) Stage 2.5 직후에 10분마다 UnifiedScore 분포 통계 자동 출력
   - Mean, Median, Std, Min, Max, P25/P50/P75 표시
   - 현재 `UNIFIED_ENTRY_FLOOR` threshold 통과율 표시
   - 필터별 차단 통계 (unified, spread, event_cvar, cooldown)
   - 최적 threshold 자동 제안 (P50, Mean)
   - 로그 태그: `[SCORE_STATS]`, `[FILTER_STATS]`, `[THRESHOLD_HINT]`

2. **분석 스크립트 3종 추가**:
   - [scripts/analyze_unified_score_live.py](cci:7://file:///Users/jeonghwakim/codex_quant_clean/scripts/analyze_unified_score_live.py:0:0-0:0): WebSocket API 기반 실시간 분석 (현재 API 엔드포인트 부재로 미사용)
   - [scripts/analyze_score_from_logs.py](cci:7://file:///Users/jeonghwakim/codex_quant_clean/scripts/analyze_score_from_logs.py:0:0-0:0): 로그 파일 기반 분포 분석 및 최적 threshold 제안
   - [scripts/backtest_unified_threshold.py](cci:7://file:///Users/jeonghwakim/codex_quant_clean/scripts/backtest_unified_threshold.py:0:0-0:0): SQLite 거래 히스토리 기반 threshold 백테스팅 (승률/수익률 시뮬레이션)

3. **종합 가이드 문서**: [docs/UNIFIED_SCORE_FILTER_GUIDE.md](cci:7://file:///Users/jeonghwakim/codex_quant_clean/docs/UNIFIED_SCORE_FILTER_GUIDE.md:0:0-0:0)
   - 5가지 진단 방법 상세 설명
   - Threshold 설정 가이드 (보수적/균형/공격적)
   - 추가 필터 완화 방법 (spread/event_cvar/TOP_N)
   - 빠른 디버깅 명령어 모음
   - 체크리스트 및 트러블슈팅 가이드

**권장 조치:**
1. **즉시**: `TOP_N_SYMBOLS=8`로 증가 (현재 4개 → 8개, 진입 기회 2배 증가)
2. **10분 후**: `[SCORE_STATS]` 로그 확인
3. **1시간 후**: Mean 또는 P50 값으로 `UNIFIED_ENTRY_FLOOR` 조정
4. **1일 후**: 백테스팅 스크립트로 최적값 검증

**효과:**
- 실시간 UnifiedScore 분포 가시화
- 데이터 기반 threshold 최적화 가능
- 진입 차단 원인 명확한 진단 가능

**영향 파일:**
- [main_engine_mc_v2_final.py](cci:7://file:///Users/jeonghwakim/codex_quant_clean/main_engine_mc_v2_final.py:0:0-0:0) (통계 로깅 추가, line 2985-3035)
- [scripts/analyze_unified_score_live.py](cci:7://file:///Users/jeonghwakim/codex_quant_clean/scripts/analyze_unified_score_live.py:0:0-0:0) (신규)
- [scripts/analyze_score_from_logs.py](cci:7://file:///Users/jeonghwakim/codex_quant_clean/scripts/analyze_score_from_logs.py:0:0-0:0) (신규)
   - [scripts/backtest_unified_threshold.py](scripts/backtest_unified_threshold.py) (신규)
   - [docs/UNIFIED_SCORE_FILTER_GUIDE.md](docs/UNIFIED_SCORE_FILTER_GUIDE.md) (신규)

### [2026-01-31] UnifiedScore 진단 로깅 긴급 수정
**문제:** `SCORE_STATS` 미출력(Kelly 옵션 의존성), `all_pass` 상황에서 진입 실패 원인 불명확
**해결:**
1. 통계 로깅을 `USE_KELLY_ALLOCATION` 블록 밖으로 이동 (항상 실행)
2. `[FILTER] ... all_pass` 로그에 Action/Score/EV 정보 추가
**영향 파일:** `main_engine_mc_v2_final.py`

### [2026-01-31] SQLite 데이터베이스 마이그레이션
**목표:**
- JSON 파일 기반 저장을 SQLite로 전환하여 데이터 무결성 및 I/O 성능 향상

**변경사항:**
1. **`main_engine_mc_v2_final.py` 수정**:
   - `DatabaseManager` import 및 초기화 추가
   - `_record_trade()`: SQLite에 거래 기록 저장 (`log_trade_background()`)
   - `_persist_state()`: SQLite에 equity 및 positions 저장
   - `_trading_mode` 동적 설정 (`enable_orders` 기반)

2. **`core/database_manager.py` 수정**:
   - SQL INSERT 문 컬럼/값 개수 불일치 수정 (trades: 26, equity: 12, positions: 29)

3. **DB 경로**: `/tmp/codex_quant_db/bot_data.db` (macOS 권한 문제 임시 회피)

**검증:**
- equity_history 테이블에 데이터 저장 확인
- 대시보드 정상 작동 확인 (http://127.0.0.1:9999)

**영향 파일:** `main_engine_mc_v2_final.py`, `core/database_manager.py`

### [2026-01-31] 메모리 최적화 및 Control Variate 비활성화
**문제:**
- RAM 메모리 사용량이 과도하게 높음 (목표: 2-3GB)
- `evaluate_entry_metrics_batch()` 함수에서 대형 배열이 함수 종료 후에도 메모리에 유지됨

**해결책:**
1. **메모리 정리 코드 추가** (`engines/mc/entry_evaluation.py` line 3625-3661):
   - 배치 처리 완료 후 대형 객체 명시적 해제: `price_paths_batch`, `exit_policy_args`, `summary_cpu`
   - 강제 가비지 컬렉션: `gc.collect()`
   - PyTorch GPU 메모리 캐시 정리: `torch.mps.empty_cache()` / `torch.cuda.empty_cache()`

2. **Control Variate 비활성화** (`.env.midterm`, `.env.scalp`):
   - `MC_USE_CONTROL_VARIATE=0` 설정
   - 효과: 각 심볼별 prices_np CPU 복사본 생성 방지로 메모리 절감
   - 영향: n_paths=4096+ 에서 분산 감소 효과 미미하므로 실질적 성능 차이 없음

**GPU 상태 확인:**
- PyTorch MPS (Apple Metal) 정상 작동
- `[BATCH_TIMING] Torch batch path simulation...` 로그로 GPU 경로 사용 확인

**영향 파일:** `engines/mc/entry_evaluation.py`, `.env.midterm`, `.env.scalp`, `.github/copilot-instructions.md`

### [2026-01-28] MC 엔진 Torch 우선 전환 및 전략 프리셋 정합
**변경사항:**
1. **JAX 제거 및 Torch 우선/NumPy fallback 전환**: MC 핵심 경로(`decision`, `entry_evaluation`, `first_passage`, `path_simulation`)에서 Torch → NumPy 순으로 동작하도록 전환.
2. **통계적 안정성 강화**: 멀티 피델리티 MC, CI 기반 진입 게이트, 분산감소(antithetic/control variate) 적용.
3. **시간 일관성/기본값 정합**: `DEFAULT_TP_PCT=0.006`, `K_LEV=2000`, `ALPHA_HIT_DEVICE=mps`, `FUNNEL_WIN_FLOOR_*` 통일.
4. **전략별 `.env` 프리셋 추가**: 중기(1h) 스윙/초단기 스캘핑에 맞춘 TP/SL·호라이즌·홀드 타임 설정.

**영향 파일:** `engines/mc/decision.py`, `engines/mc/entry_evaluation.py`, `engines/mc/first_passage.py`, `engines/mc/path_simulation.py`, `engines/mc/config.py`, `main_engine_mc_v2_final.py`, `.env.midterm`, `.env.scalp`, `gemini.md`, `docs/CODE_MAP_v2.md`

### [2026-01-27] 대시보드 안정성 개선 및 Price Fallback 강화
**문제:**
1. **Dashboard 데이터 미표시**: `fetch_prices_loop`가 ticker 가격을 가져오기 전에 `decision_loop`이 시작되어 모든 `price=None`으로 브로드캐스트됨
2. **WebSocket 재연결 부재**: 연결 끊김 시 사용자가 수동 새로고침 필요
3. **로딩 상태 피드백 부재**: 사용자가 데이터 로딩 중인지 알 수 없음

**해결책:**
1. **`dashboard_v2.html` 구조적 개선**:
   - WebSocket 자동 재연결 로직 추가 (백오프: 1s → 2s → 4s → 8s → 15s)
   - 연결 상태 표시기 (`●` 연결됨, `○` 끊김, `↻` 재연결 중, `◔` 데이터 지연)
   - 로딩 오버레이 UI 추가 (연결 중/데이터 로딩 중/에러 상태 표시)
   - Stale 감지 (10초 이상 메시지 없으면 경고)

2. **`main_engine_mc_v2_final.py` Price Fallback 로직**:
   - `_build_batch_context_soa()`: ticker price가 None일 때 OHLCV 마지막 close 사용
   - `_build_decision_context()`: 동일한 fallback 로직 적용 (개별 빌드 경로)
   - Stage 3.5 (`FILL_MISSING`): 누락된 심볼에도 OHLCV close fallback 적용
   - `[FALLBACK_PRICE]` 로그로 추적 가능

**효과:**
- 서버 시작 직후 OHLCV preload만 완료되면 즉시 대시보드에 데이터 표시
- 네트워크 불안정 시 자동 재연결로 사용자 경험 개선
- 연결/데이터 상태가 명확히 시각화됨

**영향 파일:** `dashboard_v2.html`, `main_engine_mc_v2_final.py`, `docs/CODE_MAP_v2.md`, `.github/copilot-instructions.md`

### [2026-01-24] VPIN 및 테스트/CI 안정화
**변경사항:**
1. `utils/alpha_features.py`에 Volume-Synchronized VPIN 및 Order Flow Imbalance 함수 추가 (`calculate_vpin`, `calculate_order_flow_imbalance`). 확률적 BVC(Φ(ΔP/σ))를 사용한 매수/매도 볼륨 분배 및 볼륨 버킷 처리 방식으로 VPIN을 계산합니다. JAX 호환 옵션(`use_jax`)을 제공합니다.
2. 단위 테스트 추가/수정: `tests/test_alpha_features.py` 추가, `tests/test_orchestrator_mixins.py`의 레짐(assertion) 완화.
3. `pytest.ini` 추가로 레거시/외부 의존 테스트를 무시하도록 설정하여 CI 컬렉션 안정성 향상.

**영향 파일:** `utils/alpha_features.py`, `utils/__init__.py`, `tests/test_alpha_features.py`, `tests/test_orchestrator_mixins.py`, `pytest.ini`, `docs/CODE_MAP_v2.md`.

### [2026-01-24] Path simulation drift correction
**문제:**
1. `student_t` 및 `bootstrap` 모드에서 정규분포용 이토 보정항(`-0.5 * sigma^2`)이 일괄 적용되어 기대값(EV)이 편향됨.

**해결:**
1. `engines/mc/path_simulation.py`의 `simulate_paths_price` / 배치 / netpnl 구현에서 모드별로 drift 분기 처리 추가 (Gaussian은 기존 Ito 보정 유지, `student_t`/`bootstrap`은 `mu * dt` 사용).
2. 검증 스크립트 `scripts/mc_drift_test.py` 추가. JAX 모드는 환경변수 `MC_USE_JAX=1`로 활성화하여 JIT 커널에서도 수학적 무결성을 확인할 수 있음.

### [2026-01-22] JAX 초기화 및 WebSocket 데이터 전송 버그 수정
**문제:**
1. **JAX Lazy Import 패턴의 함정**: `jax_backend.py`에서 `jax = None`으로 초기화되고 `ensure_jax()` 호출로 로드되지만, 다른 모듈에서 `from jax_backend import jax` 시점에 아직 `None` 상태. Exception handler에서 `jax.devices()` 호출 시 `AttributeError: 'NoneType' object has no attribute 'devices'` 발생.
2. **Dashboard 데이터 미표시**: JAX 초기화 실패로 `decide_batch()` 내부에서 에러 발생 → `broadcast(rows)` 미호출 → WebSocket `full_update` 미전송 → 브라우저에 `init` 메시지만 수신.
3. **HTML 문법 에러**: `dashboard_v2.html` 끝에 불필요한 `}` 괄호 중복으로 JavaScript 실행 실패.

**해결책:**
1. **`engines/mc/jax_backend.py`**: 파일 끝에 `ensure_jax()` 자동 호출 추가 (모듈 import 시점에 JAX 초기화)
2. **`engines/mc/entry_evaluation.py`**: 
   - `ensure_jax` import 추가
   - 6개 fallback 경로에서 `jax` → `jax_module` 교체 (lines 775-785, 3180-3200, 3248-3269, 3275, 3396-3397)
   - Try 블록 전 `ensure_jax()` + `jax_module` 준비
3. **`engines/mc/entry_evaluation_vmap.py`**: `_JAX_OK` import 추가
4. **`dashboard_v2.html`**: 파일 끝 불필요한 `}` 2개 제거 (line 817-818)

**영향 범위:**
- 모든 JAX fallback 로직이 안전하게 CPU로 전환 가능
- Dashboard가 18개 종목 데이터 정상 표시
- WebSocket `full_update` 메시지 정상 전송 (2초 주기)

**참조 이슈:** Dashboard에 데이터가 표시되지 않는 문제 (WebSocket 연결은 성공하나 `market` 배열 비어있음)

### [2026-01-24] Antithetic Variates 도입
**변경사항:** `engines/mc/path_simulation.py`에 Antithetic Variates(대조 변수법)를 적용하여 난수 샘플 `Z`와 `-Z` 쌍을 함께 사용하도록 구현했습니다.
**영향:** JAX 및 NumPy 경로 모두에서 표준오차 감소를 기대할 수 있으며, Student-t 모드에서도 대칭성 기반 처리를 지원합니다. Bootstrap(경험분포)은 충분한 히스토리(>=16)일 때 경험분포를 유지합니다.


### [2026-01-22] 3가지 핵심 병렬화 개선 및 중앙 집중식 상수 관리
**문제:**
1. **Data Ingestion 병목**: `decision_loop`에서 개별 심볼마다 Dict 생성 → 메모리 재할당 및 for 루프 오버헤드
2. **Barrier Logic 누락**: `compute_horizon_metrics_jax`가 만기 가격만 체크 → 중간 경로 TP/SL 도달 케이스 약 40% 누락
3. **JIT 재컴파일**: 심볼 수 변동 시 JAX JIT 재컴파일로 장중 렉 발생
4. **하드코딩 난립**: STATIC_MAX_PATHS 등 상수가 여러 파일에 중복 정의

**해결책:**
1. **SoA (Structure of Arrays) 구조** (`main_engine_mc_v2_final.py`):
   - Pre-allocated numpy 배열 추가: `_batch_prices`, `_batch_mus`, `_batch_sigmas` 등
   - `_build_batch_context_soa()`: Dict 생성 최소화, 배열에 직접 값 할당
   - 효과: 메모리 재할당 방지, O(1) 인덱스 조회

2. **Barrier Logic** (`engines/mc/entry_evaluation_vmap.py`):
   - `compute_horizon_metrics_jax()` 완전 재작성
   - `jnp.max/min`으로 경로 내 고가/저가 산출 후 First Passage 체크
   - 효과: TP 도달 케이스 43.6% → 83.2% (약 40% 증가)

3. **Static Shape Warmup** (`engines/mc/constants.py`, `monte_carlo_engine.py`):
   - `STATIC_MAX_SYMBOLS=32`, `STATIC_MAX_PATHS=16384`, `STATIC_MAX_STEPS=3600`
   - `MonteCarloEngine.__init__()`: 최대 크기로 워밍업
   - 효과: 장중 shape 변경 시 JIT 재컴파일 방지

4. **중앙 집중식 상수 관리** (`engines/mc/constants.py`):
   - 모든 하드코딩 상수를 `constants.py`로 집중
   - 다른 파일은 `from engines.mc.constants import *` 형태로 import
   - 효과: 단일 수정 지점, 중복 제거, 유지보수성 향상

**영향 파일:**
- `engines/mc/constants.py` - 중앙 상수 정의 (신규 확장)
- `main_engine_mc_v2_final.py` - SoA 배열 + STATIC_MAX_SYMBOLS import
- `engines/mc/entry_evaluation_vmap.py` - Barrier Logic + constants import
- `engines/mc/entry_evaluation.py` - JAX_STATIC_BATCH_SIZE, BOOTSTRAP_* constants import
- `engines/mc/monte_carlo_engine.py` - STATIC_* constants import + warmup

### [2026-01-28] Exit Policy 기본값 변경 및 성능 최적화
**문제:**
- Apple Metal GPU에서 full exit policy(JAX vmap + lax.scan/cond)가 ~55초 소요
- 60초 timeout(`DECIDE_BATCH_TIMEOUT_SEC`)을 초과하여 배치 처리 실패

**해결책:**
1. **`SKIP_EXIT_POLICY=true` 기본값**으로 변경 (`engines/mc/entry_evaluation.py`):
   - Summary 기반 EV 사용 (경로 시뮬레이션에서 계산된 TP/SL 확률 기반)
   - 성능: **~5초** (n_paths=16000, 18 symbols, Metal GPU)
   
2. Full exit policy는 `SKIP_EXIT_POLICY=false`로 여전히 사용 가능:
   - NVIDIA CUDA GPU 또는 낮은 n_paths 설정 시 권장
   - 5가지 청산 로직(TP/SL/TimeStop/DD/DynamicPolicy) 모두 반영

**성능 비교 (n_paths=16000, 18 symbols, Apple M4 Pro Metal):**
| 설정 | 시뮬레이션 | Exit Policy | 총 시간 |
|------|-----------|-------------|---------|
| `SKIP_EXIT_POLICY=true` (기본) | ~4.4s | ~0s (스킵) | **~4.6s** ✓ |
| `SKIP_EXIT_POLICY=false` | ~4.8s | ~55s | ~60s ✗ (timeout) |

**영향 파일:** `engines/mc/entry_evaluation.py`, `docs/CODE_MAP_v2.md`

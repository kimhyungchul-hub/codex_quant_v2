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

## 📂 프로젝트 핵심 구조 (CODE_MAP)
- `core/orchestrator/`: 믹스인 기반 오케스트레이터 (Data, Risk, Decision 분리)
- `core/data_manager.py`: 시장 데이터 관리
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

## 📋 Change Log

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

# JAX Metal (Apple Silicon) 메모리 관리 가이드

> **작성일**: 2025-01-22  
> **적용 버전**: JAX 0.4.20 + jax-metal 0.0.5

---

## 🚨 핵심 요약 (TL;DR)

| 항목 | 권장 값 | 비고 |
|------|---------|------|
| JAX | `==0.4.20` | 0.4.22+ 메모리 설정 무시 |
| jax-metal | `==0.0.5` | 0.0.6+는 JAX 0.4.22+ 필요 |
| jaxlib | `==0.4.20` | JAX 버전과 일치 |
| NumPy | `>=1.22,<2.0` | JAX 0.4.x 호환성 |

**설치 명령어:**
```bash
pip install -r requirements-jax.txt
```

---

## 🔴 문제 상황 (2025-01-22 발생)

### 증상
1. **메모리 급증**: GPU 메모리가 8GB까지 선점되어 시스템 전체 불안정
2. **Dashboard 무응답**: asyncio 이벤트 루프가 GPU 연산에 의해 블로킹
3. **환경 변수 무시**: `XLA_PYTHON_CLIENT_PREALLOCATE=false` 설정이 적용되지 않음

### 로그 증거
```
metal_plugin  | maxCacheSize: 8.00 GB  ← 전체 GPU 메모리 선점!
```

### 근본 원인
**JAX 0.4.22 이상 버전에서 Metal 백엔드의 XLA 환경 변수 처리 방식이 변경됨**

- JAX 0.4.20: `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION` 정상 작동
- JAX 0.4.22+: 해당 환경 변수를 **무시**하고 Metal 드라이버 기본값 사용
- JAX 0.9.0: `UNIMPLEMENTED: default_memory_space is not supported` 에러 발생

---

## ✅ 해결책

### 1. JAX 버전 고정 (CRITICAL)

**`requirements-jax.txt`:**
```
jax==0.4.20
jaxlib==0.4.20
jax-metal==0.0.5
numpy>=1.22,<2.0
```

**버전 검증 명령어:**
```bash
python -c "import jax; print(f'JAX: {jax.__version__}')"
# 출력: JAX: 0.4.20
```

### 2. 환경 변수 설정 (`bootstrap.py`)

```python
import os

# JAX import 전에 반드시 설정
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
os.environ.setdefault("JAX_PLATFORMS", "metal,cpu")
```

**중요**: `bootstrap.py`는 `main_engine_mc_v2_final.py` 최상단에서 import해야 함:
```python
# main_engine_mc_v2_final.py 첫 줄
import bootstrap  # 환경변수 먼저!
```

### 3. asyncio 블로킹 방지 (`ThreadPoolExecutor`)

GPU 연산은 별도 스레드에서 실행하여 asyncio 이벤트 루프를 블로킹하지 않도록 함:

```python
from concurrent.futures import ThreadPoolExecutor

GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu_worker")

async def decision_loop():
    loop = asyncio.get_running_loop()
    
    # ❌ BAD: asyncio 블로킹
    # batch_decisions = self.hub.decide_batch(ctx_list)
    
    # ✅ GOOD: 별도 스레드에서 실행
    batch_decisions = await loop.run_in_executor(
        GPU_EXECUTOR, 
        self.hub.decide_batch, 
        ctx_list
    )
```

---

## 🧪 검증 방법

### 메모리 사용량 테스트
```bash
# 간단한 JAX 연산 후 메모리 확인
python -c "
import bootstrap
import jax.numpy as jnp
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
y.block_until_ready()
" &
sleep 3
ps aux | grep -E "^USER|python" | head -5
# 예상: 300~500MB
```

### Dashboard 응답 테스트
```bash
# 엔진 실행 중 Dashboard 응답 확인
curl -s -o /dev/null -w "HTTP: %{http_code}\n" http://localhost:9999/
# 예상: HTTP: 200
```

### Kelly 엔진 작동 확인
```bash
tail -f /tmp/engine_run.log | grep -E "PORTFOLIO|KELLY"
# 예상 출력:
# [PORTFOLIO] TOP 4: [('BTC/USDT:USDT', ...)]
# [KELLY] Allocations: [('BTC/USDT:USDT', '100.00%'), ...]
```

---

## 🏗️ 아키텍처 설계 원칙

### 루프 분리 (Compute Loop vs UI Loop)

```
┌─────────────────────────────────────────────────────────┐
│                    asyncio Event Loop                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │  WebSocket  │     │   HTTP      │     │  Refresh  │ │
│  │  Handler    │     │   Handler   │     │   Loop    │ │
│  └──────┬──────┘     └──────┬──────┘     └─────┬─────┘ │
│         │                   │                   │       │
│         └───────────────────┴───────────────────┘       │
│                             │                           │
│                    Non-blocking I/O                     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              ThreadPoolExecutor                  │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │           GPU Worker Thread              │    │   │
│  │  │                                          │    │   │
│  │  │  hub.decide_batch() → JAX/Metal GPU     │    │   │
│  │  │                                          │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│                    Blocking Compute                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**핵심 원칙:**
1. **UI/WebSocket은 절대 블로킹되면 안 됨** → asyncio 이벤트 루프에서 처리
2. **GPU 연산은 시간이 오래 걸림** → ThreadPoolExecutor에서 별도 스레드로 실행
3. **두 루프 간 통신은 `await`로** → `loop.run_in_executor()` 사용

---

## 📋 트러블슈팅 체크리스트

### 메모리가 8GB로 급증할 때
- [ ] JAX 버전 확인: `python -c "import jax; print(jax.__version__)"` → `0.4.20`이어야 함
- [ ] jax-metal 버전 확인: `pip show jax-metal` → `0.0.5`이어야 함
- [ ] `bootstrap.py`가 JAX import 전에 로드되는지 확인
- [ ] 환경 변수 확인: `echo $XLA_PYTHON_CLIENT_PREALLOCATE` → `false`

### Dashboard가 응답하지 않을 때
- [ ] `ThreadPoolExecutor` 사용 여부 확인
- [ ] `await loop.run_in_executor()` 패턴 사용 여부 확인
- [ ] GPU 연산이 메인 스레드에서 직접 실행되지 않는지 확인

### JAX 초기화 에러 (`AttributeError: 'NoneType'`)
- [ ] `ensure_jax()` 호출 후 `jax` 모듈 사용하는지 확인
- [ ] Exception handler에서 `jax_module` 재import 하는지 확인
- [ ] `copilot-instructions.md`의 "JAX 모듈 초기화 규칙" 섹션 참조

---

## 📚 참고 자료

- [JAX GitHub Issues - Metal Memory](https://github.com/google/jax/issues)
- [Apple Metal Best Practices](https://developer.apple.com/metal/)
- 프로젝트 내부: `docs/CODE_MAP_v2.md`, `.github/copilot-instructions.md`

---

## 📝 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2025-01-22 | 최초 작성: JAX 버전 고정, ThreadPoolExecutor 도입 |

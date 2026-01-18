# 런타임 설정 가이드 (Runtime Configuration Guide)

이 문서는 **codex_quant** 엔진의 모든 설정 옵션을 정리합니다.

---

## 목차
1. [트레이딩 모드 (Paper vs Live)](#1-트레이딩-모드-paper-vs-live)
2. [CLI 실행 옵션](#2-cli-실행-옵션)
3. [환경변수 설정](#3-환경변수-설정)
4. [대시보드 실시간 제어](#4-대시보드-실시간-제어)
5. [설정 파일 위치](#5-설정-파일-위치)
6. [일반적인 시나리오별 설정](#6-일반적인-시나리오별-설정)
7. [🔴 라이브 전환 체크리스트](#7--라이브-전환-체크리스트)

---

## 1. 트레이딩 모드 (Paper vs Live)

### 모드 개요

| 모드 | 설명 | 실제 주문 | 잔고 |
|------|------|----------|------|
| **Paper Trading** | 시뮬레이션 모드. 실제 거래소에 주문 안 함 | ❌ | 가상 (기본 $10,000) |
| **Live Trading** | 실제 Bybit에 주문 전송 | ✅ | 실제 계좌 잔고 |

### 핵심 제어 변수

```bash
# Paper 모드 (기본값)
ENABLE_LIVE_ORDERS=0
PAPER_TRADING=1

# Live 모드
ENABLE_LIVE_ORDERS=1
PAPER_TRADING=0
```

### 모드 전환 방법

#### 방법 1: 환경변수로 시작 시 지정
```bash
# Paper 모드 (기본)
python main.py

# Live 모드
ENABLE_LIVE_ORDERS=1 python main.py
```

#### 방법 2: CLI 플래그
```bash
# Paper 모드 강제
python main.py --paper

# Paper 모드 비활성화 (Live 활성화 필요)
python main.py --no-paper
```

#### 방법 3: 대시보드 실시간 전환
대시보드의 Settings 패널에서 `enable_orders` 토글로 전환 가능 (재시작 불필요)

---

## 2. CLI 실행 옵션

```bash
python main.py [OPTIONS]
```

### 기본 옵션

| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--imports-only` | flag | 모듈 임포트만 확인하고 종료 | - |
| `--symbols` | string | 거래 심볼 (쉼표 구분) | config.SYMBOLS |
| `--port` | int | 대시보드 포트 | 9999 |
| `--no-dashboard` | flag | 대시보드 비활성화 | - |
| `--no-preload` | flag | OHLCV 사전로드 스킵 | - |

### 트레이딩 모드 옵션

| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--paper` | flag | Paper 트레이딩 활성화 | True |
| `--no-paper` | flag | Paper 트레이딩 비활성화 | - |
| `--paper-size-frac` | float | Paper 포지션 크기 비율 (0~1) | 0.10 |
| `--paper-leverage` | float | Paper 레버리지 | 5.0 |

### MC 엔진 옵션

| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--mc-n-paths-live` | int | MC 시뮬레이션 경로 수 | 16384 |
| `--mc-n-paths-exit` | int | Exit policy 경로 수 | 4096 |
| `--mc-tail-mode` | string | 꼬리 분포 모드 (`gaussian`, `student_t`, `bootstrap`) | `student_t` |
| `--mc-student-t-df` | float | Student-t 자유도 (≥2.1) | 6.0 |
| `--mc-use-jax` | flag | JAX 가속 활성화 | True |
| `--mc-no-jax` | flag | JAX 비활성화 | - |

### 실행 모드 옵션

| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--exec-mode` | string | 실행 모드 (`market`, `maker_then_market`) | `maker_then_market` |
| `--decision-refresh-sec` | float | Decision 루프 주기 (초) | 2.0 |

---

## 3. 환경변수 설정

### 거래소 연결

| 환경변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `BYBIT_API_KEY` | string | Bybit API 키 | - |
| `BYBIT_API_SECRET` | string | Bybit API 시크릿 | - |
| `BYBIT_TESTNET` | bool | 테스트넷 사용 (1/0) | 0 |
| `DATA_BYBIT_TESTNET` | bool | 데이터용 별도 테스트넷 | - |

### Paper Trading 설정

| 환경변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `PAPER_TRADING` | bool | Paper 모드 활성화 | 1 |
| `PAPER_SIZE_FRAC` | float | 기본 포지션 크기 비율 | 0.10 |
| `PAPER_LEVERAGE` | float | 기본 레버리지 | 5.0 |
| `PAPER_USE_ENGINE_SIZING` | bool | 엔진 최적 사이징 사용 | 1 |
| `PAPER_ENGINE_SIZE_MULT` | float | 엔진 사이즈 승수 | 1.0 |
| `PAPER_ENGINE_SIZE_MIN_FRAC` | float | 최소 사이즈 비율 | 0.005 |
| `PAPER_ENGINE_SIZE_MAX_FRAC` | float | 최대 사이즈 비율 | 0.20 |
| `PAPER_FEE_ROUNDTRIP` | float | 왕복 수수료 시뮬레이션 | 0.0 |
| `PAPER_SLIPPAGE_BPS` | float | 슬리피지 (bp) | 0.0 |
| `PAPER_MIN_HOLD_SEC` | int | 최소 보유 시간 (초) | 120 |
| `PAPER_MAX_HOLD_SEC` | int | 최대 보유 시간 (초) | 600 |
| `PAPER_MAX_POSITIONS` | int | 최대 동시 포지션 수 | 99999 |
| `PAPER_EXIT_POLICY_ONLY` | bool | Exit policy만 사용 | 1 |
| `PAPER_FLAT_ON_WAIT` | bool | WAIT 시 청산 | 1 |

### Exit Policy 설정

| 환경변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `PAPER_EXIT_POLICY_HORIZON_SEC` | int | 기본 호라이즌 (초) | 1800 |
| `PAPER_EXIT_POLICY_MIN_HOLD_SEC` | int | 최소 보유 시간 | 180 |
| `PAPER_EXIT_POLICY_DECISION_DT_SEC` | int | 결정 간격 | 5 |
| `PAPER_EXIT_POLICY_DD_STOP_ENABLED` | bool | DD Stop 활성화 | 1 |
| `PAPER_EXIT_POLICY_DD_STOP_ROE` | float | DD Stop ROE | -0.02 |
| `PAPER_EXIT_POLICY_P_POS_ENTER_FLOOR` | float | 진입 p_pos 하한 | 0.52 |
| `PAPER_EXIT_POLICY_P_POS_HOLD_FLOOR` | float | 보유 p_pos 하한 | 0.50 |

### MC 엔진 설정

| 환경변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `MC_N_PATHS_LIVE` | int | MC 경로 수 | 16384 |
| `MC_N_PATHS_EXIT` | int | Exit policy 경로 수 | 4096 |
| `MC_USE_JAX` | bool | JAX 사용 | 1 |
| `MC_TAIL_MODE` | string | 꼬리 분포 | `student_t` |
| `MC_STUDENT_T_DF` | float | Student-t 자유도 | 6.0 |

### 리스크/실행 설정

| 환경변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `ENABLE_LIVE_ORDERS` | bool | 실제 주문 활성화 | 0 |
| `DEFAULT_LEVERAGE` | float | 기본 레버리지 | 5.0 |
| `MAX_LEVERAGE` | float | 최대 레버리지 | 100.0 |
| `DEFAULT_SIZE_FRAC` | float | 기본 사이즈 비율 | 0.05 |
| `MAX_POSITION_HOLD_SEC` | int | 최대 보유 시간 | 600 |
| `POSITION_HOLD_MIN_SEC` | int | 최소 보유 시간 | 120 |
| `MAX_CONCURRENT_POSITIONS` | int | 최대 동시 포지션 | 99999 |
| `MAX_NOTIONAL_EXPOSURE` | float | 최대 노출 배수 | 10.0 |
| `EXEC_MODE` | string | 실행 모드 | `maker_then_market` |
| `MAKER_TIMEOUT_MS` | int | 메이커 타임아웃 (ms) | 1500 |
| `MAKER_RETRIES` | int | 메이커 재시도 횟수 | 2 |

### 대시보드/디버그 설정

| 환경변수 | 타입 | 설명 | 기본값 |
|----------|------|------|--------|
| `DECISION_REFRESH_SEC` | float | Decision 루프 주기 | 2.0 |
| `DASHBOARD_INCLUDE_DETAILS` | bool | 상세 정보 포함 | 0 |
| `LOG_STDOUT` | bool | stdout 로깅 | 0 |
| `DEBUG_MU_SIGMA` | bool | mu/sigma 디버그 | 0 |
| `DEBUG_ROW` | bool | row 디버그 | 0 |

---

## 4. 대시보드 실시간 제어

### API 엔드포인트

**GET `/api/runtime`**  
현재 런타임 설정 조회

**POST `/api/runtime`**  
런타임 설정 변경 (JSON body)

### 변경 가능한 설정 (재시작 없이)

```javascript
// 예시: fetch로 설정 변경
fetch('/api/runtime', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    enable_orders: false,              // Live/Paper 전환
    paper_trading_enabled: true,       // Paper 모드
    paper_size_frac_default: 0.05,     // 포지션 크기
    paper_leverage_default: 3.0,       // 레버리지
    paper_use_engine_sizing: true,     // 엔진 사이징 사용
    paper_exit_policy_only: true,      // Exit policy만 사용
    paper_flat_on_wait: true,          // WAIT 시 청산
    mc_n_paths_live: 8192,             // MC 경로 수
    mc_use_jax: true,                  // JAX 사용
  })
});
```

### 대시보드 UI에서 제어 가능한 항목

| 설정 | UI 컨트롤 | 설명 |
|------|----------|------|
| `enable_orders` | 토글 | Live/Paper 모드 전환 |
| `paper_trading_enabled` | 토글 | Paper 트레이딩 활성화 |
| `paper_size_frac_default` | 슬라이더 | 포지션 크기 |
| `paper_leverage_default` | 슬라이더 | 레버리지 |
| `paper_use_engine_sizing` | 토글 | 엔진 최적 사이징 |
| `paper_exit_policy_only` | 토글 | Exit policy만 사용 |
| `paper_flat_on_wait` | 토글 | WAIT 시 청산 |
| `mc_n_paths_live` | 입력 | MC 경로 수 |

---

## 5. 설정 파일 위치

### 우선순위 (높음 → 낮음)
1. **CLI 인자** (`python main.py --paper-leverage 10`)
2. **환경변수** (`PAPER_LEVERAGE=10 python main.py`)
3. **`state/bybit.env`** — 런타임 오버라이드 (`.env` 덮어씀)
4. **`.env`** — 개발자 기본 설정
5. **`config.py`** — 코드 기본값

### 파일 설명

| 파일 | 용도 | Git 추적 |
|------|------|----------|
| `.env` | 개발자 로컬 설정 | ❌ (.gitignore) |
| `state/bybit.env` | 런타임 오버라이드 | ❌ (.gitignore) |
| `state/bybit.env.example` | 템플릿 예시 | ✅ |
| `config.py` | 코드 기본값 | ✅ |

### `.env` 예시
```bash
# .env
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=0
ENABLE_LIVE_ORDERS=0
PAPER_TRADING=1
```

### `state/bybit.env` 예시
```bash
# state/bybit.env - .env보다 우선
BYBIT_TESTNET=1
ENABLE_LIVE_ORDERS=0
```

---

## 6. 일반적인 시나리오별 설정

### 시나리오 1: Mainnet Paper Trading (권장 개발 모드)
```bash
# .env 또는 환경변수
BYBIT_TESTNET=0          # 메인넷 데이터
ENABLE_LIVE_ORDERS=0     # 주문 비활성화
PAPER_TRADING=1          # Paper 모드
```
```bash
python main.py --paper
```

### 시나리오 2: Testnet Paper Trading
```bash
BYBIT_TESTNET=1
ENABLE_LIVE_ORDERS=0
PAPER_TRADING=1
```
```bash
python main.py --paper
```

### 시나리오 3: Mainnet Live Trading (실거래)
```bash
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_TESTNET=0
ENABLE_LIVE_ORDERS=1
PAPER_TRADING=0
```
```bash
python main.py --no-paper
```

### 시나리오 4: 소수 심볼만 테스트
```bash
python main.py --symbols "BTC/USDT:USDT,ETH/USDT:USDT" --paper
```

### 시나리오 5: MC 경로 수 줄여서 빠른 테스트
```bash
python main.py --mc-n-paths-live 2000 --mc-n-paths-exit 500 --paper
```

### 시나리오 6: JAX 없이 CPU 전용
```bash
python main.py --mc-no-jax --paper
```

---

## 빠른 참조 명령어

```bash
# 기본 시작 (Paper, Mainnet 데이터)
python main.py

# Live 모드 (실거래 - 주의!)
ENABLE_LIVE_ORDERS=1 python main.py --no-paper

# 대시보드 없이 시작
python main.py --no-dashboard

# 특정 포트로 시작
python main.py --port 8080

# 모듈 임포트만 확인
python main.py --imports-only

# 현재 설정 확인 (대시보드 실행 중)
curl http://localhost:9999/api/runtime
```

---

## 7. 🔴 라이브 전환 체크리스트

> ⚠️ **중요**: 실거래 전환 전 반드시 아래 항목들을 점검하세요.

### 7.1 잠재적 문제점 및 대응 방안

#### 🎯 문제 1: 주문 타입 (지정가 vs 시장가)

| 실행 모드 | 설명 | 장점 | 단점 |
|----------|------|------|------|
| `market` | 시장가 즉시 체결 | 확실한 체결 | 슬리피지 발생 |
| `maker_then_market` | 지정가 시도 → 타임아웃 시 시장가 | 수수료 절감 | 체결 지연 가능 |

**조절 옵션:**
```bash
# 시장가 전용 (빠른 체결 우선)
EXEC_MODE=market python main.py

# 메이커 시도 후 시장가 폴백 (기본값)
EXEC_MODE=maker_then_market python main.py

# 메이커 타임아웃 조절 (ms)
MAKER_TIMEOUT_MS=2000        # 기본 1500ms
MAKER_RETRIES=3              # 기본 2회
MAKER_POLL_MS=300            # 기본 200ms
```

**대시보드 실시간 전환:**
```bash
curl -X POST http://localhost:9999/api/runtime \
  -H "Content-Type: application/json" \
  -d '{"exec_mode": "market"}'
```

---

#### 🎯 문제 2: 슬리피지 처리

**현재 시스템 동작:**
- `_estimate_slippage()` 함수가 동적으로 슬리피지 추정
- 변동성(σ), 유동성(liq_score), 레버리지, OFI 기반 계산
- Paper 모드에서는 `PAPER_SLIPPAGE_BPS` 적용

**슬리피지 모델 수식:**
```
slippage = base × vol_term × liq_term × lev_term × adv_k
         × SLIPPAGE_MULT (기본 0.3)
         min(result, SLIPPAGE_CAP)  # 기본 0.0003 (3bp)
```

**조절 옵션:**
```bash
# 슬리피지 승수 (모델 출력에 곱함)
SLIPPAGE_MULT=0.5            # 기본 0.3

# 슬리피지 상한 (비율)
SLIPPAGE_CAP=0.0005          # 기본 0.0003 (3bp)

# Paper 모드 슬리피지 시뮬레이션 (bp)
PAPER_SLIPPAGE_BPS=2.0       # 기본 0.0
```

---

#### 🎯 문제 3: 체결 실패 / 부분 체결

**발생 가능 시나리오:**
1. `PostOnly` 주문이 즉시 체결 가능해서 거부됨
2. 유동성 부족으로 부분 체결
3. 가격 급변으로 체결 실패

**현재 시스템 대응:**
- `maker_then_market` 모드: 메이커 실패 시 자동으로 시장가 전환
- `MAKER_RETRIES` 횟수만큼 재시도
- 마진 부족 오류 시 `RiskManager.on_insufficient_margin_error()` 호출 → 쿨다운 + 주문 취소

**권장 설정:**
```bash
# 급변장에서 안전한 설정
EXEC_MODE=market             # 확실한 체결 우선
MAKER_TIMEOUT_MS=1000        # 빠른 폴백
```

---

#### 🎯 문제 4: 잔고/마진 관리

**RiskManager 안전장치:**

| 안전장치 | 설명 | 기본값 |
|----------|------|--------|
| `max_total_leverage` | 총 레버리지 상한 (notional/equity) | 10.0x |
| `margin_ratio_guard` | 마진 비율 경고 임계값 | 0.80 |
| `min_notional_usd` | 최소 주문 금액 (더스트 필터) | $6 |
| `insufficient_margin_cooldown_ms` | 마진 부족 후 쿨다운 | 30초 |

**조절 옵션:**
```bash
# 총 노출 제한
MAX_NOTIONAL_EXPOSURE=5.0    # 기본 10.0x

# 최대 동시 포지션
MAX_CONCURRENT_POSITIONS=5   # 기본 99999

# 레버리지 제한
DEFAULT_LEVERAGE=3.0         # 기본 5.0
MAX_LEVERAGE=20.0            # 기본 100.0
```

---

#### 🎯 문제 5: 포지션 청산 (Exit)

**정상 청산 경로:**
1. Exit Policy에 의한 자동 청산 (`paper_exit_policy_only=True`)
2. DD Stop (ROE < -2% 시 강제 청산)
3. 시간 제한 (`MAX_POSITION_HOLD_SEC`)

**비상 청산 도구:**

| 도구 | 용도 | 사용법 |
|------|------|--------|
| 대시보드 버튼 | Paper 전체 청산 | `Liquidate All` 버튼 클릭 |
| API 호출 | Paper 전체 청산 | `POST /api/liquidate_all` |
| 스크립트 | 실계좌 시장가 청산 | `python scripts/close_positions_market.py` |
| 스크립트 | 실계좌 공격적 청산 | `python scripts/close_positions_aggressive.py` |

**비상 청산 스크립트 사용:**
```bash
# 모든 포지션 즉시 시장가 청산
cd /path/to/codex_quant
python scripts/close_positions_market.py

# 가격 계속 올려가며 청산 (유동성 낮을 때)
python scripts/close_positions_aggressive.py
```

**Exit Policy 조절:**
```bash
# DD Stop (손절) 설정
PAPER_EXIT_POLICY_DD_STOP_ENABLED=1
PAPER_EXIT_POLICY_DD_STOP_ROE=-0.02   # ROE -2%에서 손절

# 최소/최대 보유 시간
PAPER_MIN_HOLD_SEC=120               # 최소 2분
PAPER_MAX_HOLD_SEC=600               # 최대 10분
```

---

#### 🎯 문제 6: 레버리지 동기화

**문제:**
- 엔진이 계산한 최적 레버리지와 거래소 설정 불일치

**현재 시스템 동작:**
- `RiskManager.sync_api_leverage()` → 거래소 레버리지 자동 동기화
- 심볼별 최대 레버리지 제한 적용
- 15초 간격 쓰로틀링 (`leverage_sync_min_interval_ms`)

**조절 옵션:**
```bash
# 시스템 최대 레버리지
MAX_LEVERAGE=50.0

# Paper 기본 레버리지
PAPER_LEVERAGE=3.0
```

---

### 7.2 라이브 전환 전 체크리스트

#### ✅ 사전 점검
- [ ] **Testnet에서 충분히 테스트했는가?**
  ```bash
  BYBIT_TESTNET=1 ENABLE_LIVE_ORDERS=1 python main.py
  ```
- [ ] **API 키 권한 확인** — 선물 거래, 주문 생성/취소 권한 필요
- [ ] **계좌 잔고 확인** — 최소 $100 이상 권장
- [ ] **네트워크 안정성** — 낮은 레이턴시 서버 권장

#### ✅ 안전장치 설정
- [ ] **레버리지 제한 설정**
  ```bash
  MAX_LEVERAGE=20.0
  DEFAULT_LEVERAGE=3.0
  ```
- [ ] **노출 제한 설정**
  ```bash
  MAX_NOTIONAL_EXPOSURE=5.0
  MAX_CONCURRENT_POSITIONS=3
  ```
- [ ] **손절 활성화**
  ```bash
  PAPER_EXIT_POLICY_DD_STOP_ENABLED=1
  PAPER_EXIT_POLICY_DD_STOP_ROE=-0.02
  ```

#### ✅ 모니터링 준비
- [ ] **대시보드 접속** — `http://localhost:9999`
- [ ] **로그 모니터링** — `tail -f logs/*.log`
- [ ] **비상 청산 스크립트 준비**
  ```bash
  # 터미널에 미리 입력해두기
  python scripts/close_positions_market.py
  ```

#### ✅ 점진적 전환
```bash
# 1단계: 소수 심볼로 테스트
python main.py --symbols "BTC/USDT:USDT" --no-paper

# 2단계: 심볼 추가
python main.py --symbols "BTC/USDT:USDT,ETH/USDT:USDT" --no-paper

# 3단계: 전체 심볼
python main.py --no-paper
```

---

### 7.3 라이브 운영 중 조절 가능 옵션 요약

| 설정 | 대시보드 | 재시작 필요 | 설명 |
|------|:--------:|:-----------:|------|
| `enable_orders` | ✅ | ❌ | Live/Paper 전환 |
| `exec_mode` | ✅ | ❌ | 주문 실행 모드 |
| `paper_leverage_default` | ✅ | ❌ | Paper 레버리지 |
| `paper_size_frac_default` | ✅ | ❌ | Paper 포지션 크기 |
| `paper_exit_policy_only` | ✅ | ❌ | Exit policy 사용 |
| `mc_n_paths_live` | ✅ | ❌ | MC 시뮬레이션 경로 수 |
| `BYBIT_TESTNET` | ❌ | ✅ | 테스트넷 사용 |
| `SYMBOLS` | ❌ | ✅ | 거래 심볼 목록 |

---

### 7.4 비상 상황 대응

#### 🚨 시나리오 1: 포지션 청산 안 됨
```bash
# 1. 대시보드 청산 버튼 클릭 (Paper만)
# 2. 또는 스크립트 실행
python scripts/close_positions_market.py

# 3. 그래도 안 되면 공격적 청산
python scripts/close_positions_aggressive.py
```

#### 🚨 시나리오 2: 마진 부족 오류 반복
```bash
# 레버리지 낮추기
curl -X POST http://localhost:9999/api/runtime \
  -d '{"paper_leverage_default": 2.0}'

# 포지션 크기 줄이기  
curl -X POST http://localhost:9999/api/runtime \
  -d '{"paper_size_frac_default": 0.02}'
```

#### 🚨 시나리오 3: 엔진 응답 없음
```bash
# 프로세스 강제 종료
kill -9 $(cat engine.pid)

# 거래소에서 직접 포지션 정리
# → Bybit 웹/앱에서 수동 청산
```

---

## 변경 로그

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-18 | 초기 문서 작성 |
| 2026-01-18 | 라이브 전환 체크리스트 섹션 추가 |

---

*작성 일자: 2026-01-18*

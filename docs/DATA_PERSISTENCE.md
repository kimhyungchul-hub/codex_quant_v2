# 데이터 저장 설계서 (Data Persistence Design)

본 문서는 codex_quant의 데이터 저장 아키텍처를 정의합니다.

---

## 목차
1. [데이터 소스 분류](#1-데이터-소스-분류)
2. [Paper vs Live 차이점](#2-paper-vs-live-차이점)
3. [SQLite 스키마](#3-sqlite-스키마)
4. [대시보드 데이터 흐름](#4-대시보드-데이터-흐름)
5. [마이그레이션 가이드](#5-마이그레이션-가이드)

---

## 1. 데이터 소스 분류

### 1.1 거래소 WebSocket에서 실시간으로 받을 수 있는 데이터

| 데이터 종류 | Bybit WS 채널 | 갱신 주기 | 설명 |
|------------|---------------|----------|------|
| **현재가 (Ticker)** | `tickers.{symbol}` | ~100ms | last price, 24h volume, bid1/ask1 |
| **오더북 (Orderbook)** | `orderbook.50.{symbol}` | ~100ms | 50레벨 depth, bid/ask |
| **캔들 (Kline)** | `kline.{interval}.{symbol}` | 간격별 | OHLCV |
| **체결 (Trade)** | `publicTrade.{symbol}` | 실시간 | 공개 체결 내역 |
| **포지션 (Position)** | `position` (private) | 변경 시 | 내 포지션 상태 |
| **주문 (Order)** | `order` (private) | 변경 시 | 내 주문 상태 |
| **지갑 (Wallet)** | `wallet` (private) | 변경 시 | 잔고, 마진 |
| **체결 (Execution)** | `execution` (private) | 체결 시 | 내 체결 내역 |

### 1.2 영속화가 필요한 데이터 (로컬 저장)

| 데이터 종류 | 이유 | 테이블 |
|------------|------|--------|
| **거래 이력 (Trade Tape)** | 감사/분석/백테스트 | `trades` |
| **자산 시계열 (Equity)** | 성과 추적/그래프 | `equity_history` |
| **포지션 메타데이터** | 재시작 복구, 정책 상태 | `positions` |
| **진입 컨텍스트** | 전략 분석 | `trades.entry_*` |
| **Exit Policy 상태** | 재시작 시 정책 연속성 | `positions.policy_*` |
| **슬리피지 분석** | 실행 품질 모니터링 | `slippage_analysis` |
| **EVPH 히스토리** | 전략 성과 추적 | `evph_history` |
| **진단 메트릭** | 디버깅/모니터링 | `diagnostics` |

### 1.3 데이터 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                        거래소 (Bybit)                                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │ Ticker  │ │Orderbook│ │ Kline   │ │Position │ │Execution│        │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │
└───────┼──────────┼──────────┼──────────┼──────────┼──────────────────┘
        │          │          │          │          │
        ▼          ▼          ▼          ▼          ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      WebSocket 수신 계층                               │
│                    (DataManager / Exchange Client)                    │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ 실시간 표시용 │ │ 엔진 입력용  │ │ 영속화 대상  │
            │ (Dashboard)  │ │ (Orchestrator)│ │ (SQLite)    │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │               │               │
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ WS Broadcast │ │ MC Engine    │ │ trades       │
            │ → Dashboard  │ │ Decision     │ │ equity_hist  │
            └──────────────┘ └──────────────┘ │ positions    │
                                              │ slippage     │
                                              └──────────────┘
```

---

## 2. Paper vs Live 차이점

### 2.1 핵심 차이점 매트릭스

| 항목 | Paper 모드 | Live 모드 | 대시보드 표시 보완 |
|------|-----------|----------|-------------------|
| **진입가** | `_paper_fill_price()` 추정 | 거래소 `execution` WS | Live: 목표가 vs 체결가 모두 표시 |
| **슬리피지** | `PAPER_SLIPPAGE_BPS` 고정 시뮬 | 실제 (목표가-체결가) | Live: 추정 vs 실제 비교 컬럼 추가 |
| **수수료** | `PAPER_FEE_ROUNDTRIP` 시뮬 | 거래소 실제 수수료 | Live: 실제 수수료율 표시 |
| **체결 확률** | PMaker 모델 추정 | maker/taker 실제 결과 | Live: 예상 vs 실제 체결률 |
| **포지션 상태** | 로컬 시뮬레이션 | 거래소 `position` WS | Live: 거래소와 로컬 동기화 상태 |
| **잔고** | 로컬 가상 잔고 | 거래소 `wallet` WS | Live: 실제 잔고 우선 표시 |
| **주문 상태** | 즉시 체결 가정 | 거래소 `order` WS | Live: 미체결/부분체결 상태 표시 |

### 2.2 대시보드 표시 보완 상세

#### 2.2.1 가격 관련 필드 (Live 추가)

```javascript
// Paper 모드
{
  "entry_price": 50000.0,    // 추정 진입가
  "slippage_bps": 2.0        // 시뮬레이션 슬리피지
}

// Live 모드 (추가 필드)
{
  "entry_price": 50000.0,           // 실제 체결가 (거래소)
  "target_price": 49990.0,          // 목표 주문가
  "slippage_actual_bps": 2.0,       // 실제 슬리피지
  "slippage_estimated_bps": 1.5,    // 사전 추정 슬리피지
  "slippage_error_bps": 0.5,        // 추정 오차
  "exec_type": "taker"              // maker/taker
}
```

#### 2.2.2 포지션 동기화 필드 (Live 추가)

```javascript
// Live 모드 추가 필드
{
  "sync_status": "synced",          // "synced" | "pending" | "mismatch"
  "exchange_size": 0.1,             // 거래소 기준 사이즈
  "local_size": 0.1,                // 로컬 기준 사이즈
  "size_diff": 0.0,                 // 차이 (mismatch 감지)
  "last_sync_ms": 1705600000000     // 마지막 동기화 시각
}
```

#### 2.2.3 주문 상태 필드 (Live 추가)

```javascript
// Live 모드 추가 필드
{
  "order_status": "filled",         // "pending" | "partial" | "filled" | "cancelled"
  "fill_progress": 1.0,             // 체결 진행률 (0~1)
  "pending_qty": 0.0,               // 미체결 수량
  "order_age_ms": 1500              // 주문 경과 시간
}
```

### 2.3 대시보드 Row 확장 스키마 (`_row()` 업데이트)

```python
# LiveOrchestrator._row() 추가 반환 필드

# Paper/Live 공통
row = {
    "symbol": sym,
    "price": price,
    "status": status,
    # ... 기존 필드 ...
}

# Live 모드 전용 추가 필드
if not self._paper_trading:
    row.update({
        # 가격/슬리피지 분석
        "target_price": pos.get("target_price"),
        "slippage_actual_bps": pos.get("slippage_actual_bps"),
        "slippage_estimated_bps": pos.get("slippage_estimated_bps"),
        "slippage_error_bps": pos.get("slippage_error_bps"),
        
        # 동기화 상태
        "sync_status": pos.get("sync_status", "synced"),
        "exchange_size": pos.get("exchange_size"),
        "local_size": pos.get("size"),
        
        # 주문 상태
        "order_status": pos.get("order_status"),
        "fill_progress": pos.get("fill_progress"),
        "pending_orders": len(self._pending_orders.get(sym, [])),
        
        # 실제 수수료
        "actual_fee": pos.get("actual_fee"),
        "actual_fee_rate": pos.get("actual_fee_rate"),
    })
```

---

## 3. SQLite 스키마

### 3.1 테이블 요약

| 테이블 | 용도 | 주요 인덱스 |
|--------|------|------------|
| `trades` | 체결 기록 | timestamp_ms, symbol, trading_mode |
| `equity_history` | 자산 시계열 | timestamp_ms, trading_mode |
| `positions` | 현재 포지션 | symbol (PK) |
| `position_history` | 포지션 변경 이력 | timestamp_ms, symbol |
| `bot_state` | KV 스토어 | key (PK) |
| `slippage_analysis` | 슬리피지 분석 | timestamp_ms, symbol |
| `diagnostics` | 진단 메트릭 | timestamp_ms, metric_type |
| `evph_history` | EVPH 히스토리 | timestamp_ms, symbol |

### 3.2 상세 스키마

상세 스키마는 `core/database_manager.py`의 `_initialize_db()` 메서드 참조.

### 3.3 성능 설정

```sql
PRAGMA journal_mode=WAL;      -- Write-Ahead Logging (동시성 향상)
PRAGMA synchronous = NORMAL;  -- 쓰기 성능 최적화
PRAGMA foreign_keys = ON;     -- 참조 무결성
```

---

## 4. 대시보드 데이터 흐름

### 4.1 데이터 제공 방식

| 데이터 | Paper 모드 | Live 모드 |
|--------|-----------|----------|
| **현재가** | WS 브로드캐스트 | WS 브로드캐스트 |
| **포지션 목록** | 메모리 → WS | 거래소 WS → 메모리 → WS |
| **잔고** | 메모리 (SQLite 백업) | 거래소 WS (SQLite 백업) |
| **거래 이력** | SQLite → API | SQLite → API |
| **Equity 그래프** | SQLite → API | SQLite → API |
| **슬리피지 통계** | N/A | SQLite → API |

### 4.2 API 엔드포인트 확장

```
GET /api/trades?mode=paper&limit=100      # 거래 이력
GET /api/equity?mode=live&since=...       # Equity 그래프
GET /api/slippage?symbol=BTC/USDT&days=7  # 슬리피지 통계 (Live)
GET /api/sync_status                       # 동기화 상태 (Live)
```

### 4.3 WebSocket 페이로드 확장

```javascript
// _build_payload() 반환 구조 확장
{
  "time_ms": 1705600000000,
  "trading_mode": "live",  // 추가
  "rows": [...],
  "portfolio": {...},
  "metrics": {...},
  
  // Live 모드 추가
  "sync_status": {
    "overall": "synced",
    "last_sync_ms": 1705600000000,
    "mismatches": []
  },
  "pending_orders": [...],
  "slippage_summary": {
    "avg_bps": 1.5,
    "max_bps": 5.0,
    "estimation_error_avg": 0.3
  }
}
```

---

## 5. 마이그레이션 가이드

### 5.1 기존 JSON → SQLite 마이그레이션

```bash
# 마이그레이션 스크립트 실행
python scripts/migrate_json_to_sqlite.py
```

### 5.2 마이그레이션 대상 파일

| JSON 파일 | SQLite 테이블 | 상태 |
|-----------|--------------|------|
| `state/paper_positions.json` | `positions` | ✅ |
| `state/paper_balance.json` | `bot_state` (key: `balance_paper`) | ✅ |
| `state/paper_trade_tape.json` | `trades` | ✅ |
| `state/paper_equity_history.json` | `equity_history` | ✅ |
| `state/evph_history.json` | `evph_history` | ✅ |
| `state/score_history.json` | `evph_history` | ✅ |
| `state/trades.json` | `trades` | ✅ |

### 5.3 호환성 유지

마이그레이션 후에도 기존 JSON 파일은 백업으로 유지됩니다.
새 코드는 SQLite를 우선 사용하고, SQLite가 없으면 JSON으로 폴백합니다.

---

## 변경 로그

- [2026-01-19] 데이터 저장 설계서 초안 작성 (DATA_PERSISTENCE.md)
- [2026-01-19] DatabaseManager 구현 (core/database_manager.py)
- [2026-01-19] Paper vs Live 차이점 및 대시보드 보완책 정의

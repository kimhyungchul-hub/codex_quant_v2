# Engine/Dashboard Runbook (Bybit Live)

## 왜 연결이 자주 끊겼는가

- 이 작업 환경에서는 일반 `nohup ... &` 방식이 부모 셸 종료/정리 시 함께 종료될 수 있습니다.
- 그 결과 `engine.pid`는 남아 있는데 실제 `main_engine_mc_v2_final.py` 프로세스는 죽어 `:9999`가 닫히는 현상이 재발합니다.

## 재발 방지 표준 기동 방식

- 반드시 `screen` 세션으로 엔진을 분리 실행합니다.
- 표준 스크립트: `scripts/engine_screen.sh`

### 시작/재시작/상태 확인

```bash
cd /Users/jeonghwakim/codex_quant_clean
scripts/engine_screen.sh restart
scripts/engine_screen.sh status
```

### 로그 확인

```bash
cd /Users/jeonghwakim/codex_quant_clean
scripts/engine_screen.sh logs
```

## 정상 상태 판정

- `scripts/engine_screen.sh status`에서 아래 3가지를 동시에 만족해야 정상입니다.
1. `main_engine_mc_v2_final.py` PID가 살아 있음
2. `127.0.0.1:9999 (LISTEN)` 존재
3. `curl http://127.0.0.1:9999/` 응답 코드 `200`

## 이번 운영 고정값

- 방향 신뢰도 임계값 상향:
  - `state/bybit.env`
  - `ALPHA_DIRECTION_MIN_CONFIDENCE=0.58`
  - `ALPHA_DIRECTION_GATE_MIN_CONF=0.58`


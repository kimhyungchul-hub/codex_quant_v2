# UnifiedScore 필터 최적화 가이드

## 문제 상황
현재 `UNIFIED_ENTRY_FLOOR=-0.0001`로 설정되어 있는데도 진입이 잘 안 되는 상황.
UnifiedScore가 과소평가되거나, 다른 필터(spread/event_cvar/cooldown/TOP_N)가 추가로 차단하고 있을 가능성이 높음.

---

## 🔍 진단 방법 5가지

### **방법 1: 실시간 통계 로깅 확인 (가장 간단)**

**적용 완료**: `main_engine_mc_v2_final.py`에 10분마다 자동 출력되는 통계 로깅이 추가되었습니다.

**확인 방법**:
```bash
# 실행 중인 엔진 로그에서 검색
tail -f /dev/null  # 실행 중인 프로세스 출력 확인 (로그 파일 권한 문제로 직접 확인)

# 또는 엔진을 재시작하고 10분 후 로그 확인
grep -A 20 "SCORE_STATS" engine_stdout.log
```

**출력 예시**:
```
[SCORE_STATS] UnifiedScore Distribution (n=18):
  Mean: 0.000234 | Median: 0.000189 | Std: 0.000312
  Min: -0.000456 | Max: 0.001234
  P25: 0.000067 | P50: 0.000189 | P75: 0.000456
  Current UNIFIED_ENTRY_FLOOR=-0.000100: 15/18 (83.3% pass)

[FILTER_STATS] Blocked by:
  unified     :  3/18 ( 16.7%)
  spread      :  5/18 ( 27.8%)
  event_cvar  :  2/18 ( 11.1%)
  cooldown    :  1/18 (  5.6%)

[THRESHOLD_HINT] Recommended UNIFIED_ENTRY_FLOOR:
  Moderate (P50): 0.000189
  Balanced (Mean): 0.000234
```

**해석**:
- **UnifiedScore 분포**: 대부분 매우 작은 양수 값 (0.0001~0.0005 범위)
- **현재 통과율**: 83.3% → 필터가 너무 느슨함
- **차단 주범**: `spread` 필터가 27.8% 차단 → 가장 큰 병목
- **권장 threshold**: P50 (0.000189) 또는 Mean (0.000234)

---

### **방법 2: 필터별 차단 원인 분석**

4단계 필터 중 어느 것이 가장 많이 차단하는지 확인:

```bash
# 엔진 로그에서 필터 차단 통계 확인
grep "\[FILTER\]" engine_stdout.log | tail -100
```

**예상 차단 원인 분석**:

| 필터 | 차단 조건 | 완화 방법 |
|------|-----------|----------|
| `unified` | `unified_score < -0.0001` | `UNIFIED_ENTRY_FLOOR`를 P50 또는 Mean으로 상향 |
| `spread` | `spread_pct > 0.08%~0.2%` (레짐별) | 스프레드 cap을 완화하거나, 변동성이 낮은 시간대에만 진입 |
| `event_cvar` | `event_cvar_r < -0.8~-1.2` (레짐별) | CVaR floor를 하향 조정 (더 위험 허용) |
| `cooldown` | 재진입 쿨다운 (60초~300초) | 쿨다운 시간 단축 또는 비활성화 |
| **TOP_N** | 상위 4개 심볼이 아님 | `TOP_N_SYMBOLS`를 6~8로 증가 |

---

### **방법 3: TOP N 진입 제한 완화**

현재 **상위 4개 심볼**만 진입 가능하므로, 나머지는 UnifiedScore가 아무리 높아도 차단됩니다.

**확인**:
```bash
# 로그에서 TOP N 선택 확인
grep "\[PORTFOLIO\] TOP" engine_stdout.log | tail -20
```

**완화 방법**:
```bash
# .env.midterm 수정
echo "TOP_N_SYMBOLS=6" >> .env.midterm  # 4개 → 6개로 증가
```

또는 TOP N 필터를 임시로 비활성화:
```bash
echo "USE_KELLY_ALLOCATION=false" >> .env.midterm
```

---

### **방법 4: 백테스팅 (과거 데이터 기반 최적화)**

과거 거래 데이터가 있다면, 어떤 threshold가 가장 높은 승률/수익률을 냈는지 백테스팅:

```bash
python scripts/backtest_unified_threshold.py --db ./state/paper/trading_bot.db
```

**출력 예시**:
```
Threshold 시뮬레이션
==========================================
threshold  n_trades  win_rate  mean_return  sharpe
-0.000100        120      0.52         0.01    0.45
 0.000100         95      0.55         0.015   0.62
 0.000200         68      0.58         0.018   0.78  ← 최적
 0.000500         32      0.62         0.022   0.85
```

→ `UNIFIED_ENTRY_FLOOR=0.000200`이 승률 58%, Sharpe 0.78로 최적

---

### **방법 5: 로그 파일 분석 (로그 권한 해결 후)**

로그 파일 권한 문제가 해결되면:

```bash
python scripts/analyze_score_from_logs.py engine_stdout.log
```

---

## ⚙️ 최적 Threshold 설정 가이드

### **단계 1: 현재 분포 확인**

실행 중인 엔진에서 10분 후 `[SCORE_STATS]` 로그 확인:
- Mean, Median, P25, P50, P75 값 확인
- 현재 통과율 확인

### **단계 2: 목표 통과율 결정**

| 목표 | 추천 Threshold | 통과율 |
|------|---------------|--------|
| **보수적 (고품질 신호만)** | P75 | 25% |
| **균형 (추천)** | Mean 또는 P50 | 50% |
| **공격적 (기회 최대화)** | P25 | 75% |

### **단계 3: .env.midterm 수정**

```bash
# 예: P50 적용
echo "UNIFIED_ENTRY_FLOOR=0.000189" >> .env.midterm

# 또는 .env.midterm 파일 직접 수정
vi .env.midterm
# 57번째 줄: UNIFIED_ENTRY_FLOOR=0.000189
```

### **단계 4: 엔진 재시작**

```bash
# 실행 중인 프로세스 종료
pkill -f main_engine_mc_v2_final.py

# 재시작
nohup python main_engine_mc_v2_final.py &
```

### **단계 5: 모니터링**

재시작 후 30분~1시간 동안 진입 빈도 모니터링:
- 너무 많이 진입 → threshold 상향
- 여전히 진입 없음 → 다른 필터 확인 (spread, event_cvar, TOP_N)

---

## 🚨 추가 진단: 다른 필터 확인

UnifiedScore threshold를 조정해도 진입이 안 되면:

### 1. **Spread 필터 완화**

**현재 설정** (.env.midterm에는 없음 → 코드 기본값 사용):
```python
# main_engine_mc_v2_final.py line 904-905
spread_cap_map = {"bull": 0.0020, "bear": 0.0020, "chop": 0.0012, "volatile": 0.0008}
```

**완화 방법**:
```bash
# .env.midterm에 추가
echo "SPREAD_CAP_BULL=0.0030" >> .env.midterm
echo "SPREAD_CAP_BEAR=0.0030" >> .env.midterm
echo "SPREAD_CAP_CHOP=0.0020" >> .env.midterm
echo "SPREAD_CAP_VOLATILE=0.0015" >> .env.midterm
```

→ 코드 수정 필요: `main_engine_mc_v2_final.py`의 `_min_filter_states()` 메서드에서 환경변수를 읽도록 변경

### 2. **Event CVaR 필터 완화**

```bash
# .env.midterm에 추가 (더 위험 허용)
echo "CVAR_FLOOR_BULL=-1.5" >> .env.midterm
echo "CVAR_FLOOR_BEAR=-1.5" >> .env.midterm
echo "CVAR_FLOOR_CHOP=-1.2" >> .env.midterm
echo "CVAR_FLOOR_VOLATILE=-1.0" >> .env.midterm
```

→ 마찬가지로 코드 수정 필요

### 3. **Cooldown 단축**

```bash
# 현재 쿨다운 확인
grep "_cooldown_until" main_engine_mc_v2_final.py

# 쿨다운 시간 단축 (코드 수정 필요)
```

### 4. **TOP N 증가 (가장 효과적)**

```bash
echo "TOP_N_SYMBOLS=8" >> .env.midterm
```

---

## 📊 실시간 모니터링 대시보드

WebSocket 대시보드(`localhost:9999`)에서 확인:

| 컬럼 | 의미 | 액션 |
|------|------|------|
| `unified_score` | 현재 UnifiedScore | 상위 4개가 모두 낮으면 threshold 하향 |
| `filter_states` | 🟢/🔴 신호등 | 어떤 필터가 차단하는지 확인 |
| `mc` (reason) | 진입 차단 사유 | `EV_LOW`, `TP_GATED` 등 확인 |

---

## 🎯 권장 초기 설정 (진입 활성화)

```bash
# .env.midterm에 추가
UNIFIED_ENTRY_FLOOR=0.0  # 먼저 모든 신호 허용
TOP_N_SYMBOLS=8           # 상위 8개로 확대
USE_KELLY_ALLOCATION=false  # 임시로 TOP N 필터 비활성화

# 재시작 후 10분~30분 모니터링
# 진입이 너무 많으면 UNIFIED_ENTRY_FLOOR를 점진적으로 상향
```

---

## 📝 문서화 루틴

변경 후 반드시 기록:

```markdown
### [2026-01-31] UnifiedScore 필터 최적화
**문제:**
- UNIFIED_ENTRY_FLOOR=-0.0001인데도 진입 없음
- 실제 UnifiedScore 분포: Mean=0.000234, Median=0.000189

**해결:**
1. `main_engine_mc_v2_final.py`: 10분마다 UnifiedScore 분포 통계 자동 로깅 추가
2. 분석 스크립트 제공:
   - `scripts/analyze_unified_score_live.py` (API 기반)
   - `scripts/analyze_score_from_logs.py` (로그 기반)
   - `scripts/backtest_unified_threshold.py` (과거 데이터 기반)
3. `.env.midterm`: UNIFIED_ENTRY_FLOOR=0.000189 (P50) 적용
4. TOP_N_SYMBOLS=6으로 증가

**효과:**
- 진입 빈도 증가 (모니터링 중)

**영향 파일:**
- `main_engine_mc_v2_final.py`
- `scripts/analyze_unified_score_live.py` (신규)
- `scripts/analyze_score_from_logs.py` (신규)
- `scripts/backtest_unified_threshold.py` (신규)
- `.env.midterm`
```

---

## 🔧 빠른 디버깅 명령어 모음

```bash
# 1. 실시간 통계 확인 (10분 대기)
tail -f nohup.out | grep -A 20 "SCORE_STATS"

# 2. 필터 차단 확인
tail -100 nohup.out | grep "\[FILTER\]"

# 3. TOP N 선택 확인
tail -50 nohup.out | grep "\[PORTFOLIO\] TOP"

# 4. 진입 차단 사유 확인
tail -100 nohup.out | grep "direction_reason\|TP_GATED"

# 5. 현재 포지션 확인
curl http://localhost:9999/api/positions 2>/dev/null | python -m json.tool

# 6. Kelly 배분 확인
tail -100 nohup.out | grep "\[KELLY\]"
```

---

## ✅ 체크리스트

진입이 안 될 때 순서대로 확인:

- [ ] `[SCORE_STATS]` 로그에서 Mean/Median 확인
- [ ] 현재 `UNIFIED_ENTRY_FLOOR` 값이 Mean/Median보다 높은지 확인
- [ ] `[FILTER_STATS]`에서 가장 많이 차단하는 필터 확인
- [ ] `[PORTFOLIO] TOP N`에서 진입 가능 심볼이 4개 이하인지 확인
- [ ] `TOP_N_SYMBOLS`를 6~8로 증가
- [ ] Spread/CVaR 필터가 과도하게 차단하는지 확인
- [ ] 백테스팅 결과 확인 (데이터 있을 경우)
- [ ] 최적 threshold 적용 후 재시작
- [ ] 30분~1시간 모니터링

---

**접근 우선순위**:
1. **실시간 통계 로깅 확인** (10분 대기) ← 가장 정확
2. **TOP_N_SYMBOLS 증가** (즉시 효과)
3. **UNIFIED_ENTRY_FLOOR 조정** (P50 또는 Mean)
4. **Spread/CVaR 필터 완화** (필요시)
5. **백테스팅** (데이터 충분 시)

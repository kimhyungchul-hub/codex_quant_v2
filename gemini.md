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

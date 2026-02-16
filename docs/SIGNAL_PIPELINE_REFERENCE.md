# Signal Pipeline Reference: mu_alpha → unified_score → Entry Filter

> **목적:** mu_alpha 신호가 어떻게 EV/unified_score로 변환되고 필터에 걸리는지 전체 체인을 문서화.
> 임계치 조정 시 이 문서를 참조하여 상류/하류 영향을 파악할 것.

---

## 0. 요약 다이어그램

```
closes + ofi_score
    │
    ▼
[STAGE 1] signal_features.py — Raw mu_alpha 생성
    ├── mu_mom = momentum × MU_MOM_SCALE (10.0)
    ├── mu_ofi = ofi × MU_OFI_SCALE (15.0)
    ├── divergence → dominant × 0.3
    ├── combined_raw = w_mom × mu_mom + w_ofi × mu_ofi
    ├── KAMA ER < 0.3 → regime_factor 감쇠 (min 0.3)
    └── cap(±5.0) → mu_alpha_raw
            │ 전형적 범위: ±(0.5 ~ 3.0) annualized
            ▼
[STAGE 2] entry_evaluation.py — Advanced Alpha 블렌딩
    ├── + mlofi(w=0.20), kf(0.20), bayes(0.10), ar(0.10)
    ├── + pf(0.10), ml(0.15), causal(0.05)
    └── mu_after_blend ≈ mu_raw × 0.80~0.95
            │
            ▼
[STAGE 3] entry_evaluation.py — 다중 계층 감쇠 (Dampen Chain)
    ├── 3.1 Hurst: H<0.45→OU, 0.45~0.55→×0.75, H>0.55→×1.15
    ├── 3.2 VPIN: damp = max(0.10, 1-0.6×vpin)
    ├── 3.3 Hawkes: +0.3×hawkes_boost
    ├── 3.4 Direction: dir_blend strength=0.6
    ├── 3.5 PMaker: fill_rate bias boost
    ├── 3.6 Cap(±5) + EMA smoothing
    └── mu_alpha_final
            │ 전형적 감쇠율: 30~40% 잔존 (chop에서)
            ▼
[STAGE 4] regime.py — adjust_mu_sigma()
    ├── bull:   mu×1.2, σ×0.9
    ├── bear:   mu×0.7, σ×1.3
    ├── chop:   mu×0.90, σ×1.25  ◀ 추가 10% 감쇠 (2026-02-14 완화)
    ├── volatile: mu×0.6, σ×1.8
    ├── session: ASIA×0.9, EU×1.0, US×1.1, OFF×0.7
    └── mu_adj, sigma_adj → MC 투입
            │
            ▼
[STAGE 5] MC Path Simulation (entry_evaluation.py L974+)
    ├── GBM: S(t) = S₀ × exp((μ-σ²/2)t + σ√t·Z)
    ├── Net PnL = dir × ((S_t/S₀) - 1) × lev - fee × lev
    └── ev_long[h], ev_short[h], cvar_long[h], cvar_short[h]
            │
            ▼
[STAGE 6] Unified Score (Ψ) (entry_evaluation.py L65-96)
    ├── Ψ(t) = (NAPV(t) - cost) / t
    ├── NAPV = Σ(utility - ρ) × e^(-ρτ) × Δτ
    ├── utility = marginal_ev - λ×|marginal_cvar|
    └── unified_score = max(Ψ_long, Ψ_short)
            │ 전형적 범위: -0.001 ~ +0.001 (chop)
            ▼
[STAGE 7] Entry Filter Chain (main_engine L5033-6230)
    ├── unified_floor = max(UNIFIED_ENTRY_FLOOR, MIN_ENTRY_SCORE)
    ├──                 + CHOP_ENTRY_FLOOR_ADD (chop시)
    ├──                 + BEAR_ENTRY_FLOOR_ADD (bear시)
    ├── unified_ok = unified_score >= unified_floor
    ├── both_ev_neg = !(long_ev ≤ 0 AND short_ev ≤ 0)
    ├── chop_guard = !(|mu_alpha| < 0.5 AND dir_conf < 0.60)
    └── ... 15개 필터 순차 체크
```

---

## 1. 환경변수 참조표 — 신호 강도 & 필터 임계치

### 1.1 신호 생성 (mu_alpha 크기 결정)

| 변수 | 기본값 | 역할 | 파일 |
|------|--------|------|------|
| `MU_MOM_SCALE` | 10.0 | 모멘텀 스케일링 계수 | signal_features.py |
| `MU_OFI_SCALE` | 15.0 | OFI 스케일링 계수 | signal_features.py |
| `ALPHA_SIGNAL_BOOST` | true | 신호 1.2× 부스트 | signal_features.py |
| `ALPHA_SCALING_FACTOR` | 1.0 | 전역 mu_alpha 스케일링 | config.py |

### 1.2 감쇠 체인 (mu_alpha 축소 단계)

| 변수 | 기본값 | 역할 | 감쇠 효과 |
|------|--------|------|-----------|
| `HURST_RANDOM_DAMPEN` | **0.75** | Hurst≈0.5 시 감쇠 | ×0.75 |
| `HURST_TREND_BOOST` | 1.15 | Hurst>0.55 시 부스트 | ×1.15 |
| `VPIN` (실시간) | 0.0~1.0 | VPIN damp = 1-0.6×vpin | ×0.4~1.0 |
| regime chop mult | **0.90** | adjust_mu_sigma() | ×0.90 |
| regime bear mult | 0.7 | adjust_mu_sigma() | ×0.70 |
| session ASIA mult | 0.9 | adjust_mu_sigma() | ×0.90 |
| session OFF mult | 0.7 | adjust_mu_sigma() | ×0.70 |

**Chop 최악 감쇠 시나리오:** Hurst×0.75 → VPIN×0.40 → Chop×0.90 → ASIA×0.90 = **총 0.243배** (75.7% 절삭)

### 1.3 MC → EV → Unified Score 변환

| 변수 | 기본값 | 역할 |
|------|--------|------|
| `UNIFIED_RHO` | 0.001 | Ψ 할인율 (낮을수록 원시 EV에 가까움) |
| `UNIFIED_LAMBDA` | **RegimePolicy별** | CVaR 가중: bull=0.05, bear=0.25, chop=0.15, volatile=0.30 |
| `DEFAULT_TP_PCT` | 0.006 | MC TP 목표 (EV에 영향). RegimePolicy별 override: bull=0.035, bear=0.015, chop=0.020 |
| `DEFAULT_SL_PCT` | **0.005** | MC SL 목표 (EV에 영향). RegimePolicy별 override: bull=0.012, bear=0.020, chop=0.015 |
| `MC_N_PATHS_LIVE` | 16000 | 경로 수 (정밀도) |

### 1.4 진입 필터 임계치

| 변수 | 수정 후 값 | 수정 전 | 역할 | ⚠️ 주의사항 |
|------|-----------|---------|------|-----------|
| `UNIFIED_ENTRY_FLOOR` | -0.0003 | -0.0003 | 기본 Ψ 임계치 | |
| `MIN_ENTRY_SCORE` | **0** | 0.0000005 | **>0이면 max()에서 negative floor 무효화** | 반드시 0 또는 음수 |
| `CHOP_ENTRY_FLOOR_ADD` | **0.0001** | 0.0015 | chop 추가 bar | 0.0015는 100% 차단 |
| `BEAR_ENTRY_FLOOR_ADD` | **0.0002** | 0.0010 | bear 추가 bar | |
| `ENTRY_BOTH_EV_NEG_NET_FLOOR` | **-0.0003** | 0.0 | Long&Short 모두 음수 시 차단 | fee > drift면 항상 발동 |
| `ENTRY_NET_EXPECTANCY_MIN` | **-0.0008** | -0.00025 | net_edge 최소 (⚠️ fee 이중 차감) | |
| `ENTRY_NET_EXPECTANCY_MIN_CHOP` | **-0.0008** | -0.00020 | chop net_edge 최소 | |
| `CHOP_ENTRY_MIN_MU_ALPHA` | **0.3** | 0.5 | chop guard |mu_alpha| 최소 | |
| `CHOP_ENTRY_MIN_DIR_CONF` | **0.55** | 0.60 | chop guard dir_conf 최소 | |

---

## 2. 유효 임계치 계산 공식

```
effective_floor = max(UNIFIED_ENTRY_FLOOR, MIN_ENTRY_SCORE)
                + (regime == "chop" ? CHOP_ENTRY_FLOOR_ADD : 0)
                + (regime == "bear" ? BEAR_ENTRY_FLOOR_ADD : 0)
```

**현재 값 (2026-02-13 수정 후):**

| Regime | 유효 floor | 계산 | 현실 score range | 통과율 |
|--------|-----------|------|-----------------|--------|
| chop | **-0.0002** | -0.0003 + 0.0001 | -0.001 ~ +0.0003 | **~60%** ✅ |
| bear | **-0.0001** | -0.0003 + 0.0002 | -0.001 ~ +0.001 | ~70% |
| bull | **-0.0003** | -0.0003 + 0 | -0.001 ~ +0.003 | ~80% |
| neutral | **-0.0003** | -0.0003 + 0 | -0.001 ~ +0.002 | ~75% |

> **⚠️ 핵심 함정:** `MIN_ENTRY_SCORE > 0`이면 `max(UNIFIED_ENTRY_FLOOR, MIN_ENTRY_SCORE)`에서
> 음수 floor가 양수로 override됩니다. 반드시 `MIN_ENTRY_SCORE=0` 유지.

---

## 3. 감쇠 체인 누적 테이블 (Chop Regime, ASIA Session)

| 단계 | 승수 | 누적 잔존 | mu 예시 (raw=1.5) |
|------|------|----------|-------------------|
| 원시 | 1.00 | 100% | 1.500 |
| Alpha blend | 0.85 | 85% | 1.275 |
| Hurst (random) | **0.75** | 63.8% | 0.956 |
| VPIN (=0.5) | 0.70 | 44.6% | 0.669 |
| Regime (chop) | **0.90** | 40.2% | 0.603 |
| Session (ASIA) | 0.90 | 36.1% | 0.542 |
| **총 감쇠** | **0.361** | **36.1%** | **0.542** |

→ 이 mu=0.542 annualized가 60s horizon의 dt=1.9e-6으로 환산되면:
- drift per step = 0.542 × 1.9e-6 = **1.03e-6**
- noise per step = σ(0.6) × √(1.9e-6) = **0.000828**
- 60s total drift = **6.2e-5**
- roundtrip fee = **0.0007**
- **drift(0.000062) << fee(0.0007) → EV ≈ -fee** (감쇠 완화로 소폭 개선)

---

## 4. 조정 가이드라인

### 4.1 Unified Score가 너무 낮을 때 (현재 상황)

**증상:** 모든 종목 unified 필터 차단, score < 0.001
**원인:** CHOP_ENTRY_FLOOR_ADD가 score 범위 대비 과도하게 높음

| 조정 대상 | 방향 | 권장값 | 효과 |
|-----------|------|--------|------|
| `CHOP_ENTRY_FLOOR_ADD` | ↓ | 0.0002~0.0003 | chop 유효 floor 내림 |
| `BEAR_ENTRY_FLOOR_ADD` | ↓ | 0.0002~0.0005 | bear 유효 floor 내림 |
| `UNIFIED_ENTRY_FLOOR` | 유지 | -0.0003 | 이미 적절 |
| `ENTRY_BOTH_EV_NEG_NET_FLOOR` | ↓ | -0.0003 | 약간의 음수 EV 허용 |

### 4.2 mu_alpha 신호가 너무 약할 때 

**증상:** MU_ALPHA_HEALTH mean < 1.0, chop_guard 빈번 차단
**원인:** 감쇠 체인 과다 

| 조정 대상 | 방향 | 범위 | 주의 |
|-----------|------|------|------|
| `HURST_RANDOM_DAMPEN` | ↑ | 현재 0.75 (이미 완화됨) | OU 블렌드 비율도 조정 필요 |
| `MU_MOM_SCALE` | ↑ | 10→15 | 과도 시 false signal |
| `ALPHA_SCALING_FACTOR` | ↑ | 1.0→1.5 | 전역 부스트 |
| `CHOP_ENTRY_MIN_MU_ALPHA` | ↓ | 0.5→0.3 | chop_guard 완화 |

### 4.3 Fee가 Drift를 지배할 때

**증상:** both_ev_neg 빈번, EV ≈ -fee
**근본 원인:** 짧은 horizon에서 fee impact가 drift를 압도

| 조정 대상 | 방향 | 효과 |
|-----------|------|------|
| `DEFAULT_TP_PCT` | ↑ | 더 큰 움직임을 기대 (horizon 연장) |
| horizon 분포 | 조정 | 짧은 horizon 비중 줄이기 |
| fee 추정치 | ↓ | 실제 fee가 낮으면 추정 조정 |

### 4.3.1 net_expectancy의 Fee 이중 차감 문제 (KNOWN ISSUE)

`_min_filter_states()`에서:
```python
net_edge = edge_raw - fee_est  # L5490
```
- `edge_raw` = `policy_ev_mix_long/short` (MC EV, **이미 fee 포함**)
- `fee_est` = `_entry_roundtrip_cost()` (슬리피지+수수료 재계산)
- **결과:** fee가 이중 차감 → `net_edge ≈ MC_EV - 2×fee`

**현재 대응:** `ENTRY_NET_EXPECTANCY_MIN*` 값을 충분히 음수로 설정 (-0.0008)
**장기 수정:** edge_raw에서 MC fee를 역산해 더하거나, net_expectancy 필터의 fee_est를 0으로 설정

### 4.4 필터 우선순위 변경 시 영향 분석

```
unified ← CHOP_ENTRY_FLOOR_ADD로 조절 (가장 직접적)
 ↑ 
 ├── both_ev_neg ← ENTRY_BOTH_EV_NEG_NET_FLOOR로 조절
 ├── chop_guard ← CHOP_ENTRY_MIN_MU_ALPHA, MIN_DIR_CONF로 조절  
 ├── dir_gate ← ALPHA_DIRECTION_GATE_MIN_*로 조절
 └── net_expectancy ← 별도 로직
```

---

## 5. 조정 시 체크리스트

- [ ] 변경 전 `[SCORE_STATS]` 로그에서 현재 score 분포 확인
- [ ] `effective_floor`를 score 분포의 P25~P50 범위로 설정
- [ ] 변경 후 10분간 `[FILTER]` 로그에서 차단률 변화 확인
- [ ] `both_ev_neg`와 `unified` 동시 차단이면 fee vs drift 문제 (신호 강도)
- [ ] `chop_guard`만 차단이면 mu_alpha 자체가 약함 (감쇠 체인 확인)
- [ ] mu_alpha 감쇠 관련 수정 시 `[COMPOUND_DAMPEN_ALERT]` 로그 모니터링
- [ ] `total_dampen_ratio > 0.90` 경고가 나오면 감쇠 과다

---

## 6. 실시간 진단 명령어

```bash
# Score 분포 확인
grep "SCORE_STATS" /tmp/engine.log | tail -3

# 필터 차단 현황
grep "FILTER" /tmp/engine.log | tail -30 | sort | uniq -c | sort -rn

# mu_alpha 건강도
grep "MU_ALPHA_HEALTH" /tmp/engine.log | tail -3

# 감쇠 경고
grep "COMPOUND_DAMPEN\|DAMPEN_ALERT" /tmp/engine.log | tail -5

# 레버리지 진단
grep "LEV_DIAG" /tmp/engine.log | tail -10

# 유효 임계치 빠른 계산
python3 -c "
import os; os.chdir('codex_quant_clean')
f=float(os.environ.get('UNIFIED_ENTRY_FLOOR','-0.0003'))
m=float(os.environ.get('MIN_ENTRY_SCORE','0'))
c=float(os.environ.get('CHOP_ENTRY_FLOOR_ADD','0.0015'))
print(f'Chop effective: {max(f,m)+c:.6f}')
"
```

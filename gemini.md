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

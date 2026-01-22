# 수학 공식 레퍼런스 (Mathematics Reference)

이 문서는 **codex_quant** 프로젝트에서 사용하는 모든 핵심 수학 공식을 정리한 레퍼런스입니다.

> **목적**: 공식의 목적, 도입 배경, 입력/출력, 구현 위치를 명확히 하여 유지보수 및 리팩토링 시 참조
> **규칙**: 수학 공식을 수정하거나 새로 추가할 때 반드시 이 문서를 갱신하세요.

---

## 목차
1. [Monte Carlo 시뮬레이션](#1-monte-carlo-시뮬레이션)
2. [Kelly Criterion (포지션 사이징)](#2-kelly-criterion-포지션-사이징)
3. [CVaR (Conditional Value at Risk)](#3-cvar-conditional-value-at-risk)
4. [확률 계산 (Probability Methods)](#4-확률-계산-probability-methods)
5. [신호 알파 (Signal Alpha)](#5-신호-알파-signal-alpha)
6. [레짐 조정 (Regime Adjustment)](#6-레짐-조정-regime-adjustment)
7. [기술적 지표](#7-기술적-지표)

---

## 1. Monte Carlo 시뮬레이션

### 1.1 GBM (Geometric Brownian Motion) 경로 생성

**목적**: 미래 가격 경로를 확률적으로 시뮬레이션하여 다양한 시나리오에서의 수익/손실 분포를 추정

**도입 배경**: 블랙-숄즈 모델에서 사용되는 표준 자산 가격 모델. 로그 수익률이 정규분포를 따른다고 가정.

**수학식**:
$$
S_t = S_0 \cdot \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right]
$$

또는 이산화된 형태:
$$
\ln\left(\frac{S_{t+\Delta t}}{S_t}\right) = \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t} \cdot Z
$$

여기서:
- $S_t$: 시점 $t$에서의 가격
- $\mu$: 드리프트 (연율 기대수익률)
- $\sigma$: 변동성 (연율 표준편차)
- $W_t$: 위너 과정 (브라운 운동)
- $Z \sim N(0,1)$: 표준정규분포 난수
- $\Delta t$: 시간 스텝 (연 단위, 예: 1초 = $\frac{1}{31536000}$)

**입력**:
| 변수 | 타입 | 설명 |
|------|------|------|
| `s0` | float | 초기 가격 |
| `mu` | float | 연율 드리프트 |
| `sigma` | float | 연율 변동성 |
| `n_paths` | int | 시뮬레이션 경로 수 |
| `n_steps` | int | 시간 스텝 수 |
| `dt` | float | 시간 스텝 크기 (연 단위) |

**출력**: `np.ndarray` shape `(n_paths, n_steps+1)` — 가격 경로 행렬

**구현 위치**:
- `engines/mc/path_simulation.py` → `simulate_paths_price()`
- `engines/simulation_methods.py` → `_mc_first_passage_tp_sl_jax_core()`

**코드 예시**:
```python
drift = (mu - 0.5 * sigma * sigma) * dt
diffusion = sigma * math.sqrt(dt)
z = np.random.standard_normal((n_paths, n_steps))
log_returns = drift + diffusion * z
prices = s0 * np.exp(np.cumsum(log_returns, axis=1))
```

---

### 1.2 Student-t 분포 꼬리 샘플링

**목적**: 팻테일(fat-tail) 리스크를 반영하기 위해 정규분포 대신 Student-t 분포 사용

**도입 배경**: 실제 금융 수익률은 정규분포보다 꼬리가 두껍다 (팻테일). Student-t 분포는 자유도(df)로 꼬리 두께 조절 가능.

**수학식**:
$$
Z_t = \frac{X}{\sqrt{U/\nu}}
$$
여기서:
- $X \sim N(0,1)$
- $U \sim \text{Gamma}(\nu/2, 1)$ (또는 $\chi^2_\nu$)
- $\nu$: 자유도 (df)

**입력**:
| 변수 | 타입 | 설명 |
|------|------|------|
| `df` | float | 자유도 (기본값: 6.0) |
| `shape` | tuple | 출력 배열 형상 |

**출력**: Student-t 분포를 따르는 난수 배열

**구현 위치**:
- `engines/simulation_methods.py` → `_sample_noise()`
- `engines/mc/path_simulation.py` → `_simulate_paths_price_jax_core()`

---

### 1.3 First-Passage Time (TP/SL 도달 확률)

**목적**: 가격이 TP(Take Profit) 또는 SL(Stop Loss)에 먼저 도달할 확률과 시간 분포 계산

**도입 배경**: 트레이딩에서 청산 전략 평가에 필수. MC 시뮬레이션으로 이론적 닫힌 형태 없이도 계산 가능.

**알고리즘**:
1. GBM 경로 생성
2. 각 경로에서 TP/SL 레벨 도달 여부 및 시점 추적
3. 통계 집계: $P(\text{TP first})$, $P(\text{SL first})$, $P(\text{timeout})$, 기대 수익률

**입력**:
| 변수 | 타입 | 설명 |
|------|------|------|
| `s0` | float | 초기 가격 |
| `tp_pct` | float | TP 목표 (예: 0.01 = 1%) |
| `sl_pct` | float | SL 한도 (예: 0.005 = 0.5%) |
| `mu`, `sigma` | float | 드리프트, 변동성 |
| `max_steps` | int | 최대 시뮬레이션 스텝 |
| `n_paths` | int | 경로 수 |

**출력**:
| 필드 | 타입 | 설명 |
|------|------|------|
| `event_p_tp` | float | TP 먼저 도달 확률 |
| `event_p_sl` | float | SL 먼저 도달 확률 |
| `event_p_timeout` | float | 타임아웃 확률 |
| `event_ev_r` | float | 기대 R-배수 (수익률/SL) |
| `event_cvar_r` | float | CVaR (조건부 VaR) |
| `event_t_median` | float | 도달 시간 중앙값 |

**구현 위치**:
- `engines/mc/first_passage.py` → `mc_first_passage_tp_sl()`
- `engines/simulation_methods.py` → `mc_first_passage_tp_sl_jax()`

---

## 2. Kelly Criterion (포지션 사이징)

### 2.1 기본 Kelly 공식

**목적**: 기대 성장률을 최대화하는 최적 베팅 비율 결정

**도입 배경**: John Kelly (1956). 정보 이론에서 유래, 복리 성장 최적화에 사용.

**수학식**:
$$
f^* = \frac{p \cdot b - q}{b} = \frac{p(b+1) - 1}{b}
$$

여기서:
- $f^*$: 최적 자본 배분 비율
- $p$: 승률
- $q = 1 - p$: 패률
- $b$: 배당률 (이익/손실 비율)

**입력**:
| 변수 | 타입 | 설명 |
|------|------|------|
| `p` | float | 승률 (0~1) |
| `b` | float | 배당률 (TP/SL 비율) |

**출력**: `float` — 최적 베팅 비율 (0~1)

**구현 위치**:
- `engines/mc_risk.py` → `kelly_fraction()`

**코드**:
```python
def kelly_fraction(p, b):
    q = 1 - p
    if b <= 0:
        return 0.0
    return max(0.0, min(1.0, (b*p - q) / b))
```

---

### 2.2 CVaR 보정 Kelly

**목적**: 꼬리 리스크(CVaR)를 반영하여 Kelly 비율을 보수적으로 조정

**도입 배경**: 순수 Kelly는 리스크를 과소평가할 수 있음. CVaR 페널티로 극단적 손실 고려.

**수학식**:
$$
f_{\text{adj}} = f^* \cdot (1 - \text{penalty})
$$
$$
\text{penalty} = \min(1.0, |\text{CVaR}| \times 10)
$$

**입력**:
| 변수 | 타입 | 설명 |
|------|------|------|
| `win_rate` | float | 승률 |
| `tp` | float | TP 수익률 |
| `sl` | float | SL 손실률 |
| `cvar` | float | CVaR (음수) |

**출력**: `float` — CVaR 보정된 Kelly 비율

**구현 위치**:
- `engines/mc_risk.py` → `kelly_with_cvar()`

---

### 2.3 다변량 Kelly (Multivariate Kelly)

**목적**: 여러 자산 간 상관관계를 고려한 최적 포트폴리오 배분

**도입 배경**: 단일 자산 Kelly를 다자산으로 확장. 공분산 행렬 사용.

**수학식**:
$$
\mathbf{f}^* = K \cdot \Sigma^{-1} \cdot \boldsymbol{\mu}
$$

여기서:
- $\mathbf{f}^*$: 최적 배분 벡터
- $K$: Kelly 승수 (예: 0.5 = half-Kelly)
- $\Sigma$: 수익률 공분산 행렬
- $\boldsymbol{\mu}$: 기대수익률 벡터 (NAPV 점수)

**입력**:
| 변수 | 타입 | 설명 |
|------|------|------|
| `candidates` | List[Dict] | `{'symbol': str, 'mu': float}` |
| `cov_matrix` | np.ndarray | 공분산 행렬 (n×n) |
| `half_kelly` | float | Kelly 승수 (기본: 1.0) |

**출력**: `Dict[str, float]` — 심볼별 배분 비중

**구현 위치**:
- `engines/kelly_allocator.py` → `KellyAllocator.compute_allocation()`

---

## 3. CVaR (Conditional Value at Risk)

### 3.1 경험적 CVaR

**목적**: 주어진 신뢰수준(α)에서 VaR을 초과하는 손실의 기대값 계산

**도입 배경**: VaR의 한계 보완. 꼬리 리스크를 평균으로 측정.

**수학식**:
$$
\text{CVaR}_\alpha = E[X | X \leq \text{VaR}_\alpha] = \frac{1}{\alpha} \int_0^\alpha \text{VaR}_u \, du
$$

이산 근사:
$$
\text{CVaR}_\alpha \approx \frac{1}{k} \sum_{i=1}^{k} X_{(i)}
$$
여기서 $k = \lfloor \alpha \cdot n \rfloor$, $X_{(i)}$는 정렬된 손실의 $i$번째 값

**입력**:
| 변수 | 타입 | 설명 |
|------|------|------|
| `pnl` | np.ndarray | PnL 샘플 배열 |
| `alpha` | float | 신뢰수준 (기본: 0.05 = 5%) |

**출력**: `float` — CVaR 값

**구현 위치**:
- `engines/cvar_methods.py` → `_cvar_empirical()`

**코드**:
```python
def _cvar_empirical(pnl, alpha=0.05):
    x = np.sort(np.asarray(pnl))
    k = max(1, int(alpha * len(x)))
    return float(x[:k].mean())
```

---

### 3.2 부트스트랩 CVaR

**목적**: 샘플링 불확실성을 줄이기 위해 부트스트랩으로 CVaR 추정

**도입 배경**: 소표본에서 경험적 CVaR은 불안정. 부트스트랩으로 중앙값 사용.

**알고리즘**:
1. 원본 데이터에서 복원 추출로 $n_{\text{boot}}$개 샘플 생성
2. 각 샘플에서 CVaR 계산
3. 중앙값 반환

**입력**:
| 변수 | 타입 | 설명 |
|------|------|------|
| `pnl` | np.ndarray | PnL 배열 |
| `alpha` | float | 신뢰수준 |
| `n_boot` | int | 부트스트랩 반복 수 (기본: 40) |
| `sample_frac` | float | 샘플 비율 (기본: 0.7) |

**출력**: `float` — 부트스트랩 CVaR

**구현 위치**:
- `engines/cvar_methods.py` → `_cvar_bootstrap()`

---

### 3.3 앙상블 CVaR

**목적**: 여러 CVaR 추정 방법을 가중 평균하여 강건한 추정치 산출

**수학식**:
$$
\text{CVaR}_{\text{ensemble}} = 0.60 \cdot \text{CVaR}_{\text{bootstrap}} + 0.25 \cdot \text{CVaR}_{\text{empirical}} + 0.15 \cdot \text{CVaR}_{\text{tail-inflate}}
$$

**구현 위치**:
- `engines/cvar_methods.py` → `cvar_ensemble()`

---

## 4. 확률 계산 (Probability Methods)

### 4.1 정규분포 CDF

**목적**: 표준정규분포의 누적분포함수 계산

**수학식**:
$$
\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

**구현 위치**:
- `engines/probability_methods.py` → `_norm_cdf()`

---

### 4.2 양의 수익 확률 및 기대 EV

**목적**: 주어진 드리프트/변동성에서 수익이 양수일 확률과 기대 EV 계산

**수학식**:
수익률: $R = \mu \cdot \tau \cdot L - \text{fee}$ (여기서 $L$=레버리지)

양의 수익 확률 (롱):
$$
P(R > 0) = \Phi\left(\frac{\mu \cdot \tau - \text{thr}}{\sigma \sqrt{\tau}}\right)
$$
여기서 $\text{thr} = \text{fee} / L$

**입력**:
| 변수 | 타입 | 설명 |
|------|------|------|
| `mu` | float | 드리프트 (연율) |
| `sigma` | float | 변동성 (연율) |
| `tau_sec` | float | 보유 시간 (초) |
| `direction` | int | 1=롱, -1=숏 |
| `leverage` | float | 레버리지 |
| `fee_roundtrip` | float | 왕복 수수료 |

**출력**: `(p_pos, ev)` — 양의 수익 확률, 기대 EV

**구현 위치**:
- `engines/probability_methods.py` → `_approx_p_pos_and_ev_hold()`

---

### 4.3 최대값 도달 확률 (Reflection Principle)

**목적**: 브라운 운동이 특정 레벨에 도달할 확률 계산

**수학식** (Reflection Principle):
$$
P\left(\max_{0 \leq s \leq T} X_s \geq a\right) = 1 - \Phi(z_1) + e^{2\mu_0 a / \sigma_0^2} \cdot \Phi(-z_1 - 2a/(\sigma_0\sqrt{T}))
$$
여기서 $z_1 = (a - \mu_0 T) / (\sigma_0 \sqrt{T})$

**구현 위치**:
- `engines/probability_methods.py` → `_prob_max_geq()`, `_prob_min_leq()`

---

### 4.4 AlphaHit 엔트로피 기반 신뢰도 가중치

**목적**: TP/SL 확률이 50:50 근처로 애매할 때 AlphaHit 영향도를 자동으로 낮춤

**수학식**:
확률 벡터 $p = (p_{tp}, p_{sl}, p_{other})$, $p_{other} = 1 - p_{tp} - p_{sl}$

샤논 엔트로피:
$$
H(p) = -\sum_i p_i \log(p_i)
$$

정규화 신뢰도:
$$
	ext{conf} = 1 - \frac{H(p)}{\log(3)}
$$

가중치 적용:
$$
\beta_{\text{eff}} = \beta \cdot \text{warm} \cdot \text{conf}
$$

**구현 위치**:
- `engines/mc/monte_carlo_engine.py` → `alpha_hit_confidence()`
- `engines/mc/entry_evaluation_new.py` → AlphaHit 확률 블렌딩

---

### 4.5 변동성 기반 TP 타겟 (Dynamic Volatility Target)

**목적**: 고정 TP 목표 대신 시장 변동성(ATR)에 따라 보상을 확장

**수학식**:
$$
	ext{atr\_frac} = \frac{\text{ATR}}{\text{price}}
$$
$$
tp_r = \max\left(tp_{\text{base}}, k_{\text{atr}} \cdot \text{atr\_frac}\right)
$$

**구현 위치**:
- `engines/mc/entry_evaluation_new.py` → TP 타겟 배치 입력 생성

---

### 4.6 PMaker 역선택(Adverse Selection) 비용

**목적**: 체결 이후 불리한 가격 이동을 entry 비용에 반영하여 EV를 보수화

**수학식** (롱 기준):
$$
	ext{adverse\_move} = \max\left(0, \frac{P_{fill} - P_{t+\Delta}}{P_{fill}}\right)
$$

숏 기준:
$$
	ext{adverse\_move} = \max\left(0, \frac{P_{t+\Delta} - P_{fill}}{P_{fill}}\right)
$$

비용 반영:
$$
	ext{cost\_entry}' = \text{cost\_entry} + k_{\text{adverse}} \cdot \text{adverse\_move}
$$

**구현 위치**:
- `core/orchestrator.py` → `_pmaker_paper_probe_tick()`
- `engines/mc/entry_evaluation_new.py` → entry 비용 보정

---

## 5. 신호 알파 (Signal Alpha)

### 5.1 연율화 (Annualization)

**목적**: 바(bar) 단위 수익률/변동성을 연율로 변환

**수학식**:
$$
\mu_{\text{annual}} = \mu_{\text{bar}} \times \frac{365 \times 24 \times 3600}{\text{bar\_seconds}}
$$
$$
\sigma_{\text{annual}} = \sigma_{\text{bar}} \times \sqrt{\frac{365 \times 24 \times 3600}{\text{bar\_seconds}}}
$$

**구현 위치**:
- `engines/mc/signal_features.py` → `_annualize()`

---

### 5.2 모멘텀 기반 알파 (mu_mom)

**목적**: 최근 가격 모멘텀에서 기대 수익률(알파) 추정

**수학식**:
$$
\text{lr}_w = \ln\left(\frac{P_{-1}}{P_{-w-1}}\right)
$$
$$
\mu_{\text{mom},w} = \frac{\text{lr}_w}{\tau_w} \times \text{SECONDS\_PER\_YEAR}
$$

가중 평균:
$$
\mu_{\text{mom}} = \frac{\sum_w w_w \cdot \mu_{\text{mom},w}}{\sum_w w_w}
$$

기본 가중치: `{15: 0.35, 30: 0.30, 60: 0.25, 120: 0.10}`

**구현 위치**:
- `engines/mc/signal_features.py` → `_signal_alpha_mu_annual_parts()`

---

### 5.3 OFI 기반 알파 (mu_ofi)

**목적**: 주문 흐름 불균형(OFI)에서 단기 알파 추정

**수학식**:
$$
\mu_{\text{ofi}} = \text{clip}(\text{ofi\_score}, -1, 1) \times \text{ofi\_scale}
$$

기본 `ofi_scale = 8.0`

**구현 위치**:
- `engines/mc/signal_features.py` → `_signal_alpha_mu_annual_parts()`

---

### 5.4 결합 알파 (mu_alpha)

**목적**: 모멘텀과 OFI 알파를 가중 결합하고 레짐 스케일 적용

**수학식**:
$$
\mu_{\alpha,\text{raw}} = w_{\text{mom}} \cdot \mu_{\text{mom}} + w_{\text{ofi}} \cdot \mu_{\text{ofi}}
$$
$$
\mu_\alpha = \mu_{\alpha,\text{raw}} \times \text{regime\_scale}
$$

기본 가중치: `w_mom = 0.70`, `w_ofi = 0.30`

**레짐 스케일**: Kaufman Efficiency Ratio 기반
$$
\text{ER} = \frac{|\text{net price change}|}{\sum |\text{incremental changes}|}
$$
$$
\text{chop\_score} = 1 - \text{ER}
$$
$$
\text{regime\_scale} = 1 - (1 - \text{scale\_min}) \times \text{chop\_score}
$$

**구현 위치**:
- `engines/mc/signal_features.py` → `_signal_alpha_mu_annual_parts()`

---

## 6. 레짐 조정 (Regime Adjustment)

### 6.1 mu/sigma 레짐 조정

**목적**: 시장 레짐에 따라 드리프트와 변동성을 조정

**조정 테이블**:
| 레짐 | mu 조정 | sigma 조정 |
|------|---------|------------|
| BULL | ×1.05 | ×0.95 |
| BEAR | ×0.95 | ×1.05 |
| VOLATILE | ×1.00 | ×1.15 |
| CHOP | ×0.90 | ×1.10 |

**구현 위치**:
- `regime/__init__.py` → `adjust_mu_sigma()`

---

## 7. 기술적 지표

### 7.1 EMA (Exponential Moving Average)

**목적**: 최근 데이터에 더 높은 가중치를 부여하는 이동평균

**수학식**:
$$
\text{EMA}_t = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_{t-1}
$$
여기서 $\alpha = \frac{2}{n + 1}$ (n = 기간)

**구현 위치**:
- `engines/mc/signal_features.py` → `ema()`

---

### 7.2 RSI (Relative Strength Index)

**목적**: 과매수/과매도 상태 판단

**수학식**:
$$
\text{RSI} = 100 - \frac{100}{1 + \text{RS}}
$$
$$
\text{RS} = \frac{\text{평균 상승폭}}{\text{평균 하락폭}}
$$

**구현 위치**:
- `utils/helpers.py` → `_calc_rsi()`

---

## 8. Exit Policy (청산 로직)

### 8.1 개요

**목적**: 몬테카를로 시뮬레이션에서 각 경로별 최적 청산 시점을 결정하여 정확한 EV 계산

**기본 설정**: `SKIP_EXIT_POLICY=false` — 모든 청산 로직을 MC 시뮬레이션에 반영

**5가지 청산 조건**:
1. **TP (Take Profit)**: 목표 수익률 도달 시 청산
2. **SL (Stop Loss)**: 손절 수준 도달 시 청산
3. **Time Stop**: 최대 보유 시간 초과 시 청산
4. **Drawdown Stop**: 미실현 손실 임계값 초과 시 청산
5. **Dynamic Policy**: 확률 기반 동적 청산 (레짐별)

---

### 8.2 TP/SL 타겟 계산 (변동성 기반)

**목적**: 각 지평(horizon)별로 변동성에 비례한 TP/SL 레벨 설정

**수학식**:
$$
\text{TP\_target\_ROE} = k_{\text{horizon}} \cdot \sigma \cdot \sqrt{\frac{h}{T}} \cdot m_{\text{TP}}
$$
$$
\text{SL\_target\_ROE} = -k_{\text{horizon}} \cdot \sigma \cdot \sqrt{\frac{h}{T}} \cdot m_{\text{SL}}
$$

여기서:
- $h$: 지평 시간 (초)
- $T$: 1년 (31,536,000초)
- $\sigma$: 연율 변동성
- $k_{\text{horizon}}$: 지평별 배수 (60s → 0.5, 300s → 1.0, 600s → 1.5, ...)
- $m_{\text{TP}}$, $m_{\text{SL}}$: TP/SL 배수 (기본값: 2.0 / 1.0)

**레버리지 반영**:
$$
\text{TP\_final} = \text{TP\_target\_ROE} \times L
$$
$$
\text{SL\_final} = \text{SL\_target\_ROE} \times L
$$
여기서 $L$은 레버리지.

**구현 위치**:
- `engines/mc/exit_policy.py` → `tp_sl_targets_for_horizon()`

---

### 8.3 Drawdown Stop (미실현 손실 청산)

**목적**: 보유 중 미실현 손실이 일정 수준 이상 발생하면 조기 청산

**수학식**:
$$
\text{DD\_ROE}(t) = \frac{P(t) - P_{\text{entry}}}{P_{\text{entry}}} \times L \times \text{direction}
$$
$$
\text{if } \text{DD\_ROE}(t) \leq \text{DD\_STOP\_ROE} \Rightarrow \text{청산}
$$

여기서:
- $P(t)$: 현재 가격
- $P_{\text{entry}}$: 진입 가격
- $L$: 레버리지
- $\text{direction}$: +1 (LONG) or -1 (SHORT)
- $\text{DD\_STOP\_ROE}$: 임계값 (기본값: -0.02, 즉 -2%)

**구현 위치**:
- `engines/mc/exit_policy.py` → `simulate_exit_policy_multi_symbol_jax()` 내부

---

### 8.4 Dynamic Policy (확률 기반 동적 청산)

**목적**: 레짐별 확률 임계값을 기준으로 보유/청산 결정

**청산 조건**:
| 조건 | 트리거 | 설명 |
|------|--------|------|
| **score_flip** | `P(pos) < 0.5` & `confirmed` | 롱→숏 또는 숏→롱 신호 반전 |
| **hold_bad** | `P(pos) < P_hold_floor` for N ticks | 보유 확률이 임계값 이하로 지속 |

**레짐별 확률 임계값**:

```python
# 진입 시 최소 확률
POLICY_P_POS_ENTER = {
    "bull": 0.52,
    "bear": 0.50,
    "chop": 0.52,
    "volatile": 0.55
}

# 보유 시 최소 확률 (이하면 청산)
POLICY_P_POS_HOLD = {
    "bull": 0.50,
    "bear": 0.48,
    "chop": 0.50,
    "volatile": 0.52
}

# SL 최대 허용 확률
POLICY_P_SL_ENTER_MAX = {
    "bull": 0.20,
    "bear": 0.25,
    "chop": 0.20,
    "volatile": 0.15
}

# TP 최소 요구 확률
POLICY_P_TP_ENTER_MIN = {
    "bull": 0.15,
    "bear": 0.12,
    "chop": 0.15,
    "volatile": 0.18
}
```

**구현 위치**:
- `engines/mc/exit_policy.py` → `ExitPolicyMixin.POLICY_P_POS_ENTER_BY_REGIME` 등

---

### 8.5 Exit Policy 계산 프로세스

**알고리즘 플로우**:
```
for each path in paths:
    t = 0
    while t < horizon_sec:
        # 1. TP/SL 체크
        if ROE(t) >= tp_target:
            exit(reason="TP", ROE=ROE(t))
        if ROE(t) <= sl_target:
            exit(reason="SL", ROE=ROE(t))
        
        # 2. Drawdown Stop 체크
        if unrealized_dd(t) <= dd_stop_roe:
            exit(reason="DD_STOP", ROE=ROE(t))
        
        # 3. Dynamic Policy 체크 (매 decision_dt_sec)
        if t % decision_dt_sec == 0:
            p_pos = compute_p_pos(path, t)
            if score_flip(p_pos):
                exit(reason="SCORE_FLIP", ROE=ROE(t))
            if hold_bad(p_pos, regime):
                exit(reason="HOLD_BAD", ROE=ROE(t))
        
        t += step_sec
    
    # 4. Time Stop (만기 청산)
    exit(reason="TIME_STOP", ROE=ROE(horizon_sec))

# EV 계산
EV = mean([ROE - fee_roundtrip for each exited path])
```

**구현 위치**:
- `engines/mc/exit_policy.py` → `compute_exit_policy_metrics_multi_symbol()`
- JAX 커널: `engines/mc/jax_backend.py` → `simulate_exit_policy_multi_symbol_jax()`

---

### 8.6 `SKIP_EXIT_POLICY` 플래그

**기본값**: `false` (모든 청산 로직 반영)

**`SKIP_EXIT_POLICY=true` 설정 시**:
- 청산 로직 생략
- 단순 만기 평균 수익으로 EV 근사:
  $$
  \text{EV\_simplified} = \mathbb{E}[\text{ROE}(T)] - \text{fee\_roundtrip}
  $$
- 속도 향상: 첫 실행 시 ~10-15초 단축 (JAX JIT 컴파일 생략)
- 정확도 하락: TP/SL/DD Stop 무시, 과대/과소평가 가능
- **프로덕션 환경에서는 권장하지 않음**

**구현 위치**:
- `engines/mc/entry_evaluation.py` → `SKIP_EXIT_POLICY` 플래그

---

## 변경 로그

| 날짜 | 변경 내용 | 관련 파일 |
|------|----------|----------|
| 2026-01-18 | 초기 문서 작성 | `docs/MATHEMATICS.md` |
| 2026-01-19 | AlphaHit 엔트로피 가중치, 변동성 TP, PMaker 역선택 비용 수식 추가 | `core/orchestrator.py`, `engines/mc/entry_evaluation_new.py`, `engines/mc/monte_carlo_engine.py` |
| 2026-01-22 | Exit Policy (청산 로직) 섹션 추가 — TP/SL/DD Stop/Dynamic Policy 수식 및 프로세스 명시 | `engines/mc/exit_policy.py`, `engines/mc/entry_evaluation.py` |

---

*작성 일자: 2026-01-18*

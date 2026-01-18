#[Quant Logic] 수학·공학·금융 기법/공식 정리

이 문서는 코드에 실제 구현된 수학적 공식과 기법을 요약한다.

## 0) 표기(Notation)
- 가격 $p_t$, 로그수익 $r_t=\ln(p_t/p_{t-1})$
- 드리프트/기대수익 $\mu$, 변동성 $\sigma$
- 연환산: $\mu_{\text{ann}}$, $\sigma_{\text{ann}}$
- 시간지평 $h$ (sec), $\tau$ (sec), $T$ (sec)
- EV(기대값), CVaR(조건부 VaR), NAPV(순현재가치)

---

## 1) 수익률·변동성 추정
**로그수익/모멘트** (`core/orchestrator.py`)
- 로그수익: $r_i=\ln(p_i/p_{i-1})$
- 평균: $\mu=\frac{1}{n}\sum_i r_i$
- 분산: $\sigma^2=\frac{1}{n}\sum_i (r_i-\mu)^2$ → $\sigma=\sqrt{\sigma^2}$

**연환산** (`engines/mc/signal_features.py`)
$$\mu_{\text{ann}}=\mu_{\text{bar}}\cdot \text{bars\_per\_year},\quad \sigma_{\text{ann}}=\sigma_{\text{bar}}\cdot\sqrt{\text{bars\_per\_year}}$$

**per-second 변환** (`engines/mc/execution_mix.py`, `core/orchestrator.py`)
$$\mu_{\text{ps}}=\frac{\mu_{\text{ann}}}{\text{SECONDS\_PER\_YEAR}},\quad \sigma_{\text{ps}}=\frac{\sigma_{\text{ann}}}{\sqrt{\text{SECONDS\_PER\_YEAR}}}$$

---

## 2) PMaker Paper 추정치
**로그수익 기반 시그마** (`core/orchestrator.py`)
$$\sigma\approx\operatorname{std}(\Delta\ln p)$$

**모멘텀 Z-score** (`core/orchestrator.py`)
$$z=\frac{\ln(p_1/p_0)}{\max(\epsilon,\sigma)\sqrt{w}}$$

---

## 3) OFI / 유동성 기반 점수
**Order Flow Imbalance** (`core/orchestrator.py`)
$$\text{OFI}=\frac{\sum w_i\,\text{bid}_i-\sum w_i\,\text{ask}_i}{\sum w_i\,\text{bid}_i+\sum w_i\,\text{ask}_i}$$

---

## 4) 멀티 타임프레임 점수
**Consensus Score** (`core/multi_timeframe_scoring.py`)
$$S=\sum_k \tilde{w}_k\,s_k,\quad \tilde{w}_k=\frac{w_k}{\sum w_k}$$
기본 가중치: $w_{5m}=0.10,\;w_{10m}=0.20,\;w_{30m}=0.30,\;w_{1h}=0.40$

**EWMA** (`core/multi_timeframe_scoring.py`)
$$S_t=\alpha x_t+(1-\alpha)S_{t-1},\quad \alpha=\frac{2}{\text{span}+1}$$

**Group/Rank Score** (`core/multi_timeframe_scoring.py`)
$$\text{group}=\text{EWMA}_{150}-1.0\cdot \operatorname{Std}(x),\quad \text{rank}=\text{EWMA}_{15}$$

**포지션 교체 비용** (`core/multi_timeframe_scoring.py`)
$$\text{switching\_cost}=\text{fee}\times4\times\text{scaling},\quad \text{switch if }g_{cand}>g_{cur}+\text{cost}$$

---

## 5) 확률 근사 (정규/반사 원리)
**정규 CDF** (`engines/probability_methods.py`)
$$\Phi(x)=\frac{1}{2}\left(1+\operatorname{erf}(x/\sqrt{2})\right)$$

**보유 EV/양의확률 근사** (`engines/probability_methods.py`)
$$m=\mu\tau,\quad v=\sigma^2\tau,\quad s=\sqrt{v},\quad \text{thr}=\frac{\text{fee}}{\text{lev}}$$
$$p_{+}=\Phi\left(\frac{m-\text{thr}}{s}\right)\;\text{(long)},\quad p_{+}=\Phi\left(\frac{-\text{thr}-m}{s}\right)\;\text{(short)}$$
$$EV=\text{direction}\cdot m\cdot\text{lev}-\text{fee}$$

**최대치 도달 확률(Reflection)** (`engines/probability_methods.py`)
$$P\left(\max_{0..T}X_t\ge a\right)\approx (1-\Phi(z_1)) + e^{\frac{2\mu_0 a}{\sigma_0^2}}\,\Phi\left(\frac{-a-\mu_0T}{\sigma_0\sqrt{T}}\right)$$
$$z_1=\frac{a-\mu_0T}{\sigma_0\sqrt{T}}$$

---

## 6) GBM 경로 시뮬레이션
**GBM 로그수익 누적** (`engines/mc/path_simulation.py`)
$$\text{drift}=(\mu-\tfrac{1}{2}\sigma^2)\,dt,\quad \text{diffusion}=\sigma\sqrt{dt}$$
$$\log P_t=\log P_0+\sum(\text{drift}+\text{diffusion}\cdot z_t),\quad P_t=P_0\exp(\cdot)$$

**Tail Sampling** (`engines/mc/tail_sampling.py`)
- Student-$t$ 샘플을 분산정규화: $z\leftarrow z/\sqrt{\nu/(\nu-2)}$
- Bootstrap: 과거 로그수익에서 리샘플링

---

## 7) Horizon 평가 (EV/Win/CVaR)
**Horizon Net Return** (`engines/mc/jax_backend.py`, `engines/mc/entry_evaluation_clean.py`)
$$\text{gross}=(P_T-P_0)/P_0$$
$$\text{net}_{long}=\text{gross}\cdot L-\text{fee},\quad \text{net}_{short}=-\text{gross}\cdot L-\text{fee}$$
$$EV=\mathbb{E}[\text{net}],\quad \text{Win}=P(\text{net}>0)$$

**CVaR (bottom $\alpha$)** (`engines/cvar_methods.py`, `engines/mc/jax_backend.py`)
$$\text{CVaR}_\alpha=\frac{1}{k}\sum_{i\le k}x_{(i)},\quad k=\lfloor\alpha n\rfloor$$
Ensemble: $0.60\cdot\text{bootstrap}+0.25\cdot\text{emp}+0.15\cdot\text{tail-inflate}$

**정책 목적함수** (`engines/mc/entry_evaluation.py`)
$$\text{Objective}=\frac{EV}{|CVaR|}\cdot\frac{1}{\sqrt{h}}$$

**Horizon 가중치** (`engines/mc/policy_weights.py`)
$$w(h)=\operatorname{normalize}(w_{prior}(h)\cdot\operatorname{softplus}(\beta\,EV_h))$$
$$w_{prior}(h)\propto \exp(-h/\text{half\_life})$$

---

## 8) NAPV (Net Added Present Value)
**EV curve 기반 NAPV** (`engines/mc/entry_evaluation.py`, `core/napv_engine_jax.py`)
$$\text{discount}=e^{-\rho t},\quad \text{cumulative}=\int_0^t (r-\rho)e^{-\rho t}dt$$
$$\text{NAPV}=\max_t\{\text{cumulative}(t)-\text{cost}\},\quad t^*=\arg\max$$

**동적 할인율 $\rho$** (`core/portfolio_management.py`)
$$\rho=\max\left(\rho_{\text{default}},\frac{\text{realized\_return\_pct}}{\text{window\_sec}}\right)$$
$$\rho_{\text{default}}=\ln(1.20)/\text{SEC\_IN\_YEAR}$$

---

## 9) Kelly 기반 사이징
**단일 Kelly** (`engines/mc_risk.py`)
$$b=\frac{TP}{SL},\quad f^*=\max\left(0,\min\left(1,\frac{bp-(1-p)}{b}\right)\right)$$

**CVaR 패널티** (`engines/mc_risk.py`)
$$f^*_{\text{cvar}}=f^*\cdot(1-\min(1,|\text{CVaR}|\cdot k))$$

**다변량 Kelly** (`engines/kelly_allocator.py`)
$$\mathbf{f}^*=K\,\Sigma^{-1}\mu$$
(수치 안정성 위해 $\Sigma+\lambda I$ 사용)

---

## 10) 레버리지 최적화
**EV 기반 레버리지 선택** (`engines/mc/leverage_optimizer_jax.py`, `engines/mc/leverage_optimizer_torch.py`)
$$\mu_{h}=(\mu_{ann}/\text{sec\_per\_year})\cdot h$$
$$EV_{long}=\mu_h\cdot L-\text{fee}\cdot L,\quad EV_{short}=-\mu_h\cdot L-\text{fee}\cdot L$$
$$\text{score}=\max(EV_{long}, EV_{short})$$

---

## 11) 실행비용/체결확률 모델
**슬리피지 추정** (`engines/mc/execution_costs.py`)
$$\text{slip}=\text{base}\cdot(1+0.5\sigma)\cdot \text{liq\_term}\cdot(1+0.1\ln(1+|L|))\cdot(1+0.6\min(2,|z_{ofi}|))$$

**Maker 체결 확률 (로지스틱)** (`engines/mc/execution_costs.py`)
$$x=0.35+0.12\ln(\text{liq})-900\cdot\text{spread}-0.25\cdot|z_{ofi}|$$
$$p=\sigma(x)=\frac{1}{1+e^{-x}},\quad p\in[0.05,0.95]$$

**Execution Mix** (`engines/mc/execution_mix.py`)
$$\text{fee\_mix}=p\cdot fee_m+(1-p)\cdot fee_t$$
$$\text{delay\_penalty}=k\cdot\sigma_{ps}\sqrt{\text{delay}}$$

---

## 12) 1차 도달(First Passage) TP/SL
**경로 생성** (`engines/mc/first_passage.py`)
$$\text{drift}=\pm(\mu-\tfrac{1}{2}\sigma^2)dt,\quad \text{diffusion}=\sigma\sqrt{dt}$$
$$P_t=P_0\exp(\text{direction}\cdot\text{logret})$$

**TP/SL 레벨** (`engines/mc/first_passage.py`)
$$\text{LONG: }TP=P_0(1+tp),\;SL=P_0(1-sl)$$
$$\text{SHORT: }TP=P_0(1-tp),\;SL=P_0(1+sl)$$

**이벤트 수익률** (`engines/mc/first_passage.py`)
- TP hit: $R=tp/sl$
- SL hit: $R=-1$
- Timeout(옵션): mark-to-market

---

## 13) OU 모델 기반 Top-K 생존 확률 (B-1)
**OU 동학** (`engines/mc/decision.py`)
$$x_{t+1}=a x_t+(1-a)\mu+\varepsilon_t,\quad a=e^{-\kappa dt}$$
$$\kappa=-\ln(a)/dt$$

**공분산 시간 변환** (`engines/mc/decision.py`)
$$Q_{ij}(dt)=D_{ij}\frac{1-e^{-(k_i+k_j)dt}}{k_i+k_j}$$

**EVPH** (`engines/mc/decision.py`)
$$\text{EVPH}=\frac{EV}{\text{hold\_hours}}$$

**Top-K 생존 확률 근사** (`engines/mc/decision.py`)
$$\sigma_T=\sigma_{base}\sqrt{\max(1,T)/3600},\quad z=\frac{\Delta}{\sigma_T},\quad p=\Phi(z)$$

---

## 14) 시그널 알파(모멘텀/OFI/레짐)
**로그수익 기반 모멘텀 연율** (`engines/mc/signal_features.py`)
$$\mu_{mom}\approx \frac{\ln(p_t/p_{t-w})}{\tau}\cdot \text{SECONDS\_PER\_YEAR}$$
**Kaufman 효율비율(Chop Score)**
$$ER=\frac{|\ln p_t-\ln p_{t-w}|}{\sum |\Delta\ln p|},\quad chop=1-ER$$
**레짐 스케일**
$$\text{scale}=1-(1-\text{scale\_min})\cdot chop$$
**알파 결합**
$$\mu_{\alpha}=\text{scale}\cdot(w_{mom}\mu_{mom}+w_{ofi}\mu_{ofi})$$

---

## 15) 기타 핵심 규칙
- **TP/SL 자동 스케일** (`engines/mc/monte_carlo_engine.py`): $\text{tp/sl} \propto \sqrt{h}\cdot(\sigma/\sigma_{ref})$
- **EVPH 생존 확률 적용** (`engines/mc/decision.py`): $EVPH_{adj}=EVPH\cdot p_{survive}$

---

## 참고 소스(직접 구현 파일)
- `core/multi_timeframe_scoring.py`
- `core/orchestrator.py`
- `core/napv_engine_jax.py`
- `core/portfolio_management.py`
- `engines/probability_methods.py`
- `engines/cvar_methods.py`
- `engines/kelly_allocator.py`
- `engines/mc/path_simulation.py`
- `engines/mc/first_passage.py`
- `engines/mc/entry_evaluation.py`
- `engines/mc/entry_evaluation_clean.py`
- `engines/mc/jax_backend.py`
- `engines/mc/signal_features.py`
- `engines/mc/policy_weights.py`
- `engines/mc/execution_costs.py`
- `engines/mc/execution_mix.py`
- `engines/mc/leverage_optimizer_jax.py`
- `engines/mc/leverage_optimizer_torch.py`
- `engines/mc/decision.py`
- `engines/mc/exit_policy.py`

## Config 핵심 파라미터 (요약)
아래 파라미터들은 `config.py`에 명시적으로 정의되어 있으며, 런타임에서는 환경변수로 덮어쓸 수 있습니다. 중요 파라미터와 기본값, 간단 설명은 다음과 같습니다.

- `MC_N_PATHS_LIVE`: 기본값 `16384` — 라이브(운영) 몬테카를로 경로 수 (생산용 높은 해상도).
- `MC_N_PATHS_EXIT`: 기본값 `4096` — 종료(출구) 정책 평가에 사용하는 경로 수.
- `MC_TAIL_MODE`: 기본값 `"student_t"` — 몬테카를로 리턴 분포 꼬리 모델 선택 (`gaussian`/`student_t`/`bootstrap`).
- `MC_STUDENT_T_DF`: 기본값 `6.0` — `student_t` 꼬리 모드의 자유도(df).
- `MC_HORIZONS_SEC`: 기본값 `""` — MC에서 사용할 호라이즌(초) CSV 문자열; 비어있으면 코드 내 기본 목록 사용.
- `EVPH_B1_ENABLE`: 기본값 `True` — B-1 결합 EVPH 스코어 사용 여부.
- `EVPH_B1_N_PATHS`: 기본값 `4096` — EVPH B-1 평가용 샘플 경로 수.
- `TOPK_SURVIVE_ENABLED`: 기본값 `True` — Top-K 생존 필터 사용 여부 (포트폴리오 선정).
- `KELLY_PORTFOLIO_ENABLE`: 기본값 `True` — Kelly 기반 포트폴리오 사이징 사용 여부.
- `KELLY_PORTFOLIO_TOPK`: 기본값 `4` — Kelly 포트폴리오에서 고려할 상위 종목 수.
- `RISK_MARGIN_RATIO_GUARD`: 기본값 `0.80` — 마진 비율(used/total) 경고/가드 임계값.
- `RISK_INSUFFICIENT_MARGIN_COOLDOWN_MS`: 기본값 `30000` — 마진 부족 감지시의 쿨다운(밀리초).
- `RISK_MAX_TOTAL_LEVERAGE`: 기본값 `10.0` — 계정 전체 허용 최대 레버리지(근사값, 설정에 따라 결정).
- `PAPER_TRADING`: 기본값 `True` — 페이퍼(모의) 트레이딩 모드 사용 여부.
- `PAPER_SIZE_FRAC`: 기본값 `DEFAULT_SIZE_FRAC` — 페이퍼 모드 기본 포지션 크기 비율(기본값은 `config.DEFAULT_SIZE_FRAC`).
- `PAPER_LEVERAGE`: 기본값 `DEFAULT_LEVERAGE` — 페이퍼 모드 기본 레버리지(기본값은 `config.DEFAULT_LEVERAGE`).
- `BYBIT_API_KEY`: 기본값 `""` — Bybit API 키 (비밀값; 가능한 경우 환경주입/시크릿 매니저 사용 권장).
- `BYBIT_API_SECRET`: 기본값 `""` — Bybit API 시크릿 (비밀값).
- `JAX_COMPILATION_CACHE_DIR`: 기본값 `None` — JAX 컴파일 캐시 디렉토리(성능 목적).
- `MC_WARMUP_MODE`: 기본값 `"mini"` — MC 워밍업 모드 (`mini`/`full` 등).
- `MC_WARMUP_SYMBOLS`: 기본값 `None` — 워밍업에 사용할 종목 리스트(쉼표 구분 문자열).

권장: 민감한 값(BYBIT API 키/시크릿 등)은 레포지토리에 직접 저장하지 말고 `state/bybit.env` 또는 런타임 환경변수/시크릿 매니저로 주입하십시오.

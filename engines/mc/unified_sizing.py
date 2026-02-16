"""
Unified Position Sizing & Leverage Engine
==========================================
기존의 11단계 레버리지 + 4중 deleveraging + 120+ 환경변수를 
단일 패스 카운터팩추얼 최적화 시스템으로 대체.

Architecture:
  1. Feature Extraction  → 시장 상태를 0~1 정규화
  2. Quality Gate         → 최소 품질 미달 시 즉시 거부
  3. Counterfactual Grid  → MC-결과 기반 utility 최적화
  4. Exchange Constraints → 거래소 한도 적용

Inputs:
  - MC 결과: ev, cvar, p_tp, p_sl, sigma
  - 시장 상태: vpin, confidence, hurst, regime, spread
  - 포트폴리오: balance, open_exposure, position_count

Outputs:
  - leverage (float), notional (float), qty (float)
  - meta (dict) — 진단 정보

총 환경변수: ~25개 (기존 120+개 → 80% 감소)
"""

from __future__ import annotations
import math
import os
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration (25 variables — 기존 120+ 대비 80% 축소)
# ──────────────────────────────────────────────

def _env_float(key: str, default: float) -> float:
    try:
        v = os.environ.get(key)
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def _env_int(key: str, default: int) -> int:
    try:
        v = os.environ.get(key)
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def _env_bool(key: str, default: bool) -> bool:
    try:
        v = os.environ.get(key)
        if v is None:
            return default
        return str(v).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return default


@dataclass
class UnifiedSizingConfig:
    """통합 사이징 설정. 환경변수로 오버라이드 가능."""

    # === Leverage 범위 === [2026-02-14 MC E안: 상향]
    lev_global_min: float = field(default_factory=lambda: _env_float("UNI_LEV_MIN", 1.0))
    lev_global_max: float = field(default_factory=lambda: _env_float("UNI_LEV_MAX", 20.0))

    # Regime별 상한 [2026-02-14 MC E안: trend 16, chop 10, bear 12, volatile 6]
    lev_max_trend: float = field(default_factory=lambda: _env_float("UNI_LEV_MAX_TREND", 16.0))
    lev_max_chop: float = field(default_factory=lambda: _env_float("UNI_LEV_MAX_CHOP", 10.0))
    lev_max_volatile: float = field(default_factory=lambda: _env_float("UNI_LEV_MAX_VOLATILE", 6.0))
    lev_max_bear: float = field(default_factory=lambda: _env_float("UNI_LEV_MAX_BEAR", 12.0))

    # === 카운터팩추얼 최적화 ===
    cf_n_candidates: int = field(default_factory=lambda: _env_int("UNI_CF_N_CANDIDATES", 15))
    cf_risk_lambda: float = field(default_factory=lambda: _env_float("UNI_CF_RISK_LAMBDA", 0.55))
    cf_equity_base: float = field(default_factory=lambda: _env_float("UNI_CF_EQUITY_BASE", 500.0))
    cf_equity_pow: float = field(default_factory=lambda: _env_float("UNI_CF_EQUITY_POW", 0.30))

    # === Quality Gate ===
    min_edge_to_enter: float = field(default_factory=lambda: _env_float("UNI_MIN_EDGE", -0.0005))
    min_confidence: float = field(default_factory=lambda: _env_float("UNI_MIN_CONFIDENCE", 0.25))
    max_vpin_hard: float = field(default_factory=lambda: _env_float("UNI_MAX_VPIN_HARD", 0.95))

    # === Position Sizing === [2026-02-14 MC E안: frac 상향]
    max_position_frac: float = field(default_factory=lambda: _env_float("UNI_MAX_POS_FRAC", 0.40))
    max_position_frac_chop: float = field(default_factory=lambda: _env_float("UNI_MAX_POS_FRAC_CHOP", 0.25))
    max_position_frac_volatile: float = field(default_factory=lambda: _env_float("UNI_MAX_POS_FRAC_VOLATILE", 0.15))
    hard_notional_cap: float = field(default_factory=lambda: _env_float("UNI_HARD_NOTIONAL_CAP", 500.0))
    min_notional_floor: float = field(default_factory=lambda: _env_float("UNI_MIN_NOTIONAL_FLOOR", 0.5))
    sizing_base_wallet_frac: float = field(default_factory=lambda: _env_float("UNI_SIZING_BASE_WALLET_FRAC", 0.15))
    sizing_base_wallet_frac_high: float = field(default_factory=lambda: _env_float("UNI_SIZING_BASE_WALLET_FRAC_HIGH", 0.45))
    high_exposure_score_threshold: float = field(default_factory=lambda: _env_float("UNI_HIGH_EXPOSURE_SCORE_THRESHOLD", 0.81))
    high_exposure_score_threshold_bull: float = field(default_factory=lambda: _env_float("UNI_HIGH_EXPOSURE_SCORE_THRESHOLD_BULL", 0.60))
    high_exposure_score_threshold_chop: float = field(default_factory=lambda: _env_float("UNI_HIGH_EXPOSURE_SCORE_THRESHOLD_CHOP", 0.81))
    high_exposure_score_threshold_bear: float = field(default_factory=lambda: _env_float("UNI_HIGH_EXPOSURE_SCORE_THRESHOLD_BEAR", 0.95))
    high_exposure_score_threshold_volatile: float = field(default_factory=lambda: _env_float("UNI_HIGH_EXPOSURE_SCORE_THRESHOLD_VOLATILE", 0.85))
    high_exposure_pos_frac_mult: float = field(default_factory=lambda: _env_float("UNI_HIGH_EXPOSURE_POS_FRAC_MULT", 1.35))

    # === Risk Budget ===
    max_single_loss_pct: float = field(default_factory=lambda: _env_float("UNI_MAX_SINGLE_LOSS_PCT", 0.03))
    max_total_exposure: float = field(default_factory=lambda: _env_float("UNI_MAX_TOTAL_EXPOSURE", 5.0))

    # === K constant (EV→Leverage 변환) ===
    k_lev: float = field(default_factory=lambda: _env_float("K_LEV", 2000.0))

    # === Liquidation safety ===
    maint_margin: float = field(default_factory=lambda: _env_float("UNI_MAINT_MARGIN", 0.005))
    liq_buffer: float = field(default_factory=lambda: _env_float("UNI_LIQ_BUFFER", 0.0025))
    min_liq_distance: float = field(default_factory=lambda: _env_float("UNI_MIN_LIQ_DIST", 0.015))

    # === SQ Probation ===
    sq_probation_cap: float = field(default_factory=lambda: _env_float("UNI_SQ_PROBATION_CAP", 2.0))


# ──────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────

def _signal_quality(
    confidence: float,
    vpin: float,
    hurst: float,
    sigma: float,
    regime: str,
) -> float:
    """
    시장 상태를 종합한 단일 품질 점수 (0.0 ~ 1.0).
    
    세 축:
      - Directional clarity: confidence × (1 - vpin) — 방향 신뢰 × 비독성
      - Trend persistence:   hurst — 트렌드 지속력  
      - Volatility suitability: 적당한 vol이 최적 (너무 낮거나 높으면 감점)
    """
    # Directional clarity (0~1)
    dir_clarity = max(0.0, min(1.0, confidence)) * max(0.0, min(1.0, 1.0 - vpin))

    # Trend persistence (0~1)
    trend_pers = max(0.0, min(1.0, hurst))

    # Vol suitability: bell curve — sigma=0.3~0.8 optimal, too low or too high penalized
    sigma_clamped = max(0.01, min(5.0, sigma))
    vol_suit = max(0.0, 1.0 - abs(math.log(sigma_clamped / 0.5)) / 2.5)

    # Regime weight adjustment
    if regime in ("trend", "bull"):
        # Trend: 방향 신뢰 중시, 변동성 허용
        quality = 0.45 * dir_clarity + 0.35 * trend_pers + 0.20 * vol_suit
    elif regime == "bear":
        # Bear: 방향 정확성 필수
        quality = 0.50 * dir_clarity + 0.30 * trend_pers + 0.20 * vol_suit
    elif regime == "volatile":
        # Volatile: 변동성 적합도 중시
        quality = 0.35 * dir_clarity + 0.25 * trend_pers + 0.40 * vol_suit
    else:
        # Chop: 보수적 — 모든 축 균등
        quality = 0.40 * dir_clarity + 0.30 * trend_pers + 0.30 * vol_suit

    return float(max(0.0, min(1.0, quality)))


def _regime_max_leverage(regime: str, cfg: UnifiedSizingConfig) -> float:
    """레짐별 레버리지 상한."""
    regime_map = {
        "trend": cfg.lev_max_trend,
        "bull": cfg.lev_max_trend,
        "bear": cfg.lev_max_bear,
        "chop": cfg.lev_max_chop,
        "volatile": cfg.lev_max_volatile,
    }
    return float(min(cfg.lev_global_max, regime_map.get(regime, cfg.lev_max_chop)))


def _max_pos_frac_for_regime(regime: str, cfg: UnifiedSizingConfig) -> float:
    """레짐별 최대 포지션 비율."""
    if regime in ("volatile",):
        return cfg.max_position_frac_volatile
    elif regime in ("chop",):
        return cfg.max_position_frac_chop
    else:
        return cfg.max_position_frac


def _liquidation_cap(cfg: UnifiedSizingConfig) -> float:
    """청산 거리 기반 레버리지 상한."""
    denom = cfg.maint_margin + cfg.liq_buffer + cfg.min_liq_distance
    return float(max(1.0, 1.0 / max(1e-9, denom)))


def counterfactual_optimal_leverage(
    *,
    ev: float,
    cvar: float,
    p_sl: float,
    sigma: float,
    vpin: float,
    confidence: float,
    hurst: float,
    spread: float,
    fee_rt: float,
    regime: str,
    balance: float,
    quality: float,
    lev_min: float,
    lev_max: float,
    cfg: UnifiedSizingConfig,
    p_tp: float = 0.5,
    tp_pct: float | None = None,
    sl_pct: float | None = None,
) -> tuple[float, dict]:
    """
    카운터팩추얼 유틸리티 최적화로 최적 레버리지를 결정.

    Mean-Variance Utility with strategy-specific variance:
      U(L) = EV × L - cost(L) - 0.5 × λ × σ²_strategy × L²

    σ²_strategy는 TP/SL 바운드된 리턴 분산을 사용 (시장 σ 대신):
      σ²_strat = p_tp × TP² + p_sl × SL² - E[R]²
    이렇게 하면 risk penalty가 실제 전략 리스크에 비례하여
    EV > 0인 경우 레버리지가 적절히 증가함.

    Returns: (optimal_leverage, diagnostic_dict)
    """
    n = cfg.cf_n_candidates
    lam = cfg.cf_risk_lambda

    # Equity scaling: larger accounts → slightly more conservative leverage
    eq_scale = 1.0
    if balance > 0 and cfg.cf_equity_base > 0:
        ratio = max(1e-6, balance / cfg.cf_equity_base)
        eq_scale = max(0.7, min(1.3, ratio ** (-cfg.cf_equity_pow)))

    if lev_max <= lev_min + 1e-9:
        return lev_min, {"cf_method": "degenerate", "lev_range": (lev_min, lev_max)}

    grid = np.linspace(lev_min, lev_max, max(5, n))

    # Quality-adjusted EV
    # [2026-02-15 CF] quality 증폭: 기존 (0.5 + q) → (0.3 + q * 1.4)
    # quality=0.20 → 0.58x, quality=0.35 → 0.79x, quality=0.50 → 1.00x, quality=0.80 → 1.42x
    # 효과: 높은 quality일 때 optimizer가 더 높은 레버리지 선택
    quality_adj_ev = ev * (0.3 + quality * 1.4)

    base_cost = max(0.0, fee_rt + spread)

    # Strategy-specific variance (TP/SL bounded returns)
    # Use actual TP/SL from config (not scaled from EV)
    import os as _os
    if tp_pct is None:
        tp_pct = float(_os.environ.get("DEFAULT_TP_PCT", "0.006"))
    if sl_pct is None:
        sl_pct = float(_os.environ.get("DEFAULT_SL_PCT", "0.004"))
    tp_pct = max(0.002, min(0.10, tp_pct))   # clamp 0.2% ~ 10%
    sl_pct = max(0.002, min(0.10, sl_pct))   # clamp 0.2% ~ 10%
    p_tp_c = max(0.01, min(0.99, p_tp))
    p_sl_c = max(0.01, min(0.99, p_sl))
    e_ret = p_tp_c * tp_pct - p_sl_c * sl_pct - base_cost
    var_strat = p_tp_c * tp_pct**2 + p_sl_c * sl_pct**2 - e_ret**2
    var_strat = max(1e-10, var_strat)

    # CVaR penalty (보조적: 극단 꼬리 리스크만)
    # 완화: 임계 0.05→0.12, 계수 0.1→0.03 (과거 EV 지배 방지)
    cvar_penalty = max(0.0, abs(cvar) - 0.12) * 0.03  # CVaR > 12%일 때만

    best_lev = lev_min
    best_u = -1e18
    utilities = []

    for lev_c in grid:
        lev_c = float(max(lev_min, lev_c))

        # EV term
        ev_term = quality_adj_ev * lev_c

        # Cost term (slight increase with leverage due to market impact)
        cost_term = base_cost * (1.0 + 0.03 * max(0.0, lev_c - 1.0))

        # Risk term: mean-variance with strategy variance
        risk_term = 0.5 * lam * var_strat * lev_c ** 2

        # CVaR tail penalty (mild)
        tail_pen = cvar_penalty * lev_c

        # Concentration penalty
        conc_pen = max(0.0, (lev_c / max(1.0, lev_max)) - 0.85) ** 2 * 0.02

        utility = (ev_term - cost_term - risk_term - tail_pen - conc_pen) * eq_scale

        utilities.append(float(utility))
        if utility > best_u:
            best_u = float(utility)
            best_lev = float(lev_c)

    # Band: leverage candidates within 15% of best utility
    band_tol = 0.15
    band_floor = best_u - abs(best_u) * band_tol
    in_band = [float(grid[i]) for i, u in enumerate(utilities) if u >= band_floor]
    if not in_band:
        in_band = [best_lev]

    meta = {
        "cf_method": "grid_mv_utility",
        "cf_best_lev": float(best_lev),
        "cf_best_utility": float(best_u),
        "cf_band_min": float(min(in_band)),
        "cf_band_max": float(max(in_band)),
        "cf_quality_adj_ev": float(quality_adj_ev),
        "cf_var_strat": float(var_strat),
        "cf_e_ret": float(e_ret),
        "cf_eq_scale": float(eq_scale),
        "cf_risk_lambda": float(lam),
        "cf_n_candidates": int(n),
    }

    return float(best_lev), meta


def counterfactual_optimal_notional(
    *,
    leverage: float,
    ev: float,
    cvar: float,
    p_sl: float,
    sigma: float,
    vpin: float,
    confidence: float,
    quality: float,
    regime: str,
    balance: float,
    open_exposure: float,
    max_total_exposure: float,
    kelly_frac: float | None,
    cfg: UnifiedSizingConfig,
    exposure_base: float | None = None,
    exposure_score: float | None = None,
) -> tuple[float, dict]:
    """
    카운터팩추얼 최적 포지션 크기(notional) 결정.
    
    세 가지 기준의 최소값:
    1. Quality-scaled fraction: balance × quality × max_pos_frac × leverage
    2. Risk budget: balance × max_single_loss / sl_pct (SL 거리 기반)
    3. Remaining exposure: (max_total_exposure × exposure_base - open_exposure)
       exposure_base: wallet_balance(총자산). balance(free_balance)와 구분.
       margin이 잠겨도 총자산 기준으로 exposure 한도를 계산해야 정확.
    
    Returns: (notional, meta)
    """
    max_frac = _max_pos_frac_for_regime(regime, cfg)

    # ── RegimePolicy 적용 ──
    try:
        from engines.mc.regime_policy import get_regime_policy
    except ImportError:
        from .regime_policy import get_regime_policy
    _rpol = get_regime_policy(regime, sigma)
    # Regime policy의 max_pos_frac가 더 보수적이면 적용
    max_frac = min(max_frac, float(_rpol.max_pos_frac))

    # [2026-02-15] exposure_base: wallet_balance 기반으로 exposure 한도 계산
    # 기존: free_balance(margin 차감 후) × 5.0 → 포지션 늘수록 한도 급감
    # 수정: wallet_balance(총자산) × 5.0 → 총자산 기준 안정적 한도
    _exposure_base = exposure_base if (exposure_base is not None and exposure_base > 0) else balance

    # [2026-02-15] 사이징 기반도 wallet_balance 사용
    # 이유: free_balance가 $1~$5일 때 quality_notional/risk_notional이 0에 수렴
    #       → min_floor($0.5)보다 작아서 모든 진입이 거부됨
    # 안전장치: max_total_exposure + max_pos_frac가 총 노출을 제한
    score_for_exposure = quality if exposure_score is None else exposure_score
    score_for_exposure = float(max(0.0, min(1.0, score_for_exposure)))

    reg = str(regime or "").lower()
    if reg in ("bull", "trend"):
        exposure_thr = float(cfg.high_exposure_score_threshold_bull)
    elif reg == "chop":
        exposure_thr = float(cfg.high_exposure_score_threshold_chop)
    elif reg == "bear":
        exposure_thr = float(cfg.high_exposure_score_threshold_bear)
    elif reg == "volatile":
        exposure_thr = float(cfg.high_exposure_score_threshold_volatile)
    else:
        exposure_thr = float(cfg.high_exposure_score_threshold)

    high_exposure_mode = bool(score_for_exposure >= exposure_thr)
    wallet_frac = float(cfg.sizing_base_wallet_frac_high if high_exposure_mode else cfg.sizing_base_wallet_frac)
    wallet_frac = float(max(0.05, min(1.00, wallet_frac)))

    _sizing_base = max(balance, _exposure_base * wallet_frac)

    # 1. Quality-scaled position
    # quality: 0~1, higher → larger position
    max_frac_eff = float(max_frac)
    if high_exposure_mode:
        max_frac_eff = float(min(0.95, max_frac_eff * max(1.0, float(cfg.high_exposure_pos_frac_mult))))

    quality_notional = _sizing_base * quality * max_frac_eff * max(1.0, leverage)

    # 2. Risk budget: max single loss / SL distance (not probability!)
    # [2026-02-14 FIX] p_sl은 SL "확률"이지 SL "거리"가 아님.
    # SL 거리(sl_pct)를 사용하여: notional × sl_pct = _sizing_base × max_single_loss_pct
    # → notional = _sizing_base × max_single_loss_pct / sl_pct
    sl_pct = max(0.002, float(_rpol.sl_pct))  # regime 정책 SL 사용
    risk_notional = _sizing_base * cfg.max_single_loss_pct / sl_pct

    # 3. Remaining exposure in portfolio — wallet_balance 기준!
    remaining = max(0.0, max_total_exposure * _exposure_base - open_exposure)
    exposure_notional = remaining

    # 4. Kelly fraction (soft scaling, not hard cap)
    # Kelly가 매우 작으면(음수 EV 등) hard cap으로 쓰면 진입 자체가 불가.
    # Kelly를 soft scale 팩터로 사용: notional에 max(0.5, kelly_weight)를 곱함.
    kelly_scale = 1.0
    if kelly_frac is not None and kelly_frac > 0:
        # kelly_frac 범위: 보통 0.001~0.1
        # scale: kelly_frac=0.01일 때 0.6, kelly_frac=0.05일 때 1.0
        kelly_scale = max(0.5, min(1.5, kelly_frac * 20.0))

    # Take minimum of quality, risk, exposure constraints
    notional = min(quality_notional, risk_notional, exposure_notional)

    # Apply kelly as soft scale (not hard cap)
    notional *= kelly_scale

    # Hard cap
    notional = min(notional, cfg.hard_notional_cap)

    # Floor
    notional = max(0.0, notional)

    meta = {
        "quality_notional": float(quality_notional),
        "risk_notional": float(risk_notional),
        "exposure_notional": float(exposure_notional),
        "exposure_score": float(score_for_exposure),
        "high_exposure_mode": bool(high_exposure_mode),
        "high_exposure_threshold": float(exposure_thr),
        "sizing_base_wallet_frac": float(wallet_frac),
        "sizing_base": float(_sizing_base),
        "kelly_scale": float(kelly_scale),
        "kelly_frac": float(kelly_frac) if kelly_frac is not None else None,
        "binding_constraint": "quality" if notional >= quality_notional * kelly_scale - 0.01
            else "risk" if notional >= risk_notional * kelly_scale - 0.01
            else "exposure",
        "max_pos_frac": float(max_frac),
        "max_pos_frac_eff": float(max_frac_eff),
    }

    return float(notional), meta


def compute_unified_sizing(
    *,
    # MC Results
    ev: float,
    cvar: float,
    p_tp: float,
    p_sl: float,

    # Market State
    sigma: float,
    vpin: float,
    confidence: float,
    hurst: float,
    regime: str,
    spread: float,
    fee_rt: float,

    # Portfolio State
    balance: float,
    open_exposure: float,
    position_count: int,
    price: float,

    # Exposure base (wallet_balance, separate from free_balance)
    exposure_base: float | None = None,
    exposure_score: float | None = None,

    # Kelly (optional)
    kelly_frac: float | None = None,

    # Overrides
    sq_probation: bool = False,
    cfg: UnifiedSizingConfig | None = None,
) -> dict:
    """
    통합 포지션 사이징 — 단일 호출로 leverage + notional + qty 결정.
    
    Returns dict with keys:
      leverage, notional, qty, quality, meta
    """
    if cfg is None:
        cfg = UnifiedSizingConfig()

    meta: dict[str, Any] = {}

    # ── Step 1: Signal Quality (0~1) ──
    quality = _signal_quality(confidence, vpin, hurst, sigma, regime)
    meta["quality"] = float(quality)

    # ── Step 2: Quality Gate ──
    if quality < cfg.min_confidence:
        meta["gate_reject"] = "low_quality"
        logger.warning(f"[UNI_SIZING] gate_reject=low_quality quality={quality:.4f} < min={cfg.min_confidence:.4f} conf={confidence:.4f} vpin={vpin:.4f} hurst={hurst:.4f} sigma={sigma:.4f}")
        return {"leverage": 1.0, "notional": 0.0, "qty": 0.0, "quality": quality, "meta": meta}

    if vpin >= cfg.max_vpin_hard:
        meta["gate_reject"] = "high_vpin"
        logger.warning(f"[UNI_SIZING] gate_reject=high_vpin vpin={vpin:.4f} >= max={cfg.max_vpin_hard:.4f}")
        return {"leverage": 1.0, "notional": 0.0, "qty": 0.0, "quality": quality, "meta": meta}

    # ── Step 3: Regime-based leverage cap ──
    # RegimePolicy에서 일관된 레버리지/포지션/TP/SL 정책 가져오기
    try:
        from engines.mc.regime_policy import get_regime_policy
    except ImportError:
        from .regime_policy import get_regime_policy
    _rpol = get_regime_policy(regime, sigma)

    regime_max = min(float(_rpol.max_leverage), _regime_max_leverage(regime, cfg))
    liq_cap = _liquidation_cap(cfg)
    lev_max = min(regime_max, liq_cap, cfg.lev_global_max)

    # SQ probation override
    if sq_probation:
        lev_max = min(lev_max, cfg.sq_probation_cap)
        meta["sq_probation"] = True

    lev_min = cfg.lev_global_min

    # ── Step 4: Counterfactual Optimal Leverage (regime-adaptive) ──
    # RegimePolicy의 TP/SL과 cf_risk_lambda 사용
    _tp_pct = float(_rpol.tp_pct)
    _sl_pct = float(_rpol.sl_pct)
    # Regime-adaptive CF risk lambda: 환경변수 → regime policy → config fallback
    _cf_lam_override = _rpol.cf_risk_lambda
    _orig_cf_lam = cfg.cf_risk_lambda
    cfg.cf_risk_lambda = _cf_lam_override  # 임시 override
    opt_lev, lev_meta = counterfactual_optimal_leverage(
        ev=ev, cvar=cvar, p_sl=p_sl, sigma=sigma,
        vpin=vpin, confidence=confidence, hurst=hurst,
        spread=spread, fee_rt=fee_rt, regime=regime,
        balance=balance, quality=quality,
        lev_min=lev_min, lev_max=lev_max, cfg=cfg,
        p_tp=p_tp, tp_pct=_tp_pct, sl_pct=_sl_pct,
    )
    cfg.cf_risk_lambda = _orig_cf_lam  # 복원
    lev_meta["regime_policy_cf_lambda"] = float(_cf_lam_override)
    lev_meta["regime_policy_tp_pct"] = float(_tp_pct)
    lev_meta["regime_policy_sl_pct"] = float(_sl_pct)
    meta.update(lev_meta)

    # Final leverage clamp
    leverage = max(lev_min, min(lev_max, opt_lev))
    meta["leverage_final"] = float(leverage)
    meta["lev_min"] = float(lev_min)
    meta["lev_max"] = float(lev_max)
    meta["regime_max"] = float(regime_max)
    meta["liq_cap"] = float(liq_cap)

    # ── Step 5: Counterfactual Optimal Notional ──
    max_total = cfg.max_total_exposure
    notional, size_meta = counterfactual_optimal_notional(
        leverage=leverage, ev=ev, cvar=cvar, p_sl=p_sl,
        sigma=sigma, vpin=vpin, confidence=confidence,
        quality=quality, regime=regime, balance=balance,
        open_exposure=open_exposure, max_total_exposure=max_total,
        kelly_frac=kelly_frac, cfg=cfg,
        exposure_base=exposure_base,
        exposure_score=exposure_score,
    )
    meta.update(size_meta)

    # ── Step 6: Floor check ──
    if notional < cfg.min_notional_floor:
        meta["size_reject"] = "below_min_notional"
        logger.warning(f"[UNI_SIZING] size_reject=below_min_notional notional={notional:.4f} < floor={cfg.min_notional_floor:.4f} quality={quality:.4f} lev={leverage:.2f} balance={balance:.2f}")
        return {"leverage": leverage, "notional": 0.0, "qty": 0.0, "quality": quality, "meta": meta}

    # ── Step 7: Quantity ──
    qty = notional / price if price > 0 else 0.0
    size_frac = notional / (balance * max(1.0, leverage)) if balance > 0 and leverage > 0 else 0.0

    meta["size_frac"] = float(size_frac)
    meta["notional_final"] = float(notional)

    logger.info(f"[UNI_SIZING] OK notional={notional:.2f} qty={qty:.6f} lev={leverage:.2f} quality={quality:.4f} balance={balance:.2f} price={price:.2f}")

    return {
        "leverage": float(leverage),
        "notional": float(notional),
        "qty": float(qty),
        "quality": float(quality),
        "size_frac": float(size_frac),
        "meta": meta,
    }


# ──────────────────────────────────────────────
# Convenience: 기존 인터페이스 호환 래퍼
# ──────────────────────────────────────────────

_DEFAULT_CFG: UnifiedSizingConfig | None = None

def get_config() -> UnifiedSizingConfig:
    """싱글톤 설정 (환경변수 재로드 시 새로 생성)."""
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        _DEFAULT_CFG = UnifiedSizingConfig()
    return _DEFAULT_CFG

def reset_config():
    """환경변수 변경 후 설정 리로드."""
    global _DEFAULT_CFG
    _DEFAULT_CFG = None

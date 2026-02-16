"""
Regime-Adaptive Policy Engine
==============================
중앙집중식 레짐별 파라미터 관리.  모든 진입/청산/레버리지/자본 배분 전략이
이 모듈의 `get_regime_policy(regime)` 로 반환되는 `RegimePolicy` 에 기반한다.

설계 원칙:
  1. 단일 소스 (Single Source of Truth) — 레짐별 수치가 여기서만 정의됨
  2. 환경변수 오버라이드 — 모든 필드를 env 로 런타임 변경 가능
  3. 변동성 정규화 — σ_ref 기반으로 objective/threshold/TP/SL 스케일 자체를 시장에 맞춤
"""
from __future__ import annotations

import math
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _ef(name: str, default: float) -> float:
    """Env float helper."""
    try:
        v = os.environ.get(name)
        if v is None or str(v).strip() == "":
            return float(default)
        return float(v)
    except (ValueError, TypeError):
        return float(default)


def _eb(name: str, default: bool) -> bool:
    """Env bool helper."""
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ────────────────────────────────────────────────────────────
#  RegimePolicy dataclass
# ────────────────────────────────────────────────────────────
@dataclass
class RegimePolicy:
    """하나의 레짐에 대한 전 전략 파라미터 묶음."""

    regime: str = "chop"

    # ── Objective Function ──
    objective_mode: str = "ratio_time_var"   # ratio_time | ev_var | new_objective | ratio_time_var
    lambda_var: float = 2.0                  # variance penalty (큰 값 = 보수적)
    risk_lambda: float = 1.0                 # unified Ψ risk lambda
    use_gross_score: bool = False            # True면 비용 제외 gross 기반 점수
    sigma_ref: float = 0.50                  # 변동성 정규화 기준 σ (연율)

    # ── Entry Thresholds ──
    entry_floor: float = -0.0001             # unified_score 최저선
    entry_floor_add: float = 0.0             # regime-specific 추가 floor
    score_entry_threshold: float = 0.0003    # horizon-adaptive threshold base
    score_entry_floor: float = 0.00015       # horizon-adaptive threshold floor
    min_ev_gap: float = 0.0002               # long-short gap 최소 요구
    net_expectancy_min: float = 0.0          # net EV 하한
    chop_guard_mu_min: float = 0.2           # chop guard |mu_alpha| 최소
    chop_guard_conf_min: float = 0.55        # chop guard dir_conf 최소

    # ── TP/SL (가격 기준 %) ──
    tp_pct: float = 0.025                    # take profit %
    sl_pct: float = 0.018                    # stop loss %
    tp_sl_ratio: float = 1.4                 # TP/SL 비율 (정보 전용)
    tp_base_roe: float = 0.0015              # MC 내부 TP ROE
    sl_base_roe: float = 0.0020              # MC 내부 SL ROE

    # ── Exit ──
    min_hold_sec: float = 120.0              # 최소 보유 시간 (초)
    dd_regime_mult: float = 1.0              # unrealized DD multiplier
    trailing_atr_mult: float = 2.0           # trailing stop ATR 배수
    max_hold_sec: float = 0.0               # 최대 보유 시간 (0 = 무제한)

    # ── Leverage ──
    max_leverage: float = 16.0               # 레짐별 레버리지 상한
    cf_risk_lambda: float = 0.20             # CF utility λ_risk
    cf_quality_base: float = 0.3             # quality 기반 EV 보정 기저

    # ── Capital / Position Sizing ──
    max_pos_frac: float = 0.40               # 잔고 대비 최대 포지션
    high_exposure_score_threshold: float = 0.60  # 고노출 시 필요 점수

    # ── Spread & Risk Filter ──
    spread_cap: float = 0.0020               # 최대 스프레드 허용
    cvar_floor: float = -1.2                 # CVaR 하한
    funnel_win_floor: float = 0.50           # 승률 하한
    funnel_event_cvar_floor: float = -1.25   # event CVaR 하한

    # ── Consensus / Neighbor ──
    local_consensus_alpha: float = 0.3       # 이웃 horizon 보정 강도

    # ── Direction Gate ──
    allow_counter_trend: bool = True         # regime 반대 방향 진입 허용

    # ── MC Simulation ──
    mc_n_paths: int = 24576                  # Monte Carlo 경로 수


def _regime_env_prefix(regime: str) -> str:
    """Regime을 대문자 환경변수 접두사로 변환."""
    r = str(regime).strip().upper()
    return {"BULL": "BULL", "BEAR": "BEAR", "CHOP": "CHOP",
            "VOLATILE": "VOLATILE", "TREND": "BULL"}.get(r, "CHOP")


def _rp_float(regime_prefix: str, base_key: str, default: float) -> float:
    """레짐별 환경변수 검색: {BASE_KEY}_{REGIME} → {BASE_KEY} → default."""
    for suffix in (f"_{regime_prefix}", ""):
        full = f"{base_key}{suffix}"
        val = os.environ.get(full)
        if val is not None and str(val).strip():
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return float(default)


def _rp_bool(regime_prefix: str, base_key: str, default: bool) -> bool:
    for suffix in (f"_{regime_prefix}", ""):
        full = f"{base_key}{suffix}"
        val = os.environ.get(full)
        if val is not None and str(val).strip():
            return str(val).strip().lower() in ("1", "true", "yes", "on")
    return default


# ────────────────────────────────────────────────────────────
#  Built-in Regime Defaults   (env 미설정 시 사용)
# ────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, dict] = {
    "bull": dict(
        objective_mode="ratio_time",      # 추세 추종: 방향성 위주
        lambda_var=0.5,                   # 낮은 분산 페널티 (추세 탈 때 허용)
        risk_lambda=0.05,                 # Ψ λ: 매우 공격적 (CVaR 패널티 최소)
        entry_floor=0.000010,             # Ψ 양수면 진입 허용
        entry_floor_add=0.0,
        score_entry_threshold=0.000040,   # 낮은 threshold (추세 진입 적극)
        score_entry_floor=0.000010,
        min_ev_gap=0.0001,                # 추세장에서는 gap 완화
        net_expectancy_min=-0.0002,       # 약간 음수도 허용 (추세 프리미엄 기대)
        tp_pct=0.035,                     # 넓은 TP (추세 라이딩)
        sl_pct=0.012,                     # 타이트한 SL (빠른 손절)
        tp_base_roe=0.0020,
        sl_base_roe=0.0015,
        min_hold_sec=60.0,                # 짧은 최소 보유 (빠른 진입 허용)
        dd_regime_mult=1.2,               # 추세장 DD 허용 더 넓게
        trailing_atr_mult=2.5,            # 넓은 trailing
        max_leverage=20.0,
        cf_risk_lambda=0.12,              # 공격적 레버리지
        max_pos_frac=0.45,
        high_exposure_score_threshold=0.50,
        spread_cap=0.0025,                # 스프레드 관대
        cvar_floor=-1.5,
        funnel_event_cvar_floor=-1.50,
        local_consensus_alpha=0.20,       # consensus 약화 (best horizon 신뢰)
        allow_counter_trend=False,        # bull에서 SHORT 차단
        mc_n_paths=16384,
    ),
    "bear": dict(
        objective_mode="new_objective",   # 리스크 조정 점수 (|CVaR|+2σ 분모)
        lambda_var=3.0,                   # 높은 분산 페널티
        risk_lambda=0.25,                 # Ψ λ: 보수적 (일부 진입만)
        entry_floor=0.000020,             # 양수 floor — 확실한 기회만
        entry_floor_add=0.000010,
        score_entry_threshold=0.000060,
        score_entry_floor=0.000020,
        min_ev_gap=0.0003,
        net_expectancy_min=0.0,           # net 양수 필수
        tp_pct=0.015,                     # 타이트한 TP (빠른 이익 실현)
        sl_pct=0.020,                     # 넓은 SL (bear rally 방어)
        tp_base_roe=0.0012,
        sl_base_roe=0.0025,
        min_hold_sec=180.0,               # 보수적 보유
        dd_regime_mult=0.8,               # 타이트한 DD
        trailing_atr_mult=1.5,
        max_leverage=12.0,
        cf_risk_lambda=0.35,              # 보수적 레버리지
        max_pos_frac=0.30,
        high_exposure_score_threshold=0.80,
        spread_cap=0.0020,
        cvar_floor=-1.0,
        funnel_event_cvar_floor=-1.10,
        local_consensus_alpha=0.35,
        allow_counter_trend=False,        # bear에서 LONG 차단
        mc_n_paths=24576,
    ),
    "chop": dict(
        objective_mode="ev_var",          # 분산 페널티 모드 (노이즈 억제)
        lambda_var=4.0,                   # 강한 분산 페널티
        risk_lambda=0.15,                 # Ψ λ: 중도 (횡보에서도 기회 포착)
        entry_floor=0.000010,             # Ψ > 0.00001 이면 진입 고려
        entry_floor_add=0.0,              # 추가 장벽 없음 (risk_lambda가 이미 필터)
        score_entry_threshold=0.000050,   # Ψ 양수 범위에 맞춘 threshold
        score_entry_floor=0.000010,
        min_ev_gap=0.0004,                # 방향 확실해야 진입
        net_expectancy_min=0.0002,        # net 확실히 양수
        tp_pct=0.010,                     # 짧은 TP (mean-reversion)
        sl_pct=0.008,                     # 짧은 SL (빠른 손절)
        tp_base_roe=0.0010,
        sl_base_roe=0.0012,
        min_hold_sec=300.0,               # 충분히 기다려야 mean-revert
        dd_regime_mult=0.7,               # 타이트한 DD
        trailing_atr_mult=1.2,
        max_leverage=8.0,                 # 보수적 레버리지
        cf_risk_lambda=0.45,
        max_pos_frac=0.25,
        high_exposure_score_threshold=0.85,
        spread_cap=0.0012,                # 타이트한 스프레드
        cvar_floor=-0.8,
        funnel_event_cvar_floor=-1.00,
        local_consensus_alpha=0.45,       # consensus 강화 (다 horizon 동의 필요)
        allow_counter_trend=True,
        mc_n_paths=32768,                 # 많은 경로 (방향 불확실)
    ),
    "volatile": dict(
        objective_mode="new_objective",   # signal reliability (|CVaR|+2σ)
        lambda_var=5.0,                   # 매우 강한 분산 페널티
        risk_lambda=0.30,                 # Ψ λ: 높음 (확실한 기회만)
        entry_floor=0.000020,             # Ψ 양수 범위
        entry_floor_add=0.000010,
        score_entry_threshold=0.000050,
        score_entry_floor=0.000020,
        min_ev_gap=0.0005,
        net_expectancy_min=0.0004,
        tp_pct=0.020,                     # 중간 TP (큰 움직임 활용)
        sl_pct=0.025,                     # 넓은 SL (whipsaw 방어)
        tp_base_roe=0.0018,
        sl_base_roe=0.0030,
        min_hold_sec=120.0,               # volatile에서도 빠른 탈출 허용
        dd_regime_mult=0.6,               # 매우 타이트한 DD
        trailing_atr_mult=3.0,            # 넓은 trailing (high vol)
        max_leverage=5.0,                 # 매우 보수적 레버리지
        cf_risk_lambda=0.55,
        max_pos_frac=0.15,
        high_exposure_score_threshold=0.90,
        spread_cap=0.0008,
        cvar_floor=-0.6,
        funnel_event_cvar_floor=-0.90,
        local_consensus_alpha=0.50,
        allow_counter_trend=True,
        mc_n_paths=32768,
    ),
}


def get_regime_policy(regime: str, realized_sigma: float = 0.5) -> RegimePolicy:
    """
    레짐 + 실현변동성에 따른 동적 정책 생성.

    Parameters
    ----------
    regime : str
        "bull", "bear", "chop", "volatile"
    realized_sigma : float
        현재 실현 연율 변동성 σ (기본 0.5 = 50%)

    Returns
    -------
    RegimePolicy
        해당 레짐에 최적화된 전 전략 파라미터
    """
    r = str(regime).strip().lower()
    if r in ("trend",):
        r = "bull"
    if r not in _DEFAULTS:
        r = "chop"

    pfx = _regime_env_prefix(r)
    d = _DEFAULTS[r]

    # ── 변동성 정규화 스케일 ──
    # σ_ref = 0.50 (기본 50% 연율 변동성)
    # σ가 높으면 threshold/TP/SL/floor 등을 비례 확대
    sigma_ref = _ef("REGIME_SIGMA_REF", 0.50)
    sigma_safe = max(float(realized_sigma), 0.01)
    vol_scale = sigma_safe / max(sigma_ref, 0.01)
    # Clamp: 0.3 ~ 3.0 range
    vol_scale = max(0.3, min(3.0, vol_scale))

    policy = RegimePolicy(
        regime=r,

        # Objective
        objective_mode=str(os.environ.get(f"POLICY_OBJECTIVE_MODE_{pfx}",
                          os.environ.get("POLICY_OBJECTIVE_MODE", d["objective_mode"]))).strip().lower(),
        lambda_var=_rp_float(pfx, "POLICY_LAMBDA_VAR", d["lambda_var"]),
        risk_lambda=_rp_float(pfx, "UNIFIED_RISK_LAMBDA", d["risk_lambda"]),
        use_gross_score=_rp_bool(pfx, "USE_GROSS_SCORE", d.get("use_gross_score", False)),
        sigma_ref=sigma_ref,

        # Entry
        entry_floor=_rp_float(pfx, "UNIFIED_ENTRY_FLOOR", d["entry_floor"]),
        entry_floor_add=_rp_float(pfx, "ENTRY_FLOOR_ADD", d["entry_floor_add"]),
        score_entry_threshold=_rp_float(pfx, "SCORE_ENTRY_THRESHOLD", d["score_entry_threshold"]),
        score_entry_floor=_rp_float(pfx, "SCORE_ENTRY_FLOOR", d["score_entry_floor"]),
        min_ev_gap=_rp_float(pfx, "POLICY_MIN_EV_GAP", d["min_ev_gap"]),
        net_expectancy_min=_rp_float(pfx, "ENTRY_NET_EXPECTANCY_MIN", d["net_expectancy_min"]),
        chop_guard_mu_min=_rp_float(pfx, "CHOP_ENTRY_MIN_MU_ALPHA", d.get("chop_guard_mu_min", 0.2)),
        chop_guard_conf_min=_rp_float(pfx, "CHOP_DIRECTION_CONF_MIN", d.get("chop_guard_conf_min", 0.55)),

        # TP/SL — 변동성 스케일 적용
        tp_pct=_rp_float(pfx, "DEFAULT_TP_PCT", d["tp_pct"]) * vol_scale,
        sl_pct=_rp_float(pfx, "DEFAULT_SL_PCT", d["sl_pct"]) * vol_scale,
        tp_sl_ratio=d.get("tp_sl_ratio", 1.4),
        tp_base_roe=_rp_float(pfx, "MC_TP_BASE_ROE", d["tp_base_roe"]) * vol_scale,
        sl_base_roe=_rp_float(pfx, "MC_SL_BASE_ROE", d["sl_base_roe"]) * vol_scale,

        # Exit
        min_hold_sec=_rp_float(pfx, "EXIT_MIN_HOLD_SEC", d["min_hold_sec"]),
        dd_regime_mult=_rp_float(pfx, "UNREALIZED_DD_REGIME_MULT", d["dd_regime_mult"]),
        trailing_atr_mult=_rp_float(pfx, "TRAIL_ATR_MULT", d["trailing_atr_mult"]),
        max_hold_sec=_rp_float(pfx, "POLICY_MAX_HOLD_SEC", d.get("max_hold_sec", 0.0)),

        # Leverage
        max_leverage=_rp_float(pfx, "UNI_LEV_MAX", d["max_leverage"]),
        cf_risk_lambda=_rp_float(pfx, "UNI_CF_RISK_LAMBDA", d["cf_risk_lambda"]),
        cf_quality_base=_rp_float(pfx, "UNI_CF_QUALITY_BASE", d.get("cf_quality_base", 0.3)),

        # Capital
        max_pos_frac=_rp_float(pfx, "UNI_MAX_POS_FRAC", d["max_pos_frac"]),
        high_exposure_score_threshold=_rp_float(pfx, "UNI_HIGH_EXPOSURE_SCORE_THRESHOLD", d["high_exposure_score_threshold"]),

        # Spread & Risk
        spread_cap=_rp_float(pfx, "SPREAD_CAP", d["spread_cap"]),
        cvar_floor=_rp_float(pfx, "FUNNEL_EVENT_CVAR", d["cvar_floor"]),
        funnel_win_floor=_rp_float(pfx, "FUNNEL_WIN_FLOOR", d.get("funnel_win_floor", 0.50)),
        funnel_event_cvar_floor=_rp_float(pfx, "FUNNEL_EVENT_CVAR_FLOOR", d["funnel_event_cvar_floor"]),

        # Consensus
        local_consensus_alpha=_rp_float(pfx, "POLICY_LOCAL_CONSENSUS_ALPHA", d["local_consensus_alpha"]),

        # Direction
        allow_counter_trend=_rp_bool(pfx, "ALLOW_COUNTER_TREND", d["allow_counter_trend"]),

        # MC
        mc_n_paths=int(_rp_float(pfx, "MC_N_PATHS_LIVE", d["mc_n_paths"])),
    )

    return policy


def compute_adaptive_objective(
    ev_net: float,
    cvar_abs: float,
    std_dev: float,
    horizon_sec: float,
    regime: str,
    realized_sigma: float = 0.5,
    cost_total_roe: float = 0.0,
    policy: Optional[RegimePolicy] = None,
) -> float:
    """
    레짐 + 변동성 적응형 objective score 계산.

    기존 `j_new = (EV / (|CVaR| + 2σ)) × (1/√T)` 대체.

    개선점:
    1. 레짐별 objective_mode 자동 선택
    2. 변동성 정규화: σ_realized / σ_ref 로 스코어 스케일 보정
    3. 레짐별 차등 λ_var 적용
    4. Gross/Net 선택 → 비용에 의한 스코어 압축 완화

    Parameters
    ----------
    ev_net : float
        비용 차감 후 기대 수익 (ROE)
    cvar_abs : float
        |CVaR| (양수)
    std_dev : float
        수익률 표준편차
    horizon_sec : float
        평가 horizon (초)
    regime : str
        현재 레짐
    realized_sigma : float
        현재 실현 연율 변동성 (default 0.5)
    cost_total_roe : float
        총 거래비용 (ROE 단위)
    policy : RegimePolicy, optional
        사전 생성된 정책 (없으면 내부 생성)

    Returns
    -------
    float
        adaptive objective score (높을수록 좋은 기회)
    """
    if policy is None:
        policy = get_regime_policy(regime, realized_sigma)

    ev = ev_net
    if policy.use_gross_score:
        ev = ev_net + cost_total_roe  # 비용 재추가 → gross 평가

    # 변동성 정규화 계수
    sigma_ref = max(policy.sigma_ref, 0.01)
    sigma_safe = max(realized_sigma, 0.01)
    vol_norm = sigma_ref / sigma_safe  # σ 높으면 norm < 1 → 스코어 축소 (이미 vol가 분모에 있으므로)

    mode = policy.objective_mode
    lam = policy.lambda_var
    time_w = 1.0 / math.sqrt(max(horizon_sec, 1.0))
    cvar_eps = 1e-6

    if mode in ("ratio_time", "time_ratio"):
        # 추세장 최적: 방향성 대비 tail risk
        denom = max(cvar_abs + cvar_eps, cvar_eps)
        score = (ev / denom) * time_w

    elif mode in ("ev_var", "var_penalty"):
        # 횡보장 최적: EV에서 분산 패널티
        var_val = std_dev ** 2
        score = (ev - lam * var_val) * time_w

    elif mode in ("new_objective", "signal_reliability"):
        # 변동성/하락장: CVaR + 2σ 분모 (기존 j_new)
        denom = max(cvar_abs + 2.0 * std_dev, cvar_eps)
        score = (ev / denom) * time_w

    else:
        # default: ratio_time_var (기존 기본값)
        denom = max(cvar_abs + cvar_eps, cvar_eps)
        var_val = std_dev ** 2
        score = (ev / denom) * time_w - lam * var_val

    # 변동성 정규화 적용
    # σ가 기준보다 높으면 score를 보정 (threshold도 같이 스케일되므로 상대적 유효)
    score *= vol_norm

    return float(score)


def log_regime_policy(policy: RegimePolicy, sym: str = "") -> None:
    """디버그용 레짐 정책 로깅."""
    logger.info(
        f"[REGIME_POLICY] {sym} regime={policy.regime} "
        f"obj_mode={policy.objective_mode} λ_var={policy.lambda_var:.2f} "
        f"entry_floor={policy.entry_floor:.6f}+{policy.entry_floor_add:.4f} "
        f"tp/sl={policy.tp_pct:.4f}/{policy.sl_pct:.4f} "
        f"max_lev={policy.max_leverage:.0f} cf_λ={policy.cf_risk_lambda:.3f} "
        f"max_frac={policy.max_pos_frac:.2f} "
        f"min_hold={policy.min_hold_sec:.0f}s spread_cap={policy.spread_cap:.5f}"
    )

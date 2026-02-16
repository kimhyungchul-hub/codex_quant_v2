"""
Direction Model: mu_alpha 부호 기반 방향 결정 + 보정된 확신도 모델

## 구조적 문제 해결
1. MC 시뮬레이션의 EV_long ≈ EV_short (drift negligible vs noise)
   - mu_alpha(연간화) / 31,536,000 = ~3e-7 per step drift
   - sigma * sqrt(dt) = ~9e-5 per step noise
   - SNR per step ≈ 0.002 → 방향 결정이 noise-dominated

2. Ψ score 비교 (기존)는 사실상 난수 결정
   - EV_long - EV_short = 2*L*E[gross_ret] ≈ 0.0014%p
   - 이 미세한 차이로 LONG/SHORT을 결정 → 사실상 random

## 새로운 방향 결정 로직
1. mu_alpha 부호가 방향을 결정 (mu > 0 → LONG, mu < 0 → SHORT)
2. 다중 신호 합의(consensus)로 방향 확신도 산출
3. MC는 EV 크기와 리스크(CVaR) 평가에만 사용
4. 확신도 = f(mu_alpha magnitude, signal consensus, regime, hurst, vpin)
"""

from __future__ import annotations

import math
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DirectionResult:
    """방향 결정 결과"""
    direction: int          # +1 (LONG) or -1 (SHORT)
    confidence: float       # [0.0, 1.0] 보정된 확신도
    consensus_score: float  # [-1.0, +1.0] 다중 신호 합의 점수
    signal_count: int       # 합의에 참여한 신호 수
    dominant_source: str    # 가장 강한 신호 출처
    mu_alpha_sign: int      # mu_alpha 부호 (+1/-1/0)
    ev_sign: int            # MC EV가 선호하는 방향 (+1/-1)
    agreement: bool         # mu_alpha와 EV 방향 일치 여부
    debug: Dict[str, Any]   # 디버깅 정보


class DirectionModel:
    """
    다중 신호 합의 기반 방향 결정 모델.
    
    기존 문제: MC 시뮬레이션의 Ψ 비교는 drift가 noise에 묻혀 방향 식별 불가.
    해결: mu_alpha 부호 + alpha 구성요소 합의 + regime 필터로 방향 결정.
    MC EV는 크기(진입 가치) 평가에만 사용.

    ## 신호 소스 및 가중치
    1. mu_alpha (종합 alpha): 최고 우선순위
    2. mu_mom (모멘텀): 가격 추세
    3. mu_ofi (OFI): 주문 흐름
    4. mu_kf (칼만 필터): 상태 추정
    5. mu_bayes (베이지안): 사전분포 업데이트
    6. mu_ml (ML 예측): 기계학습 예측
    7. mu_ar (AR 모델): 자기회귀
    8. mu_pf (파티클 필터): 비선형 추정
    9. mu_ou (OU 프로세스): 평균회귀
    10. hawkes_boost: 이벤트 강도
    """

    # 신호 소스별 기본 가중치 (중요도 순)
    DEFAULT_SIGNAL_WEIGHTS = {
        "mu_alpha":     0.30,   # 종합 alpha (이미 가중 합산)
        "mu_mom":       0.15,   # 모멘텀 (가격 추세)
        "mu_ofi":       0.12,   # OFI (주문 흐름 불균형)
        "mu_kf":        0.10,   # 칼만 필터 (상태 추정)
        "mu_bayes":     0.08,   # 베이지안 (사전분포)
        "mu_ml":        0.08,   # ML 예측
        "mu_mlofi":     0.07,   # 다중레벨 OFI
        "mu_pf":        0.05,   # 파티클 필터
        "mu_ar":        0.03,   # AR 모델
        "hawkes_boost": 0.02,   # 이벤트 강도
    }

    # Regime별 확신도 스케일링
    REGIME_CONFIDENCE_SCALE = {
        "bull":         1.0,    # 추세장: 방향 신뢰
        "bear":         1.0,    # 추세장: 방향 신뢰
        "trend":        0.95,   # 일반 추세
        "chop":         0.65,   # 횡보: 방향 불확실
        "mean_revert":  0.55,   # 평균회귀: 역방향 가능
        "range":        0.60,   # 박스권
    }

    def __init__(self):
        self._signal_weights = dict(self.DEFAULT_SIGNAL_WEIGHTS)
        # WR 기반 온라인 보정을 위한 버퍼
        self._calibration_buffer: list[dict] = []
        self._calibration_max_size = 500
        # Platt scaling 파라미터 (초기값: 항등 함수)
        self._platt_a = 1.0
        self._platt_b = 0.0
        self._platt_fitted = False

    def determine_direction(
        self,
        *,
        mu_alpha: float,
        meta: Dict[str, Any],
        ctx: Dict[str, Any],
        ev_long: float = 0.0,
        ev_short: float = 0.0,
        score_long: float = 0.0,
        score_short: float = 0.0,
    ) -> DirectionResult:
        """
        다중 신호 합의 기반 방향 결정.
        
        Args:
            mu_alpha: 종합 mu_alpha 값
            meta: MC 엔진 메타데이터 (alpha 구성요소 포함)
            ctx: 엔진 컨텍스트 (regime, hurst, vpin 등)
            ev_long/ev_short: MC EV (참고용)
            score_long/score_short: MC Ψ score (참고용)
        
        Returns:
            DirectionResult with direction, calibrated confidence, etc.
        """
        debug = {}
        
        # 1. 개별 신호 부호 추출 및 가중 합의 계산
        signals: Dict[str, float] = {}
        signals["mu_alpha"] = float(mu_alpha)
        
        # meta에서 개별 alpha 구성요소 추출
        for key in ("mu_mom", "mu_ofi", "mu_kf", "mu_bayes", "mu_ml",
                     "mu_mlofi", "mu_pf", "mu_ar", "hawkes_boost"):
            val = meta.get(key)
            if val is not None:
                try:
                    signals[key] = float(val)
                except (TypeError, ValueError):
                    pass

        # mu_ou는 방향이 반대이므로 별도 처리
        mu_ou = meta.get("mu_ou")
        hurst_regime = str(meta.get("hurst_regime", "")).lower()
        if mu_ou is not None and hurst_regime == "mean_revert":
            try:
                signals["mu_ou"] = float(mu_ou)
            except (TypeError, ValueError):
                pass

        # 2. 가중 합의 점수 계산 (consensus_score)
        weighted_sum = 0.0
        total_weight = 0.0
        signal_count = 0
        dominant_source = "mu_alpha"
        dominant_strength = 0.0

        for name, val in signals.items():
            w = self._signal_weights.get(name, 0.0)
            if w <= 0 or abs(val) < 1e-9:
                continue
            
            # 부호 기반 기여 (크기는 무시, 부호 합의만)
            sign_contribution = math.copysign(1.0, val) * w
            weighted_sum += sign_contribution
            total_weight += w
            signal_count += 1

            if abs(val) * w > dominant_strength:
                dominant_strength = abs(val) * w
                dominant_source = name
        
        if total_weight > 0:
            consensus_score = weighted_sum / total_weight  # [-1, +1]
        else:
            consensus_score = 0.0

        debug["signals"] = {k: round(v, 6) for k, v in signals.items()}
        debug["weighted_sum"] = round(weighted_sum, 6)
        debug["total_weight"] = round(total_weight, 6)
        debug["signal_count"] = signal_count

        # 3. 방향 결정: consensus 부호 우선, 동점이면 mu_alpha 부호
        mu_alpha_sign = int(math.copysign(1, mu_alpha)) if abs(mu_alpha) > 1e-6 else 0
        ev_sign = 1 if score_long >= score_short else -1
        
        if abs(consensus_score) > 0.01:
            direction = 1 if consensus_score > 0 else -1
        elif mu_alpha_sign != 0:
            direction = mu_alpha_sign
        else:
            # mu_alpha도 0이면 MC EV 참조 (마지막 수단)
            direction = ev_sign

        agreement = (mu_alpha_sign == ev_sign) if mu_alpha_sign != 0 else True

        # 4. Raw 확신도 계산 (다중 요소 기반)
        raw_confidence = self._compute_raw_confidence(
            consensus_score=consensus_score,
            mu_alpha=mu_alpha,
            signal_count=signal_count,
            ctx=ctx,
            meta=meta,
            agreement=agreement,
            debug=debug,
        )

        # 5. Platt scaling으로 보정된 확신도
        calibrated_confidence = self._calibrate_confidence(raw_confidence)

        debug["raw_confidence"] = round(raw_confidence, 6)
        debug["calibrated_confidence"] = round(calibrated_confidence, 6)
        debug["direction_source"] = (
            "consensus" if abs(consensus_score) > 0.01
            else "mu_alpha" if mu_alpha_sign != 0
            else "ev_fallback"
        )

        return DirectionResult(
            direction=direction,
            confidence=calibrated_confidence,
            consensus_score=consensus_score,
            signal_count=signal_count,
            dominant_source=dominant_source,
            mu_alpha_sign=mu_alpha_sign,
            ev_sign=ev_sign,
            agreement=agreement,
            debug=debug,
        )

    def _compute_raw_confidence(
        self,
        *,
        consensus_score: float,
        mu_alpha: float,
        signal_count: int,
        ctx: Dict[str, Any],
        meta: Dict[str, Any],
        agreement: bool,
        debug: Dict[str, Any],
    ) -> float:
        """
        Raw 확신도 계산.
        
        구성 요소:
        1. consensus 강도 (40%): 신호 합의가 강할수록 높음
        2. mu_alpha 크기 (25%): mu_alpha 절대값이 클수록 높음  
        3. signal diversity (15%): 합의에 참여한 신호 수가 많을수록 높음
        4. regime 조정 (10%): 추세장에서 높고, 횡보장에서 낮음
        5. hurst/vpin 조정 (10%): 추세 지속성 높고 독성 낮을수록 높음
        """
        # 1. Consensus 강도 [0, 1]
        consensus_strength = min(1.0, abs(consensus_score))
        
        # 2. mu_alpha 크기 → 확신도 (sigmoid 변환)
        # |mu_alpha| = 0 → 0.5, |mu_alpha| = 5 → 0.92, |mu_alpha| = 10 → 0.99
        mu_abs = abs(mu_alpha)
        try:
            mu_cap = float(os.environ.get("MU_ALPHA_CAP", "10.0") or "10.0")
        except Exception:
            mu_cap = 10.0
        mu_norm = min(1.0, mu_abs / max(mu_cap, 1.0))
        mu_confidence = 2.0 / (1.0 + math.exp(-4.0 * mu_norm)) - 1.0  # sigmoid [0, 1)

        # 3. Signal diversity: 3개 이상 동의 시 보너스
        diversity_bonus = min(1.0, max(0.0, (signal_count - 1) / 5.0))

        # 4. Regime 조정
        regime = str(ctx.get("regime", "chop")).lower()
        regime_scale = self.REGIME_CONFIDENCE_SCALE.get(regime, 0.7)

        # 5. Hurst + VPIN 조정
        hurst = float(ctx.get("hurst", 0.5) or 0.5)
        vpin = float(ctx.get("vpin", 0.5) or 0.5)
        
        # Hurst > 0.55 = trend persistence (방향 확신 ↑)
        # Hurst < 0.45 = mean reversion (방향 확신 ↓)
        hurst_factor = min(1.0, max(0.3, (hurst - 0.3) / 0.4))  # [0.3, 1.0]
        
        # VPIN > 0.7 = high toxicity (방향 불확실 ↑)
        vpin_factor = min(1.0, max(0.4, 1.0 - 0.8 * max(0.0, vpin - 0.3)))  # [0.4, 1.0]

        # mu_alpha↔EV 방향 불일치 시 페널티
        agree_factor = 1.0 if agreement else 0.75

        # 가중 합산
        raw = (
            0.40 * consensus_strength +
            0.25 * mu_confidence +
            0.15 * diversity_bonus +
            0.10 * regime_scale +
            0.10 * (hurst_factor * 0.5 + vpin_factor * 0.5)
        ) * agree_factor

        # [0.0, 1.0] 범위로 클리핑
        raw = max(0.0, min(1.0, raw))

        debug["conf_components"] = {
            "consensus_strength": round(consensus_strength, 4),
            "mu_confidence": round(mu_confidence, 4),
            "diversity_bonus": round(diversity_bonus, 4),
            "regime_scale": round(regime_scale, 4),
            "hurst_factor": round(hurst_factor, 4),
            "vpin_factor": round(vpin_factor, 4),
            "agree_factor": round(agree_factor, 4),
        }

        return raw

    def _calibrate_confidence(self, raw_conf: float) -> float:
        """
        Platt scaling으로 확신도 보정.
        
        fitted 상태가 아니면 raw_conf를 [0.5, 0.85] 범위로 보수적 매핑.
        온라인 Platt scaling이 fit되면 그 파라미터를 사용.
        """
        if self._platt_fitted:
            # Platt scaling: P(correct | raw_conf) = 1 / (1 + exp(a * raw + b))
            z = self._platt_a * raw_conf + self._platt_b
            z = max(-30.0, min(30.0, z))
            calibrated = 1.0 / (1.0 + math.exp(-z))
        else:
            # 보수적 매핑: raw [0, 1] → calibrated [0.5, 0.85]
            # 0.85 상한 → 과도한 확신 방지
            calibrated = 0.5 + 0.35 * raw_conf
        
        return max(0.0, min(1.0, calibrated))

    def update_calibration(self, raw_conf: float, was_correct: bool):
        """
        거래 결과로 확신도 보정 파라미터 온라인 업데이트.
        
        Args:
            raw_conf: 진입 시점의 raw confidence
            was_correct: 방향이 맞았는지 (PnL > 0)
        """
        self._calibration_buffer.append({
            "raw_conf": float(raw_conf),
            "correct": 1.0 if was_correct else 0.0,
        })
        
        # 버퍼 초과 시 오래된 데이터 제거
        if len(self._calibration_buffer) > self._calibration_max_size:
            self._calibration_buffer = self._calibration_buffer[-self._calibration_max_size:]
        
        # 최소 50개 이상 모이면 Platt scaling fit
        if len(self._calibration_buffer) >= 50:
            self._fit_platt_scaling()

    def _fit_platt_scaling(self):
        """
        Platt scaling 파라미터 (a, b) 피팅.
        
        minimize NLL: -Σ [y*log(σ(ax+b)) + (1-y)*log(1-σ(ax+b))]
        Gradient descent (Newton's method simplified).
        """
        if len(self._calibration_buffer) < 50:
            return
        
        raw_confs = np.array([d["raw_conf"] for d in self._calibration_buffer])
        labels = np.array([d["correct"] for d in self._calibration_buffer])
        
        # Simple Platt scaling via logistic regression (1D)
        a, b = self._platt_a, self._platt_b
        lr = 0.01
        
        for _ in range(100):  # 100 iterations of gradient descent
            z = a * raw_confs + b
            z = np.clip(z, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            
            # Gradient
            err = p - labels
            grad_a = float(np.mean(err * raw_confs))
            grad_b = float(np.mean(err))
            
            a -= lr * grad_a
            b -= lr * grad_b
        
        self._platt_a = float(a)
        self._platt_b = float(b)
        self._platt_fitted = True
        
        # 보정 후 WR 확인
        z_final = a * raw_confs + b
        z_final = np.clip(z_final, -30, 30)
        p_final = 1.0 / (1.0 + np.exp(-z_final))
        
        # 빈별 WR 체크
        for th in [0.55, 0.60, 0.65, 0.70]:
            mask = p_final >= th
            if mask.sum() > 5:
                wr = float(labels[mask].mean())
                logger.info(
                    f"[DIR_MODEL_CALIB] conf>={th:.2f}: n={int(mask.sum())} WR={wr*100:.1f}%"
                )


def compute_direction_override(
    mu_alpha: float,
    meta: Dict[str, Any],
    ctx: Dict[str, Any],
    score_long: float,
    score_short: float,
    ev_long: float = 0.0,
    ev_short: float = 0.0,
) -> Dict[str, Any]:
    """
    decision.py에서 호출하는 편의 함수.
    mu_alpha 부호 기반 방향 결정 + 보정된 확신도.
    
    Returns:
        {
            "direction": int,        # +1 or -1
            "confidence": float,     # [0.0, 1.0]
            "consensus_score": float,
            "direction_source": str,
            "agreement": bool,
            "debug": dict,
        }
    """
    model = _get_global_model()
    result = model.determine_direction(
        mu_alpha=mu_alpha,
        meta=meta,
        ctx=ctx,
        ev_long=ev_long,
        ev_short=ev_short,
        score_long=score_long,
        score_short=score_short,
    )
    return {
        "direction": result.direction,
        "confidence": result.confidence,
        "raw_confidence": result.debug.get("raw_confidence", result.confidence),
        "consensus_score": result.consensus_score,
        "signal_count": result.signal_count,
        "dominant_source": result.dominant_source,
        "mu_alpha_sign": result.mu_alpha_sign,
        "ev_sign": result.ev_sign,
        "agreement": result.agreement,
        "direction_source": result.debug.get("direction_source", "unknown"),
        "debug": result.debug,
    }


# Global singleton
_GLOBAL_MODEL: Optional[DirectionModel] = None

def _get_global_model() -> DirectionModel:
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = DirectionModel()
    return _GLOBAL_MODEL

def get_direction_model() -> DirectionModel:
    """외부에서 모델 인스턴스에 접근 (calibration update 등)"""
    return _get_global_model()

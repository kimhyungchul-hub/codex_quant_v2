from __future__ import annotations

import math
import os
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from engines.mc.config import config

from engines.mc.constants import MC_VERBOSE_PRINT, SECONDS_PER_YEAR

logger = logging.getLogger(__name__)


def _calculate_refined_mu_alpha_with_debug(
    closes: np.ndarray,
    ofi_score: float,
    current_vol: float,
    long_term_vol: float,
    regime_kama: float,
    base_params: Dict[str, Any],
) -> tuple[float, Dict[str, Any]]:
    """
    Refined mu_alpha with acceleration + OFI divergence + vol-adaptive weighting.
    Returns (final_mu, debug_dict).
    """
    windows = [5, 10, 20]
    velocities: list[float] = []
    accelerations: list[float] = []
    for w in windows:
        if closes.size < w + 2:
            continue
        r_now = float(np.log(closes[-1] / closes[-w]))
        if closes.size >= 2 * w:
            r_prev = float(np.log(closes[-w] / closes[-2 * w]))
        else:
            r_prev = r_now
        vel = r_now
        acc = r_now - r_prev
        velocities.append(vel)
        accelerations.append(acc)

    if velocities:
        weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)[: len(velocities)]
        weights = weights / np.sum(weights)
        avg_velocity = float(np.dot(velocities, weights))
        avg_acceleration = float(np.dot(accelerations, weights))
    else:
        avg_velocity = 0.0
        avg_acceleration = 0.0

    momentum_score = float(avg_velocity)
    accel_dampen = False
    if (avg_velocity > 0 and avg_acceleration < 0) or (avg_velocity < 0 and avg_acceleration > 0):
        momentum_score *= 0.5
        accel_dampen = True
        logger.debug("[MU_DEBUG] Acceleration divergence detected. Dampening momentum.")

    mu_mom_scale = float(base_params.get("mu_mom_scale", 10.0))
    mu_mom = float(momentum_score * mu_mom_scale)

    mu_ofi_scale = float(base_params.get("mu_ofi_scale", 15.0))
    mu_ofi = float(ofi_score * mu_ofi_scale)

    divergence = False
    combined_raw = 0.0
    w_mom = None
    w_ofi = None
    if np.sign(mu_mom) != np.sign(mu_ofi) and abs(mu_ofi) > 0.5:
        divergence = True
        # [FIX 2026-02-09] 발산 시 dominant signal의 30% 유지 (이전: 0으로 완전 소멸)
        # Chop에서 mom↔OFI 부호 불일치는 흔하므로, 방향 정보를 완전 제거하면 안 됨
        if abs(mu_mom) >= abs(mu_ofi):
            combined_raw = float(mu_mom * 0.3)
            w_mom = 0.3
            w_ofi = 0.0
        else:
            combined_raw = float(mu_ofi * 0.3)
            w_mom = 0.0
            w_ofi = 0.3
    else:
        vol_ratio = float(current_vol / (long_term_vol + 1e-9))
        w_ofi = float(min(0.7, max(0.2, 0.5 * vol_ratio)))
        w_mom = float(1.0 - w_ofi)
        combined_raw = float((mu_mom * w_mom) + (mu_ofi * w_ofi))

    chop_threshold = float(base_params.get("chop_threshold", 0.3))
    if regime_kama < chop_threshold:
        # [FIX 2026-02-09] 이차 댐핑(quadratic) → 선형 댐핑으로 변경
        # 이전: (regime_kama/0.3)^2 → ER=0.1에서 0.11배 (89% 감쇄)
        # 변경: max(0.3, regime_kama/0.3) → ER=0.1에서 0.33배 (66% 감쇄)
        # regime.py에서 chop×0.8 추가 적용되므로 총 감쇄를 합리적 수준으로 제한
        regime_factor = float(max(0.3, regime_kama / chop_threshold))
        final_mu = float(combined_raw * regime_factor)
    else:
        boost = 1.2 if bool(base_params.get("boost_enabled", False)) else 1.0
        final_mu = float(combined_raw * boost)
        regime_factor = float(boost)

    alpha_scaling = float(base_params.get("alpha_scaling_factor", 1.0))
    final_mu = float(final_mu * alpha_scaling)

    mu_clipped = False

    debug = {
        "avg_velocity": float(avg_velocity),
        "avg_acceleration": float(avg_acceleration),
        "accel_dampen": bool(accel_dampen),
        "mu_mom": float(mu_mom),
        "mu_ofi": float(mu_ofi),
        "divergence": bool(divergence),
        "combined_raw": float(combined_raw),
        "w_mom": float(w_mom) if w_mom is not None else None,
        "w_ofi": float(w_ofi) if w_ofi is not None else None,
        "vol_ratio": float(current_vol / (long_term_vol + 1e-9)),
        "regime_factor": float(regime_factor),
        "alpha_scaling_factor": float(alpha_scaling),
        "mu_alpha_cap": None,
        "mu_alpha_floor": None,
        "mu_alpha_clipped": bool(mu_clipped),
    }
    return final_mu, debug


def calculate_refined_mu_alpha(
    closes: np.ndarray,
    ofi_score: float,
    current_vol: float,
    long_term_vol: float,
    regime_kama: float,
    base_params: dict,
) -> float:
    """
    [개선된 Mu Alpha 산출 로직]
    기존의 단순 가중평균을 넘어, 가속도(Acceleration)와 다이버전스를 반영하여
    변곡점에서의 손실을 방지합니다.
    """
    final_mu, _ = _calculate_refined_mu_alpha_with_debug(
        closes, ofi_score, current_vol, long_term_vol, regime_kama, base_params
    )
    return float(final_mu)


def ema(values: Sequence[float], period: int) -> Optional[float]:
    if values is None or len(values) < 2:
        return None
    v = np.asarray(values, dtype=np.float64)
    period = max(2, int(period))
    alpha = 2.0 / (period + 1.0)
    e = float(v[0])
    for x in v[1:]:
        e = alpha * float(x) + (1.0 - alpha) * e
    return float(e)


class MonteCarloSignalFeaturesMixin:
    @staticmethod
    def _annualize(mu_bar: float, sigma_bar: float, bar_seconds: float) -> Tuple[float, float]:
        bars_per_year = (365.0 * 24.0 * 3600.0) / float(bar_seconds)
        mu_base = float(mu_bar) * bars_per_year
        sigma_ann = float(sigma_bar) * math.sqrt(bars_per_year)
        return float(mu_base), float(max(sigma_ann, 1e-6))

    @staticmethod
    def _trend_direction(price: float, closes: Sequence[float]) -> int:
        # EMA200 없으면 EMA50/20로 대체
        if closes is None or len(closes) < 30:
            return 1
        p = float(price)
        e_slow = ema(closes, 200) if len(closes) >= 200 else ema(closes, min(50, len(closes)))
        if e_slow is None:
            return 1
        return 1 if p >= float(e_slow) else -1

    @staticmethod
    def _signal_alpha_mu_annual(closes: Sequence[float], bar_seconds: float, ofi_score: float, regime: str) -> float:
        """
        최근 평균수익(mu_bar)을 쓰지 않고, "신호(모멘텀/OFI)"로부터 조건부 기대수익(알파) μ(연율)를 만든다.
        - 출력 단위: 연율(log-return drift, per-year)
        - 방향: 양수=가격상승 기대, 음수=가격하락 기대
        """
        parts = MonteCarloSignalFeaturesMixin._signal_alpha_mu_annual_parts(closes, bar_seconds, ofi_score, regime)
        try:
            return float(parts.get("mu_alpha") or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _signal_alpha_mu_annual_parts(closes: Sequence[float], bar_seconds: float, ofi_score: float, regime: str) -> Dict[str, Any]:
        """
        _signal_alpha_mu_annual()의 분해/디버그 버전.
        - mu_mom_{15,30,60,120}: 각 모멘텀 창의 연율(로그수익 기울기) 추정치
        - mu_mom: 가중 평균
        - mu_ofi: OFI 기반 단기 알파 항
        - mu_alpha_raw: regime_scale 및 결합 후, cap 적용 전
        - mu_alpha: cap 적용 후 (실제 사용)
        """
        if not closes or len(closes) < 8:
            return {
                "bar_seconds": float(bar_seconds) if float(bar_seconds) > 0 else 60.0,
                "n_closes": int(len(closes) if closes else 0),
                "regime": str(regime or ""),
                "regime_scale": 1.0,
                "ofi_score_clipped": 0.0,
                "mu_ofi": 0.0,
                "mu_mom": 0.0,
                "mu_alpha_raw": 0.0,
                "mu_alpha_cap": None,
                "mu_alpha": 0.0,
                "reason": "insufficient_closes",
            }

        bs = float(bar_seconds) if float(bar_seconds) > 0 else 60.0
        n = len(closes)

        # OFI clipping
        try:
            ofi = float(ofi_score)
        except Exception:
            ofi = 0.0
        ofi = float(max(-1.0, min(1.0, ofi)))

        # Kaufman Efficiency Ratio (0~1)
        chop_window_bars = max(10, int(config.mu_alpha_chop_window_bars))
        er = None
        chop_score = None
        try:
            x = np.asarray(closes, dtype=np.float64)
            if x.size >= 3:
                win = min(int(chop_window_bars), int(x.size - 1))
                if win >= 2:
                    xx = np.log(x[-(win + 1) :])
                    net = abs(float(xx[-1] - xx[0]))
                    den = float(np.sum(np.abs(np.diff(xx))))
                    er_val = (net / den) if den > 1e-12 else 0.0
                    er_val = float(max(0.0, min(1.0, er_val)))
                    er = float(er_val)
                    chop_score = float(1.0 - er_val)
        except Exception:
            er = None
            chop_score = None

        if chop_score is None:
            chop_score = 1.0 if str(regime or "").lower() == "chop" else 0.0

        # Volatility windows for adaptive weighting
        vol_short = None
        vol_long = None
        try:
            x = np.asarray(closes, dtype=np.float64)
            rets = np.diff(np.log(x))
            if rets.size >= 5:
                sb = min(int(config.mu_alpha_vol_short_bars), int(rets.size))
                lb = min(int(config.mu_alpha_vol_long_bars), int(rets.size))
                vol_short = float(np.std(rets[-sb:])) if sb >= 2 else None
                vol_long = float(np.std(rets[-lb:])) if lb >= 2 else None
        except Exception:
            vol_short = None
            vol_long = None

        current_vol = float(vol_short) if vol_short is not None else 0.0
        long_term_vol = float(vol_long) if vol_long is not None else max(current_vol, 1e-9)

        base_params = {
            "mu_mom_scale": float(os.environ.get("MU_MOM_SCALE", 10.0) or 10.0),
            "mu_ofi_scale": float(getattr(config, "mu_ofi_scale", 10.0) or 10.0),
            "boost_enabled": bool(config.alpha_signal_boost),
            "alpha_scaling_factor": float(config.alpha_scaling_factor),
            "chop_threshold": 0.3,
        }

        mu_alpha, dbg = _calculate_refined_mu_alpha_with_debug(
            np.asarray(closes, dtype=np.float64),
            ofi,
            current_vol,
            long_term_vol,
            float(er or 0.0),
            base_params,
        )

        out: Dict[str, Any] = {
            "bar_seconds": float(bs),
            "n_closes": int(n),
            "regime": str(regime or ""),
            "er": float(er) if er is not None else None,
            "chop_score": float(chop_score) if chop_score is not None else None,
            "ofi_score_clipped": float(ofi),
            "vol_short": float(vol_short) if vol_short is not None else None,
            "vol_long": float(vol_long) if vol_long is not None else None,
            "mu_mom": float(dbg.get("mu_mom", 0.0)),
            "mu_ofi": float(dbg.get("mu_ofi", 0.0)),
            "mu_alpha_raw": float(dbg.get("combined_raw", 0.0)),
            "mu_alpha_cap": None,
            "mu_alpha_clipped": bool(dbg.get("mu_alpha_clipped", False)),
            "mu_alpha": float(mu_alpha),
            "mu_alpha_accel_dampen": bool(dbg.get("accel_dampen", False)),
            "mu_alpha_divergence": bool(dbg.get("divergence", False)),
            "mu_alpha_w_mom": dbg.get("w_mom"),
            "mu_alpha_w_ofi": dbg.get("w_ofi"),
            "mu_alpha_vol_ratio": dbg.get("vol_ratio"),
            "alpha_scaling_factor": float(dbg.get("alpha_scaling_factor", config.alpha_scaling_factor)),
        }

        if MC_VERBOSE_PRINT:
            print(
                f"[ALPHA_DEBUG] mu_mom={out['mu_mom']:.6f} mu_ofi={out['mu_ofi']:.6f} mu_alpha_raw={out['mu_alpha_raw']:.6f} mu_alpha={out['mu_alpha']:.6f}"
            )

        return out

    @staticmethod
    def _cluster_regime(closes: Sequence[float]) -> str:
        if closes is None or len(closes) < 40:
            return "chop"
        x = np.asarray(closes, dtype=np.float64)
        rets = np.diff(np.log(x))
        if rets.size < 30:
            return "chop"
        # 특징: 단기 추세, 변동성
        slope = float(x[-1] - x[-10]) / max(1e-6, float(x[-10]))
        vol = float(rets[-30:].std())
        feats = np.array([[slope, vol]], dtype=np.float64)
        # 초기 중심 (bear/chop/bull 가정)
        centers = np.array([
            [-0.002, vol * 1.2],
            [0.0, vol],
            [0.002, max(vol * 0.8, 1e-6)]
        ], dtype=np.float64)
        # 미니 k-means 3회
        for _ in range(3):
            d = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d.argmin(axis=1)
            for k in range(3):
                mask = labels == k
                if mask.any():
                    centers[k] = feats[mask].mean(axis=0)
        label = int(labels[0])
        if label == 0:
            return "bear"
        if label == 2:
            return "bull"
        return "volatile" if vol > 0.01 else "chop"

    # -----------------------------
    # Slippage model
    # -----------------------------

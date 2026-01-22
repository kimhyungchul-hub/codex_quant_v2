from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from engines.mc.config import config

from engines.mc.constants import MC_VERBOSE_PRINT, SECONDS_PER_YEAR


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
                "mu_alpha_cap": 40.0,
                "mu_alpha": 0.0,
                "reason": "insufficient_closes",
            }

        bs = float(bar_seconds) if float(bar_seconds) > 0 else 60.0
        n = len(closes)

        def _mom_mu_ann(window_bars: int) -> tuple[float, float, float]:
            w = int(window_bars)
            if w <= 1 or n <= w:
                return 0.0, 0.0, 0.0
            p0 = float(closes[-w - 1])
            p1 = float(closes[-1])
            if p0 <= 0.0 or p1 <= 0.0:
                return 0.0, 0.0, 0.0
            try:
                lr = math.log(p1 / p0)
            except Exception:
                return 0.0, 0.0, 0.0
            # Stabilize annualization: short-window momentum can explode when scaled to per-year.
            # Use a tau floor (in seconds) and optional log-return clipping.
            lr_cap = config.mu_mom_lr_cap
            if lr_cap > 0:
                lr = float(max(-lr_cap, min(lr_cap, float(lr))))

            tau_floor = config.mu_mom_tau_floor_sec
            tau = max(1e-9, float(w) * bs, float(max(0.0, tau_floor)))
            
            # Calculate annualized momentum
            mu_ann_raw = float((lr / tau) * SECONDS_PER_YEAR)
            
            # ✅ NEW: Cap annualized momentum directly to prevent explosion
            mu_ann_cap = config.mu_mom_ann_cap
            if mu_ann_cap > 0:
                mu_ann = float(max(-mu_ann_cap, min(mu_ann_cap, mu_ann_raw)))
            else:
                mu_ann = mu_ann_raw
            
            return float(mu_ann), float(lr), float(tau)

        mom_cfg = ((15, 0.35), (30, 0.30), (60, 0.25), (120, 0.10))
        mom_terms_w = []
        mom_w = []
        mom_each: Dict[str, Any] = {}
        for w, wt in mom_cfg:
            if n > int(w) + 1:
                mu_w, lr_w, tau_w = _mom_mu_ann(int(w))
                mom_each[f"mu_mom_{int(w)}"] = float(mu_w)
                mom_each[f"lr_mom_{int(w)}"] = float(lr_w)
                mom_each[f"tau_mom_{int(w)}_sec"] = float(tau_w)
                mom_terms_w.append(float(mu_w) * float(wt))
                mom_w.append(float(wt))
        mu_mom = float(sum(mom_terms_w) / max(1e-12, float(sum(mom_w)))) if mom_terms_w else 0.0

        # OFI는 매우 단기 알파로 취급 (연율로 스케일)
        try:
            ofi = float(ofi_score)
        except Exception:
            ofi = 0.0
        ofi = float(max(-1.0, min(1.0, ofi)))
        try:
            ofi_scale = config.mu_ofi_scale
        except Exception:
            ofi_scale = 8.0
        mu_ofi = float(ofi * ofi_scale)

        r = str(regime or "").lower()

        # Smooth regime scaling (always-on): use a continuous "chop_score" (0=trend, 1=chop)
        # based on Kaufman efficiency ratio over recent window.
        # This avoids hard-switch flicker at regime boundaries.
        scale_min = config.mu_alpha_scale_min
        scale_min = float(max(0.0, min(1.0, scale_min)))
        chop_window_bars = config.mu_alpha_chop_window_bars
        chop_window_bars = max(10, int(chop_window_bars))

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

        # Fallback if chop_score can't be computed (should be rare).
        if chop_score is None:
            if r == "chop":
                chop_score = 1.0
            else:
                chop_score = 0.0

        regime_scale = float(1.0 - (1.0 - scale_min) * float(chop_score))

        # Adaptive weighting (always-on): vol이 낮으면 OFI 비중↑, vol이 높으면 모멘텀 비중↑
        w_mom = 0.70
        w_ofi = 0.30
        vol_short = None
        vol_long = None
        vol_ratio = None
        try:
            w_mom_base = config.mu_alpha_w_mom_base
            w_mom_min = config.mu_alpha_w_mom_min
            w_mom_max = config.mu_alpha_w_mom_max
            short_bars = config.mu_alpha_vol_short_bars
            long_bars = config.mu_alpha_vol_long_bars
            short_bars = max(5, short_bars)
            long_bars = max(short_bars, long_bars)

            x = np.asarray(closes, dtype=np.float64)
            rets = np.diff(np.log(x))
            if rets.size >= 5:
                sb = min(int(short_bars), int(rets.size))
                lb = min(int(long_bars), int(rets.size))
                vol_short = float(np.std(rets[-sb:])) if sb >= 2 else None
                vol_long = float(np.std(rets[-lb:])) if lb >= 2 else None
                if vol_short is not None and vol_long is not None and vol_long > 1e-12:
                    vol_ratio = float(vol_short / vol_long)
                    # 변동성이 평소보다 크면 모멘텀 비중↑ (base*ratio)
                    w_mom = float(max(w_mom_min, min(w_mom_max, w_mom_base * vol_ratio)))
                    w_ofi = float(1.0 - w_mom)
        except Exception:
            # fallback to fixed weights
            w_mom = 0.70
            w_ofi = 0.30

        mu_alpha_raw = float(regime_scale * (w_mom * mu_mom + w_ofi * mu_ofi))
        
        # ✅ Apply dynamic scaling factor
        scaling = config.alpha_scaling_factor
        mu_alpha_raw_before = mu_alpha_raw
        
        # ✅ NEW: Trend strength boost - 추세가 강할수록 신호 증폭
        # er (efficiency ratio)가 높으면 추세가 강함 → 부스터 적용
        trend_boost = 1.0
        if config.alpha_signal_boost and er is not None:
            # er=1.0 (perfect trend) → boost=1.5, er=0.0 (chop) → boost=1.0
            # Normalized from 2.0 max to 1.5 max for stability
            trend_boost = 1.0 + 0.5 * float(er)  # 1.0 ~ 1.5
        
        mu_alpha_raw = mu_alpha_raw * scaling * trend_boost
        
        # Debug: Print scaling factor (first time only)
        if not hasattr(config, '_scaling_logged'):
            print(f"[ALPHA_SCALING_DEBUG] alpha_scaling_factor={scaling}, trend_boost={trend_boost:.2f}, mu_alpha_raw_before={mu_alpha_raw_before:.6f}, mu_alpha_raw_after={mu_alpha_raw:.6f}")
            config._scaling_logged = True

        mu_cap = config.mu_alpha_cap
        mu_floor = config.mu_alpha_floor
        mu_alpha = float(mu_alpha_raw)
        mu_alpha_clipped = False
        
        # Apply asymmetric bounds
        if mu_alpha > mu_cap:
            mu_alpha = mu_cap
            mu_alpha_clipped = True
        elif mu_alpha < mu_floor:
            mu_alpha = mu_floor
            mu_alpha_clipped = True

        out: Dict[str, Any] = {
            "bar_seconds": float(bs),
            "n_closes": int(n),
            "regime": str(regime or ""),
            "regime_scale": float(regime_scale),
            "er": float(er) if er is not None else None,
            "chop_score": float(chop_score) if chop_score is not None else None,
            "scale_min": float(scale_min),
            "w_mom": float(w_mom),
            "w_ofi": float(w_ofi),
            "vol_short": float(vol_short) if vol_short is not None else None,
            "vol_long": float(vol_long) if vol_long is not None else None,
            "vol_ratio": float(vol_ratio) if vol_ratio is not None else None,
            "mu_mom_tau_floor_sec": config.mu_mom_tau_floor_sec,
            "mu_mom_lr_cap": config.mu_mom_lr_cap,
            "ofi_score_clipped": float(ofi),
            "ofi_scale": float(ofi_scale),
            "mu_ofi": float(mu_ofi),
            "mu_mom": float(mu_mom),
            "alpha_scaling_factor": float(scaling),
            "trend_boost": float(trend_boost),
            "mu_alpha_raw_before": float(mu_alpha_raw_before),
            "mu_alpha_raw": float(mu_alpha_raw),
            "mu_alpha_cap": float(mu_cap),
            "mu_alpha_clipped": bool(mu_alpha_clipped),
            "mu_alpha": float(mu_alpha),
        }
        out.update(mom_each)
        
        # Debug: Print mu_alpha calculation
        if MC_VERBOSE_PRINT:
            print(
                f"[ALPHA_DEBUG] mu_mom={mu_mom:.6f} mu_ofi={mu_ofi:.6f} regime_scale={regime_scale:.2f} mu_alpha_raw={mu_alpha_raw:.6f} mu_alpha={mu_alpha:.6f}"
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

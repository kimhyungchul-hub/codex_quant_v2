"""
MarketDataMixin - 시장 데이터 계산 기능
======================================

LiveOrchestrator에서 분리된 시장 데이터 분석 메서드들.
- 수익률/변동성 계산 (_compute_returns_and_vol, _annualize_mu_sigma)
- 방향성 판단 (_direction_bias)
- 레짐 추정 (_infer_regime)
- 호가창 분석 (_compute_ofi_score, _liquidity_score)
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from regime import MarketRegimeDetector

if TYPE_CHECKING:
    pass


class MarketDataMixin:
    """시장 데이터 계산 믹스인"""

    _regime_detectors: Dict[str, MarketRegimeDetector]
    _regime_meta: Dict[str, Dict[str, Any]]

    def _compute_returns_and_vol(
        self, 
        closes: List[float], 
        lookback: int = 20
    ) -> Tuple[float, float]:
        """
        최근 종가에서 수익률과 변동성 계산.
        
        Args:
            closes: 종가 리스트
            lookback: 계산에 사용할 캔들 수
            
        Returns:
            (평균 로그 수익률, 변동성 std)
        """
        if len(closes) < 2:
            return 0.0, 0.0
        
        closes = closes[-lookback:]
        arr = np.array(closes, dtype=np.float64)
        
        # 로그 수익률
        log_returns = np.diff(np.log(arr))
        
        if len(log_returns) == 0:
            return 0.0, 0.0
            
        mu = float(np.mean(log_returns))
        sigma = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0
        
        return mu, sigma

    def _direction_bias(self, closes: List[float], lookback: int = 10) -> float:
        """
        최근 가격 방향성 편향 계산.
        
        Args:
            closes: 종가 리스트
            lookback: 계산에 사용할 캔들 수
            
        Returns:
            -1.0 ~ 1.0 사이의 방향성 지표
        """
        if len(closes) < 2:
            return 0.0
        
        closes = closes[-lookback:]
        arr = np.array(closes, dtype=np.float64)
        
        # 상승 캔들 비율
        changes = np.diff(arr)
        up_ratio = float(np.sum(changes > 0) / len(changes)) if len(changes) > 0 else 0.5
        
        return 2.0 * up_ratio - 1.0  # -1 ~ 1 변환

    def _infer_regime(
        self,
        closes: List[float],
        symbol: Optional[str] = None,
        threshold_low: float = 0.005,
        threshold_high: float = 0.02,
    ) -> str:
        """온라인 GMM 기반 레짐 추정 (폴백: 단순 변동성 분류).

        Args:
            closes: 종가 리스트
            symbol: 심볼(없으면 공용 디텍터 사용)
            threshold_low: 폴백용 낮은 변동성 임계값
            threshold_high: 폴백용 높은 변동성 임계값

        Returns:
            레짐 라벨 문자열
        """
        if not closes or len(closes) < 2:
            return "-"

        detector = self._get_regime_detector(symbol)
        try:
            meta = detector.detect_regime(closes, assume_returns=False)
            if symbol:
                self._regime_meta[symbol] = meta
            else:
                self._regime_meta["__default__"] = meta
            return meta.get("regime", "-")
        except Exception:
            # 폴백: 기존 변동성 기반 분류 유지
            _, sigma = self._compute_returns_and_vol(closes, lookback=20)
            if sigma < threshold_low:
                return "LOW_VOL"
            if sigma > threshold_high:
                return "HIGH_VOL"
            return "MED_VOL"

    def _compute_ofi_score(
        self, 
        orderbook: Optional[Dict[str, Any]]
    ) -> float:
        """
        주문서 불균형(OFI) 점수 계산.
        
        Args:
            orderbook: {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
            
        Returns:
            -1.0 ~ 1.0 사이의 OFI 점수 (양수: 매수 우세)
        """
        if not orderbook:
            return 0.0
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return 0.0
        
        try:
            # 상위 5레벨 수량 합산
            bid_vol = sum(float(b[1]) for b in bids[:5])
            ask_vol = sum(float(a[1]) for a in asks[:5])
            
            total = bid_vol + ask_vol
            if total == 0:
                return 0.0
            
            # 정규화된 불균형
            return (bid_vol - ask_vol) / total
        except Exception:
            return 0.0

    def _liquidity_score(
        self, 
        orderbook: Optional[Dict[str, Any]], 
        depth_levels: int = 10
    ) -> float:
        """
        주문서 유동성 점수 계산.
        
        Args:
            orderbook: 호가창 데이터
            depth_levels: 분석할 호가 레벨 수
            
        Returns:
            0.0 ~ 1.0 사이의 유동성 점수
        """
        if not orderbook:
            return 0.0
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return 0.0
        
        try:
            mid = (float(bids[0][0]) + float(asks[0][0])) / 2
            if mid == 0:
                return 0.0
            
            # 스프레드 기반 유동성
            spread = float(asks[0][0]) - float(bids[0][0])
            spread_pct = spread / mid
            
            # 깊이 기반 유동성
            bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:depth_levels])
            ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:depth_levels])
            total_depth = bid_depth + ask_depth
            
            # 스프레드가 작고 깊이가 클수록 유동성 높음
            # 간단한 휴리스틱: 스프레드 < 0.1%면 기본 0.5, 깊이에 따라 조정
            base_score = max(0.0, 1.0 - spread_pct * 100)  # 스프레드 1%면 0
            depth_bonus = min(0.5, total_depth / (mid * 1000))  # 깊이 보너스
            
            return min(1.0, base_score * 0.7 + depth_bonus * 0.3)
        except Exception:
            return 0.0

    def _annualize_mu_sigma(
        self, 
        mu: float, 
        sigma: float, 
        interval_sec: int = 60
    ) -> Tuple[float, float]:
        """
        캔들 기반 mu/sigma를 연율화.
        
        Args:
            mu: 캔들 기준 평균 로그 수익률
            sigma: 캔들 기준 변동성
            interval_sec: 캔들 간격 (초)
            
        Returns:
            (연율화 mu, 연율화 sigma)
        """
        # 1년 = 365일 * 24시간 * 3600초
        periods_per_year = (365 * 24 * 3600) / interval_sec
        
        # mu는 기간에 비례, sigma는 sqrt(기간)에 비례
        ann_mu = mu * periods_per_year
        ann_sigma = sigma * math.sqrt(periods_per_year)
        
        return ann_mu, ann_sigma

    # --------- helpers ---------
    def _get_regime_detector(self, symbol: Optional[str]) -> MarketRegimeDetector:
        """심볼별(또는 공용) 레짐 디텍터를 반환/초기화."""
        if not hasattr(self, "_regime_detectors"):
            self._regime_detectors = {}
        if not hasattr(self, "_regime_meta"):
            self._regime_meta = {}

        key = symbol or "__default__"
        detector = self._regime_detectors.get(key)
        if detector is None:
            detector = MarketRegimeDetector(
                n_states=getattr(self, "REGIME_N_STATES", 3),
                window=getattr(self, "REGIME_WINDOW", 256),
                alpha=getattr(self, "REGIME_ALPHA", 0.05),
                use_jax=getattr(self, "REGIME_USE_JAX", False),
            )
            self._regime_detectors[key] = detector
        return detector

"""
DecisionMixin - 의사결정 기능
============================

LiveOrchestrator에서 분리된 의사결정 메서드들.
- 결정 컨텍스트 구성 (_build_decision_context)
- 합의 로직 (_consensus_action, _consensus_used_flag)
- EMA 업데이트 (_ema_update)
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    pass


class DecisionMixin:
    """의사결정 믹스인"""

    def _build_decision_context(
        self,
        sym: str,
        ts_ms: int
    ) -> Optional[Dict[str, Any]]:
        """
        심볼에 대한 의사결정 컨텍스트 구성.
        
        Args:
            sym: 심볼
            ts_ms: 현재 타임스탬프 (ms)
            
        Returns:
            결정 컨텍스트 dict 또는 None (데이터 부족 시)
        """
        # 가격 정보
        price = self._latest_prices.get(sym)
        
        # OHLCV 데이터
        ohlcv = self._ohlcv_buffer.get(sym, {})
        closes = ohlcv.get("closes", [])
        
        # 가격 폴백: ticker 미수신 시 마지막 close 사용
        if price is None and closes:
            price = closes[-1]
        
        if not price or not closes or len(closes) < 10:
            return None
        
        # 변동성 계산
        mu, sigma = self._compute_returns_and_vol(closes, lookback=20)
        
        # 연율화
        ann_mu, ann_sigma = self._annualize_mu_sigma(
            mu, sigma, 
            interval_sec=getattr(self, "OHLCV_INTERVAL_SEC", 60)
        )
        
        # 호가창 데이터
        orderbook = self._orderbook_buffer.get(sym, {})
        ofi = self._compute_ofi_score(orderbook)
        liquidity = self._liquidity_score(orderbook)
        
        # 레짐
        regime = self._infer_regime(closes, symbol=sym)
        
        # 방향성
        direction = self._direction_bias(closes)
        
        # 포지션 상태
        pos = self.positions.get(sym)
        in_position = pos is not None
        position_side = pos.get("action") if pos else None
        hold_sec = 0
        if pos:
            hold_sec = (ts_ms - pos.get("entry_time", ts_ms)) / 1000
        
        ctx = {
            "symbol": sym,
            "price": price,
            "mu": ann_mu,
            "sigma": ann_sigma,
            "closes": closes[-50:],  # 최근 50개만
            "orderbook": orderbook,
            "ofi": ofi,
            "liquidity": liquidity,
            "regime": regime,
            "direction": direction,
            "in_position": in_position,
            "position_side": position_side,
            "hold_sec": hold_sec,
            "balance": self.balance,
            "leverage": self.leverage,
            "ts_ms": ts_ms,
        }
        
        return ctx

    def _consensus_action(
        self,
        decisions: List[Dict[str, Any]],
        threshold: float = 0.6
    ) -> Tuple[str, float]:
        """
        여러 결정에서 합의 액션 도출.
        
        Args:
            decisions: 결정 리스트 [{"action": str, "confidence": float}, ...]
            threshold: 합의 임계값
            
        Returns:
            (합의 액션, 합의 신뢰도)
        """
        if not decisions:
            return "HOLD", 0.0
        
        # 투표 집계
        votes: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        
        for dec in decisions:
            action = dec.get("action", "HOLD")
            conf = float(dec.get("confidence", 0.5))
            
            if action in votes:
                votes[action] += conf
        
        total = sum(votes.values())
        if total == 0:
            return "HOLD", 0.0
        
        # 최다 득표 액션
        best_action = max(votes.keys(), key=lambda k: votes[k])
        best_ratio = votes[best_action] / total
        
        # 임계값 미달 시 HOLD
        if best_ratio < threshold:
            return "HOLD", best_ratio
        
        return best_action, best_ratio

    def _consensus_used_flag(
        self,
        decisions: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """
        합의에 사용된 플래그들 집계.
        
        Args:
            decisions: 결정 리스트
            
        Returns:
            통합 플래그 dict
        """
        flags: Dict[str, bool] = {}
        
        for dec in decisions:
            for key, value in dec.items():
                if isinstance(value, bool):
                    # OR 합성: 하나라도 True면 True
                    flags[key] = flags.get(key, False) or value
        
        return flags

    def _ema_update(
        self,
        old_value: float,
        new_value: float,
        alpha: float = 0.1
    ) -> float:
        """
        지수이동평균 업데이트.
        
        Args:
            old_value: 이전 EMA 값
            new_value: 새 관측값
            alpha: 스무딩 팩터 (0~1, 높을수록 빠른 반응)
            
        Returns:
            업데이트된 EMA 값
        """
        if math.isnan(old_value) or math.isinf(old_value):
            return new_value
        if math.isnan(new_value) or math.isinf(new_value):
            return old_value
        
        return alpha * new_value + (1 - alpha) * old_value

    def _apply_decision(
        self,
        sym: str,
        decision: Dict[str, Any],
        ts_ms: int
    ) -> Optional[Dict[str, Any]]:
        """
        결정을 적용하여 액션 생성.
        
        Args:
            sym: 심볼
            decision: 엔진 결정 결과
            ts_ms: 현재 타임스탬프
            
        Returns:
            수행할 액션 dict 또는 None
        """
        action = decision.get("action", "HOLD")
        
        if action == "HOLD":
            return None
        
        confidence = float(decision.get("confidence", 0.5))
        ev = float(decision.get("ev", 0.0))
        cap_frac = float(decision.get("cap_frac", 0.01))
        leverage = float(decision.get("leverage", self.leverage))
        
        # 신뢰도 필터
        if confidence < self.MIN_CONFIDENCE:
            return None
        
        # EV 필터
        if ev < self.MIN_EV:
            return None
        
        price = self._latest_prices.get(sym)
        if not price:
            return None
        
        return {
            "symbol": sym,
            "action": action,
            "price": price,
            "confidence": confidence,
            "ev": ev,
            "cap_frac": cap_frac,
            "leverage": leverage,
            "ts_ms": ts_ms,
        }

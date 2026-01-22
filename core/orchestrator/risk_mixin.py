"""
RiskMixin - 리스크 관리 기능
============================

LiveOrchestrator에서 분리된 리스크 관리 메서드들.
- 동적 레버리지 조정 (_dynamic_leverage_risk)
- 청산 조건 평가 (_maybe_exit_position, _evaluate_event_exit)
- EV 기반 청산 (_check_ema_ev_exit, _check_ev_drop_exit)
"""

from __future__ import annotations
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    pass


class RiskMixin:
    """리스크 관리 믹스인"""

    def _dynamic_leverage_risk(
        self, 
        base_leverage: float, 
        volatility: float,
        confidence: float = 0.5
    ) -> float:
        """
        변동성과 신뢰도 기반 동적 레버리지 조정.
        
        Args:
            base_leverage: 기본 레버리지
            volatility: 현재 변동성 (연율화 sigma)
            confidence: 신호 신뢰도 (0~1)
            
        Returns:
            조정된 레버리지
        """
        # 변동성 역비례 조정
        # vol > 0.5 (50%) -> 레버리지 감소
        # vol < 0.2 (20%) -> 레버리지 유지/증가
        vol_factor = max(0.3, min(1.0, 0.3 / max(0.01, volatility)))
        
        # 신뢰도 비례 조정
        conf_factor = 0.5 + 0.5 * confidence
        
        adjusted = base_leverage * vol_factor * conf_factor
        
        # 범위 제한
        return max(1.0, min(self.MAX_LEVERAGE, adjusted))

    def _maybe_exit_position(
        self,
        sym: str,
        pos: Dict[str, Any],
        current_price: float,
        ts_ms: int
    ) -> Tuple[bool, str]:
        """
        포지션 청산 여부 판단.
        
        Args:
            sym: 심볼
            pos: 포지션 정보
            current_price: 현재 가격
            ts_ms: 현재 타임스탬프 (ms)
            
        Returns:
            (청산 여부, 청산 이유)
        """
        entry_price = float(pos.get("entry_price", current_price))
        entry_time = int(pos.get("entry_time", ts_ms))
        action = pos.get("action", "BUY")
        hold_limit = int(pos.get("hold_limit", self.MAX_POSITION_HOLD_SEC * 1000))
        
        # 수익률 계산
        if action == "BUY":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # 1. 손절 체크
        if pnl_pct < -self.STOP_LOSS_PCT:
            return True, "STOP_LOSS"
        
        # 2. 익절 체크
        if pnl_pct > self.TAKE_PROFIT_PCT:
            return True, "TAKE_PROFIT"
        
        # 3. 홀딩 시간 초과
        hold_time = ts_ms - entry_time
        if hold_time > hold_limit:
            return True, "TIME_LIMIT"
        
        # 4. 트레일링 스탑
        if hasattr(self, "TRAILING_STOP_PCT") and self.TRAILING_STOP_PCT > 0:
            peak = float(pos.get("peak_price", entry_price))
            if action == "BUY":
                if current_price > peak:
                    pos["peak_price"] = current_price
                elif (peak - current_price) / peak > self.TRAILING_STOP_PCT:
                    return True, "TRAILING_STOP"
            else:
                if current_price < peak:
                    pos["peak_price"] = current_price
                elif (current_price - peak) / peak > self.TRAILING_STOP_PCT:
                    return True, "TRAILING_STOP"
        
        return False, ""

    def _evaluate_event_exit(
        self,
        sym: str,
        pos: Dict[str, Any],
        current_price: float,
        event_type: str
    ) -> Tuple[bool, str]:
        """
        이벤트 기반 청산 평가.
        
        Args:
            sym: 심볼
            pos: 포지션 정보
            current_price: 현재 가격
            event_type: 이벤트 유형
            
        Returns:
            (청산 여부, 청산 이유)
        """
        # 킬스위치 발동
        if event_type == "KILL_SWITCH":
            return True, "KILL_SWITCH"
        
        # 극단적 변동성
        if event_type == "EXTREME_VOL":
            entry_price = float(pos.get("entry_price", current_price))
            pnl_pct = abs(current_price - entry_price) / entry_price
            # 급격한 움직임 + 수익 중이면 청산
            if pnl_pct > 0.02:  # 2% 이상 움직임
                return True, "EXTREME_VOL_EXIT"
        
        # 유동성 고갈
        if event_type == "LOW_LIQUIDITY":
            return True, "LIQUIDITY_EXIT"
        
        return False, ""

    def _check_ema_ev_exit(
        self,
        sym: str,
        pos: Dict[str, Any],
        current_ev: float
    ) -> Tuple[bool, str]:
        """
        EMA EV 기반 청산 체크.
        
        진입 시 EV 대비 현재 EV가 크게 하락하면 청산.
        
        Args:
            sym: 심볼
            pos: 포지션 정보
            current_ev: 현재 EV
            
        Returns:
            (청산 여부, 청산 이유)
        """
        entry_ev = float(pos.get("entry_ev", 0.0))
        
        if entry_ev <= 0:
            return False, ""
        
        # EV가 진입 시의 30% 이하로 떨어지면 청산
        if current_ev < entry_ev * 0.3:
            return True, "EV_COLLAPSE"
        
        # EV가 음수로 전환되면 청산
        if current_ev < 0 and entry_ev > 0:
            return True, "EV_NEGATIVE"
        
        return False, ""

    def _check_ev_drop_exit(
        self,
        sym: str,
        pos: Dict[str, Any],
        ev_history: list
    ) -> Tuple[bool, str]:
        """
        EV 급락 패턴 감지 기반 청산.
        
        Args:
            sym: 심볼
            pos: 포지션 정보
            ev_history: 최근 EV 이력
            
        Returns:
            (청산 여부, 청산 이유)
        """
        if len(ev_history) < 3:
            return False, ""
        
        recent = ev_history[-3:]
        
        # 연속 하락 패턴
        if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
            drop_rate = (recent[0] - recent[-1]) / max(0.0001, abs(recent[0]))
            if drop_rate > 0.5:  # 50% 이상 급락
                return True, "EV_DROP_PATTERN"
        
        return False, ""

    def _check_drawdown_stop(self) -> Tuple[bool, str]:
        """
        포트폴리오 전체 드로다운 체크.
        
        Returns:
            (킬스위치 발동 여부, 이유)
        """
        if not self.initial_equity or self.initial_equity <= 0:
            return False, ""
        
        current_equity = self.balance + self._unrealized_pnl()
        drawdown = (self.initial_equity - current_equity) / self.initial_equity
        
        if drawdown > self.MAX_DRAWDOWN:
            return True, f"MAX_DRAWDOWN_{drawdown:.1%}"
        
        return False, ""

    def _unrealized_pnl(self) -> float:
        """
        미실현 손익 계산.
        
        Returns:
            총 미실현 손익
        """
        total = 0.0
        for sym, pos in self.positions.items():
            price = self._latest_prices.get(sym)
            if not price:
                continue
            
            entry = float(pos.get("entry_price", price))
            qty = float(pos.get("quantity", 0.0))
            action = pos.get("action", "BUY")
            
            if action == "BUY":
                total += (price - entry) * qty
            else:
                total += (entry - price) * qty
        
        return total

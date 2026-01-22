"""
PositionMixin - 포지션 관리 기능
================================

LiveOrchestrator에서 분리된 포지션 관리 메서드들.
- 노출 계산 (_total_open_notional)
- 진입 허용 판단 (_can_enter_position, _entry_permit)
- 사이징 (_calc_position_size)
- 진입/청산 (_enter_position, _close_position, _liquidate_all_positions)
- 리밸런싱 (_rebalance_position)
"""

from __future__ import annotations
import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    pass


class PositionMixin:
    """포지션 관리 믹스인"""

    def _total_open_notional(self) -> float:
        """
        현재 열린 모든 포지션의 총 명목가치 계산.
        
        Returns:
            총 노출 금액 (USDT)
        """
        total = 0.0
        for pos in self.positions.values():
            total += abs(float(pos.get("notional", 0.0)))
        return total

    def _can_enter_position(self, sym: str, notional: float) -> bool:
        """
        신규 포지션 진입 가능 여부 판단.
        
        Args:
            sym: 심볼
            notional: 진입하려는 명목가치
            
        Returns:
            진입 가능 여부
        """
        # 이미 포지션 보유 중
        if sym in self.positions:
            return False
        
        # 쿨다운 중
        if self._cooldown_until.get(sym, 0) > time.time():
            return False
        
        # 최대 노출 체크
        current_notional = self._total_open_notional()
        max_notional = self.balance * self.MAX_CAPITAL_UTIL
        
        if current_notional + notional > max_notional:
            return False
        
        # 최대 포지션 수
        if len(self.positions) >= self.MAX_POSITIONS:
            return False
        
        return True

    def _entry_permit(self, sym: str) -> Tuple[bool, str]:
        """
        진입 허용 여부와 거부 이유 반환.
        
        Args:
            sym: 심볼
            
        Returns:
            (허용 여부, 거부 시 이유)
        """
        # 이미 포지션 보유
        if sym in self.positions:
            return False, "ALREADY_IN_POSITION"
        
        # 쿨다운 체크
        cooldown = self._cooldown_until.get(sym, 0)
        now = time.time()
        if cooldown > now:
            remaining = int(cooldown - now)
            return False, f"COOLDOWN_{remaining}s"
        
        # 킬스위치
        if self._kill_switch_on:
            return False, "KILL_SWITCH"
        
        # 최대 포지션 수
        if len(self.positions) >= self.MAX_POSITIONS:
            return False, "MAX_POSITIONS"
        
        # 최대 노출
        if self._total_open_notional() >= self.balance * self.MAX_CAPITAL_UTIL:
            return False, "MAX_EXPOSURE"
        
        return True, "OK"

    def _calc_position_size(
        self, 
        sym: str, 
        price: float, 
        action: str,
        leverage: float,
        cap_frac: float
    ) -> Tuple[float, float]:
        """
        진입 수량 및 명목가치 계산.
        
        Args:
            sym: 심볼
            price: 현재 가격
            action: "BUY" 또는 "SELL"
            leverage: 레버리지
            cap_frac: 자본 할당 비율
            
        Returns:
            (수량, 명목가치)
        """
        if price <= 0 or cap_frac <= 0:
            return 0.0, 0.0
        
        # 할당 가능 자본
        available = self.balance * self.MAX_CAPITAL_UTIL
        existing = self._total_open_notional()
        remaining = max(0.0, available - existing)
        
        # 이번 진입에 사용할 자본
        target_notional = min(
            self.balance * cap_frac * leverage,
            remaining
        )
        
        # 최소/최대 제한
        target_notional = max(self.MIN_NOTIONAL, target_notional)
        target_notional = min(self.MAX_NOTIONAL_PER_POSITION, target_notional)
        
        quantity = target_notional / price
        
        return quantity, target_notional

    async def _enter_position(
        self,
        sym: str,
        action: str,
        price: float,
        quantity: float,
        notional: float,
        leverage: float,
        cap_frac: float,
        reason: str = ""
    ) -> bool:
        """
        포지션 진입 실행.
        
        Args:
            sym: 심볼
            action: "BUY" 또는 "SELL"
            price: 진입 가격
            quantity: 수량
            notional: 명목가치
            leverage: 레버리지
            cap_frac: 자본 할당 비율
            reason: 진입 이유
            
        Returns:
            성공 여부
        """
        from . import now_ms
        
        if quantity <= 0 or notional <= 0:
            return False
        
        # 진입 가능 여부 재확인
        can_enter, deny_reason = self._entry_permit(sym)
        if not can_enter:
            self._log(f"[DENY] {sym} entry blocked: {deny_reason}")
            return False
        
        ts = now_ms()
        
        # 포지션 기록
        pos = {
            "symbol": sym,
            "action": action,
            "entry_price": price,
            "quantity": quantity,
            "notional": notional,
            "leverage": leverage,
            "cap_frac": cap_frac,
            "entry_time": ts,
            "hold_limit": self.MAX_POSITION_HOLD_SEC * 1000,
            "reason": reason,
            "peak_equity": self.balance,
        }
        self.positions[sym] = pos
        
        # 브로커에 주문 전송 (PAPER/LIVE 모드에 따라)
        try:
            await self._submit_order(sym, action, quantity, price)
        except Exception as e:
            self._log_err(f"[ERR] {sym} order failed: {e}")
            del self.positions[sym]
            return False
        
        self._log(f"[ENTER] {sym} {action} qty={quantity:.4f} @ {price:.4f} notional={notional:.1f}")
        self._entry_streak[sym] = self._entry_streak.get(sym, 0) + 1
        
        # 상태 저장
        self._persist_state()
        
        return True

    async def _close_position(
        self,
        sym: str,
        exit_price: float,
        reason: str = "NORMAL"
    ) -> Optional[Dict[str, Any]]:
        """
        포지션 청산 실행.
        
        Args:
            sym: 심볼
            exit_price: 청산 가격
            reason: 청산 이유
            
        Returns:
            청산된 포지션 정보 또는 None
        """
        from . import now_ms
        
        pos = self.positions.get(sym)
        if not pos:
            return None
        
        ts = now_ms()
        entry_price = float(pos.get("entry_price", exit_price))
        quantity = float(pos.get("quantity", 0.0))
        action = pos.get("action", "BUY")
        
        # PnL 계산
        if action == "BUY":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        # 청산 주문 전송
        close_action = "SELL" if action == "BUY" else "BUY"
        try:
            await self._submit_order(sym, close_action, quantity, exit_price)
        except Exception as e:
            self._log_err(f"[ERR] {sym} close order failed: {e}")
            return None
        
        # 잔고 업데이트
        self.balance += pnl
        
        # 거래 기록
        trade = {
            "symbol": sym,
            "action": action,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": pnl,
            "reason": reason,
            "entry_time": pos.get("entry_time"),
            "exit_time": ts,
            "hold_sec": (ts - pos.get("entry_time", ts)) / 1000,
        }
        self.trade_tape.append(trade)
        
        # 포지션 제거
        closed_pos = self.positions.pop(sym, None)
        
        self._log(f"[CLOSE] {sym} @ {exit_price:.4f} PnL={pnl:+.2f} ({reason})")
        
        # 쿨다운 설정
        self._mark_exit_and_cooldown(sym, reason, ts)
        
        # 상태 저장
        self._persist_state()
        
        return closed_pos

    async def _liquidate_all_positions(self, reason: str = "EMERGENCY") -> int:
        """
        모든 포지션 강제 청산.
        
        Args:
            reason: 청산 이유
            
        Returns:
            청산된 포지션 수
        """
        closed_count = 0
        symbols = list(self.positions.keys())
        
        for sym in symbols:
            price = self._latest_prices.get(sym)
            if price:
                result = await self._close_position(sym, price, reason)
                if result:
                    closed_count += 1
        
        self._log(f"[LIQUIDATE] Closed {closed_count}/{len(symbols)} positions ({reason})")
        return closed_count

    async def _rebalance_position(
        self,
        sym: str,
        new_cap_frac: float,
        current_price: float
    ) -> bool:
        """
        기존 포지션 리밸런싱.
        
        Args:
            sym: 심볼
            new_cap_frac: 새 자본 할당 비율
            current_price: 현재 가격
            
        Returns:
            성공 여부
        """
        pos = self.positions.get(sym)
        if not pos:
            return False
        
        old_cap_frac = float(pos.get("cap_frac", 0.0))
        leverage = float(pos.get("leverage", self.leverage))
        quantity = float(pos.get("quantity", 0.0))
        action = pos.get("action", "BUY")
        
        # 변화량 계산
        frac_delta = new_cap_frac - old_cap_frac
        if abs(frac_delta) < 0.01:  # 1% 미만 변화는 무시
            return True
        
        notional_delta = self.balance * frac_delta * leverage
        qty_delta = abs(notional_delta / current_price)
        
        try:
            if notional_delta > 0:
                # 증가: 추가 매수/매도
                await self._submit_order(sym, action, qty_delta, current_price)
                pos["quantity"] = quantity + qty_delta
            else:
                # 감소: 부분 청산
                close_action = "SELL" if action == "BUY" else "BUY"
                qty_delta = min(qty_delta, quantity)
                await self._submit_order(sym, close_action, qty_delta, current_price)
                pos["quantity"] = quantity - qty_delta
            
            pos["cap_frac"] = new_cap_frac
            pos["notional"] = pos["quantity"] * current_price
            
            self._log(f"[REBAL] {sym} cap_frac {old_cap_frac:.2%} -> {new_cap_frac:.2%}")
            return True
            
        except Exception as e:
            self._log_err(f"[ERR] {sym} rebalance failed: {e}")
            return False

    async def _submit_order(
        self,
        sym: str,
        side: str,
        quantity: float,
        price: float
    ) -> None:
        """
        주문 제출 (브로커로 전달).
        
        하위 클래스 또는 조합에서 구현해야 함.
        Paper 모드: paper_broker 사용
        Live 모드: 거래소 API 사용
        """
        # 기본 구현은 pass - 실제 오케스트레이터에서 오버라이드
        pass

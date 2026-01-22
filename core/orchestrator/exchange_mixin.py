"""
ExchangeMixin - 거래소 API 관련 기능
====================================

LiveOrchestrator에서 분리된 거래소 연동 메서드들.
- CCXT 호출 래퍼 (_ccxt_call)
- 거래소 설정 초기화 (init_exchange_settings)
- 포지션 모드/레버리지 동기화 (_sync_position_mode, _sync_leverage)
"""

from __future__ import annotations
import asyncio
import random
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass


class ExchangeMixin:
    """거래소 API 믹스인"""
    
    # 상수 (OrchestratorBase에서 초기화됨)
    MAX_RETRY: int = 3
    RETRY_BASE_SEC: float = 0.5
    
    async def _ccxt_call(
        self,
        label: str,
        fn: Callable,
        *args,
        semaphore=None,
        **kwargs,
    ) -> Any:
        """
        Best-effort CCXT 호출 래퍼.
        
        기능:
        - 동시성 제한 (semaphore)
        - 지수 백오프 + 지터를 사용한 재시도
        
        Args:
            label: 로깅용 라벨 (예: "fetch_ticker")
            fn: 호출할 async 함수
            *args: 함수 인자
            semaphore: 동시성 제한용 세마포어 (없으면 self._net_sem 사용)
            **kwargs: 함수 키워드 인자
            
        Returns:
            함수 반환값
            
        Raises:
            Exception: 재시도 횟수 초과 또는 재시도 불가능한 에러
        """
        for attempt in range(1, self.MAX_RETRY + 1):
            try:
                async with (semaphore or self._net_sem):
                    return await fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                is_retryable = any(k in msg for k in [
                    "RequestTimeout", "DDoSProtection", "ExchangeNotAvailable",
                    "NetworkError", "ETIMEDOUT", "ECONNRESET", "502", "503", "504"
                ])
                if (attempt >= self.MAX_RETRY) or (not is_retryable):
                    raise
                backoff = (self.RETRY_BASE_SEC * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                self._log_err(f"[WARN] {label} retry {attempt}/{self.MAX_RETRY} err={msg} sleep={backoff:.2f}s")
                await asyncio.sleep(backoff)

    async def init_exchange_settings(self) -> None:
        """
        거래소 설정 초기화.
        
        순서:
        1. 마켓 정보 로드
        2. 포지션 모드 동기화 (Hedge/One-way)
        3. 레버리지 동기화
        """
        try:
            await self.exchange.load_markets()
        except Exception:
            pass
        await self._sync_position_mode()
        await self._sync_leverage()

    async def _sync_position_mode(self) -> None:
        """
        거래소 포지션 모드 확인 및 동기화.
        
        Bybit Hedge 모드인지 One-way 모드인지 확인하여
        self._is_hedge_mode와 self._position_mode를 설정.
        """
        if not hasattr(self.exchange, "fetch_position_mode"):
            return
        
        try:
            mode = await self.exchange.fetch_position_mode()
            hedge = False
            
            if isinstance(mode, bool):
                hedge = bool(mode)
            elif isinstance(mode, str):
                hedge = mode.lower() in ("hedge", "both", "dual")
            elif isinstance(mode, dict):
                if "hedgeMode" in mode:
                    hedge = bool(mode.get("hedgeMode"))
                elif "positionMode" in mode:
                    hedge = str(mode.get("positionMode")).lower() in ("hedge", "both", "dual")
                elif "mode" in mode:
                    hedge = str(mode.get("mode")).lower() in ("hedge", "both", "dual")
            
            self._is_hedge_mode = bool(hedge)
            self._position_mode = "hedge" if self._is_hedge_mode else "oneway"
            self._log(f"[EXCHANGE] position mode: {self._position_mode}")
            
        except Exception as e:
            self._log_err(f"[EXCHANGE] fetch_position_mode failed: {e}")

    async def _sync_leverage(self) -> None:
        """
        모든 심볼에 대해 레버리지 설정 동기화.
        """
        if not hasattr(self.exchange, "set_leverage"):
            return
        
        target = float(self.leverage)
        
        for sym in self.SYMBOLS:
            try:
                await self._ccxt_call(
                    "set_leverage",
                    self.exchange.set_leverage,
                    int(target),
                    sym,
                )
                self._log(f"[EXCHANGE] leverage synced: {sym} -> {target:.1f}x")
            except Exception as e:
                self._log_err(f"[EXCHANGE] set_leverage failed: {sym} target={target:.1f}x err={e}")

    def _position_idx_for_side(self, position_side: str) -> int:
        """
        Bybit Hedge 모드에서 positionIdx 결정.
        
        Args:
            position_side: "LONG" 또는 "SHORT"
            
        Returns:
            0 (One-way), 1 (Hedge-Long), 2 (Hedge-Short)
        """
        if self._is_hedge_mode:
            return 1 if position_side == "LONG" else 2
        return 0

    def _should_force_market(self, sym: str) -> bool:
        """
        시장 상황에 따라 시장가 주문을 강제할지 결정.
        
        조건:
        - 오더북이 준비되지 않음
        - 스프레드가 너무 넓음
        - 변동성이 너무 높음
        
        Returns:
            True면 시장가 주문 사용
        """
        ob = self.orderbook.get(sym)
        if not ob or not ob.get("ready"):
            return True
        
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return True
        
        try:
            bid = float(bids[0][0])
            ask = float(asks[0][0])
        except Exception:
            return True
        
        mid = (bid + ask) / 2.0 if bid and ask else 0.0
        if mid <= 0:
            return True
        
        spread_pct = (ask - bid) / mid
        if spread_pct > self.SPREAD_PCT_MAX:
            return True
        
        closes = list(self.ohlcv_buffer.get(sym, []))
        mu_bar, sigma_bar = self._compute_returns_and_vol(closes)
        if sigma_bar is not None and float(sigma_bar) > float(self.VOLATILITY_MARKET_THRESHOLD):
            return True
        
        return False

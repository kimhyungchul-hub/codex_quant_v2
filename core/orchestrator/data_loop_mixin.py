"""
DataLoopMixin - 데이터 수집 루프 기능
=====================================

LiveOrchestrator에서 분리된 비동기 데이터 수집 메서드들.
- 티커 가격 수집 (fetch_prices_loop)
- OHLCV 수집 (preload_all_ohlcv, fetch_ohlcv_loop)
- 호가창 수집 (fetch_orderbook_loop)
"""

from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass


class DataLoopMixin:
    """데이터 수집 루프 믹스인"""

    async def fetch_prices_loop(self) -> None:
        """
        티커 가격을 주기적으로 수집하는 루프.
        
        self._ticker_interval_sec 간격으로 모든 심볼의 가격을 업데이트.
        """
        interval = getattr(self, "_ticker_interval_sec", 0.5)
        
        while not self._shutdown:
            try:
                await self._fetch_all_prices()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log_err(f"[ERR] fetch_prices: {e}")
            
            await asyncio.sleep(interval)

    async def _fetch_all_prices(self) -> None:
        """
        모든 심볼의 티커 가격 수집.
        """
        if not self.exchange:
            return
        
        try:
            # Bybit의 경우 한 번에 모든 티커 가져오기
            tickers = await self._ccxt_call("fetch_tickers", self.exchange.fetch_tickers, self.symbols)
            
            for sym in self.symbols:
                ticker = tickers.get(sym)
                if ticker:
                    price = ticker.get("last") or ticker.get("close")
                    if price:
                        self._latest_prices[sym] = float(price)
        except Exception as e:
            # 개별 심볼 시도
            for sym in self.symbols:
                try:
                    ticker = await self._ccxt_call("fetch_ticker", self.exchange.fetch_ticker, sym)
                    if ticker:
                        price = ticker.get("last") or ticker.get("close")
                        if price:
                            self._latest_prices[sym] = float(price)
                except Exception:
                    pass

    async def preload_all_ohlcv(self) -> None:
        """
        시작 시 모든 심볼의 OHLCV 데이터 프리로드.
        
        변동성 계산 등에 필요한 초기 캔들 데이터를 수집.
        """
        if not self.exchange:
            return
        
        limit = getattr(self, "OHLCV_PRELOAD_LIMIT", 100)
        timeframe = getattr(self, "OHLCV_TIMEFRAME", "1m")
        
        tasks = []
        for sym in self.symbols:
            tasks.append(self._fetch_ohlcv_single(sym, timeframe, limit))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        loaded = sum(1 for s in self.symbols if s in self._ohlcv_buffer)
        self._log(f"[OHLCV] Preloaded {loaded}/{len(self.symbols)} symbols")

    async def _fetch_ohlcv_single(
        self, 
        sym: str, 
        timeframe: str = "1m",
        limit: int = 100
    ) -> None:
        """
        단일 심볼의 OHLCV 수집.
        
        Args:
            sym: 심볼
            timeframe: 캔들 주기
            limit: 가져올 캔들 수
        """
        try:
            ohlcv = await self._ccxt_call(
                "fetch_ohlcv",
                self.exchange.fetch_ohlcv, 
                sym, 
                timeframe, 
                limit=limit
            )
            
            if not ohlcv:
                return
            
            # [timestamp, open, high, low, close, volume]
            opens = [float(c[1]) for c in ohlcv]
            highs = [float(c[2]) for c in ohlcv]
            lows = [float(c[3]) for c in ohlcv]
            closes = [float(c[4]) for c in ohlcv]
            volumes = [float(c[5]) for c in ohlcv]
            timestamps = [int(c[0]) for c in ohlcv]
            
            self._ohlcv_buffer[sym] = {
                "timestamps": timestamps,
                "opens": opens,
                "highs": highs,
                "lows": lows,
                "closes": closes,
                "volumes": volumes,
                "last_update": timestamps[-1] if timestamps else 0,
            }
        except Exception as e:
            self._log_err(f"[ERR] OHLCV {sym}: {e}")

    async def fetch_ohlcv_loop(self) -> None:
        """
        OHLCV를 주기적으로 업데이트하는 루프.
        """
        interval = getattr(self, "_ohlcv_interval_sec", 60)
        timeframe = getattr(self, "OHLCV_TIMEFRAME", "1m")
        
        while not self._shutdown:
            try:
                for sym in self.symbols:
                    await self._fetch_ohlcv_single(sym, timeframe, limit=5)
                    
                    # 레이트 리밋 방지
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log_err(f"[ERR] fetch_ohlcv_loop: {e}")
            
            await asyncio.sleep(interval)

    async def fetch_orderbook_loop(self) -> None:
        """
        호가창을 주기적으로 수집하는 루프.
        """
        interval = getattr(self, "_orderbook_interval_sec", 2.0)
        depth = getattr(self, "ORDERBOOK_DEPTH", 10)
        
        while not self._shutdown:
            try:
                await self._fetch_all_orderbooks(depth)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log_err(f"[ERR] fetch_orderbook: {e}")
            
            await asyncio.sleep(interval)

    async def _fetch_all_orderbooks(self, depth: int = 10) -> None:
        """
        모든 심볼의 호가창 수집.
        
        Args:
            depth: 호가 깊이
        """
        if not self.exchange:
            return
        
        # 병렬 수집
        tasks = []
        for sym in self.symbols:
            tasks.append(self._fetch_orderbook_single(sym, depth))
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_orderbook_single(
        self, 
        sym: str, 
        depth: int = 10
    ) -> None:
        """
        단일 심볼의 호가창 수집.
        
        Args:
            sym: 심볼
            depth: 호가 깊이
        """
        try:
            ob = await self._ccxt_call(
                "fetch_order_book",
                self.exchange.fetch_order_book, 
                sym, 
                limit=depth
            )
            
            if ob:
                self._orderbook_buffer[sym] = {
                    "bids": ob.get("bids", []),
                    "asks": ob.get("asks", []),
                    "timestamp": ob.get("timestamp", 0),
                }
        except Exception:
            pass  # 호가창 수집 실패는 조용히 무시

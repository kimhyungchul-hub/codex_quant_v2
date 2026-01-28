import asyncio
import time
import numpy as np
import os
from collections import deque
from config import *
from utils.helpers import now_ms
import config

class DataManager:
    def __init__(self, orchestrator, symbols=None, data_exchange=None):
        self.orch = orchestrator
        # Use a dedicated public-data exchange when provided (so live orders can be testnet
        # while MC inputs match paper/mainnet market data distribution).
        self.exchange = data_exchange or getattr(orchestrator, "data_exchange", None) or orchestrator.exchange
        # Ensure symbols are canonical (accept user shorthand)
        from utils.helpers import normalize_symbol

        if symbols is not None:
            raw_symbols = list(symbols)
            self.symbols = [normalize_symbol(s) for s in symbols]
            print(f"[DATA_DEBUG] DataManager.__init__: raw_symbols={raw_symbols[:3]} -> normalized={self.symbols[:3]}", flush=True)
        else:
            self.symbols = SYMBOLS
            print(f"[DATA_DEBUG] DataManager.__init__: using config SYMBOLS={self.symbols[:3]}", flush=True)
        
        self.market = {s: {"price": None, "last": None, "bid": None, "ask": None, "ts": 0} for s in self.symbols}
        self.ohlcv_buffer = {s: deque(maxlen=OHLCV_PRELOAD_LIMIT) for s in self.symbols}
        self.orderbook = {s: {"ts": 0, "ready": False, "bids": [], "asks": []} for s in self.symbols}
        
        self._last_kline_ts = {s: 0 for s in self.symbols}
        self._last_kline_ok_ms = {s: 0 for s in self.symbols}
        self._preloaded = {s: False for s in self.symbols}
        self._last_feed_ok_ms = 0
        self._markets_loaded = False

        # Reduce dashboard/log spam: only log orderbook readiness transitions.
        # (Logging every fetch produces huge payloads and slows UI updates.)
        self._orderbook_ready_prev = {s: None for s in self.symbols}
        
        self.data_updated_event = asyncio.Event()

    def _valid_symbols(self):
        markets = getattr(self.exchange, "markets", {}) or {}
        if markets:
            valid = [s for s in self.symbols if s in markets]
            # 디버그: 심볼 매칭 상태 출력 (첫 5회만)
            if not hasattr(self, "_valid_symbols_debug_count"):
                self._valid_symbols_debug_count = 0
            self._valid_symbols_debug_count += 1
            if self._valid_symbols_debug_count <= 5:
                missing = [s for s in self.symbols if s not in markets]
                market_samples = list(markets.keys())[:5]
                print(f"[DATA_DEBUG] _valid_symbols: self.symbols={self.symbols[:3]} | valid={valid[:3]} | missing={missing[:3]} | market_samples={market_samples}", flush=True)
            return valid
        return list(self.symbols)

    async def fetch_prices_loop(self):
        count = 0
        while True:
            count += 1
            if count % 10 == 0:
                print(f"💓 [HEARTBEAT] fetch_prices_loop active (cycle {count})", flush=True)
            try:
                if not self._markets_loaded:
                    try:
                        await self.orch._ccxt_call("load_markets(prices)", self.exchange.load_markets)
                        self._markets_loaded = True
                    except Exception:
                        pass
                valid_syms = self._valid_symbols()
                tickers = await self.orch._ccxt_call("fetch_tickers", self.exchange.fetch_tickers, valid_syms)
                ts = now_ms()
                ok_any = False
                if count % 10 == 0:
                    print(f"DEBUG: valid_syms count={len(valid_syms)} samples={valid_syms[:2]}", flush=True)
                    ticker_keys_sample = list(tickers.keys())[:5] if tickers else []
                    print(f"DEBUG: tickers keys sample={ticker_keys_sample}", flush=True)
                    print(f"DEBUG: tickers sample for {valid_syms[0]}: {tickers.get(valid_syms[0])}", flush=True)
                for s in valid_syms:
                    base = s.split(":")[0] if isinstance(s, str) else s
                    t = (tickers.get(s) or tickers.get(base) or {})
                    # 추가 디버그: 첫 3회에는 ticker 매칭 상태 출력
                    if count <= 3:
                        t_direct = tickers.get(s)
                        t_base = tickers.get(base)
                        print(f"[TICKER_DEBUG] {s} | direct={t_direct is not None} | base={base} base_found={t_base is not None} | last={t.get('last')}", flush=True)
                    base = s.split(":")[0] if isinstance(s, str) else s
                    t = (tickers.get(s) or tickers.get(base) or {})
                    last = t.get("last")
                    bid = t.get("bid")
                    ask = t.get("ask")
                    # Some venues/markets may omit `last` but still provide bid/ask.
                    # Use mid as a fallback so paper PnL marking stays live.
                    if last is None and bid is not None and ask is not None:
                        try:
                            last = 0.5 * (float(bid) + float(ask))
                        except Exception:
                            last = None
                    if last is not None:
                        self.market[s]["price"] = float(last)
                        self.market[s]["last"] = float(last)
                        if count % 10 == 0 and s == valid_syms[0]:
                            print(f"[DEBUG] Market updated {s}: price={last}", flush=True)
                        ok_any = True
                    if bid is not None:
                        self.market[s]["bid"] = float(bid)
                        ok_any = True
                    if ask is not None:
                        self.market[s]["ask"] = float(ask)
                        ok_any = True
                    if (last is not None) or (bid is not None) or (ask is not None):
                        self.market[s]["ts"] = ts
                if ok_any:
                    self._last_feed_ok_ms = ts
                    self.data_updated_event.set()
            except Exception as e:
                self.orch._log_err(f"[ERR] fetch_tickers: {e}")
            ticker_sleep = float(getattr(config, "TICKER_SLEEP_SEC", 1.0))
            await asyncio.sleep(ticker_sleep)

    async def preload_all_ohlcv(self, limit: int = OHLCV_PRELOAD_LIMIT):
        # Ensure markets are loaded once up-front (avoids slow implicit load_markets per call).
        try:
            if hasattr(self.orch, "_ccxt_call"):
                await self.orch._ccxt_call("load_markets(preload)", self.exchange.load_markets)
            else:
                await self.exchange.load_markets()
        except Exception as e:
            print(f"[WARN] preload load_markets: {e}", flush=True)

        async def _fetch_one_symbol(sym: str):
            """Helper to fetch OHLCV for a single symbol."""
            try:
                print(f"[PRELOAD] Fetching {sym}...", flush=True)
                if hasattr(self.orch, "_ccxt_call"):
                    ohlcv = await self.orch._ccxt_call(
                        f"fetch_ohlcv(preload) {sym}",
                        self.exchange.fetch_ohlcv,
                        sym,
                        timeframe=TIMEFRAME,
                        limit=limit,
                    )
                else:
                    ohlcv = await self.exchange.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=limit)
                if not ohlcv:
                    print(f"[PRELOAD] {sym} - no data received", flush=True)
                    return
                self.ohlcv_buffer[sym].clear()
                last_ts = 0
                for c in ohlcv:
                    ts_ms = int(c[0])
                    close_price = float(c[4])
                    self.ohlcv_buffer[sym].append(close_price)
                    last_ts = ts_ms
                self._last_kline_ts[sym] = last_ts
                self._last_kline_ok_ms[sym] = now_ms()
                self._preloaded[sym] = True
                msg = f"[PRELOAD] {sym} candles={len(self.ohlcv_buffer[sym])}"
                print(msg, flush=True)
                self.orch._log(msg)
            except Exception as e:
                err_msg = f"[ERR] preload_ohlcv {sym}: {e}"
                print(err_msg, flush=True)
                self.orch._log_err(err_msg)

        # Fetch all symbols concurrently
        await asyncio.gather(*[_fetch_one_symbol(sym) for sym in self.symbols])
        self.data_updated_event.set()



    async def fetch_ohlcv_loop(self):
        while True:
            start = now_ms()
            try:
                valid_syms = self._valid_symbols()
                
                async def _update_one_ohlcv(sym: str):
                    try:
                        ohlcv = await self.orch._ccxt_call(
                            f"fetch_ohlcv {sym}",
                            self.exchange.fetch_ohlcv,
                            sym, timeframe=TIMEFRAME, limit=OHLCV_REFRESH_LIMIT
                        )
                        if not ohlcv:
                            return
                        last = ohlcv[-1]
                        ts_ms = int(last[0])
                        close_price = float(last[4])

                        if ts_ms != self._last_kline_ts[sym]:
                            self.ohlcv_buffer[sym].append(close_price)
                            self._last_kline_ts[sym] = ts_ms
                            self._last_kline_ok_ms[sym] = now_ms()
                            self.data_updated_event.set()
                    except Exception as e_sym:
                        self.orch._log_err(f"[ERR] fetch_ohlcv {sym}: {e_sym}")

                # Parallel fetch OHLCV for all symbols
                await asyncio.gather(*[_update_one_ohlcv(sym) for sym in valid_syms])
            except Exception as e:
                self.orch._log_err(f"[ERR] fetch_ohlcv(loop): {e}")

            elapsed = (now_ms() - start) / 1000.0
            sleep_left = max(0.5, OHLCV_SLEEP_SEC - elapsed) # Faster refresh if needed
            await asyncio.sleep(sleep_left)

    async def fetch_orderbook_loop(self):
        count = 0
        while True:
            count += 1
            if count % 5 == 0:
                print(f"💓 [HEARTBEAT] fetch_orderbook_loop active (cycle {count})", flush=True)
            start = now_ms()
            valid_syms = self._valid_symbols()
            
            # Use Semaphore to avoid overwhelming the exchange API
            sem = asyncio.Semaphore(5) 
            
            async def _update_one_ob(sym: str):
                async with sem:
                    try:
                        ob = await self.orch._ccxt_call(
                            f"fetch_orderbook {sym}",
                            self.exchange.fetch_order_book,
                            sym, limit=ORDERBOOK_DEPTH
                        )
                        bids = (ob.get("bids") or [])[:ORDERBOOK_DEPTH]
                        asks = (ob.get("asks") or [])[:ORDERBOOK_DEPTH]
                        ready = bool(bids) and bool(asks)
                        self.orderbook[sym]["bids"] = bids
                        self.orderbook[sym]["asks"] = asks
                        self.orderbook[sym]["ready"] = ready
                        self.orderbook[sym]["ts"] = now_ms()
                        prev_ready = self._orderbook_ready_prev.get(sym)
                        if ready and prev_ready is not True:
                            self.orch._log(f"[ORDERBOOK] {sym} ready bids={len(bids)} asks={len(asks)}")
                        if (not ready) and prev_ready is True:
                            self.orch._log_err(f"[ORDERBOOK] {sym} became empty bids={len(bids)} asks={len(asks)}")
                        self._orderbook_ready_prev[sym] = bool(ready)
                    except Exception as e_sym:
                        self.orderbook[sym]["ready"] = False
                        self._orderbook_ready_prev[sym] = False
                        self.orch._log_err(f"[ERR] fetch_orderbook {sym}: {repr(e_sym)}")
                    
                    # Small delay between symbol batches to stay within rate limits if needed
                    await asyncio.sleep(ORDERBOOK_SYMBOL_INTERVAL_SEC)

            # Parallel fetch orderbooks
            await asyncio.gather(*[_update_one_ob(sym) for sym in valid_syms])

            self.data_updated_event.set()
            elapsed = (now_ms() - start) / 1000.0
            sleep_left = max(0.1, ORDERBOOK_SLEEP_SEC - elapsed)
            await asyncio.sleep(sleep_left)

    def get_btc_corr(self, sym: str, window: int = 60) -> float:
        """
        BTC와의 최근 수익률 상관관계를 계산 (최근 window개 캔들 기준)
        """
        if sym.startswith("BTC"):
            return 1.0
        
        btc_sym = self.symbols[0] if self.symbols else "BTC/USDT:USDT"
        if not btc_sym.startswith("BTC"):
            # SYMBOLS[0]이 BTC가 아니면 명시적으로 찾기
            for s in self.symbols:
                if s.startswith("BTC"):
                    btc_sym = s
                    break
        
        btc_closes = list(self.ohlcv_buffer.get(btc_sym) or [])
        sym_closes = list(self.ohlcv_buffer.get(sym) or [])
        
        if len(btc_closes) < window + 1 or len(sym_closes) < window + 1:
            return 0.0
            
        # 길이 맞추기
        min_len = min(len(btc_closes), len(sym_closes))
        btc_closes = btc_closes[-min_len:]
        sym_closes = sym_closes[-min_len:]
        
        try:
            btc_ret = np.diff(np.log(np.array(btc_closes[-window-1:], dtype=np.float64)))
            sym_ret = np.diff(np.log(np.array(sym_closes[-window-1:], dtype=np.float64)))
            
            if len(btc_ret) < window or len(sym_ret) < window:
                return 0.0
                
            corr = np.corrcoef(btc_ret, sym_ret)[0, 1]
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except Exception:
            return 0.0

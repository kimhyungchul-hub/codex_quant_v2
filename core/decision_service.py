from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

from engines.engine_hub import EngineHub
from utils.helpers import now_ms


class DecisionService:
    """Handles decision computation: context building, engine calls, caching.

    Isolated from orchestrator to keep decision logic modular and testable.
    """

    def __init__(
        self,
        *,
        hub: EngineHub,
        symbols: list[str],
        get_market_data: Callable[[str], Dict[str, Any]],
        get_positions: Callable[[], Dict[str, Dict[str, Any]]],
        get_balance: Callable[[], float],
        log: Callable[[str], None],
        log_err: Callable[[str], None],
        decision_cache_maxlen: int = 1000,
        decision_eval_min_interval_sec: float = 0.2,
    ) -> None:
        self.hub = hub
        self.symbols = symbols
        self._get_market_data = get_market_data
        self._get_positions = get_positions
        self._get_balance = get_balance
        self._log = log
        self._log_err = log_err

        self.decision_cache: Dict[str, Dict[str, Any]] = {}
        self.decision_cache_maxlen = decision_cache_maxlen
        self.decision_eval_min_interval_sec = decision_eval_min_interval_sec
        self._last_decision_eval_ms: Dict[str, int] = {}

    def _build_decide_ctx(
        self,
        *,
        sym: str,
        ts_ms: int,
        positions: Dict[str, Dict[str, Any]],
        balance: float,
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Simplified ctx building - in real, more data
        ctx = {
            "symbol": sym,
            "ts_ms": ts_ms,
            "balance": balance,
            "positions": positions,
            "market": market_data,
        }
        return ctx

    async def get_decision(
        self,
        *,
        sym: str,
        ts_ms: int,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        # Check cache and interval
        cache_key = f"{sym}_{ts_ms}"
        if not force and cache_key in self.decision_cache:
            return self.decision_cache[cache_key]

        last_eval = self._last_decision_eval_ms.get(sym, 0)
        if not force and (ts_ms - last_eval) / 1000 < self.decision_eval_min_interval_sec:
            return None

        # Build ctx
        positions = self._get_positions()
        balance = self._get_balance()
        market_data = self._get_market_data(sym)
        ctx = self._build_decide_ctx(
            sym=sym, ts_ms=ts_ms, positions=positions, balance=balance, market_data=market_data
        )

        # Call engine
        try:
            decision = await self.hub.decide(sym, ctx)
            if decision:
                self.decision_cache[cache_key] = decision
                if len(self.decision_cache) > self.decision_cache_maxlen:
                    # Simple LRU - remove oldest
                    oldest_key = next(iter(self.decision_cache))
                    del self.decision_cache[oldest_key]
                self._last_decision_eval_ms[sym] = ts_ms
            return decision
        except Exception as e:
            self._log_err(f"[DECISION_ERR] {sym}: {e}")
            return None

    async def decision_loop_step(self, ts_ms: int) -> None:
        # Simplified - decide for all symbols
        tasks = [self.get_decision(sym=sym, ts_ms=ts_ms) for sym in self.symbols]
        await asyncio.gather(*tasks, return_exceptions=True)
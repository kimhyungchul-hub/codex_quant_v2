from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from utils.helpers import now_ms


class DashboardAdapter:
    """Handles dashboard row generation and broadcasting.

    Isolated from orchestrator to keep UI logic modular.
    """

    def __init__(
        self,
        *,
        symbols: list[str],
        get_market_data: Callable[[str], Dict[str, Any]],
        get_positions: Callable[[], Dict[str, Dict[str, Any]]],
        get_balance: Callable[[], float],
        get_decision_cache: Callable[[], Dict[str, Dict[str, Any]]],
        get_exec_stats: Callable[[], Dict[str, Dict[str, Any]]],
        get_logs: Callable[[], List[Dict[str, Any]]],
        get_equity_history: Callable[[], List[Dict[str, Any]]],
        log: Callable[[str], None],
        log_err: Callable[[str], None],
    ) -> None:
        self.symbols = symbols
        self._get_market_data = get_market_data
        self._get_positions = get_positions
        self._get_balance = get_balance
        self._get_decision_cache = get_decision_cache
        self._get_exec_stats = get_exec_stats
        self._get_logs = get_logs
        self._get_equity_history = get_equity_history
        self._log = log
        self._log_err = log_err

    def _row(
        self,
        sym: str,
        price: Any,
        ts: int,
        decision: Optional[Dict[str, Any]],
        candles: int,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Simplified row generation - in real, more fields
        positions = self._get_positions()
        balance = self._get_balance()
        market_data = self._get_market_data(sym)

        row = {
            "symbol": sym,
            "price": price,
            "ts": ts,
            "decision": decision,
            "candles": candles,
            "balance": balance,
            "position": positions.get(sym),
            "market": market_data,
        }
        return row

    def _rows_snapshot(self, ts_ms: int, *, apply_trades: bool = False) -> list[Dict[str, Any]]:
        rows = []
        for sym in self.symbols:
            # Simplified snapshot
            market_data = self._get_market_data(sym)
            price = market_data.get("price")
            candles = len(market_data.get("candles", []))
            decision_cache = self._get_decision_cache()
            decision = decision_cache.get(sym, {}).get("decision")

            row = self._row(sym, price, ts_ms, decision, candles)
            rows.append(row)
        return rows

    async def _rows_snapshot_cached(self, ts_ms: int) -> list[Dict[str, Any]]:
        # Simplified cached version - in real, add caching logic
        return self._rows_snapshot(ts_ms)

    async def decision_loop_step(self, ts: int) -> None:
        # Simplified - just snapshot and broadcast if dashboard exists
        # In real, would call dashboard.broadcast
        pass
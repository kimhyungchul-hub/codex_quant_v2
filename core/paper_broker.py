from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

from engines.probability_methods import _approx_p_pos_and_ev_hold
from engines.mc.constants import SECONDS_PER_YEAR
from utils.helpers import now_ms


class PaperBroker:
    """Handles paper trading operations: entry, exit, position management, exit policies.

    Isolated from orchestrator to keep trading logic modular and testable.
    """

    def __init__(
        self,
        *,
        symbols: list[str],
        update_balance: Callable[[float], None],
        update_positions: Callable[[str, Optional[Dict[str, Any]]], None],
        append_trade: Callable[[Dict[str, Any]], None],
        get_balance: Callable[[], float],
        get_max_leverage: Callable[[], float],
        log: Callable[[str], None],
        log_err: Callable[[str], None],
    ) -> None:
        self.symbols = symbols
        self._update_balance = update_balance
        self._update_positions = update_positions
        self._append_trade = append_trade
        self._get_balance = get_balance
        self._get_max_leverage = get_max_leverage
        self._log = log
        self._log_err = log_err

        # Exit policy state per symbol
        self._exit_policy_state: Dict[str, Dict[str, Any]] = {}

    def _pmaker_paper_sigma(self, closes: list[float], window: int = 60) -> float:
        if len(closes) < window:
            return 0.0
        prices = np.array(closes[-window:])
        returns = np.diff(np.log(prices))
        return float(np.std(returns, ddof=1) * np.sqrt(SECONDS_PER_YEAR))

    def _pmaker_paper_momentum_z(self, closes: list[float], sigma: float, window: int = 10) -> float:
        if len(closes) < window + 1 or sigma <= 0:
            return 0.0
        prices = np.array(closes[-(window + 1):])
        momentum = (prices[-1] - prices[0]) / prices[0]
        return float(momentum / (sigma * np.sqrt(window / SECONDS_PER_YEAR)))

    def _pmaker_paper_probe_tick(self, *, sym: str, ts_ms: int, ctx: Dict[str, Any]) -> None:
        # Placeholder for probe tick logic if needed
        pass

    def _paper_mark_position(self, sym: str, mark_price: Optional[float], ts_ms: int) -> None:
        # Placeholder - mark-to-market logic if needed
        pass

    def _paper_append_trade(self, trade: Dict[str, Any]) -> None:
        self._append_trade(trade)

    def _paper_fee_roundtrip_from_engine_meta(self, meta: Dict[str, Any]) -> Optional[float]:
        fee_rate = meta.get("fee_rate", 0.001)
        return float(fee_rate * 2)  # roundtrip

    @staticmethod
    def _paper_exec_mode() -> str:
        return "paper"

    @staticmethod
    def _paper_p_maker_from_detail(detail: Optional[Dict[str, Any]], *, leg: str) -> float:
        if detail is None:
            return 0.5
        p_maker = detail.get(f"p_maker_{leg}", 0.5)
        return float(p_maker)

    def _paper_fill_price(
        self,
        *,
        sym: str,
        side: str,
        size: float,
        limit_price: Optional[float],
        ts_ms: int,
        ctx: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> float:
        # Simplified fill logic - in real paper, could simulate slippage
        if limit_price is not None:
            return limit_price
        # Use mid price or last close
        closes = ctx.get("closes", [])
        if closes:
            return closes[-1]
        return 100.0  # fallback

    def _paper_open_position(
        self,
        *,
        sym: str,
        side: str,
        entry_price: float,
        ts_ms: int,
        cap_frac: float,
        leverage: float,
        fee_roundtrip: float,
        reason: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        side_u = str(side).upper()
        if side_u not in ("LONG", "SHORT"):
            return
        if entry_price <= 0:
            return

        cap = float(max(0.0, min(1.0, cap_frac)))
        lev = float(max(0.0, min(float(leverage), float(self._get_max_leverage()))))
        if cap <= 0.0 or lev <= 0.0:
            return

        margin = float(self._get_balance()) * cap
        notional = margin * lev
        qty = notional / float(entry_price)
        if qty <= 0:
            return

        position = {
            "symbol": sym,
            "side": side_u,
            "entry_price": float(entry_price),
            "entry": float(entry_price),
            "time": int(ts_ms),
            "leverage": float(lev),
            "cap_frac": float(cap),
            "margin": float(margin),
            "notional": float(notional),
            "size": float(qty),
            "quantity": float(qty),
            "fee_roundtrip": float(max(0.0, fee_roundtrip)),
            "current": float(entry_price),
            "price": float(entry_price),
            "unrealized_pnl": 0.0,
            "pnl": 0.0,
            "roe": 0.0,
            "age_sec": 0.0,
        }

        # Dynamic Horizon Hold
        if detail is not None:
            meta = detail.get("meta") if isinstance(detail.get("meta"), dict) else {}
            entry_h = 0
            if side_u == "LONG":
                entry_h = int(meta.get("policy_best_h_long") or 0)
            else:
                entry_h = int(meta.get("policy_best_h_short") or 0)

            if entry_h > 0:
                hold_frac = float(max(0.0, min(1.0, 0.25)))  # self.entry_min_hold_frac
                dyn_hold = int(max(0, min(entry_h, int(entry_h * hold_frac))))
                position["entry_horizon_sec"] = int(entry_h)
                position["policy_horizon_sec"] = int(entry_h)
                position["dynamic_min_hold"] = int(dyn_hold)

            # Score-Based Exit (UnifiedScore preferred)
            score_long = float(meta.get("unified_score_long") or detail.get("unified_score_long") or meta.get("policy_ev_score_long") or detail.get("policy_ev_score_long") or 0.0)
            score_short = float(meta.get("unified_score_short") or detail.get("unified_score_short") or meta.get("policy_ev_score_short") or detail.get("policy_ev_score_short") or 0.0)
            entry_score_side = score_long if side_u == "LONG" else score_short
            position["entry_score_long"] = score_long
            position["entry_score_short"] = score_short
            position["entry_score_side"] = entry_score_side
            position["max_score_long"] = score_long
            position["max_score_short"] = score_short
            position["entry_direction_reason"] = str(meta.get("policy_direction_reason") or "")
            position["entry_policy_direction"] = int(meta.get("policy_direction") or 0)

        self._update_positions(sym, position)
        fee = notional * fee_roundtrip / 2  # entry fee
        self._update_balance(-fee)

        trade = {
            "time": int(ts_ms),
            "sym": sym,
            "type": "ENTER",
            "action_type": "ENTER",
            "side": side_u,
            "price": float(entry_price),
            "leverage": float(lev),
            "cap_frac": float(cap),
            "notional": float(notional),
            "reason": str(reason or ""),
        }
        self._paper_append_trade(trade)

    def _paper_close_position(
        self,
        *,
        sym: str,
        exit_price: float,
        ts_ms: int,
        reason: str,
        position: Dict[str, Any],
    ) -> None:
        entry_price = position["entry_price"]
        size = position["size"]
        fee_roundtrip = position["fee_roundtrip"]

        pnl = (exit_price - entry_price) * size if position["side"] == "LONG" else (entry_price - exit_price) * size
        fee = size * exit_price * fee_roundtrip / 2  # exit fee
        net_pnl = pnl - fee

        self._update_balance(net_pnl)
        self._update_positions(sym, None)  # remove position

        trade = {
            "symbol": sym,
            "side": "SELL" if position["side"] == "LONG" else "BUY",
            "price": exit_price,
            "size": size,
            "fee": fee,
            "pnl": pnl,
            "net_pnl": net_pnl,
            "time": ts_ms,
            "reason": reason,
            "mode": "paper",
        }
        self._paper_append_trade(trade)

    def _paper_init_exit_policy_state(self, sym: str, detail: Optional[Dict[str, Any]], ts_ms: int) -> None:
        # Initialize exit policy state for symbol
        self._exit_policy_state[sym] = {
            "init_time": ts_ms,
            "detail": detail or {},
        }

    def _paper_exit_policy_signal(
        self,
        *,
        sym: str,
        ts_ms: int,
        ctx: Dict[str, Any],
        position: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> Optional[str]:
        # Simplified exit policy - could be expanded
        # For now, just check if position is old
        init_time = self._exit_policy_state.get(sym, {}).get("init_time", ts_ms)
        if ts_ms - init_time > 3600 * 1000:  # 1 hour
            return "time_exit"
        return None

    def _paper_trade_step(
        self,
        *,
        sym: str,
        ts_ms: int,
        ctx: Dict[str, Any],
        meta: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> None:
        # Handle entry/exit based on decision
        action = decision.get("action")
        if action == "enter":
            # Open position
            side = decision.get("side", "LONG")
            size = decision.get("size", 1.0)
            entry_price = self._paper_fill_price(
                sym=sym, side=side, size=size, limit_price=None, ts_ms=ts_ms, ctx=ctx, meta=meta
            )
            fee_roundtrip = self._paper_fee_roundtrip_from_engine_meta(meta) or 0.002
            self._paper_open_position(
                sym=sym, side=side, size=size, entry_price=entry_price, ts_ms=ts_ms, ctx=ctx, meta=meta, fee_roundtrip=fee_roundtrip
            )
            self._paper_init_exit_policy_state(sym, decision.get("detail"), ts_ms)
        elif action == "exit":
            # Close position if exists
            # Assume position is passed or retrieved
            pass  # Placeholder

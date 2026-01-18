from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover
    from core.orchestrator import LiveOrchestrator


@dataclass
class AccountSummary:
    total_equity: float
    wallet_balance: float
    free_balance: Optional[float]
    total_initial_margin: Optional[float]
    total_maintenance_margin: Optional[float]
    margin_ratio: Optional[float]  # initial_margin / equity


class RiskManager:
    """
    Cross-margin risk guardrails for live trading.

    Key goals:
      - Sync per-symbol API leverage (so Kelly sizing doesn't conflict with exchange settings).
      - Use Total Equity (wallet + uPnL) as the sizing base in cross margin.
      - Monitor margin ratio; if > threshold, block new entries and de-lever existing positions.
      - Handle insufficient margin errors by triggering group recalculation and cooldown.
    """

    def __init__(self, orch: "LiveOrchestrator"):
        self.orch = orch

        # System max total leverage (notional / equity). Requirement: 700% = 7x.
        try:
            self.max_total_leverage = float(getattr(orch, "max_total_leverage", 0.0) or 0.0)
        except Exception:
            self.max_total_leverage = 0.0
        if self.max_total_leverage <= 0:
            self.max_total_leverage = 10.0

        # Margin-ratio guard (initial margin / equity).
        self.margin_ratio_guard = 0.80
        self.margin_ratio_target = 0.80
        self.total_leverage_guard_eps = 0.01  # 1% hysteresis to avoid thrash

        # Cooldown after insufficient margin / rejected new order.
        self.insufficient_margin_cooldown_ms = 30_000
        self._blocked_new_entries_until_ms = 0

        self._acct: Optional[AccountSummary] = None
        self._last_margin_guard_ms = 0
        self._last_total_lev_guard_ms = 0
        self._margin_guard_min_interval_ms = 3_000

        # Leverage sync throttling (avoid spamming set_leverage).
        self._last_leverage_sync_ms_by_sym: Dict[str, int] = {}
        self.leverage_sync_min_interval_ms = 15_000

        # Minimum order value/qty filters (skip dust allocations).
        self.min_notional_usd = 6.0
        self._emergency_stop_triggered = False
        self.initial_equity: Optional[float] = None
        try:
            self.max_drawdown_limit = float(getattr(orch, "max_drawdown_limit", 0.10) or 0.10)
        except Exception:
            self.max_drawdown_limit = 0.10
        self._dd_stop_triggered = False

    # -----------------------------
    # account summary / equity basis
    # -----------------------------
    def update_account_summary(
        self,
        *,
        wallet_balance: Optional[float],
        total_equity: Optional[float],
        free_balance: Optional[float],
        total_initial_margin: Optional[float],
        total_maintenance_margin: Optional[float],
    ) -> None:
        try:
            w = float(wallet_balance) if wallet_balance is not None else 0.0
        except Exception:
            w = 0.0
        try:
            e = float(total_equity) if total_equity is not None else 0.0
        except Exception:
            e = 0.0
        if e <= 0:
            e = w

        im = None
        mm = None
        try:
            if total_initial_margin is not None:
                im = float(total_initial_margin)
        except Exception:
            im = None
        try:
            if total_maintenance_margin is not None:
                mm = float(total_maintenance_margin)
        except Exception:
            mm = None

        mr = None
        try:
            if im is not None and e > 0:
                mr = float(im) / float(e)
        except Exception:
            mr = None

        self._acct = AccountSummary(
            total_equity=float(e),
            wallet_balance=float(w),
            free_balance=free_balance,
            total_initial_margin=im,
            total_maintenance_margin=mm,
            margin_ratio=mr,
        )

    def get_total_equity(self, fallback_wallet: float) -> float:
        if self._acct is not None and self._acct.total_equity > 0:
            return float(self._acct.total_equity)
        try:
            eq = float(getattr(self.orch, "_live_equity", 0.0) or 0.0)
            if eq > 0:
                return eq
        except Exception:
            pass
        return float(max(1.0, float(fallback_wallet)))

    def get_margin_ratio(self) -> Optional[float]:
        if self._acct is not None:
            return self._acct.margin_ratio
        return None

    def check_emergency_stop(self, current_equity: float) -> bool:
        if self._emergency_stop_triggered:
            return True
        try:
            equity = float(current_equity)
        except Exception:
            return False
        if self.initial_equity is None:
            try:
                baseline = float(getattr(self.orch, "initial_equity", equity) or equity)
            except Exception:
                baseline = equity
            self.initial_equity = baseline
        baseline = float(self.initial_equity or 0.0)
        if baseline <= 0:
            return False
        dd = (equity - baseline) / baseline
        if dd > -float(self.max_drawdown_limit):
            return False
        self._emergency_stop_triggered = True
        return True

    def check_drawdown_stop(self, current_equity: float, initial_equity: float) -> bool:
        if self._dd_stop_triggered:
            return True
        try:
            baseline = float(initial_equity)
            equity = float(current_equity)
        except Exception:
            return False
        if baseline <= 0:
            return False
        max_dd = float(getattr(self.orch, "max_drawdown_pct", 0.05) or 0.05)
        dd = (equity - baseline) / baseline
        if dd > -max_dd:
            return False
        self._dd_stop_triggered = True
        try:
            setattr(self.orch, "safety_mode", True)
        except Exception:
            pass
        self.orch._log_err(f"[RISK] DD stop triggered: equity={equity:.2f} baseline={baseline:.2f} dd={dd:.2%}")
        for sym, pos in list(getattr(self.orch, "positions", {}).items()):
            px = None
            try:
                px = getattr(self.orch, "market", {}).get(sym, {}).get("price")
            except Exception:
                px = None
            if px is None:
                continue
            try:
                self.orch._close_position(sym, float(px), "drawdown_stop", exit_kind="KILL")
            except Exception as e:
                self.orch._log_err(f"[RISK] panic close failed: {sym} err={e}")
        return True

    # -----------------------------
    # leverage sync
    # -----------------------------
    async def sync_api_leverage(self, sym: str, *, target_leverage: float, ts_ms: int) -> float:
        """
        Ensure the exchange-side leverage setting is >= target_leverage (clamped to symbol max).
        Returns the leverage we attempted to apply.
        """
        try:
            now = int(ts_ms)
        except Exception:
            now = int(time.time() * 1000)
        last = int(self._last_leverage_sync_ms_by_sym.get(str(sym), 0) or 0)
        if (now - last) < int(self.leverage_sync_min_interval_ms):
            return float(target_leverage)

        # Clamp target to symbol max leverage (from markets metadata).
        max_lev = None
        try:
            if hasattr(self.orch, "_max_exchange_leverage"):
                max_lev = self.orch._max_exchange_leverage(sym)  # type: ignore[attr-defined]
        except Exception:
            max_lev = None
        lev = float(target_leverage)
        if max_lev is not None and float(max_lev) > 0:
            lev = float(max(1.0, min(float(lev), float(max_lev))))

        self._last_leverage_sync_ms_by_sym[str(sym)] = now
        try:
            await self.orch._ensure_live_leverage(sym, float(lev))  # type: ignore[attr-defined]
            self.orch._log(f"[RISK] API leverage sync: {sym} -> {float(lev):.2f}x")
        except Exception as e:
            self.orch._log_err(f"[RISK] API leverage sync failed: {sym} target={float(lev):.2f}x err={e}")
        return float(lev)

    # -----------------------------
    # margin guard / deleveraging
    # -----------------------------
    def _cooldown_new_entries(self, ts_ms: int, *, reason: str) -> None:
        until = int(ts_ms) + int(self.insufficient_margin_cooldown_ms)
        if until > int(self._blocked_new_entries_until_ms):
            self._blocked_new_entries_until_ms = int(until)
        self.orch._log_err(f"[RISK] New entries blocked for {self.insufficient_margin_cooldown_ms}ms: {reason}")

    def allow_new_entry_now(self, ts_ms: int) -> bool:
        try:
            return int(ts_ms) >= int(self._blocked_new_entries_until_ms)
        except Exception:
            return True

    def _estimate_min_notional_usd(self, sym: str) -> float:
        """
        Prefer exchange market metadata minNotionalValue when available; fallback to configured default.
        """
        try:
            ex = getattr(self.orch, "exchange", None)
            if ex is None:
                return float(self.min_notional_usd)
            mkt = None
            markets = getattr(ex, "markets", None) or {}
            if isinstance(markets, dict) and str(sym) in markets:
                mkt = markets.get(str(sym))
            if mkt is None and hasattr(ex, "market"):
                mkt = ex.market(str(sym))
            info = mkt.get("info") if isinstance(mkt, dict) else None
            if isinstance(info, dict):
                lot = info.get("lotSizeFilter")
                if isinstance(lot, dict) and lot.get("minNotionalValue") is not None:
                    return float(lot.get("minNotionalValue"))
        except Exception:
            pass
        return float(self.min_notional_usd)

    def maybe_trigger_margin_guard(self, ts_ms: int) -> bool:
        """
        If margin ratio is above guard threshold, block new entries and submit reduceOnly deleveraging.
        Returns True if guard is active.
        """
        mr = self.get_margin_ratio()
        if mr is None:
            return False
        if float(mr) <= float(self.margin_ratio_guard):
            return False

        try:
            now = int(ts_ms)
        except Exception:
            now = int(time.time() * 1000)
        if (now - int(self._last_margin_guard_ms)) < int(self._margin_guard_min_interval_ms):
            return True
        self._last_margin_guard_ms = now

        try:
            reduction = float(self.margin_ratio_target) / max(1e-9, float(mr))
            reduction = float(max(0.0, min(1.0, reduction)))
        except Exception:
            reduction = 0.0

        self.orch._log_err(f"[RISK] Margin ratio high: {float(mr):.3f} > {self.margin_ratio_guard:.2f} (scale={reduction:.3f})")

        # Emergency deleveraging: shrink each open position proportionally.
        positions = getattr(self.orch, "positions", {}) or {}
        if not isinstance(positions, dict) or not positions:
            return True

        for sym, pos in list(positions.items()):
            if not isinstance(pos, dict):
                continue
            try:
                size = float(pos.get("size", pos.get("quantity", pos.get("qty", 0.0))) or 0.0)
            except Exception:
                size = 0.0
            if abs(size) <= 0.0:
                continue

            # Reduce by (1 - reduction) of the current size.
            reduce_frac = 1.0 - float(reduction)
            if reduce_frac <= 0.0:
                continue
            qty = abs(size) * float(reduce_frac)
            if qty <= 0.0:
                continue

            side = str(pos.get("side") or "").upper()
            close_side = "buy" if side == "SHORT" else ("sell" if side == "LONG" else None)
            if close_side is None:
                continue

            params: Dict[str, Any] = {"reduceOnly": True}
            try:
                pidx = pos.get("position_idx")
                if pidx is not None:
                    params["positionIdx"] = int(pidx)
            except Exception:
                pass

            try:
                self.orch._schedule_live_order(str(sym), close_side, float(qty), price=None, leverage=None, params=params)
            except Exception as e:
                self.orch._log_err(f"[RISK] Deleverage order failed: {sym} qty={qty:.6f} err={e}")

        return True

    def maybe_trigger_total_leverage_guard(self, ts_ms: int) -> bool:
        """
        Enforce max total leverage (total_notional / equity <= max_total_leverage).
        If exceeded, block increases and submit proportional reduceOnly deleveraging.
        Returns True if guard is active.
        """
        eq = self.get_total_equity(getattr(self.orch, "balance", 0.0) or 0.0)
        if eq <= 0:
            return False

        # Include pending notional reservations when available.
        try:
            if hasattr(self.orch, "_live_notional_used"):
                open_notional = float(self.orch._live_notional_used(ts_ms=int(ts_ms)))  # type: ignore[attr-defined]
            else:
                open_notional = float(getattr(self.orch, "_total_open_notional")())
        except Exception:
            open_notional = 0.0

        ratio = float(open_notional) / float(eq) if eq > 0 else 0.0
        if ratio <= float(self.max_total_leverage) * (1.0 + float(self.total_leverage_guard_eps)):
            return False

        try:
            now = int(ts_ms)
        except Exception:
            now = int(time.time() * 1000)
        if (now - int(self._last_total_lev_guard_ms)) < int(self._margin_guard_min_interval_ms):
            return True
        self._last_total_lev_guard_ms = now

        target_total = float(eq) * float(self.max_total_leverage)
        need_reduce_notional = float(open_notional) - float(target_total)
        if need_reduce_notional <= 0:
            return False

        self.orch._log_err(
            f"[RISK] Total leverage high: util={ratio:.3f}x > {float(self.max_total_leverage):.2f}x (need_reduce_notional={need_reduce_notional:.2f})"
        )

        positions = getattr(self.orch, "positions", {}) or {}
        if not isinstance(positions, dict) or not positions:
            return True

        # Reduce largest notionals first to avoid "dust" reduce orders that fall below min order quantity.
        ranked = []
        for sym, pos in list(positions.items()):
            if not isinstance(pos, dict):
                continue
            try:
                size = float(pos.get("size", pos.get("quantity", pos.get("qty", 0.0))) or 0.0)
            except Exception:
                size = 0.0
            if abs(size) <= 0.0:
                continue
            try:
                notional = float(pos.get("notional") or 0.0)
            except Exception:
                notional = 0.0
            ranked.append((float(notional), str(sym), pos))
        ranked.sort(reverse=True, key=lambda x: x[0])

        ex = getattr(self.orch, "exchange", None)
        remaining = float(need_reduce_notional)
        for notional, sym, pos in ranked:
            if remaining <= 0:
                break
            try:
                px = float(pos.get("price") or pos.get("current") or 0.0)
            except Exception:
                px = 0.0
            if px <= 0:
                continue
            try:
                size = float(pos.get("size", pos.get("quantity", pos.get("qty", 0.0))) or 0.0)
            except Exception:
                size = 0.0
            if abs(size) <= 0:
                continue

            reduce_notional = min(float(remaining), float(notional))
            qty = float(reduce_notional) / float(px)
            qty = float(min(abs(size), max(0.0, qty)))
            if qty <= 0:
                continue

            # Best-effort: apply amount precision early to increase chance of passing min checks.
            try:
                if ex is not None and hasattr(ex, "amount_to_precision"):
                    qty = float(ex.amount_to_precision(sym, qty))
            except Exception:
                pass

            side = str(pos.get("side") or "").upper()
            close_side = "buy" if side == "SHORT" else ("sell" if side == "LONG" else None)
            if close_side is None:
                continue

            params: Dict[str, Any] = {"reduceOnly": True}
            try:
                pidx = pos.get("position_idx")
                if pidx is not None:
                    params["positionIdx"] = int(pidx)
            except Exception:
                pass

            try:
                self.orch._schedule_live_order(sym, close_side, float(qty), price=None, leverage=None, params=params)
                remaining -= float(qty) * float(px)
            except Exception as e:
                self.orch._log_err(f"[RISK] Total leverage deleverage failed: {sym} qty={qty:.6f} err={e}")

        # While above the cap, block new entries briefly to prevent re-overshoot.
        self._cooldown_new_entries(ts_ms, reason="total_leverage_guard")
        return True

    # -----------------------------
    # sizing helpers (equity-based)
    # -----------------------------
    def adjust_entry(
        self,
        *,
        sym: str,
        cap_frac: float,
        leverage: float,
        entry_price: float,
        ts_ms: int,
        margin_used_est: float,
    ) -> Tuple[float, float]:
        """
        Adjust entry cap/leverage for:
          - global max total leverage (notional/equity)
          - margin ratio guard
          - new-entry cooldown
          - minimum notional filter
        Returns (cap_frac, leverage); may return (0, lev) to indicate "skip".
        """
        if entry_price <= 0:
            return 0.0, float(leverage)

        # Guards: always run both (both may schedule reduceOnly deleveraging),
        # and block new entries if either is active.
        guard_active = False
        try:
            guard_active = bool(self.maybe_trigger_total_leverage_guard(ts_ms)) or guard_active
        except Exception:
            pass
        try:
            guard_active = bool(self.maybe_trigger_margin_guard(ts_ms)) or guard_active
        except Exception:
            pass
        if guard_active:
            return 0.0, float(leverage)

        if not self.allow_new_entry_now(ts_ms):
            return 0.0, float(leverage)

        eq = self.get_total_equity(getattr(self.orch, "balance", 0.0) or 0.0)
        cap = float(max(0.0, min(1.0, float(cap_frac))))
        lev = float(max(1.0, float(leverage)))

        # Notional filter: drop tiny orders for fee efficiency.
        notional = float(eq) * float(cap) * float(lev)
        min_notional = self._estimate_min_notional_usd(sym)
        if notional < float(min_notional):
            self.orch._log(f"[RISK] Skip dust allocation: {sym} notional={notional:.2f} < min_notional={min_notional:.2f}")
            return 0.0, float(lev)

        # Max total leverage: enforce notional/equity <= max_total_leverage.
        try:
            if hasattr(self.orch, "_live_notional_used"):
                open_notional = float(self.orch._live_notional_used(ts_ms=int(ts_ms)))  # type: ignore[attr-defined]
            else:
                open_notional = float(getattr(self.orch, "_total_open_notional")())
        except Exception:
            open_notional = 0.0
        allowed_total = float(eq) * float(self.max_total_leverage)
        remaining = float(allowed_total) - float(open_notional)
        if remaining <= 0.0:
            return 0.0, float(lev)
        if notional > remaining:
            old_cap = cap
            cap = float(remaining) / (float(eq) * float(lev))
            cap = float(max(0.0, min(1.0, cap)))
            self.orch._log(f"[RISK] Scale entry cap by leverage cap: {sym} {old_cap:.3f}->{cap:.3f} (remaining_notional={remaining:.2f})")
            notional = float(eq) * float(cap) * float(lev)
            if cap <= 0.0:
                return 0.0, float(lev)
            if notional < float(min_notional):
                return 0.0, float(lev)

        # Margin budget: in cross margin, cap_frac is the intended margin fraction.
        req_margin = float(eq) * float(cap)
        max_margin = float(eq) * float(getattr(self.orch, "live_max_margin_frac", 0.95) or 0.95)
        remaining_margin = float(max_margin) - float(max(0.0, margin_used_est))
        if remaining_margin <= 0.0:
            return 0.0, float(lev)
        if req_margin > remaining_margin:
            old_cap = cap
            cap = float(remaining_margin) / float(eq)
            cap = float(max(0.0, min(1.0, cap)))
            self.orch._log(f"[RISK] Scale entry cap by margin budget: {sym} {old_cap:.3f}->{cap:.3f} (remaining_margin={remaining_margin:.2f})")
            if cap <= 0.0:
                return 0.0, float(lev)

        return float(cap), float(lev)

    # -----------------------------
    # error handling
    # -----------------------------
    def on_insufficient_margin_error(self, ts_ms: int, *, sym: str, err_msg: str) -> None:
        """
        Triggered when ccxt/order returns an insufficient margin-like error.
        """
        self._cooldown_new_entries(ts_ms, reason=f"{sym} {err_msg}")
        # Ask orchestrator to cancel stale open non-reduceOnly orders to free IM.
        try:
            until = int(ts_ms) + int(self.insufficient_margin_cooldown_ms)
            if until > int(getattr(self.orch, "_live_force_open_order_cleanup_until_ms", 0) or 0):
                setattr(self.orch, "_live_force_open_order_cleanup_until_ms", int(until))
                setattr(self.orch, "_live_force_open_order_cleanup_reason", f"insufficient_margin {sym}")
        except Exception:
            pass
        try:
            if hasattr(self.orch, "_recalculate_groups"):
                self.orch._recalculate_groups(force=True)
        except Exception:
            pass

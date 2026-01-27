from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, List

import ccxt.async_support as ccxt
import numpy as np

import config
from core.data_manager import DataManager
from core.database_manager import DatabaseManager, TradingMode
from engines.engine_hub import EngineHub
from engines.pmaker_manager import PMakerManager
from engines.probability_methods import _approx_p_pos_and_ev_hold
from engines.mc.constants import SECONDS_PER_YEAR
from utils.helpers import _env_bool, _env_float, _env_int, _load_env_file, _safe_float, now_ms


class LiveOrchestrator:
    def __init__(
        self,
        exchange: Any,
        symbols: Optional[list[str]] = None,
        data_exchange: Any = None,
        trading_mode: str = "paper",
    ):
        self.exchange = exchange
        self.data_exchange = data_exchange or exchange
        self.symbols = list(symbols) if symbols is not None else list(config.SYMBOLS)
        
        # ─────────────────────────────────────────────────────────────────────
        # Trading Mode: paper vs live
        # ─────────────────────────────────────────────────────────────────────
        self.trading_mode = trading_mode.lower() if trading_mode else "paper"
        self.is_live_mode = self.trading_mode == "live"
        
        # Database Manager 초기화 (SQLite 영속 저장소)
        db_filename = "bot_data_live.db" if self.is_live_mode else "bot_data_paper.db"
        try:
            self.db: Optional[DatabaseManager] = DatabaseManager(db_path=f"state/{db_filename}")
            print(f"[DB] DatabaseManager initialized: state/{db_filename}")
        except Exception as e:
            print(f"[DB] Failed to initialize DatabaseManager: {e}")
            self.db = None

        self.hub = EngineHub()
        self._mc_ready = True

        self._net_sem = asyncio.Semaphore(int(config.MAX_INFLIGHT_REQ))
        self.clients: set[Any] = set()
        self.logs = deque(maxlen=300)

        self.balance = 10_000.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.exposure_cap_enabled = bool(getattr(config, "EXPOSURE_CAP_ENABLED", True))

        self.leverage = float(config.DEFAULT_LEVERAGE)
        self.max_leverage = float(config.MAX_LEVERAGE)
        self.enable_orders = bool(config.ENABLE_LIVE_ORDERS)
        self.decision_refresh_sec = float(_env_float("DECISION_REFRESH_SEC", float(getattr(config, "DECISION_REFRESH_SEC", 2.0))))
        # Heavy decision compute is decoupled from dashboard refresh; this sets a minimum per-symbol re-eval interval.
        self.decision_eval_min_interval_sec = float(
            _env_float("DECISION_EVAL_MIN_INTERVAL_SEC", float(self.decision_refresh_sec))
        )
        self.decision_worker_sleep_sec = float(_env_float("DECISION_WORKER_SLEEP_SEC", 0.0))

        # paper trading (no exchange orders)
        self.paper_trading_enabled = _env_bool("PAPER_TRADING", True) and (not self.enable_orders)
        self.paper_flat_on_wait = _env_bool("PAPER_FLAT_ON_WAIT", True)
        # Paper sizing: by default use orchestrator defaults (fixed size/leverage).
        # If enabled, use engine-provided optimal sizing/leverage (often Kelly-like and can be extremely small).
        self.paper_use_engine_sizing = _env_bool("PAPER_USE_ENGINE_SIZING", True)
        # If engine sizing is enabled, apply a multiplier and a floor/cap so paper positions don't become too tiny.
        self.paper_engine_size_mult = float(_env_float("PAPER_ENGINE_SIZE_MULT", 1.0))
        self.paper_engine_size_min_frac = float(_env_float("PAPER_ENGINE_SIZE_MIN_FRAC", 0.005))
        self.paper_engine_size_max_frac = float(_env_float("PAPER_ENGINE_SIZE_MAX_FRAC", 0.20))
        self.paper_size_frac_default = float(_env_float("PAPER_SIZE_FRAC", float(getattr(config, "DEFAULT_SIZE_FRAC", 0.10))))
        self.paper_leverage_default = float(_env_float("PAPER_LEVERAGE", float(getattr(config, "DEFAULT_LEVERAGE", 5.0))))
        self.paper_fee_roundtrip = float(_env_float("PAPER_FEE_ROUNDTRIP", 0.0))
        self.paper_slippage_bps = float(_env_float("PAPER_SLIPPAGE_BPS", 0.0))
        self.paper_min_hold_sec = int(_env_int("PAPER_MIN_HOLD_SEC", int(getattr(config, "POSITION_HOLD_MIN_SEC", 0))))
        self.paper_max_hold_sec = int(_env_int("PAPER_MAX_HOLD_SEC", int(getattr(config, "MAX_POSITION_HOLD_SEC", 600))))
        self.paper_max_positions = int(_env_int("PAPER_MAX_POSITIONS", int(getattr(config, "MAX_CONCURRENT_POSITIONS", 99999))))
        # If enabled, exits/flips are driven by MC exit-policy heuristics (hold_bad/score_flip/dd_stop/time_stop),
        # and legacy "WAIT->FLAT" / "max_hold" paper rules are ignored for open positions.
        self.paper_exit_policy_only = _env_bool("PAPER_EXIT_POLICY_ONLY", True)
        self.paper_exit_policy_horizon_sec_default = int(_env_int("PAPER_EXIT_POLICY_HORIZON_SEC", 1800))
        self.paper_exit_policy_min_hold_sec = int(_env_int("PAPER_EXIT_POLICY_MIN_HOLD_SEC", 180))
        self.paper_exit_policy_decision_dt_sec = int(_env_int("PAPER_EXIT_POLICY_DECISION_DT_SEC", 5))
        self.paper_exit_policy_flip_confirm_ticks = int(_env_int("PAPER_EXIT_POLICY_FLIP_CONFIRM_TICKS", 3))
        self.paper_exit_policy_hold_bad_ticks = int(_env_int("PAPER_EXIT_POLICY_HOLD_BAD_TICKS", 3))
        self.paper_exit_policy_score_margin = float(_env_float("PAPER_EXIT_POLICY_SCORE_MARGIN", 0.0001))
        self.paper_exit_policy_soft_floor = float(_env_float("PAPER_EXIT_POLICY_SOFT_FLOOR", -0.001))
        self.paper_exit_policy_p_pos_enter_floor = float(_env_float("PAPER_EXIT_POLICY_P_POS_ENTER_FLOOR", 0.52))
        self.paper_exit_policy_p_pos_hold_floor = float(_env_float("PAPER_EXIT_POLICY_P_POS_HOLD_FLOOR", 0.50))
        self.paper_exit_policy_dd_stop_enabled = _env_bool("PAPER_EXIT_POLICY_DD_STOP_ENABLED", True)
        self.paper_exit_policy_dd_stop_roe = float(_env_float("PAPER_EXIT_POLICY_DD_STOP_ROE", -0.02))

        # MC runtime tuning (ctx/instance overrides)
        self.mc_n_paths_live = int(_env_int("MC_N_PATHS_LIVE", int(getattr(config, "MC_N_PATHS_LIVE", 10000))))
        self.mc_n_paths_exit = int(_env_int("MC_N_PATHS_EXIT", int(getattr(config, "MC_N_PATHS_EXIT", 512))))
        self.mc_tail_mode = str(os.environ.get("MC_TAIL_MODE", "student_t")).strip().lower() or "student_t"
        self.mc_student_t_df = float(_env_float("MC_STUDENT_T_DF", 6.0))
        self._last_trade_event_by_sym: Dict[str, str] = {}

        self.exec_stats: Dict[str, Dict[str, Any]] = {}
        self.trade_tape = deque(maxlen=20_000)
        self._equity_history = deque(maxlen=20_000)

        self._loop_ms: Optional[int] = None
        self._last_rows: Optional[list[Dict[str, Any]]] = None
        self._decision_cache: Dict[str, Dict[str, Any]] = {}
        self._decision_rr_index: int = 0
        self._decide_cycle_ms: Optional[int] = None

        # Portfolio-level rebalancing config
        self.rebalance_top_n = int(_env_int("PORTFOLIO_TOP_N", 4))
        self.rebalance_cost_mult = float(_env_float("PORTFOLIO_SWITCH_COST_MULT", 1.2))
        self.rebalance_kelly_cap = float(_env_float("PORTFOLIO_KELLY_CAP", 5.0))
        self.portfolio_joint_interval_sec = float(_env_float("PORTFOLIO_JOINT_INTERVAL_SEC", 15.0))
        self._rebalance_last_decision: Dict[str, str] = {}
        self._last_portfolio_joint_ts: float = 0.0
        self._last_portfolio_report: Optional[Dict[str, Any]] = None

        # Back-compat state used by dashboard/aux modules
        self._group_info: Dict[str, Any] = {}
        self._latest_rankings: list[Any] = []
        self._symbol_t_star: Dict[str, Any] = {}
        self._symbol_scores: Dict[str, Any] = {}

        self.dashboard = None  # set in main
        self.data = DataManager(self, self.symbols, data_exchange=self.data_exchange)

        # Back-compat aliases (some modules still reference orch.market/orderbook directly)
        self.market = self.data.market
        self.ohlcv_buffer = self.data.ohlcv_buffer
        self.orderbook = self.data.orderbook
        self._last_kline_ok_ms = self.data._last_kline_ok_ms

        self.pmaker = PMakerManager(self)
        self._apply_mc_runtime_to_engines()

        # PMaker paper-mode self-training (simulated maker fills).
        # 목적: paper에서도 심볼별 fill_rate 통계를 누적해서 mu_alpha 보정/대시보드 표시가 가능하도록 함.
        self.pmaker_paper_enabled = _env_bool("PMAKER_PAPER_ENABLE", True)
        self.pmaker_paper_probe_interval_sec = float(_env_float("PMAKER_PAPER_PROBE_INTERVAL_SEC", 2.0))
        self.pmaker_paper_probe_timeout_ms = int(_env_int("PMAKER_PAPER_PROBE_TIMEOUT_MS", int(getattr(config, "MAKER_TIMEOUT_MS", 1500))))
        self.pmaker_paper_train_every_n = int(_env_int("PMAKER_PAPER_TRAIN_EVERY_N", 25))
        self.pmaker_paper_save_every_sec = float(_env_float("PMAKER_PAPER_SAVE_EVERY_SEC", 30.0))
        self.pmaker_paper_adverse_delay_sec = float(_env_float("PMAKER_ADVERSE_DELAY_SEC", 5.0))
        self.pmaker_paper_adverse_ema_alpha = float(_env_float("PMAKER_ADVERSE_EMA_ALPHA", 0.2))
        self._pmaker_paper_active: Dict[str, Dict[str, Any]] = {}
        self._pmaker_paper_next_side: Dict[str, str] = {}
        self._pmaker_paper_last_start_ms: Dict[str, int] = {}
        self._pmaker_paper_updates = 0
        self._pmaker_paper_last_save_ms = 0
        self._pmaker_paper_adverse_pending: Dict[str, Dict[str, Any]] = {}
        self._pmaker_paper_adverse_ema: Dict[str, float] = {}
        self._pmaker_paper_adverse_n: Dict[str, int] = {}

        # persistence
        self.state_dir = config.BASE_DIR / "state"
        self.state_dir.mkdir(exist_ok=True)
        self.state_files = {
            "equity": self.state_dir / "equity_history.json",
            "trade": self.state_dir / "trade_tape.json",
            "positions": self.state_dir / "positions.json",
            "balance": self.state_dir / "balance.json",
        }
        self._last_state_persist_ms = 0
        self._load_persisted_state()

    def _apply_mc_runtime_to_engines(self) -> None:
        for eng in getattr(self.hub, "engines", []) or []:
            if hasattr(eng, "N_PATHS_EXIT_POLICY"):
                try:
                    setattr(eng, "N_PATHS_EXIT_POLICY", int(self.mc_n_paths_exit))
                except Exception:
                    pass

    def _total_open_notional(self) -> float:
        total = 0.0
        for sym, pos in (self.positions or {}).items():
            try:
                size = float(pos.get("size", pos.get("quantity", pos.get("qty", 0.0))) or 0.0)
            except Exception:
                size = 0.0
            if size == 0.0:
                continue

            notional = pos.get("notional")
            if notional is None:
                price = pos.get("price") or pos.get("entry_price") or (self.market.get(sym) or {}).get("price")
                try:
                    px = float(price) if price is not None else 0.0
                except Exception:
                    px = 0.0
                notional = abs(size) * px
            try:
                total += abs(float(notional) or 0.0)
            except Exception:
                continue
        return float(total)

    @staticmethod
    def _pmaker_paper_sigma(closes: list[float], window: int = 60) -> float:
        if not closes:
            return 0.01
        n = len(closes)
        w = int(max(5, min(int(window), n)))
        xs = closes[-w:]
        try:
            arr = np.asarray(xs, dtype=np.float64)
            if arr.size < 6:
                return 0.01
            rets = np.diff(np.log(arr))
            if rets.size < 5:
                return 0.01
            sig = float(np.std(rets))
            if not math.isfinite(sig) or sig <= 0:
                return 0.01
            return float(max(1e-6, min(0.10, sig)))
        except Exception:
            return 0.01

    @staticmethod
    def _pmaker_paper_momentum_z(closes: list[float], sigma: float, window: int = 10) -> float:
        if not closes or len(closes) < int(window) + 2:
            return 0.0
        try:
            w = int(max(2, min(int(window), len(closes) - 1)))
            p0 = float(closes[-w - 1])
            p1 = float(closes[-1])
            if p0 <= 0 or p1 <= 0:
                return 0.0
            lr = float(math.log(p1 / p0))
            denom = float(max(1e-6, float(sigma))) * math.sqrt(float(w))
            z = float(lr / denom)
            if not math.isfinite(z):
                return 0.0
            return float(max(-8.0, min(8.0, z)))
        except Exception:
            return 0.0

    def _pmaker_paper_probe_tick(self, *, sym: str, ts_ms: int, ctx: Dict[str, Any]) -> None:
        """
        Paper-only PMaker training: simulate maker-limit fills at touch and update PMakerSurvivalMLP.
        """
        if not self.paper_trading_enabled:
            return
        if not self.pmaker_paper_enabled:
            return
        pm = getattr(self, "pmaker", None)
        surv = getattr(pm, "surv", None) if pm is not None else None
        if pm is None or (not getattr(pm, "enabled", False)) or surv is None:
            return

        price = ctx.get("price")
        best_bid = ctx.get("best_bid")
        best_ask = ctx.get("best_ask")
        spread_pct = ctx.get("spread_pct")
        closes = ctx.get("closes") or []
        ofi_score = float(ctx.get("ofi_score") or 0.0)
        liq_score = float(ctx.get("liquidity_score") or 1.0)

        try:
            px = float(price) if price is not None else None
        except Exception:
            px = None
        if px is None or px <= 0:
            return

        pending_adv = self._pmaker_paper_adverse_pending.get(sym)
        if pending_adv is not None:
            delay_sec = float(pending_adv.get("delay_sec", self.pmaker_paper_adverse_delay_sec) or 0.0)
            delay_ms = int(max(0.0, delay_sec) * 1000.0)
            fill_px = pending_adv.get("fill_px")
            side = str(pending_adv.get("side") or "")
            start_ms = int(pending_adv.get("fill_ms") or ts_ms)
            if (ts_ms - start_ms) >= delay_ms:
                try:
                    fill_px_f = float(fill_px) if fill_px is not None else None
                except Exception:
                    fill_px_f = None
                if fill_px_f is not None and fill_px_f > 0:
                    if side == "buy":
                        adverse_move = max(0.0, (fill_px_f - float(px)) / float(fill_px_f))
                    elif side == "sell":
                        adverse_move = max(0.0, (float(px) - fill_px_f) / float(fill_px_f))
                    else:
                        adverse_move = 0.0
                    prev = float(self._pmaker_paper_adverse_ema.get(sym, adverse_move) or 0.0)
                    alpha = float(self.pmaker_paper_adverse_ema_alpha)
                    alpha = float(max(0.0, min(1.0, alpha)))
                    ema = (1.0 - alpha) * prev + alpha * float(adverse_move)
                    self._pmaker_paper_adverse_ema[sym] = float(max(0.0, ema))
                    self._pmaker_paper_adverse_n[sym] = int(self._pmaker_paper_adverse_n.get(sym, 0) or 0) + 1
                self._pmaker_paper_adverse_pending.pop(sym, None)

        try:
            bid = float(best_bid) if best_bid is not None else None
        except Exception:
            bid = None
        try:
            ask = float(best_ask) if best_ask is not None else None
        except Exception:
            ask = None
        try:
            sp = float(spread_pct) if spread_pct is not None else 0.0
        except Exception:
            sp = 0.0

        active = self._pmaker_paper_active.get(sym)
        if active is not None:
            side = str(active.get("side") or "")
            limit_px = active.get("limit_px")
            start_ms = int(active.get("start_ms") or ts_ms)
            timeout_ms = int(active.get("timeout_ms") or self.pmaker_paper_probe_timeout_ms)

            try:
                limit_px_f = float(limit_px) if limit_px is not None else None
            except Exception:
                limit_px_f = None
            if limit_px_f is None or limit_px_f <= 0:
                self._pmaker_paper_active.pop(sym, None)
                return

            filled = False
            if side == "buy":
                filled = bool(px <= float(limit_px_f))
            elif side == "sell":
                filled = bool(px >= float(limit_px_f))

            age_ms = int(ts_ms) - int(start_ms)
            if filled or (age_ms >= int(timeout_ms)):
                try:
                    first_fill_delay_ms = int(max(0, age_ms)) if filled else None
                    x = active.get("x")
                    if x is not None:
                        surv.update_one_attempt(
                            sym=sym,
                            x=x,
                            timeout_ms=int(timeout_ms),
                            first_fill_delay_ms=first_fill_delay_ms,
                            qty_attempt=1.0,
                            qty_filled=1.0 if filled else 0.0,
                        )
                        self._pmaker_paper_updates += 1
                        if self.pmaker_paper_train_every_n > 0 and (self._pmaker_paper_updates % int(self.pmaker_paper_train_every_n) == 0):
                            try:
                                surv.train_from_replay(steps=int(getattr(pm, "train_steps", 1)), batch_size=int(getattr(pm, "batch", 32)))
                            except Exception:
                                pass
                        if self.pmaker_paper_save_every_sec > 0:
                            if (ts_ms - int(self._pmaker_paper_last_save_ms)) >= int(self.pmaker_paper_save_every_sec * 1000):
                                self._pmaker_paper_last_save_ms = int(ts_ms)
                                try:
                                    pm.save_model()
                                except Exception:
                                    pass
                    if filled:
                        self._pmaker_paper_adverse_pending[sym] = {
                            "side": side,
                            "fill_px": float(px),
                            "fill_ms": int(ts_ms),
                            "delay_sec": float(self.pmaker_paper_adverse_delay_sec),
                        }
                except Exception:
                    pass
                finally:
                    self._pmaker_paper_active.pop(sym, None)
            return

        # start a new probe (rate-limited)
        last_start = int(self._pmaker_paper_last_start_ms.get(sym) or 0)
        if (ts_ms - last_start) < int(max(0.1, float(self.pmaker_paper_probe_interval_sec)) * 1000):
            return
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return

        next_side = str(self._pmaker_paper_next_side.get(sym) or "buy")
        side = next_side if next_side in ("buy", "sell") else "buy"
        self._pmaker_paper_next_side[sym] = "sell" if side == "buy" else "buy"

        limit_px = float(bid) if side == "buy" else float(ask)
        mid = 0.5 * (float(bid) + float(ask))
        rel_px = (float(limit_px) - float(mid)) / float(mid) if mid > 0 else 0.0

        sigma = self._pmaker_paper_sigma(list(closes))
        mom_z = self._pmaker_paper_momentum_z(list(closes), sigma, window=10)

        try:
            x = surv.featurize(
                {
                    "spread_pct": float(sp),
                    "sigma": float(sigma),
                    "ofi_z": float(ofi_score),
                    "momentum_z": float(mom_z),
                    "liq_score": float(liq_score),
                    "attempt_idx": 0.0,
                    "rel_px": float(rel_px),
                }
            )
        except Exception:
            return

        self._pmaker_paper_active[sym] = {
            "side": side,
            "limit_px": float(limit_px),
            "start_ms": int(ts_ms),
            "timeout_ms": int(max(250, int(self.pmaker_paper_probe_timeout_ms))),
            "x": x,
        }
        self._pmaker_paper_last_start_ms[sym] = int(ts_ms)

    def runtime_config(self) -> Dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "decision_refresh_sec": float(self.decision_refresh_sec),
            "decision_eval_min_interval_sec": float(self.decision_eval_min_interval_sec),
            "decision_worker_sleep_sec": float(self.decision_worker_sleep_sec),
            "enable_orders": bool(self.enable_orders),
            "paper_trading_enabled": bool(self.paper_trading_enabled),
            "paper_flat_on_wait": bool(self.paper_flat_on_wait),
            "paper_use_engine_sizing": bool(self.paper_use_engine_sizing),
            "paper_engine_size_mult": float(self.paper_engine_size_mult),
            "paper_engine_size_min_frac": float(self.paper_engine_size_min_frac),
            "paper_engine_size_max_frac": float(self.paper_engine_size_max_frac),
            "paper_size_frac_default": float(self.paper_size_frac_default),
            "paper_leverage_default": float(self.paper_leverage_default),
            "paper_fee_roundtrip": float(self.paper_fee_roundtrip),
            "paper_slippage_bps": float(self.paper_slippage_bps),
            "paper_min_hold_sec": int(self.paper_min_hold_sec),
            "paper_max_hold_sec": int(self.paper_max_hold_sec),
            "paper_max_positions": int(self.paper_max_positions),
            "paper_exit_policy_only": bool(self.paper_exit_policy_only),
            "mc_n_paths_live": int(self.mc_n_paths_live),
            "mc_n_paths_exit": int(self.mc_n_paths_exit),
            "mc_tail_mode": str(self.mc_tail_mode),
            "mc_student_t_df": float(self.mc_student_t_df),
            "exec_mode": str(os.environ.get("EXEC_MODE", config.EXEC_MODE)).strip().lower(),
        }

    def set_enable_orders(self, enabled: bool) -> None:
        self.enable_orders = bool(enabled)
        try:
            config.ENABLE_LIVE_ORDERS = bool(self.enable_orders)
        except Exception:
            pass
        # Keep paper mode consistent with live ordering.
        self.paper_trading_enabled = _env_bool("PAPER_TRADING", True) and (not self.enable_orders)

    def score_debug_for_symbol(self, sym: str) -> Dict[str, Any]:
        sym = str(sym).strip()
        return {
            "symbol": sym,
            "market": (getattr(self, "data", None).market.get(sym) if getattr(self, "data", None) is not None else None),
            "orderbook": (getattr(self, "data", None).orderbook.get(sym) if getattr(self, "data", None) is not None else None),
            "position": (self.positions.get(sym) if isinstance(getattr(self, "positions", None), dict) else None),
            "decision_cache": (self._decision_cache.get(sym) if isinstance(getattr(self, "_decision_cache", None), dict) else None),
            "ts_ms": int(now_ms()),
        }

    def liquidate_all_positions(self) -> None:
        if self.enable_orders and (not self.paper_trading_enabled):
            self._log_err("[WARN] liquidate_all_positions called in live mode; ignoring")
            return

        ts_ms = int(now_ms())
        for sym in list((self.positions or {}).keys()):
            pos = (self.positions or {}).get(sym) or {}
            mkt = getattr(self, "data", None).market.get(sym) if getattr(self, "data", None) is not None else {}
            exit_price = mkt.get("price") or mkt.get("last") or pos.get("current") or pos.get("price") or pos.get("entry_price")
            try:
                exit_price_f = float(exit_price)
            except Exception:
                continue
            if exit_price_f > 0:
                self._paper_close_position(sym=sym, exit_price=exit_price_f, ts_ms=ts_ms, reason="liquidate_all")

    async def liquidate_all_positions_live(self) -> None:
        # Stub: real live liquidation is intentionally not implemented here.
        self._log_err("[WARN] liquidate_all_positions_live is not implemented")

    async def live_sync_loop(self) -> None:
        # Stub loop for main.py compatibility when enable_orders=True.
        warned = False
        while True:
            if self.enable_orders and not warned:
                self._log_err("[WARN] live_sync_loop is a stub (no live sync)")
                warned = True
            await asyncio.sleep(float(_env_float("LIVE_SYNC_SLEEP_SEC", 2.0)))

    def _paper_mark_position(self, sym: str, mark_price: Optional[float], ts_ms: int) -> None:
        pos = self.positions.get(sym)
        if not pos or mark_price is None:
            return
        try:
            entry = float(pos.get("entry_price") or 0.0)
            qty = float(pos.get("size", 0.0) or 0.0)
            side = str(pos.get("side") or "").upper()
            if entry <= 0 or qty == 0:
                return
            if side == "SHORT":
                pnl = (entry - float(mark_price)) * qty
            else:
                pnl = (float(mark_price) - entry) * qty
            pos["current"] = float(mark_price)
            pos["price"] = float(mark_price)
            pos["unrealized_pnl"] = float(pnl)
            pos["pnl"] = float(pnl)

            margin = float(pos.get("margin") or 0.0)
            roe = (float(pnl) / margin) if margin > 0 else 0.0
            pos["roe"] = float(roe)
            
            # ✅ Track Max ROE for Trailing Stop
            current_max = float(pos.get("max_roe") or -999.0)
            if roe > current_max:
                pos["max_roe"] = float(roe)

            t0 = int(pos.get("time") or ts_ms)
            pos["age_sec"] = max(0.0, (int(ts_ms) - t0) / 1000.0)
        except Exception:
            return

    @staticmethod
    def _mark_price(best_bid: Optional[float], best_ask: Optional[float], last_price: Optional[float]) -> Optional[float]:
        try:
            if best_bid is not None and best_ask is not None:
                b = float(best_bid)
                a = float(best_ask)
                if b > 0 and a > 0:
                    return 0.5 * (b + a)
        except Exception:
            pass
        try:
            if last_price is not None:
                p = float(last_price)
                if p > 0:
                    return p
        except Exception:
            pass
        return None

    def _paper_append_trade(self, trade: Dict[str, Any]) -> None:
        # Prefer a human-readable clock time for the dashboard, while keeping the epoch in `ts`.
        try:
            ts_ms = trade.get("ts")
            if ts_ms is None:
                ts_ms = trade.get("ts_ms")
            if ts_ms is None:
                ts_ms = trade.get("time")
            ts_i = int(ts_ms) if ts_ms is not None else None
            if ts_i is not None and ts_i > 0:
                trade.setdefault("ts", int(ts_i))
                # keep `time` stable even if it's already a string from legacy data
                if not isinstance(trade.get("time"), str):
                    trade["time"] = time.strftime("%H:%M:%S", time.localtime(int(ts_i) / 1000.0))
        except Exception:
            pass
        self.trade_tape.append(trade)
        
        # DB에 trade 기록
        if self.db:
            try:
                mode = TradingMode.LIVE if self.is_live_mode else TradingMode.PAPER
                trade_with_mode = dict(trade)
                trade_with_mode["trading_mode"] = self.trading_mode
                self.db.log_trade_background(trade_with_mode, mode=mode)
            except Exception as e:
                self._log_err(f"[ERR] DB log_trade: {e}")

    def _paper_fee_roundtrip_from_engine_meta(self, meta: Dict[str, Any]) -> Optional[float]:
        """
        Engine `fee_roundtrip_total` includes `expected_spread_cost`.
        Paper trading entries/exits are filled at bid/ask, so spread is already realized in PnL.
        Subtract spread here to avoid double-counting costs in paper PnL.
        """
        if not isinstance(meta, dict):
            return None
        try:
            total = float(meta.get("fee_roundtrip_total") or 0.0)
        except Exception:
            return None
        if total <= 0.0 or (not math.isfinite(total)):
            return None
        try:
            spread = float(meta.get("expected_spread_cost") or 0.0)
        except Exception:
            spread = 0.0
        if (not math.isfinite(spread)) or spread < 0.0:
            spread = 0.0
        fee = float(total) - float(spread)
        if (not math.isfinite(fee)) or fee < 0.0:
            fee = 0.0
        return float(fee)

    @staticmethod
    def _paper_exec_mode() -> str:
        return str(os.environ.get("EXEC_MODE", config.EXEC_MODE)).strip().lower()

    @staticmethod
    def _paper_p_maker_from_detail(detail: Optional[Dict[str, Any]], *, leg: str) -> float:
        if not isinstance(detail, dict):
            return 0.0
        meta = detail.get("meta") if isinstance(detail.get("meta"), dict) else {}
        keys = ("pmaker_entry", "p_maker") if leg == "entry" else ("pmaker_exit", "pmaker_entry", "p_maker")
        p = None
        for k in keys:
            if meta.get(k) is not None:
                p = meta.get(k)
                break
        try:
            v = float(p) if p is not None else 0.0
        except Exception:
            v = 0.0
        if not math.isfinite(v):
            v = 0.0
        return float(max(0.0, min(1.0, v)))

    def _paper_fill_price(
        self,
        *,
        side: str,
        leg: str,
        best_bid: Optional[float],
        best_ask: Optional[float],
        mark_price: Optional[float],
        detail: Optional[Dict[str, Any]],
    ) -> tuple[Optional[float], str]:
        """
        Paper execution model aligned to EXEC_MODE:
        - market: always taker at bid/ask
        - maker_then_market: Bernoulli(p_maker) maker at touch, else taker

        Returns (fill_price, exec_tag) where exec_tag ∈ {"maker","taker","mark"}.
        """
        side_u = str(side or "").upper()
        if side_u not in ("LONG", "SHORT"):
            return None, "mark"
        leg_u = str(leg or "").lower()
        if leg_u not in ("entry", "exit"):
            leg_u = "entry"

        exec_mode = self._paper_exec_mode()

        def _f(x):
            try:
                v = float(x) if x is not None else None
            except Exception:
                v = None
            if v is None or (not math.isfinite(v)) or v <= 0.0:
                return None
            return float(v)

        bid = _f(best_bid)
        ask = _f(best_ask)
        mark = _f(mark_price)

        if leg_u == "entry":
            px_taker = ask if side_u == "LONG" else bid
            px_maker = bid if side_u == "LONG" else ask
            is_buy = side_u == "LONG"
        else:
            px_taker = bid if side_u == "LONG" else ask
            px_maker = ask if side_u == "LONG" else bid
            is_buy = side_u == "SHORT"

        px = px_taker
        tag = "taker"
        if exec_mode == "maker_then_market":
            p_maker = self._paper_p_maker_from_detail(detail, leg=leg_u)
            if p_maker > 0.0 and px_maker is not None and random.random() < p_maker:
                px = px_maker
                tag = "maker"
        if px is None:
            return (mark, "mark") if mark is not None else (None, "mark")

        # Optional slippage (bps): apply to taker fills only (maker fills assume near-zero slip).
        slip = float(max(0.0, self.paper_slippage_bps)) / 10000.0
        if tag == "taker" and slip > 0:
            if is_buy:
                px = float(px) * (1.0 + slip)
            else:
                px = float(px) * (1.0 - slip)
        return float(px), tag

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
        lev = float(max(0.0, min(float(leverage), float(self.max_leverage))))
        if cap <= 0.0 or lev <= 0.0:
            return

        margin = float(self.balance) * cap
        notional = margin * lev
        qty = notional / float(entry_price)
        if qty <= 0:
            return

        self.positions[sym] = {
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

        # ✅ Dynamic Horizon Hold: 진입 시 결정된 Horizon을 최소 보유 시간으로 설정
        if detail is not None:
            meta = detail.get("meta") or {}
            entry_h = 0
            if side_u == "LONG":
                entry_h = int(meta.get("policy_best_h_long") or 0)
            else:
                entry_h = int(meta.get("policy_best_h_short") or 0)
            
            if entry_h > 0:
                self.positions[sym]["dynamic_min_hold"] = int(entry_h)
            
            # ✅ Score-Based Exit: Track entry scores and max scores for trailing stop
            score_long = float(detail.get("policy_ev_score_long") or 0.0)
            score_short = float(detail.get("policy_ev_score_short") or 0.0)
            self.positions[sym]["entry_score_long"] = score_long
            self.positions[sym]["entry_score_short"] = score_short
            self.positions[sym]["max_score_long"] = score_long
            self.positions[sym]["max_score_short"] = score_short

        self._last_trade_event_by_sym[sym] = "ENTER"
        self._paper_append_trade(
            {
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
        )

    def _paper_close_position(
        self,
        *,
        sym: str,
        exit_price: float,
        ts_ms: int,
        reason: str,
    ) -> None:
        pos = self.positions.get(sym)
        if not pos:
            return
        try:
            side = str(pos.get("side") or "").upper()
            entry = float(pos.get("entry_price") or 0.0)
            qty = float(pos.get("size") or 0.0)
            margin = float(pos.get("margin") or 0.0)
            notional = float(pos.get("notional") or 0.0)
            fee_roundtrip = float(pos.get("fee_roundtrip") or 0.0)
        except Exception:
            return
        if entry <= 0 or qty == 0 or exit_price <= 0:
            return

        gross_pnl = ((entry - float(exit_price)) * qty) if side == "SHORT" else ((float(exit_price) - entry) * qty)
        fee_cost = abs(notional) * float(max(0.0, fee_roundtrip))
        realized_pnl = float(gross_pnl) - float(fee_cost)
        roe = (realized_pnl / margin) if margin > 0 else 0.0

        self.balance = float(self.balance) + float(realized_pnl)
        self._last_trade_event_by_sym[sym] = "EXIT"
        self._paper_append_trade(
            {
                "time": int(ts_ms),
                "sym": sym,
                "type": "EXIT",
                "action_type": "EXIT",
                "side": side,
                "entry_price": float(entry),
                "price": float(exit_price),
                "pnl": float(realized_pnl),
                "roe": float(roe),
                "leverage": float(pos.get("leverage") or 0.0),
                "cap_frac": float(pos.get("cap_frac") or 0.0),
                "notional": float(notional),
                "reason": str(reason or ""),
            }
        )
        self.positions.pop(sym, None)

    async def _close_position(
        self,
        *,
        sym: str,
        exit_price: float,
        ts_ms: int,
        reason: str,
    ) -> None:
        # Back-compat shim used by older dashboard/tests.
        if self.paper_trading_enabled or (not self.enable_orders):
            self._paper_close_position(sym=sym, exit_price=exit_price, ts_ms=ts_ms, reason=reason)
            return
        # Safety: avoid implicit real orders through legacy call sites.
        self._log_err(f"[WARN] _close_position called in live mode; ignoring (sym={sym})")

    def _paper_init_exit_policy_state(self, sym: str, detail: Optional[Dict[str, Any]], ts_ms: int) -> None:
        pos = self.positions.get(sym)
        if not pos:
            return

        meta = detail.get("meta") if isinstance(detail, dict) and isinstance(detail.get("meta"), dict) else {}
        horizon = None
        # Prefer the targeted horizon (Plan A: best single horizon) over weighted effective horizon.
        for k in ("policy_horizon_eff_sec", "policy_h_eff_sec", "best_horizon_steps"):
            try:
                v = meta.get(k)
                if v is None:
                    continue
                horizon = int(max(0, float(v)))
                break
            except Exception:
                continue
        if horizon is None or horizon <= 0:
            horizon = int(max(1, int(self.paper_exit_policy_horizon_sec_default)))

        pos["policy_horizon_sec"] = int(horizon)
        pos["policy_flip_streak"] = 0
        pos["policy_hold_bad"] = 0
        pos["policy_last_eval_ms"] = int(ts_ms)
        try:
            pos["policy_mu_annual"] = float(meta.get("mu_adjusted")) if meta.get("mu_adjusted") is not None else None
        except Exception:
            pos["policy_mu_annual"] = None
        try:
            sigma_v = None
            for k in ("sigma_annual", "policy_paths_sigma_annual", "sigma_sim"):
                if meta.get(k) is not None:
                    sigma_v = meta.get(k)
                    break
            pos["policy_sigma_annual"] = float(sigma_v) if sigma_v is not None else None
        except Exception:
            pos["policy_sigma_annual"] = None

    def _paper_exit_policy_signal(
        self,
        *,
        sym: str,
        pos: Dict[str, Any],
        ts_ms: int,
        best_bid: Optional[float],
        best_ask: Optional[float],
        mark_price: Optional[float],
        detail: Optional[Dict[str, Any]],
    ) -> tuple[str, str]:
        """
        Returns (action, reason) where action ∈ {"HOLD","EXIT","FLIP_LONG","FLIP_SHORT"}.
        This is a lightweight, real-time approximation of the engine's exit-policy rules.
        """
        side = str(pos.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return "HOLD", "policy:bad_side"
        side_now = 1 if side == "LONG" else -1
        alt_side = -side_now
        alt_label = "LONG" if alt_side == 1 else "SHORT"

        age_sec = float(pos.get("age_sec") or 0.0)
        try:
            t0 = int(pos.get("time") or ts_ms)
            age_sec = max(age_sec, max(0.0, (int(ts_ms) - t0) / 1000.0))
        except Exception:
            pass

        min_hold = int(max(0, int(self.paper_exit_policy_min_hold_sec)))
        horizon = int(max(1, int(pos.get("policy_horizon_sec") or 0) or int(self.paper_exit_policy_horizon_sec_default)))
        tau = float(max(0.0, float(horizon) - float(age_sec)))

        # time_stop (Relaxed: Do not exit solely on time, rely on score/trailing_stop)
        # if horizon > 0 and age_sec >= float(horizon):
        #     px_exit = best_bid if side == "LONG" else best_ask
        #     if px_exit is None:
        #         px_exit = mark_price
        #     if px_exit is None:
        #         return "HOLD", "policy:time_stop(no_px)"
        #     return "EXIT", f"policy:time_stop(h={horizon}s age={age_sec:.1f}s)"

        # dd_stop (realized path based)
        if self.paper_exit_policy_dd_stop_enabled and age_sec >= float(min_hold):
            try:
                roe = float(pos.get("roe") or 0.0)
            except Exception:
                roe = 0.0
            if roe <= float(self.paper_exit_policy_dd_stop_roe):
                return "EXIT", f"policy:dd_stop(roe={roe:.4f}<= {float(self.paper_exit_policy_dd_stop_roe):.4f})"

        # ✅ Profit Preservation: Trailing Stop & Breakeven (Dynamic Fee Barrier)
        try:
            roe = float(pos.get("roe") or 0.0)
            max_roe = float(pos.get("max_roe") or -999.0)
            lev = float(pos.get("leverage") or 1.0)
            
            # Dynamic Fee Calculation:
            # Check execution mode to determine fee basis.
            # Maker (0.02% + 0.01% slip ~ 0.03% RT) vs Taker (0.06% + 0.06% slip ~ 0.12% RT)
            exec_mode = self._paper_exec_mode()
            if exec_mode == "maker_then_market":
                fee_base = 0.0003 # 0.03% Roundtrip for Maker Strategy
            else:
                fee_base = 0.0012 # 0.12% Roundtrip for Taker Strategy

            # Fee in ROE units = fee_base * Leverage
            fee_roe_total = fee_base * lev
            
            # Dynamic Barriers (Wider Buffers for Slippage)
            # Slippage on exit (market order) can be 0.1%~0.3% ROE impact (at high lev).
            # We need larger safety margin.
            trailing_activation = fee_roe_total + 0.010  # Fee + 1.0% Pure Profit (Buffer for slippage)
            breakeven_activation = fee_roe_total + 0.005 # Fee + 0.5% Buffer
            
            # 1. Trailing Stop
            if max_roe >= trailing_activation:
                # Trail by 0.5% or more to allow breathing room, but lock in profits.
                drop_threshold = max_roe - 0.005
                if roe <= drop_threshold:
                     return "EXIT", f"policy:trailing_stop(max={max_roe:.4f} cur={roe:.4f} fee_roe={fee_roe_total:.4f})"
            
            # 2. Breakeven
            if max_roe >= breakeven_activation:
                # Exit if we fall back to just above fee cost (give 0.3% buffer above fee for slippage)
                exit_floor = fee_roe_total + 0.003
                if roe <= exit_floor:
                    return "EXIT", f"policy:breakeven(max={max_roe:.4f} cur={roe:.4f} floor={exit_floor:.4f})"
        except Exception:
            pass

        # throttle decision ticks to the policy DT (avoid counter explosion on fast loops)
        dt_sec = int(max(1, int(self.paper_exit_policy_decision_dt_sec)))
        last_eval = pos.get("policy_last_eval_ms")
        if last_eval is not None:
            try:
                if (int(ts_ms) - int(last_eval)) < int(dt_sec * 1000):
                    return "HOLD", "policy:dt_skip"
            except Exception:
                pass
        pos["policy_last_eval_ms"] = int(ts_ms)

        # ✅ Score-Based Exit: Update and check score trailing stop
        if detail is not None:
            try:
                # Get current scores from detail
                current_score_long = float(detail.get("policy_ev_score_long") or 0.0)
                current_score_short = float(detail.get("policy_ev_score_short") or 0.0)
                
                # Update max_score
                max_score_long = pos.get("max_score_long", -999.0)
                max_score_short = pos.get("max_score_short", -999.0)
                if current_score_long > max_score_long:
                    pos["max_score_long"] = current_score_long
                    max_score_long = current_score_long
                if current_score_short > max_score_short:
                    pos["max_score_short"] = current_score_short
                    max_score_short = current_score_short
                
                # Get trailing factor from env (default 0.6 = 60% retention)
                try:
                    trailing_factor = float(os.environ.get("POLICY_SCORE_TRAILING_FACTOR", "0.6") or 0.6)
                except Exception:
                    trailing_factor = 0.6
                trailing_factor = float(max(0.3, min(0.9, trailing_factor)))
                
                # Get flip margin from env (default 0.001)
                try:
                    flip_margin = float(os.environ.get("POLICY_SCORE_FLIP_MARGIN", "0.001") or 0.001)
                except Exception:
                    flip_margin = 0.001
                flip_margin = float(max(0.0, flip_margin))
                
                # 1. Score Trailing Stop: exit if score drops 40% from peak
                if side == "LONG":
                    if max_score_long > 0 and current_score_long < max_score_long * trailing_factor:
                        return "EXIT", f"score_trailing(max={max_score_long:.6f} cur={current_score_long:.6f} factor={trailing_factor:.2f})"
                elif side == "SHORT":
                    if max_score_short > 0 and current_score_short < max_score_short * trailing_factor:
                        return "EXIT", f"score_trailing(max={max_score_short:.6f} cur={current_score_short:.6f} factor={trailing_factor:.2f})"
                
                # 2. Weighted Consensus Flip: exit/flip if opposite direction becomes stronger
                if side == "LONG":
                    if current_score_short > current_score_long + flip_margin:
                        return "FLIP_SHORT", f"consensus_flip(L={current_score_long:.6f} S={current_score_short:.6f} margin={flip_margin:.6f})"
                elif side == "SHORT":
                    if current_score_long > current_score_short + flip_margin:
                        return "FLIP_LONG", f"consensus_flip(L={current_score_long:.6f} S={current_score_short:.6f} margin={flip_margin:.6f})"
            except Exception:
                pass

        meta = detail.get("meta") if isinstance(detail, dict) and isinstance(detail.get("meta"), dict) else {}
        mu_annual = meta.get("mu_adjusted")
        sigma_annual = None
        for k in ("sigma_annual", "policy_paths_sigma_annual", "sigma_sim"):
            if meta.get(k) is not None:
                sigma_annual = meta.get(k)
                break
        if mu_annual is None:
            mu_annual = pos.get("policy_mu_annual")
        if sigma_annual is None:
            sigma_annual = pos.get("policy_sigma_annual")
        try:
            mu_annual_f = float(mu_annual)
        except Exception:
            return "HOLD", "policy:no_mu"
        try:
            sigma_annual_f = float(sigma_annual)
        except Exception:
            return "HOLD", "policy:no_sigma"
        if sigma_annual_f <= 0:
            return "HOLD", "policy:bad_sigma"

        # convert annualized -> per-second (dt=1/SECONDS_PER_YEAR)
        mu_ps = float(mu_annual_f) / float(SECONDS_PER_YEAR)
        sigma_ps = float(sigma_annual_f) / math.sqrt(float(SECONDS_PER_YEAR))

        try:
            lev = float(pos.get("leverage") or 0.0)
        except Exception:
            lev = 0.0
        if lev <= 0:
            return "HOLD", "policy:bad_lev"

        try:
            fee_rt = float(pos.get("fee_roundtrip") or 0.0)
        except Exception:
            fee_rt = 0.0
        # `fee_roundtrip` is stored as a per-notional rate (paper PnL: fee = notional * fee_rt).
        # `_approx_p_pos_and_ev_hold` operates in ROE units, so scale costs by leverage.
        fee_exit_only_notional = max(0.0, 0.5 * float(fee_rt))
        fee_exit_only = float(fee_exit_only_notional) * float(lev)
        exec_oneway = float(fee_exit_only)
        switch_cost = float(max(0.0, 2.0 * float(exec_oneway)))

        # scores / probabilities (deterministic approximation)
        p_pos_cur, score_cur = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, side_now, lev, fee_exit_only)
        p_pos_alt, score_alt_raw = _approx_p_pos_and_ev_hold(mu_ps, sigma_ps, tau, alt_side, lev, fee_exit_only)
        score_alt = float(score_alt_raw) - float(switch_cost)
        gap_eff = float(score_cur) - float(score_alt)

        mgn = float(max(0.0, float(self.paper_exit_policy_score_margin)))
        hold_value_ok = bool(gap_eff >= -mgn)
        hold_ok = bool(hold_value_ok and (float(p_pos_cur) >= float(self.paper_exit_policy_p_pos_hold_floor)))

        hold_bad = int(pos.get("policy_hold_bad") or 0)
        if (float(score_cur) < -mgn) or (not hold_ok):
            hold_bad += 1
        else:
            hold_bad = 0
        pos["policy_hold_bad"] = int(hold_bad)

        alt_value_after_cost = float(score_alt_raw) - float(exec_oneway)
        flip_ok = bool(
            (float(p_pos_alt) >= float(self.paper_exit_policy_p_pos_enter_floor))
            and ((alt_value_after_cost > 0.0) or (alt_value_after_cost > float(self.paper_exit_policy_soft_floor)))
        )
        pref_side = alt_side if (gap_eff < -mgn and flip_ok) else None

        flip_streak = int(pos.get("policy_flip_streak") or 0)
        if pref_side is not None and int(pref_side) == int(alt_side):
            flip_streak += 1
        else:
            flip_streak = 0
        pos["policy_flip_streak"] = int(flip_streak)

        # emit actions only after min_hold (Dynamic)
        dynamic_hold = int(pos.get("dynamic_min_hold") or 0)
        effective_min_hold = max(float(min_hold), float(dynamic_hold))
        
        if age_sec >= effective_min_hold:
            if flip_streak >= int(max(1, int(self.paper_exit_policy_flip_confirm_ticks))):
                return (
                    f"FLIP_{alt_label}",
                    f"policy:score_flip(gap={gap_eff:.6f} p_cur={p_pos_cur:.3f} p_alt={p_pos_alt:.3f} score_cur={score_cur:.6f} score_alt={score_alt_raw:.6f})",
                )
            if hold_bad >= int(max(1, int(self.paper_exit_policy_hold_bad_ticks))):
                return (
                    "EXIT",
                    f"policy:hold_bad(n={hold_bad} p_cur={p_pos_cur:.3f} score_cur={score_cur:.6f} gap={gap_eff:.6f})",
                )

        return "HOLD", f"policy:hold(p={p_pos_cur:.3f} gap={gap_eff:.6f} tau={tau:.0f}s)"

    def _paper_trade_step(
        self,
        *,
        sym: str,
        desired_action: str,
        ts_ms: int,
        best_bid: Optional[float],
        best_ask: Optional[float],
        mark_price: Optional[float],
        detail: Optional[Dict[str, Any]],
        decision_reason: str,
    ) -> None:
        if not self.paper_trading_enabled:
            return
        if sym is None:
            return

        # mark existing position before applying signals
        self._paper_mark_position(sym, mark_price, ts_ms)

        pos = self.positions.get(sym)
        age_sec = float(pos.get("age_sec") or 0.0) if pos else 0.0
        if pos and self.paper_exit_policy_only:
            action, pol_reason = self._paper_exit_policy_signal(
                sym=sym,
                pos=pos,
                ts_ms=ts_ms,
                best_bid=best_bid,
                best_ask=best_ask,
                mark_price=mark_price,
                detail=detail,
            )
            if action == "EXIT":
                px_exit, exec_tag = self._paper_fill_price(
                    side=str(pos.get("side") or ""),
                    leg="exit",
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mark_price=mark_price,
                    detail=detail,
                )
                if px_exit is not None:
                    self._paper_close_position(
                        sym=sym,
                        exit_price=float(px_exit),
                        ts_ms=ts_ms,
                        reason=f"{pol_reason} | {decision_reason} | exec={exec_tag}",
                    )
                return
            if action in ("FLIP_LONG", "FLIP_SHORT"):
                new_side = "LONG" if action == "FLIP_LONG" else "SHORT"
                # close
                px_exit, exec_tag_exit = self._paper_fill_price(
                    side=str(pos.get("side") or ""),
                    leg="exit",
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mark_price=mark_price,
                    detail=detail,
                )
                if px_exit is not None:
                    self._paper_close_position(
                        sym=sym,
                        exit_price=float(px_exit),
                        ts_ms=ts_ms,
                        reason=f"{pol_reason} | {decision_reason} | exec={exec_tag_exit}",
                    )
                # open
                px_entry, exec_tag_entry = self._paper_fill_price(
                    side=new_side,
                    leg="entry",
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mark_price=mark_price,
                    detail=detail,
                )
                if px_entry is None:
                    return
                cap_frac = float(pos.get("cap_frac") or self.paper_size_frac_default)
                leverage = float(pos.get("leverage") or self.paper_leverage_default)
                # Apply engine-suggested leverage on flips as well (if provided).
                if detail:
                    try:
                        lev_opt = detail.get("optimal_leverage")
                        if lev_opt is not None and float(lev_opt) > 0:
                            leverage = float(lev_opt)
                    except Exception:
                        pass
                fee_roundtrip = float(pos.get("fee_roundtrip") or self.paper_fee_roundtrip)
                if fee_roundtrip <= 0.0 and detail:
                    meta = detail.get("meta") if isinstance(detail.get("meta"), dict) else {}
                    fee_from_meta = self._paper_fee_roundtrip_from_engine_meta(meta)
                    if fee_from_meta is not None:
                        fee_roundtrip = float(fee_from_meta)
                self._paper_open_position(
                    sym=sym,
                    side=new_side,
                    entry_price=float(px_entry),
                    ts_ms=ts_ms,
                    cap_frac=cap_frac,
                    leverage=leverage,
                    fee_roundtrip=fee_roundtrip,
                    reason=f"{pol_reason} | {decision_reason} | exec={exec_tag_entry}",
                )
                self._paper_init_exit_policy_state(sym, detail, ts_ms)
                return
            # HOLD: ignore legacy WAIT->FLAT / max_hold / signal flips for open positions
            return
        max_hold_sec = None
        if pos is not None:
            try:
                max_hold_sec = float(pos.get("policy_horizon_sec") or self.paper_max_hold_sec)
            except Exception:
                max_hold_sec = float(self.paper_max_hold_sec)
        if pos and max_hold_sec and (float(max_hold_sec) > 0) and (age_sec >= float(max_hold_sec)):
            px_exit, exec_tag = self._paper_fill_price(
                side=str(pos.get("side") or ""),
                leg="exit",
                best_bid=best_bid,
                best_ask=best_ask,
                mark_price=mark_price,
                detail=detail,
            )
            if px_exit is not None:
                self._paper_close_position(
                    sym=sym,
                    exit_price=float(px_exit),
                    ts_ms=ts_ms,
                    reason=f"TIME_EXIT {decision_reason} | exec={exec_tag}",
                )
            return

        action = str(desired_action or "WAIT").upper()
        want_side = action if action in ("LONG", "SHORT") else None

        # default: WAIT -> flat
        if want_side is None:
            if pos and self.paper_flat_on_wait and (age_sec >= float(self.paper_min_hold_sec)):
                px_exit, exec_tag = self._paper_fill_price(
                    side=str(pos.get("side") or ""),
                    leg="exit",
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mark_price=mark_price,
                    detail=detail,
                )
                if px_exit is not None:
                    self._paper_close_position(
                        sym=sym,
                        exit_price=float(px_exit),
                        ts_ms=ts_ms,
                        reason=f"EXIT_WAIT {decision_reason} | exec={exec_tag}",
                    )
            return

        # respect min hold on flips
        if pos and str(pos.get("side") or "").upper() != str(want_side).upper() and (age_sec < float(self.paper_min_hold_sec)):
            return

        # choose sizing from engine detail when available
        cap_frac = float(self.paper_size_frac_default)
        leverage = float(self.paper_leverage_default)
        fee_roundtrip = float(self.paper_fee_roundtrip)
        # Apply engine-suggested leverage even when engine sizing is disabled (keeps sizing toggle about cap_frac).
        if detail:
            try:
                lev_opt = detail.get("optimal_leverage")
                if lev_opt is not None and float(lev_opt) > 0:
                    leverage = float(lev_opt)
                    self._log(f"🔍 [PAPER_DEBUG] {sym} | Applied OPTIMAL_LEVERAGE from engine: {leverage}")
                else:
                    self._log(f"🔍 [PAPER_DEBUG] {sym} | No optimal leverage in detail, using default: {leverage}")
            except Exception:
                pass
        if self.paper_use_engine_sizing and detail:
            engine_size = None
            try:
                v = detail.get("optimal_size")
                if v is None:
                    v = detail.get("size_frac")
                if v is not None:
                    engine_size = float(v)
            except Exception:
                engine_size = None
            if engine_size is not None and engine_size > 0:
                mult = float(max(0.0, float(self.paper_engine_size_mult)))
                mn = float(max(0.0, float(self.paper_engine_size_min_frac)))
                mx = float(max(mn, min(1.0, float(self.paper_engine_size_max_frac))))
                cap_frac = float(engine_size) * mult
                cap_frac = float(max(mn, min(mx, cap_frac)))
            meta = detail.get("meta") if isinstance(detail.get("meta"), dict) else {}
            if fee_roundtrip <= 0.0 and isinstance(meta, dict):
                try:
                    fee_from_meta = self._paper_fee_roundtrip_from_engine_meta(meta)
                    if fee_from_meta is not None:
                        fee_roundtrip = float(fee_from_meta)
                except Exception:
                    fee_roundtrip = 0.0

        cap_frac = float(max(0.0, min(1.0, cap_frac)))
        leverage = float(max(0.0, min(float(leverage), float(self.max_leverage))))

        # exposure / position cap
        if not pos:
            if (self.paper_max_positions > 0) and (len(self.positions) >= int(self.paper_max_positions)):
                return
            try:
                max_expo = float(getattr(config, "MAX_NOTIONAL_EXPOSURE", 0.0) or 0.0)
            except Exception:
                max_expo = 0.0
            if self.exposure_cap_enabled and max_expo > 0 and self.balance > 0:
                new_notional = float(self.balance) * cap_frac * leverage
                if (self._total_open_notional() + new_notional) / max(1.0, float(self.balance)) > max_expo:
                    return

        # execute: close/flip/open
        if pos and str(pos.get("side") or "").upper() != want_side:
            px_exit, exec_tag = self._paper_fill_price(
                side=str(pos.get("side") or ""),
                leg="exit",
                best_bid=best_bid,
                best_ask=best_ask,
                mark_price=mark_price,
                detail=detail,
            )
            if px_exit is not None:
                self._paper_close_position(
                    sym=sym,
                    exit_price=float(px_exit),
                    ts_ms=ts_ms,
                    reason=f"FLIP_EXIT {decision_reason} | exec={exec_tag}",
                )
            pos = None

        if not pos:
            px_entry, exec_tag = self._paper_fill_price(
                side=want_side,
                leg="entry",
                best_bid=best_bid,
                best_ask=best_ask,
                mark_price=mark_price,
                detail=detail,
            )
            if px_entry is None:
                return
            
            self._paper_open_position(
                sym=sym,
                side=want_side,
                entry_price=float(px_entry),
                ts_ms=ts_ms,
                cap_frac=cap_frac,
                leverage=leverage,
                fee_roundtrip=fee_roundtrip,
                reason=f"{decision_reason} | exec={exec_tag}",
                detail=detail,
            )
            self._paper_init_exit_policy_state(sym, detail, ts_ms)
            return

        # mark after
        self._paper_mark_position(sym, mark_price, ts_ms)

    # -----------------------------
    # logging / retry helpers
    # -----------------------------
    def _log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": "INFO", "msg": str(text)})
        if config.LOG_STDOUT:
            print(text)

    def _log_err(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": "ERROR", "msg": str(text)})
        if config.LOG_STDOUT:
            print(text)

    async def _ccxt_call(self, label: str, fn, *args, **kwargs):
        timeout_sec = float(getattr(config, "CCXT_TIMEOUT_MS", 0) or 0) / 1000.0
        retry_keys = (
            "TimeoutError",
            "RequestTimeout",
            "DDoSProtection",
            "ExchangeNotAvailable",
            "NetworkError",
            "ETIMEDOUT",
            "ECONNRESET",
            "502",
            "503",
            "504",
        )
        for attempt in range(1, int(config.MAX_RETRY) + 1):
            try:
                async with self._net_sem:
                    coro = fn(*args, **kwargs)
                    if timeout_sec > 0:
                        return await asyncio.wait_for(coro, timeout=timeout_sec)
                    return await coro
            except Exception as e:
                exc_name = type(e).__name__
                msg = str(e) or exc_name
                is_retryable = isinstance(e, asyncio.TimeoutError) or any(k in msg for k in retry_keys) or any(
                    k in exc_name for k in retry_keys
                )
                if (attempt >= int(config.MAX_RETRY)) or (not is_retryable):
                    raise
                backoff = (float(config.RETRY_BASE_SEC) * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                self._log_err(f"[WARN] {label} retry {attempt}/{config.MAX_RETRY} err={msg} sleep={backoff:.2f}s")
                await asyncio.sleep(backoff)

    # -----------------------------
    # persistence (best-effort)
    # -----------------------------
    def _load_json(self, path: Path, default):
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self._log_err(f"[ERR] load {path.name}: {e}")
        return default

    def _load_persisted_state(self) -> None:
        # Load balance
        try:
            val = self._load_json(self.state_files["balance"], None)
            if val is not None:
                self.balance = float(val)
        except Exception:
            pass

        # Load positions with stale check
        try:
            pos_data = self._load_json(self.state_files["positions"], {})
            if isinstance(pos_data, list):  # Handle list format from json
                # Convert list to dict keyed by symbol if needed, or assume it's lost structure
                # For safety, if it's a list, we just drop it or try to recover.
                # The current code expects a dict.
                pos_data = {} 
            
            if isinstance(pos_data, dict) and pos_data:
                valid_positions = {}
                now = now_ms()
                stale_threshold_ms = 24 * 3600 * 1000  # 24 hours
                
                discarded_count = 0
                for sym, p in pos_data.items():
                    # Check age
                    entry_time = int(p.get("time") or p.get("entry_time") or 0)
                    if now - entry_time > stale_threshold_ms:
                        discarded_count += 1
                        continue
                    valid_positions[sym] = p
                
                if discarded_count > 0:
                    self._log(f"[WARN] Discarded {discarded_count} stale positions (>24h old) from state.")
                
                self.positions = valid_positions
        except Exception:
             self.positions = {}

        # Load trade tape
        try:
            tape = self._load_json(self.state_files["trade"], [])
            if isinstance(tape, list):
                self.trade_tape = deque(tape, maxlen=20_000)
        except Exception:
            pass

    def _persist_state(self, force: bool = False) -> None:
        ts = now_ms()
        if not force and (ts - self._last_state_persist_ms < 5_000):
            return
        self._last_state_persist_ms = ts
        
        # JSON 파일 저장 (레거시 호환)
        try:
            self.state_files["balance"].write_text(str(self.balance), encoding="utf-8")
            self.state_files["positions"].write_text(json.dumps(self.positions), encoding="utf-8")
            self.state_files["trade"].write_text(json.dumps(list(self.trade_tape)), encoding="utf-8")
        except Exception:
            pass
        
        # SQLite DB 저장
        if self.db:
            try:
                mode = TradingMode.LIVE if self.is_live_mode else TradingMode.PAPER
                
                # Balance 저장
                self.db.save_state_background("balance", self.balance)
                self.db.save_state_background("trading_mode", self.trading_mode)
                
                # Positions 저장
                for sym, pos in self.positions.items():
                    if isinstance(pos, dict):
                        self.db.save_position_background(sym, pos, mode=mode)
                
                # 최신 equity 기록
                if self._equity_history:
                    latest = self._equity_history[-1]
                    if isinstance(latest, dict):
                        equity_data = {
                            "timestamp_ms": latest.get("time", ts),
                            "total_equity": latest.get("equity", self.balance),
                            "wallet_balance": self.balance,
                        }
                        self.db.log_equity_background(equity_data, mode=mode)
            except Exception as e:
                self._log_err(f"[ERR] persist state (DB): {e}")

    # -----------------------------
    # market feature helpers
    # -----------------------------
    def _compute_returns_and_vol(self, prices):
        if prices is None or len(prices) < 10:
            return None, None
        log_returns = []
        for i in range(1, len(prices)):
            p0, p1 = prices[i - 1], prices[i]
            if p0 and p1 and p0 > 0 and p1 > 0:
                log_returns.append(math.log(p1 / p0))
        if len(log_returns) < 5:
            return None, None
        mu = sum(log_returns) / len(log_returns)
        var = sum((r - mu) ** 2 for r in log_returns) / len(log_returns)
        sigma = math.sqrt(var)
        return mu, sigma

    def _infer_regime(self, closes) -> str:
        if not closes or len(closes) < 30:
            return "chop"
        fast_period = min(80, len(closes))
        slow_period = min(200, len(closes))
        fast = sum(closes[-fast_period:]) / fast_period
        slow = sum(closes[-slow_period:]) / slow_period
        slope_short = closes[-1] - closes[max(0, len(closes) - 6)]
        slope_long = closes[-1] - closes[max(0, len(closes) - 40)]
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, min(len(closes), 180))]
        vol = float(np.std(rets)) if rets else 0.0
        if vol > 0.01 and abs(slope_short) < closes[-1] * 0.0015:
            return "volatile"
        if fast > slow and slope_long > 0 and slope_short > 0:
            return "bull"
        if fast < slow and slope_long < 0 and slope_short < 0:
            return "bear"
        return "chop"

    def _compute_ofi_score(self, sym: str) -> float:
        ob = self.orderbook.get(sym) or {}
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        bid_vol = sum(float(b[1]) * weights[i] for i, b in enumerate(bids[: len(weights)]) if len(b) >= 2)
        ask_vol = sum(float(a[1]) * weights[i] for i, a in enumerate(asks[: len(weights)]) if len(a) >= 2)
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return float((bid_vol - ask_vol) / denom)

    def _liquidity_score(self, sym: str) -> float:
        ob = self.orderbook.get(sym) or {}
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        vol = sum(float(b[1]) for b in bids[:5] if len(b) >= 2) + sum(float(a[1]) for a in asks[:5] if len(a) >= 2)
        return float(max(vol, 1.0))

    # -----------------------------
    # dashboard rows
    # -----------------------------
    def _extract_mc_meta(self, decision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not decision:
            return {}
        for d in decision.get("details", []) or []:
            if d.get("_engine") in ("mc_barrier", "mc_engine", "mc"):
                return d.get("meta", {}) or {}
        return decision.get("meta", {}) or {}

    def _decision_metrics(self, decision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        mc_meta = self._extract_mc_meta(decision)
        ev = _safe_float(decision.get("ev", mc_meta.get("ev", 0.0)) if decision else 0.0, 0.0)
        exec_cost = _safe_float(
            mc_meta.get("execution_cost", mc_meta.get("fee_roundtrip_fee_mix", 0.0)),
            0.0,
        )
        kelly = _safe_float(mc_meta.get("kelly", 0.0), 0.0)
        ev_adj = float(ev) - float(self.rebalance_cost_mult) * float(exec_cost)
        # Debug: surface potentially-missing MC outputs
        try:
            if float(ev) == 0.0:
                self._log(f"[METRICS] ev==0 exec_cost={exec_cost:.6f} kelly={kelly:.6f} meta_keys={list(mc_meta.keys())[:8]}")
        except Exception:
            pass
        return {"ev": ev, "exec_cost": exec_cost, "kelly": kelly, "ev_adj": ev_adj, "meta": mc_meta}

    def _run_portfolio_joint_sync(self, symbols: List[str], ai_scores: Dict[str, float]) -> Optional[Dict[str, Any]]:
        if not symbols:
            return None

        try:
            from engines.mc.portfolio_joint_sim import PortfolioJointSimEngine, PortfolioConfig
            from engines.mc.config import config as mc_config
        except Exception as e:  # pragma: no cover - defensive import guard
            self._log_err(f"[PORTFOLIO_JOINT] import failed: {e}")
            return None

        ohlcv_map = {}
        for sym in symbols:
            candles = self.data.ohlcv_buffer.get(sym, [])
            if candles:
                try:
                    ohlcv_map[sym] = [
                        (
                            float(c.get("open")),
                            float(c.get("high")),
                            float(c.get("low")),
                            float(c.get("close")),
                            float(c.get("volume")),
                        )
                        for c in candles
                    ]
                except Exception:
                    continue

        if not ohlcv_map:
            return None

        cfg = PortfolioConfig(
            days=int(getattr(mc_config, "portfolio_days", 3)),
            simulations=int(getattr(mc_config, "portfolio_simulations", 12000)),
            batch_size=int(getattr(mc_config, "portfolio_batch_size", 4000)),
            block_size=int(getattr(mc_config, "portfolio_block_size", 12)),
            min_history=int(getattr(mc_config, "portfolio_min_history", 180)),
            drift_k=float(getattr(mc_config, "portfolio_drift_k", 0.35)),
            score_clip=float(getattr(mc_config, "portfolio_score_clip", 1.0)),
            tilt_strength=float(getattr(mc_config, "portfolio_tilt_strength", 0.6)),
            use_jumps=bool(getattr(mc_config, "portfolio_use_jumps", True)),
            p_jump_market=float(getattr(mc_config, "portfolio_p_jump_market", 0.005)),
            p_jump_idio=float(getattr(mc_config, "portfolio_p_jump_idio", 0.007)),
            target_leverage=float(getattr(mc_config, "portfolio_target_leverage", 10.0)),
            individual_cap=float(getattr(mc_config, "portfolio_individual_cap", 3.0)),
            risk_aversion=float(getattr(mc_config, "portfolio_risk_aversion", 0.5)),
            var_alpha=float(getattr(mc_config, "portfolio_var_alpha", 0.05)),
            leverage=float(self.max_leverage),
            seed=None,
        )

        try:
            # Debug: report input sizes
            try:
                self._log(f"[PORTFOLIO_JOINT] building ohlcv_map entries={len(ohlcv_map)} ai_scores={list(ai_scores.keys())[:8]} cfg_target_lev={cfg.target_leverage}")
            except Exception:
                pass

            engine = PortfolioJointSimEngine(ohlcv_map, ai_scores, cfg)
            weights, report = engine.build_portfolio(symbols)
            report = dict(report)
            report["weights"] = weights
            report["target_leverage"] = float(cfg.target_leverage)
            try:
                self._log(f"[PORTFOLIO_JOINT] report E[PnL]={report.get('expected_portfolio_pnl', 0.0):.6f} cvar={report.get('cvar', 0.0):.6f}")
            except Exception:
                pass
            return report
        except Exception as e:  # pragma: no cover - runtime guard
            import traceback

            self._log_err(f"[PORTFOLIO_JOINT] run failed: {e}")
            try:
                self._log_err(traceback.format_exc())
            except Exception:
                pass
            return None

    async def _maybe_portfolio_joint(self, symbols: List[str], ai_scores: Dict[str, float]) -> Optional[Dict[str, Any]]:
        now = time.time()
        if (now - float(self._last_portfolio_joint_ts)) < float(self.portfolio_joint_interval_sec):
            self._log(f"[PORTFOLIO_JOINT] skipping; last_run={self._last_portfolio_joint_ts} interval={self.portfolio_joint_interval_sec}")
            return self._last_portfolio_report
        report = await asyncio.to_thread(self._run_portfolio_joint_sync, symbols, ai_scores)
        if report:
            self._last_portfolio_joint_ts = now
            self._last_portfolio_report = report
            self._log(
                f"[PORTFOLIO_JOINT] E[PnL]={report.get('expected_portfolio_pnl', 0.0):.6f} "
                f"CVaR={report.get('cvar', 0.0):.6f} weights={report.get('weights', {})}"
            )
        else:
            self._log("[PORTFOLIO_JOINT] report is None or empty")
        return report

    def _build_rebalance_plan(
        self,
        *,
        decision_map: Dict[str, Dict[str, Any]],
        metrics_map: Dict[str, Dict[str, Any]],
        joint_report: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not metrics_map:
            return {}

        entries = [(sym, m.get("ev_adj", 0.0)) for sym, m in metrics_map.items()]
        entries.sort(key=lambda x: float(x[1]), reverse=True)

        top_syms = [sym for sym, ev_adj in entries if ev_adj > 0][: int(self.rebalance_top_n)]
        plan: Dict[str, Dict[str, Any]] = {}
        joint_weights = (joint_report or {}).get("weights") or {}
        total_lev = float((joint_report or {}).get("target_leverage", 0.0) or (joint_report or {}).get("total_leverage_allocated", 0.0) or 0.0)

        for sym, metrics in metrics_map.items():
            ev_adj = float(metrics.get("ev_adj", 0.0))
            exec_cost = float(metrics.get("exec_cost", 0.0))
            kelly = float(metrics.get("kelly", 0.0))
            allowed = bool(sym in top_syms and ev_adj > 0)
            target_leverage = min(max(kelly, 0.0), float(self.rebalance_kelly_cap)) if allowed else None
            cap_frac = None
            if allowed and sym in joint_weights and total_lev > 0:
                cap_frac = max(0.0, min(1.0, float(joint_weights[sym]) / float(total_lev)))

            plan[sym] = {
                "allow_trade": allowed,
                "ev_adj": ev_adj,
                "exec_cost": exec_cost,
                "target_leverage": target_leverage,
                "target_cap_frac": cap_frac,
            }

        # cost-aware HOLD logging for symbols we choose not to rotate out
        best_ev_adj = entries[0][1] if entries else 0.0
        for sym, metrics in metrics_map.items():
            if sym in top_syms:
                self._rebalance_last_decision[sym] = "SWITCH"
                continue
            if sym in self.positions:
                delta_ev = float(best_ev_adj) - float(metrics.get("ev_adj", 0.0))
                threshold = float(self.rebalance_cost_mult) * float(metrics.get("exec_cost", 0.0))
                if delta_ev <= threshold:
                    prev = self._rebalance_last_decision.get(sym)
                    if prev != "HOLD":
                        self._log(
                            f"Rebalance Decision: HOLD {sym} (ΔEV={delta_ev:.6f} <= cost={threshold:.6f})"
                        )
                    self._rebalance_last_decision[sym] = "HOLD"
                    plan.setdefault(sym, {})["allow_trade"] = False
                else:
                    self._rebalance_last_decision[sym] = "SWITCH"

        # cache last plan for dashboard access
        try:
            self._last_rebalance_plan = plan
        except Exception:
            pass

        return plan

    def _row(self, sym: str, price: Any, ts: int, decision: Optional[Dict[str, Any]], candles: int, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        status = "WAIT"
        conf = 0.0
        reason = "-"

        meta = decision.get("meta", {}) if decision else {}
        mc_meta = self._extract_mc_meta(decision)
        mc_detail = None
        if decision:
            try:
                for d in decision.get("details", []) or []:
                    if d.get("_engine") in ("mc_barrier", "mc_engine", "mc"):
                        mc_detail = d
                        break
            except Exception:
                mc_detail = None

        if decision:
            status = str(decision.get("action", "WAIT"))
            conf = _safe_float(decision.get("confidence", 0.0), 0.0)
            reason = str(decision.get("reason", "")) or "-"

        market_ts = (self.market.get(sym) or {}).get("ts") or 0
        trade_age = ts - int(market_ts) if market_ts else None
        kline_age = ts - int((self._last_kline_ok_ms.get(sym) or 0)) if self._last_kline_ok_ms.get(sym) else None

        ob_ts = (self.orderbook.get(sym) or {}).get("ts") or 0
        ob_age = ts - int(ob_ts) if ob_ts else None
        ob_ready = bool((self.orderbook.get(sym) or {}).get("ready"))
        ev = _safe_float(decision.get("ev", mc_meta.get("ev", meta.get("ev", 0.0))) if decision else 0.0, 0.0)
        ev_raw = _safe_float(decision.get("ev_raw", mc_meta.get("ev_raw", meta.get("ev_raw", 0.0))) if decision else 0.0, 0.0)
        kelly = _safe_float(mc_meta.get("kelly", meta.get("kelly", 0.0)), 0.0)
        regime = (ctx or {}).get("regime") or meta.get("regime") or "-"

        def _opt_float(val):
            if val is None:
                return None
            try:
                return float(val)
            except Exception:
                return None

        # MC event diagnostics
        event_p_tp = _opt_float(mc_meta.get("event_p_tp", meta.get("event_p_tp")))
        event_p_timeout = _opt_float(mc_meta.get("event_p_timeout", meta.get("event_p_timeout")))
        event_t_median = _opt_float(mc_meta.get("event_t_median", meta.get("event_t_median")))
        event_ev_r = _opt_float(mc_meta.get("event_ev_r", meta.get("event_ev_r")))
        event_cvar_r = _opt_float(mc_meta.get("event_cvar_r", meta.get("event_cvar_r")))

        mc_win_rate = _opt_float(mc_meta.get("win_rate"))
        mc_cvar = _opt_float(mc_meta.get("cvar05"))

        funnel_reason = mc_meta.get("funnel_reason") or meta.get("funnel_reason")

        # mu_alpha diagnostics (direct fields for dashboard)
        mu_alpha = _opt_float(mc_meta.get("mu_alpha"))
        mu_alpha_raw = _opt_float(mc_meta.get("mu_alpha_raw"))
        mu_alpha_pmaker_fill_rate = _opt_float(mc_meta.get("mu_alpha_pmaker_fill_rate"))
        mu_alpha_pmaker_boost = _opt_float(mc_meta.get("mu_alpha_pmaker_boost"))

        pos = self.positions.get(sym) or {}
        pos_roe = _opt_float(pos.get("roe")) or 0.0
        pos_leverage = _opt_float(pos.get("leverage"))
        pos_cap_frac = _opt_float(pos.get("cap_frac"))
        pos_pnl = _opt_float(pos.get("unrealized_pnl"))
        action_type = self._last_trade_event_by_sym.get(sym, "-")

        # Diff 2/3/4/5/6/7 payload keys (mostly forwarded from mc_meta)
        policy_horizons = mc_meta.get("policy_horizons")
        policy_w_h = mc_meta.get("policy_w_h")
        policy_h_eff_sec = _opt_float(mc_meta.get("policy_h_eff_sec"))
        policy_ev_mix_long = _opt_float(mc_meta.get("policy_ev_mix_long"))
        policy_ev_mix_short = _opt_float(mc_meta.get("policy_ev_mix_short"))
        paths_reused = mc_meta.get("paths_reused")
        policy_ev_target = _opt_float(mc_meta.get("policy_ev_target"))
        policy_ev_bonus = _opt_float(mc_meta.get("policy_ev_bonus"))
        policy_ev_penalty = _opt_float(mc_meta.get("policy_ev_penalty"))
        policy_ev_adjust = _opt_float(mc_meta.get("policy_ev_adjust"))
        policy_ev_score_long = _opt_float(mc_meta.get("policy_ev_score_long"))
        policy_ev_score_short = _opt_float(mc_meta.get("policy_ev_score_short"))
        policy_best_ev_gap = _opt_float(mc_meta.get("policy_best_ev_gap"))
        policy_ev_gap = _opt_float(mc_meta.get("policy_ev_gap"))
        policy_ev_neighbor_veto_abs = _opt_float(mc_meta.get("policy_ev_neighbor_veto_abs"))

        policy_signal_strength = _opt_float(mc_meta.get("policy_signal_strength"))
        policy_h_eff_sec_prior = _opt_float(mc_meta.get("policy_h_eff_sec_prior"))
        policy_w_short_sum = _opt_float(mc_meta.get("policy_w_short_sum"))

        policy_exit_reason_counts_per_h_long = mc_meta.get("policy_exit_reason_counts_per_h_long")
        policy_exit_reason_counts_per_h_short = mc_meta.get("policy_exit_reason_counts_per_h_short")
        policy_exit_unrealized_dd_frac = _opt_float(mc_meta.get("policy_exit_unrealized_dd_frac"))
        policy_exit_hold_bad_frac = _opt_float(mc_meta.get("policy_exit_hold_bad_frac"))
        policy_exit_score_flip_frac = _opt_float(mc_meta.get("policy_exit_score_flip_frac"))

        exec_mode = str(os.environ.get("EXEC_MODE", config.EXEC_MODE)).strip().lower()
        pmaker_entry = _opt_float(mc_meta.get("pmaker_entry"))
        pmaker_entry_delay_sec = _opt_float(mc_meta.get("pmaker_entry_delay_sec"))
        pmaker_entry_delay_penalty_r = _opt_float(mc_meta.get("pmaker_entry_delay_penalty_r"))
        policy_entry_shift_steps = mc_meta.get("policy_entry_shift_steps")

        fee_roundtrip_fee_mix = _opt_float(mc_meta.get("fee_roundtrip_fee_mix"))
        fee_roundtrip_fee_taker = _opt_float(mc_meta.get("fee_roundtrip_fee_taker"))
        fee_roundtrip_fee_maker = _opt_float(mc_meta.get("fee_roundtrip_fee_maker"))
        execution_cost = _opt_float(mc_meta.get("execution_cost"))
        expected_spread_cost = _opt_float(mc_meta.get("expected_spread_cost"))
        slippage_dyn = _opt_float(mc_meta.get("slippage_dyn"))
        spread_pct_used = _opt_float(mc_meta.get("spread_pct"))

        optimal_leverage = _opt_float((mc_detail or {}).get("optimal_leverage"))

        # 포지션 출처 결정 (ws=거래소 WS, db=내부 DB/메모리)
        pos_source = "ws" if self.is_live_mode and pos.get("from_exchange") else "db"

        return {
            "symbol": sym,
            "price": price,
            "status": status,
            "action_type": action_type,
            "trading_mode": self.trading_mode,
            "pos_source": pos_source,
            "conf": conf,
            "mc": reason,
            "ev": ev,
            "ev_raw": ev_raw,
            "mc_win_rate": mc_win_rate,
            "mc_cvar": mc_cvar,
            "kelly": kelly,
            "optimal_leverage": optimal_leverage,
            "regime": regime,
            "candles": candles,
            "pos_roe": pos_roe,
            "pos_leverage": pos_leverage,
            "pos_cap_frac": pos_cap_frac,
            "pos_pnl": pos_pnl,
            "event_p_tp": event_p_tp,
            "event_p_timeout": event_p_timeout,
            "event_t_median": event_t_median,
            "event_ev_r": event_ev_r,
            "event_cvar_r": event_cvar_r,
            "funnel_reason": funnel_reason,
            "mu_alpha": mu_alpha,
            "mu_alpha_raw": mu_alpha_raw,
            "mu_alpha_pmaker_fill_rate": mu_alpha_pmaker_fill_rate,
            "mu_alpha_pmaker_boost": mu_alpha_pmaker_boost,

            # freshness
            "trade_age": trade_age,
            "kline_age": kline_age,
            "orderbook_age": ob_age,
            "orderbook_ready": ob_ready,

            # Diff 2: Multi-Horizon Policy Mix
            "policy_horizons": policy_horizons,
            "policy_w_h": policy_w_h,
            "policy_h_eff_sec": policy_h_eff_sec,
            "policy_ev_mix_long": policy_ev_mix_long,
            "policy_ev_mix_short": policy_ev_mix_short,
            "paths_reused": paths_reused,
            "policy_ev_target": policy_ev_target,
            "policy_ev_bonus": policy_ev_bonus,
            "policy_ev_penalty": policy_ev_penalty,
            "policy_ev_adjust": policy_ev_adjust,
            "policy_ev_score_long": policy_ev_score_long,
            "policy_ev_score_short": policy_ev_score_short,
            "policy_best_ev_gap": policy_best_ev_gap,
            "policy_ev_gap": policy_ev_gap,
            "policy_ev_neighbor_veto_abs": policy_ev_neighbor_veto_abs,

            # Diff 3: Rule-based Dynamic Weights
            "policy_signal_strength": policy_signal_strength,
            "policy_h_eff_sec_prior": policy_h_eff_sec_prior,
            "policy_w_short_sum": policy_w_short_sum,

            # Diff 4: Exit Reason 통계
            "policy_exit_reason_counts_per_h_long": policy_exit_reason_counts_per_h_long,
            "policy_exit_reason_counts_per_h_short": policy_exit_reason_counts_per_h_short,
            "policy_exit_unrealized_dd_frac": policy_exit_unrealized_dd_frac,
            "policy_exit_hold_bad_frac": policy_exit_hold_bad_frac,
            "policy_exit_score_flip_frac": policy_exit_score_flip_frac,

            # Diff 5: Maker → Market 혼합 실행
            "exec_mode": exec_mode,
            "pmaker_entry": pmaker_entry,
            "fee_roundtrip_fee_mix": fee_roundtrip_fee_mix,
            "fee_roundtrip_fee_taker": fee_roundtrip_fee_taker,
            "fee_roundtrip_fee_maker": fee_roundtrip_fee_maker,
            "execution_cost": execution_cost,
            "expected_spread_cost": expected_spread_cost,
            "slippage_dyn": slippage_dyn,
            "spread_pct_used": spread_pct_used,
            "p_maker": pmaker_entry,

            # Diff 6: Maker fill delay (if present)
            "pmaker_entry_delay_sec": pmaker_entry_delay_sec,

            # Diff 7: delay penalty + horizon shift
            "pmaker_entry_delay_penalty_r": pmaker_entry_delay_penalty_r,
            "policy_entry_shift_steps": policy_entry_shift_steps,

            # extra
            "pmaker": self.pmaker.status_dict() if getattr(self, "pmaker", None) is not None else None,
            "details": (decision.get("details", []) if (decision and bool(getattr(config, "DASHBOARD_INCLUDE_DETAILS", False))) else []),
            # Rebalance diagnostics (last joint-run decisions)
            "rebalance_decision": self._rebalance_last_decision.get(sym) if hasattr(self, "_rebalance_last_decision") else None,
            "rebalance_weight": (self._last_portfolio_report.get("weights", {}).get(sym) if (hasattr(self, "_last_portfolio_report") and self._last_portfolio_report) else None),
            # Detailed rebalance diagnostics
            "rebalance_delta_ev": (self._last_rebalance_plan.get(sym, {}).get("ev_adj") if (hasattr(self, "_last_rebalance_plan") and self._last_rebalance_plan) else None),
            "rebalance_exec_cost": (self._last_rebalance_plan.get(sym, {}).get("exec_cost") if (hasattr(self, "_last_rebalance_plan") and self._last_rebalance_plan) else None),
            "rebalance_target_leverage": (self._last_rebalance_plan.get(sym, {}).get("target_leverage") if (hasattr(self, "_last_rebalance_plan") and self._last_rebalance_plan) else None),
            "rebalance_allow_trade": (self._last_rebalance_plan.get(sym, {}).get("allow_trade") if (hasattr(self, "_last_rebalance_plan") and self._last_rebalance_plan) else None),
        }

    def _snapshot_inputs(
        self, sym: str
    ) -> tuple[Any, list[float], int, Optional[float], Optional[float], Optional[float]]:
        mkt = self.market.get(sym) or {}
        price = mkt.get("price")
        closes = list(self.ohlcv_buffer.get(sym) or [])
        candles = len(closes)

        # ✅ FIX: price가 None이지만 closes가 있으면 마지막 close 사용
        if price is None and closes:
            price = closes[-1]
            # 디버그: 폴백 사용
            if not hasattr(self, "_price_fallback_debug"):
                self._price_fallback_debug = set()
            if sym not in self._price_fallback_debug:
                print(f"[PRICE_FALLBACK] {sym} using last close={price:.2f} (ticker not yet available)", flush=True)
                self._price_fallback_debug.add(sym)
        candles = len(closes)

        best_bid = None
        best_ask = None
        try:
            if mkt.get("bid") is not None:
                best_bid = float(mkt.get("bid"))
        except Exception:
            best_bid = None
        try:
            if mkt.get("ask") is not None:
                best_ask = float(mkt.get("ask"))
        except Exception:
            best_ask = None

        ob = self.orderbook.get(sym) or {}
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if bids and asks:
            try:
                ob_bid = float(bids[0][0])
                ob_ask = float(asks[0][0])

                ref = None
                try:
                    if price is not None:
                        ref = float(price)
                    elif best_bid is not None and best_ask is not None:
                        ref = 0.5 * (float(best_bid) + float(best_ask))
                except Exception:
                    ref = None

                ok = True
                if ob_bid <= 0 or ob_ask <= 0 or ob_ask < ob_bid:
                    ok = False
                if ok and ref is not None and ref > 0:
                    # Orderbook sanity: reject wildly off quotes (prevents bad fills when API returns broken bids/asks).
                    lo = float(ref) * 0.5
                    hi = float(ref) * 1.5
                    if not (lo <= ob_bid <= hi and lo <= ob_ask <= hi):
                        ok = False

                if ok:
                    best_bid = ob_bid
                    best_ask = ob_ask
            except Exception:
                pass

        spread_pct = None
        # Final quote sanity vs last price (some venues occasionally return broken bid/ask like bid<<last<<ask).
        try:
            ref_price = float(price) if price is not None else None
        except Exception:
            ref_price = None
        if ref_price is not None and ref_price > 0:
            lo = ref_price * 0.5
            hi = ref_price * 1.5
            if best_bid is not None and not (lo <= float(best_bid) <= hi):
                best_bid = None
            if best_ask is not None and not (lo <= float(best_ask) <= hi):
                best_ask = None
            # If one side is missing, fall back to last price so paper fills don't blow up.
            if best_bid is None and best_ask is None:
                best_bid = ref_price
                best_ask = ref_price
            elif best_bid is None and best_ask is not None:
                best_bid = min(ref_price, float(best_ask))
            elif best_ask is None and best_bid is not None:
                best_ask = max(ref_price, float(best_bid))
            if best_bid is not None and best_ask is not None and float(best_ask) < float(best_bid):
                best_bid = ref_price
                best_ask = ref_price

        if best_bid is not None and best_ask is not None:
            mid = 0.5 * (best_bid + best_ask)
            if mid > 0:
                spread_pct = (best_ask - best_bid) / mid

        return price, closes, candles, best_bid, best_ask, spread_pct

    def _build_decide_ctx(
        self,
        *,
        sym: str,
        price: Any,
        closes: list[float],
        candles: int,
        best_bid: Optional[float],
        best_ask: Optional[float],
        spread_pct: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        try:
            min_candles = int(_env_int("DECISION_MIN_CANDLES", 20))
        except Exception:
            min_candles = 20

        # 디버그: ctx 구축 실패 원인 추적
        if not hasattr(self, "_ctx_debug_count"):
            self._ctx_debug_count = {}
        self._ctx_debug_count[sym] = self._ctx_debug_count.get(sym, 0) + 1
        debug_this = (self._ctx_debug_count[sym] <= 5)  # 심볼당 처음 5회만

        if price is None or candles < min_candles:
            if debug_this:
                print(f"[CTX_DEBUG] {sym} SKIP: price={price} candles={candles} min_candles={min_candles}", flush=True)
            return None
        try:
            px = float(price)
        except Exception:
            if debug_this:
                print(f"[CTX_DEBUG] {sym} SKIP: price not float-able: {price}", flush=True)
            return None
        if px <= 0:
            if debug_this:
                print(f"[CTX_DEBUG] {sym} SKIP: px <= 0: {px}", flush=True)
            return None

        regime = self._infer_regime(closes)
        ofi = self._compute_ofi_score(sym)
        liq = self._liquidity_score(sym)
        
        # ✅ FIX: mu_base와 sigma 계산 추가
        mu_sim, sigma_sim = self._compute_returns_and_vol(closes)

        # Fallback for short histories (useful for dry-run profiling when DECISION_MIN_CANDLES is lowered)
        if mu_sim is None or sigma_sim is None:
            try:
                if closes and len(closes) >= 2:
                    arr = np.asarray(closes, dtype=np.float64)
                    rets = np.diff(np.log(np.maximum(arr, 1e-12)))
                    if rets.size >= 1:
                        mu_sim = float(np.mean(rets))
                        sigma_sim = float(np.std(rets))
                if mu_sim is None:
                    mu_sim = 0.0
                if sigma_sim is None or (not math.isfinite(float(sigma_sim))) or float(sigma_sim) <= 0:
                    sigma_sim = 0.01
            except Exception:
                mu_sim = 0.0
                sigma_sim = 0.01
        
        ctx = {
            "symbol": sym,
            "price": px,
            "closes": closes,
            "regime": regime,
            # ✅ FIX: mu_sim과 sigma_sim 추가
            "mu_sim": mu_sim,
            "sigma_sim": sigma_sim,
            "mu_base": mu_sim,  # backward compat
            "sigma": sigma_sim,  # backward compat
            # Back-compat: engines expect `ofi_score` (legacy ctx used `ofi`).
            "ofi_score": float(ofi),
            "ofi": float(ofi),
            "liquidity_score": float(liq),
            "spread_pct": float(spread_pct) if spread_pct is not None else 0.0,
            # Back-compat: some code expects `bid`/`ask`.
            "bid": best_bid,
            "ask": best_ask,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bar_seconds": 60.0,
            "leverage": float(self.leverage),
            "max_leverage": float(self.max_leverage),
            "n_paths": int(self.mc_n_paths_live),
            "tail_mode": str(self.mc_tail_mode),
            "student_t_df": float(self.mc_student_t_df),
        }
        # PMaker: provide survival model for mu_alpha boost (paper-mode training updates sym_fill_mean).
        try:
            pm = getattr(self, "pmaker", None)
            surv = getattr(pm, "surv", None) if pm is not None else None
            if pm is not None and bool(getattr(pm, "enabled", False)) and surv is not None:
                ctx["pmaker_surv"] = surv
                ctx["pmaker_timeout_ms"] = int(os.environ.get("MAKER_TIMEOUT_MS", str(getattr(config, "MAKER_TIMEOUT_MS", 1500))))
            ctx["pmaker_adverse_move"] = float(self._pmaker_paper_adverse_ema.get(sym, 0.0) or 0.0)
        except Exception:
            pass
        
        # 디버그: ctx 구축 성공
        if debug_this:
            print(f"[CTX_DEBUG] {sym} OK: price={px:.2f} candles={candles} mu_sim={mu_sim:.6f} sigma_sim={sigma_sim:.6f} regime={regime}", flush=True)
        
        return ctx

    def _rows_snapshot(self, ts_ms: int, *, apply_trades: bool = False) -> list[Dict[str, Any]]:
        # Batch all valid contexts and call EngineHub.decide_batch for parallel evaluation.
        rows: list[Dict[str, Any]] = []

        ctx_list: list[Dict[str, Any]] = []
        ctx_syms: list[str] = []
        ctx_meta: dict[str, dict] = {}

        for sym in self.symbols:
            price, closes, candles, best_bid, best_ask, spread_pct = self._snapshot_inputs(sym)

            mark_price = self._mark_price(best_bid, best_ask, float(price) if price is not None else None)
            # mark existing positions for freshness
            self._paper_mark_position(sym, mark_price, ts_ms)

            ctx = self._build_decide_ctx(
                sym=sym,
                price=price,
                closes=closes,
                candles=candles,
                best_bid=best_bid,
                best_ask=best_ask,
                spread_pct=spread_pct,
            )
            if ctx is None:
                # no ctx -> produce row with no decision
                rows.append(self._row(sym, price, ts_ms, None, candles, ctx=None))
                continue

            # PMaker probe tick updates internal state used by evaluation
            try:
                self._pmaker_paper_probe_tick(sym=sym, ts_ms=ts_ms, ctx=ctx)
            except Exception:
                pass

            ctx_list.append(ctx)
            ctx_syms.append(sym)
            ctx_meta[sym] = {"best_bid": best_bid, "best_ask": best_ask, "mark_price": mark_price, "candles": candles}

        # If no contexts, return early
        if not ctx_list:
            return rows

        # Synchronous batch call (EngineHub.decide_batch is sync); wrap in try/except
        try:
            decisions = self.hub.decide_batch(ctx_list)
        except Exception as e:
            self._log_err(f"[ERR] decide_batch: {e}")
            # fallback to sequential decisions
            decisions = []
            for ctx in ctx_list:
                try:
                    d = self.hub.decide(ctx)
                except Exception as e2:
                    self._log_err(f"[ERR] decide fallback: {e2}")
                    d = None
                decisions.append(d)

        # Map decisions back to symbols and optionally apply trades
        for i, sym in enumerate(ctx_syms):
            decision = decisions[i] if i < len(decisions) else None
            meta = ctx_meta.get(sym, {})
            best_bid = meta.get("best_bid")
            best_ask = meta.get("best_ask")
            mark_price = meta.get("mark_price")
            candles = int(meta.get("candles") or 0)

            if apply_trades and decision:
                details = decision.get("details", []) if isinstance(decision, dict) else []
                best_detail = None
                if isinstance(details, list) and details:
                    cand = [d for d in details if isinstance(d, dict)]
                    if cand:
                        best_detail = max(cand, key=lambda d: float(d.get("ev", 0.0) or 0.0))
                decision_reason = str(decision.get("reason", "") or "") if isinstance(decision, dict) else ""
                desired_action = str(decision.get("action", "WAIT") if isinstance(decision, dict) else "WAIT")
                try:
                    self._paper_trade_step(
                        sym=sym,
                        desired_action=desired_action,
                        ts_ms=ts_ms,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        mark_price=mark_price,
                        detail=best_detail,
                        decision_reason=decision_reason,
                    )
                except Exception:
                    pass

            rows.append(self._row(sym, (self.market.get(sym) or {}).get("price"), ts_ms, decision, candles, ctx=ctx_list[i]))

        return rows

    def _rows_snapshot_cached(self, ts_ms: int) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        for sym in self.symbols:
            price, closes, candles, _best_bid, _best_ask, _spread_pct = self._snapshot_inputs(sym)
            mark_price = self._mark_price(_best_bid, _best_ask, float(price) if price is not None else None)
            self._paper_mark_position(sym, mark_price, ts_ms)
            cached = self._decision_cache.get(sym) or {}
            decision = cached.get("decision")
            regime = self._infer_regime(closes) if candles >= 20 else "-"
            rows.append(self._row(sym, price, ts_ms, decision, candles, ctx={"regime": regime}))
        return rows

    async def decision_worker_loop(self):
        """
        Continuously refresh per-symbol decisions in a background task.
        Uses asyncio.to_thread() so HTTP/WebSocket tasks stay responsive even when MC is heavy.
        """
        while True:
            cycle_t0 = time.time()
            base_syms = list(self.symbols)
            if not base_syms:
                await asyncio.sleep(0.25)
                continue

            start = int(self._decision_rr_index) % len(base_syms)
            rotated = base_syms[start:] + base_syms[:start]
            self._decision_rr_index = (start + 1) % len(base_syms)

            # Prioritize open positions for faster exit/management responsiveness.
            open_set: set[str] = set()
            for sym, pos in (self.positions or {}).items():
                try:
                    size = float(pos.get("size", pos.get("quantity", pos.get("qty", 0.0))) or 0.0)
                except Exception:
                    size = 0.0
                if size != 0.0:
                    open_set.add(sym)
            symbols = [s for s in rotated if s in open_set] + [s for s in rotated if s not in open_set]

            # Build a batch of contexts to evaluate in one call to EngineHub.decide_batch
            ctx_list = []
            ctx_syms: list[str] = []
            ctx_meta: dict[str, dict] = {}
            try:
                min_gap_ms_def = int(max(0.0, float(self.decision_eval_min_interval_sec)) * 1000.0)
            except Exception:
                min_gap_ms_def = 0

            for sym in symbols:
                ts_ms = now_ms()
                last_ts = (self._decision_cache.get(sym) or {}).get("ts")
                if last_ts and min_gap_ms_def > 0:
                    try:
                        if int(ts_ms) - int(last_ts) < int(min_gap_ms_def):
                            continue
                    except Exception:
                        pass

                price, closes, candles, best_bid, best_ask, spread_pct = self._snapshot_inputs(sym)
                mark_price = self._mark_price(best_bid, best_ask, float(price) if price is not None else None)
                ctx = self._build_decide_ctx(
                    sym=sym,
                    price=price,
                    closes=closes,
                    candles=candles,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    spread_pct=spread_pct,
                )
                if ctx is None:
                    continue

                # PMaker paper training (simulated maker fills) — run now so batch ctx contains pmaker state
                try:
                    self._pmaker_paper_probe_tick(sym=sym, ts_ms=ts_ms, ctx=ctx)
                except Exception:
                    pass

                ctx_list.append(ctx)
                ctx_syms.append(sym)
                ctx_meta[sym] = {"ts_ms": int(ts_ms), "best_bid": best_bid, "best_ask": best_ask, "mark_price": mark_price}

            if not ctx_list:
                await asyncio.sleep(0)
                self._decide_cycle_ms = int((time.time() - cycle_t0) * 1000)
                sleep_sec = float(self.decision_worker_sleep_sec) if self.decision_worker_sleep_sec else 0.0
                if sleep_sec > 0:
                    await asyncio.sleep(sleep_sec)
                else:
                    await asyncio.sleep(0)
                continue

            # Evaluate batch in a thread to avoid blocking the event loop
            t0 = time.time()
            try:
                decisions = await asyncio.to_thread(self.hub.decide_batch, ctx_list)
            except Exception as e:
                self._log_err(f"[ERR] decide_batch: {e}")
                # Fallback: evaluate sequentially
                decisions = []
                for ctx in ctx_list:
                    try:
                        d = await asyncio.to_thread(self.hub.decide, ctx)
                    except Exception:
                        d = None
                    decisions.append(d)
            decide_ms = int((time.time() - t0) * 1000)

            # Build decision maps for portfolio-aware gating
            decision_map: Dict[str, Dict[str, Any]] = {}
            metrics_map: Dict[str, Dict[str, Any]] = {}
            ai_scores: Dict[str, float] = {}

            for i, sym in enumerate(ctx_syms):
                decision = decisions[i] if i < len(decisions) else None
                decision_map[sym] = decision
                metrics = self._decision_metrics(decision)
                metrics_map[sym] = metrics
                # Debug: log per-symbol metrics that look suspicious
                try:
                    ev = float(metrics.get("ev", 0.0) or 0.0)
                    kelly = float(metrics.get("kelly", 0.0) or 0.0)
                    exec_cost = float(metrics.get("exec_cost", 0.0) or 0.0)
                    if ev == 0.0:
                        self._log(f"[DECIDE] {sym} ev=0 kelly={kelly:.6f} exec_cost={exec_cost:.6f} decision_mc_reason={decision.get('reason') if isinstance(decision, dict) else None}")
                except Exception:
                    pass
                if math.isfinite(metrics.get("ev", 0.0)):
                    ai_scores[sym] = float(metrics.get("ev", 0.0))

            joint_report = await self._maybe_portfolio_joint(list(ai_scores.keys()), ai_scores)
            rebalance_plan = self._build_rebalance_plan(
                decision_map=decision_map,
                metrics_map=metrics_map,
                joint_report=joint_report,
            )

            # Map back decisions to symbols and apply
            for i, sym in enumerate(ctx_syms):
                ts_ms = int(ctx_meta.get(sym, {}).get("ts_ms") or now_ms())
                best_bid = ctx_meta.get(sym, {}).get("best_bid")
                best_ask = ctx_meta.get(sym, {}).get("best_ask")
                mark_price = ctx_meta.get(sym, {}).get("mark_price")
                decision = decision_map.get(sym)

                self._decision_cache[sym] = {"decision": decision, "ctx": ctx_list[i], "ts": int(ts_ms), "decide_ms": int(decide_ms)}

                if self.paper_trading_enabled and decision:
                    details = decision.get("details", []) if isinstance(decision, dict) else []
                    best_detail = None
                    if isinstance(details, list) and details:
                        cand = [d for d in details if isinstance(d, dict)]
                        if cand:
                            best_detail = max(cand, key=lambda d: float(d.get("ev", 0.0) or 0.0))
                    plan = rebalance_plan.get(sym, {})
                    desired_action = str(decision.get("action", "WAIT") if isinstance(decision, dict) else "WAIT")
                    if not plan.get("allow_trade", True):
                        desired_action = "WAIT"
                    if best_detail is not None:
                        best_detail = dict(best_detail)
                    if plan.get("target_leverage") is not None:
                        best_detail = best_detail or {}
                        best_detail["optimal_leverage"] = float(plan.get("target_leverage"))
                    if plan.get("target_cap_frac") is not None:
                        best_detail = best_detail or {}
                        best_detail["optimal_size"] = float(plan.get("target_cap_frac"))
                    decision_reason = str(decision.get("reason", "") or "") if isinstance(decision, dict) else ""
                    self._paper_trade_step(
                        sym=sym,
                        desired_action=desired_action,
                        ts_ms=ts_ms,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        mark_price=mark_price,
                        detail=best_detail,
                        decision_reason=decision_reason,
                    )

                await asyncio.sleep(0)

            self._decide_cycle_ms = int((time.time() - cycle_t0) * 1000)
            sleep_sec = float(self.decision_worker_sleep_sec) if self.decision_worker_sleep_sec else 0.0
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)
            else:
                await asyncio.sleep(0)

    async def decision_loop(self):
        while True:
            t0 = time.time()
            ts = now_ms()
            rows = self._rows_snapshot_cached(ts)
            self._last_rows = rows
            self._loop_ms = int((time.time() - t0) * 1000)

            # mark-to-market equity history for charts
            try:
                unreal = sum(float((p or {}).get("unrealized_pnl", 0.0) or 0.0) for p in (self.positions or {}).values())
                equity = float(self.balance) + float(unreal)
                self._equity_history.append({"time": int(ts), "equity": float(equity)})
            except Exception:
                pass

            if self.dashboard is not None:
                try:
                    await self.dashboard.broadcast(rows)
                except Exception as e:
                    self._log_err(f"[ERR] dashboard broadcast: {e}")

            self._persist_state(force=False)
            elapsed = time.time() - t0
            sleep_left = max(0.0, float(self.decision_refresh_sec) - float(elapsed))
            await asyncio.sleep(sleep_left)

    def reset_tape_and_equity(self) -> None:
        """
        Resets the trade tape, balance (to 10k), positions, and equity history.
        Used by the dashboard 'Clear Trade Tape' button to provide a fresh start.
        """
        import json
        
        # 1. Reset Trade Tape
        self.trade_tape.clear()
        try:
            self.state_files["trade"].write_text(json.dumps(list(self.trade_tape)), encoding="utf-8")
        except Exception as e:
            self._log_err(f"[RESET_ERR] trade: {e}")

        # 2. Reset Balance
        self.balance = 10000.0
        try:
            # Balance is saved as raw string in _persist_state
            self.state_files["balance"].write_text(str(self.balance), encoding="utf-8")
        except Exception as e:
            self._log_err(f"[RESET_ERR] balance: {e}")

        # 3. Reset Positions (New)
        self.positions.clear()
        try:
            self.state_files["positions"].write_text(json.dumps({}), encoding="utf-8")
        except Exception as e:
            self._log_err(f"[RESET_ERR] positions: {e}")

        # 4. Reset Equity History
        self._equity_history.clear()
        self._equity_history.append(10000.0)
        try:
            # Equity history is not currently persisted in _persist_state, but we write it here just in case
            self.state_files["equity"].write_text(json.dumps(list(self._equity_history)), encoding="utf-8")
        except Exception as e:
            pass
        
        # 5. Clear execution stats
        self.exec_stats.clear()
        
        self._log("[RESET] Trade tape, balance, positions, and equity history reset to initial state (10k).")


async def build_exchange() -> ccxt.Exchange:
    _load_env_file(str(config.BASE_DIR / "state" / "bybit.env"))
    # Only load the example template when the real env file is missing.
    # (The example often enables BYBIT_TESTNET=1 which would otherwise hijack prices.)
    if not (config.BASE_DIR / "state" / "bybit.env").exists():
        _load_env_file(str(config.BASE_DIR / "state" / "bybit.env.example"))

    # Public-data mode (paper trading / dashboards) should not require auth.
    # Invalid keys can break `load_markets()` on Bybit because it may hit private endpoints.
    ex_cfg: Dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": int(config.CCXT_TIMEOUT_MS),
    }
    if bool(getattr(config, "ENABLE_LIVE_ORDERS", False)):
        ex_cfg.update(
            {
                "apiKey": getattr(config, "API_KEY", ""),
                "secret": getattr(config, "API_SECRET", ""),
            }
        )
    exchange = ccxt.bybit(ex_cfg)
    if str(os.environ.get("BYBIT_TESTNET", "0")).strip().lower() in ("1", "true", "yes"):
        exchange.set_sandbox_mode(True)
    return exchange


async def build_data_exchange() -> ccxt.Exchange:
    """Build a Bybit CCXT client dedicated to market-data.

    This is intentionally unauthenticated so paper/dashboard runs don't break on invalid keys.
    Sandbox selection is controlled via DATA_BYBIT_TESTNET (separate from BYBIT_TESTNET).
    """

    _load_env_file(str(config.BASE_DIR / "state" / "bybit.env"))
    if not (config.BASE_DIR / "state" / "bybit.env").exists():
        _load_env_file(str(config.BASE_DIR / "state" / "bybit.env.example"))

    ex_cfg: Dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": int(config.CCXT_TIMEOUT_MS),
    }
    exchange = ccxt.bybit(ex_cfg)

    if str(os.environ.get("DATA_BYBIT_TESTNET", "0")).strip().lower() in ("1", "true", "yes"):
        exchange.set_sandbox_mode(True)
    return exchange

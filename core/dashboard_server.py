import json
import asyncio
import os
import time
from aiohttp import web
from config import PORT, DASHBOARD_FILE, DASHBOARD_HISTORY_MAX, DASHBOARD_TRADE_TAPE_MAX, EXEC_MODE, MAKER_TIMEOUT_MS, MAKER_RETRIES, MAKER_POLL_MS, MAX_NOTIONAL_EXPOSURE
from utils.helpers import now_ms, _sanitize_for_json

def _fallback_rows(orch, ts: int):
    rows = []
    symbols = list(getattr(orch, "symbols", []) or [])
    market = getattr(orch, "market", {}) or {}
    for sym in symbols:
        m = market.get(sym) or {}
        rows.append(
            {
                "symbol": sym,
                "price": m.get("price"),
                "bid": m.get("bid"),
                "ask": m.get("ask"),
                "ts": m.get("ts") or ts,
                "status": "WAIT",
                "regime": "-",
                "ev": 0.0,
                "conf": 0.0,
                "meta": {"boot": True},
            }
        )
    return rows

def _exec_stats_snapshot(orch):
    # Logic from LiveOrchestrator._exec_stats_snapshot
    res = {}
    for sym, st in orch.exec_stats.items():
        res[sym] = {
            "maker_fills": st.get("maker_limit_filled", 0),
            "taker_fills": st.get("taker_market_filled", 0),
            # ... more stats ...
        }
    return res

def _compute_portfolio(orch):
    """Compute portfolio data from orchestrator positions."""
    try:
        ts = now_ms()
        wallet = float(getattr(orch, 'balance', 10000.0))
        equity = wallet
        unreal = 0.0
        pos_list = []
        
        # Get positions
        positions = getattr(orch, 'positions', {})
        if not isinstance(positions, dict):
            positions = {}
        
        for sym, pos in positions.items():
            try:
                pos_size = float(pos.get("size", pos.get("quantity", 0.0)) or 0.0)
                if pos_size != 0:
                    unreal += float(pos.get("unrealized_pnl", 0.0))
                    
                    # Enrich position data
                    enriched_pos = dict(pos)
                    # Live close notes (e.g. low-liquidity close order waiting)
                    try:
                        notes = getattr(orch, "_live_close_note_by_sym", None) or {}
                        if isinstance(notes, dict):
                            rec = notes.get(sym)
                            if isinstance(rec, dict):
                                until = rec.get("until_ms")
                                try:
                                    until_i = int(until) if until is not None else 0
                                except Exception:
                                    until_i = 0
                                if until_i > int(ts):
                                    enriched_pos["close_pending"] = True
                                    enriched_pos["close_pending_note"] = rec.get("note")
                                    enriched_pos["close_pending_price"] = rec.get("price")
                                    enriched_pos["close_pending_qty"] = rec.get("qty")
                                else:
                                    try:
                                        notes.pop(sym, None)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    
                    # Group info - safely access _group_info
                    if hasattr(orch, '_group_info') and isinstance(orch._group_info, dict):
                        enriched_pos["group"] = pos.get("entry_group") or orch._group_info.get(sym, {}).get("group", "OTHER")
                    else:
                        enriched_pos["group"] = pos.get("entry_group", "OTHER")
                    
                    # Rank info
                    entry_rank = pos.get("entry_rank")
                    current_rank = orch._get_current_rank(sym) if hasattr(orch, '_get_current_rank') else None
                    enriched_pos["rank"] = entry_rank if entry_rank is not None else current_rank
                    enriched_pos["entry_rank"] = entry_rank
                    enriched_pos["entry_order"] = pos.get("entry_order", 0)
                    enriched_pos["current_rank"] = current_rank
                    enriched_pos["entry_group"] = pos.get("entry_group")
                    enriched_pos["optimal_horizon_sec"] = pos.get("optimal_horizon_sec")
                    
                    # entry_t_star with fallback to current T*
                    entry_t_star = pos.get("entry_t_star")
                    current_t_star = orch._symbol_t_star.get(sym, 3600.0) if hasattr(orch, '_symbol_t_star') else 3600.0
                    
                    if entry_t_star is None:
                        entry_t_star = current_t_star
                        
                    enriched_pos["entry_t_star"] = entry_t_star
                    enriched_pos["current_optimal_horizon_sec"] = current_t_star
                    enriched_pos["rebalance_target_cap_frac"] = getattr(orch, "_rebalance_targets", {}).get(sym)
                    
                    pos_list.append(enriched_pos)
            except Exception as e:
                print(f"[ERR] _compute_portfolio: Error processing position {sym}: {e}")
                continue
                
        live_mode = bool(getattr(orch, "enable_orders", False)) and (not bool(getattr(orch, "paper_trading_enabled", False)))
        if live_mode and getattr(orch, "_live_equity", None) is not None:
            try:
                equity = float(getattr(orch, "_live_equity"))
            except Exception:
                equity = wallet + unreal
        else:
            equity = wallet + unreal

        total_notional = orch._total_open_notional() if hasattr(orch, "_total_open_notional") else 0.0
        # In cross-margin live mode, show utilization against Total Equity (wallet + uPnL),
        # because equity is the true collateral base.
        util_equity = float(total_notional) / max(float(equity), 1.0)
        util_wallet = float(total_notional) / max(float(wallet), 1.0)
        util = util_equity if live_mode else util_wallet
        return equity, unreal, util, pos_list
    except Exception as e:
        print(f"[ERR] _compute_portfolio: {e}")
        return 10000.0, 0.0, 0.0, []

def _compute_eval_metrics(orch):
    # Compute eval metrics from equity history
    try:
        if len(orch._equity_history) >= 2:
            eq_list = list(orch._equity_history)
            initial = eq_list[0]
            current = eq_list[-1]
            if initial > 0:
                total_return = (current - initial) / initial
            else:
                total_return = 0.0
            return {"total_return": total_return}
        return {"total_return": 0.0}
    except Exception:
        return {"total_return": 0.0}

def _build_payload(orch, rows, include_history, include_trade_tape, *, include_logs: bool = True):
    equity, unreal, util, pos_list = _compute_portfolio(orch)
    eval_metrics = _compute_eval_metrics(orch)
    ts = now_ms()

    feed_connected = (orch.data._last_feed_ok_ms > 0) and (ts - orch.data._last_feed_ok_ms < 10_000)
    feed = {
        "connected": bool(feed_connected),
        "last_msg_age": (ts - orch.data._last_feed_ok_ms) if orch.data._last_feed_ok_ms else None
    }

    history = []
    if include_history:
        history = list(orch._equity_history)[-int(DASHBOARD_HISTORY_MAX):]

    trade_tape = []
    if include_trade_tape:
        trade_tape = list(orch.trade_tape)[-int(DASHBOARD_TRADE_TAPE_MAX):]

    live_mode = bool(getattr(orch, "enable_orders", False)) and (not bool(getattr(orch, "paper_trading_enabled", False)))
    util_cap = None
    try:
        if live_mode and hasattr(orch, "max_total_leverage"):
            util_cap = float(getattr(orch, "max_total_leverage"))
        else:
            util_cap = float(MAX_NOTIONAL_EXPOSURE) if orch.exposure_cap_enabled else None
    except Exception:
        util_cap = float(MAX_NOTIONAL_EXPOSURE) if orch.exposure_cap_enabled else None

    payload = {
        "type": "full_update",
        "server_time": ts,
        "kill_switch": bool(getattr(orch, "safety_mode", False)),
        "engine": {
            "modules_ok": True,
            "run_id": getattr(orch, "run_id", None),
            "start_ms": getattr(orch, "start_ms", None),
            "ws_clients": len(orch.clients),
            "loop_ms": getattr(orch, "_loop_ms", None),
            "decide_cycle_ms": getattr(orch, "_decide_cycle_ms", None),
            "mc_ready": bool(getattr(orch, "_mc_ready", False)),
            "enable_orders": bool(orch.enable_orders),
            "paper_trading": bool(getattr(orch, "paper_trading_enabled", False)),
            "record_mode": str(getattr(orch, "_record_mode", "live" if bool(getattr(orch, "enable_orders", False)) else "paper")),
            "live_sync_age_ms": (ts - int(getattr(orch, "_last_live_sync_ms", 0) or 0)) if getattr(orch, "_last_live_sync_ms", 0) else None,
            "live_sync_err": getattr(orch, "_last_live_sync_err", None),
            "live_wallet_balance": getattr(orch, "_live_wallet_balance", None),
            "live_equity": getattr(orch, "_live_equity", None),
            "live_free_balance": getattr(orch, "_live_free_balance", None),
            "live_total_initial_margin": getattr(orch, "_live_total_initial_margin", None),
            "live_total_maintenance_margin": getattr(orch, "_live_total_maintenance_margin", None),
            "live_margin_ratio": getattr(getattr(orch, "risk", None), "get_margin_ratio", lambda: None)(),
            "decision_refresh_sec": getattr(orch, "decision_refresh_sec", None),
            "decision_eval_min_interval_sec": getattr(orch, "decision_eval_min_interval_sec", None),
            "mc_n_paths_live": getattr(orch, "mc_n_paths_live", None),
            "mc_n_paths_exit": getattr(orch, "mc_n_paths_exit", None),
            "exec_mode": str(os.environ.get("EXEC_MODE", EXEC_MODE)).strip().lower(),
            "maker_timeout_ms": int(os.environ.get("MAKER_TIMEOUT_MS", str(MAKER_TIMEOUT_MS))),
            "maker_retries": int(os.environ.get("MAKER_RETRIES", str(MAKER_RETRIES))),
            "maker_poll_ms": int(os.environ.get("MAKER_POLL_MS", str(MAKER_POLL_MS))),
            "exec_stats": _exec_stats_snapshot(orch),
            "pmaker": orch.pmaker.status_dict(),
        },
        "feed": feed,
        "market": rows,
        "portfolio": {
            "balance": float(orch.balance),
            "equity": float(equity),
            "unrealized_pnl": float(unreal),
            "utilization": util,
            "utilization_cap": util_cap,
            "margin_used": sum(float((p or {}).get("margin") or 0.0) for p in (pos_list or [])),
            "margin_max_frac": getattr(orch, "live_max_margin_frac", None),
            "positions": pos_list,
            "history": history,
        },
        "eval_metrics": eval_metrics,
        "trade_tape": trade_tape,
    }
    logs_data = list(orch.logs)[-100:] if include_logs and hasattr(orch, 'logs') else []
    payload["logs"] = logs_data
    payload["alerts"] = list(getattr(orch, "anomalies", [])) if hasattr(orch, "anomalies") else []

    return payload

# HTTP API Route Handlers
async def handle_api_status(request):
    """API endpoint for /api/status"""
    try:
        orch = request.app['orchestrator']
        rows = orch._last_rows if hasattr(orch, '_last_rows') and orch._last_rows else _fallback_rows(orch, now_ms())
        payload = _build_payload(orch, rows, include_history=True, include_trade_tape=True, include_logs=False)
        return web.json_response(payload)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_api_positions(request):
    """API endpoint for /api/positions"""
    try:
        orch = request.app['orchestrator']
        equity, unreal, util, pos_list = _compute_portfolio(orch)
        return web.json_response({
            "positions": pos_list,
            "equity": equity,
            "unrealized_pnl": unreal,
            "utilization": util
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_api_score_debug(request):
    """API endpoint for /api/score_debug?symbol=..."""
    try:
        orch = request.app["orchestrator"]
        sym = request.query.get("symbol") or request.query.get("sym") or ""
        sym = str(sym).strip()
        if not sym:
            return web.json_response({"error": "missing symbol"}, status=400)
        if not hasattr(orch, "score_debug_for_symbol"):
            return web.json_response({"error": "orchestrator lacks score_debug_for_symbol"}, status=500)
        out = orch.score_debug_for_symbol(sym)
        return web.json_response(_sanitize_for_json(out))
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_dashboard(request):
    """Serve dashboard HTML"""
    try:
        dashboard_path = DASHBOARD_FILE
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            html = f.read()
        return web.Response(text=html, content_type='text/html')
    except Exception as e:
        return web.Response(text=f"Dashboard file not found: {e}", status=404)

def serve_dashboard(orchestrator):
    return web.FileResponse(str(DASHBOARD_FILE))

async def index_handler(request):
    return web.FileResponse(str(DASHBOARD_FILE))

async def debug_payload_handler(request):
    orch = request.app["orchestrator"]
    ts = now_ms()
    rows = getattr(orch, "_last_rows", None)
    if rows is None:
        rows = _fallback_rows(orch, ts)
    payload = _build_payload(orch, rows, include_history=False, include_trade_tape=False, include_logs=True)
    payload = _sanitize_for_json(payload)
    return web.json_response(payload, dumps=lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":")))

async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    orch = request.app["orchestrator"]
    orch.clients.add(ws)
    
    try:
        ts0 = now_ms()
        snap_rows0 = getattr(orch, "_last_rows", None)
        if snap_rows0 is None:
            snap_rows0 = _fallback_rows(orch, ts0)
        snap0 = _build_payload(orch, snap_rows0, include_history=False, include_trade_tape=False, include_logs=True)
        snap0 = _sanitize_for_json(snap0)
        await ws.send_str(json.dumps(snap0, ensure_ascii=False, separators=(",", ":")))
    except Exception as e:
        print(f"[ERR] ws_handler initial snapshot: {e}")

    await ws.send_str(json.dumps({"type": "init", "msg": "connected"}, ensure_ascii=False, separators=(",", ":")))

    try:
        ts1 = now_ms()
        snap_rows = getattr(orch, "_last_rows", None)
        if snap_rows is None:
            snap_rows = _fallback_rows(orch, ts1)
        snap_payload = _build_payload(orch, snap_rows, include_history=True, include_trade_tape=True, include_logs=True)
        snap_payload = _sanitize_for_json(snap_payload)
        await ws.send_str(json.dumps(snap_payload, ensure_ascii=False, separators=(",", ":")))
    except Exception as e:
        print(f"[ERR] ws_handler full snapshot: {e}")

    async for _ in ws:
        pass

    orch.clients.discard(ws)
    return ws


async def runtime_get_handler(request):
    orch = request.app["orchestrator"]
    cfg = orch.runtime_config() if hasattr(orch, "runtime_config") else {}
    cfg = _sanitize_for_json(cfg)
    return web.json_response(cfg, dumps=lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":")))


async def runtime_post_handler(request):
    orch = request.app["orchestrator"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

    if not isinstance(body, dict):
        return web.json_response({"ok": False, "error": "invalid_payload"}, status=400)

    def _b(x):
        if isinstance(x, bool):
            return x
        if x is None:
            return None
        return str(x).strip().lower() in ("1", "true", "yes", "y", "on")

    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    def _i(x):
        try:
            return int(x)
        except Exception:
            return None
            

    # orchestrator-level toggles
    if "paper_trading_enabled" in body:
        v = _b(body.get("paper_trading_enabled"))
        if v is not None:
            orch.paper_trading_enabled = bool(v) and (not getattr(orch, "enable_orders", False))
    if "enable_orders" in body:
        v = _b(body.get("enable_orders"))
        old_enable_orders = bool(getattr(orch, "enable_orders", False))
        if v is not None and hasattr(orch, "set_enable_orders"):
            new_enable_orders = bool(v)
            orch.set_enable_orders(new_enable_orders)
            # Broadcast mode change event to all connected clients
            if old_enable_orders != new_enable_orders:
                mode = "live" if new_enable_orders else "paper"
                message = "실시간 거래소 주문이 활성화되었습니다" if new_enable_orders else "시뮬레이션 잔고로 전환되었습니다"
                mode_event = {
                    "type": "mode_change",
                    "mode": mode,
                    "message": message
                }
                mode_event = _sanitize_for_json(mode_event)
                mode_json = json.dumps(mode_event, ensure_ascii=False, separators=(",", ":"))
                for ws in list(orch.clients):
                    try:
                        asyncio.create_task(ws.send_str(mode_json))
                    except Exception:
                        orch.clients.discard(ws)
    if "paper_flat_on_wait" in body:
        v = _b(body.get("paper_flat_on_wait"))
        if v is not None:
            orch.paper_flat_on_wait = bool(v)
    if "paper_use_engine_sizing" in body:
        v = _b(body.get("paper_use_engine_sizing"))
        if v is not None:
            orch.paper_use_engine_sizing = bool(v)
    if "paper_engine_size_mult" in body:
        v = _f(body.get("paper_engine_size_mult"))
        if v is not None:
            orch.paper_engine_size_mult = float(max(0.0, v))
    if "paper_engine_size_min_frac" in body:
        v = _f(body.get("paper_engine_size_min_frac"))
        if v is not None:
            orch.paper_engine_size_min_frac = float(max(0.0, min(1.0, v)))
    if "paper_engine_size_max_frac" in body:
        v = _f(body.get("paper_engine_size_max_frac"))
        if v is not None:
            orch.paper_engine_size_max_frac = float(max(0.0, min(1.0, v)))
    if "paper_exit_policy_only" in body:
        v = _b(body.get("paper_exit_policy_only"))
        if v is not None:
            orch.paper_exit_policy_only = bool(v)
    if "paper_size_frac_default" in body:
        v = _f(body.get("paper_size_frac_default"))
        if v is not None:
            orch.paper_size_frac_default = float(max(0.0, min(1.0, v)))
    if "paper_leverage_default" in body:
        v = _f(body.get("paper_leverage_default"))
        if v is not None:
            orch.paper_leverage_default = float(max(0.0, v))
    if "paper_fee_roundtrip" in body:
        v = _f(body.get("paper_fee_roundtrip"))
        if v is not None:
            orch.paper_fee_roundtrip = float(max(0.0, v))
    if "paper_slippage_bps" in body:
        v = _f(body.get("paper_slippage_bps"))
        if v is not None:
            orch.paper_slippage_bps = float(max(0.0, v))
    if "paper_min_hold_sec" in body:
        v = _i(body.get("paper_min_hold_sec"))
        if v is not None:
            orch.paper_min_hold_sec = int(max(0, v))
    if "paper_max_hold_sec" in body:
        v = _i(body.get("paper_max_hold_sec"))
        if v is not None:
            orch.paper_max_hold_sec = int(max(0, v))

    if "decision_refresh_sec" in body:
        v = _f(body.get("decision_refresh_sec"))
        if v is not None:
            orch.decision_refresh_sec = float(max(0.2, v))
    if "decision_eval_min_interval_sec" in body:
        v = _f(body.get("decision_eval_min_interval_sec"))
        if v is not None:
            orch.decision_eval_min_interval_sec = float(max(0.0, v))
    if "decision_worker_sleep_sec" in body:
        v = _f(body.get("decision_worker_sleep_sec"))
        if v is not None:
            orch.decision_worker_sleep_sec = float(max(0.0, v))

    # MC tuning (ctx / instance)
    if "mc_n_paths_live" in body:
        v = _i(body.get("mc_n_paths_live"))
        if v is not None:
            orch.mc_n_paths_live = int(max(200, min(200000, v)))
    if "mc_n_paths_exit" in body:
        v = _i(body.get("mc_n_paths_exit"))
        if v is not None:
            orch.mc_n_paths_exit = int(max(200, min(200000, v)))
            if hasattr(orch, "_apply_mc_runtime_to_engines"):
                orch._apply_mc_runtime_to_engines()
    if "mc_tail_mode" in body:
        v = str(body.get("mc_tail_mode") or "").strip().lower()
        if v in ("gaussian", "student_t", "bootstrap"):
            orch.mc_tail_mode = v
    if "mc_student_t_df" in body:
        v = _f(body.get("mc_student_t_df"))
        if v is not None:
            orch.mc_student_t_df = float(max(2.1, v))

    if "exec_mode" in body:
        v = str(body.get("exec_mode") or "").strip().lower()
        if v in ("market", "maker_then_market"):
            os.environ["EXEC_MODE"] = v

    if "pmaker_enabled" in body:
        v = _b(body.get("pmaker_enabled"))
        if v is not None:
            orch.pmaker.enabled = bool(v)
    
    if "alpha_hit_enabled" in body:
        v = _b(body.get("alpha_hit_enabled"))
        if v is not None:
            for eng in getattr(orch.hub, "engines", []):
                if hasattr(eng, "alpha_hit_enabled"):
                    eng.alpha_hit_enabled = bool(v)

    cfg = orch.runtime_config() if hasattr(orch, "runtime_config") else {}
    cfg = _sanitize_for_json(cfg)
    cfg["ok"] = True
    cfg = _sanitize_for_json(cfg)
    cfg["ok"] = True
    return web.json_response(cfg, dumps=lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":")))

async def api_reset_tape_handler(request):
    orch = request.app["orchestrator"]
    if hasattr(orch, "reset_tape_and_equity"):
        orch.reset_tape_and_equity()
    return web.json_response({"ok": True}, dumps=lambda x: json.dumps(x, ensure_ascii=False))

async def api_liquidate_all_handler(request):
    orch = request.app["orchestrator"]
    orch._log("🛰️ [API] Received liquidate_all request from dashboard.")
    live_mode = bool(getattr(orch, "enable_orders", False)) and (not bool(getattr(orch, "paper_trading_enabled", False)))
    if live_mode and hasattr(orch, "liquidate_all_positions_live"):
        res = orch.liquidate_all_positions_live()
        if asyncio.iscoroutine(res):
            await res
    elif hasattr(orch, "liquidate_all_positions"):
        res = orch.liquidate_all_positions()
        if asyncio.iscoroutine(res):
            await res
    else:
        orch._log_err("❌ [API] LiveOrchestrator has no 'liquidate_all_positions' method!")
    return web.json_response(
        {"ok": True, "paper_trading_enabled": bool(getattr(orch, "paper_trading_enabled", False))},
        dumps=lambda x: json.dumps(x, ensure_ascii=False),
    )

class DashboardServer:
    def __init__(self, orchestrator, *, port: int | None = None):
        self.orchestrator = orchestrator
        self.port = int(port) if port is not None else int(PORT)
        self._last_trade_tape_sig: tuple[int, object, object, object] | None = None
        self._last_logs_sig: tuple[int, object, object, object] | None = None
        self.app = web.Application()
        self.app["orchestrator"] = orchestrator
        self.app.add_routes([
            web.get("/", index_handler),
            web.get("/ws", ws_handler),
            web.get("/debug/payload", debug_payload_handler),
            web.get("/api/runtime", runtime_get_handler),
            web.post("/api/runtime", runtime_post_handler),
            web.post("/api/reset_tape", api_reset_tape_handler),
            web.post("/api/liquidate_all", api_liquidate_all_handler),
            web.get('/api/status', handle_api_status),
            web.get('/api/positions', handle_api_positions),
            web.get('/api/score_debug', handle_api_score_debug),
        ])
        self.runner = None
        self.site = None

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", int(self.port))
        await self.site.start()
        print(f"🚀 Dashboard: http://localhost:{int(self.port)}")

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

    async def broadcast(self, rows):
        if not self.orchestrator.clients:
            return
        include_trade_tape = False
        try:
            tape = getattr(self.orchestrator, "trade_tape", None)
            cur_len = len(tape) if tape is not None else 0
            last = tape[-1] if (tape is not None and cur_len) else None
            last_ts = None
            last_type = None
            last_sym = None
            if isinstance(last, dict):
                last_ts = last.get("ts") or last.get("ts_ms") or last.get("time")
                last_type = last.get("action_type") or last.get("type") or last.get("event")
                last_sym = last.get("sym") or last.get("symbol")
            cur_sig = (int(cur_len), last_ts, last_type, last_sym)
        except Exception:
            cur_sig = None

        if cur_sig is not None and cur_sig != self._last_trade_tape_sig:
            self._last_trade_tape_sig = cur_sig
            include_trade_tape = bool(cur_sig[0] > 0)

        include_logs = False
        try:
            logs = getattr(self.orchestrator, "logs", None)
            cur_len = len(logs) if logs is not None else 0
            last = logs[-1] if (logs is not None and cur_len) else None
            last_time = None
            last_level = None
            last_msg = None
            if isinstance(last, dict):
                last_time = last.get("time")
                last_level = last.get("level")
                last_msg = last.get("msg")
            cur_sig = (int(cur_len), last_time, last_level, last_msg)
        except Exception:
            cur_sig = None
        if cur_sig is not None and cur_sig != self._last_logs_sig:
            self._last_logs_sig = cur_sig
            include_logs = True

        payload = _build_payload(
            self.orchestrator,
            rows,
            include_history=False,
            include_trade_tape=include_trade_tape,
            include_logs=include_logs,
        )
        payload = _sanitize_for_json(payload)
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        
        for ws in list(self.orchestrator.clients):
            try:
                await ws.send_str(data)
            except Exception:
                self.orchestrator.clients.discard(ws)

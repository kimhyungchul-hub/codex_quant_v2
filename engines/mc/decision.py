from __future__ import annotations

# Decision policy for MonteCarloEngine.

import logging
import os
import time
from typing import Any, Dict

import numpy as np

from engines.mc.constants import MC_VERBOSE_PRINT
from engines.mc_risk import kelly_with_cvar
from utils.helpers import now_ms

# ✅ GPU-Accelerated Leverage Optimization
try:
    from engines.mc.leverage_optimizer_jax import find_optimal_leverage
    USE_GPU_LEVERAGE_OPT = True
except ImportError:
    USE_GPU_LEVERAGE_OPT = False

logger = logging.getLogger(__name__)

_LAST_LOG_MS: dict[tuple[str, str], int] = {}


def _throttled_log(symbol: str, key: str, interval_ms: int) -> bool:
    try:
        now = int(time.time() * 1000)
        k = (str(symbol), str(key))
        last = int(_LAST_LOG_MS.get(k, 0) or 0)
        if now - last >= int(interval_ms):
            _LAST_LOG_MS[k] = now
            return True
        return False
    except Exception:
        return True


from engines.mc.config import config

# EMA memory for leverage smoothing
_LEV_EMA_MEM: dict[str, float] = {}


class MonteCarloDecisionMixin:
    def decide(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(ctx.get("symbol", "UNKNOWN"))
        price = float(ctx.get("price", 0.0))
        regime_ctx = str(ctx.get("regime", "chop"))
        params = self._get_params(regime_ctx, ctx)

        # ✅ Deterministic seed for stability (anchored to 5s window)
        ts_ms = int(ctx.get("ts") or (time.time() * 1000))
        seed_window = int(ts_ms // 5000)
        seed = int((hash(symbol) ^ seed_window) & 0xFFFFFFFF)

        # ✅ [EV_DEBUG] mc_engine.decide 시작 로그
        boost_val = float(ctx.get("boost", 1.0))
        if MC_VERBOSE_PRINT:
            print(f"[EV_DEBUG] mc_engine.decide: START symbol={symbol} price={price} regime={regime_ctx} boost={boost_val:.2f}")
            logger.info(f"[EV_DEBUG] mc_engine.decide: START symbol={symbol} price={price} regime={regime_ctx} boost={boost_val:.2f}")

        # [Paper Trading] Inject pmaker_entry for simulation if not present
        if "pmaker_entry" not in ctx:
            ctx["pmaker_entry"] = config.pmaker_prob

        # ✅ 최적 레버리지 자동 산출
        leverage_val = ctx.get("leverage")
        optimal_leverage = float(leverage_val if leverage_val is not None else 5.0)
        optimal_size = 0.0
        best_net_ev = None
        
        # ============================================================
        # Funnel Structure Entry Filter Logic
        # ✅ Step 0: NAPV (Net Added Present Value) Filter
        # ============================================================
        action = "WAIT"
        filter_reason = None
        
        use_napv_filter = config.funnel_use_napv_filter
        napv_val = float(ctx.get("napv", 0.0))
        napv_threshold = config.funnel_napv_threshold
        group = ctx.get("group", "OTHER")
        
        if use_napv_filter:
            if napv_val < napv_threshold:
                filter_reason = f"NAPV_FILTER: napv={napv_val:.6f} < threshold={napv_threshold:.6f} (group={group})"
                logger.info(f"[FUNNEL_FILTER] {symbol} | {filter_reason}")
        
        try:
            # 1. 기초 Pass
            ctx_base = ctx.copy()
            ctx_base["leverage"] = 1.0
            ctx_base["_mu_alpha_ema_skip_update"] = True
            metrics_base = self.evaluate_entry_metrics(ctx_base, params, seed=seed)
            ev_raw = float(metrics_base.get("ev_raw", metrics_base.get("ev", 0.0)) or 0.0)
            
            if config.score_only_mode:
                score_gap = abs(float(metrics_base.get("policy_ev_gap", 0.0) or 0.0))
                if score_gap > 0:
                    ev_raw = max(abs(ev_raw), score_gap)

            sigma_raw = float(metrics_base.get("sigma_sim", metrics_base.get("sigma_annual", 0.0)) or 0.0)
            win_rate = float(metrics_base.get("win", 0.0) or 0.0)
            cvar_raw = float(metrics_base.get("cvar", 0.0) or 0.0)
            
            if ev_raw != 0.0 or sigma_raw > 0:
                execution_cost_base = float(metrics_base.get("execution_cost", 0.0) or 0.0)
                fee_roundtrip_total_base = float(metrics_base.get("fee_roundtrip_total", execution_cost_base) or execution_cost_base)
                fee_rate_total = fee_roundtrip_total_base
                max_leverage = float(ctx.get("max_leverage", 100.0) or 100.0)
                MAX_POSITION_SIZE_CAP = 1.0
                
                # GPU Optimization
                if USE_GPU_LEVERAGE_OPT and config.use_gpu_leverage:
                    try:
                        mu_annual = float(metrics_base.get("mu_adjusted", 0.0) or 0.0)
                        sigma_annual = float(metrics_base.get("sigma_annual", sigma_raw) or sigma_raw)
                        horizon_sec = int(metrics_base.get("policy_horizon_eff_sec", 300) or 300)
                        fee_base = fee_rate_total
                        
                        optimal_leverage_gpu, optimal_score_gpu = find_optimal_leverage(
                            mu_annual=mu_annual, sigma_annual=sigma_annual,
                            horizon_sec=horizon_sec, fee_base=fee_base, use_gpu=True,
                        )
                        optimal_leverage_gpu = float(max(1.0, min(max_leverage, optimal_leverage_gpu)))
                        
                        if optimal_leverage_gpu > 1.0:
                            optimal_leverage = optimal_leverage_gpu
                            # Normalize Ratio against 100x instead of max_leverage to avoid 100% distortion
                            # KellyAllocator will still use this as a relative weight.
                            raw_size = optimal_leverage_gpu / 100.0
                            boost = float(ctx.get("boost", 1.0))
                            if config.kelly_boost_enabled:
                                optimal_size = min(raw_size * boost, MAX_POSITION_SIZE_CAP)
                                msg = f"[KELLY_BOOST] {symbol} | GPU raw_size={raw_size:.4f} * boost={boost:.2f} -> optimal_size={optimal_size:.4f}"
                                logger.info(msg)
                                if MC_VERBOSE_PRINT: print(msg)
                            else:
                                optimal_size = min(raw_size, MAX_POSITION_SIZE_CAP)
                            best_net_ev = optimal_score_gpu
                            candidate_leverages = []
                    except Exception as e:
                        logger.warning(f"[GPU_LEVERAGE] {symbol} | GPU failed: {e}")

                # Kelly Fallback
                if not (USE_GPU_LEVERAGE_OPT and config.use_gpu_leverage and optimal_leverage > 1.0):
                    candidate_leverages = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100]
                    candidate_leverages = [L for L in candidate_leverages if L <= max_leverage]
                    best_leverage, best_kelly, best_net_ev_local, best_size = 1.0, 0.0, -1e18, 0.0
                    
                    for L in candidate_leverages:
                        net_ev = (ev_raw * L) - (fee_rate_total * L)
                        adj_sig = sigma_raw * L
                        if net_ev > 0 and adj_sig > 1e-6:
                            tp_est = abs(ev_raw) * 2.0
                            sl_est = abs(cvar_raw) if cvar_raw < 0 else abs(ev_raw) * 0.5
                            kelly_cvar = kelly_with_cvar(win_rate, tp_est, sl_est, cvar_raw * L)
                            growth = kelly_cvar * net_ev
                            if growth > best_kelly:
                                best_leverage, best_kelly, best_net_ev_local, best_size = float(L), growth, net_ev, min(kelly_cvar, 1.0)
                    
                    if best_net_ev_local > 0:
                        optimal_leverage = best_leverage
                        boost = float(ctx.get("boost", 1.0))
                        if config.kelly_boost_enabled:
                            optimal_size = min(best_size * boost, MAX_POSITION_SIZE_CAP)
                            msg = f"[KELLY_BOOST] {symbol} | Kelly best_size={best_size:.4f} * boost={boost:.2f} -> optimal_size={optimal_size:.4f}"
                            logger.info(msg)
                            if MC_VERBOSE_PRINT: print(msg)
                        else:
                            optimal_size = best_size
                        best_net_ev = best_net_ev_local
                    else:
                        # Falling back to Dynamic Leverage (e.g. for score-only)
                        k_lev = config.k_lev
                        sig_strength = float(metrics_base.get("policy_signal_strength", 0.0) or 0.0)
                        if sig_strength > 0 and config.score_only_mode:
                            boost = float(ctx.get("boost", 1.0))
                            if config.kelly_boost_enabled:
                                optimal_leverage = float(max(1.0, min(max_leverage, sig_strength * k_lev * boost)))
                                msg = f"[DYNAMIC_BOOST] {symbol} | leverage={sig_strength*k_lev:.2f} * boost={boost:.2f} -> final={optimal_leverage:.2f}"
                                logger.info(msg)
                                if MC_VERBOSE_PRINT: print(msg)
                            else:
                                optimal_leverage = float(max(1.0, min(max_leverage, sig_strength * k_lev)))
                            best_net_ev = None
        except Exception as e:
            logger.error(f"[LEVERAGE_ERROR] {symbol} | {e}")
            import traceback
            logger.error(traceback.format_exc())

        # smoothing size EMA
        ema_alpha = 0.3
        prev_size = _LEV_EMA_MEM.get(symbol, optimal_size)
        smoothed_size = prev_size + ema_alpha * (optimal_size - prev_size)
        _LEV_EMA_MEM[symbol] = smoothed_size
        
        ctx_final = ctx.copy()
        ctx_final["leverage"] = optimal_leverage
        ctx_final["size_frac"] = smoothed_size
        
        metrics = self.evaluate_entry_metrics(ctx_final, params, seed=seed)
        # Ensure score fields are always present for ranking/Kelly, even when entry is filtered.
        # Some evaluation modes may not populate policy_ev_score_long/short; fall back to policy_ev_mix_long/short.
        try:
            if isinstance(metrics, dict):
                def _as_float(v, default=None):
                    try:
                        if v is None:
                            return default
                        return float(v)
                    except Exception:
                        return default

                sL = _as_float(metrics.get("policy_ev_score_long"), default=None)
                sS = _as_float(metrics.get("policy_ev_score_short"), default=None)
                if sL is None:
                    sL = _as_float(metrics.get("policy_ev_mix_long"), default=0.0)
                    metrics["policy_ev_score_long"] = sL
                if sS is None:
                    sS = _as_float(metrics.get("policy_ev_mix_short"), default=0.0)
                    metrics["policy_ev_score_short"] = sS
        except Exception:
            pass
        best_h = int(metrics.get("best_h", 300))
        best_desc = f"{best_h}s"
        win_value = float(metrics.get("win", 0.0) or 0.0)
        cvar_value = float(metrics.get("cvar", 0.0) or 0.0)
        policy_ev_mix = float(metrics.get("policy_ev_mix", 0.0) or 0.0)
        direction_policy = int(metrics.get("direction", 0) or 0)
        
        if config.score_only_mode and not filter_reason:
            target_action = "LONG" if direction_policy == 1 else "SHORT" if direction_policy == -1 else "WAIT"
            return {
                "action": target_action, "ev": policy_ev_mix, "ev_raw": float(metrics.get("ev_raw", policy_ev_mix)),
                "confidence": win_value, "reason": f"SCORE({best_desc}) {regime_ctx} EV {policy_ev_mix*100:.2f}%",
                "meta": metrics, "size_frac": smoothed_size, "optimal_leverage": optimal_leverage, "optimal_size": smoothed_size,
            }
        
        if not filter_reason:
            regime = str(regime_ctx).upper() if regime_ctx else "CHOP"
            if regime not in ["BULL", "BEAR", "CHOP", "VOLATILE"]: regime = "CHOP"
            win_floors = {"BULL": config.funnel_win_floor_bull, "BEAR": config.funnel_win_floor_bear, "CHOP": config.funnel_win_floor_chop, "VOLATILE": config.funnel_win_floor_volatile}
            cvar_floors = {"BULL": config.funnel_cvar_floor_bull, "BEAR": config.funnel_cvar_floor_bear, "CHOP": config.funnel_cvar_floor_chop, "VOLATILE": config.funnel_cvar_floor_volatile}
            event_cvar_floors = {"BULL": config.funnel_event_cvar_floor_bull, "BEAR": config.funnel_event_cvar_floor_bear, "CHOP": config.funnel_event_cvar_floor_chop, "VOLATILE": config.funnel_event_cvar_floor_volatile}
            
            ev_for_filter = policy_ev_mix if policy_ev_mix != 0.0 else float(metrics.get("ev", 0.0))
            win_rate_f = float(metrics.get("policy_p_pos_mix", metrics.get("win", 0.0)) or 0.0)
            cvar1 = float(metrics.get("cvar1", metrics.get("cvar", 0.0)) or 0.0)
            event_cvar_r = metrics.get("event_cvar_r")
            
            win_floor = win_floors.get(regime, 0.0) if config.funnel_use_winrate_filter else 0.0
            
            if ev_for_filter <= 0.0:
                filter_reason = f"EV_FILTER: ev={ev_for_filter:.6f} <= 0.0"
            elif win_rate_f < win_floor:
                filter_reason = f"WINRATE_FILTER: win={win_rate_f:.4f} < {win_floor:.4f}"
            elif cvar1 < cvar_floors.get(regime, -0.10):
                filter_reason = f"CVAR_FILTER: cvar1={cvar1:.4f}"
            elif event_cvar_r is not None and event_cvar_r < event_cvar_floors.get(regime, -1.0):
                filter_reason = f"EVENT_FILTER: {event_cvar_r:.4f}"
            
            if not filter_reason:
                if direction_policy == 1: action = "LONG"
                elif direction_policy == -1: action = "SHORT"
                else: filter_reason = "DIRECTION_ZERO"
        
        if filter_reason and MC_VERBOSE_PRINT:
            logger.info(f"[FUNNEL_FILTER] {symbol} | {filter_reason}")

        ev_value = policy_ev_mix if policy_ev_mix != 0.0 else float(metrics.get("ev", 0.0))

        res = {
            "action": action, 
            "ev": ev_value,
            "ev_raw": float(metrics.get("ev_raw", ev_value)),
            "confidence": win_rate_f if 'win_rate_f' in locals() else win_rate,
            "reason": f"MC({best_desc}) {regime_ctx} EV {ev_value*100:.2f}%" if not filter_reason else f"FILTERED: {filter_reason}",
            "meta": metrics, 
            "size_frac": smoothed_size, 
            "optimal_leverage": optimal_leverage, 
            "optimal_size": smoothed_size,
            "boost": boost_val,
        }
        if MC_VERBOSE_PRINT:
            logger.info(f"[DECIDE_FINAL] {symbol} | action={res['action']} ev={res['ev']:.6f} reason={res['reason']}")
            print(f"[DECIDE_FINAL] {symbol} | action={res['action']} ev={res['ev']:.6f} reason={res['reason']}")
        return res

    def _get_params(self, regime: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "n_paths": int(ctx.get("n_paths", config.n_paths_live)),
            "use_jax": bool(ctx.get("use_jax", True)),
            "tail_mode": str(ctx.get("tail_mode", "student_t")),
            "student_t_df": float(ctx.get("student_t_df", 3.0)),
        }
        return params

    def evaluate_entry_metrics(self, ctx: Dict[str, Any], params: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
        if hasattr(self, '_evaluate_entry_metrics_impl'):
            return self._evaluate_entry_metrics_impl(ctx, params, seed=seed)
        return {"ev": 0.0, "win": 0.0, "cvar": 0.0, "can_enter": False, "best_h": 300}
    def decide_batch(self, ctx_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        GLOBAL BATCH DECISION: 여러 심볼의 의사결정을 한 번에 처리한다.
        """
        if not ctx_list:
            return []
        import time
        num_symbols = len(ctx_list)
        ts_ms = now_ms()
        seed_window = int(ts_ms // 5000)
        print(f"[DECIDE_BATCH] START symbols={num_symbols} ts={ts_ms}")
        
        # 1. Prepare tasks for evaluation
        tasks = []
        for ctx in ctx_list:
            symbol = str(ctx.get("symbol", "UNKNOWN"))
            regime = str(ctx.get("regime", "chop"))
            params = self._get_params(regime, ctx)
            seed = int((hash(symbol) ^ seed_window) & 0xFFFFFFFF)
            
            # Simplified NAPV filter check for batching
            napv_val = float(ctx.get("napv", 0.0))
            napv_threshold = config.funnel_napv_threshold
            if config.funnel_use_napv_filter and napv_val < napv_threshold:
                continue # Skip evaluation for this symbol
                
            tasks.append({"ctx": ctx, "params": params, "seed": seed})
            
        if not tasks:
            return [{"ok": False, "reason": "ALL_FILTERED"} for _ in ctx_list]
            
        # 2. Call Batch Evaluation (The Global JAX Vmap!)
        import time
        t_eval_start = time.perf_counter()
        print(f"[DECIDE_BATCH] evaluate_entry_metrics_batch START num_tasks={len(tasks)}")
        batch_metrics = self.evaluate_entry_metrics_batch(tasks)
        t_eval_end = time.perf_counter()
        print(f"[DECIDE_BATCH] evaluate_entry_metrics_batch END elapsed={(t_eval_end-t_eval_start):.3f}s")
        
        # 3. Post-process and map back to input order
        results_map = {t["ctx"].get("symbol"): batch_metrics[i] for i, t in enumerate(tasks)}
            
        final_decisions = []
        import math
        
        for ctx in ctx_list:
            sym = ctx.get("symbol")
            if sym in results_map:
                metrics = results_map[sym]
                
                policy_score = float(metrics.get("score", 0.0))
                if math.isnan(policy_score): policy_score = 0.0
                
                direction = 0
                if policy_score > 1e-9: direction = 1
                elif policy_score < -1e-9: direction = -1
                
                metric_copy = metrics.copy()
                # Explicitly set fields for dashboard
                metric_copy["score"] = policy_score
                metric_copy["ev"] = metrics.get("ev", policy_score)
                metric_copy["confidence"] = metrics.get("win_prob", 0.0)
                metric_copy["mc_win_rate"] = metrics.get("win_prob", 0.0)
                metric_copy["mc_cvar"] = metrics.get("cvar", 0.0)
                metric_copy["best_h"] = int(metrics.get("best_h", 300))
                
                # Standard Filtering Logic for Batch
                target_action = "WAIT"
                filter_reason = None
                
                # Get metrics for filtering - use positive EV for both Long/Short
                ev_val = float(metrics.get("ev", 0.0))
                win_rate_f = float(metrics.get("win_prob", 0.0))
                cvar1 = float(metrics.get("cvar", 0.0))
                
                # Apply filters if not in score_only_mode
                if not config.score_only_mode:
                    regime = str(ctx.get("regime", "CHOP")).upper()
                    if regime not in ["BULL", "BEAR", "CHOP", "VOLATILE"]: regime = "CHOP"
                    
                    win_floors = {"BULL": config.funnel_win_floor_bull, "BEAR": config.funnel_win_floor_bear, "CHOP": config.funnel_win_floor_chop, "VOLATILE": config.funnel_win_floor_volatile}
                    cvar_floors = {"BULL": config.funnel_cvar_floor_bull, "BEAR": config.funnel_cvar_floor_bear, "CHOP": config.funnel_cvar_floor_chop, "VOLATILE": config.funnel_cvar_floor_volatile}
                    
                    win_floor = win_floors.get(regime, 0.0) if config.funnel_use_winrate_filter else 0.0
                    cvar_floor = cvar_floors.get(regime, -0.10)
                    
                    if ev_val <= 0.0:
                        filter_reason = f"EV_FILTER: ev={ev_val:.6f}"
                    elif win_rate_f < win_floor:
                        filter_reason = f"WINRATE_FILTER: win={win_rate_f:.4f} < {win_floor:.4f}"
                    elif cvar1 < cvar_floor:
                        filter_reason = f"CVAR_FILTER: cvar1={cvar1:.4f}"
                    
                    if not filter_reason:
                        if direction == 1: target_action = "LONG"
                        elif direction == -1: target_action = "SHORT"
                        else: filter_reason = "DIRECTION_ZERO"
                else:
                    # Score only mode: simply follow the direction
                    if direction == 1: target_action = "LONG"
                    elif direction == -1: target_action = "SHORT"
                    else: target_action = "WAIT"
                    filter_reason = None

                metric_copy["action"] = target_action
                metric_copy["status"] = target_action # Sync status with action for dashboard
                if filter_reason:
                    metric_copy["reason"] = f"FILTERED: {filter_reason}"
                else:
                    best_h = int(metrics.get("best_h", 3600))
                    metric_copy["reason"] = f"BATCH({best_h}s) {ctx.get('regime','chop')} EV {ev_val*100:.2f}%"
                
                # Compatibility with LiveOrchestrator._paper_trade_step
                meta = metrics.get("meta", {})
                metric_copy["optimal_leverage"] = float(meta.get("optimal_leverage") or meta.get("total_leverage_allocated") or 20.0)
                metric_copy["optimal_size"] = float(meta.get("optimal_size") or meta.get("size_frac") or 0.05)
                
                # Avoid recursion: details list contains a shallow copy without 'details' key
                detail_copy = metric_copy.copy()
                metric_copy["details"] = [detail_copy]

                final_decisions.append(metric_copy)
            else:
                final_decisions.append({"ok": False, "reason": "FILTERED", "action": "WAIT", "score": 0.0})
        print(f"[DECIDE_BATCH] END symbols={num_symbols} elapsed={(time.time()*1000 - ts_ms):.0f}ms")
        return final_decisions

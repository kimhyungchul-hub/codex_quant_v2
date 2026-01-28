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
from core.economic_brain import EconomicBrain

# ✅ GPU-Accelerated Leverage Optimization
try:
    from engines.mc.leverage_optimizer_jax import find_optimal_leverage
    USE_GPU_LEVERAGE_OPT = True
except ImportError:
    USE_GPU_LEVERAGE_OPT = False

logger = logging.getLogger(__name__)

_LAST_LOG_MS: dict[tuple[str, str], int] = {}
_UNIFIED_BRAIN = EconomicBrain()


def _calc_unified_score(
    horizons: list[int] | list[float],
    cumulative_ev: list[float],
    cumulative_cvar: list[float],
    *,
    cost: float,
    rho: float,
    lambda_param: float,
) -> tuple[float, float]:
    try:
        return _UNIFIED_BRAIN.calculate_unified_score(
            horizons_sec=np.asarray(horizons, dtype=float),
            cumulative_ev=np.asarray(cumulative_ev, dtype=float),
            cumulative_cvar=np.asarray(cumulative_cvar, dtype=float),
            cost=float(cost),
            rho=float(rho),
            lambda_param=float(lambda_param),
        )
    except Exception:
        return 0.0, 0.0


def _get_vector(metrics: Dict[str, Any], key: str) -> list[float]:
    v = metrics.get(key)
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in v]
        except Exception:
            return []
    return []


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
        
        action = "WAIT"
        
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
        win_value = float(metrics.get("win", 0.0) or 0.0)

        horizons_long = _get_vector(metrics, "horizon_seq_long")
        horizons_short = _get_vector(metrics, "horizon_seq_short")
        ev_long = _get_vector(metrics, "ev_by_horizon_long")
        ev_short = _get_vector(metrics, "ev_by_horizon_short")
        cvar_long = _get_vector(metrics, "cvar_by_horizon_long")
        cvar_short = _get_vector(metrics, "cvar_by_horizon_short")

        lev_val = float(ctx_final.get("leverage") or 1.0)
        cost_base = float(metrics.get("fee_roundtrip_total", metrics.get("execution_cost", 0.0)) or 0.0)
        cost_roe = float(cost_base * lev_val)
        cost_roe_exit = float(0.5 * cost_base * lev_val)
        rho_val = float(ctx_final.get("rho", config.unified_rho))
        lambda_val = float(ctx_final.get("unified_lambda", config.unified_risk_lambda))

        # Convert net cumulative vectors to gross for marginal extraction
        ev_long_gross = [v + cost_roe for v in ev_long]
        ev_short_gross = [v + cost_roe for v in ev_short]
        cvar_long_gross = [v + cost_roe for v in cvar_long]
        cvar_short_gross = [v + cost_roe for v in cvar_short]

        score_long, t_long = _calc_unified_score(
            horizons_long, ev_long_gross, cvar_long_gross,
            cost=cost_roe, rho=rho_val, lambda_param=lambda_val,
        ) if horizons_long and ev_long_gross and cvar_long_gross else (0.0, 0.0)
        score_short, t_short = _calc_unified_score(
            horizons_short, ev_short_gross, cvar_short_gross,
            cost=cost_roe, rho=rho_val, lambda_param=lambda_val,
        ) if horizons_short and ev_short_gross and cvar_short_gross else (0.0, 0.0)

        score_long_hold, _ = _calc_unified_score(
            horizons_long, ev_long_gross, cvar_long_gross,
            cost=cost_roe_exit, rho=rho_val, lambda_param=lambda_val,
        ) if horizons_long and ev_long_gross and cvar_long_gross else (0.0, 0.0)
        score_short_hold, _ = _calc_unified_score(
            horizons_short, ev_short_gross, cvar_short_gross,
            cost=cost_roe_exit, rho=rho_val, lambda_param=lambda_val,
        ) if horizons_short and ev_short_gross and cvar_short_gross else (0.0, 0.0)

        if score_long >= score_short:
            best_score = float(score_long)
            best_dir = 1
            best_t = float(t_long)
        else:
            best_score = float(score_short)
            best_dir = -1
            best_t = float(t_short)

        hold_score = None
        pos_side = int(ctx_final.get("position_side", 0) or 0)
        if pos_side == 1:
            hold_score = float(score_long_hold)
        elif pos_side == -1:
            hold_score = float(score_short_hold)

        if config.score_only_mode:
            if best_dir == 1:
                action = "LONG"
            elif best_dir == -1:
                action = "SHORT"
            else:
                action = "WAIT"
        else:
            if best_score > 0:
                action = "LONG" if best_dir == 1 else "SHORT"
            else:
                action = "WAIT"

        best_desc = f"{int(best_t)}s" if best_t > 0 else f"{int(metrics.get('best_h', 300))}s"
        reason = f"UNIFIED({best_desc}) score {best_score*100:.2f}%"

        metrics["unified_score"] = float(best_score)
        metrics["unified_score_long"] = float(score_long)
        metrics["unified_score_short"] = float(score_short)
        metrics["unified_score_hold"] = float(hold_score) if hold_score is not None else None
        metrics["unified_t_star"] = float(best_t)
        metrics["unified_t_star_long"] = float(t_long)
        metrics["unified_t_star_short"] = float(t_short)
        metrics["unified_lambda"] = float(lambda_val)
        metrics["unified_rho"] = float(rho_val)
        metrics["unified_cost_roe"] = float(cost_roe)
        metrics["unified_cost_roe_exit"] = float(cost_roe_exit)
        metrics["unified_direction"] = int(best_dir)
        if best_t > 0:
            metrics["best_h"] = int(best_t)
        metrics["best_horizon_steps"] = int(best_t) if best_t > 0 else int(metrics.get("best_h", 300))

        res = {
            "action": action,
            "ev": best_score,
            "ev_raw": best_score,
            "confidence": win_value,
            "reason": reason,
            "meta": metrics,
            "size_frac": smoothed_size,
            "optimal_leverage": optimal_leverage,
            "optimal_size": smoothed_size,
            "boost": boost_val,
            "unified_score": float(best_score),
            "unified_score_long": float(score_long),
            "unified_score_short": float(score_short),
            "unified_score_hold": hold_score,
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

                policy_score = float(metrics.get("unified_score", metrics.get("score", 0.0)) or 0.0)
                if math.isnan(policy_score):
                    policy_score = 0.0

                score_long = float(metrics.get("unified_score_long", metrics.get("policy_ev_score_long", 0.0)) or 0.0)
                score_short = float(metrics.get("unified_score_short", metrics.get("policy_ev_score_short", 0.0)) or 0.0)
                meta = metrics.get("meta", {}) if isinstance(metrics, dict) else {}
                hold_score = None
                try:
                    src = meta if isinstance(meta, dict) else metrics
                    horizons = _get_vector(src, "horizon_seq")
                    ev_long_vec = _get_vector(src, "ev_by_horizon_long")
                    ev_short_vec = _get_vector(src, "ev_by_horizon_short")
                    cvar_long_vec = _get_vector(src, "cvar_by_horizon_long")
                    cvar_short_vec = _get_vector(src, "cvar_by_horizon_short")
                    if horizons and ev_long_vec and ev_short_vec and cvar_long_vec and cvar_short_vec:
                        lev_val = float(src.get("leverage", metrics.get("leverage", 1.0)) or 1.0)
                        cost_base = float(src.get("fee_roundtrip_total", metrics.get("fee_roundtrip_total", 0.0)) or 0.0)
                        cost_roe_full = float(cost_base * lev_val)
                        cost_roe_exit = float(0.5 * cost_roe_full)
                        rho_val = float(src.get("unified_rho", config.unified_rho))
                        lambda_val = float(src.get("unified_lambda", config.unified_risk_lambda))
                        ev_long_gross = [v + cost_roe_full for v in ev_long_vec]
                        ev_short_gross = [v + cost_roe_full for v in ev_short_vec]
                        cvar_long_gross = [v + cost_roe_full for v in cvar_long_vec]
                        cvar_short_gross = [v + cost_roe_full for v in cvar_short_vec]
                        score_long_hold, _ = _calc_unified_score(
                            horizons, ev_long_gross, cvar_long_gross,
                            cost=cost_roe_exit, rho=rho_val, lambda_param=lambda_val,
                        )
                        score_short_hold, _ = _calc_unified_score(
                            horizons, ev_short_gross, cvar_short_gross,
                            cost=cost_roe_exit, rho=rho_val, lambda_param=lambda_val,
                        )
                        pos_side = int(ctx.get("position_side", 0) or 0)
                        if pos_side == 1:
                            hold_score = float(score_long_hold)
                        elif pos_side == -1:
                            hold_score = float(score_short_hold)
                except Exception:
                    hold_score = None

                direction = 0
                if score_long > score_short:
                    direction = 1
                elif score_short > score_long:
                    direction = -1
                elif policy_score > 0:
                    direction = 1
                elif policy_score < 0:
                    direction = -1

                metric_copy = metrics.copy()
                metric_copy["score"] = policy_score
                metric_copy["ev"] = policy_score
                metric_copy["confidence"] = metrics.get("win", metrics.get("win_prob", 0.0))
                metric_copy["mc_win_rate"] = metrics.get("win_prob", 0.0)
                metric_copy["mc_cvar"] = metrics.get("cvar", 0.0)
                metric_copy["best_h"] = int(metrics.get("best_h", 300))
                metric_copy["unified_score"] = policy_score
                metric_copy["unified_score_long"] = float(score_long)
                metric_copy["unified_score_short"] = float(score_short)

                pos_side = int(ctx.get("position_side", 0) or 0)
                if hold_score is None:
                    if pos_side == 1:
                        hold_score = float(score_long)
                    elif pos_side == -1:
                        hold_score = float(score_short)
                if hold_score is not None:
                    metric_copy["unified_score_hold"] = float(hold_score)

                target_action = "WAIT"
                if config.score_only_mode:
                    if direction == 1:
                        target_action = "LONG"
                    elif direction == -1:
                        target_action = "SHORT"
                else:
                    if policy_score > 0:
                        target_action = "LONG" if direction == 1 else "SHORT"

                metric_copy["action"] = target_action
                metric_copy["status"] = target_action
                best_h = int(metrics.get("best_h", 3600))
                metric_copy["reason"] = f"UNIFIED({best_h}s) {ctx.get('regime','chop')} score {policy_score*100:.2f}%"
                
                # Compatibility with LiveOrchestrator._paper_trade_step
                meta = metrics.get("meta", {})
                metric_copy["optimal_leverage"] = float(meta.get("optimal_leverage") or meta.get("total_leverage_allocated") or 20.0)
                metric_copy["optimal_size"] = float(meta.get("optimal_size") or meta.get("size_frac") or 0.05)
                
                # Avoid recursion: details list contains a shallow copy without 'details' key
                detail_copy = metric_copy.copy()
                metric_copy["details"] = [detail_copy]

                final_decisions.append(metric_copy)
            else:
                final_decisions.append({
                    "ok": False, "reason": "NO_METRICS", "action": "WAIT", "score": 0.0,
                })
        print(f"[DECIDE_BATCH] END symbols={num_symbols} elapsed={(time.time()*1000 - ts_ms):.0f}ms")
        return final_decisions

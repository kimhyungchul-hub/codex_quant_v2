from __future__ import annotations

# Decision policy for MonteCarloEngine.

import logging
import math
import os
import time
from dataclasses import replace
from typing import Any, Dict

import numpy as np

from engines.mc.constants import MC_VERBOSE_PRINT
from engines.mc_risk import kelly_with_cvar
from utils.helpers import now_ms
from core.economic_brain import EconomicBrain

# ✅ GPU-Accelerated Leverage Optimization (PyTorch, lazy import)
USE_GPU_LEVERAGE_OPT = False
_FIND_OPTIMAL_LEVERAGE = None


def _maybe_load_leverage_optimizer() -> None:
    global USE_GPU_LEVERAGE_OPT, _FIND_OPTIMAL_LEVERAGE
    if _FIND_OPTIMAL_LEVERAGE is not None or USE_GPU_LEVERAGE_OPT:
        return
    use_torch_env = str(os.environ.get("MC_USE_TORCH", "1")).strip().lower()
    if use_torch_env in ("0", "false", "no", "off"):
        USE_GPU_LEVERAGE_OPT = False
        return
    try:
        from engines.mc.leverage_optimizer_torch import find_optimal_leverage as _find_optimal_leverage
        _FIND_OPTIMAL_LEVERAGE = _find_optimal_leverage
        USE_GPU_LEVERAGE_OPT = True
    except Exception:
        USE_GPU_LEVERAGE_OPT = False

logger = logging.getLogger(__name__)

_LAST_LOG_MS: dict[tuple[str, str], int] = {}
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
_UNIFIED_BRAIN = EconomicBrain()


def _is_auto_value(value: Any) -> bool:
    try:
        return str(value).strip().lower() in ("auto", "dyn", "adaptive")
    except Exception:
        return False


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


def _pick_ev_expected(metrics: Dict[str, Any], best_dir: int) -> float | None:
    def _get(key: str):
        v = metrics.get(key)
        if v is None:
            meta = metrics.get("meta")
            if isinstance(meta, dict):
                v = meta.get(key)
        return v

    v = _get("policy_ev_target")
    if v is None:
        if best_dir == 1:
            v = _get("policy_best_ev_long")
        elif best_dir == -1:
            v = _get("policy_best_ev_short")
        else:
            v_long = _get("policy_best_ev_long")
            v_short = _get("policy_best_ev_short")
            candidates = []
            for x in (v_long, v_short):
                try:
                    candidates.append(float(x))
                except Exception:
                    pass
            v = max(candidates) if candidates else None
    try:
        return float(v)
    except Exception:
        return None


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

        use_hybrid = str(os.environ.get("MC_USE_HYBRID_PLANNER", "0")).strip().lower() in ("1", "true", "yes", "on")
        hybrid_only_env = os.environ.get("MC_HYBRID_ONLY")
        hybrid_only = use_hybrid if hybrid_only_env is None else str(hybrid_only_env).strip().lower() in ("1", "true", "yes", "on")

        if hybrid_only:
            return self._decide_hybrid_only(ctx, params, seed, symbol, price, regime_ctx, boost_val)

        # ✅ 최적 레버리지 자동 산출
        try:
            lev_floor_env = float(os.environ.get("LEVERAGE_MIN", 1.0) or 1.0)
        except Exception:
            lev_floor_env = 1.0
        lev_floor_env = float(max(1.0, lev_floor_env))
        optimal_leverage = float(lev_floor_env)
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
                _maybe_load_leverage_optimizer()
                if USE_GPU_LEVERAGE_OPT and config.use_gpu_leverage and _FIND_OPTIMAL_LEVERAGE is not None:
                    try:
                        mu_annual = float(metrics_base.get("mu_adjusted", 0.0) or 0.0)
                        sigma_annual = float(metrics_base.get("sigma_annual", sigma_raw) or sigma_raw)
                        horizon_sec = int(metrics_base.get("policy_horizon_eff_sec", 300) or 300)
                        fee_base = fee_rate_total
                        
                        optimal_leverage_gpu, optimal_score_gpu = _FIND_OPTIMAL_LEVERAGE(
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
                        else:
                            # No positive edge from Kelly scan -> keep leverage at floor (avoid sticky default 5x).
                            optimal_leverage = float(max(1.0, min(max_leverage, lev_floor_env)))
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

        ev_expected = _pick_ev_expected(metrics, best_dir)

        hold_score = None
        pos_side = int(ctx_final.get("position_side", 0) or 0)
        if pos_side == 1:
            hold_score = float(score_long_hold)
        elif pos_side == -1:
            hold_score = float(score_short_hold)

        entry_floor = float(os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
        if config.score_only_mode:
            if best_dir == 1:
                action = "LONG"
            elif best_dir == -1:
                action = "SHORT"
            else:
                action = "WAIT"
        else:
            if best_score > entry_floor:
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
        metrics["ev_expected"] = float(ev_expected) if ev_expected is not None else None
        metrics["ev_best"] = float(ev_expected) if ev_expected is not None else None

        # Use true EV from metrics; keep unified score separately for ranking.
        ev_val = float(metrics.get("ev", best_score) or 0.0)
        ev_raw_val = float(metrics.get("ev_raw", ev_val) or ev_val)

        res = {
            "action": action,
            "ev": ev_val,
            "ev_raw": ev_raw_val,
            "ev_expected": float(ev_expected) if ev_expected is not None else None,
            "ev_best": float(ev_expected) if ev_expected is not None else None,
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

        # Hybrid planner override (optional, gated by env)
        if use_hybrid:
            try:
                mc_paths = self._build_hybrid_mc_paths(ctx_final, params, seed)
                hybrid_decision, hybrid_meta = self.decide_with_hybrid_planner(
                    symbol, ctx_final, mc_paths, return_detail=True
                )
                # Normalize HOLD -> WAIT (EXIT is preserved for explicit exit handling)
                if hybrid_decision == "HOLD":
                    hybrid_action_norm = "WAIT"
                else:
                    hybrid_action_norm = hybrid_decision
                res["action"] = hybrid_action_norm
                meta = res.get("meta") or {}
                meta["hybrid_action"] = hybrid_decision
                meta["hybrid_action_idx"] = int(hybrid_meta.get("action_idx", 0) or 0)
                score_logw = float(hybrid_meta.get("score", 0.0) or 0.0)
                current_capital = ctx_final.get("equity", ctx_final.get("balance", 1.0) or 1.0)
                try:
                    current_capital = float(current_capital)
                except Exception:
                    current_capital = 1.0
                score_growth = float(score_logw - math.log(max(current_capital, 1e-9)))
                meta["hybrid_score_logw"] = score_logw
                meta["hybrid_score"] = score_growth
                # Directional proxies from LSM (if available)
                try:
                    raw = hybrid_meta.get("raw") if isinstance(hybrid_meta, dict) else None
                    if isinstance(raw, dict):
                        exp_vals = (raw.get("debug_lsm") or {}).get("exp_vals")
                        if exp_vals is not None:
                            if hasattr(exp_vals, "detach"):
                                exp_list = exp_vals.detach().cpu().tolist()
                            else:
                                exp_list = list(exp_vals)
                            if len(exp_list) > 1:
                                meta["hybrid_score_long"] = float(exp_list[1])
                            if len(exp_list) > 2:
                                meta["hybrid_score_short"] = float(exp_list[2])
                except Exception:
                    pass
                res["hybrid_score"] = score_growth
                res["hybrid_score_logw"] = score_logw
                if "hybrid_score_long" in meta:
                    res["hybrid_score_long"] = meta.get("hybrid_score_long")
                if "hybrid_score_short" in meta:
                    res["hybrid_score_short"] = meta.get("hybrid_score_short")
                res["meta"] = meta
                res["reason"] = f"{res['reason']} | HYBRID"
            except Exception as e:
                logger.warning(f"[HYBRID] {symbol} | fallback to default action: {e}")

        if MC_VERBOSE_PRINT:
            logger.info(f"[DECIDE_FINAL] {symbol} | action={res['action']} ev={res['ev']:.6f} reason={res['reason']}")
            print(f"[DECIDE_FINAL] {symbol} | action={res['action']} ev={res['ev']:.6f} reason={res['reason']}")
        return res

    def _decide_hybrid_only(
        self,
        ctx: Dict[str, Any],
        params,
        seed: int,
        symbol: str,
        price: float,
        regime_ctx: str,
        boost_val: float,
    ) -> Dict[str, Any]:
        # Optionally compute leverage before simulation (Kelly-style via leverage optimizer).
        ctx_for_sim = ctx
        optimal_leverage = None
        mc_vol_step = None
        mc_vol_ann = None
        try:
            use_kelly_lev = str(os.environ.get("HYBRID_USE_KELLY_LEV", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            use_kelly_lev = True
        if use_kelly_lev:
            try:
                mu_annual = float(ctx.get("mu_base", ctx.get("mu_sim", 0.0)) or 0.0)
            except Exception:
                mu_annual = 0.0
            try:
                sigma_annual = float(ctx.get("sigma", ctx.get("sigma_sim", 0.0)) or 0.0)
            except Exception:
                sigma_annual = 0.0
            # Horizon seconds for leverage optimizer
            h_steps = 0
            try:
                h_env = os.environ.get("MC_HYBRID_HORIZON_STEPS")
                if h_env is not None:
                    h_steps = int(h_env)
            except Exception:
                h_steps = 0
            if h_steps <= 0:
                try:
                    h_steps = int(getattr(getattr(self, "hybrid_planner", None), "cfg", None).lsm_horizon_steps)
                except Exception:
                    h_steps = 0
            try:
                step_env = os.environ.get("MC_HYBRID_TIME_STEP_SEC")
                if step_env is not None:
                    step_sec_lev = float(step_env)
                else:
                    step_sec_lev = float(ctx.get("bar_seconds", getattr(self, "time_step_sec", 1)) or 1)
            except Exception:
                step_sec_lev = float(ctx.get("bar_seconds", getattr(self, "time_step_sec", 1)) or 1)
            step_sec_lev = float(max(1.0, step_sec_lev))
            horizon_sec = int(max(1.0, float(max(1, h_steps)) * step_sec_lev))

            # Fee proxy (roundtrip)
            try:
                base_fee = float(os.environ.get("HYBRID_BASE_FEE_RATE", getattr(self, "fee_roundtrip_base", 0.0002)) or 0.0002)
            except Exception:
                base_fee = float(getattr(self, "fee_roundtrip_base", 0.0002) or 0.0002)
            try:
                slippage_bps = float(os.environ.get("HYBRID_SLIPPAGE_BPS", 0.0) or 0.0)
            except Exception:
                slippage_bps = 0.0
            fee_roundtrip = float(2.0 * (base_fee + slippage_bps * 1e-4))

            try:
                max_leverage = float(ctx.get("max_leverage", os.environ.get("MAX_LEVERAGE", 100.0) or 100.0) or 100.0)
            except Exception:
                max_leverage = 100.0
            try:
                no_cap = str(os.environ.get("HYBRID_LEV_UNCAPPED", "0")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                no_cap = False

            _maybe_load_leverage_optimizer()
            if _FIND_OPTIMAL_LEVERAGE is not None:
                try:
                    lev_opt, _ = _FIND_OPTIMAL_LEVERAGE(
                        mu_annual=mu_annual,
                        sigma_annual=sigma_annual,
                        horizon_sec=horizon_sec,
                        fee_base=fee_roundtrip,
                        use_gpu=USE_GPU_LEVERAGE_OPT,
                    )
                    optimal_leverage = float(lev_opt)
                except Exception:
                    optimal_leverage = None
            if optimal_leverage is None:
                try:
                    optimal_leverage = float(ctx.get("leverage", 1.0) or 1.0)
                except Exception:
                    optimal_leverage = 1.0
            if not no_cap:
                optimal_leverage = float(max(1.0, min(max_leverage, optimal_leverage)))
            try:
                min_lev_env = float(os.environ.get("LEVERAGE_MIN", 1.0) or 1.0)
            except Exception:
                min_lev_env = 1.0
            if max_leverage and min_lev_env > float(max_leverage):
                min_lev_env = float(max_leverage)
            if min_lev_env > 0:
                optimal_leverage = float(max(min_lev_env, optimal_leverage))

        # Precompute liquidation params (reused by leverage sweep)
        try:
            maint_rate = float(os.environ.get("MAINT_MARGIN_RATE", 0.005) or 0.005)
        except Exception:
            maint_rate = 0.005
        try:
            liq_buf = float(os.environ.get("LIQUIDATION_BUFFER", 0.0025) or 0.0025)
        except Exception:
            liq_buf = 0.0025

        liq_prob_long = None
        liq_prob_short = None
        liq_price_long = None
        liq_price_short = None
        liq_from_sweep = False
        lev_sweep_meta = None
        mc_paths = None
        hybrid_decision = None
        hybrid_meta = None

        def _calc_liq_probs(price_paths_local, lev_for_liq: float):
            lp_long = None
            lp_short = None
            lp_price_long = None
            lp_price_short = None
            try:
                if lev_for_liq > 0 and price > 0:
                    barrier = (1.0 / lev_for_liq) + float(maint_rate) + float(liq_buf)
                    lp_price_long = float(price) * float(math.exp(-barrier))
                    lp_price_short = float(price) * float(math.exp(barrier))
                    if price_paths_local is not None:
                        if hasattr(price_paths_local, "device") and hasattr(price_paths_local, "dtype"):
                            import torch
                            p = price_paths_local
                            if hasattr(p, "ndim") and int(p.ndim) == 2:
                                lp_long = float((p <= lp_price_long).any(dim=1).float().mean().item())
                                lp_short = float((p >= lp_price_short).any(dim=1).float().mean().item())
                        else:
                            p = np.asarray(price_paths_local)
                            if p.ndim == 2:
                                lp_long = float((p <= lp_price_long).any(axis=1).mean())
                                lp_short = float((p >= lp_price_short).any(axis=1).mean())
            except Exception:
                pass
            return lp_long, lp_short, lp_price_long, lp_price_short

        # Optional: leverage sweep with liquidation probability constraint
        try:
            sweep_on = str(os.environ.get("HYBRID_LEV_SWEEP", "0")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            sweep_on = False
        try:
            force_lev = str(os.environ.get("HYBRID_FORCE_LEVERAGE", "0")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            force_lev = False
        if force_lev:
            sweep_on = False

        if sweep_on:
            raw_cands = os.environ.get("HYBRID_LEV_CANDIDATES", os.environ.get("LEVERAGE_CANDIDATES", ""))
            candidates = []
            if raw_cands:
                for part in str(raw_cands).replace(";", ",").split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if ".." in part or "-" in part:
                        sep = ".." if ".." in part else "-"
                        try:
                            a, b = part.split(sep, 1)
                        except Exception:
                            a, b = None, None
                        if a is None or b is None:
                            continue
                        try:
                            start = float(a.strip())
                            end = float(b.strip())
                        except Exception:
                            continue
                        if start <= 0 or end <= 0:
                            continue
                        if end < start:
                            start, end = end, start
                        start_i = int(math.floor(start))
                        end_i = int(math.floor(end))
                        for v in range(start_i, end_i + 1):
                            if v > 0:
                                candidates.append(float(v))
                        continue
                    try:
                        candidates.append(float(part))
                    except Exception:
                        continue
            if not candidates:
                candidates = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100]
            try:
                max_lev_env = float(ctx.get("max_leverage", os.environ.get("MAX_LEVERAGE", 100.0) or 100.0) or 100.0)
            except Exception:
                max_lev_env = 100.0
            try:
                max_lev_env = min(max_lev_env, float(os.environ.get("HYBRID_LEV_MAX", max_lev_env) or max_lev_env))
            except Exception:
                pass
            try:
                min_lev_env = float(os.environ.get("LEVERAGE_MIN", 1.0) or 1.0)
            except Exception:
                min_lev_env = 1.0
            try:
                min_lev_env = max(min_lev_env, float(os.environ.get("HYBRID_LEV_MIN", min_lev_env) or min_lev_env))
            except Exception:
                pass

            filt = []
            for v in candidates:
                try:
                    lv = float(v)
                except Exception:
                    continue
                if lv <= 0:
                    continue
                if max_lev_env > 0:
                    lv = min(lv, max_lev_env)
                if min_lev_env > 0:
                    lv = max(lv, min_lev_env)
                filt.append(lv)
            candidates = sorted(set(filt))
            if not candidates:
                candidates = [min_lev_env if min_lev_env > 0 else 1.0]

            # Prepare annual stats for Kelly/GBM windows
            try:
                mu_annual = float(ctx.get("mu_base", ctx.get("mu_sim", 0.0)) or 0.0)
            except Exception:
                mu_annual = 0.0
            try:
                sigma_annual = float(ctx.get("sigma", ctx.get("sigma_sim", 0.0)) or 0.0)
            except Exception:
                sigma_annual = 0.0

            # Optional: narrow candidates around Kelly estimate
            try:
                kelly_window_on = str(os.environ.get("HYBRID_LEV_KELLY_WINDOW", "0")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                kelly_window_on = False
            kelly_est = None
            if kelly_window_on:
                try:
                    sig2 = float(sigma_annual) ** 2
                except Exception:
                    sig2 = 0.0
                if sig2 > 0:
                    try:
                        kelly_est = float(mu_annual) / float(sig2)
                    except Exception:
                        kelly_est = None
                if kelly_est is not None and math.isfinite(kelly_est):
                    try:
                        k_low = float(os.environ.get("HYBRID_LEV_KELLY_LOW", 0.5) or 0.5)
                        k_high = float(os.environ.get("HYBRID_LEV_KELLY_HIGH", 1.5) or 1.5)
                    except Exception:
                        k_low, k_high = 0.5, 1.5
                    if k_high < k_low:
                        k_low, k_high = k_high, k_low
                    kelly_est = max(min_lev_env, min(max_lev_env, kelly_est))
                    low_b = max(min_lev_env, kelly_est * k_low)
                    high_b = max(min_lev_env, kelly_est * k_high)
                    if high_b < low_b:
                        low_b, high_b = high_b, low_b
                    cand_k = [c for c in candidates if low_b <= c <= high_b]
                    if cand_k:
                        candidates = cand_k

            # Optional: compute GBM closed-form estimate (init-only or windowing)
            try:
                gbm_window_on = str(os.environ.get("HYBRID_LEV_GBM_WINDOW", "0")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                gbm_window_on = False
            try:
                gbm_init_on = str(os.environ.get("HYBRID_LEV_GBM_INIT", "0")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                gbm_init_on = False
            gbm_est = None
            if gbm_window_on or gbm_init_on:
                try:
                    sig2 = float(sigma_annual) ** 2
                except Exception:
                    sig2 = 0.0
                if sig2 > 0:
                    # Approximate annualized cost from roundtrip fee
                    try:
                        base_fee = float(os.environ.get("HYBRID_BASE_FEE_RATE", getattr(self, "fee_roundtrip_base", 0.0002)) or 0.0002)
                    except Exception:
                        base_fee = float(getattr(self, "fee_roundtrip_base", 0.0002) or 0.0002)
                    try:
                        slippage_bps = float(os.environ.get("HYBRID_SLIPPAGE_BPS", 0.0) or 0.0)
                    except Exception:
                        slippage_bps = 0.0
                    fee_roundtrip = float(2.0 * (base_fee + slippage_bps * 1e-4))
                    # Estimate horizon in seconds (for fee annualization)
                    try:
                        h_steps = int(os.environ.get("MC_HYBRID_HORIZON_STEPS", 0) or 0)
                    except Exception:
                        h_steps = 0
                    if h_steps <= 0:
                        try:
                            h_steps = int(getattr(getattr(self, "hybrid_planner", None), "cfg", None).lsm_horizon_steps)
                        except Exception:
                            h_steps = 0
                    try:
                        step_env = os.environ.get("MC_HYBRID_TIME_STEP_SEC")
                        if step_env is not None:
                            step_sec_est = float(step_env)
                        else:
                            step_sec_est = float(ctx.get("bar_seconds", getattr(self, "time_step_sec", 1)) or 1)
                    except Exception:
                        step_sec_est = float(ctx.get("bar_seconds", getattr(self, "time_step_sec", 1)) or 1)
                    step_sec_est = float(max(1.0, step_sec_est))
                    horizon_sec_est = int(max(1.0, float(max(1, h_steps)) * step_sec_est))
                    cost_annual = float(fee_roundtrip) * (SECONDS_PER_YEAR / float(max(1, horizon_sec_est)))
                    try:
                        rf = float(os.environ.get("HYBRID_RISK_FREE_RATE", 0.0) or 0.0)
                    except Exception:
                        rf = 0.0
                    try:
                        gbm_est = (float(mu_annual) - float(rf) - float(cost_annual)) / float(sig2)
                    except Exception:
                        gbm_est = None
                if gbm_est is not None and math.isfinite(gbm_est):
                    gbm_est = max(min_lev_env, min(max_lev_env, float(gbm_est)))
                    try:
                        g_low = float(os.environ.get("HYBRID_LEV_GBM_LOW", 0.5) or 0.5)
                        g_high = float(os.environ.get("HYBRID_LEV_GBM_HIGH", 1.5) or 1.5)
                    except Exception:
                        g_low, g_high = 0.5, 1.5
                    if g_high < g_low:
                        g_low, g_high = g_high, g_low
                    low_b = max(min_lev_env, gbm_est * g_low)
                    high_b = max(min_lev_env, gbm_est * g_high)
                    if high_b < low_b:
                        low_b, high_b = high_b, low_b
                    if gbm_window_on:
                        cand_g = [c for c in candidates if low_b <= c <= high_b]
                        if cand_g:
                            candidates = cand_g

            try:
                max_liq = float(os.environ.get("HYBRID_LIQ_PROB_MAX", 0.05) or 0.05)
            except Exception:
                max_liq = 0.05

            # Build paths once (price paths are leverage-invariant)
            ctx_for_sim = dict(ctx)
            if "leverage" not in ctx_for_sim:
                ctx_for_sim["leverage"] = float(min_lev_env if min_lev_env > 0 else 1.0)
            mc_paths = self._build_hybrid_mc_paths(ctx_for_sim, params, seed)
            price_paths = mc_paths.get("prices")
            torch_mod = getattr(self, "_torch", None)
            if torch_mod is None or price_paths is None:
                mc_paths = None
            else:
                p = price_paths
                if not isinstance(p, torch_mod.Tensor):
                    p = torch_mod.as_tensor(p, device=self.device, dtype=torch_mod.float32)
                # Simple returns for liquidation checks
                p0 = p[:, [0]].clamp(min=1e-12)
                rel = (p / p0) - 1.0
                min_ret = torch_mod.min(rel, dim=1).values
                max_ret = torch_mod.max(rel, dim=1).values
                final_ret = rel[:, -1]
                mc_vol_step = None
                mc_vol_ann = None
                try:
                    lr = torch_mod.log(p[:, 1:].clamp(min=1e-12) / p[:, :-1].clamp(min=1e-12))
                    if lr.numel() > 0:
                        vol_path = torch_mod.std(lr, dim=1, unbiased=False)
                        mc_vol_step = float(vol_path.mean().item())
                        dt_used = mc_paths.get("dt_used") if isinstance(mc_paths, dict) else None
                        try:
                            dt_used = float(dt_used) if dt_used is not None else None
                        except Exception:
                            dt_used = None
                        if dt_used is None or dt_used <= 0:
                            try:
                                step_sec_f = float(mc_paths.get("time_step_sec", 0.0) or 0.0)
                            except Exception:
                                step_sec_f = 0.0
                            dt_used = float(step_sec_f) / float(SECONDS_PER_YEAR) if step_sec_f > 0 else None
                        if dt_used is not None and dt_used > 0:
                            mc_vol_ann = float(mc_vol_step) / float(math.sqrt(dt_used))
                except Exception:
                    mc_vol_step = None
                    mc_vol_ann = None

                lev_t = torch_mod.tensor(candidates, device=p.device, dtype=torch_mod.float32)
                barrier = (1.0 / lev_t) + float(maint_rate) + float(liq_buf)
                thr_long = torch_mod.exp(-barrier) - 1.0
                thr_short = torch_mod.exp(barrier) - 1.0

                min_ret_e = min_ret.unsqueeze(1)
                max_ret_e = max_ret.unsqueeze(1)
                final_ret_e = final_ret.unsqueeze(1)
                lev_e = lev_t.unsqueeze(0)

                liq_long_mask = min_ret_e <= thr_long.unsqueeze(0)
                liq_short_mask = max_ret_e >= thr_short.unsqueeze(0)
                liq_prob_long_vec = liq_long_mask.float().mean(dim=0)
                liq_prob_short_vec = liq_short_mask.float().mean(dim=0)

                pnl_long = final_ret_e * lev_e - float(fee_roundtrip) * lev_e
                pnl_short = (-final_ret_e) * lev_e - float(fee_roundtrip) * lev_e
                pnl_long = torch_mod.where(liq_long_mask, torch_mod.full_like(pnl_long, -1.0), pnl_long)
                pnl_short = torch_mod.where(liq_short_mask, torch_mod.full_like(pnl_short, -1.0), pnl_short)

                ev_long = pnl_long.mean(dim=0)
                ev_short = pnl_short.mean(dim=0)
                best_ev = torch_mod.maximum(ev_long, ev_short)
                best_dir = torch_mod.where(ev_long >= ev_short, 1.0, -1.0)

                # --- Trend strength bonuses (combine 1,2,3) ---
                trend_meta = {
                    "mc": None,
                    "hyb": None,
                    "ev_cvar": None,
                    "combined": None,
                    "w": None,
                }
                try:
                    trend_on = str(os.environ.get("HYBRID_LEV_TREND_BONUS", "1")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    trend_on = True
                best_ev_adj = best_ev
                if trend_on:
                    # 1) MC trend: |mean(final_ret)| / std(final_ret)
                    try:
                        mu_ret = torch_mod.mean(final_ret)
                        std_ret = torch_mod.std(final_ret)
                        mc_trend = float(torch_mod.abs(mu_ret).item()) / float(max(std_ret.item(), 1e-9))
                    except Exception:
                        mc_trend = 0.0
                    # 2) Hybrid direction: (hyb_long - hyb_short)
                    try:
                        hyb_long = float(ctx.get("hybrid_score_long") or ctx.get("score_long") or 0.0)
                        hyb_short = float(ctx.get("hybrid_score_short") or ctx.get("score_short") or 0.0)
                        hyb_trend = float(hyb_long - hyb_short)
                    except Exception:
                        hyb_trend = 0.0
                    # 3) Event EV / CVaR ratio
                    try:
                        ev_r = float(ctx.get("event_ev_r") or 0.0)
                        cvar_r = float(ctx.get("event_cvar_r") or 0.0)
                        ev_cvar = float(ev_r) / float(abs(cvar_r) + 1e-9)
                    except Exception:
                        ev_cvar = 0.0

                    try:
                        w1 = float(os.environ.get("HYBRID_LEV_TREND_W1", 1.0) or 1.0)
                        w2 = float(os.environ.get("HYBRID_LEV_TREND_W2", 1.0) or 1.0)
                        w3 = float(os.environ.get("HYBRID_LEV_TREND_W3", 1.0) or 1.0)
                    except Exception:
                        w1 = w2 = w3 = 1.0
                    trend_score = (w1 * mc_trend) + (w2 * hyb_trend) + (w3 * ev_cvar)
                    try:
                        trend_strength = float(os.environ.get("HYBRID_LEV_TREND_STRENGTH", 0.01) or 0.01)
                    except Exception:
                        trend_strength = 0.01
                    lev_ratio = lev_t / max(float(max_lev_env or 1.0), 1e-6)
                    best_ev_adj = best_ev + (float(trend_strength) * float(trend_score) * lev_ratio)
                    trend_meta = {
                        "mc": float(mc_trend),
                        "hyb": float(hyb_trend),
                        "ev_cvar": float(ev_cvar),
                        "combined": float(trend_score),
                        "w": [float(w1), float(w2), float(w3)],
                        "strength": float(trend_strength),
                    }

                # --- MC future volatility bonus (from simulated paths) ---
                mc_future_vol_meta = None
                try:
                    future_vol_on = str(os.environ.get("MC_FUTURE_VOL_BONUS", "0")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    future_vol_on = False
                if future_vol_on and mc_vol_step is not None:
                    try:
                        vol_ref = float(os.environ.get("MC_FUTURE_VOL_REF", 0.002) or 0.002)
                    except Exception:
                        vol_ref = 0.002
                    try:
                        vol_k = float(os.environ.get("MC_FUTURE_VOL_K", 0.002) or 0.002)
                    except Exception:
                        vol_k = 0.002
                    try:
                        vol_max = float(os.environ.get("MC_FUTURE_VOL_MAX", 0.01) or 0.01)
                    except Exception:
                        vol_max = 0.01
                    bonus_ratio = 0.0
                    if vol_ref > 0:
                        bonus_ratio = max(0.0, float(mc_vol_step) / float(vol_ref) - 1.0)
                    bonus = float(min(float(vol_max), float(vol_k) * float(bonus_ratio)))
                    if bonus > 0:
                        lev_ratio = lev_t / max(float(max_lev_env or 1.0), 1e-6)
                        best_ev_adj = best_ev_adj + float(bonus) * lev_ratio
                    mc_future_vol_meta = {
                        "enabled": True,
                        "mc_vol_step": float(mc_vol_step),
                        "mc_vol_ann": float(mc_vol_ann) if mc_vol_ann is not None else None,
                        "vol_ref": float(vol_ref),
                        "bonus": float(bonus),
                        "k": float(vol_k),
                        "max": float(vol_max),
                    }

                # --- Bad volatility penalty (downside risk) ---
                bad_vol_meta = None
                try:
                    bad_vol_on = str(os.environ.get("HYBRID_LEV_BADVOL_PENALTY", "1")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    bad_vol_on = True
                if bad_vol_on:
                    try:
                        max_p_sl = float(os.environ.get("HYBRID_LEV_BADVOL_PSL_REF", 0.6) or 0.6)
                    except Exception:
                        max_p_sl = 0.6
                    try:
                        bad_strength = float(os.environ.get("HYBRID_LEV_BADVOL_STRENGTH", 0.01) or 0.01)
                    except Exception:
                        bad_strength = 0.01
                    try:
                        ev_p_sl = float(ctx.get("event_p_sl") or 0.0)
                    except Exception:
                        ev_p_sl = 0.0
                    bad_ratio = 0.0
                    if max_p_sl > 0:
                        bad_ratio = max(0.0, (ev_p_sl - max_p_sl) / max_p_sl)
                    if bad_strength > 0 and bad_ratio > 0:
                        lev_ratio = lev_t / max(float(max_lev_env or 1.0), 1e-6)
                        best_ev_adj = best_ev_adj - (float(bad_strength) * float(bad_ratio) * lev_ratio)
                    bad_vol_meta = {
                        "enabled": True,
                        "event_p_sl": float(ev_p_sl),
                        "ref": float(max_p_sl),
                        "ratio": float(bad_ratio),
                        "strength": float(bad_strength),
                    }

                # --- Optional: penalize low-volatility symbols when selecting leverage
                try:
                    vol_pen_on = str(os.environ.get("HYBRID_LEV_VOL_PENALTY", "0")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    vol_pen_on = False
                vol_pen_meta = None
                if vol_pen_on:
                    try:
                        use_mc_vol = str(os.environ.get("HYBRID_LEV_VOL_PENALTY_USE_MC", "0")).strip().lower() in ("1", "true", "yes", "on")
                    except Exception:
                        use_mc_vol = False
                    try:
                        vol_ref = float(os.environ.get("HYBRID_LEV_VOL_PENALTY_REF", os.environ.get("HYBRID_VOL_REF", 0.6) or 0.6) or 0.6)
                    except Exception:
                        vol_ref = 0.6
                    try:
                        vol_gamma = float(os.environ.get("HYBRID_LEV_VOL_PENALTY_GAMMA", 1.0) or 1.0)
                    except Exception:
                        vol_gamma = 1.0
                    try:
                        vol_strength = float(os.environ.get("HYBRID_LEV_VOL_PENALTY_STRENGTH", 0.005) or 0.005)
                    except Exception:
                        vol_strength = 0.005
                    try:
                        if use_mc_vol and mc_vol_ann is not None:
                            sigma_safe = float(mc_vol_ann)
                        else:
                            sigma_safe = float(sigma_annual)
                    except Exception:
                        sigma_safe = 0.0
                    sigma_safe = max(0.0, sigma_safe)
                    vol_ref_safe = max(float(vol_ref), 1e-6)
                    low_vol = max(0.0, (vol_ref_safe - sigma_safe) / vol_ref_safe)
                    if vol_gamma and vol_gamma != 1.0:
                        try:
                            low_vol = float(low_vol) ** float(vol_gamma)
                        except Exception:
                            pass
                    if vol_strength > 0 and low_vol > 0:
                        lev_ratio = lev_t / max(float(max_lev_env or 1.0), 1e-6)
                        penalty = float(vol_strength) * float(low_vol) * lev_ratio
                        best_ev_adj = best_ev_adj - penalty
                    vol_pen_meta = {
                        "enabled": True,
                        "sigma_annual": float(sigma_safe),
                        "vol_ref": float(vol_ref),
                        "low_vol": float(low_vol),
                        "strength": float(vol_strength),
                        "gamma": float(vol_gamma),
                    }

                liq_best = torch_mod.where(best_dir > 0, liq_prob_long_vec, liq_prob_short_vec)
                # --- Optional: relax liquidation threshold on strong trend ---
                try:
                    liq_relax_on = str(os.environ.get("HYBRID_LIQ_TREND_RELAX", "1")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    liq_relax_on = True
                max_liq_eff = float(max_liq)
                if liq_relax_on and trend_meta.get("combined") is not None:
                    try:
                        relax_k = float(os.environ.get("HYBRID_LIQ_TREND_RELAX_K", 0.5) or 0.5)
                    except Exception:
                        relax_k = 0.5
                    max_liq_eff = float(max_liq) * float(1.0 + max(0.0, float(trend_meta.get("combined") or 0.0)) * float(relax_k))
                ok_mask = liq_best <= float(max_liq_eff)

                passed = int(ok_mask.sum().item()) if ok_mask.numel() else 0
                if ok_mask.any():
                    neg_inf = torch_mod.tensor(-1e18, device=p.device, dtype=torch_mod.float32)
                    masked_ev = torch_mod.where(ok_mask, best_ev_adj, neg_inf)
                    idx = int(torch_mod.argmax(masked_ev).item())
                else:
                    idx = int(torch_mod.argmax(best_ev_adj).item())

                lev_sel = float(lev_t[idx].item())
                optimal_leverage = float(lev_sel)
                ctx_for_sim["leverage"] = float(lev_sel)

                liq_prob_long = float(liq_prob_long_vec[idx].item())
                liq_prob_short = float(liq_prob_short_vec[idx].item())
                liq_price_long = float(price) * float(math.exp(-float(barrier[idx].item())))
                liq_price_short = float(price) * float(math.exp(float(barrier[idx].item())))

                # Override exposure for planner with selected leverage
                try:
                    size_frac = float(mc_paths.get("size_frac", ctx_for_sim.get("size_frac", 1.0)) or 1.0)
                except Exception:
                    size_frac = 1.0
                try:
                    max_exposure = float(os.environ.get("HYBRID_MAX_EXPOSURE", 5.0) or 5.0)
                except Exception:
                    max_exposure = 5.0
                exposure = max(0.0, float(lev_sel) * float(size_frac))
                if max_exposure > 0:
                    exposure = min(exposure, max_exposure)
                mc_paths["exposure"] = exposure
                mc_paths["leverage_used"] = float(lev_sel)
                mc_paths["size_frac"] = float(size_frac)

                hybrid_decision, hybrid_meta = self.decide_with_hybrid_planner(
                    symbol, ctx_for_sim, mc_paths, return_detail=True
                )
                liq_from_sweep = True
                lev_sweep_meta = {
                    "enabled": True,
                    "candidates": [float(x) for x in candidates],
                    "max_liq_prob": float(max_liq),
                    "passed": int(passed),
                    "selected": float(lev_sel),
                    "kelly_est": float(kelly_est) if kelly_est is not None and math.isfinite(kelly_est) else None,
                    "gbm_est": float(gbm_est) if gbm_est is not None and math.isfinite(gbm_est) else None,
                }
                lev_sweep_meta["max_liq_eff"] = float(max_liq_eff)
                lev_sweep_meta["trend"] = trend_meta
                if bad_vol_meta is not None:
                    lev_sweep_meta["bad_vol"] = bad_vol_meta
                if vol_pen_meta is not None:
                    lev_sweep_meta["vol_penalty"] = vol_pen_meta
                if vol_pen_meta is not None:
                    lev_sweep_meta["vol_penalty"] = vol_pen_meta
                if mc_future_vol_meta is not None:
                    lev_sweep_meta["future_vol"] = mc_future_vol_meta
        if mc_paths is None:
            if optimal_leverage is not None:
                ctx_for_sim = dict(ctx)
                ctx_for_sim["leverage"] = float(optimal_leverage)

            # Build hybrid MC inputs + run planner
            mc_paths = self._build_hybrid_mc_paths(ctx_for_sim, params, seed)
            hybrid_decision, hybrid_meta = self.decide_with_hybrid_planner(
                symbol, ctx_for_sim, mc_paths, return_detail=True
            )

        # Normalize action labels for engine compatibility
        action = "WAIT" if hybrid_decision == "HOLD" else hybrid_decision
        action_idx = int(hybrid_meta.get("action_idx", 0) or 0)

        # --- Liquidation-aware diagnostics (maintenance margin proxy) ---
        if not liq_from_sweep:
            try:
                lev_for_liq = float(ctx_for_sim.get("leverage", ctx.get("leverage", 1.0)) or 1.0)
            except Exception:
                lev_for_liq = 1.0
            try:
                if lev_for_liq > 0 and price > 0:
                    barrier = (1.0 / lev_for_liq) + float(maint_rate) + float(liq_buf)
                    liq_price_long = float(price) * float(math.exp(-barrier))
                    liq_price_short = float(price) * float(math.exp(barrier))
                    price_paths = mc_paths.get("prices")
                    if price_paths is not None:
                        # torch path (preferred)
                        if hasattr(price_paths, "device") and hasattr(price_paths, "dtype"):
                            import torch
                            p = price_paths
                            if hasattr(p, "ndim") and int(p.ndim) == 2:
                                liq_prob_long = float((p <= liq_price_long).any(dim=1).float().mean().item())
                                liq_prob_short = float((p >= liq_price_short).any(dim=1).float().mean().item())
                        else:
                            p = np.asarray(price_paths)
                            if p.ndim == 2:
                                liq_prob_long = float((p <= liq_price_long).any(axis=1).mean())
                                liq_prob_short = float((p >= liq_price_short).any(axis=1).mean())
            except Exception:
                pass

        # Score: convert logW -> log-growth (subtract log(capital))
        score_logw = float(hybrid_meta.get("score", 0.0) or 0.0)
        current_capital = ctx.get("equity", ctx.get("balance", 1.0) or 1.0)
        try:
            current_capital = float(current_capital)
        except Exception:
            current_capital = 1.0
        log_cap = math.log(max(current_capital, 1e-9))
        score = float(score_logw - log_cap)
        score_raw = score
        score_steps = int(hybrid_meta.get("horizon_steps", 0) or 0) if isinstance(hybrid_meta, dict) else 0
        per_step = str(os.environ.get("HYBRID_SCORE_PER_STEP", "0")).strip().lower() in ("1", "true", "yes", "on")
        scale_mode = str(os.environ.get("HYBRID_SCORE_TIME_SCALE", "")).strip().lower()
        if not scale_mode:
            scale_mode = "per_step" if per_step else "raw"
        step_sec = None
        if isinstance(hybrid_meta, dict):
            step_sec = hybrid_meta.get("time_step_sec") or hybrid_meta.get("time_step_seconds")
        if step_sec is None and isinstance(ctx, dict):
            step_sec = ctx.get("time_step_sec") or ctx.get("bar_seconds") or ctx.get("timeframe_sec")
        try:
            step_sec = float(step_sec)
        except Exception:
            step_sec = float(getattr(self, "time_step_sec", 1) or 1)
        step_sec = float(max(1.0, step_sec))
        total_seconds = float(score_steps) * step_sec if score_steps > 0 else None

        # Use LSM exp_vals as directional scores (proxy)
        score_long = None
        score_short = None
        score_hold = None
        exp_list = None
        raw = hybrid_meta.get("raw") if isinstance(hybrid_meta, dict) else None
        if isinstance(raw, dict):
            debug_lsm = raw.get("debug_lsm") or {}
            exp_vals = debug_lsm.get("exp_vals")
            try:
                if exp_vals is not None:
                    if hasattr(exp_vals, "detach"):
                        exp_list = exp_vals.detach().cpu().tolist()
                    else:
                        exp_list = list(exp_vals)
                    if len(exp_list) > 1:
                        score_long = float(exp_list[1])
                    if len(exp_list) > 2:
                        score_short = float(exp_list[2])
                    pos_side = ctx.get("position_side", ctx.get("position", 0))
                    pos_idx = 0
                    if isinstance(pos_side, str):
                        if pos_side.upper() == "LONG":
                            pos_idx = 1
                        elif pos_side.upper() == "SHORT":
                            pos_idx = 2
                    else:
                        try:
                            pos_idx = 1 if float(pos_side) > 0 else (2 if float(pos_side) < 0 else 0)
                        except Exception:
                            pos_idx = 0
                    if len(exp_list) > pos_idx:
                        score_hold = float(exp_list[pos_idx])
            except Exception:
                score_long = None
                score_short = None
                score_hold = None
                exp_list = None

        # Optional: use exp_vals-based scoring instead of beam score
        score_source = str(os.environ.get("HYBRID_SCORE_SOURCE", "beam")).strip().lower()
        if score_source in ("exp", "exp_vals", "lsm", "exp_diff", "exp_delta") and exp_list:
            try:
                if score_source in ("exp_diff", "exp_delta"):
                    # Compare best action vs current position (cash baseline)
                    curr_pos_idx = 0
                    pos_side = ctx.get("position_side", ctx.get("position", 0))
                    if isinstance(pos_side, str):
                        if pos_side.upper() == "LONG":
                            curr_pos_idx = 1
                        elif pos_side.upper() == "SHORT":
                            curr_pos_idx = 2
                    else:
                        try:
                            curr_pos_idx = 1 if float(pos_side) > 0 else (2 if float(pos_side) < 0 else 0)
                        except Exception:
                            curr_pos_idx = 0
                    curr_pos_idx = int(max(0, min(curr_pos_idx, len(exp_list) - 1)))
                    action_idx_local = int(hybrid_meta.get("action_idx", 0) or 0)
                    action_idx_local = int(max(0, min(action_idx_local, len(exp_list) - 1)))
                    base_score = float(exp_list[action_idx_local]) - float(exp_list[curr_pos_idx])
                else:
                    action_idx_local = int(hybrid_meta.get("action_idx", 0) or 0)
                    action_idx_local = int(max(0, min(action_idx_local, len(exp_list) - 1)))
                    base_score = float(exp_list[action_idx_local])
                score_raw = float(base_score)
                score = float(base_score)
            except Exception:
                # fallback to beam-based score
                score_source = "beam"

        # Precompute impulse/momentum signals for entry bias/bonus
        try:
            momentum_z = float(ctx.get("momentum_z", 0.0) or 0.0)
        except Exception:
            momentum_z = 0.0
        try:
            impulse_score = float(ctx.get("impulse_score", 0.0) or 0.0)
        except Exception:
            impulse_score = 0.0
        try:
            impulse_dir = int(ctx.get("impulse_dir", 0) or 0)
        except Exception:
            impulse_dir = 0
        impulse_active = bool(ctx.get("impulse_active", False)) or (impulse_score > 0.0 and impulse_dir != 0)

        try:
            rt_score = float(ctx.get("rt_breakout_score", 0.0) or 0.0)
        except Exception:
            rt_score = 0.0
        try:
            rt_dir = int(ctx.get("rt_breakout_dir", 0) or 0)
        except Exception:
            rt_dir = 0
        rt_active = bool(ctx.get("rt_breakout_active", False)) or (rt_score > 0.0 and rt_dir != 0)

        # Tick-level microstructure features (fast direction/volatility)
        try:
            tick_trend = float(ctx.get("tick_trend", 0.0) or 0.0)
        except Exception:
            tick_trend = 0.0
        try:
            tick_ret = float(ctx.get("tick_ret", 0.0) or 0.0)
        except Exception:
            tick_ret = 0.0
        try:
            tick_vol = float(ctx.get("tick_vol", 0.0) or 0.0)
        except Exception:
            tick_vol = 0.0
        try:
            tick_dir = int(ctx.get("tick_dir", 0) or 0)
        except Exception:
            tick_dir = 0
        try:
            tick_breakout_score = float(ctx.get("tick_breakout_score", 0.0) or 0.0)
        except Exception:
            tick_breakout_score = 0.0
        try:
            tick_breakout_dir = int(ctx.get("tick_breakout_dir", 0) or 0)
        except Exception:
            tick_breakout_dir = 0
        tick_breakout_active = bool(ctx.get("tick_breakout_active", False)) or (tick_breakout_score > 0.0 and tick_breakout_dir != 0)

        bias_eps_extra = 0.0
        try:
            impulse_bias = float(os.environ.get("IMPULSE_ENTRY_BIAS_EPS", 0.0) or 0.0)
        except Exception:
            impulse_bias = 0.0
        if impulse_active and impulse_bias > 0:
            bias_eps_extra += float(impulse_score) * float(impulse_bias)
        try:
            rt_bias = float(os.environ.get("REALTIME_BREAKOUT_BIAS_EPS", 0.0) or 0.0)
        except Exception:
            rt_bias = 0.0
        if rt_active and rt_bias > 0:
            bias_eps_extra += float(rt_score) * float(rt_bias)
        try:
            mom_bias_k = float(os.environ.get("HYBRID_MOMENTUM_BIAS_K", 0.0) or 0.0)
        except Exception:
            mom_bias_k = 0.0
        try:
            mom_bias_cap = float(os.environ.get("HYBRID_MOMENTUM_BIAS_CAP", 0.001) or 0.001)
        except Exception:
            mom_bias_cap = 0.001
        if mom_bias_k > 0 and momentum_z != 0.0:
            bias_eps_extra += min(float(mom_bias_cap), abs(float(momentum_z)) * float(mom_bias_k))
        try:
            tick_bias_k = float(os.environ.get("TICK_TREND_BIAS_K", 0.0) or 0.0)
        except Exception:
            tick_bias_k = 0.0
        try:
            tick_bias_cap = float(os.environ.get("TICK_TREND_BIAS_CAP", 0.001) or 0.001)
        except Exception:
            tick_bias_cap = 0.001
        if tick_bias_k > 0 and tick_trend != 0.0:
            bias_eps_extra += min(float(tick_bias_cap), abs(float(tick_trend)) * float(tick_bias_k))

        # Entry bias: if cash best but near-alternative within epsilon, prefer entry
        try:
            bias_eps = float(os.environ.get("HYBRID_ENTRY_BIAS_EPS", 0.0) or 0.0)
        except Exception:
            bias_eps = 0.0
        if bias_eps_extra > 0:
            bias_eps = float(bias_eps) + float(bias_eps_extra)
        biased_entry = False
        if bias_eps > 0 and exp_list and action_idx == 0 and len(exp_list) > 1:
            try:
                best_alt_idx = 1 + int(max(range(len(exp_list) - 1), key=lambda k: exp_list[k + 1]))
                if float(exp_list[best_alt_idx]) >= float(exp_list[0]) - bias_eps:
                    action_idx = int(best_alt_idx)
                    hybrid_decision = "LONG" if action_idx == 1 else "SHORT"
                    action = hybrid_decision
                    biased_entry = True
                    # Recompute score if using exp-based source
                    if score_source in ("exp", "exp_vals", "lsm", "exp_diff", "exp_delta"):
                        if score_source in ("exp_diff", "exp_delta"):
                            base_score = float(exp_list[action_idx]) - float(exp_list[0])
                        else:
                            base_score = float(exp_list[action_idx])
                        score_raw = float(base_score)
                        score = float(base_score)
            except Exception:
                pass

        # Apply score time scaling after any bias adjustments.
        if scale_mode == "per_step":
            if score_steps > 0:
                score = float(score_raw / max(1, score_steps))
            else:
                score = float(score_raw)
        elif scale_mode in ("per_sec", "per_second", "sec", "second"):
            if total_seconds and total_seconds > 0:
                score = float(score_raw / max(total_seconds, 1e-9))
            else:
                score = float(score_raw)
        elif scale_mode in ("per_min", "per_minute", "minute", "min"):
            if total_seconds and total_seconds > 0:
                score = float(score_raw / max(total_seconds / 60.0, 1e-9))
            else:
                score = float(score_raw)
        elif scale_mode in ("per_hour", "hour", "hr"):
            if total_seconds and total_seconds > 0:
                score = float(score_raw / max(total_seconds / 3600.0, 1e-9))
            else:
                score = float(score_raw)
        elif scale_mode in ("annual", "per_year", "year", "annualized"):
            if total_seconds and total_seconds > 0:
                years = float(total_seconds) / float(SECONDS_PER_YEAR)
                score = float(score_raw / max(years, 1e-12))
            else:
                score = float(score_raw)
        else:
            score = float(score_raw)

        # Apply liquidation penalty to score (optional)
        try:
            liq_penalty = float(os.environ.get("LIQUIDATION_SCORE_PENALTY", 0.0) or 0.0)
        except Exception:
            liq_penalty = 0.0
        if liq_penalty > 0:
            liq_prob = None
            if action == "LONG":
                liq_prob = liq_prob_long
            elif action == "SHORT":
                liq_prob = liq_prob_short
            if liq_prob is not None:
                score = float(score - (liq_penalty * float(liq_prob)))

        # Momentum/impulse bonus for early breakout capture
        impulse_bonus = 0.0
        momentum_bonus = 0.0
        rt_breakout_bonus = 0.0
        tick_bonus = 0.0
        tick_breakout_bonus = 0.0
        try:
            impulse_enabled = str(os.environ.get("IMPULSE_ENTRY_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            impulse_enabled = True
        try:
            rt_enabled = str(os.environ.get("REALTIME_BREAKOUT_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            rt_enabled = True
        try:
            impulse_boost = float(os.environ.get("IMPULSE_SCORE_BOOST", 0.0015) or 0.0015)
        except Exception:
            impulse_boost = 0.0015
        try:
            rt_boost = float(os.environ.get("REALTIME_BREAKOUT_SCORE_BOOST", 0.0020) or 0.0020)
        except Exception:
            rt_boost = 0.0020
        try:
            momentum_k = float(os.environ.get("HYBRID_MOMENTUM_BONUS_K", 0.001) or 0.001)
        except Exception:
            momentum_k = 0.001
        try:
            momentum_cap = float(os.environ.get("HYBRID_MOMENTUM_BONUS_CAP", 0.004) or 0.004)
        except Exception:
            momentum_cap = 0.004
        try:
            tick_bonus_k = float(os.environ.get("TICK_TREND_BONUS_K", 0.0006) or 0.0006)
        except Exception:
            tick_bonus_k = 0.0006
        try:
            tick_bonus_cap = float(os.environ.get("TICK_TREND_BONUS_CAP", 0.003) or 0.003)
        except Exception:
            tick_bonus_cap = 0.003
        try:
            tick_breakout_boost = float(os.environ.get("TICK_BREAKOUT_SCORE_BOOST", 0.0020) or 0.0020)
        except Exception:
            tick_breakout_boost = 0.0020

        if action in ("LONG", "SHORT"):
            if impulse_enabled and impulse_active and impulse_dir != 0:
                if (action == "LONG" and impulse_dir > 0) or (action == "SHORT" and impulse_dir < 0):
                    impulse_bonus = float(impulse_score) * float(impulse_boost)
            if rt_enabled and rt_active and rt_dir != 0:
                if (action == "LONG" and rt_dir > 0) or (action == "SHORT" and rt_dir < 0):
                    rt_breakout_bonus = float(rt_score) * float(rt_boost)
            if momentum_k > 0 and momentum_z != 0.0:
                if action == "LONG" and momentum_z > 0:
                    momentum_bonus = min(float(momentum_cap), float(momentum_z) * float(momentum_k))
                elif action == "SHORT" and momentum_z < 0:
                    momentum_bonus = min(float(momentum_cap), abs(float(momentum_z)) * float(momentum_k))
            if tick_bonus_k > 0 and tick_trend != 0.0:
                if action == "LONG" and tick_trend > 0:
                    tick_bonus = min(float(tick_bonus_cap), abs(float(tick_trend)) * float(tick_bonus_k))
                elif action == "SHORT" and tick_trend < 0:
                    tick_bonus = min(float(tick_bonus_cap), abs(float(tick_trend)) * float(tick_bonus_k))
            if tick_breakout_active and tick_breakout_dir != 0:
                if (action == "LONG" and tick_breakout_dir > 0) or (action == "SHORT" and tick_breakout_dir < 0):
                    tick_breakout_bonus = float(tick_breakout_score) * float(tick_breakout_boost)
            if impulse_bonus or momentum_bonus or rt_breakout_bonus or tick_bonus or tick_breakout_bonus:
                score_adj = float(impulse_bonus + momentum_bonus + rt_breakout_bonus + tick_bonus + tick_breakout_bonus)
                score = float(score + score_adj)
                score_raw = float(score_raw + score_adj)

        # Optional: entry bonus from MC future volatility (encourage high-opportunity regimes)
        mc_entry_bonus = 0.0
        try:
            mc_entry_on = str(os.environ.get("MC_FUTURE_VOL_ENTRY_BONUS", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            mc_entry_on = True
        if mc_entry_on and mc_vol_step is not None and action in ("LONG", "SHORT"):
            try:
                vol_ref = float(os.environ.get("MC_FUTURE_VOL_REF", 0.002) or 0.002)
            except Exception:
                vol_ref = 0.002
            try:
                entry_k = float(os.environ.get("MC_FUTURE_VOL_ENTRY_K", 0.0005) or 0.0005)
            except Exception:
                entry_k = 0.0005
            try:
                entry_cap = float(os.environ.get("MC_FUTURE_VOL_ENTRY_MAX", 0.003) or 0.003)
            except Exception:
                entry_cap = 0.003
            if vol_ref > 0 and entry_k > 0:
                ratio = max(0.0, float(mc_vol_step) / float(vol_ref) - 1.0)
                mc_entry_bonus = min(float(entry_cap), float(entry_k) * float(ratio))
                if mc_entry_bonus > 0:
                    score = float(score + mc_entry_bonus)
                    score_raw = float(score_raw + mc_entry_bonus)

        # Entry floor gate (hybrid)
        entry_floor_env = os.environ.get("HYBRID_ENTRY_FLOOR", os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
        auto_entry = _is_auto_value(entry_floor_env) or str(os.environ.get("HYBRID_AUTO_ENTRY", "0")).strip().lower() in ("1", "true", "yes", "on")
        entry_floor_dyn = ctx.get("hybrid_entry_floor_dyn") if auto_entry else None
        if entry_floor_dyn is not None:
            entry_floor = float(entry_floor_dyn)
        else:
            try:
                entry_floor = float(entry_floor_env or 0.0)
            except Exception:
                entry_floor = 0.0
        pos_side = ctx.get("position_side", ctx.get("position", 0) or 0)
        flat = False
        if isinstance(pos_side, str):
            flat = pos_side.upper() not in ("LONG", "SHORT")
        else:
            try:
                flat = float(pos_side) == 0.0
            except Exception:
                flat = True
        impulse_override = False
        rt_breakout_override = False
        if action == "WAIT" and flat:
            # Tick breakout override (fast microstructure breakout)
            if tick_breakout_active and tick_breakout_dir != 0:
                try:
                    tick_margin = float(os.environ.get("TICK_BREAKOUT_ENTRY_MARGIN", 0.0) or 0.0)
                except Exception:
                    tick_margin = 0.0
                score_candidate = float(score)
                if tick_breakout_boost > 0:
                    score_candidate = float(score_candidate + float(tick_breakout_score) * float(tick_breakout_boost))
                if score_candidate >= float(entry_floor - tick_margin):
                    action = "LONG" if tick_breakout_dir > 0 else "SHORT"
                    action_idx = 1 if tick_breakout_dir > 0 else 2
                    tick_breakout_bonus = float(tick_breakout_score) * float(tick_breakout_boost)
                    score = float(score_candidate)
                    score_raw = float(score_raw + tick_breakout_bonus)

            # Realtime breakout override (current price vs recent highs/lows)
            if rt_active and rt_dir != 0:
                try:
                    rt_margin = float(os.environ.get("REALTIME_BREAKOUT_ENTRY_MARGIN", 0.0) or 0.0)
                except Exception:
                    rt_margin = 0.0
                score_candidate = float(score)
                if rt_enabled and rt_boost > 0:
                    score_candidate = float(score_candidate + float(rt_score) * float(rt_boost))
                if score_candidate >= float(entry_floor - rt_margin):
                    action = "LONG" if rt_dir > 0 else "SHORT"
                    action_idx = 1 if rt_dir > 0 else 2
                    rt_breakout_override = True
                    rt_breakout_bonus = float(rt_score) * float(rt_boost) if rt_enabled else 0.0
                    score = float(score_candidate)
                    score_raw = float(score_raw + rt_breakout_bonus)

            # Impulse override (closed-bar breakout)
            if action == "WAIT" and impulse_active and impulse_dir != 0:
                try:
                    impulse_margin = float(os.environ.get("IMPULSE_ENTRY_MARGIN", 0.0) or 0.0)
                except Exception:
                    impulse_margin = 0.0
                score_candidate = float(score)
                if impulse_enabled and impulse_boost > 0:
                    score_candidate = float(score_candidate + float(impulse_score) * float(impulse_boost))
                if score_candidate >= float(entry_floor - impulse_margin):
                    action = "LONG" if impulse_dir > 0 else "SHORT"
                    action_idx = 1 if impulse_dir > 0 else 2
                    impulse_override = True
                    impulse_bonus = float(impulse_score) * float(impulse_boost) if impulse_enabled else 0.0
                    score = float(score_candidate)
                    score_raw = float(score_raw + impulse_bonus)
        if (not biased_entry) and flat and action in ("LONG", "SHORT") and score < entry_floor:
            action = "WAIT"

        # Directional guard: avoid entering against strong tick trend
        tick_guard_blocked = False
        try:
            tick_guard_on = str(os.environ.get("TICK_DIR_GUARD", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            tick_guard_on = True
        if tick_guard_on and flat and action in ("LONG", "SHORT"):
            try:
                tick_guard_min = float(os.environ.get("TICK_DIR_GUARD_MIN", 0.6) or 0.6)
            except Exception:
                tick_guard_min = 0.6
            if abs(float(tick_trend)) >= float(tick_guard_min):
                oppose = (action == "LONG" and tick_trend < 0) or (action == "SHORT" and tick_trend > 0)
                if oppose:
                    allow_override = False
                    if tick_breakout_active and tick_breakout_dir != 0:
                        allow_override = (action == "LONG" and tick_breakout_dir > 0) or (action == "SHORT" and tick_breakout_dir < 0)
                    if (not allow_override) and rt_active and rt_dir != 0:
                        allow_override = (action == "LONG" and rt_dir > 0) or (action == "SHORT" and rt_dir < 0)
                    if (not allow_override) and impulse_active and impulse_dir != 0:
                        allow_override = (action == "LONG" and impulse_dir > 0) or (action == "SHORT" and impulse_dir < 0)
                    if not allow_override:
                        action = "WAIT"
                        tick_guard_blocked = True

        # Optional: force LONG/SHORT direction by EV (entry evaluation)
        ev_force_used = False
        ev_force_dir = None
        ev_force_gap = None
        ev_force_long = None
        ev_force_short = None
        try:
            force_ev = str(os.environ.get("FORCE_EV_DIRECTION", "0")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            force_ev = False
        if force_ev:
            try:
                ctx_ev = dict(ctx)
                try:
                    ctx_ev["leverage"] = float(lev_used or ctx_ev.get("leverage", 1.0) or 1.0)
                except Exception:
                    ctx_ev["leverage"] = 1.0
                ctx_ev["_mu_alpha_ema_skip_update"] = True
                params_ev = params
                try:
                    n_paths_ev = int(os.environ.get("FORCE_EV_N_PATHS", 0) or 0)
                except Exception:
                    n_paths_ev = 0
                if n_paths_ev > 0:
                    try:
                        params_ev = replace(params, n_paths=int(n_paths_ev))
                    except Exception:
                        params_ev = params
                metrics_ev = self.evaluate_entry_metrics(ctx_ev, params_ev, seed=seed)
                if isinstance(metrics_ev, dict):
                    meta_ev = metrics_ev.get("meta") if isinstance(metrics_ev.get("meta"), dict) else {}
                    ev_force_long = metrics_ev.get("policy_ev_mix_long", meta_ev.get("policy_ev_mix_long"))
                    ev_force_short = metrics_ev.get("policy_ev_mix_short", meta_ev.get("policy_ev_mix_short"))
                    try:
                        ev_force_long = float(ev_force_long) if ev_force_long is not None else None
                    except Exception:
                        ev_force_long = None
                    try:
                        ev_force_short = float(ev_force_short) if ev_force_short is not None else None
                    except Exception:
                        ev_force_short = None
                    if ev_force_long is not None and ev_force_short is not None:
                        ev_force_gap = float(ev_force_long - ev_force_short)
                        try:
                            min_gap = float(os.environ.get("FORCE_EV_GAP_MIN", 0.0) or 0.0)
                        except Exception:
                            min_gap = 0.0
                        try:
                            require_pos = str(os.environ.get("FORCE_EV_REQUIRE_POSITIVE", "1")).strip().lower() in ("1", "true", "yes", "on")
                        except Exception:
                            require_pos = True
                        if abs(ev_force_gap) >= float(min_gap):
                            if (not require_pos) or max(ev_force_long, ev_force_short) > 0:
                                ev_force_dir = "LONG" if ev_force_gap > 0 else "SHORT"
            except Exception:
                ev_force_dir = None

        if ev_force_dir is not None and action in ("LONG", "SHORT"):
            if action != ev_force_dir:
                action = ev_force_dir
                action_idx = 1 if ev_force_dir == "LONG" else 2
                ev_force_used = True

        # Confidence proxy from score
        conf_scale_env = os.environ.get("HYBRID_CONF_SCALE", "0.01")
        auto_conf = _is_auto_value(conf_scale_env) or str(os.environ.get("HYBRID_AUTO_CONF", "0")).strip().lower() in ("1", "true", "yes", "on")
        conf_scale_dyn = ctx.get("hybrid_conf_scale_dyn") if auto_conf else None
        if conf_scale_dyn is not None:
            conf_scale = float(conf_scale_dyn)
        else:
            try:
                conf_scale = float(conf_scale_env or 0.01)
            except Exception:
                conf_scale = 0.01
        try:
            conf = 0.5 + 0.5 * math.tanh(score / max(1e-9, conf_scale))
        except Exception:
            conf = 0.5

        lev_used = optimal_leverage
        if lev_used is None:
            try:
                lev_used = float(ctx_for_sim.get("leverage", ctx.get("leverage", 1.0)) or 1.0)
            except Exception:
                lev_used = 1.0

        # Optional: compute optimal hold time (t*) via entry evaluation for exit alignment
        tstar = None
        tstar_src = None
        tstar_n_paths = None
        obs_mu_alpha = None
        obs_mu_alpha_raw = None
        obs_mu_dir_conf = None
        obs_mu_dir_edge = None
        obs_mu_dir_prob_long = None
        obs_policy_score_threshold = None
        obs_policy_event_exit_min_score = None
        obs_policy_unrealized_dd_floor = None
        try:
            tstar_on = str(os.environ.get("HYBRID_TSTAR_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            tstar_on = False
        if tstar_on and flat and action in ("LONG", "SHORT"):
            try:
                ctx_eval = dict(ctx)
                try:
                    ctx_eval["leverage"] = float(lev_used or ctx_eval.get("leverage", 1.0) or 1.0)
                except Exception:
                    ctx_eval["leverage"] = 1.0
                # Avoid side effects on alpha/EMA updates during auxiliary eval
                ctx_eval["_mu_alpha_ema_skip_update"] = True

                params_eval = params
                try:
                    tstar_n_paths = int(os.environ.get("HYBRID_TSTAR_N_PATHS", 0) or 0)
                except Exception:
                    tstar_n_paths = 0
                if tstar_n_paths > 0:
                    try:
                        params_eval = replace(params, n_paths=int(tstar_n_paths))
                    except Exception:
                        params_eval = params

                metrics_t = self.evaluate_entry_metrics(ctx_eval, params_eval, seed=seed)
                if isinstance(metrics_t, dict):
                    tstar = metrics_t.get("unified_t_star") or metrics_t.get("best_h")
                    meta_t = metrics_t.get("meta") if isinstance(metrics_t.get("meta"), dict) else {}

                    def _pick_obs(*keys: str):
                        for kk in keys:
                            try:
                                vv = metrics_t.get(kk)
                                if vv is not None:
                                    return vv
                            except Exception:
                                pass
                            try:
                                vv = meta_t.get(kk)
                                if vv is not None:
                                    return vv
                            except Exception:
                                pass
                        return None

                    obs_mu_alpha = _pick_obs("mu_alpha", "event_exit_dynamic_mu_alpha")
                    obs_mu_alpha_raw = _pick_obs("mu_alpha_raw")
                    obs_mu_dir_conf = _pick_obs("mu_dir_conf")
                    obs_mu_dir_edge = _pick_obs("mu_dir_edge")
                    obs_mu_dir_prob_long = _pick_obs("mu_dir_prob_long")
                    obs_policy_score_threshold = _pick_obs("policy_score_threshold_eff", "policy_score_threshold", "score_threshold")
                    obs_policy_event_exit_min_score = _pick_obs("event_exit_min_score")
                    obs_policy_unrealized_dd_floor = _pick_obs("unrealized_dd_floor_dyn")

                    if tstar is None:
                        tstar = meta_t.get("unified_t_star") or meta_t.get("policy_horizon_eff_sec") or meta_t.get("best_h")
                    try:
                        if tstar is not None:
                            tstar = float(tstar)
                            if not math.isfinite(tstar) or tstar <= 0:
                                tstar = None
                    except Exception:
                        tstar = None
                    if tstar is not None:
                        tstar_src = "entry_eval"
            except Exception as e:
                logger.warning(f"[HYBRID_TSTAR] {symbol} | compute failed: {e}")

        def _opt_float(v):
            try:
                if v is None:
                    return None
                fv = float(v)
                if not math.isfinite(fv):
                    return None
                return fv
            except Exception:
                return None

        mu_alpha_meta = _opt_float(obs_mu_alpha)
        if mu_alpha_meta is None:
            mu_alpha_meta = _opt_float(ctx.get("mu_alpha"))
        mu_alpha_raw_meta = _opt_float(obs_mu_alpha_raw)
        if mu_alpha_raw_meta is None:
            mu_alpha_raw_meta = _opt_float(ctx.get("mu_alpha_raw"))
        mu_dir_conf_meta = _opt_float(obs_mu_dir_conf)
        if mu_dir_conf_meta is None:
            mu_dir_conf_meta = _opt_float(ctx.get("mu_dir_conf"))
        mu_dir_edge_meta = _opt_float(obs_mu_dir_edge)
        if mu_dir_edge_meta is None:
            mu_dir_edge_meta = _opt_float(ctx.get("mu_dir_edge"))
        mu_dir_prob_long_meta = _opt_float(obs_mu_dir_prob_long)
        if mu_dir_prob_long_meta is None:
            mu_dir_prob_long_meta = _opt_float(ctx.get("mu_dir_prob_long"))

        meta = {
            "hybrid_action": hybrid_decision,
            "hybrid_action_idx": action_idx,
            "hybrid_score": float(score),
            "hybrid_score_raw": float(score_raw),
            "hybrid_score_logw": float(score_logw),
            "hybrid_score_source": score_source,
            "hybrid_score_scale_mode": scale_mode,
            "hybrid_score_long": score_long,
            "hybrid_score_short": score_short,
            "hybrid_score_hold": score_hold,
            "hybrid_entry_floor": entry_floor,
            "hybrid_biased_entry": bool(biased_entry),
            "hybrid_conf_scale": float(conf_scale),
            "hybrid_score_steps": int(score_steps),
            "hybrid_score_per_step": bool(per_step),
            "hybrid_score_time_step_sec": float(step_sec),
            "hybrid_score_total_sec": float(total_seconds) if total_seconds is not None else None,
            "momentum_z": float(momentum_z),
            "impulse_score": float(impulse_score),
            "impulse_dir": int(impulse_dir),
            "impulse_active": bool(impulse_active),
            "impulse_override": bool(impulse_override),
            "impulse_bonus": float(impulse_bonus),
            "momentum_bonus": float(momentum_bonus),
            "bias_eps_extra": float(bias_eps_extra),
            "rt_breakout_score": float(rt_score),
            "rt_breakout_dir": int(rt_dir),
            "rt_breakout_active": bool(rt_active),
            "rt_breakout_override": bool(rt_breakout_override),
            "rt_breakout_bonus": float(rt_breakout_bonus),
            "tick_ret": float(tick_ret),
            "tick_vol": float(tick_vol),
            "tick_trend": float(tick_trend),
            "tick_dir": int(tick_dir),
            "tick_breakout_active": bool(tick_breakout_active),
            "tick_breakout_dir": int(tick_breakout_dir),
            "tick_breakout_score": float(tick_breakout_score),
            "tick_breakout_bonus": float(tick_breakout_bonus),
            "tick_bonus": float(tick_bonus),
            "tick_guard_blocked": bool(tick_guard_blocked),
            "force_ev_used": bool(ev_force_used),
            "force_ev_dir": ev_force_dir,
            "force_ev_gap": float(ev_force_gap) if ev_force_gap is not None else None,
            "force_ev_long": float(ev_force_long) if ev_force_long is not None else None,
            "force_ev_short": float(ev_force_short) if ev_force_short is not None else None,
            "mc_future_vol_step": float(mc_vol_step) if mc_vol_step is not None else None,
            "mc_future_vol_ann": float(mc_vol_ann) if mc_vol_ann is not None else None,
            "mc_future_vol_entry_bonus": float(mc_entry_bonus),
            "optimal_leverage": float(lev_used),
            "maint_margin_rate": float(maint_rate) if maint_rate is not None else None,
            "liq_buffer": float(liq_buf) if liq_buf is not None else None,
            "liq_price_long": liq_price_long,
            "liq_price_short": liq_price_short,
            "liq_prob_long": liq_prob_long,
            "liq_prob_short": liq_prob_short,
            "regime": regime_ctx,
            "opt_hold_sec": float(tstar) if tstar is not None else None,
            "opt_hold_src": tstar_src,
            "opt_hold_n_paths": int(tstar_n_paths) if (tstar is not None and tstar_n_paths) else None,
            # Keep alpha observability populated even in HYBRID_ONLY mode.
            "mu_alpha": mu_alpha_meta,
            "mu_alpha_raw": mu_alpha_raw_meta,
            "mu_dir_conf": mu_dir_conf_meta,
            "mu_dir_edge": mu_dir_edge_meta,
            "mu_dir_prob_long": mu_dir_prob_long_meta,
            "vpin": _opt_float(ctx.get("vpin")),
            "hurst": _opt_float(ctx.get("hurst")),
            "policy_score_threshold_eff": _opt_float(obs_policy_score_threshold),
            "event_exit_min_score": _opt_float(obs_policy_event_exit_min_score),
            "unrealized_dd_floor_dyn": _opt_float(obs_policy_unrealized_dd_floor),
        }
        if lev_sweep_meta is not None:
            meta["lev_sweep"] = lev_sweep_meta

        res = {
            "action": action,
            "ev": float(score),
            "ev_raw": float(score),
            "confidence": float(conf),
            "reason": f"HYBRID score={score:.6f}",
            "meta": meta,
            "hybrid_score": float(score),
            "hybrid_score_logw": float(score_logw),
            "hybrid_score_long": score_long,
            "hybrid_score_short": score_short,
            "hybrid_score_hold": score_hold,
            "liq_prob_long": liq_prob_long,
            "liq_prob_short": liq_prob_short,
            "liq_price_long": liq_price_long,
            "liq_price_short": liq_price_short,
            "size_frac": float(ctx.get("size_frac", 0.0) or 0.0),
            "optimal_leverage": float(lev_used),
            "leverage": float(lev_used),
            "optimal_size": float(ctx.get("size_frac", 0.0) or 0.0),
            "boost": float(boost_val),
            # Unified-score compatibility (now mapped to hybrid)
            "unified_score": float(score),
            "unified_score_long": score_long,
            "unified_score_short": score_short,
            "unified_score_hold": score_hold,
            "mu_alpha": mu_alpha_meta,
            "mu_alpha_raw": mu_alpha_raw_meta,
            "mu_dir_conf": mu_dir_conf_meta,
            "mu_dir_edge": mu_dir_edge_meta,
            "mu_dir_prob_long": mu_dir_prob_long_meta,
            "details": [
                {
                    "_engine": "mc_barrier",
                    "_weight": 1.0,
                    "action": action,
                    "ev": float(score),
                    "confidence": float(conf),
                    "reason": "HYBRID_ONLY",
                    "meta": meta,
                }
            ],
        }
        if MC_VERBOSE_PRINT:
            logger.info(f"[HYBRID_ONLY] {symbol} | action={res['action']} score={res['ev']:.6f}")
            print(f"[HYBRID_ONLY] {symbol} | action={res['action']} score={res['ev']:.6f}")

        # Optional deep debug logging for hybrid score scale/inputs
        try:
            debug_on = str(os.environ.get("HYBRID_DEBUG", "0")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            debug_on = False
        if debug_on:
            interval_ms = int(os.environ.get("HYBRID_DEBUG_EVERY_SEC", "30") or 30) * 1000
            if _throttled_log(symbol, "hybrid_debug", interval_ms):
                try:
                    try:
                        print(f"[HYBRID_DEBUG] {symbol} debug_on", flush=True)
                    except Exception:
                        pass
                    torch = getattr(self, "_torch", None)
                    price_paths = mc_paths.get("prices")
                    if torch is not None and isinstance(price_paths, torch.Tensor):
                        price_t = price_paths
                    elif torch is not None:
                        price_t = torch.as_tensor(price_paths, dtype=torch.float32)
                    else:
                        price_t = None
                    lr_mean = None
                    lr_std = None
                    n_paths = None
                    if price_t is not None:
                        if price_t.ndim == 2:
                            n_paths = int(price_t.shape[0])
                            log_ret = torch.log(price_t[:, 1:].clamp(min=1e-12) / price_t[:, :-1].clamp(min=1e-12))
                        else:
                            n_paths = int(price_t.shape[1])
                            log_ret = torch.log(price_t[0, :, 1:].clamp(min=1e-12) / price_t[0, :, :-1].clamp(min=1e-12))
                        lr_mean = float(log_ret.mean().item())
                        lr_std = float(log_ret.std().item())
                    score_lsm = None
                    score_beam = None
                    action_idx_dbg = None
                    if isinstance(hybrid_meta, dict):
                        try:
                            action_idx_dbg = int(hybrid_meta.get("action_idx", 0) or 0)
                        except Exception:
                            action_idx_dbg = None
                        raw = hybrid_meta.get("raw") or {}
                        if isinstance(raw, dict):
                            score_lsm = raw.get("score_lsm")
                            score_beam = raw.get("score_beam")
                    mu_dbg = mc_paths.get("mu_used", ctx.get("mu_base", ctx.get("mu_alpha", ctx.get("mu_sim"))))
                    sigma_dbg = mc_paths.get("sigma_used", ctx.get("sigma", ctx.get("sigma_sim")))
                    dt_dbg = mc_paths.get("dt_used", getattr(self, "dt", None))
                    exposure_dbg = mc_paths.get("exposure", ctx.get("exposure"))
                    lev_used = mc_paths.get("leverage_used")
                    lev_cap = mc_paths.get("leverage_cap")
                    size_frac_dbg = mc_paths.get("size_frac")
                    mu_str = f"{float(mu_dbg):.6f}" if mu_dbg is not None else "None"
                    sigma_str = f"{float(sigma_dbg):.6f}" if sigma_dbg is not None else "None"
                    dt_str = f"{float(dt_dbg):.6e}" if dt_dbg is not None else "None"
                    lr_mean_str = f"{lr_mean:.6e}" if lr_mean is not None else "None"
                    lr_std_str = f"{lr_std:.6e}" if lr_std is not None else "None"
                    msg = (
                        "[HYBRID_DEBUG] %s action=%s idx=%s src=%s score=%.6e score_raw=%.6e logw=%.6e log_cap=%.6e steps=%s per_step=%s "
                        "scale=%s step_sec=%.3f entry=%.6e conf_scale=%.6e mu=%s sigma=%s dt=%s exp=%s lev=%s lev_cap=%s size=%s lr_mean=%s lr_std=%s n_paths=%s "
                        "score_lsm=%s score_beam=%s hold=%s long=%s short=%s"
                    )
                    logger.info(
                        msg,
                        symbol,
                        action,
                        action_idx_dbg,
                        score_source,
                        float(score),
                        float(score_raw),
                        float(score_logw),
                        float(log_cap),
                        score_steps,
                        per_step,
                        scale_mode,
                        float(step_sec),
                        float(entry_floor),
                        float(conf_scale),
                        mu_str,
                        sigma_str,
                        dt_str,
                        exposure_dbg,
                        lev_used,
                        lev_cap,
                        size_frac_dbg,
                        lr_mean_str,
                        lr_std_str,
                        n_paths,
                        score_lsm,
                        score_beam,
                        score_hold,
                        score_long,
                        score_short,
                    )
                    try:
                        print(
                            msg
                            % (
                                symbol,
                                action,
                                action_idx_dbg,
                                score_source,
                                float(score),
                                float(score_raw),
                                float(score_logw),
                                float(log_cap),
                                score_steps,
                                per_step,
                                float(entry_floor),
                                float(conf_scale),
                                mu_str,
                                sigma_str,
                                dt_str,
                                exposure_dbg,
                                lev_used,
                                lev_cap,
                                size_frac_dbg,
                                lr_mean_str,
                                lr_std_str,
                                n_paths,
                                score_lsm,
                                score_beam,
                                score_hold,
                                score_long,
                                score_short,
                            ),
                            flush=True,
                        )
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        print(f"[HYBRID_DEBUG] {symbol} debug_error: {e}", flush=True)
                    except Exception:
                        pass
        return res

    def _get_params(self, regime: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        use_torch_default = str(os.environ.get("MC_USE_TORCH", "1")).strip().lower() in ("1", "true", "yes", "on")
        params = {
            "n_paths": int(ctx.get("n_paths", config.n_paths_live)),
            "use_torch": bool(ctx.get("use_torch", ctx.get("use_jax", use_torch_default))),
            "use_jax": False,
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

        use_hybrid = str(os.environ.get("MC_USE_HYBRID_PLANNER", "0")).strip().lower() in ("1", "true", "yes", "on")
        hybrid_only_env = os.environ.get("MC_HYBRID_ONLY")
        hybrid_only = use_hybrid if hybrid_only_env is None else str(hybrid_only_env).strip().lower() in ("1", "true", "yes", "on")
        use_hybrid_batch = str(os.environ.get("MC_HYBRID_IN_BATCH", "0")).strip().lower() in ("1", "true", "yes", "on")

        if hybrid_only:
            # Hybrid-only: run per-symbol decisions (keeps legacy batch path untouched)
            return [self.decide(ctx) for ctx in ctx_list]
        
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
            
        # 2. Call Batch Evaluation (Multi-fidelity optional)
        import time
        t_eval_start = time.perf_counter()
        print(f"[DECIDE_BATCH] evaluate_entry_metrics_batch START num_tasks={len(tasks)}")

        mf_enabled = str(os.environ.get("MC_MULTI_FIDELITY_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        if mf_enabled:
            n_paths_stage1 = int(os.environ.get("MC_N_PATHS_STAGE1", max(512, int(config.n_paths_live // 4))))
            n_paths_stage2 = int(os.environ.get("MC_N_PATHS_STAGE2", int(config.n_paths_live)))
            mf_topk = int(os.environ.get("MC_MULTI_FIDELITY_TOPK", 8))
            mf_score_min = float(os.environ.get("MC_MULTI_FIDELITY_SCORE_MIN", 0.0))

            stage1_metrics = self.evaluate_entry_metrics_batch(tasks, n_paths_override=n_paths_stage1)
            for m in stage1_metrics:
                try:
                    m_meta = m.get("meta", {})
                    m_meta["mf_stage"] = 1
                    m_meta["mf_n_paths"] = int(n_paths_stage1)
                    m["meta"] = m_meta
                except Exception:
                    pass

            # Select candidates for stage2
            scored = []
            for i, m in enumerate(stage1_metrics):
                score_val = float(m.get("unified_score", m.get("ev", 0.0)) or 0.0)
                if score_val >= mf_score_min:
                    scored.append((i, score_val))
            scored.sort(key=lambda x: x[1], reverse=True)
            if mf_topk > 0:
                scored = scored[: mf_topk]
            idxs_stage2 = [i for i, _ in scored]

            if idxs_stage2:
                tasks_stage2 = [tasks[i] for i in idxs_stage2]
                stage2_metrics = self.evaluate_entry_metrics_batch(tasks_stage2, n_paths_override=n_paths_stage2)
                for j, i in enumerate(idxs_stage2):
                    m2 = stage2_metrics[j]
                    try:
                        m_meta = m2.get("meta", {})
                        m_meta["mf_stage"] = 2
                        m_meta["mf_n_paths"] = int(n_paths_stage2)
                        m2["meta"] = m_meta
                    except Exception:
                        pass
                    stage1_metrics[i] = m2

            batch_metrics = stage1_metrics
        else:
            batch_metrics = self.evaluate_entry_metrics_batch(tasks)

        t_eval_end = time.perf_counter()
        print(f"[DECIDE_BATCH] evaluate_entry_metrics_batch END elapsed={(t_eval_end-t_eval_start):.3f}s")
        
        # 3. Post-process and map back to input order
        results_map = {t["ctx"].get("symbol"): batch_metrics[i] for i, t in enumerate(tasks)}
        params_map = {t["ctx"].get("symbol"): t["params"] for t in tasks}
        seed_map = {t["ctx"].get("symbol"): t["seed"] for t in tasks}
            
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

                # Recompute unified scores if missing/zero but vectors are available
                try:
                    if abs(score_long) < 1e-12 and abs(score_short) < 1e-12:
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
                            rho_val = float(src.get("unified_rho", config.unified_rho))
                            lambda_val = float(src.get("unified_lambda", config.unified_risk_lambda))
                            ev_long_gross = [v + cost_roe_full for v in ev_long_vec]
                            ev_short_gross = [v + cost_roe_full for v in ev_short_vec]
                            cvar_long_gross = [v + cost_roe_full for v in cvar_long_vec]
                            cvar_short_gross = [v + cost_roe_full for v in cvar_short_vec]
                            score_long, _ = _calc_unified_score(
                                horizons, ev_long_gross, cvar_long_gross,
                                cost=cost_roe_full, rho=rho_val, lambda_param=lambda_val,
                            )
                            score_short, _ = _calc_unified_score(
                                horizons, ev_short_gross, cvar_short_gross,
                                cost=cost_roe_full, rho=rho_val, lambda_param=lambda_val,
                            )
                except Exception:
                    pass

                direction = 0
                if score_long > score_short:
                    direction = 1
                elif score_short > score_long:
                    direction = -1
                elif policy_score > 0:
                    direction = 1
                elif policy_score < 0:
                    direction = -1

                # Refresh policy_score after any recompute
                policy_score = float(score_long) if float(score_long) >= float(score_short) else float(score_short)

                ev_expected = _pick_ev_expected(metrics, direction)

                metric_copy = metrics.copy()
                def _as_float(v, default=0.0):
                    try:
                        if v is None:
                            return float(default)
                        return float(v)
                    except Exception:
                        return float(default)

                ev_val = _as_float(metrics.get("ev", policy_score), policy_score)
                ev_raw_val = _as_float(metrics.get("ev_raw", ev_val), ev_val)

                metric_copy["score"] = policy_score
                metric_copy["ev"] = ev_val
                metric_copy["ev_raw"] = ev_raw_val
                metric_copy["ev_expected"] = float(ev_expected) if ev_expected is not None else None
                metric_copy["ev_best"] = float(ev_expected) if ev_expected is not None else None
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

                entry_floor = float(os.environ.get("UNIFIED_ENTRY_FLOOR", 0.0) or 0.0)
                target_action = "WAIT"
                if config.score_only_mode:
                    if direction == 1:
                        target_action = "LONG"
                    elif direction == -1:
                        target_action = "SHORT"
                else:
                    if policy_score > entry_floor:
                        target_action = "LONG" if direction == 1 else "SHORT"

                metric_copy["action"] = target_action
                metric_copy["status"] = target_action
                best_h = int(metrics.get("best_h", 3600))
                metric_copy["reason"] = f"UNIFIED({best_h}s) {ctx.get('regime','chop')} score {policy_score*100:.2f}%"
                
                # Compatibility with LiveOrchestrator._paper_trade_step
                meta = metrics.get("meta", {})
                if isinstance(meta, dict):
                    meta["ev_expected"] = float(ev_expected) if ev_expected is not None else None
                    meta["ev_best"] = float(ev_expected) if ev_expected is not None else None
                metric_copy["optimal_leverage"] = float(meta.get("optimal_leverage") or meta.get("total_leverage_allocated") or 20.0)
                metric_copy["optimal_size"] = float(meta.get("optimal_size") or meta.get("size_frac") or 0.05)
                
                # Avoid recursion: details list contains a shallow copy without 'details' key
                detail_copy = metric_copy.copy()
                metric_copy["details"] = [detail_copy]

                if use_hybrid and use_hybrid_batch:
                    try:
                        params = params_map.get(sym)
                        seed = int(seed_map.get(sym, seed_window))
                        mc_paths = self._build_hybrid_mc_paths(ctx, params, seed)
                        hybrid_decision, hybrid_meta = self.decide_with_hybrid_planner(
                            sym, ctx, mc_paths, return_detail=True
                        )
                        if hybrid_decision in ("HOLD", "EXIT"):
                            hybrid_action_norm = "WAIT"
                        else:
                            hybrid_action_norm = hybrid_decision
                        metric_copy["action"] = hybrid_action_norm
                        metric_copy["status"] = hybrid_action_norm
                        meta = metric_copy.get("meta", {})
                        if isinstance(meta, dict):
                            meta["hybrid_action"] = hybrid_decision
                            meta["hybrid_action_idx"] = int(hybrid_meta.get("action_idx", 0) or 0)
                            meta["hybrid_score"] = float(hybrid_meta.get("score", 0.0) or 0.0)
                        metric_copy["reason"] = f"{metric_copy.get('reason','')} | HYBRID"
                    except Exception as e:
                        logger.warning(f"[HYBRID_BATCH] {sym} | fallback to default action: {e}")

                final_decisions.append(metric_copy)
            else:
                final_decisions.append({
                    "ok": False, "reason": "NO_METRICS", "action": "WAIT", "score": 0.0,
                })
        print(f"[DECIDE_BATCH] END symbols={num_symbols} elapsed={(time.time()*1000 - ts_ms):.0f}ms")
        return final_decisions

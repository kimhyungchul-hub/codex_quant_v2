from __future__ import annotations

import math
import logging
import os
import config as base_config
import time
from statistics import NormalDist
from typing import Any, Dict, Optional, Sequence

import numpy as np

from engines.cvar_methods import cvar_ensemble
from engines.mc.constants import MC_VERBOSE_PRINT, SECONDS_PER_YEAR
from engines.mc import jax_backend as jax_backend
from engines.mc.jax_backend import summarize_gbm_horizons_multi_symbol_jax
from engines.mc.params import MCParams
from engines.mc.config import config
from regime import adjust_mu_sigma

# ✅ Exit Policy Flag - Full exit policy calculation by default to include all exit logic in MC simulation
# Set SKIP_EXIT_POLICY=true to use simplified summary-based EV (faster but less accurate)
# Default: false (empty string or "false" → full exit policy calculation)
SKIP_EXIT_POLICY = bool(getattr(config, "skip_exit_policy", False))

# ✅ Clean evaluator for horizon processing (fixes IndexError bug)
try:
    from engines.mc.entry_evaluation_clean import get_clean_evaluator
    _CLEAN_EVALUATOR_AVAILABLE = True
except ImportError:
    _CLEAN_EVALUATOR_AVAILABLE = False
    logger.warning("[CLEAN_EVAL] Clean evaluator not available, using legacy code")

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


def _summary_to_numpy(res: Any) -> Any:
    """Convert torch-backed summary dict to numpy arrays."""
    if isinstance(res, dict):
        out: dict[str, Any] = {}
        for k, v in res.items():
            try:
                out[k] = jax_backend.to_numpy(v)
            except Exception:
                out[k] = v
        return out
    return res


def _calc_unified_score_from_cumulative(
    horizons_sec: Sequence[float],
    cumulative_ev: Sequence[float],
    cumulative_cvar: Sequence[float],
    *,
    cost: float,
    rho: float,
    lambda_param: float,
) -> tuple[float, float]:
    h = np.asarray(horizons_sec, dtype=float)
    ev = np.asarray(cumulative_ev, dtype=float)
    cv = np.asarray(cumulative_cvar, dtype=float)
    if h.ndim != 1 or ev.ndim != 1 or cv.ndim != 1:
        return 0.0, 0.0
    n = min(h.size, ev.size, cv.size)
    if n < 2:
        return 0.0, 0.0
    h = h[:n]
    ev = ev[:n]
    cv = cv[:n]
    dt = np.diff(h, prepend=0.0)
    safe_dt = np.where(dt > 0.0, dt, 1.0)
    marginal_ev = np.diff(ev, prepend=0.0) / safe_dt
    marginal_cvar = np.diff(cv, prepend=0.0) / safe_dt
    utility_rate = marginal_ev - float(lambda_param) * np.abs(marginal_cvar)
    discount = np.exp(-float(rho) * h)
    gross_napv = np.cumsum((utility_rate - float(rho)) * discount * dt)
    denom = np.where(h > 0.0, h, 1.0)
    psi_score = (gross_napv - float(cost)) / denom
    best_idx = int(np.argmax(psi_score))
    return float(psi_score[best_idx]), float(h[best_idx])


class MonteCarloEntryEvaluationMixin:
    def _get_execution_costs(self, ctx: Dict[str, Any], params: Optional[MCParams] = None) -> Dict[str, float]:
        """
        Helper to calculate execution costs for batch processing.
        Simplified extraction from evaluate_entry_metrics logic.
        USE_MAKER_ORDERS=true일 때 Maker 수수료 적용 (0.02% roundtrip)
        """
        # Defaults - Maker 우선 사용
        _use_maker = bool(getattr(base_config, "USE_MAKER_ORDERS", True))
        fee_taker = float(ctx.get("fee_taker", 0.0006))
        fee_maker = float(ctx.get("fee_maker", 0.0001))
        
        # PMaker params - Maker 주문 우선시 pmaker_entry를 높게 설정
        pmaker_entry = float(ctx.get("pmaker_entry", 0.95 if _use_maker else 0.9))
        exec_mode = "maker_then_market"
        
        if params:
            if hasattr(params, "pmaker_entry"):
                try:
                    pmaker_entry = float(params.pmaker_entry)
                except (ValueError, TypeError):
                    pass
            if hasattr(params, "exec_mode"):
                try:
                    exec_mode = str(params.exec_mode)
                except (ValueError, TypeError):
                    pass

        # Dynamic adjustments
        p_maker = pmaker_entry
        if exec_mode != "maker_then_market":
            p_maker = 0.0 # Taker only
            
        # Fee Mix
        fee_fee_mix = p_maker * fee_maker + (1.0 - p_maker) * fee_taker
        
        # Slippage / Spread (Defaults)
        slippage_dyn_raw = float(ctx.get("slippage_dyn_raw", 0.0002))
        expected_spread_cost_raw = float(ctx.get("expected_spread_cost_raw", 0.0001))
        
        # Effective Slippage/Spread (reduced by maker probability)
        slippage_dyn = (1.0 - p_maker) * slippage_dyn_raw
        expected_spread_cost = (1.0 - p_maker) * expected_spread_cost_raw
        
        # Total
        fee_roundtrip_total = fee_fee_mix + slippage_dyn + expected_spread_cost
        
        # Cap
        MAX_FEE_ROUNDTRIP = 0.003
        fee_roundtrip_total = min(fee_roundtrip_total, MAX_FEE_ROUNDTRIP)
        
        return {
            "fee_roundtrip": fee_roundtrip_total,
            "exec_oneway": fee_roundtrip_total / 2.0,
            "impact_cost": slippage_dyn + expected_spread_cost
        }

    def evaluate_entry_metrics(self, ctx: Dict[str, Any], params: MCParams, seed: int) -> Dict[str, Any]:
        """
        튜너가 과거 ctx로 candidate 파라미터 평가할 때도 사용.
        """
        symbol = str(ctx.get("symbol", ""))
        if MC_VERBOSE_PRINT:
            logger.info(
                f"[EV_DEBUG] {symbol} | evaluate={ctx.get('price')} mu={ctx.get('mu_sim')} sig={ctx.get('sigma_sim')}"
            )
        def _s(val, default=0.0) -> float:
            try:
                if val is None:
                    return float(default)
                return float(val)
            except Exception:
                return float(default)

        # symbol은 위에서 이미 가져옴
        price = _s(ctx.get("price"), 0.0)
        
        # mu/sigma 입력은 유지하되, "μ는 최근 평균수익"이 아니라 아래에서 계산하는 알파 μ로 완전 대체한다.
        # (mu_base는 디버깅/추적용으로만 남긴다)
        mu_base = ctx.get("mu_sim")
        if mu_base is None:
            mu_base = ctx.get("mu_base")
        sigma = ctx.get("sigma_sim")  # sigma_sim 우선
        if sigma is None:
            sigma = ctx.get("sigma")  # fallback
        
        # ✅ [EV_DEBUG] mu_base, sigma 값 확인
        if MC_VERBOSE_PRINT:
            print(f"[EV_DEBUG] {symbol} | evaluate_entry_metrics: mu_base={mu_base} sigma={sigma} price={price}")
        if mu_base is None or sigma is None or sigma <= 0:
            if MC_VERBOSE_PRINT:
                print(f"[EV_DEBUG] {symbol} | ⚠️  WARNING: mu_base or sigma is invalid! mu_base={mu_base} sigma={sigma}")
                logger.warning(f"[EV_DEBUG] {symbol} | ⚠️  WARNING: mu_base or sigma is invalid! mu_base={mu_base} sigma={sigma}")
            elif _throttled_log(symbol, "MU_SIGMA_INVALID", 60_000):
                logger.warning(f"[EV_DEBUG] {symbol} | ⚠️  WARNING: mu_base or sigma is invalid! mu_base={mu_base} sigma={sigma}")
        
        closes = ctx.get("closes")
        liq_score = _s(ctx.get("liquidity_score"), 1.0)
        bar_seconds = _s(ctx.get("bar_seconds", 60.0), 60.0)
        # tail mode plumbing
        self._use_torch = bool(ctx.get("use_torch", ctx.get("use_jax", True)))
        self._use_jax = False
        self._tail_mode = str(ctx.get("tail_mode", self.default_tail_mode))
        self._student_t_df = _s(ctx.get("student_t_df", self.default_student_t_df), self.default_student_t_df)
        br = ctx.get("bootstrap_returns")
        if br is not None:
            try:
                self._bootstrap_returns = np.asarray(br, dtype=np.float64)
            except Exception:
                self._bootstrap_returns = None
        else:
            if self._tail_mode == "bootstrap" and closes is not None and len(closes) >= 64:
                x = np.asarray(closes, dtype=np.float64)
                rets = np.diff(np.log(np.maximum(x, 1e-12)))
                from engines.mc.constants import BOOTSTRAP_HISTORY_LEN, BOOTSTRAP_MIN_SAMPLES
                self._bootstrap_returns = rets[-BOOTSTRAP_HISTORY_LEN:].astype(np.float64) if rets.size >= BOOTSTRAP_MIN_SAMPLES else None
            else:
                self._bootstrap_returns = None
        regime_ctx = str(ctx.get("regime", "chop"))

        # ✅ Step 1: mu_base, sigma 초기 계산 전 상태 기록
        mu_base_input = mu_base
        sigma_input = sigma
        closes_len = len(closes) if closes is not None else 0
        returns_window_len = None
        vol_src = "ctx_input"
        
        def _clean_logrets(closes_local: Sequence[float]) -> np.ndarray:
            x = np.asarray(closes_local, dtype=np.float64)
            if x.size < 2:
                return np.zeros((0,), dtype=np.float64)
            # Guard against bad candles (0/negatives) and extreme spikes that can blow up annualization.
            logp = np.log(np.maximum(x, 1e-12))
            rets0 = np.diff(logp)
            rets0 = rets0[np.isfinite(rets0)]
            if rets0.size == 0:
                return rets0
            try:
                lr_cap = config.sigma_lr_cap
            except Exception:
                lr_cap = 0.02
            if lr_cap > 0:
                rets0 = np.clip(rets0, -lr_cap, lr_cap)
            return rets0.astype(np.float64)

        # sigma가 없는 경우만 closes에서 보충 (μ는 사용하지 않음)
        if (sigma is None) and closes is not None and len(closes) >= 10:
            rets = _clean_logrets(closes)
            if rets.size >= 5:
                returns_window_len = int(rets.size)
                vol_src = f"closes_diff_n={returns_window_len}"
                mu_bar = float(rets.mean())
                sigma_bar = float(rets.std())
                _, sigma = self._annualize(mu_bar, sigma_bar, bar_seconds=bar_seconds)

        # 다중 지평 블렌딩으로 추정치 안정화
        if closes is not None and len(closes) >= 30:
            rets = _clean_logrets(closes)
            windows = [30, 90, 180]
            mu_blend = []
            sigma_blend = []
            for w in windows:
                if rets.size >= w:
                    rw = rets[-w:]
                    mu_blend.append(float(rw.mean()))
                    sigma_blend.append(float(rw.std()))
            if mu_blend and sigma_blend:
                returns_window_len = rets.size  # 업데이트
                vol_src = f"blend_windows={windows}_n={returns_window_len}"
                mu_bar_mix = float(np.mean(mu_blend))
                sigma_bar_mix = float(np.mean(sigma_blend))
                mu_mix, sigma_mix = self._annualize(mu_bar_mix, sigma_bar_mix, bar_seconds=bar_seconds)
                if sigma is None:
                    sigma = sigma_mix
                else:
                    sigma = 0.5 * float(sigma) + 0.5 * float(sigma_mix)

        if sigma is None or price <= 0:
            logger.warning(
                f"[MU_SIGMA_DEBUG] {symbol} | early return: mu_base_input={mu_base_input}, sigma_input={sigma_input}, "
                f"mu_base_after={mu_base}, sigma_after={sigma}, price={price}, closes_len={closes_len}, "
                f"returns_window_len={returns_window_len}, vol_src={vol_src}, regime={regime_ctx}"
            )
            if MC_VERBOSE_PRINT:
                print(f"[EARLY_RETURN_1] {symbol} | sigma={sigma} price={price} (returning ev=0)")
            return {"can_enter": False, "ev": 0.0, "ev_raw": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": 1, "kelly": 0.0, "size_frac": 0.0}

        sigma = float(sigma) if sigma is not None else 0.0
        
        # ✅ Step 1: sigma가 0이면 경고 및 키 확인
        if sigma <= 0:
            ctx_keys = list(ctx.keys())
            mu_sigma_keys = [k for k in ctx_keys if 'mu' in k.lower() or 'sigma' in k.lower()]
            logger.error(
                f"[MU_SIGMA_ERROR] {symbol} | sigma <= 0 before adjust! "
                f"sigma={sigma}, mu_base={mu_base}, "
                f"ctx_keys_with_mu_sigma={mu_sigma_keys}, "
                f"mu_sim={ctx.get('mu_sim')}, sigma_sim={ctx.get('sigma_sim')}, "
                f"mu_base_ctx={ctx.get('mu_base')}, sigma_ctx={ctx.get('sigma')}"
            )
            # 절대 0 fallback하지 않음 - 에러를 명확히 보고
            return {"can_enter": False, "ev": 0.0, "ev_raw": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": 1, "kelly": 0.0, "size_frac": 0.0}
        
        sigma = max(sigma, 1e-6)  # 최소값 보정 (0이 아닌 경우에만)
        
        # 레짐 기반 기대수익/변동성 조정
        mu_base_before_adjust = float(mu_base) if mu_base is not None else 0.0
        sigma_before_adjust = sigma
        # ✅ μ는 신호 조건부 기대수익(알파)로 완전 대체
        ofi_score = _s(ctx.get("ofi_score"), 0.0)
        mu_alpha_parts = self._signal_alpha_mu_annual_parts(closes or [], bar_seconds, ofi_score, regime_ctx)
        mu_alpha_raw = float(mu_alpha_parts.get("mu_alpha") or 0.0)

        # -----------------------------
        # Advanced alpha components (optional)
        # -----------------------------
        try:
            use_mlofi = bool(getattr(config, "alpha_use_mlofi", False))
            use_vpin = bool(getattr(config, "alpha_use_vpin", False))
            use_kf = bool(getattr(config, "alpha_use_kf", False))
            use_hurst = bool(getattr(config, "alpha_use_hurst", False))
            use_garch = bool(getattr(config, "alpha_use_garch", False))
            use_bayes = bool(getattr(config, "alpha_use_bayes", False))
            use_arima = bool(getattr(config, "alpha_use_arima", False))
            use_ml = bool(getattr(config, "alpha_use_ml", False))
            use_hawkes = bool(getattr(config, "alpha_use_hawkes", False))
            use_pf = bool(getattr(config, "alpha_use_pf", False))
            use_causal = bool(getattr(config, "alpha_use_causal", False))
        except Exception:
            use_mlofi = use_vpin = use_kf = use_hurst = False
            use_garch = use_bayes = use_arima = use_ml = False
            use_hawkes = use_pf = use_causal = False

        comp_sum_w = 0.0
        comp_sum = 0.0

        def _add_comp(name: str, val: float, w: float) -> None:
            nonlocal comp_sum_w, comp_sum
            if w is None:
                return
            w = float(w)
            if w <= 0:
                return
            comp_sum_w += w
            comp_sum += w * float(val)
            mu_alpha_parts[name] = float(val)
            mu_alpha_parts[f"{name}_w"] = float(w)

        if use_mlofi:
            mlofi = _s(ctx.get("mlofi"), 0.0)
            mlofi_scale = float(getattr(config, "mlofi_scale", 8.0))
            mu_mlofi = float(mlofi) * mlofi_scale
            _add_comp("mu_mlofi", mu_mlofi, float(getattr(config, "mlofi_weight", 0.20)))

        if use_kf:
            mu_kf = _s(ctx.get("mu_kf"), 0.0)
            _add_comp("mu_kf", mu_kf, float(getattr(config, "kf_weight", 0.20)))

        if use_bayes:
            mu_bayes = _s(ctx.get("mu_bayes"), 0.0)
            _add_comp("mu_bayes", mu_bayes, float(getattr(config, "bayes_weight", 0.10)))
            mu_alpha_parts["mu_bayes_var"] = _s(ctx.get("mu_bayes_var"), 0.0)

        if use_arima:
            mu_ar = _s(ctx.get("mu_ar"), 0.0)
            _add_comp("mu_ar", mu_ar, float(getattr(config, "arima_weight", 0.10)))

        if use_pf:
            mu_pf = _s(ctx.get("mu_pf"), 0.0)
            _add_comp("mu_pf", mu_pf, float(getattr(config, "pf_weight", 0.10)))

        if use_ml:
            mu_ml = _s(ctx.get("mu_ml"), 0.0)
            _add_comp("mu_ml", mu_ml, float(getattr(config, "ml_weight", 0.15)))

        if use_causal:
            mu_causal = _s(ctx.get("mu_causal"), 0.0)
            _add_comp("mu_causal", mu_causal, float(getattr(config, "causal_weight", 0.05)))

        # Combine with base mu_alpha
        if comp_sum_w > 1.0:
            scale = 1.0 / comp_sum_w
            comp_sum *= scale
            comp_sum_w = 1.0
        base_w = max(0.0, 1.0 - comp_sum_w)
        mu_alpha_raw = float(base_w * mu_alpha_raw + comp_sum)
        mu_alpha_parts["mu_alpha_mix_base_w"] = float(base_w)
        mu_alpha_parts["mu_alpha_mix_sum_w"] = float(comp_sum_w)
        mu_alpha_parts["mu_alpha_after_mix"] = float(mu_alpha_raw)

        # Hurst-driven regime switch / OU blend
        if use_hurst:
            hurst = ctx.get("hurst")
            if hurst is not None:
                hurst = float(hurst)
                mu_alpha_parts["hurst"] = float(hurst)
                low = float(getattr(config, "hurst_low", 0.45))
                high = float(getattr(config, "hurst_high", 0.55))
                if hurst < low:
                    mu_ou = _s(ctx.get("mu_ou"), 0.0)
                    ou_w = float(getattr(config, "ou_weight", 0.7))
                    mu_alpha_raw = float((1.0 - ou_w) * mu_alpha_raw + ou_w * mu_ou)
                    mu_alpha_parts["hurst_regime"] = "mean_revert"
                elif hurst > high:
                    boost = float(getattr(config, "hurst_trend_boost", 1.15))
                    mu_alpha_raw = float(mu_alpha_raw * boost)
                    mu_alpha_parts["hurst_regime"] = "trend"
                else:
                    damp = float(getattr(config, "hurst_random_dampen", 0.25))
                    mu_alpha_raw = float(mu_alpha_raw * damp)
                    mu_alpha_parts["hurst_regime"] = "random"

        # VPIN risk adjustment (damp + sigma expand + extreme contrarian OU blend)
        if use_vpin:
            vpin = ctx.get("vpin")
            if vpin is not None:
                vpin = max(0.0, min(1.0, float(vpin)))
                gamma = float(getattr(config, "vpin_gamma", 0.6))
                damp_floor = float(getattr(config, "vpin_damp_floor", 0.10))
                damp = max(damp_floor, 1.0 - gamma * vpin)

                extreme_th = float(getattr(config, "vpin_extreme_threshold", 0.90))
                if vpin >= extreme_th:
                    mu_ou = _s(ctx.get("mu_ou"), 0.0)
                    ou_ext_w = float(getattr(config, "vpin_extreme_ou_weight", 0.85))
                    ou_ext_w = min(1.0, max(0.0, ou_ext_w))
                    if abs(mu_ou) > 0:
                        mu_alpha_raw = float((1.0 - ou_ext_w) * mu_alpha_raw + ou_ext_w * mu_ou)
                        mu_alpha_parts["vpin_extreme_mode"] = "ou_contrarian"
                        mu_alpha_parts["vpin_extreme_ou_weight"] = float(ou_ext_w)
                    else:
                        mu_alpha_parts["vpin_extreme_mode"] = "dampen_only"

                mu_alpha_raw = float(mu_alpha_raw * damp)
                mu_alpha_parts["vpin"] = float(vpin)
                mu_alpha_parts["vpin_damp"] = float(damp)

                # sigma boost with hard cap to avoid blow-up
                vpin_sigma_k = float(getattr(config, "vpin_sigma_k", 0.8))
                sigma_cap_mult = float(getattr(config, "vpin_sigma_cap_mult", 2.5))
                sigma_mult = float(1.0 + vpin_sigma_k * vpin)
                sigma_mult = min(sigma_mult, max(1.0, sigma_cap_mult))
                sigma = float(sigma) * sigma_mult
                mu_alpha_parts["vpin_sigma_mult"] = float(sigma_mult)

        # Hawkes boost
        if use_hawkes:
            hawkes_boost = _s(ctx.get("hawkes_boost"), 0.0)
            hawkes_k = float(getattr(config, "hawkes_boost_k", 0.3))
            mu_alpha_raw = float(mu_alpha_raw + hawkes_k * hawkes_boost)
            mu_alpha_parts["hawkes_boost"] = float(hawkes_boost)
            mu_alpha_parts["hawkes_k"] = float(hawkes_k)

        # GARCH sigma blending (optional)
        if use_garch:
            sigma_garch = ctx.get("sigma_garch")
            if sigma_garch is not None and float(sigma_garch) > 0:
                w_g = float(getattr(config, "garch_sigma_weight", 0.5))
                sigma = float((1.0 - w_g) * float(sigma) + w_g * float(sigma_garch))
                mu_alpha_parts["sigma_garch"] = float(sigma_garch)
                mu_alpha_parts["sigma_garch_w"] = float(w_g)
        mu_alpha_parts["mu_alpha_raw"] = float(mu_alpha_raw)

        # Optional: directional correction layer from lightweight classifier (logistic baseline).
        # This adjusts mu_alpha sign only in high-confidence direction regimes.
        try:
            use_dir_corr = bool(getattr(config, "alpha_direction_use", False))
        except Exception:
            use_dir_corr = False
        if use_dir_corr:
            dir_edge = _s(ctx.get("mu_dir_edge"), 0.0)
            dir_conf = _s(ctx.get("mu_dir_conf"), abs(dir_edge))
            dir_prob_long = _s(ctx.get("mu_dir_prob_long"), 0.5)
            dir_strength = float(getattr(config, "alpha_direction_strength", 0.6))
            dir_conf_th = float(getattr(config, "alpha_direction_min_confidence", 0.55))
            dir_conf_floor = float(getattr(config, "alpha_direction_confidence_floor", 0.45))
            dir_conf_th = float(max(dir_conf_th, dir_conf_floor))
            dir_conf = float(max(0.0, min(1.0, dir_conf)))
            dir_blend = 0.0
            if abs(dir_edge) > 0.0 and dir_conf >= dir_conf_th and dir_strength > 0.0:
                dir_target = math.copysign(abs(mu_alpha_raw), float(dir_edge))
                dir_blend = float(min(1.0, max(0.0, dir_strength * dir_conf)))
                mu_alpha_raw = float((1.0 - dir_blend) * float(mu_alpha_raw) + dir_blend * float(dir_target))
                mu_alpha_parts["mu_alpha_raw"] = float(mu_alpha_raw)
            mu_alpha_parts["mu_dir_prob_long"] = float(dir_prob_long)
            mu_alpha_parts["mu_dir_edge"] = float(dir_edge)
            mu_alpha_parts["mu_dir_conf"] = float(dir_conf)
            mu_alpha_parts["mu_dir_conf_min_required"] = float(dir_conf_th)
            mu_alpha_parts["mu_dir_blend"] = float(dir_blend)

        # ✅ [FIX 2] PMaker fill 결과를 mu_alpha에 반영하여 개선
        # fill rate가 높으면 alpha 신뢰도 증가 → mu_alpha 증가
        # fill rate가 낮으면 alpha 신뢰도 감소 → mu_alpha 감소
        mu_alpha_pmaker_adjusted = mu_alpha_raw
        pmaker_surv = ctx.get("pmaker_surv")
        pmaker_mu_alpha_boost_enabled = config.pmaker_mu_alpha_boost_enabled
        pmaker_mu_alpha_boost_k = config.pmaker_mu_alpha_boost_k
        
        
        if pmaker_mu_alpha_boost_enabled and pmaker_surv is not None:
            try:
                # 심볼별 평균 fill rate 가져오기
                fill_rate_mean = pmaker_surv.sym_fill_mean(symbol)
                # 중립 기준을 0.5로 잡되, fill_rate가 낮다고 mu_alpha를 추가로 깎지는 않는다(부정적 신호와 이중 반영 방지).
                fill_rate_bias = max(0.0, (fill_rate_mean - 0.5) * 2.0)  # [0, 1] 범위로 변환
                # fill_rate가 0.5보다 높을 때만 alpha를 보정(추가 가산), 낮을 때는 0으로 두어 악화 방지
                mu_alpha_boost = fill_rate_bias * pmaker_mu_alpha_boost_k * abs(mu_alpha_raw)
                mu_alpha_pmaker_adjusted = mu_alpha_raw + mu_alpha_boost
                
                # mu_alpha_parts에 조정값 저장
                mu_alpha_parts["mu_alpha_pmaker_fill_rate"] = float(fill_rate_mean)
                mu_alpha_parts["mu_alpha_pmaker_boost"] = float(mu_alpha_boost)
                mu_alpha_parts["mu_alpha_before_pmaker"] = float(mu_alpha_raw)
                
                logger.info(
                    f"[MU_ALPHA_PMAKER_BOOST] {symbol} | "
                    f"fill_rate={fill_rate_mean:.4f} fill_rate_bias={fill_rate_bias:.4f} | "
                    f"mu_alpha_raw={mu_alpha_raw:.6f} boost={mu_alpha_boost:.6f} mu_alpha_adjusted={mu_alpha_pmaker_adjusted:.6f}"
                )
            except Exception as e:
                logger.warning(f"[MU_ALPHA_PMAKER_BOOST] {symbol} | Failed to apply PMaker boost: {e}")
                mu_alpha_parts["mu_alpha_pmaker_fill_rate"] = None
                mu_alpha_parts["mu_alpha_pmaker_boost"] = 0.0
        
        # mu_alpha cap 적용 (pmaker 조정 후)
        try:
            mu_cap = config.mu_alpha_cap
        except Exception:
            mu_cap = 40.0
        mu_alpha_final = float(mu_alpha_pmaker_adjusted)
        if mu_alpha_final > mu_cap:
            mu_alpha_final = mu_cap
        elif mu_alpha_final < -mu_cap:
            mu_alpha_final = -mu_cap

        # Optional: EMA smoothing (residual alpha / inertia) to reduce signal flicker.
        # - Enabled when MU_ALPHA_EMA_ALPHA in (0, 1].
        # - To avoid double-updating during multi-pass evaluation (e.g. leverage=1 base pass),
        #   callers can set ctx["_mu_alpha_ema_skip_update"]=True.
        mu_alpha_pre_ema = float(mu_alpha_final)
        try:
            mu_alpha_ema_alpha = config.mu_alpha_ema_alpha
        except Exception:
            mu_alpha_ema_alpha = 0.0
        if 0.0 < float(mu_alpha_ema_alpha) <= 1.0:
            try:
                ema_state = getattr(self, "_mu_alpha_ema", None)
                if ema_state is None:
                    ema_state = {}
                    setattr(self, "_mu_alpha_ema", ema_state)
                prev = ema_state.get(symbol)
                if prev is None:
                    mu_alpha_final = float(mu_alpha_pre_ema)
                else:
                    mu_alpha_final = float((1.0 - float(mu_alpha_ema_alpha)) * float(prev) + float(mu_alpha_ema_alpha) * float(mu_alpha_pre_ema))
                # safety clamp
                if mu_alpha_final > mu_cap:
                    mu_alpha_final = mu_cap
                elif mu_alpha_final < -mu_cap:
                    mu_alpha_final = -mu_cap

                mu_alpha_parts["mu_alpha_before_ema"] = float(mu_alpha_pre_ema)
                mu_alpha_parts["mu_alpha_ema_alpha"] = float(mu_alpha_ema_alpha)
                mu_alpha_parts["mu_alpha"] = float(mu_alpha_final)

                skip_update = bool(ctx.get("_mu_alpha_ema_skip_update"))
                if not skip_update:
                    ema_state[symbol] = float(mu_alpha_final)
            except Exception as e:
                logger.warning(f"[MU_ALPHA_EMA] {symbol} | Failed to apply EMA smoothing: {e}")

        mu_alpha_parts["mu_alpha"] = float(mu_alpha_final)  # 최종 mu_alpha 업데이트
        try:
            ctx["mu_alpha"] = float(mu_alpha_final)
        except Exception:
            ctx["mu_alpha"] = 0.0
        
        mu_base, sigma = adjust_mu_sigma(float(mu_alpha_final), sigma, regime_ctx)
        
        # ✅ Step 1: 조정 후에도 sigma가 0이면 경고
        if sigma <= 0:
            logger.error(
                f"[MU_SIGMA_ERROR] {symbol} | sigma <= 0 after adjust_mu_sigma! "
                f"sigma_before={sigma_before_adjust:.8f}, sigma_after={sigma:.8f}, "
                f"mu_before={mu_base_before_adjust:.8f}, mu_after={mu_base:.8f}, regime={regime_ctx}"
            )
            if MC_VERBOSE_PRINT:
                print(f"[EARLY_RETURN_2] {symbol} | sigma={sigma} after adjust_mu_sigma (returning ev=0)")
            return {"can_enter": False, "ev": 0.0, "ev_raw": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": 1, "kelly": 0.0, "size_frac": 0.0}
        
        sigma = max(sigma, 1e-6)  # 조정 후 최소값 보정
        
        # ✅ [FIX] meta 변수를 초기화 (line 2413에서 사용되기 전에)
        meta = {}
        
        # ✅ Step 1: mu_sim, sigma_sim 추적 로그 (조정 후)
        if MC_VERBOSE_PRINT:
            logger.info(
                f"[MU_SIGMA_DEBUG] {symbol} | "
                f"mu_base_input={mu_base_input}, sigma_input={sigma_input} | "
                f"mu_base_before_adjust={mu_base_before_adjust:.8f}, sigma_before_adjust={sigma_before_adjust:.8f} | "
                f"mu_base_after_adjust={mu_base:.8f}, sigma_after_adjust={sigma:.8f} | "
                f"closes_len={closes_len}, returns_window_len={returns_window_len}, vol_src={vol_src} | "
                f"regime={regime_ctx}"
            )

        direction = int(ctx.get("direction") or self._trend_direction(price, closes or []))
        regime_ctx_for_cluster = ctx.get("regime")
        if regime_ctx_for_cluster == "chop" and closes:
            regime_ctx_for_cluster = self._cluster_regime(closes)

        mu_adj = float(mu_base)
        signal_mu = float(mu_alpha_parts.get("mu_alpha", mu_alpha_raw))  # ✅ FIX: pmaker 조정된 mu_alpha 사용
        mu_alpha_for_ev = float(signal_mu)
        mu_adjusted_for_ev = float(mu_adj)
        # Keep ctx/meta aligned with the exact mu used for EV construction.
        if isinstance(meta, dict):
            meta["mu_alpha_for_ev"] = float(mu_alpha_for_ev)
            meta["mu_adjusted_for_ev"] = float(mu_adjusted_for_ev)
            meta["mu_alpha"] = float(mu_alpha_for_ev)
        
        # ✅ [EV_VALIDATION 1] mu_alpha 자체가 음수/미약 검증 (pmaker 조정 후)
        mu_alpha_value = float(mu_alpha_parts.get("mu_alpha", 0.0))
        mu_alpha_mom = float(mu_alpha_parts.get("mu_mom", 0.0))
        mu_alpha_ofi = float(mu_alpha_parts.get("mu_ofi", 0.0))
        mu_alpha_warning = []
        if abs(mu_alpha_value) < 0.01:  # 1% annualized threshold
            mu_alpha_warning.append(f"mu_alpha={mu_alpha_value:.6f} is too weak (|mu_alpha|<0.01)")
        if mu_alpha_value < -5.0:
            mu_alpha_warning.append(f"mu_alpha={mu_alpha_value:.6f} is strongly negative (<-5.0)")
        if abs(mu_alpha_mom) < 0.005 and abs(mu_alpha_ofi) < 0.5:
            mu_alpha_warning.append(f"Both mu_mom={mu_alpha_mom:.6f} and mu_ofi={mu_alpha_ofi:.6f} are weak")
        if mu_alpha_warning:
            if MC_VERBOSE_PRINT or _throttled_log(symbol, "MU_ALPHA_ISSUES", 60_000):
                logger.warning(f"[EV_VALIDATION_1] {symbol} | mu_alpha issues: {'; '.join(mu_alpha_warning)}")
                if MC_VERBOSE_PRINT:
                    print(f"[EV_VALIDATION_1] {symbol} | mu_alpha issues: {'; '.join(mu_alpha_warning)}")
        
        leverage = _s(ctx.get("leverage", 5.0), 5.0)

        # OFI z-score by regime/session (abs) for slippage adverse factor
        key = (str(ctx.get("regime", "chop")), str(ctx.get("session", "OFF")))
        hist = self._ofi_hist.setdefault(key, [])
        hist.append(ofi_score)
        if len(hist) > 500:
            hist.pop(0)
        mean = 0.0
        std = 0.05
        ofi_z_abs = 0.0
        if len(hist) >= 5:
            arr = np.asarray(hist, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std())
            std = std if std > 1e-6 else 0.05
            ofi_z_abs = abs(ofi_score - mean) / std

        # 레버리지/변동성/유동성 기반 슬리피지 모델
        slippage_dyn_raw = self._estimate_slippage(leverage, sigma, liq_score, ofi_z_abs=ofi_z_abs)
        slippage_base_raw = self._estimate_slippage(1.0, sigma, liq_score, ofi_z_abs=ofi_z_abs)
        fee_fee_taker = float(self.fee_roundtrip_base)
        fee_fee_maker = float(getattr(self, "fee_roundtrip_maker_base", 0.0002))
        spread_pct = ctx.get("spread_pct")
        if spread_pct is None:
            spread_pct = 0.0002  # 2bp fallback
        try:
            spread_pct = float(spread_pct)
        except Exception:
            spread_pct = 0.0002
        spread_cap = config.spread_pct_max
        if spread_cap > 0.0 and spread_pct > spread_cap:
            spread_pct = spread_cap
        expected_spread_cost_raw = 0.5 * float(spread_pct) * 1.0  # adverse selection factor=1.0

        # maker_then_market 모드: 기대비용을 (maker/taker 혼합)으로 근사
        exec_mode = config.exec_mode
        p_maker = 0.0
        fee_fee_mix = fee_fee_taker
        slippage_dyn = float(slippage_dyn_raw)
        slippage_base = float(slippage_base_raw)
        expected_spread_cost = float(expected_spread_cost_raw)
        if exec_mode == "maker_then_market":
            p_maker = float(self._estimate_p_maker(spread_pct=float(spread_pct), liq_score=float(liq_score), ofi_z_abs=float(ofi_z_abs)))
            fee_fee_mix = float(p_maker * fee_fee_maker + (1.0 - p_maker) * fee_fee_taker)
            # maker fill 시 spread/slippage를 대부분 피한다고 가정(보수적으로 taker 비중만 반영)
            slippage_dyn = float((1.0 - p_maker) * float(slippage_dyn_raw))
            slippage_base = float((1.0 - p_maker) * float(slippage_base_raw))
            expected_spread_cost = float((1.0 - p_maker) * float(expected_spread_cost_raw))

        # --- PMAKER survival override (from orchestrator decision.meta) ---
        pmaker_entry = 0.0
        pmaker_entry_delay_sec = 0.0
        pmaker_entry_short = 0.0
        pmaker_entry_short_delay_sec = 0.0
        pmaker_exit = 0.0
        pmaker_exit_delay_sec = 0.0
        pmaker_override_used = False
        try:
            meta_in = ctx.get("meta") or {}
            pmaker_entry = float(ctx.get("pmaker_entry") or meta_in.get("pmaker_entry") or 0.0)
            if MC_VERBOSE_PRINT:
                print(
                    f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry={pmaker_entry:.4f} from ctx.get={ctx.get('pmaker_entry')} meta_in.get={meta_in.get('pmaker_entry')}"
                )
                logger.info(
                    f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry={pmaker_entry:.4f} from ctx.get={ctx.get('pmaker_entry')} meta_in.get={meta_in.get('pmaker_entry')}"
                )
        except Exception as e:
            if MC_VERBOSE_PRINT:
                print(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry read failed: {e}")
            logger.warning(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry read failed: {e}")
            pmaker_entry = 0.0
        try:
            meta_in = ctx.get("meta") or {}
            pmaker_entry_delay_sec = float(ctx.get("pmaker_entry_delay_sec") or meta_in.get("pmaker_entry_delay_sec") or 0.0)
            if MC_VERBOSE_PRINT:
                logger.info(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry_delay_sec={pmaker_entry_delay_sec:.4f}")
        except Exception as e:
            logger.warning(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_entry_delay_sec read failed: {e}")
            pmaker_entry_delay_sec = 0.0

        # --- Directional PMAKER (Short side) ---
        try:
            meta_in = ctx.get("meta") or {}
            pmaker_entry_short = float(ctx.get("pmaker_entry_short") or meta_in.get("pmaker_entry_short") or pmaker_entry)
        except Exception:
            pmaker_entry_short = pmaker_entry
        
        try:
            meta_in = ctx.get("meta") or {}
            pmaker_entry_short_delay_sec = float(ctx.get("pmaker_entry_short_delay_sec") or meta_in.get("pmaker_entry_short_delay_sec") or pmaker_entry_delay_sec)
        except Exception:
            pmaker_entry_short_delay_sec = pmaker_entry_delay_sec
        try:
            meta_in = ctx.get("meta") or {}
            pmaker_exit = float(ctx.get("pmaker_exit") or meta_in.get("pmaker_exit") or pmaker_entry)
            if MC_VERBOSE_PRINT:
                print(
                    f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit={pmaker_exit:.4f} from ctx.get={ctx.get('pmaker_exit')} meta_in.get={meta_in.get('pmaker_exit')}"
                )
                logger.info(
                    f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit={pmaker_exit:.4f} from ctx.get={ctx.get('pmaker_exit')} meta_in.get={meta_in.get('pmaker_exit')}"
                )
        except Exception as e:
            if MC_VERBOSE_PRINT:
                print(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit read failed: {e}")
            logger.warning(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit read failed: {e}")
            pmaker_exit = pmaker_entry
        try:
            meta_in = ctx.get("meta") or {}
            pmaker_exit_delay_sec = float(ctx.get("pmaker_exit_delay_sec") or meta_in.get("pmaker_exit_delay_sec") or pmaker_entry_delay_sec)
            if MC_VERBOSE_PRINT:
                logger.info(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit_delay_sec={pmaker_exit_delay_sec:.4f}")
        except Exception as e:
            logger.warning(f"[PMAKER_DEBUG] {symbol} | evaluate_entry_metrics: pmaker_exit_delay_sec read failed: {e}")
            pmaker_exit_delay_sec = pmaker_entry_delay_sec

        # Guard against NaN/Inf leaking from ctx/meta (would silently become null in JSON).
        try:
            if not math.isfinite(float(pmaker_entry)):
                pmaker_entry = 0.0
        except Exception:
            pmaker_entry = 0.0
        try:
            if not math.isfinite(float(pmaker_entry_delay_sec)):
                pmaker_entry_delay_sec = 0.0
        except Exception:
            pmaker_entry_delay_sec = 0.0
        try:
            if not math.isfinite(float(pmaker_exit)):
                pmaker_exit = float(pmaker_entry)
        except Exception:
            pmaker_exit = float(pmaker_entry)
        try:
            if not math.isfinite(float(pmaker_exit_delay_sec)):
                pmaker_exit_delay_sec = float(pmaker_entry_delay_sec)
        except Exception:
            pmaker_exit_delay_sec = float(pmaker_entry_delay_sec)


        if exec_mode == "maker_then_market" and pmaker_entry > 0.0:
            # Use PMaker's calibrated P(fill<=timeout) instead of heuristic p_maker.
            pmaker_entry = float(min(1.0, max(0.0, pmaker_entry)))
            p_maker = pmaker_entry
            fee_fee_mix = float(p_maker * fee_fee_maker + (1.0 - p_maker) * fee_fee_taker)
            slippage_dyn = float((1.0 - p_maker) * float(slippage_dyn_raw))
            slippage_base = float((1.0 - p_maker) * float(slippage_base_raw))
            expected_spread_cost = float((1.0 - p_maker) * float(expected_spread_cost_raw))
            pmaker_override_used = True

        # [DIFF 5 VALIDATION] Validate Maker → Market 혼합 실행 + 기대 비용 모델
        diff5_validation_warnings = []
        
        # Validation point 1: exec_mode should be "maker_then_market" when using maker mode
        if exec_mode == "maker_then_market":
            # Validation point 2: pmaker_entry should exist and be > 0 when using maker_then_market
            # ✅ [FIX] pmaker_entry_local이 아직 정의되지 않았으므로 pmaker_entry를 직접 사용
            if pmaker_entry <= 0.0:
                diff5_validation_warnings.append(
                    f"pmaker_entry={pmaker_entry:.6f} <= 0.0 in maker_then_market mode"
                )
            
            # Validation point 3: fee_roundtrip_fee_mix should be < fee_taker (maker mix should be cheaper)
            if fee_fee_mix >= fee_fee_taker:
                diff5_validation_warnings.append(
                    f"fee_roundtrip_fee_mix={fee_fee_mix:.8f} >= fee_taker={fee_fee_taker:.8f} "
                    f"(p_maker={p_maker:.4f} fee_maker={fee_fee_maker:.8f})"
                )
            elif p_maker > 0.0 and fee_fee_maker >= fee_fee_taker:
                # If p_maker > 0 but fee_maker >= fee_taker, then fee_mix cannot be < fee_taker
                diff5_validation_warnings.append(
                    f"fee_maker={fee_fee_maker:.8f} >= fee_taker={fee_fee_taker:.8f} "
                    f"but p_maker={p_maker:.4f} > 0 (fee_mix cannot be cheaper)"
                )
        
        # Additional validation: fee_roundtrip_fee_mix calculation
        if exec_mode == "maker_then_market" and p_maker > 0.0:
            fee_mix_expected = float(p_maker * fee_fee_maker + (1.0 - p_maker) * fee_fee_taker)
            fee_mix_diff = abs(fee_fee_mix - fee_mix_expected)
            if fee_mix_diff > 1e-6:
                diff5_validation_warnings.append(
                    f"fee_roundtrip_fee_mix calculation mismatch: computed={fee_fee_mix:.8f} "
                    f"expected={fee_mix_expected:.8f} diff={fee_mix_diff:.8f}"
                )
        
        # Log validation warnings
        if diff5_validation_warnings:
            logger.warning(
                f"[DIFF5_VALIDATION] {symbol} | exec_mode={exec_mode} "
                f"pmaker_entry={pmaker_entry:.6f} p_maker={p_maker:.4f} "
                f"fee_roundtrip_fee_mix={fee_fee_mix:.8f} fee_taker={fee_fee_taker:.8f} "
                f"fee_maker={fee_fee_maker:.8f} | Warnings: {'; '.join(diff5_validation_warnings)}"
            )
        
        # Store validation metrics in meta
        if diff5_validation_warnings:
            meta["diff5_validation_warnings"] = diff5_validation_warnings

        # 수수료는 레버리지와 무관하게 고정, 슬리피지는 노출(lev) 가중
        # NOTE: fee_roundtrip_total은 "roundtrip 총비용"으로 한 번만 구성해야 한다.
        # 기존 구현은 slippage_dyn/slippage_base를 fee_rt에 포함시킨 뒤 다시 더해 중복 합산되는 버그가 있었음.
        fee_rt = float(fee_fee_mix)  # (fee only; spread/slippage는 아래에서 합산)
        # gate용 baseline(lev=1)
        fee_rt_base = float(fee_fee_mix)  # (fee only; spread/slippage는 아래에서 합산)
        
        # ✅ Step C: 원인 분리 A/B 테스트 플래그
        _force_zero_cost = getattr(self, "_force_zero_cost", False)  # 비용 0으로 강제
        _force_horizon_600 = getattr(self, "_force_horizon_600", False)  # horizon 600초 고정
        
        if _force_zero_cost:
            fee_rt = 0.0
            fee_rt_base = 0.0
            expected_spread_cost = 0.0
            expected_spread_cost_raw = 0.0
            slippage_dyn = 0.0
            slippage_base = 0.0
            slippage_dyn_raw = 0.0
            slippage_base_raw = 0.0
            p_maker = 0.0
            fee_fee_mix = 0.0

        # ✅ total execution cost (roundtrip) used in net simulation
        # NOTE: simulate_paths_netpnl already subtracts fee_roundtrip_total inside the path net,
        # so we MUST NOT subtract it again later.
        # slippage/spread는 roundtrip 기준으로 1회만 합산한다(중복 합산 방지).
        fee_roundtrip_total = float(fee_rt) + float(expected_spread_cost) + float(slippage_dyn)
        fee_roundtrip_total_base = float(fee_rt_base) + float(expected_spread_cost) + float(slippage_base)

        # For logs/gates, treat fee_rt as the effective roundtrip execution cost.
        fee_rt = float(fee_roundtrip_total)
        fee_rt_base = float(fee_roundtrip_total_base)
        
        # ✅ 하드 캡 적용: 비정상적으로 높은 수수료 방지
        MAX_FEE_ROUNDTRIP = 0.003  # 0.3% 상한선
        fee_roundtrip_total = min(fee_roundtrip_total, MAX_FEE_ROUNDTRIP)
        fee_roundtrip_total_base = min(fee_roundtrip_total_base, MAX_FEE_ROUNDTRIP)
        
        execution_cost = float(fee_roundtrip_total)
        
        # ✅ 모든 심볼의 비용 검증 (LINK 제외, 비용이 너무 큰 경우)
        if not symbol.startswith("LINK"):
            # cost_entry는 fee_roundtrip_total의 절반이므로, fee_roundtrip_total이 크면 cost_entry도 큼
            # 일반적으로 fee_roundtrip_total은 0.001~0.002 정도여야 함 (0.1%~0.2%)
            if fee_roundtrip_total > 0.002:  # 0.2% 이상이면 비정상적으로 큼
                allow_log = MC_VERBOSE_PRINT or _throttled_log(symbol, "FEE_ROUNDTRIP_TOO_HIGH", 60_000)
                if allow_log:
                    if MC_VERBOSE_PRINT:
                        logger.warning(
                            f"[EV_VALIDATION_NEG] {symbol} | ⚠️  비정상적으로 큰 fee_roundtrip_total: {fee_roundtrip_total:.6f} (일반적으로 0.001~0.002)"
                        )
                        logger.warning(
                            f"[EV_VALIDATION_NEG] {symbol} | fee_rt={fee_rt:.6f} expected_spread_cost={expected_spread_cost:.6f} fee_fee_mix={fee_fee_mix:.6f} slippage_dyn={slippage_dyn:.6f}"
                        )
                        print(f"[EV_VALIDATION_NEG] {symbol} | ⚠️  비정상적으로 큰 fee_roundtrip_total: {fee_roundtrip_total:.6f}")
                        print(
                            f"[EV_VALIDATION_NEG] {symbol} | fee_rt={fee_rt:.6f} expected_spread_cost={expected_spread_cost:.6f} fee_fee_mix={fee_fee_mix:.6f} slippage_dyn={slippage_dyn:.6f}"
                        )
                    else:
                        logger.warning(
                            f"[EV_VALIDATION_NEG] {symbol} | fee_roundtrip_total={fee_roundtrip_total:.6f} fee_rt={fee_rt:.6f} spread={expected_spread_cost:.6f} slippage={slippage_dyn:.6f}"
                        )
        
        # ✅ Step C: horizon 선택
        horizons_for_sim = (600,) if _force_horizon_600 else self.horizons
        
        # ✅ 즉시 확인: fee_roundtrip_total (execution_cost) 값
        if MC_VERBOSE_PRINT:
            logger.info(
                f"[COST_QUICK] {symbol} | exec_mode={exec_mode} p_maker={p_maker:.3f} | "
                f"execution_cost={execution_cost:.6f} | fee_taker={fee_fee_taker:.6f} fee_maker={fee_fee_maker:.6f} fee_mix={fee_fee_mix:.6f} | "
                f"slippage_dyn_eff={slippage_dyn:.6f} spread_eff={expected_spread_cost:.6f} | "
                f"slippage_dyn_raw={float(slippage_dyn_raw):.6f} spread_raw={float(expected_spread_cost_raw):.6f} | "
                f"_force_zero_cost={_force_zero_cost}"
            )
            # [COST_BREAKDOWN] Enhanced version
            print(f"[COST_BREAKDOWN] {symbol} | fee_roundtrip_total={fee_roundtrip_total:.8f} | "
                  f"fee_fee_mix={fee_fee_mix:.8f} (p_maker={p_maker:.2f}, taker={fee_fee_taker:.6f}, maker={fee_fee_maker:.6f}) | "
                  f"expected_spread_cost={expected_spread_cost:.8f} (spread_pct={spread_pct:.6f}) | "
                  f"slippage_dyn={slippage_dyn:.8f} (slippage_dyn_raw={slippage_dyn_raw:.8f})")

            # [MU_ALPHA_VS_COST] Comparison at a reference horizon
            ref_h = 180
            ann_factor = SECONDS_PER_YEAR / ref_h
            net_exp_ann = mu_alpha_final - (fee_roundtrip_total * ann_factor)
            print(f"[MU_ALPHA_VS_COST] {symbol} | mu_alpha={mu_alpha_final:.4f} | Cost@180s={fee_roundtrip_total*ann_factor:.4f} | "
                  f"Net@180s={net_exp_ann:.4f} {'✅' if net_exp_ann > 0 else '✗'}")

            # ✅ 즉시 확인: self.horizons 값
            logger.info(
                f"[HORIZON_DEBUG] {symbol} | self.horizons={self.horizons} | horizons_for_sim={horizons_for_sim} | _force_horizon_600={_force_horizon_600}"
            )
            # ✅ Step 1: direction이 왜 SHORT로 잡히는지 확인
            logger.info(
                f"[DIR_DEBUG] {symbol} direction={direction} mu_adj={mu_adj:.10f} sigma={sigma:.10f} lev={leverage} dt={self.dt} fee_rt={fee_rt:.6f}"
            )
        if MC_VERBOSE_PRINT:
            print(f"[DIR_DEBUG] {symbol} direction={direction} mu_adj={mu_adj:.10f} sigma={sigma:.10f} lev={leverage} dt={self.dt} fee_rt={fee_rt:.6f}")

        # ✅ Step 2: LONG/SHORT 둘 다 평가
        t_start = time.perf_counter()
        step_sec = int(getattr(self, "time_step_sec", 1) or 1)
        step_sec = int(max(1, step_sec))
        n_paths = int(getattr(params, "n_paths", config.n_paths_live))
        
        # [A] Price paths generation (1st call)
        max_h_sec = int(max(horizons_for_sim)) if horizons_for_sim else 0
        max_steps = int(math.ceil(max_h_sec / float(step_sec))) if max_h_sec > 0 else 0
        t0_gen1 = time.perf_counter()
        price_paths = self.simulate_paths_price(
            seed=seed,
            s0=float(price),
            mu=float(mu_adj),
            sigma=float(sigma),
            n_paths=n_paths,
            n_steps=max_steps,
            dt=float(self.dt),
            return_torch=True,
        )
        t1_gen1 = time.perf_counter()

        # net (ROE) = direction * ((S_t - S0)/S0) * leverage - fee_roundtrip_total * leverage
        # NOTE: `fee_roundtrip_total` is per-notional (used by paper fills as % of notional).
        # To convert to ROE (per-margin), multiply by leverage.
        s0_f = float(price)
        lev_f = float(leverage)
        fee_rt_total_f = float(fee_roundtrip_total)
        fee_rt_total_roe_f = float(fee_rt_total_f) * float(lev_f)
        fee_rt_total_base_f = float(fee_roundtrip_total_base)

        # [B] GPU summary for horizons_for_sim (End-to-End JAX)
        t0_cpu_sum = time.perf_counter()
        h_cols = sorted(list(set(horizons_for_sim)))
        h_indices = np.array(
            [min(int(max_steps), int(math.ceil(int(h) / float(step_sec)))) for h in h_cols],
            dtype=np.int32,
        )
        policy_horizons_all = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", h_cols)) or list(h_cols)
        
        # Calculate stats (PyTorch/NumPy backend)
        try:
            res_summary = jax_backend.summarize_gbm_horizons_jax(
                price_paths=price_paths,
                s0=s0_f,
                leverage=lev_f,
                fee_rt_total_roe=fee_rt_total_roe_f,
                horizons_indices=h_indices,
                alpha=params.cvar_alpha,
            )
            res_summary_cpu = _summary_to_numpy(res_summary)
        except Exception as e:
            logger.warning(f"⚠️ [MC_FALLBACK] summarize_gbm_horizons failed: {e}. Falling back to NumPy.")
            from engines.mc.jax_backend import summarize_gbm_horizons_numpy
            res_summary_cpu = summarize_gbm_horizons_numpy(
                np.asarray(jax_backend.to_numpy(price_paths)),
                s0_f,
                lev_f,
                fee_rt_total_roe_f,
                h_indices,
                params.cvar_alpha,
            )
        
        ev_L_arr = res_summary_cpu["ev_long"]
        win_L_arr = res_summary_cpu["win_long"]
        cvar_L_arr = res_summary_cpu["cvar_long"]
        
        ev_S_arr = res_summary_cpu["ev_short"]
        win_S_arr = res_summary_cpu["win_short"]
        cvar_S_arr = res_summary_cpu["cvar_short"]

        net_by_h_long = {h_cols[i]: ev_L_arr[i] for i in range(len(h_cols))}
        net_by_h_short = {h_cols[i]: ev_S_arr[i] for i in range(len(h_cols))}
        
        # For compatibility with downstream logic (summarize returns tuples)
        best_idx_L = int(np.argmax(ev_L_arr))
        best_h_L = int(h_cols[best_idx_L])
        best_ev_L = float(ev_L_arr[best_idx_L])
        ev_L = float(np.mean(ev_L_arr))
        win_L = float(np.mean(win_L_arr))
        cvar_L = float(np.mean(cvar_L_arr))
        dbg_L = (ev_L_arr.tolist(), win_L_arr.tolist(), cvar_L_arr.tolist(), h_cols)

        best_idx_S = int(np.argmax(ev_S_arr))
        best_h_S = int(h_cols[best_idx_S])
        best_ev_S = float(ev_S_arr[best_idx_S])
        ev_S = float(np.mean(ev_S_arr))
        win_S = float(np.mean(win_S_arr))
        cvar_S = float(np.mean(cvar_S_arr))
        dbg_S = (ev_S_arr.tolist(), win_S_arr.tolist(), cvar_S_arr.tolist(), h_cols)
        
        t1_cpu_sum = time.perf_counter()


        # Sanity check: long/short symmetry under identical costs.
        # For each horizon: EV_long(h) + EV_short(h) should equal -2 * cost_roundtrip_ROE.
        # (Because gross_short = -gross_long and both subtract the same fee.)
        try:
            evL = dbg_L[0] if isinstance(dbg_L, tuple) and len(dbg_L) >= 1 else None
            evS = dbg_S[0] if isinstance(dbg_S, tuple) and len(dbg_S) >= 1 else None
            if isinstance(evL, list) and isinstance(evS, list) and len(evL) == len(evS) and len(evL) > 0:
                expected_sum = -2.0 * float(fee_rt_total_roe_f)
                max_err = 0.0
                for a, b in zip(evL, evS):
                    try:
                        err = abs((float(a) + float(b)) - expected_sum)
                        if err > max_err:
                            max_err = err
                    except Exception:
                        continue
                if max_err > 1e-6 and (MC_VERBOSE_PRINT or _throttled_log(symbol, "EV_SYMMETRY_FAIL", 60_000)):
                    logger.warning(
                        f"[EV_SANITY] {symbol} | long+short symmetry off: max_err={max_err:.6e} expected_sum={expected_sum:.6e}"
                    )
        except Exception:
            pass

        policy_horizons = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", (120, 180, 300)))
        if not policy_horizons:
            policy_horizons = [int(getattr(self, "POLICY_HORIZON_SEC", 1800))]
        max_policy_h = int(max(policy_horizons)) if policy_horizons else int(getattr(self, "POLICY_HORIZON_SEC", 1800))
        
        # ✅ [EV_DEBUG] policy_horizons 확인 (성능 개선: MC_VERBOSE_PRINT로 조건부)
        if MC_VERBOSE_PRINT:
            logger.info(f"[EV_DEBUG] {symbol} | policy_horizons={policy_horizons} max_policy_h={max_policy_h}")
            print(f"[EV_DEBUG] {symbol} | policy_horizons={policy_horizons} max_policy_h={max_policy_h}")

        # ✅ Check 1) horizon별로 path를 새로 생성하지 않도록, 최대 horizon 길이 경로를 1번만 만들고 재사용
        # - 기본: 이미 생성한 price_paths(hold/horizon용)를 재사용 (동일하게 1초 스텝 경로임)
        # - 단, 디버그로 horizons_for_sim이 짧아졌을 때(_force_horizon_600), policy horizon이 더 길면 별도 생성
        t0_gen2 = time.perf_counter()
        price_paths_policy = price_paths
        paths_reused = True
        paths_seed_base = int(seed)
        max_policy_steps = int(math.ceil(int(max_policy_h) / float(step_sec))) if max_policy_h > 0 else 0
        if int(price_paths_policy.shape[1]) < int(max_policy_steps + 1):
            price_paths_policy = self.simulate_paths_price(
                seed=paths_seed_base,
                s0=float(price),
                mu=float(mu_adj),
                sigma=float(sigma),
                n_paths=int(params.n_paths),
                n_steps=int(max_policy_steps),
                dt=float(self.dt),
                return_torch=True,
            )
            paths_reused = False
        t1_gen2 = time.perf_counter()


        momentum_z = 0.0
        if closes is not None and len(closes) >= 6:
            rets = np.diff(np.log(np.asarray(closes, dtype=np.float64)))
            if rets.size >= 5:
                window = int(min(20, rets.size))
                subset = rets[-window:]
                mean_r = float(subset.mean())
                std_r = float(subset.std())
                if std_r > 1e-9:
                    momentum_z = float((subset[-1] - mean_r) / std_r)

        ofi_z = 0.0
        if len(hist) >= 5:
            std_local = float(max(std, 1e-9))
            ofi_z = float((ofi_score - mean) / std_local) if std_local > 1e-9 else 0.0

        signal_strength = float(abs(momentum_z) + 0.7 * abs(ofi_z))
        s_clip = float(np.clip(signal_strength, 0.0, 4.0))
        base_half_life = config.policy_w_prior_half_life_base_sec
        policy_half_life_sec = float(max(1.0, base_half_life)) / (1.0 + s_clip)
        
        # Compute prior weights (rule-based) - used as base for EV-based weights
        w_prior = self._weights_for_horizons(policy_horizons, signal_strength)
        
        # [D] AlphaHitMLP: Predict horizon-specific TP/SL probabilities using OnlineAlphaTrainer
        alpha_hit = None
        features_np = None
        alpha_hit_features_np = None
        alpha_hit_buffer_n = 0
        alpha_hit_loss = None
        alpha_hit_beta_eff = 0.0
        if self.alpha_hit_enabled and self.alpha_hit_trainer is not None:
            try:
                features = self._extract_alpha_hit_features(
                    symbol=symbol,
                    price=price,
                    mu=mu_adj,
                    sigma=sigma,
                    momentum_z=momentum_z,
                    ofi_z=ofi_z,
                    regime=regime_ctx,
                    leverage=leverage,
                    ctx=ctx,
                )
                if features is not None:
                    features_np = features[0].detach().cpu().numpy()  # [F]
                    alpha_hit_features_np = features_np
                    # Use trainer's predict method
                    pred = self.alpha_hit_trainer.predict(features_np)
                    # Convert to 1D arrays aligned with horizons (avoid [1, H] len==1 bug)
                    alpha_hit = {
                        "p_tp_long": pred["p_tp_long"].detach().cpu().numpy().reshape(-1),
                        "p_sl_long": pred["p_sl_long"].detach().cpu().numpy().reshape(-1),
                        "p_tp_short": pred["p_tp_short"].detach().cpu().numpy().reshape(-1),
                        "p_sl_short": pred["p_sl_short"].detach().cpu().numpy().reshape(-1),
                    }
                    # Train on previous samples (async, non-blocking)
                    try:
                        train_stats = self.alpha_hit_trainer.train_tick()
                        try:
                            alpha_hit_buffer_n = int(getattr(self.alpha_hit_trainer, "buffer_size", 0))
                        except Exception:
                            alpha_hit_buffer_n = 0
                        if alpha_hit_buffer_n <= 0:
                            try:
                                alpha_hit_buffer_n = int(train_stats.get("buffer_n", train_stats.get("n_samples", 0)) or 0)
                            except Exception:
                                alpha_hit_buffer_n = 0
                        alpha_hit_loss = train_stats.get("loss")
                        if alpha_hit_loss is not None:
                            logger.debug(f"[ALPHA_HIT] Training loss: {alpha_hit_loss:.6f}, buffer: {alpha_hit_buffer_n}")
                    except Exception as e:
                        logger.warning(f"[ALPHA_HIT] Training failed: {e}")

                    # Warmup: don't trust random-init AlphaHit preds until at least one train step happened.
                    min_buf = config.alpha_hit_min_buffer
                    try:
                        base_beta = float(getattr(self, "alpha_hit_beta", 1.0))
                    except Exception:
                        base_beta = 1.0
                    base_beta = float(np.clip(base_beta, 0.0, 1.0))
                    warm = 1.0
                    max_loss = config.alpha_hit_max_loss
                    if min_buf > 0:
                        warm = min(1.0, float(alpha_hit_buffer_n) / float(min_buf))
                    if alpha_hit_loss is None or (not math.isfinite(float(alpha_hit_loss))) or float(alpha_hit_loss) > float(max_loss):
                        warm = 0.0
                    alpha_hit_beta_eff = float(base_beta * warm)
            except Exception as e:
                logger.warning(f"[ALPHA_HIT] Failed to predict: {e}")
                alpha_hit = None
        # [D] TP/SL 확률 기반 EV 계산 (완전 전환)
        # --- NEW: alpha hit model predictions (per horizon, long/short) ---
        # Costs: maker survival + delay are supplied per symbol (from LiveOrchestrator)
        # Note: pmaker_entry, pmaker_entry_delay_sec, pmaker_exit, pmaker_exit_delay_sec are already read above (line 2270-2294)
        # We use them here for delay penalty calculation
        # ✅ Use already-read values from line 2301-2314 instead of re-reading from ctx
        pmaker_entry_local = float(pmaker_entry)  # Use value read at line 2301
        pmaker_delay_sec_local = float(pmaker_entry_delay_sec)  # Use value read at line 2310
        
        # [C] PMaker delay penalty: k * sigma_per_sqrt_sec * sqrt(delay)
        # (approximates adverse selection / drift uncertainty during waiting)
        delay_penalty_k = config.pmaker_delay_penalty_k
        # sigma_bar is per-sec vol proxy (from meta or computed)
        mu_sigma_meta = ctx.get("meta", {})
        sigma_bar = float(mu_sigma_meta.get("sigma_bar", 0.0) or 0.0)
        if sigma_bar <= 0:
            # Fallback: use annualized sigma converted to per-second
            sigma_bar = float(sigma) / math.sqrt(float(SECONDS_PER_YEAR))
        delay_penalty = delay_penalty_k * sigma_bar * math.sqrt(max(0.0, pmaker_delay_sec_local))
        
        # [D] TP/SL EV per horizon
        # TP table per horizon (horizon-specific TP returns)
        tp_r_by_h = getattr(self, "TP_R_BY_H", {})
        sl_r_by_h = getattr(self, "SL_R_BY_H", {})
        # Trailing SL is ATR based; in EV space we approximate SL_r as risk-bound
        sl_r_fixed = config.sl_r_fixed
        atr_frac = float(ctx.get("atr_frac", 0.0) or 0.0)  # ATR/price from orchestrator
        sl_atr_mult = config.trail_atr_mult
        sl_r = (sl_atr_mult * atr_frac) if atr_frac > 0 else sl_r_fixed
        
        # ✅ [EV_VALIDATION_3] 비용 분리: entry 1회 + exit 1회로 분리하여 한 번만 차감
        # fee_rt_total_f는 roundtrip 비용 (entry + exit 합계)
        # Entry 비용: fee_entry + spread_entry + slippage_entry (1회만 차감)
        # Exit 비용: fee_exit + spread_exit + slippage_exit (1회만 차감)
        # 계획: entry 비용은 여기서 차감, exit 비용은 compute_exit_policy_metrics에서 exec_oneway로 차감
        # 따라서 compute_exit_policy_metrics에는 exit 비용만 전달 (fee_roundtrip에 exit 비용만)
        fee_roundtrip_total = float(fee_rt_total_f)
        # Entry 비용: roundtrip의 절반 (entry fee + spread_entry + slippage_entry)
        # 실제로는 fee_rt_total_f가 이미 entry+exit 합계이므로, 절반을 entry로 사용
        cost_entry = fee_roundtrip_total / 2.0  # Entry 비용 (roundtrip의 절반) [per-notional]
        cost_exit = fee_roundtrip_total / 2.0  # Exit 비용 (roundtrip의 절반) [per-notional]
        cost_entry_roe = float(cost_entry) * float(leverage)
        cost_exit_roe = float(cost_exit) * float(leverage)
        # delay_penalty는 horizon별로 스케일링되어 적용됨 (아래에서 처리)
        
        # ✅ [EV_VALIDATION_3] maker delay + spread + slippage가 gross EV를 잠식 검증
        # Gross EV approximation: mu_adj * leverage * time_horizon (rough estimate for 5min; short-term 0~5m)
        gross_ev_approx_5min = float(mu_adj) / float(SECONDS_PER_YEAR) * float(leverage) * 300.0  # 5min
        cost_fraction = cost_entry_roe / max(1e-6, abs(gross_ev_approx_5min)) if gross_ev_approx_5min != 0.0 else float("inf")
        cost_warning_val = None
        if cost_fraction > 0.5:  # 비용이 gross EV의 50% 이상
            cost_warning_val = f"costs are {cost_fraction*100:.1f}% of gross_ev_approx_5min (cost_entry_roe={cost_entry_roe:.6f}, gross={gross_ev_approx_5min:.6f})"
            if MC_VERBOSE_PRINT or _throttled_log(symbol, "EV_VALIDATION_3_COST", 60_000):
                logger.warning(f"[EV_VALIDATION_3] {symbol} | {cost_warning_val}")
                if MC_VERBOSE_PRINT:
                    print(f"[EV_VALIDATION_3] {symbol} | {cost_warning_val}")
            # Store in meta (meta will be initialized/accessed later)
            meta["ev_validation_3_warning"] = cost_warning_val
        
        # [D] Compute EV_h for long/short based on predicted hit probs
        ev_long_h = []
        ev_short_h = []
        ppos_long_h = []
        ppos_short_h = []
        
        # Keep MC simulation results for meta/diagnostics (exit-policy rollforward)
        evs_long = []
        evs_short = []
        pps_long = []
        pps_short = []
        cvars_long = []
        cvars_short = []
        per_h_long = []
        per_h_short = []
        mc_p_tp_long = []
        mc_p_sl_long = []
        mc_p_tp_short = []
        mc_p_sl_short = []

        # compute_exit_policy_metrics expects annualized (per-year) μ/σ.
        # (It converts to per-second internally using self.dt.)
        mu_exit = float(mu_adj)
        sigma_exit = float(sigma)
        impact_cost = 0.0
        exec_oneway = float(execution_cost) * float(leverage) / 2.0  # ROE one-way cost
        
        # [C] PMaker survival function and delay penalty per horizon
        pmaker_surv = ctx.get("pmaker_surv")  # PMakerSurvivalMLP instance from orchestrator
        pmaker_timeout_ms = int(ctx.get("pmaker_timeout_ms", 1500))  # 1.5s default
        pmaker_features = ctx.get("pmaker_features")  # Features for PMaker prediction
        
        # [DIFF 6] Prepare decision_meta for compute_exit_policy_metrics (pmaker_entry/exit from ctx)
        # We will create these inside the horizon loop to allow for long/short directional differentiation.
        # Use already-read values from evaluate_entry_metrics (line 2270-2294)
        # These are already extracted from ctx, so we can use them directly
        decision_meta_for_exit_long = {}
        # Always include pmaker_entry/exit even if 0 (compute_exit_policy_metrics needs to know they exist)
        if pmaker_entry is not None:
            decision_meta_for_exit_long["pmaker_entry"] = float(pmaker_entry)
        if pmaker_entry_delay_sec is not None:
            decision_meta_for_exit_long["pmaker_entry_delay_sec"] = float(pmaker_entry_delay_sec)
        if pmaker_exit is not None:
            decision_meta_for_exit_long["pmaker_exit"] = float(pmaker_exit)
        if pmaker_exit_delay_sec is not None:
            decision_meta_for_exit_long["pmaker_exit_delay_sec"] = float(pmaker_exit_delay_sec)
        decision_meta_for_exit_long["fee_roundtrip_is_exit_only"] = True

        decision_meta_for_exit_short = {}
        # Use short-side directional pmaker values for the short side.
        if pmaker_entry_short is not None:
            decision_meta_for_exit_short["pmaker_entry"] = float(pmaker_entry_short)
        if pmaker_entry_short_delay_sec is not None:
            decision_meta_for_exit_short["pmaker_entry_delay_sec"] = float(pmaker_entry_short_delay_sec)
        if pmaker_exit is not None: # Note: exit is usually taker or shared logic, but we could differentiate if needed
            decision_meta_for_exit_short["pmaker_exit"] = float(pmaker_exit)
        if pmaker_exit_delay_sec is not None:
            decision_meta_for_exit_short["pmaker_exit_delay_sec"] = float(pmaker_exit_delay_sec)
        decision_meta_for_exit_short["fee_roundtrip_is_exit_only"] = True
        if MC_VERBOSE_PRINT:
            print(f"[PMAKER_DEBUG] {symbol} | decision_meta_for_exit_long={decision_meta_for_exit_long} short={decision_meta_for_exit_short}")
            logger.info(f"[PMAKER_DEBUG] {symbol} | decision_meta_for_exit_long={decision_meta_for_exit_long} short={decision_meta_for_exit_short}")
        
        # ✅ [FIX] meta는 이미 초기화됨 (line 2229에서)
        # meta = {}  # 이미 초기화되어 있으므로 주석 처리
        
        # [D] Compute TP/SL 확률 기반 EV_h for each horizon
        # ✅ [EV_DEBUG] alpha_hit 상태 확인 (성능 개선: MC_VERBOSE_PRINT로 조건부)
        if MC_VERBOSE_PRINT:
            logger.info(f"[EV_DEBUG] {symbol} | alpha_hit_enabled={self.alpha_hit_enabled} alpha_hit_trainer={self.alpha_hit_trainer is not None} alpha_hit={alpha_hit is not None}")
            print(f"[EV_DEBUG] {symbol} | alpha_hit_enabled={self.alpha_hit_enabled} alpha_hit_trainer={self.alpha_hit_trainer is not None} alpha_hit={alpha_hit is not None}")
            if alpha_hit is None:
                logger.warning(f"[EV_DEBUG] {symbol} | alpha_hit is None - will use compute_exit_policy_metrics results only")
                print(f"[EV_DEBUG] {symbol} | alpha_hit is None - will use compute_exit_policy_metrics results only")

        def _normalize_tp_sl(p_tp: float, p_sl: float) -> tuple[float, float, float]:
            """
            Keep TP/SL probabilities in [0,1] and on the simplex: p_tp + p_sl + p_other == 1.
            AlphaHit predicts TP/SL independently, so p_tp+p_sl can exceed 1 without this.
            """
            try:
                tp = float(np.clip(float(p_tp), 0.0, 1.0))
            except Exception:
                tp = 0.0
            try:
                sl = float(np.clip(float(p_sl), 0.0, 1.0))
            except Exception:
                sl = 0.0
            s = tp + sl
            if s > 1.0:
                tp = tp / s
                sl = sl / s
                s = 1.0
            other = max(0.0, 1.0 - s)
            return tp, sl, other
        
        # ✅ [EV_DEBUG] horizon 루프 시작 확인 (성능 개선: MC_VERBOSE_PRINT로 조건부)
        if MC_VERBOSE_PRINT:
            logger.info(f"[EV_DEBUG] {symbol} | Starting horizon loop: policy_horizons={policy_horizons} len={len(policy_horizons)}")
            print(f"[EV_DEBUG] {symbol} | Starting horizon loop: policy_horizons={policy_horizons} len={len(policy_horizons)}")
        
        # Debug: Confirm horizon loop starts
        if MC_VERBOSE_PRINT:
            print(f"[HORIZON_LOOP_DEBUG] {symbol} | Starting horizon loop with {len(policy_horizons)} horizons: {policy_horizons}")
        
        
        # ✅ JAX VMAP OPTIMIZATION: Prepare batched inputs for all horizons and directions
        # This allows running all horizons for both directions in a SINGLE JAX call on GPU.
        batch_horizons = []
        batch_directions = []
        tp_target_roe_batch = []
        sl_target_roe_batch = []
        dd_stop_roe_batch = []
        
        # We'll use a fixed drawdown stop ROE for the batch (usually -0.01 or from environment)
        dd_stop_roe_val = float(getattr(base_config, "PAPER_EXIT_POLICY_DD_STOP_ROE", -0.01))
        
        for idx, h in enumerate(policy_horizons):
            tp_r = float(tp_r_by_h.get(int(h), tp_r_by_h.get(str(int(h)), 0.0)))
            
            # Long direction
            batch_horizons.append(h)
            batch_directions.append(1)
            tp_target_roe_batch.append(float(tp_r * leverage))
            sl_target_roe_batch.append(float(sl_r * leverage))
            dd_stop_roe_batch.append(dd_stop_roe_val)
            
            # Short direction
            batch_horizons.append(h)
            batch_directions.append(-1)
            tp_target_roe_batch.append(float(tp_r * leverage))
            sl_target_roe_batch.append(float(sl_r * leverage))
            dd_stop_roe_batch.append(dd_stop_roe_val)

        if MC_VERBOSE_PRINT or _throttled_log(symbol, "JAX_SHAPE_DEBUG", 60_000):
            print(f"[SHAPE_DEBUG] {symbol} | price_paths_policy shape={getattr(price_paths_policy, 'shape', 'N/A')} mu={mu_exit:.6f} sigma={sigma_exit:.6f}")
        
        # Call the batched version (massive parallelization on GPU)
        t0_jax = time.perf_counter()
        batch_results_list = self.compute_exit_policy_metrics_batched(
            symbol=symbol,
            price=price,
            mu=mu_exit,
            sigma=sigma_exit,
            leverage=leverage,
            fee_roundtrip=cost_exit_roe,
            exec_oneway=exec_oneway,
            impact_cost=impact_cost,
            regime=regime_ctx,
            batch_directions=np.array(batch_directions),
            batch_horizons=np.array(batch_horizons),
            price_paths=price_paths_policy,
            cvar_alpha=params.cvar_alpha,
            tp_target_roe_batch=np.array(tp_target_roe_batch),
            sl_target_roe_batch=np.array(sl_target_roe_batch),
            enable_dd_stop=True,
            dd_stop_roe_batch=np.array(dd_stop_roe_batch),
        )
        t1_jax = time.perf_counter()
        # Detailed batch perf logging
        try:
            # Unconditional perf print for batch internals (always emit)
            total = time.perf_counter() - t_start
            print(f"[BATCH_PERF] gen={(t1_gen-t0_gen):.3f}s sum={(t1_sum-t1_gen):.3f}s exit_jax={(t1_jax-t0_jax):.3f}s total={total:.3f}s num_symbols={num_symbols} n_paths={n_paths}")
        except Exception:
            pass
        if MC_VERBOSE_PRINT or (t1_jax - t0_jax) > 1.0:
            print(f"[RE-VAMP_DEBUG] {symbol} | compute_exit_policy_metrics_batched took {(t1_jax - t0_jax)*1000:.2f}ms")
            logger.info(f"[RE-VAMP_DEBUG] {symbol} | compute_exit_policy_metrics_batched took {(t1_jax - t0_jax)*1000:.2f}ms")





        t_end = time.perf_counter()
        if MC_VERBOSE_PRINT or (t_end - t_start) > 0.5:
            msg = (f"[PERF_BREAKDOWN] {symbol} | Total: {(t_end - t_start)*1000:.1f}ms | "
                   f"Gen1: {(t1_gen1 - t0_gen1)*1000:.1f}ms | "
                   f"CPUSum: {(t1_cpu_sum - t0_cpu_sum)*1000:.1f}ms | "
                   f"Gen2: {(t1_gen2 - t0_gen2)*1000:.1f}ms | "
                   f"JAXBat: {(t1_jax - t0_jax)*1000:.2f}ms")
            print(msg)
            logger.info(msg)

        # ✅ FORCE CLEAN EVALUATOR (No fallback)
        print(f"[DEBUG] Force-using clean evaluator. _CLEAN_EVALUATOR_AVAILABLE={_CLEAN_EVALUATOR_AVAILABLE}")
        
        # Use clean evaluator to process horizons correctly
        clean_eval = get_clean_evaluator()
        
        # Convert price_paths_policy to numpy if needed
        if hasattr(price_paths_policy, 'block_until_ready'):
            price_paths_np = np.array(price_paths_policy)
        else:
            price_paths_np = price_paths_policy
        
        # Prepare TP/SL targets per horizon
        tp_targets_clean = []
        sl_targets_clean = []
        for h in policy_horizons:
            tp_r_h = float(tp_r_by_h.get(int(h), tp_r_by_h.get(str(int(h)), tp_r)))
            sl_r_h = float(sl_r_by_h.get(int(h), sl_r_by_h.get(str(int(h)), sl_r)))
            tp_targets_clean.append(tp_r_h * leverage)
            sl_targets_clean.append(sl_r_h * leverage)
        
        # Evaluate horizons
        horizon_results = clean_eval.evaluate_horizons(
            price_paths_np,
            policy_horizons,
            leverage,
            cost_exit_roe,
            tp_targets_clean,
            sl_targets_clean,
            params.cvar_alpha,
        )
        
        # Extract results
        ev_long_h = horizon_results['ev_long_h']
        ev_short_h = horizon_results['ev_short_h']
        ppos_long_h = horizon_results['p_pos_long_h']
        ppos_short_h = horizon_results['p_pos_short_h']
        cvar_long_h = horizon_results['cvar_long_h']
        cvar_short_h = horizon_results['cvar_short_h']
        mc_p_tp_long = horizon_results['p_tp_long']
        mc_p_sl_long = horizon_results['p_sl_long']
        mc_p_tp_short = horizon_results['p_tp_short']
        mc_p_sl_short = horizon_results['p_sl_short']
        
        # Apply entry cost
        ev_long_h = [ev - cost_entry_roe for ev in ev_long_h]
        ev_short_h = [ev - cost_entry_roe for ev in ev_short_h]
        cvar_long_h = [cv - cost_entry_roe for cv in cvar_long_h]
        cvar_short_h = [cv - cost_entry_roe for cv in cvar_short_h]
        
        # Convert to numpy arrays for downstream code
        ev_long_h = np.array(ev_long_h)
        ev_short_h = np.array(ev_short_h)
        ppos_long_h = np.array(ppos_long_h)
        ppos_short_h = np.array(ppos_short_h)
        cvar_long_h = np.array(cvar_long_h)
        cvar_short_h = np.array(cvar_short_h)

        # AlphaHit blending (EV recompute using blended TP/SL probabilities)
        if alpha_hit is not None:
            try:
                tp_targets_clean = [float(x) for x in tp_targets_clean]
                sl_targets_clean = [float(x) for x in sl_targets_clean]
                H = len(policy_horizons)
                ev_long_h_new = np.asarray(ev_long_h, dtype=np.float64).copy()
                ev_short_h_new = np.asarray(ev_short_h, dtype=np.float64).copy()
                ppos_long_h_new = np.asarray(ppos_long_h, dtype=np.float64).copy()
                ppos_short_h_new = np.asarray(ppos_short_h, dtype=np.float64).copy()

                for i in range(H):
                    tp_r = float(tp_targets_clean[i]) if i < len(tp_targets_clean) else 0.0
                    sl_r_raw = float(sl_targets_clean[i]) if i < len(sl_targets_clean) else 0.0
                    sl_r = -abs(sl_r_raw) if sl_r_raw >= 0 else float(sl_r_raw)

                    # MC probs from clean evaluator
                    p_tp_L_mc = float(mc_p_tp_long[i]) if i < len(mc_p_tp_long) else 0.0
                    p_sl_L_mc = float(mc_p_sl_long[i]) if i < len(mc_p_sl_long) else 0.0
                    p_tp_S_mc = float(mc_p_tp_short[i]) if i < len(mc_p_tp_short) else 0.0
                    p_sl_S_mc = float(mc_p_sl_short[i]) if i < len(mc_p_sl_short) else 0.0

                    # AlphaHit probs
                    tp_pL_pred = float(alpha_hit["p_tp_long"][i]) if i < len(alpha_hit["p_tp_long"]) else 0.0
                    sl_pL_pred = float(alpha_hit["p_sl_long"][i]) if i < len(alpha_hit["p_sl_long"]) else 0.0
                    tp_pS_pred = float(alpha_hit["p_tp_short"][i]) if i < len(alpha_hit["p_tp_short"]) else 0.0
                    sl_pS_pred = float(alpha_hit["p_sl_short"][i]) if i < len(alpha_hit["p_sl_short"]) else 0.0

                    # Normalize
                    tp_pL_pred, sl_pL_pred, _ = _normalize_tp_sl(tp_pL_pred, sl_pL_pred)
                    tp_pS_pred, sl_pS_pred, _ = _normalize_tp_sl(tp_pS_pred, sl_pS_pred)
                    p_tp_L_mc, p_sl_L_mc, _ = _normalize_tp_sl(p_tp_L_mc, p_sl_L_mc)
                    p_tp_S_mc, p_sl_S_mc, _ = _normalize_tp_sl(p_tp_S_mc, p_sl_S_mc)

                    beta_eff = float(alpha_hit_beta_eff) if alpha_hit_beta_eff is not None else 0.0
                    beta_eff = float(np.clip(beta_eff, 0.0, 1.0))
                    conf_L = float(self.alpha_hit_confidence(tp_pL_pred, sl_pL_pred))
                    conf_S = float(self.alpha_hit_confidence(tp_pS_pred, sl_pS_pred))
                    beta_eff_L = float(np.clip(beta_eff * conf_L, 0.0, 1.0))
                    beta_eff_S = float(np.clip(beta_eff * conf_S, 0.0, 1.0))

                    tp_pL_val = beta_eff_L * tp_pL_pred + (1.0 - beta_eff_L) * p_tp_L_mc
                    sl_pL_val = beta_eff_L * sl_pL_pred + (1.0 - beta_eff_L) * p_sl_L_mc
                    tp_pS_val = beta_eff_S * tp_pS_pred + (1.0 - beta_eff_S) * p_tp_S_mc
                    sl_pS_val = beta_eff_S * sl_pS_pred + (1.0 - beta_eff_S) * p_sl_S_mc

                    tp_pL_val, sl_pL_val, p_other_L = _normalize_tp_sl(tp_pL_val, sl_pL_val)
                    tp_pS_val, sl_pS_val, p_other_S = _normalize_tp_sl(tp_pS_val, sl_pS_val)

                    # Implied other_r from MC mean (net of entry cost)
                    evL = float(ev_long_h[i]) if i < len(ev_long_h) else 0.0
                    evS = float(ev_short_h[i]) if i < len(ev_short_h) else 0.0
                    denom_L = max(1e-12, p_other_L)
                    denom_S = max(1e-12, p_other_S)
                    other_r_L = (evL + float(cost_entry_roe) - (tp_pL_val * tp_r + sl_pL_val * sl_r)) / denom_L
                    other_r_S = (evS + float(cost_entry_roe) - (tp_pS_val * tp_r + sl_pS_val * sl_r)) / denom_S

                    ev_long_h_new[i] = tp_pL_val * tp_r + sl_pL_val * sl_r + p_other_L * other_r_L - float(cost_entry_roe)
                    ev_short_h_new[i] = tp_pS_val * tp_r + sl_pS_val * sl_r + p_other_S * other_r_S - float(cost_entry_roe)
                    ppos_long_h_new[i] = tp_pL_val
                    ppos_short_h_new[i] = tp_pS_val

                ev_long_h = ev_long_h_new
                ev_short_h = ev_short_h_new
                ppos_long_h = ppos_long_h_new
                ppos_short_h = ppos_short_h_new
            except Exception as e:
                logger.warning(f"[ALPHA_HIT] Blend failed: {e}")

        # Mock per_h_long/short to prevent NaNs in downstream calculations
        # (Since current CleanEvaluator does not calculate variance/liquidation yet)
        mock_metrics = {'var_exit_policy': 0.0, 'p_liq_exit_policy': 0.0, 'dd_min_exit_policy': 0.0, 'profit_cost_exit_policy': 999.0}
        per_h_long = [mock_metrics.copy() for _ in policy_horizons]
        per_h_short = [mock_metrics.copy() for _ in policy_horizons]
        
        # Always refresh legacy helpers with CleanEvaluator outputs (needed for unified scoring + hold stats)
        evs_long = ev_long_h
        evs_short = ev_short_h
        pps_long = ppos_long_h
        pps_short = ppos_short_h
        cvars_long = cvar_long_h
        cvars_short = cvar_short_h
        dbg_L = (ev_long_h.tolist(), ppos_long_h.tolist(), cvar_long_h.tolist(), policy_horizons)
        dbg_S = (ev_short_h.tolist(), ppos_short_h.tolist(), cvar_short_h.tolist(), policy_horizons)
        net_by_h_long_base = [None] * len(policy_horizons)
        net_by_h_short_base = [None] * len(policy_horizons)
        net_by_h_long = [None] * len(policy_horizons)
        net_by_h_short = [None] * len(policy_horizons)

        if MC_VERBOSE_PRINT:
            print(f"[CLEAN_EVAL] {symbol} | Processed {len(policy_horizons)} horizons successfully")
            print(f"[CLEAN_EVAL] {symbol} | ev_long_h shape: {ev_long_h.shape}, values: {ev_long_h}")


        if self.alpha_hit_enabled and self.alpha_hit_trainer is not None and features_np is not None:
            try:
                y_tp_long = np.clip(np.asarray(mc_p_tp_long, dtype=np.float32), 0.0, 1.0)
                y_sl_long = np.clip(np.asarray(mc_p_sl_long, dtype=np.float32), 0.0, 1.0)
                y_tp_short = np.clip(np.asarray(mc_p_tp_short, dtype=np.float32), 0.0, 1.0)
                y_sl_short = np.clip(np.asarray(mc_p_sl_short, dtype=np.float32), 0.0, 1.0)
                if len(y_tp_long) == len(policy_horizons):
                    self.alpha_hit_trainer.add_sample(
                        x=features_np,
                        y={
                            "tp_long": y_tp_long,
                            "sl_long": y_sl_long,
                            "tp_short": y_tp_short,
                            "sl_short": y_sl_short,
                        },
                        ts_ms=int(time.time() * 1000),
                        symbol=symbol,
                    )
            except Exception as e:
                logger.warning(f"[ALPHA_HIT] Failed to add MC soft-label sample: {e}")
        
        # [D] Convert to numpy arrays
        ev_long_h = np.asarray(ev_long_h, dtype=np.float64)
        ev_short_h = np.asarray(ev_short_h, dtype=np.float64)
        ppos_long_h = np.asarray(ppos_long_h, dtype=np.float64)
        ppos_short_h = np.asarray(ppos_short_h, dtype=np.float64)
        
        # [D] Learned contribution -> w(h) uses EV_h (but keep rule-based half-life prior)
        w_prior_arr = np.asarray(w_prior, dtype=np.float64)
        # EV shaping: positive-only contribution (avoid negative EV horizon hijacking weight)
        beta = config.policy_w_ev_beta
        contrib_long = np.log1p(np.exp(ev_long_h * beta))
        contrib_short = np.log1p(np.exp(ev_short_h * beta))
        
        # ✅ [EV_VALIDATION_2] exit 분포 기반 페널티 계산 및 적용
        # 페널티: p(exit <= min_hold+Δ) 또는 exit_reason_counts(hold_bad/flip 비율) 기반
        min_hold_sec_val = float(self.MIN_HOLD_SEC_DIRECTIONAL)
        exit_early_delta_sec = config.exit_early_delta_sec
        exit_early_penalty_k = config.exit_early_penalty_k
        
        penalty_long = np.ones(len(per_h_long), dtype=np.float64)
        penalty_short = np.ones(len(per_h_short), dtype=np.float64)
        
        for i, (m_long_h, m_short_h) in enumerate(zip(per_h_long, per_h_short)):
            # Option 1: p(exit <= min_hold + Δ) 계산
            exit_t_long = m_long_h.get("exit_t")
            exit_t_short = m_short_h.get("exit_t")
            
            if exit_t_long is not None:
                exit_t_arr_long = np.asarray(exit_t_long, dtype=np.float64)
                if exit_t_arr_long.size > 0:
                    # p_early_exit = p(exit <= min_hold + Δ)
                    p_early_exit_long = float(np.mean(exit_t_arr_long <= (min_hold_sec_val + exit_early_delta_sec)))
                    penalty_long[i] = max(0.0, 1.0 - exit_early_penalty_k * p_early_exit_long)
            
            if exit_t_short is not None:
                exit_t_arr_short = np.asarray(exit_t_short, dtype=np.float64)
                if exit_t_arr_short.size > 0:
                    p_early_exit_short = float(np.mean(exit_t_arr_short <= (min_hold_sec_val + exit_early_delta_sec)))
                    penalty_short[i] = max(0.0, 1.0 - exit_early_penalty_k * p_early_exit_short)
            
            # Option 2: exit_reason_counts에서 hold_bad/flip 비율 계산 (보조 지표)
            exit_reason_counts_long = m_long_h.get("exit_reason_counts", {}) or {}
            exit_reason_counts_short = m_short_h.get("exit_reason_counts", {}) or {}
            
            total_exits_long = sum(int(v) for v in exit_reason_counts_long.values()) if exit_reason_counts_long else 1
            total_exits_short = sum(int(v) for v in exit_reason_counts_short.values()) if exit_reason_counts_short else 1
            
            if total_exits_long > 0:
                p_early_bad_long = float((exit_reason_counts_long.get("hold_bad", 0) + exit_reason_counts_long.get("score_flip", 0))) / float(total_exits_long)
                # Option 1과 Option 2 중 더 보수적인 값 사용 (더 낮은 penalty = 더 큰 페널티)
                penalty_long[i] = min(penalty_long[i], max(0.0, 1.0 - exit_early_penalty_k * p_early_bad_long))
            
            if total_exits_short > 0:
                p_early_bad_short = float((exit_reason_counts_short.get("hold_bad", 0) + exit_reason_counts_short.get("score_flip", 0))) / float(total_exits_short)
                penalty_short[i] = min(penalty_short[i], max(0.0, 1.0 - exit_early_penalty_k * p_early_bad_short))
        
        penalty_long = np.asarray(penalty_long, dtype=np.float64)
        penalty_short = np.asarray(penalty_short, dtype=np.float64)
        
        # ✅ [EV_VALIDATION_2] 페널티를 w(h)에 적용: w(h) = w_prior(h) * contrib(EV_h) * penalty
        w_long = w_prior_arr * contrib_long * penalty_long
        w_short = w_prior_arr * contrib_short * penalty_short
        w_long = w_long / (w_long.sum() + 1e-12)
        w_short = w_short / (w_short.sum() + 1e-12)
        
        # ✅ [EV_VALIDATION_2] 페널티 로그
        if MC_VERBOSE_PRINT:
            logger.info(
                f"[EV_VALIDATION_2] {symbol} | exit_early_penalty: penalty_long={penalty_long.tolist()} penalty_short={penalty_short.tolist()} (k={exit_early_penalty_k}, delta={exit_early_delta_sec}s)"
            )
            print(
                f"[EV_VALIDATION_2] {symbol} | exit_early_penalty: penalty_long={penalty_long.tolist()} penalty_short={penalty_short.tolist()} (k={exit_early_penalty_k}, delta={exit_early_delta_sec}s)"
            )
        
        # [D] Mix EV using learned weights
        # ✅ [EV_DEBUG] 가중치 및 EV 배열 확인
        if MC_VERBOSE_PRINT:
            logger.info(f"[EV_DEBUG] {symbol} | Before mix: ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
            logger.info(f"[EV_DEBUG] {symbol} | Before mix: w_long={w_long.tolist()} w_short={w_short.tolist()}")
            logger.info(f"[EV_DEBUG] {symbol} | Before mix: evs_long={evs_long} evs_short={evs_short}")
            print(f"[EV_DEBUG] {symbol} | Before mix: ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
            print(f"[EV_DEBUG] {symbol} | Before mix: w_long={w_long.tolist()} w_short={w_short.tolist()}")
            print(f"[EV_DEBUG] {symbol} | Before mix: evs_long={evs_long} evs_short={evs_short}")
        
        policy_ev_mix_long = float((w_long * ev_long_h).sum())
        policy_ev_mix_short = float((w_short * ev_short_h).sum())
        policy_p_pos_mix_long = float((w_long * ppos_long_h).sum())
        policy_p_pos_mix_short = float((w_short * ppos_short_h).sum())
        
        # ✅ [EV_DEBUG] policy_ev_mix 계산 결과 로그
        if MC_VERBOSE_PRINT:
            print(f"[EV_DEBUG] {symbol} | policy_ev_mix calculation: ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
            print(f"[EV_DEBUG] {symbol} | policy_ev_mix calculation: w_long={w_long.tolist()} w_short={w_short.tolist()}")
            print(
                f"[EV_DEBUG] {symbol} | policy_ev_mix calculation: policy_ev_mix_long={policy_ev_mix_long:.8f} policy_ev_mix_short={policy_ev_mix_short:.8f}"
            )

        # Always-on short summary for troubleshooting (helps when MC_VERBOSE_PRINT is False)
        try:
            el_len = len(evs_long) if ("evs_long" in locals() and hasattr(evs_long, '__len__')) else 0
            es_len = len(evs_short) if ("evs_short" in locals() and hasattr(evs_short, '__len__')) else 0
        except Exception:
            el_len = 0
            es_len = 0
        print(f"[EV_SUMMARY] {symbol} n_paths={n_paths} ev_mix_long={policy_ev_mix_long:.8f} ev_mix_short={policy_ev_mix_short:.8f} evs_long_len={el_len} evs_short_len={es_len}")
        logger.info(f"[EV_DEBUG] {symbol} | policy_ev_mix calculation: policy_ev_mix_long={policy_ev_mix_long:.8f} policy_ev_mix_short={policy_ev_mix_short:.8f}")
        
        # ✅ 모든 심볼의 policy_ev_mix가 음수인 경우 검증 (LINK 제외)
        if not symbol.startswith("LINK") and policy_ev_mix_long < 0 and policy_ev_mix_short < 0:
            allow_log = MC_VERBOSE_PRINT or _throttled_log(symbol, "POLICY_EV_MIX_NEG", 60_000)
            if allow_log:
                if MC_VERBOSE_PRINT:
                    logger.warning(
                        f"[EV_VALIDATION_NEG] {symbol} | ⚠️  policy_ev_mix가 모두 음수: long={policy_ev_mix_long:.6f} short={policy_ev_mix_short:.6f}"
                    )
                    logger.warning(f"[EV_VALIDATION_NEG] {symbol} | ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
                    logger.warning(f"[EV_VALIDATION_NEG] {symbol} | w_long={w_long.tolist()} w_short={w_short.tolist()}")
                    print(
                        f"[EV_VALIDATION_NEG] {symbol} | ⚠️  policy_ev_mix가 모두 음수: long={policy_ev_mix_long:.6f} short={policy_ev_mix_short:.6f}"
                    )
                    print(f"[EV_VALIDATION_NEG] {symbol} | ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
                    print(f"[EV_VALIDATION_NEG] {symbol} | w_long={w_long.tolist()} w_short={w_short.tolist()}")
                else:
                    logger.warning(
                        f"[EV_VALIDATION_NEG] {symbol} | policy_ev_mix both negative: long={policy_ev_mix_long:.6f} short={policy_ev_mix_short:.6f}"
                    )
        
        # ✅ 모든 심볼의 policy_ev_mix 통계 (LINK 제외)
        if not symbol.startswith("LINK"):
            # horizon별 EV 값이 모두 음수인지 확인
            all_negative_long = all(ev < 0 for ev in ev_long_h) if len(ev_long_h) > 0 else False
            all_negative_short = all(ev < 0 for ev in ev_short_h) if len(ev_short_h) > 0 else False
            
            if all_negative_long and all_negative_short:
                allow_log = MC_VERBOSE_PRINT or _throttled_log(symbol, "ALL_HORIZON_EV_NEG", 60_000)
                if allow_log:
                    if MC_VERBOSE_PRINT:
                        logger.warning(
                            f"[EV_VALIDATION_NEG] {symbol} | ⚠️  모든 horizon에서 EV가 음수: ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}"
                        )
                        print(
                            f"[EV_VALIDATION_NEG] {symbol} | ⚠️  모든 horizon에서 EV가 음수: ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}"
                        )
                    else:
                        logger.warning(f"[EV_VALIDATION_NEG] {symbol} | all horizons EV negative")
        
        # ✅ [EV_DEBUG] policy_ev_mix 계산 결과 로그
        if MC_VERBOSE_PRINT:
            logger.info(f"[EV_DEBUG] {symbol} | policy_ev_mix_long={policy_ev_mix_long:.6f} policy_ev_mix_short={policy_ev_mix_short:.6f}")
            print(f"[EV_DEBUG] {symbol} | policy_ev_mix_long={policy_ev_mix_long:.6f} policy_ev_mix_short={policy_ev_mix_short:.6f}")
        
        # ✅ [EV_DEBUG] policy_ev_mix가 0인 경우 원인 파악
        if abs(policy_ev_mix_long) < 1e-6 and abs(policy_ev_mix_short) < 1e-6:
            allow_log = MC_VERBOSE_PRINT or _throttled_log(symbol, "POLICY_EV_MIX_NEAR_ZERO", 60_000)
            if allow_log:
                if MC_VERBOSE_PRINT:
                    logger.warning(f"[EV_DEBUG] {symbol} | ⚠️  policy_ev_mix_long and policy_ev_mix_short are both near 0!")
                    logger.warning(f"[EV_DEBUG] {symbol} | ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
                    logger.warning(f"[EV_DEBUG] {symbol} | w_long={w_long.tolist()} w_short={w_short.tolist()}")
                    print(f"[EV_DEBUG] {symbol} | ⚠️  policy_ev_mix_long and policy_ev_mix_short are both near 0!")
                    print(f"[EV_DEBUG] {symbol} | ev_long_h={ev_long_h.tolist()} ev_short_h={ev_short_h.tolist()}")
                    print(f"[EV_DEBUG] {symbol} | w_long={w_long.tolist()} w_short={w_short.tolist()}")
                else:
                    logger.warning(f"[EV_DEBUG] {symbol} | policy_ev_mix near 0")
        
        # [A] Objective-based direction choice (0~5m).
        # 기본 목표(권장): J = (EV_net / |CVaR_95|) * (1 / sqrt(T)) - λ * Var_EV
        # Hard constraints:
        #   - p_liq < 0.01%  (practically 0 with finite MC paths)
        #   - Profit/Cost > 1.2
        #   - Max DD < 1.0% (worst drawdown over the horizon)
        h_arr_pol = np.asarray(policy_horizons, dtype=np.int64)
        n_pol = int(h_arr_pol.size)

        # objective / constraint knobs
        # Available modes:
        # - "ratio_time_var": default mode (EV/CVaR * time_weight - lambda*var)
        # - "new_objective": new objective with signal reliability penalty (EV / (CVaR + 2*Std_Dev) * time_weight)
        # - "signal_reliability": alias for "new_objective"
        obj_mode = config.policy_objective_mode
        lambda_var = config.policy_lambda_var
        cvar_eps = config.policy_cvar_eps
        max_p_liq = config.policy_max_p_liq
        min_profit_cost = config.policy_min_profit_cost
        max_dd_abs = config.policy_max_dd_abs

        min_gap = config.policy_min_ev_gap
        # In chop/volatile, require a larger directional edge to avoid noisy long/short flips.
        try:
            reg = str(regime_ctx or "").strip().lower()
        except Exception:
            reg = "chop"
        min_gap_eff = float(min_gap)
        if reg == "chop":
            min_gap_eff = float(max(min_gap_eff, 0.0006))
        elif reg == "volatile":
            min_gap_eff = float(max(min_gap_eff, 0.0008))

        def _metric_arr(per_h: list[dict], key: str) -> np.ndarray:
            out = []
            for i in range(n_pol):
                v = None
                if i < len(per_h):
                    try:
                        v = per_h[i].get(key)
                    except Exception:
                        v = None
                try:
                    out.append(float(v))
                except Exception:
                    out.append(float("nan"))
            return np.asarray(out, dtype=np.float64)

        def _ensure_len(arr: np.ndarray, target_len: int, fill: float = 0.0) -> np.ndarray:
            """Patched: Static warmup shape and empty array defense."""
            a = np.asarray(arr, dtype=np.float64)
            if a.size == 0:
                return np.full(target_len, fill, dtype=np.float64)
            if a.size < target_len:
                out = np.full(target_len, fill, dtype=np.float64)
                out[: a.size] = a
                return out
            if a.size > target_len:
                return a[:target_len]
            return a

        # risk stats from exit-policy rollforward (per horizon)
        var_long_h = _metric_arr(per_h_long, "var_exit_policy")
        var_short_h = _metric_arr(per_h_short, "var_exit_policy")
        p_liq_long_h = _metric_arr(per_h_long, "p_liq_exit_policy")
        p_liq_short_h = _metric_arr(per_h_short, "p_liq_exit_policy")
        dd_min_long_h = _metric_arr(per_h_long, "dd_min_exit_policy")
        dd_min_short_h = _metric_arr(per_h_short, "dd_min_exit_policy")

        # CVaR arrays are already net-of-entry-cost (see append above)
        cvar_long_h = _ensure_len(cvars_long, n_pol, 0.0)
        cvar_short_h = _ensure_len(cvars_short, n_pol, 0.0)

        # Profit/Cost ratio uses total execution cost (entry+exit) in ROE units.
        cost_total_roe = float(cost_entry_roe) + float(cost_exit_roe)
        denom_cost = float(max(1e-12, cost_total_roe))
        profit_cost_long_h = (ev_long_h + float(cost_total_roe)) / denom_cost
        profit_cost_short_h = (ev_short_h + float(cost_total_roe)) / denom_cost

        # Model components
        abs_cvar_long = np.abs(cvar_long_h) + float(cvar_eps)
        abs_cvar_short = np.abs(cvar_short_h) + float(cvar_eps)
        
        # Calculate standard deviation from variance
        std_long_h = np.sqrt(np.maximum(var_long_h, 0.0))
        std_short_h = np.sqrt(np.maximum(var_short_h, 0.0))
        
        # New objective function: J = (EV_net) / (CVaR + (2.0 * Std_Dev)) * (1 / sqrt(T))
        # This replaces the previous ratio-based approach
        denominator_long = abs_cvar_long + (2.0 * std_long_h)
        denominator_short = abs_cvar_short + (2.0 * std_short_h)
        
        # Avoid division by zero
        denominator_long = np.maximum(denominator_long, 1e-12)
        denominator_short = np.maximum(denominator_short, 1e-12)
        
        # Time efficiency weight: 1 / sqrt(T)
        time_w = 1.0 / np.sqrt(np.maximum(h_arr_pol.astype(np.float64), 1.0))
        
        # ✅ Gross Score 옵션: 비용 제외한 순수 방향성 평가
        use_gross_score = config.use_gross_score
        
        if use_gross_score:
            # Gross EV (비용 제외)
            ev_long_gross = ev_long_h + float(cost_total_roe)
            ev_short_gross = ev_short_h + float(cost_total_roe)
            
            # Gross Score
            j_new_long = (ev_long_gross / denominator_long) * time_w
            j_new_short = (ev_short_gross / denominator_short) * time_w
        else:
            # Net EV (비용 포함, 기본값)
            j_new_long = (ev_long_h / denominator_long) * time_w
            j_new_short = (ev_short_h / denominator_short) * time_w
        
        # Keep existing components for backward compatibility
        j_ratio_long = ev_long_h / abs_cvar_long
        j_ratio_short = ev_short_h / abs_cvar_short
        j_ratio_time_long = j_ratio_long * time_w
        j_ratio_time_short = j_ratio_short * time_w
        j_ev_var_long = ev_long_h - float(lambda_var) * var_long_h
        j_ev_var_short = ev_short_h - float(lambda_var) * var_short_h

        if obj_mode in ("ratio", "cvar_ratio"):
            obj_long_raw = j_ratio_long
            obj_short_raw = j_ratio_short
        elif obj_mode in ("ratio_time", "time_ratio"):
            obj_long_raw = j_ratio_time_long
            obj_short_raw = j_ratio_time_short
        elif obj_mode in ("ev_var", "var_penalty"):
            obj_long_raw = j_ev_var_long
            obj_short_raw = j_ev_var_short
        elif obj_mode in ("new_objective", "signal_reliability"):
            # Use the new objective function with signal reliability penalty
            obj_long_raw = j_new_long
            obj_short_raw = j_new_short
        else:
            # default: combine (1)+(3) with (2) penalty
            obj_long_raw = j_ratio_time_long - float(lambda_var) * var_long_h
            obj_short_raw = j_ratio_time_short - float(lambda_var) * var_short_h

        # TP-hit reliability weighting (per-horizon). Penalize low TP hit on positive scores.
        tp_floor = float(max(0.0, min(1.0, config.score_tp_floor)))
        tp_power = float(max(0.0, config.score_tp_power))
        tp_weight = float(max(0.0, min(1.0, config.score_tp_weight)))
        try:
            tp_long_h = np.asarray(mc_p_tp_long, dtype=np.float64)
            tp_short_h = np.asarray(mc_p_tp_short, dtype=np.float64)
        except Exception:
            tp_long_h = np.asarray([], dtype=np.float64)
            tp_short_h = np.asarray([], dtype=np.float64)
        if tp_long_h.size != obj_long_raw.size:
            tp_long_h = np.ones_like(obj_long_raw, dtype=np.float64)
        if tp_short_h.size != obj_short_raw.size:
            tp_short_h = np.ones_like(obj_short_raw, dtype=np.float64)

        def _tp_weight(tp_arr: np.ndarray) -> np.ndarray:
            tp_arr = np.clip(tp_arr, 0.0, 1.0)
            if tp_floor <= 0.0:
                w = tp_arr
            else:
                w = (tp_arr - tp_floor) / max(1e-6, 1.0 - tp_floor)
                w = np.clip(w, 0.0, 1.0)
            if tp_power != 1.0:
                w = np.power(w, tp_power)
            if tp_weight < 1.0:
                w = (1.0 - tp_weight) + (tp_weight * w)
            return w

        tp_weight_long_h = _tp_weight(tp_long_h)
        tp_weight_short_h = _tp_weight(tp_short_h)
        obj_long_raw = np.where(obj_long_raw > 0.0, obj_long_raw * tp_weight_long_h, obj_long_raw)
        obj_short_raw = np.where(obj_short_raw > 0.0, obj_short_raw * tp_weight_short_h, obj_short_raw)

        # ✅ SCORE_ONLY: EV > 0 제약조건 제거 (음수 EV도 Score 계산 가능)
        use_score_only = config.score_only_mode
        
        if use_score_only:
            # Score 모드: 극단적인 케이스만 차단 (무한값, NaN 등)
            valid_long = (
                np.isfinite(obj_long_raw)
                & np.isfinite(ev_long_h)
                & np.isfinite(cvar_long_h)
            )
            valid_short = (
                np.isfinite(obj_short_raw)
                & np.isfinite(ev_short_h)
                & np.isfinite(cvar_short_h)
            )
        else:
            # 레거시 모드: 모든 제약조건 적용
            p_liq_ok_L = p_liq_long_h < float(max_p_liq)
            profit_cost_ok_L = profit_cost_long_h > float(min_profit_cost)
            dd_ok_L = ((-dd_min_long_h) <= float(max_dd_abs))
            
            valid_long = (
                np.isfinite(obj_long_raw)
                & np.isfinite(p_liq_long_h)
                & np.isfinite(dd_min_long_h)
                & np.isfinite(profit_cost_long_h)
                & p_liq_ok_L
                & profit_cost_ok_L
                & dd_ok_L
            )
            if MC_VERBOSE_PRINT and not np.all(valid_long):
                 # Log why it's invalid for the best raw index if possible
                 best_raw_idx = np.argmax(obj_long_raw)
                 if not valid_long[best_raw_idx]:
                     logger.info(f"[LEGACY_FILTER] {symbol} LONG h={h_arr_pol[best_raw_idx]} | obj={obj_long_raw[best_raw_idx]:.6f} p_liq={p_liq_long_h[best_raw_idx]:.4f}(ok={p_liq_ok_L[best_raw_idx]}) profit_cost={profit_cost_long_h[best_raw_idx]:.2f}(ok={profit_cost_ok_L[best_raw_idx]}) dd={-dd_min_long_h[best_raw_idx]:.4f}(ok={dd_ok_L[best_raw_idx]})")

            p_liq_ok_S = p_liq_short_h < float(max_p_liq)
            profit_cost_ok_S = profit_cost_short_h > float(min_profit_cost)
            dd_ok_S = ((-dd_min_short_h) <= float(max_dd_abs))

            valid_short = (
                np.isfinite(obj_short_raw)
                & np.isfinite(p_liq_short_h)
                & np.isfinite(dd_min_short_h)
                & np.isfinite(profit_cost_short_h)
                & p_liq_ok_S
                & profit_cost_ok_S
                & dd_ok_S
            )
            if MC_VERBOSE_PRINT and not np.all(valid_short):
                best_raw_idx_s = np.argmax(obj_short_raw)
                if not valid_short[best_raw_idx_s]:
                    logger.info(f"[LEGACY_FILTER] {symbol} SHORT h={h_arr_pol[best_raw_idx_s]} | obj={obj_short_raw[best_raw_idx_s]:.6f} p_liq={p_liq_short_h[best_raw_idx_s]:.4f}(ok={p_liq_ok_S[best_raw_idx_s]}) profit_cost={profit_cost_short_h[best_raw_idx_s]:.2f}(ok={profit_cost_ok_S[best_raw_idx_s]}) dd={-dd_min_short_h[best_raw_idx_s]:.4f}(ok={dd_ok_S[best_raw_idx_s]})")


        
        obj_long_h = np.where(valid_long, obj_long_raw, -np.inf)
        obj_short_h = np.where(valid_short, obj_short_raw, -np.inf)

        def _safe_argmax(x: np.ndarray) -> int:
            try:
                if x.size <= 0 or (not np.isfinite(x).any()):
                    return 0
                return int(np.argmax(x))
            except Exception:
                return 0

        # Select best horizon per side by objective (after constraints)
        best_idx_long = _safe_argmax(obj_long_h)
        best_idx_short = _safe_argmax(obj_short_h)
        best_h_long = int(h_arr_pol[best_idx_long]) if n_pol > 0 else 0
        best_h_short = int(h_arr_pol[best_idx_short]) if n_pol > 0 else 0

        best_obj_long = float(obj_long_h[best_idx_long]) if obj_long_h.size else float("-inf")
        best_obj_short = float(obj_short_h[best_idx_short]) if obj_short_h.size else float("-inf")

        # Keep EV/PPos/CVaR for the chosen horizons (for display/gating)
        best_ev_long = float(ev_long_h[best_idx_long]) if ev_long_h.size else 0.0
        best_ev_short = float(ev_short_h[best_idx_short]) if ev_short_h.size else 0.0
        best_p_pos_long = float(ppos_long_h[best_idx_long]) if ppos_long_h.size else 0.0
        best_p_pos_short = float(ppos_short_h[best_idx_short]) if ppos_short_h.size else 0.0
        best_cvar_long = float(cvar_long_h[best_idx_long]) if cvar_long_h.size else 0.0
        best_cvar_short = float(cvar_short_h[best_idx_short]) if cvar_short_h.size else 0.0

        # Neighbor reinforcement (local multi-horizon): reward consistency on adjacent horizons.
        # 목표: 단일 best-horizon만 보고 방향/EV를 결정하는 불안정성을 줄이되,
        #       청산 타겟(policy_best_h)은 그대로 유지하고, 진입 점수(EV)를 "이웃 호라이즌 지지"만큼 보너스.
        neighbor_w = config.policy_neighbor_bonus_w
        neighbor_cap = config.policy_neighbor_bonus_cap
        neighbor_pen_w = config.policy_neighbor_penalty_w
        neighbor_pen_cap = config.policy_neighbor_penalty_cap

        # Default: enable veto with a short-term meaningful threshold (profit_target).
        veto_env = str(getattr(config, "policy_neighbor_oppose_veto_abs", "") or "")
        if veto_env is None or (not str(veto_env).strip()):
            neighbor_veto_abs = float(getattr(params, "profit_target", 0.0) or 0.0)
        else:
            try:
                neighbor_veto_abs = float(veto_env)
            except Exception:
                neighbor_veto_abs = float(getattr(params, "profit_target", 0.0) or 0.0)
        neighbor_veto_abs = float(max(0.0, neighbor_veto_abs))

        def _neighbor_bonus(best_idx: int, side: int) -> float:
            try:
                n = int(h_arr_pol.size)
            except Exception:
                n = 0
            if n <= 0:
                return 0.0
            bonus = 0.0
            # Note: Legcay neighbor_bonus is being phased out in favor of unified consensus,
            # but we keep the structure for now and simply dampen its additive effect to avoid double counting.
            # Unified consensus already handles neighbor support.
            neighbor_damp = 0.5 
            for j in (int(best_idx) - 1, int(best_idx) + 1):
                if j < 0 or j >= n:
                    continue
                oL_j = float(obj_long_h[j])
                oS_j = float(obj_short_h[j])
                if int(side) == 1:
                    if (oL_j > 0.0) and (oL_j >= oS_j):
                        bonus += float(neighbor_w) * oL_j * neighbor_damp
                else:
                    if (oS_j > 0.0) and (oS_j > oL_j):
                        bonus += float(neighbor_w) * oS_j * neighbor_damp
            if neighbor_cap > 0.0:
                bonus = float(min(neighbor_cap, bonus))
            return float(bonus)

        def _neighbor_penalty(best_idx: int, side: int) -> tuple[float, float]:
            """
            Penalize when adjacent horizons are positive AND prefer the opposite side.
            Returns (penalty, opp_max_obj).
            """
            try:
                n = int(h_arr_pol.size)
            except Exception:
                n = 0
            if n <= 0:
                return 0.0, 0.0
            penalty = 0.0
            opp_max = 0.0
            for j in (int(best_idx) - 1, int(best_idx) + 1):
                if j < 0 or j >= n:
                    continue
                oL_j = float(obj_long_h[j])
                oS_j = float(obj_short_h[j])
                if int(side) == 1:
                    # Opposite of LONG is SHORT; mirror short-bonus condition
                    if (oS_j > 0.0) and (oS_j > oL_j):
                        penalty += float(neighbor_pen_w) * oS_j
                        opp_max = float(max(opp_max, oS_j))
                else:
                    # Opposite of SHORT is LONG; mirror long-bonus condition
                    if (oL_j > 0.0) and (oL_j >= oS_j):
                        penalty += float(neighbor_pen_w) * oL_j
                        opp_max = float(max(opp_max, oL_j))
            if neighbor_pen_cap > 0.0:
                penalty = float(min(neighbor_pen_cap, penalty))
            return float(penalty), float(opp_max)


        # ========================================
        # Local Weighted Consensus
        # ========================================
        # Each horizon is a candidate. Neighbors contribute weight based on:
        # 1. Distance (closer = more weight)
        # 2. Direction agreement (same side, positive)
        local_consensus_alpha = config.policy_local_consensus_alpha
        base_horizon = config.policy_local_consensus_base_h
        
        n_horizons = int(h_arr_pol.size) if h_arr_pol.size > 0 else 0
        
        if n_horizons <= 0:
            # Fallback: no horizons available
            weighted_score_long_h = np.array([float(best_obj_long)])
            weighted_score_short_h = np.array([float(best_obj_short)])
            best_idx_long_weighted = 0
            best_idx_short_weighted = 0
        else:
            # Initialize weighted scores with base objective values
            weighted_score_long_h = np.copy(obj_long_h)
            weighted_score_short_h = np.copy(obj_short_h)
            
            # For each horizon i, add distance-weighted contributions from neighbors
            for i in range(n_horizons):
                h_i = float(h_arr_pol[i])
                
                # --- [Refinement] Differentiated Thresholds ---
                # Formula: Th_h = Th_base * sqrt(h / 180s)
                # Short horizons get lower thresholds (more sensitive).
                th_base = config.score_entry_threshold
                th_h = th_base * math.sqrt(h_i / 180.0)
                th_h = max(0.0005, th_h) # Absolute floor
                
                # --- [Refinement] Lead Signal Logic (Lag Reduction) ---
                # If short-term horizon (e.g. <= 60s) is extremely strong, pulse its contribution.
                is_lead = (h_i <= 60.0)
                lead_boost = 1.0
                if is_lead:
                    obj_i = float(obj_long_h[i]) if float(obj_long_h[i]) >= float(obj_short_h[i]) else float(obj_short_h[i])
                    if obj_i > th_h * 1.5:
                        lead_boost = 1.5 # 50% more weight to lead signal neighbors
                
                for j in range(n_horizons):
                    if i == j:
                        continue
                    
                    h_j = float(h_arr_pol[j])
                    oL_j = float(obj_long_h[j])
                    oS_j = float(obj_short_h[j])
                    
                    # Distance weight: closer horizons have more influence
                    distance = abs(h_i - h_j)
                    w_dist = 1.0 / (1.0 + distance / base_horizon)
                    
                    # Add contribution if neighbor j supports long (positive and >= short)
                    if oL_j > 0.0 and oL_j >= oS_j:
                        weighted_score_long_h[i] += oL_j * w_dist * local_consensus_alpha * lead_boost
                    
                    # Add contribution if neighbor j supports short (positive and > long)
                    if oS_j > 0.0 and oS_j > oL_j:
                        weighted_score_short_h[i] += oS_j * w_dist * local_consensus_alpha * lead_boost
            
            # Select horizon with highest weighted score
            best_idx_long_weighted = int(np.argmax(weighted_score_long_h))
            best_idx_short_weighted = int(np.argmax(weighted_score_short_h))
        
        # Use weighted scores as final scores
        best_obj_long_weighted = float(weighted_score_long_h[best_idx_long_weighted])
        best_obj_short_weighted = float(weighted_score_short_h[best_idx_short_weighted])
        best_h_long_weighted = int(h_arr_pol[best_idx_long_weighted]) if n_horizons > 0 else 60
        best_h_short_weighted = int(h_arr_pol[best_idx_short_weighted]) if n_horizons > 0 else 60
        
        if MC_VERBOSE_PRINT:
            logger.info(
                f"[LOCAL_CONSENSUS] {symbol} | "
                f"Long: horizon={best_h_long_weighted}s base_obj={float(obj_long_h[best_idx_long_weighted]):.6f} "
                f"weighted={best_obj_long_weighted:.6f}"
            )
            logger.info(
                f"[LOCAL_CONSENSUS] {symbol} | "
                f"Short: horizon={best_h_short_weighted}s base_obj={float(obj_short_h[best_idx_short_weighted]):.6f} "
                f"weighted={best_obj_short_weighted:.6f}"
            )

        neighbor_bonus_long = float(_neighbor_bonus(best_idx_long, 1))
        neighbor_bonus_short = float(_neighbor_bonus(best_idx_short, -1))
        neighbor_penalty_long, neighbor_opp_max_long = _neighbor_penalty(best_idx_long, 1)
        neighbor_penalty_short, neighbor_opp_max_short = _neighbor_penalty(best_idx_short, -1)
        
        # Final score: use weighted objective + neighbor adjustments
        score_long = float(best_obj_long_weighted) + float(neighbor_bonus_long) - float(neighbor_penalty_long)
        score_short = float(best_obj_short_weighted) + float(neighbor_bonus_short) - float(neighbor_penalty_short)

        # Keep names for existing dashboard/tooling keys, but values are now objective gaps.
        best_ev_gap = float(best_obj_long) - float(best_obj_short)
        ev_gap = float(score_long) - float(score_short)
        p_pos_gap = float(best_p_pos_long) - float(best_p_pos_short)
        side_best = 1 if float(score_long) >= float(score_short) else -1

        # ✅ [EV_DEBUG] direction_policy 결정 로그
        if MC_VERBOSE_PRINT:
            logger.info(
                f"[EV_DEBUG] {symbol} | best_obj_long={best_obj_long:.6f}@{best_h_long}s best_obj_short={best_obj_short:.6f}@{best_h_short}s | "
                f"score_long={score_long:.6f} score_short={score_short:.6f} | "
                f"score_gap={ev_gap:.6f} min_gap={min_gap_eff:.6f} mode={obj_mode}"
            )
            print(
                f"[EV_DEBUG] {symbol} | best_obj_long={best_obj_long:.6f}@{best_h_long}s best_obj_short={best_obj_short:.6f}@{best_h_short}s | "
                f"score_long={score_long:.6f} score_short={score_short:.6f} | "
                f"score_gap={ev_gap:.6f} min_gap={min_gap_eff:.6f} mode={obj_mode}"
            )

        # ✅ SCORE-BASED DIRECTION: 롱과 숏을 독립적으로 판단 (차등 임계점 적용)
        policy_direction_reason = None
        
        # Determine effective threshold for the best horizon
        best_h_chosen = (best_h_long_weighted if score_long >= score_short else best_h_short_weighted)
        th_base = float(max(0.0, config.score_entry_threshold))
        score_threshold_eff = th_base * math.sqrt(float(best_h_chosen) / 180.0)
        score_threshold_eff = float(max(config.score_entry_floor, score_threshold_eff))

        # TP-hit gate at a fixed horizon (default 5m) to suppress entries with near-zero TP hit.
        tp_entry_min = float(max(0.0, min(1.0, config.score_tp_entry_min)))
        tp_entry_hard = float(max(0.0, min(1.0, config.score_tp_entry_hard)))
        tp_gate_mult = float(max(0.0, config.score_tp_entry_gate_mult))
        tp_entry_h = int(max(1, int(config.score_tp_entry_horizon_sec)))
        tp5_idx = None
        tp_long_5m = None
        tp_short_5m = None
        try:
            h_vals = [int(x) for x in h_arr_pol.tolist()] if hasattr(h_arr_pol, "tolist") else [int(x) for x in h_arr_pol]
            if h_vals:
                tp5_idx = min(range(len(h_vals)), key=lambda i: abs(int(h_vals[i]) - tp_entry_h))
        except Exception:
            tp5_idx = None
        try:
            if tp5_idx is not None and tp5_idx >= 0:
                tp_long_5m = float(tp_long_h[tp5_idx])
                tp_short_5m = float(tp_short_h[tp5_idx])
        except Exception:
            tp_long_5m = None
            tp_short_5m = None

        def _tp_gate(tp_val: Optional[float]) -> tuple[float, bool]:
            if tp_val is None:
                return 1.0, True
            tp_v = float(max(0.0, min(1.0, tp_val)))
            if tp_entry_min <= 0.0:
                return 1.0, True
            if tp_v <= tp_entry_hard:
                return 1.0, False
            if tp_v < tp_entry_min:
                mult = 1.0 + ((tp_entry_min - tp_v) / max(1e-6, tp_entry_min)) * tp_gate_mult
                return float(max(1.0, mult)), True
            return 1.0, True

        tp_mult_long, tp_ok_long = _tp_gate(tp_long_5m)
        tp_mult_short, tp_ok_short = _tp_gate(tp_short_5m)

        # ✅ [DEBUG] 진입 결정 분기 직전 상태 로깅
        if MC_VERBOSE_PRINT or _throttled_log(symbol, "ENTRY_DECISION_DEBUG", 30_000):
            logger.info(
                f"[ENTRY_DECISION] {symbol} | "
                f"scoreL={score_long:.6f} scoreS={score_short:.6f} threshold={score_threshold_eff:.6f} | "
                f"tp_okL={tp_ok_long} tp_okS={tp_ok_short} tp_multL={tp_mult_long:.2f} tp_multS={tp_mult_short:.2f} | "
                f"tp_5m_L={tp_long_5m} tp_5m_S={tp_short_5m} | "
                f"score_tp_entry_min={tp_entry_min:.4f} score_tp_entry_hard={tp_entry_hard:.4f}"
            )

        score_long_valid = (
            tp_ok_long
            and math.isfinite(float(score_long))
            and float(score_long) > (score_threshold_eff * tp_mult_long)
        )
        score_short_valid = (
            tp_ok_short
            and math.isfinite(float(score_short))
            and float(score_short) > (score_threshold_eff * tp_mult_short)
        )
        
        # ✅ [DEBUG] 진입 결정 결과 로깅
        if MC_VERBOSE_PRINT or _throttled_log(symbol, "ENTRY_VALID_DEBUG", 30_000):
            logger.info(
                f"[ENTRY_VALID] {symbol} | "
                f"score_long_valid={score_long_valid} score_short_valid={score_short_valid} | "
                f"scoreL({score_long:.6f}) > threshold({score_threshold_eff * tp_mult_long:.6f})? {score_long > (score_threshold_eff * tp_mult_long)} | "
                f"scoreS({score_short:.6f}) > threshold({score_threshold_eff * tp_mult_short:.6f})? {score_short > (score_threshold_eff * tp_mult_short)}"
            )
        try:
            both_ev_neg_on = str(os.environ.get("ENTRY_BOTH_EV_NEG_FILTER_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            both_ev_neg_on = True
        try:
            both_ev_neg_floor = float(os.environ.get("ENTRY_BOTH_EV_NEG_NET_FLOOR", 0.0) or 0.0)
        except Exception:
            both_ev_neg_floor = 0.0
        both_ev_negative = bool(
            float(policy_ev_mix_long) <= float(both_ev_neg_floor)
            and float(policy_ev_mix_short) <= float(both_ev_neg_floor)
        )

        if both_ev_neg_on and both_ev_negative:
            direction_policy = 0
            policy_direction_reason = (
                f"both_ev_negative_wait (evL={float(policy_ev_mix_long):.6f}, "
                f"evS={float(policy_ev_mix_short):.6f}, floor={float(both_ev_neg_floor):.6f})"
            )
        elif not score_long_valid and not score_short_valid:
            # 둘 다 임계값 미달 → WAIT
            direction_policy = 0
            policy_direction_reason = f"both_scores_invalid (scoreL={score_long:.6f}, scoreS={score_short:.6f}, threshold={score_threshold_eff:.6f}, h={best_h_chosen})"
        elif score_long_valid and score_short_valid:
            # 둘 다 양수 → gap이 충분히 크면 큰 쪽 선택, 아니면 WAIT
            if abs(ev_gap) >= min_gap_eff:
                direction_policy = 1 if ev_gap > 0 else -1
                policy_direction_reason = (
                    f"both_positive_gap_ok (scoreL={score_long:.6f}, scoreS={score_short:.6f}, "
                    f"gap={ev_gap:.6f}, min_gap_eff={min_gap_eff:.6f})"
                )
            else:
                side_candidate = 1 if ev_gap > 0 else -1
                side_conf = float(best_p_pos_long) if side_candidate == 1 else float(best_p_pos_short)
                reg_key = "CHOP"
                if reg == "trend":
                    reg_key = "TREND"
                elif reg == "volatile":
                    reg_key = "VOLATILE"

                def _sg_env_float(key: str, fallback: float) -> float:
                    try:
                        v = os.environ.get(f"{key}_{reg_key}")
                        if v is None or str(v).strip() == "":
                            v = os.environ.get(key)
                        if v is None or str(v).strip() == "":
                            return float(fallback)
                        return float(v)
                    except Exception:
                        return float(fallback)

                def _sg_env_bool(key: str, fallback: bool) -> bool:
                    try:
                        v = os.environ.get(f"{key}_{reg_key}")
                        if v is None or str(v).strip() == "":
                            v = os.environ.get(key)
                        if v is None:
                            return bool(fallback)
                        txt = str(v).strip().lower()
                        if txt in ("1", "true", "yes", "on", "y"):
                            return True
                        if txt in ("0", "false", "no", "off", "n"):
                            return False
                    except Exception:
                        pass
                    return bool(fallback)

                allow_high_conf = _sg_env_bool(
                    "POLICY_SMALL_GAP_ALLOW_HIGH_CONF",
                    bool(getattr(config, "policy_small_gap_allow_high_conf", True)),
                )
                conf_th = _sg_env_float(
                    "POLICY_SMALL_GAP_CONFIDENCE",
                    float(getattr(config, "policy_small_gap_confidence", 0.60)),
                )
                dir_conf = float(mu_alpha_parts.get("mu_dir_conf") or 0.0) if isinstance(mu_alpha_parts, dict) else 0.0
                dir_edge = abs(float(mu_alpha_parts.get("mu_dir_edge") or 0.0)) if isinstance(mu_alpha_parts, dict) else 0.0
                dir_prob_long = float(mu_alpha_parts.get("mu_dir_prob_long") or 0.5) if isinstance(mu_alpha_parts, dict) else 0.5
                side_prob = float(dir_prob_long) if side_candidate == 1 else float(1.0 - dir_prob_long)
                dir_conf_th = _sg_env_float(
                    "POLICY_SMALL_GAP_DIR_CONFIDENCE",
                    float(getattr(config, "policy_small_gap_dir_confidence", 0.58)),
                )
                dir_edge_th = _sg_env_float(
                    "POLICY_SMALL_GAP_DIR_EDGE",
                    float(getattr(config, "policy_small_gap_dir_edge", 0.08)),
                )
                side_prob_th = _sg_env_float(
                    "POLICY_SMALL_GAP_SIDE_PROB",
                    float(getattr(config, "policy_small_gap_side_prob", 0.56)),
                )
                if (
                    allow_high_conf
                    and side_conf >= conf_th
                    and dir_conf >= dir_conf_th
                    and dir_edge >= dir_edge_th
                    and side_prob >= side_prob_th
                ):
                    direction_policy = side_candidate
                    policy_direction_reason = (
                        f"both_positive_small_gap_high_conf (scoreL={score_long:.6f}, scoreS={score_short:.6f}, "
                        f"gap={ev_gap:.6f}, min_gap_eff={min_gap_eff:.6f}, p_pos={side_conf:.4f}, conf_th={conf_th:.4f}, "
                        f"dir_conf={dir_conf:.4f}, dir_edge={dir_edge:.4f}, side_prob={side_prob:.4f}/{side_prob_th:.4f})"
                    )
                else:
                    direction_policy = 0
                    policy_direction_reason = (
                        f"both_positive_small_gap_wait (scoreL={score_long:.6f}, scoreS={score_short:.6f}, "
                        f"gap={ev_gap:.6f}, min_gap_eff={min_gap_eff:.6f}, p_pos={side_conf:.4f}, conf_th={conf_th:.4f}, "
                        f"dir_conf={dir_conf:.4f}/{dir_conf_th:.4f}, dir_edge={dir_edge:.4f}/{dir_edge_th:.4f}, "
                        f"side_prob={side_prob:.4f}/{side_prob_th:.4f})"
                    )
        elif score_long_valid:
            # 롱만 임계값 통과 → LONG
            direction_policy = 1
            policy_direction_reason = f"long_only_positive (scoreL={score_long:.6f}, scoreS={score_short:.6f}, threshold={score_threshold_eff:.6f})"
        else:
            # 숏만 임계값 통과 → SHORT
            direction_policy = -1
            policy_direction_reason = f"short_only_positive (scoreL={score_long:.6f}, scoreS={score_short:.6f}, threshold={score_threshold_eff:.6f})"

        # Optional veto: if an adjacent horizon strongly prefers the opposite side, don't trade (uncertainty).
        if neighbor_veto_abs > 0.0:
            # Veto is evaluated in EV space (stable threshold in ROE units).
            def _neighbor_opp_max_ev(best_idx: int, side: int) -> float:
                try:
                    n = int(h_arr_pol.size)
                except Exception:
                    n = 0
                if n <= 0:
                    return 0.0
                opp_max_ev = 0.0
                for j in (int(best_idx) - 1, int(best_idx) + 1):
                    if j < 0 or j >= n:
                        continue
                    evL_j = float(ev_long_h[j])
                    evS_j = float(ev_short_h[j])
                    if int(side) == 1:
                        if (evS_j > 0.0) and (evS_j > evL_j):
                            opp_max_ev = float(max(opp_max_ev, evS_j))
                    else:
                        if (evL_j > 0.0) and (evL_j >= evS_j):
                            opp_max_ev = float(max(opp_max_ev, evL_j))
                return float(opp_max_ev)

            neighbor_opp_max_long_ev = float(_neighbor_opp_max_ev(best_idx_long, 1))
            neighbor_opp_max_short_ev = float(_neighbor_opp_max_ev(best_idx_short, -1))

            if int(side_best) == 1 and float(neighbor_opp_max_long_ev) >= float(neighbor_veto_abs):
                direction_policy = 0
                policy_direction_reason = (
                    f"neighbor_veto (opp_short_ev={neighbor_opp_max_long_ev:.6f}>=veto {neighbor_veto_abs:.6f})"
                )
            elif int(side_best) == -1 and float(neighbor_opp_max_short_ev) >= float(neighbor_veto_abs):
                direction_policy = 0
                policy_direction_reason = (
                    f"neighbor_veto (opp_long_ev={neighbor_opp_max_short_ev:.6f}>=veto {neighbor_veto_abs:.6f})"
                )

        # The chosen side for meta/diagnostics: when no-trade, still expose the better side.
        side_for_meta = int(direction_policy) if int(direction_policy) != 0 else int(side_best)

        if side_for_meta == 1:
            policy_ev_target = float(best_ev_long)
            policy_ev_bonus = 0.0
            policy_ev_penalty = 0.0
            policy_ev_mix = float(policy_ev_target)
            policy_p_pos_mix = float(best_p_pos_long)
            policy_best_h = int(best_h_long)
            policy_best_cvar = float(best_cvar_long)
        else:
            policy_ev_target = float(best_ev_short)
            policy_ev_bonus = 0.0
            policy_ev_penalty = 0.0
            policy_ev_mix = float(policy_ev_target)
            policy_p_pos_mix = float(best_p_pos_short)
            policy_best_h = int(best_h_short)
            policy_best_cvar = float(best_cvar_short)

        # Trend regime: keep higher-quality entries longer by uplifting target hold horizon.
        trend_hold_uplift = False
        trend_hold_from_h = int(policy_best_h)
        trend_hold_to_h = int(policy_best_h)
        try:
            trend_hold_on = str(os.environ.get("POLICY_TREND_MIN_HOLD_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            trend_hold_on = True
        regime_tag = str(regime_ctx_for_cluster or regime_ctx or "").strip().lower()
        is_trend_regime = ("trend" in regime_tag) or ("bull" in regime_tag) or ("bear" in regime_tag)
        if trend_hold_on and is_trend_regime and policy_horizons:
            try:
                trend_min_hold = int(float(os.environ.get("POLICY_TREND_MIN_HOLD_SEC", 300) or 300))
            except Exception:
                trend_min_hold = 300
            trend_min_hold = int(max(1, trend_min_hold))
            try:
                trend_min_dir_conf = float(os.environ.get("POLICY_TREND_MIN_HOLD_DIR_CONF", 0.58) or 0.58)
            except Exception:
                trend_min_dir_conf = 0.58
            try:
                trend_max_vpin = float(os.environ.get("POLICY_TREND_MIN_HOLD_MAX_VPIN", 0.80) or 0.80)
            except Exception:
                trend_max_vpin = 0.80
            dir_conf_now = float(mu_alpha_parts.get("mu_dir_conf") or 0.0) if isinstance(mu_alpha_parts, dict) else 0.0
            try:
                vpin_now = float(mu_alpha_parts.get("vpin") or 0.0) if isinstance(mu_alpha_parts, dict) else 0.0
            except Exception:
                vpin_now = 0.0
            if float(dir_conf_now) >= float(trend_min_dir_conf) and float(vpin_now) <= float(trend_max_vpin):
                h_sorted = sorted(int(h) for h in policy_horizons)
                h_candidate = None
                for h_val in h_sorted:
                    if int(h_val) >= int(trend_min_hold):
                        h_candidate = int(h_val)
                        break
                if h_candidate is None and h_sorted:
                    h_candidate = int(h_sorted[-1])
                if h_candidate is not None and int(h_candidate) > int(policy_best_h):
                    try:
                        h_idx = int(list(policy_horizons).index(int(h_candidate)))
                    except Exception:
                        h_idx = -1
                    if 0 <= h_idx < int(len(policy_horizons)):
                        trend_hold_uplift = True
                        trend_hold_to_h = int(h_candidate)
                        policy_best_h = int(h_candidate)
                        if side_for_meta == 1:
                            policy_ev_target = float(ev_long_h[h_idx])
                            policy_p_pos_mix = float(ppos_long_h[h_idx])
                            policy_best_cvar = float(cvar_long_h[h_idx])
                        else:
                            policy_ev_target = float(ev_short_h[h_idx])
                            policy_p_pos_mix = float(ppos_short_h[h_idx])
                            policy_best_cvar = float(cvar_short_h[h_idx])
                        policy_ev_mix = float(policy_ev_target)

        # Expose objective score diagnostics (used for dashboard/leveraging)
        policy_score_target = float(best_obj_long) if side_for_meta == 1 else float(best_obj_short)
        policy_score_bonus = float(neighbor_bonus_long) if side_for_meta == 1 else float(neighbor_bonus_short)
        policy_score_penalty = float(neighbor_penalty_long) if side_for_meta == 1 else float(neighbor_penalty_short)
        policy_score_mix = float(score_long) if side_for_meta == 1 else float(score_short)

        # Weighted CVaR is still useful for diagnostics; gate value follows the chosen horizon.
        w_arr_for_cvar = w_long if side_for_meta == 1 else w_short
        cvars_arr = np.asarray(cvars_long if side_for_meta == 1 else cvars_short, dtype=np.float64)
        policy_cvar_mix_weighted = float((w_arr_for_cvar * cvars_arr).sum()) if cvars_arr.size > 0 else 0.0
        policy_cvar_mix = float(policy_best_cvar)

        # ✅ [EV_DEBUG] 선택된 목표 호라이즌 및 EV
        if MC_VERBOSE_PRINT:
            logger.info(
                f"[EV_DEBUG] {symbol} | policy_target_h={policy_best_h}s "
                f"ev_target={policy_ev_target:.6f} ev_gate={policy_ev_mix:.6f} "
                f"score_target={policy_score_target:.6f} score_bonus={policy_score_bonus:.6f} score_pen={policy_score_penalty:.6f} score_mix={policy_score_mix:.6f} "
                f"(direction_policy={direction_policy}, side_for_meta={side_for_meta}, cvar_target={policy_cvar_mix:.6f}, cvar_w={policy_cvar_mix_weighted:.6f}, mode={obj_mode})"
            )
            print(
                f"[EV_DEBUG] {symbol} | policy_target_h={policy_best_h}s "
                f"ev_target={policy_ev_target:.6f} ev_gate={policy_ev_mix:.6f} "
                f"score_target={policy_score_target:.6f} score_bonus={policy_score_bonus:.6f} score_pen={policy_score_penalty:.6f} score_mix={policy_score_mix:.6f} "
                f"(direction_policy={direction_policy}, side_for_meta={side_for_meta}, cvar_target={policy_cvar_mix:.6f}, cvar_w={policy_cvar_mix_weighted:.6f}, mode={obj_mode})"
            )
        
        # [D] Expose per-horizon and weights in meta (meta already initialized before horizon loop)
        meta["policy_horizons"] = [int(h) for h in policy_horizons]
        meta["horizon_seq"] = [int(h) for h in policy_horizons]
        meta["policy_ev_by_h_long"] = ev_long_h.tolist()
        meta["policy_ev_by_h_short"] = ev_short_h.tolist()
        meta["policy_w_h_long"] = w_long.tolist()
        meta["policy_w_h_short"] = w_short.tolist()
        meta["policy_ev_mix_long"] = policy_ev_mix_long
        meta["policy_ev_mix_short"] = policy_ev_mix_short
        meta["policy_ev_mix_weighted_long"] = float(policy_ev_mix_long)
        meta["policy_ev_mix_weighted_short"] = float(policy_ev_mix_short)
        meta["policy_p_pos_mix_long"] = policy_p_pos_mix_long
        meta["policy_p_pos_mix_short"] = policy_p_pos_mix_short
        # Objective/constraint diagnostics
        meta["policy_obj_mode"] = str(obj_mode)
        meta["policy_lambda_var"] = float(lambda_var)
        meta["policy_max_p_liq"] = float(max_p_liq)
        meta["policy_min_profit_cost"] = float(min_profit_cost)
        meta["policy_max_dd_abs"] = float(max_dd_abs)
        meta["policy_score_target"] = float(policy_score_target)
        meta["policy_score_bonus"] = float(policy_score_bonus)
        meta["policy_score_penalty"] = float(policy_score_penalty)
        meta["policy_score_mix"] = float(policy_score_mix)
        meta["policy_obj_by_h_long"] = obj_long_raw.tolist()
        meta["policy_obj_by_h_short"] = obj_short_raw.tolist()
        meta["policy_obj_valid_long"] = [bool(x) for x in valid_long.tolist()] if hasattr(valid_long, "tolist") else []
        meta["policy_obj_valid_short"] = [bool(x) for x in valid_short.tolist()] if hasattr(valid_short, "tolist") else []
        meta["policy_cvar_by_h_long"] = cvar_long_h.tolist()
        meta["policy_cvar_by_h_short"] = cvar_short_h.tolist()
        meta["policy_var_by_h_long"] = var_long_h.tolist()
        meta["policy_var_by_h_short"] = var_short_h.tolist()
        meta["policy_p_liq_by_h_long"] = p_liq_long_h.tolist()
        meta["policy_p_liq_by_h_short"] = p_liq_short_h.tolist()
        meta["policy_dd_min_by_h_long"] = dd_min_long_h.tolist()
        meta["policy_dd_min_by_h_short"] = dd_min_short_h.tolist()
        meta["policy_profit_cost_by_h_long"] = profit_cost_long_h.tolist()
        meta["policy_profit_cost_by_h_short"] = profit_cost_short_h.tolist()
        meta["policy_direction"] = direction_policy
        meta["policy_direction_reason"] = str(policy_direction_reason) if policy_direction_reason else None
        meta["policy_both_ev_negative"] = bool(both_ev_negative)
        meta["policy_both_ev_neg_floor"] = float(both_ev_neg_floor)
        meta["policy_score_threshold_eff"] = float(score_threshold_eff)
        meta["policy_min_ev_gap"] = min_gap_eff
        meta["policy_best_ev_gap"] = float(best_ev_gap)
        meta["policy_ev_gap"] = float(ev_gap)
        meta["policy_p_pos_gap"] = float(p_pos_gap)
        meta["policy_tp_5m_long"] = float(tp_long_5m) if tp_long_5m is not None else None
        meta["policy_tp_5m_short"] = float(tp_short_5m) if tp_short_5m is not None else None
        meta["policy_tp_gate_mult_long"] = float(tp_mult_long)
        meta["policy_tp_gate_mult_short"] = float(tp_mult_short)
        meta["policy_tp_gate_block_long"] = bool(not tp_ok_long)
        meta["policy_tp_gate_block_short"] = bool(not tp_ok_short)
        meta["policy_best_h_long"] = int(best_h_long)
        meta["policy_best_h_short"] = int(best_h_short)
        meta["policy_best_ev_long"] = float(best_ev_long)
        meta["policy_best_ev_short"] = float(best_ev_short)
        meta["policy_ev_score_long"] = float(score_long)
        meta["policy_ev_score_short"] = float(score_short)
        meta["policy_ev_neighbor_bonus_long"] = float(neighbor_bonus_long)
        meta["policy_ev_neighbor_bonus_short"] = float(neighbor_bonus_short)
        meta["policy_ev_neighbor_w"] = float(neighbor_w)
        meta["policy_ev_neighbor_cap"] = float(neighbor_cap)
        meta["policy_ev_neighbor_penalty_long"] = float(neighbor_penalty_long)
        meta["policy_ev_neighbor_penalty_short"] = float(neighbor_penalty_short)
        meta["policy_ev_neighbor_pen_w"] = float(neighbor_pen_w)
        meta["policy_ev_neighbor_pen_cap"] = float(neighbor_pen_cap)
        meta["policy_ev_neighbor_veto_abs"] = float(neighbor_veto_abs)
        meta["policy_p_tp_long"] = [float(x) for x in tp_long_h.tolist()] if hasattr(tp_long_h, "tolist") else []
        meta["policy_p_tp_short"] = [float(x) for x in tp_short_h.tolist()] if hasattr(tp_short_h, "tolist") else []
        meta["policy_tp_weight_long"] = [float(x) for x in tp_weight_long_h.tolist()] if hasattr(tp_weight_long_h, "tolist") else []
        meta["policy_tp_weight_short"] = [float(x) for x in tp_weight_short_h.tolist()] if hasattr(tp_weight_short_h, "tolist") else []
        meta["policy_local_consensus_alpha"] = float(local_consensus_alpha)
        meta["policy_local_consensus_base_h"] = float(base_horizon)
        meta["policy_best_obj_long_weighted"] = float(best_obj_long_weighted)
        meta["policy_best_obj_short_weighted"] = float(best_obj_short_weighted)
        meta["policy_best_h_long_weighted"] = int(best_h_long_weighted)
        meta["policy_best_h_short_weighted"] = int(best_h_short_weighted)
        meta["policy_trend_hold_uplift"] = bool(trend_hold_uplift)
        meta["policy_trend_hold_from_h"] = int(trend_hold_from_h)
        meta["policy_trend_hold_to_h"] = int(trend_hold_to_h)
        meta["policy_ev_target"] = float(policy_ev_target)
        meta["policy_ev_bonus"] = float(policy_ev_bonus)
        meta["policy_ev_penalty"] = float(policy_ev_penalty)
        meta["policy_ev_adjust"] = float(policy_ev_bonus) - float(policy_ev_penalty)
        meta["policy_horizon_eff_sec"] = int(policy_best_h)
        meta["policy_cvar_mix_weighted"] = float(policy_cvar_mix_weighted)
        
        # Store validation warnings in meta
        if mu_alpha_warning:
            meta["ev_validation_1_warnings"] = mu_alpha_warning
        # cost_warning will be added to meta later after it's computed (around line 2670)
        
        # [C] Cost diagnostics (maker delay integrated)
        meta["pmaker_entry"] = pmaker_entry_local
        meta["pmaker_entry_delay_sec"] = pmaker_delay_sec_local
        meta["pmaker_delay_penalty"] = delay_penalty
        meta["policy_cost_entry"] = cost_entry  # ✅ [EV_VALIDATION_3] entry 비용만 저장 (exit은 compute_exit_policy_metrics에서 처리)
        meta["sl_r_used"] = sl_r
        meta["fee_roundtrip_total"] = float(fee_roundtrip_total)
        if "gross_ev_approx_5min" in locals():
            meta["gross_ev_approx_5min"] = float(gross_ev_approx_5min)
        
        # [D] Compute policy_h_eff_sec and policy_w_short_sum from learned weights
        policy_h_eff_sec = 0.0
        policy_w_short_sum = 0.0
        policy_h_eff_sec_prior = 0.0  # For validation: w_prior only (rule-based)
        try:
            h_arr_pol = np.asarray(policy_horizons, dtype=np.float64)
            w_arr_pol = w_long if side_for_meta == 1 else w_short
            policy_h_eff_sec = float(np.sum(w_arr_pol * h_arr_pol)) if w_arr_pol.size else 0.0
            policy_w_short_sum = float(np.sum(w_arr_pol[h_arr_pol <= 60.0])) if w_arr_pol.size else 0.0
            # Compute policy_h_eff_sec from w_prior only (for validation)
            if w_prior_arr.size > 0 and h_arr_pol.size == w_prior_arr.size:
                policy_h_eff_sec_prior = float(np.sum(w_prior_arr * h_arr_pol))
        except Exception:
            pass  # Already initialized to 0.0
        
        # [DIFF 3 VALIDATION] Validate dynamic weight behavior
        # Validation point 1: signal_strength ↑ → policy_h_eff_sec ↓
        # Check: w_prior-based policy_h_eff_sec should decrease as signal_strength increases
        # (Note: Final weights include EV contribution, so we validate w_prior separately)
        validation_warning = []
        if policy_h_eff_sec_prior > 0.0:
            # Expected: higher signal_strength → lower half_life → lower policy_h_eff_sec_prior
            # half_life = 1800.0 / (1.0 + s_clip)
            # For s_clip=0: half_life=1800, for s_clip=4: half_life=360
            # Higher s_clip → lower half_life → more weight on short horizons → lower policy_h_eff_sec_prior
            expected_half_life = policy_half_life_sec
            if expected_half_life > 0.0:
                # Sanity check: policy_h_eff_sec_prior should be reasonable given half_life
                # For exponential decay, effective horizon ≈ half_life * ln(2) for uniform distribution
                # But with multiple horizons, it's more complex. Just log for monitoring.
                pass  # Will be logged in meta
        
        # Validation point 2: policy_w_short_sum should be in [0, 1]
        if policy_w_short_sum < 0.0 or policy_w_short_sum > 1.0:
            validation_warning.append(
                f"policy_w_short_sum={policy_w_short_sum:.6f} is out of range [0, 1]"
            )
        
        # Log validation warnings
        if validation_warning:
            logger.warning(
                f"[DIFF3_VALIDATION] {symbol} | signal_strength={signal_strength:.4f} "
                f"policy_h_eff_sec={policy_h_eff_sec:.2f} policy_h_eff_sec_prior={policy_h_eff_sec_prior:.2f} "
                f"policy_w_short_sum={policy_w_short_sum:.6f} | Warnings: {'; '.join(validation_warning)}"
            )
        
        # Store validation metrics in meta
        meta["policy_h_eff_sec_prior"] = float(policy_h_eff_sec_prior)
        if validation_warning:
            meta["diff3_validation_warnings"] = validation_warning

        # [DIFF 2 VALIDATION] Validate Multi-Horizon Policy Mix
        diff2_validation_warnings = []
        
        # Validation point 1: policy_w_h normalization (sum should be ~1.0)
        w_arr = w_long if side_for_meta == 1 else w_short
        w_arr_sum = float(np.sum(w_arr)) if w_arr.size > 0 else 0.0
        if abs(w_arr_sum - 1.0) > 1e-5:
            diff2_validation_warnings.append(
                f"policy_w_h sum={w_arr_sum:.8f} is not ~1.0 (expected: 1.0)"
            )
        
        # Validation point 2: policy_ev_mix_long/short calculation
        # Verify: policy_ev_mix_long = sum(w_long[i] * ev_long_h[i])
        if ev_long_h.size > 0 and w_long.size == ev_long_h.size:
            ev_mix_long_manual = float((w_long * ev_long_h).sum())
            ev_mix_long_diff = abs(ev_mix_long_manual - policy_ev_mix_long)
            if ev_mix_long_diff > 1e-6:
                diff2_validation_warnings.append(
                    f"policy_ev_mix_long calculation mismatch: computed={policy_ev_mix_long:.8f} "
                    f"manual={ev_mix_long_manual:.8f} diff={ev_mix_long_diff:.8f}"
                )
        
        if ev_short_h.size > 0 and w_short.size == ev_short_h.size:
            ev_mix_short_manual = float((w_short * ev_short_h).sum())
            ev_mix_short_diff = abs(ev_mix_short_manual - policy_ev_mix_short)
            if ev_mix_short_diff > 1e-6:
                diff2_validation_warnings.append(
                    f"policy_ev_mix_short calculation mismatch: computed={policy_ev_mix_short:.8f} "
                    f"manual={ev_mix_short_manual:.8f} diff={ev_mix_short_diff:.8f}"
                )
        
        # Validation point 3: policy_horizons should match engine's multi-horizon config (short-term 0~5m).
        expected_horizons = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", policy_horizons))
        if list(policy_horizons) != expected_horizons:
            diff2_validation_warnings.append(
                f"policy_horizons={list(policy_horizons)} != expected {expected_horizons}"
            )
        
        # Validation point 4: policy_w_h length should match policy_horizons length
        if len(w_arr) != len(policy_horizons):
            diff2_validation_warnings.append(
                f"policy_w_h length={len(w_arr)} != policy_horizons length={len(policy_horizons)}"
            )
        
        # Forced logging for visibility (Goal: user can see scores)
        logger.warning(
            f"✅ [MC_SCORE] {symbol} | EV_Long: {policy_ev_mix_long*100:.4f}% | EV_Short: {policy_ev_mix_short*100:.4f}% | "
            f"mu_alpha: {float(mu_alpha_for_ev):.6f}"
        )

        # Log validation warnings separately if any
        if diff2_validation_warnings:
            logger.warning(
                f"⚠️ [DIFF2_WARN] {symbol} | policy_horizons={list(policy_horizons)} | "
                f"Warnings: {'; '.join(diff2_validation_warnings)}"
            )
        
        # Store validation metrics in meta
        if diff2_validation_warnings:
            meta["diff2_validation_warnings"] = diff2_validation_warnings
        meta["policy_w_h_sum"] = float(w_arr_sum)  # For validation

        # [D] Use TP/SL 확률 기반 EV for decision (already computed above)
        # MC simulation results are kept for meta/diagnostics only
        evs_long_arr = np.asarray(evs_long, dtype=np.float64)  # MC simulation results (for meta)
        evs_short_arr = np.asarray(evs_short, dtype=np.float64)
        pps_long_arr = np.asarray(pps_long, dtype=np.float64)
        pps_short_arr = np.asarray(pps_short, dtype=np.float64)
        cvars_long_arr = np.asarray(cvars_long, dtype=np.float64)
        cvars_short_arr = np.asarray(cvars_short, dtype=np.float64)

        # [D] Legacy MC simulation mix (for meta/diagnostics only)
        # Keep these side-consistent even when direction_policy==0 (no-trade).
        ev_mix_long = float((w_long * evs_long_arr).sum()) if (w_long.size and evs_long_arr.size and w_long.size == evs_long_arr.size) else 0.0
        ev_mix_short = float((w_short * evs_short_arr).sum()) if (w_short.size and evs_short_arr.size and w_short.size == evs_short_arr.size) else 0.0
        ppos_mix_long = float((w_long * pps_long_arr).sum()) if (w_long.size and pps_long_arr.size and w_long.size == pps_long_arr.size) else 0.0
        ppos_mix_short = float((w_short * pps_short_arr).sum()) if (w_short.size and pps_short_arr.size and w_short.size == pps_short_arr.size) else 0.0
        cvar_mix_long = float((w_long * cvars_long_arr).sum()) if (w_long.size and cvars_long_arr.size and w_long.size == cvars_long_arr.size) else 0.0
        cvar_mix_short = float((w_short * cvars_short_arr).sum()) if (w_short.size and cvars_short_arr.size and w_short.size == cvars_short_arr.size) else 0.0

        # -----------------------------
        # EV model choice:
        # - legacy: "exit_policy" (compute_exit_policy_metrics rollforward)
        # - unified exit mode: approximate with "hold" EV so entry/exit stay consistent
        # -----------------------------
        exit_mode = config.exit_mode or str(ctx.get("exit_mode", "")).strip().lower()
        use_hold_ev = exit_mode in ("unified", "entry", "entry_like", "symmetric")

        # Pre-compute hold aggregates for both directions (from net_by_h_* summaries).
        def _agg_from_dbg(dbg):
            ev_list0, win_list0, cvar_list0, h_list0 = dbg
            if not h_list0:
                return 0.0, 0.0, 0.0
            h_arr0 = np.asarray(h_list0, dtype=np.float64)
            w0 = np.exp(-h_arr0 * math.log(2) / 120.0)
            w0 = w0 / max(1e-12, float(np.sum(w0)))
            evs0 = np.asarray(ev_list0, dtype=np.float64)
            wins0 = np.asarray(win_list0, dtype=np.float64)
            cvars0 = np.asarray(cvar_list0, dtype=np.float64)
            ev_agg0 = float(np.sum(w0 * evs0))
            win_agg0 = float(np.sum(w0 * wins0))
            cvar_agg0 = float(np.quantile(cvars0, 0.25)) if cvars0.size else 0.0
            return ev_agg0, win_agg0, cvar_agg0

        hold_ev_mix_long, hold_win_mix_long, hold_cvar_mix_long = _agg_from_dbg(dbg_L)
        hold_ev_mix_short, hold_win_mix_short, hold_cvar_mix_short = _agg_from_dbg(dbg_S)
        # ✅ direction_policy=0일 때도 per_h_best/cvars_best를 정의 (더 나은 방향 사용)
        if direction_policy == 1:
            per_h_best = per_h_long
            evs_best = evs_long
            pps_best = pps_long
            cvars_best = cvars_long
        elif direction_policy == -1:
            per_h_best = per_h_short
            evs_best = evs_short
            pps_best = pps_short
            cvars_best = cvars_short
        else:  # direction_policy == 0
            # 더 나은 방향 선택 (BEST horizon 기준)
            if side_for_meta == 1:
                per_h_best = per_h_long
                evs_best = evs_long
                pps_best = pps_long
                cvars_best = cvars_long
            else:
                per_h_best = per_h_short
                evs_best = evs_short
                pps_best = pps_short
                cvars_best = cvars_short
        exit_reason_counts_per_h_long = [
            self._compress_reason_counts(m.get("exit_reason_counts"), top_k=3) for m in per_h_long
        ]
        exit_reason_counts_per_h_short = [
            self._compress_reason_counts(m.get("exit_reason_counts"), top_k=3) for m in per_h_short
        ]
        exit_reason_counts_per_h_best = exit_reason_counts_per_h_long if side_for_meta == 1 else exit_reason_counts_per_h_short
        exit_t_mean_per_h_long = [float(m.get("exit_t_mean_sec", 0.0) or 0.0) for m in per_h_long]
        exit_t_mean_per_h_short = [float(m.get("exit_t_mean_sec", 0.0) or 0.0) for m in per_h_short]
        exit_t_p50_per_h_long = [float(m.get("exit_t_p50_sec", 0.0) or 0.0) for m in per_h_long]
        exit_t_p50_per_h_short = [float(m.get("exit_t_p50_sec", 0.0) or 0.0) for m in per_h_short]
        exit_t_mean_per_h_best = exit_t_mean_per_h_long if side_for_meta == 1 else exit_t_mean_per_h_short
        exit_t_p50_per_h_best = exit_t_p50_per_h_long if side_for_meta == 1 else exit_t_p50_per_h_short
        
        # ✅ [EV_VALIDATION 2] exit이 min_hold 근처에서 반복적으로 발생 검증
        min_hold_sec_val = float(self.MIN_HOLD_SEC_DIRECTIONAL)
        exit_t_mean_avg_long = float(np.mean(exit_t_mean_per_h_long)) if exit_t_mean_per_h_long else 0.0
        exit_t_mean_avg_short = float(np.mean(exit_t_mean_per_h_short)) if exit_t_mean_per_h_short else 0.0
        exit_t_mean_avg_best = exit_t_mean_avg_long if side_for_meta == 1 else exit_t_mean_avg_short
        if exit_t_mean_avg_best > 0:
            hold_ratio = exit_t_mean_avg_best / max(1.0, min_hold_sec_val)
            if 0.9 <= hold_ratio <= 1.2:  # min_hold의 90%~120% 범위
                if MC_VERBOSE_PRINT or _throttled_log(symbol, "EV_VALIDATION_2_HOLD_RATIO", 60_000):
                    logger.warning(
                        f"[EV_VALIDATION_2] {symbol} | exit_t_mean_avg={exit_t_mean_avg_best:.1f}s is near min_hold={min_hold_sec_val:.1f}s (ratio={hold_ratio:.2f}) - frequent early exits"
                    )
                    if MC_VERBOSE_PRINT:
                        print(
                            f"[EV_VALIDATION_2] {symbol} | exit_t_mean_avg={exit_t_mean_avg_best:.1f}s is near min_hold={min_hold_sec_val:.1f}s (ratio={hold_ratio:.2f}) - frequent early exits"
                        )
        # Pick exit-policy diagnostics from the targeted horizon (Plan A: policy_best_h).
        policy_best_idx = -1
        try:
            policy_best_idx = int(list(policy_horizons).index(int(policy_best_h)))
        except Exception:
            policy_best_idx = -1
        per_h_idx = policy_best_idx if (per_h_best and 0 <= policy_best_idx < len(per_h_best)) else ((len(per_h_best) - 1) if per_h_best else -1)
        if per_h_idx >= 0:
            exit_reason_counts_policy = per_h_best[per_h_idx].get("exit_reason_counts") or {}
            exit_t_mean_sec = float(per_h_best[per_h_idx].get("exit_t_mean_sec", 0.0) or 0.0)
        else:
            exit_reason_counts_policy = {}
            exit_t_mean_sec = 0.0
        weight_peak_h = int(policy_horizons[int(np.argmax(w_arr))]) if policy_horizons else 0
        best_h = int(policy_best_h)

        # [D] Drive return values from TP/SL 확률 기반 policy mix (not hold EV)
        # ✅ [EV_VALIDATION_4 FIX] 방향 선택 및 게이트는 항상 policy_ev_mix 기반으로 고정
        # use_hold_ev는 backward compatibility를 위해 유지되지만, 실제 방향 결정이나 게이트에는 사용되지 않음
        # hold_ev_mix는 메타/진단용으로만 사용됨
        if MC_VERBOSE_PRINT:
            logger.info(f"[EV_DEBUG] {symbol} | use_hold_ev={use_hold_ev} exit_mode={exit_mode} (policy_ev_mix 기반으로 고정)")
            print(f"[EV_DEBUG] {symbol} | use_hold_ev={use_hold_ev} exit_mode={exit_mode} (policy_ev_mix 기반으로 고정)")
        
        # ✅ [EV_VALIDATION_4] 방향 선택은 항상 policy_ev_mix 기반으로 고정 (use_hold_ev 무관)
        direction = direction_policy
        
        # ✅ [EV_VALIDATION_4] direction_policy=0일 때도 policy_ev_mix 값을 반환 (대시보드에서 표시하기 위해)
        # can_enter=False로 설정되어 거래는 하지 않지만, EV 값은 표시됨
        if direction_policy == 0:
            # Plan A: even on no-trade, expose the targeted horizon metrics of the better side.
            ev = float(policy_ev_mix)
            win = float(policy_p_pos_mix)
            # CVaR는 보수적으로 더 나쁜 쪽 사용
            # CVaR는 보수적으로 더 나쁜 쪽 사용
            if hasattr(cvars_best, 'size') and cvars_best.size > 0:
                cvar_gate = float(np.min(cvars_best))
            elif isinstance(cvars_best, list) and cvars_best:
                cvar_gate = float(min(cvars_best))
            else:
                cvar_gate = float(policy_cvar_mix)
            cvar = float(cvar_gate)
            if MC_VERBOSE_PRINT:
                logger.info(
                    f"[EV_DEBUG] {symbol} | direction_policy=0: using better policy_ev_mix={ev:.6f} win={win:.4f} cvar={cvar:.6f} (no trade, but ev shown)"
                )
                print(f"[EV_DEBUG] {symbol} | direction_policy=0: using better policy_ev_mix={ev:.6f} win={win:.4f} cvar={cvar:.6f} (no trade, but ev shown)")
        else:
            # ✅ [EV_VALIDATION_4] EV, win, CVaR는 항상 policy 기반 값 사용
            ev = float(policy_ev_mix)
            win = float(policy_p_pos_mix)
            # ✅ [EV_VALIDATION_4] CVaR 게이트는 policy_cvar_mix 사용 (보수적으로 최소값 사용)
            # ✅ [EV_VALIDATION_4] CVaR 게이트는 policy_cvar_mix 사용 (보수적으로 최소값 사용)
            if hasattr(cvars_best, 'size') and cvars_best.size > 0:
                cvar_gate = float(np.min(cvars_best))
            elif isinstance(cvars_best, list) and cvars_best:
                cvar_gate = float(min(cvars_best))
            else:
                cvar_gate = float(policy_cvar_mix)
            cvar = float(cvar_gate)
            
            # ✅ [EV_DEBUG] 최종 ev 값 확인
            if MC_VERBOSE_PRINT:
                logger.info(
                    f"[EV_DEBUG] {symbol} | Final ev={ev:.6f} (policy_ev_mix={policy_ev_mix:.6f}) win={win:.4f} cvar={cvar:.6f} direction={direction}"
                )
                print(f"[EV_DEBUG] {symbol} | Final ev={ev:.6f} (policy_ev_mix={policy_ev_mix:.6f}) win={win:.4f} cvar={cvar:.6f} direction={direction}")
            
            # ✅ [EV_DEBUG] ev가 0인 경우 경고
            if abs(ev) < 1e-6:
                if MC_VERBOSE_PRINT or _throttled_log(symbol, "FINAL_EV_NEAR_ZERO", 60_000):
                    logger.warning(
                        f"[EV_DEBUG] {symbol} | ⚠️  Final ev is near 0! policy_ev_mix={policy_ev_mix:.6f} direction_policy={direction_policy}"
                    )
                    if MC_VERBOSE_PRINT:
                        print(
                            f"[EV_DEBUG] {symbol} | ⚠️  Final ev is near 0! policy_ev_mix={policy_ev_mix:.6f} direction_policy={direction_policy}"
                        )
        
        # exit-policy metrics were picked from the targeted horizon (policy_best_h) above.
        
        # ✅ [EV_DEBUG] 최종 ev 결정 로그
        if MC_VERBOSE_PRINT:
            logger.info(
                f"[EV_DEBUG] {symbol} | Final ev decision: direction={direction} ev={ev:.6f} policy_ev_mix={policy_ev_mix:.6f} direction_policy={direction_policy} (policy 기반 고정)"
            )
            print(
                f"[EV_DEBUG] {symbol} | Final ev decision: direction={direction} ev={ev:.6f} policy_ev_mix={policy_ev_mix:.6f} direction_policy={direction_policy} (policy 기반 고정)"
            )
        
        # ✅ [EV_VALIDATION_4] hold_ev_mix는 메타/진단용으로만 저장 (방향 결정이나 게이트에 사용하지 않음)
        # hold_ev_mix 값은 이미 계산되어 메타에 저장됨 (아래에서 저장)

        # Exit-policy diagnostics: which rule dominates exits?
        policy_exit_unrealized_dd_frac = None
        policy_exit_hold_bad_frac = None
        policy_exit_score_flip_frac = None
        try:
            cnt = exit_reason_counts_policy or {}
            tot = float(sum(int(v) for v in cnt.values())) if isinstance(cnt, dict) else 0.0
            if tot > 0:
                policy_exit_unrealized_dd_frac = float(cnt.get("unrealized_dd", 0)) / tot
                policy_exit_hold_bad_frac = float(cnt.get("hold_bad", 0)) / tot
                policy_exit_score_flip_frac = float(cnt.get("score_flip", 0)) / tot
        except Exception:
            policy_exit_unrealized_dd_frac = None
            policy_exit_hold_bad_frac = None
            policy_exit_score_flip_frac = None
        
        # [DIFF 4 VALIDATION] Validate Exit Reason statistics
        diff4_validation_warnings = []
        
        # Validation point 1: exit reason counts structure
        # Check that exit_reason_counts_per_h_long/short are lists with correct length
        if exit_reason_counts_per_h_long is not None:
            if not isinstance(exit_reason_counts_per_h_long, list):
                diff4_validation_warnings.append(
                    f"exit_reason_counts_per_h_long is not a list: {type(exit_reason_counts_per_h_long)}"
                )
            elif len(exit_reason_counts_per_h_long) != len(policy_horizons):
                diff4_validation_warnings.append(
                    f"exit_reason_counts_per_h_long length={len(exit_reason_counts_per_h_long)} != "
                    f"policy_horizons length={len(policy_horizons)}"
                )
        
        if exit_reason_counts_per_h_short is not None:
            if not isinstance(exit_reason_counts_per_h_short, list):
                diff4_validation_warnings.append(
                    f"exit_reason_counts_per_h_short is not a list: {type(exit_reason_counts_per_h_short)}"
                )
            elif len(exit_reason_counts_per_h_short) != len(policy_horizons):
                diff4_validation_warnings.append(
                    f"exit_reason_counts_per_h_short length={len(exit_reason_counts_per_h_short)} != "
                    f"policy_horizons length={len(policy_horizons)}"
                )
        
        # Validation point 2: short-term config sanity (0~5m 정책에서 min_hold가 너무 크면 이상)
        min_hold_sec_actual = int(self.MIN_HOLD_SEC_DIRECTIONAL)
        if min_hold_sec_actual > 60:
            diff4_validation_warnings.append(
                f"min_hold_sec={min_hold_sec_actual} seems too high for 0~5m policy"
            )
        
        # Validation point 3: Check exit reason fractions sum to reasonable value
        # (They should be fractions of total exits, so sum should be <= 1.0)
        if policy_exit_unrealized_dd_frac is not None and policy_exit_hold_bad_frac is not None and policy_exit_score_flip_frac is not None:
            frac_sum = policy_exit_unrealized_dd_frac + policy_exit_hold_bad_frac + policy_exit_score_flip_frac
            if frac_sum > 1.0 + 1e-5:  # Allow small floating point error
                diff4_validation_warnings.append(
                    f"Exit reason fractions sum={frac_sum:.6f} > 1.0 "
                    f"(unrealized_dd={policy_exit_unrealized_dd_frac:.6f} "
                    f"hold_bad={policy_exit_hold_bad_frac:.6f} "
                    f"score_flip={policy_exit_score_flip_frac:.6f})"
                )
        
        # Validation point 4: Check that exit times respect min_hold_sec
        # (This is a soft check - we can't enforce it strictly as some exits may be before min_hold)
        # But we can log if there are many early exits
        if exit_t_mean_per_h_long and exit_t_mean_per_h_short:
            min_hold_sec_float = float(min_hold_sec_actual)
            early_exits_long = sum(1 for t in exit_t_mean_per_h_long if t < min_hold_sec_float)
            early_exits_short = sum(1 for t in exit_t_mean_per_h_short if t < min_hold_sec_float)
            if early_exits_long > len(exit_t_mean_per_h_long) * 0.5:  # More than 50% early exits
                diff4_validation_warnings.append(
                    f"Many early exits in LONG: {early_exits_long}/{len(exit_t_mean_per_h_long)} "
                    f"exits before min_hold_sec={min_hold_sec_actual}s"
                )
            if early_exits_short > len(exit_t_mean_per_h_short) * 0.5:
                diff4_validation_warnings.append(
                    f"Many early exits in SHORT: {early_exits_short}/{len(exit_t_mean_per_h_short)} "
                    f"exits before min_hold_sec={min_hold_sec_actual}s"
                )
        
        # Log validation warnings
        if diff4_validation_warnings:
            logger.warning(
                f"[DIFF4_VALIDATION] {symbol} | min_hold_sec={min_hold_sec_actual} "
                f"policy_exit_unrealized_dd_frac={policy_exit_unrealized_dd_frac} "
                f"policy_exit_hold_bad_frac={policy_exit_hold_bad_frac} "
                f"policy_exit_score_flip_frac={policy_exit_score_flip_frac} | "
                f"Warnings: {'; '.join(diff4_validation_warnings)}"
            )
        
        # Store validation metrics in meta
        if diff4_validation_warnings:
            meta["diff4_validation_warnings"] = diff4_validation_warnings
        meta["min_hold_sec"] = int(min_hold_sec_actual)  # For validation
        
        side_for_calc = int(direction) if int(direction) != 0 else int(side_for_meta)
        picked_side = "LONG" if side_for_calc == 1 else "SHORT"
        ev_model = "hold" if use_hold_ev else "exit_policy"
        if MC_VERBOSE_PRINT:
            logger.info(
                f"[SIDE_CHOICE] {symbol} | model={ev_model} picked={picked_side} | "
                f"policy_ev_mix={policy_ev_mix:.6f} policy_p_pos_mix={policy_p_pos_mix:.4f} | "
                f"hold_best_ev_L={best_ev_L:.6f} hold_best_ev_S={best_ev_S:.6f} | "
                f"horizon_policy={best_h}"
            )
            print(
                f"[SIDE_CHOICE] {symbol} | model={ev_model} picked={picked_side} | "
                f"policy_ev_mix={policy_ev_mix:.6f} policy_p_pos_mix={policy_p_pos_mix:.4f} | "
                f"horizon_policy={best_h}"
            )
        
        # hold-to-horizon debug lists (match picked direction for consistency)
        ev_list, win_list, cvar_list, h_list = dbg_L if int(side_for_calc) == 1 else dbg_S

        # ✅ Step B: ev_raw가 어디서 만들어지는지 확인 (horizon별 ev_h, win_h 출력)
        # Numpy-safe check
        has_ev = ev_list.size > 0 if hasattr(ev_list, 'size') else bool(ev_list)
        has_win = win_list.size > 0 if hasattr(win_list, 'size') else bool(win_list)

        if has_ev and has_win:
            log_msg = f"[NET_STATS] {symbol} | ev_h={ev_list} win_h={win_list} fee_rt={fee_rt:.6f} horizons={h_list}"
            if MC_VERBOSE_PRINT:
                logger.info(log_msg)
                print(log_msg)
        
        # gate용 baseline(lev=1): 동일 price path에서 direction만 반영
        net_by_h_base = net_by_h_long_base if int(side_for_calc) == 1 else net_by_h_short_base
        net_by_h = net_by_h_long if int(side_for_calc) == 1 else net_by_h_short

        # Numpy-safe check for h_list
        has_h = h_list.size > 0 if hasattr(h_list, 'size') else bool(h_list)

        if not has_h:
            logger.warning(f"[COST_DEBUG] {symbol} | h_list is empty, early return")
            if MC_VERBOSE_PRINT:
                print(f"[EARLY_RETURN_3] {symbol} | fee_roundtrip_total={fee_roundtrip_total} > 0.01 (returning ev=0)")
            return {"can_enter": False, "ev": 0.0, "ev_raw": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": direction, "kelly": 0.0, "size_frac": 0.0}

        # exp-decay weights (half-life=120s)
        h_arr = np.asarray(h_list, dtype=np.float64)
        w = np.exp(-h_arr * math.log(2) / 120.0)
        w = w / np.sum(w)
        evs = np.asarray(ev_list, dtype=np.float64)
        wins = np.asarray(win_list, dtype=np.float64)
        cvars = np.asarray(cvar_list, dtype=np.float64)

        ev_agg = float(np.sum(w * evs))
        win_agg = float(np.sum(w * wins))
        cvar_agg = float(np.quantile(cvars, 0.25))
        horizon_weights = {int(h): float(w_i) for h, w_i in zip(h_list, w)}

        # ✅ Step C: horizon 600초 고정 테스트
        _force_horizon_600 = getattr(self, "_force_horizon_600", False)
        if _force_horizon_600:
            if 600 in h_list:
                best_h = 600
                best_ev = ev_list[h_list.index(600)]
            else:
                # 600초가 없으면 첫 번째 horizon 사용
                if h_list:
                    best_h = h_list[0]
                    best_ev = ev_list[0]
        
        if best_h is None:
            logger.warning(f"[COST_DEBUG] {symbol} | best_h is None, early return")
            if MC_VERBOSE_PRINT:
                print(f"[EARLY_RETURN_4] {symbol} | MC validation failed (returning ev=0)")
            return {"can_enter": False, "ev": 0.0, "ev_raw": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": direction, "kelly": 0.0, "size_frac": 0.0}

        # ✅ Step 2: best_h에 해당하는 net_mat로 검증 로그 출력
        net_mat_for_verification = None
        if best_h is not None and best_h in net_by_h:
            net_mat_for_verification = net_by_h[best_h]
            net_arr = np.asarray(net_mat_for_verification, dtype=np.float64)
            net_mat_mean = float(net_arr.mean())
            net_mat_win = float((net_arr > 0).mean())
            net_mat_std = float(net_arr.std())
            net_mat_min = float(net_arr.min())
            net_mat_max = float(net_arr.max())
            if MC_VERBOSE_PRINT:
                logger.info(
                    f"[DBG_NET] {symbol} | best_h={best_h} | "
                    f"net_mat_mean={net_mat_mean:.8f} net_mat_win={net_mat_win:.4f} net_mat_std={net_mat_std:.8f} | "
                    f"net_mat_min={net_mat_min:.8f} net_mat_max={net_mat_max:.8f} | "
                    f"sigma_used={sigma:.8f} | "
                    f"ev_agg={ev_agg:.8f} win_agg={win_agg:.4f} cvar_agg={cvar_agg:.8f}"
                )
        
        # hold-to-horizon metrics
        ev_hold = float(ev_agg)
        p_pos_hold = float(win_agg)
        cvar_hold = float(cvar_agg)

        # ✅ "정책-청산 반영 EV/승률"을 진입/홀드 판단에 사용 (완전 대체)

        exit_reason_counts = exit_reason_counts_policy

        # ev is net ROE already; reconstruct a gross ROE proxy by adding ROE execution cost back.
        execution_cost_roe = float(execution_cost) * float(leverage)
        ev_gross = float(ev) + float(execution_cost_roe)
        ev_raw = float(ev_gross)
        
        # ✅ [EV_DEBUG] ev_raw 계산 로그
        if MC_VERBOSE_PRINT:
            logger.info(
                f"[EV_DEBUG] {symbol} | ev_raw calculation: ev={ev:.6f} execution_cost={execution_cost:.6f} ev_gross={ev_gross:.6f} ev_raw={ev_raw:.6f}"
            )
            print(f"[EV_DEBUG] {symbol} | ev_raw calculation: ev={ev:.6f} execution_cost={execution_cost:.6f} ev_gross={ev_gross:.6f} ev_raw={ev_raw:.6f}")

        # [C] PMaker delay penalty is now applied per-horizon in the loop above
        # Legacy discount logic removed - delay penalty is computed horizon-specifically
        # EV already includes horizon-specific delay penalty from MC simulation
        pmaker_discount = 1.0  # Kept for backward compatibility in meta

        # EV decompose (quick sanity): drift-only gross vs fee for 1m/5m (0~5분 단기 기준)
        try:
            mu_exit_per_sec = float(mu_adj) / float(SECONDS_PER_YEAR)
        except Exception:
            mu_exit_per_sec = 0.0
        gross_long_60 = float(mu_exit_per_sec * 60.0 * float(leverage))
        gross_long_300 = float(mu_exit_per_sec * 300.0 * float(leverage))
        gross_short_60 = float(-gross_long_60)
        gross_short_300 = float(-gross_long_300)
        fee_rt_total_for_decomp = float(fee_rt_total_f) * float(leverage)  # ROE cost proxy
        ev_net_approx_long_60 = float(gross_long_60 - fee_rt_total_for_decomp)
        ev_net_approx_long_300 = float(gross_long_300 - fee_rt_total_for_decomp)
        ev_net_approx_short_60 = float((-gross_long_60) - fee_rt_total_for_decomp)
        ev_net_approx_short_300 = float((-gross_long_300) - fee_rt_total_for_decomp)
        # Breakeven drift (annual) required to offset execution costs (ignores sigma/tails; drift-only proxy).
        # mu_req = fee * SECONDS_PER_YEAR / (h * leverage)
        mu_req_60 = None
        mu_req_300 = None
        mu_req_exit_mean = None
        try:
            lev_eff = max(1e-9, float(leverage))
            mu_req_60 = float(fee_rt_total_for_decomp) * float(SECONDS_PER_YEAR) / max(1e-9, 60.0 * lev_eff)
            mu_req_300 = float(fee_rt_total_for_decomp) * float(SECONDS_PER_YEAR) / max(1e-9, 300.0 * lev_eff)
            # Use the *policy*'s typical exit time (mean of chosen horizon rollforward) to contextualize EV negativity.
            # If exit_mean is short, required drift becomes huge and EV will often be negative even when |mu| looks large.
            exit_mean = float(exit_t_mean_sec) if (exit_t_mean_sec is not None) else 0.0
            if exit_mean > 1e-9:
                mu_req_exit_mean = float(fee_rt_total_for_decomp) * float(SECONDS_PER_YEAR) / max(1e-9, exit_mean * lev_eff)
        except Exception:
            mu_req_60 = None
            mu_req_300 = None
            mu_req_exit_mean = None
        # gate용(base lev=1): use base simulation (already net after base costs)
        try:
            ev1_by_h = [float(net_by_h_base[int(h)].mean()) for h in h_list]
            win1_by_h = [float((net_by_h_base[int(h)] > 0).mean()) for h in h_list]
            cvar1_by_h = [float(cvar_ensemble(net_by_h_base[int(h)], alpha=params.cvar_alpha)) for h in h_list]
            ev1 = float(np.sum(w * np.asarray(ev1_by_h, dtype=np.float64)))
            win1 = float(np.sum(w * np.asarray(win1_by_h, dtype=np.float64)))
            cvar1 = float(np.quantile(np.asarray(cvar1_by_h, dtype=np.float64), 0.25))
        except Exception:
            ev1 = float(ev)  # fallback
            win1 = float(win)
            cvar1 = float(cvar)
        
        # ✅ A) 비용 디버그 로그 (각 심볼마다 출력)
        execution_cost_oneway = float(execution_cost) / 2.0  # oneway는 roundtrip의 절반
        logger.info(
            f"[COST_DEBUG] {symbol} | "
            f"fee_roundtrip={fee_rt:.6f} | "
            f"execution_cost_oneway={execution_cost_oneway:.6f} | "
            f"expected_spread_cost={expected_spread_cost:.6f} | "
            f"slippage_dyn={slippage_dyn:.6f} | "
            f"leverage={leverage:.2f} | "
            f"ev_hold={ev_hold:.6f} p_pos_hold={p_pos_hold:.4f} | "
            f"policy_ev_mix={policy_ev_mix:.6f} policy_p_pos_mix={policy_p_pos_mix:.4f} exit_t_mean={exit_t_mean_sec:.1f}s"
        )
        
        # ✅ B) win 계산 디버그 로그 (각 심볼마다 출력)
        w_sum = float(np.sum(w))
        logger.info(
            f"[WIN_DEBUG] {symbol} | "
            f"best_h={best_h} | "
            f"horizons={h_list} | "
            f"w_sum={w_sum:.6f} | "
            f"w={[f'{w_i:.4f}' for w_i in w]} | "
            f"win_h={[f'{win_h:.4f}' for win_h in win_list]} | "
            f"ev_h={[f'{ev_h:.6f}' for ev_h in ev_list]} | "
            f"win_agg={win_agg:.4f}"
        )
        mid_h = []
        ev_mid = None
        win_mid = None
        cvar_mid = None
        if mid_h:
            mid_concat = np.concatenate(mid_h, axis=0)
            ev_mid = float(mid_concat.mean())
            win_mid = float((mid_concat > 0).mean())
            cvar_mid = float(cvar_ensemble(mid_concat, alpha=params.cvar_alpha))

        # entry gating: 비용을 충분히 이길 때만, 승률/꼬리/중기 필터
        cost_floor = float(fee_rt) * float(leverage)  # ROE cost
        # Short-term (0~5m): gate thresholds are driven by regime params (net EV after costs).
        ev_floor = float(params.profit_target)
        win_floor = float(params.min_win)
        cvar_floor_abs = cost_floor * 3.0  # 요구: cvar_agg > -cvar_floor_abs
        # 비용 대비 마진 게이트(옵션): ev_for_gate >= k * fee_roundtrip_total
        ev_cost_mult_gate = 0.0
        ev_cost_floor = None
        ev_cost_mult_gate = config.ev_cost_mult_gate
        if ev_cost_mult_gate > 0:
            ev_cost_floor = float(ev_cost_mult_gate) * float(fee_roundtrip_total) * float(leverage)

        # ✅ SCORE-ONLY MODE: Entry Gate 비활성화
        # Score 기반 진입만 사용하므로 여기서는 항상 통과
        use_score_only = config.score_only_mode
        
        if use_score_only:
            can_enter = True  # Score에서만 판단
            blocked_by = []
            if MC_VERBOSE_PRINT:
                logger.info(f"[SCORE_ONLY] {symbol} | Entry gate bypassed - using score-based entry")
        else:
            # 기존 Entry Gate 로직 (레거시)
            can_enter = False
            blocked_by = []
            
            # 기본 게이트 체크
            if ev <= ev_floor:
                blocked_by.append(f"ev_gate(ev={ev:.6f} <= ev_floor={ev_floor:.6f})")
            if ev_cost_floor is not None and ev <= float(ev_cost_floor):
                blocked_by.append(f"ev_cost_gate(ev={ev:.6f} <= k*fee={float(ev_cost_floor):.6f}, k={ev_cost_mult_gate:.2f}, fee_rt_total={float(fee_roundtrip_total):.6f})")
            if win < win_floor:
                blocked_by.append(f"win_gate(win={win:.4f} < win_floor={win_floor:.4f})")
            if cvar <= -cvar_floor_abs:
                blocked_by.append(f"cvar_gate(cvar={cvar:.6f} <= -cvar_floor_abs={-cvar_floor_abs:.6f})")
            
            ev_ok = ev > ev_floor
            if ev_cost_floor is not None:
                ev_ok = ev_ok and (ev > float(ev_cost_floor))
            if ev_ok and win >= win_floor and cvar > -cvar_floor_abs:
                mid_ok = True
                mid_cut_reason = None
                if ev_mid is not None:
                    if ev_mid < 0.0:
                        mid_ok = False
                        mid_cut_reason = f"ev_mid={ev_mid:.6f} < 0.0"
                if win_mid is not None and mid_ok:
                    if win_mid < 0.50:
                        mid_ok = False
                        mid_cut_reason = f"win_mid={win_mid:.4f} < 0.50"
                if not mid_ok and mid_cut_reason is not None:
                    blocked_by.append(f"mid_gate({mid_cut_reason})")
                can_enter = mid_ok
            else:
                # 기본 게이트에서 막힌 경우 mid 체크는 스킵
                pass
        
        # 진입게이트 상세 로그 (항상 출력)
        if True:  # ✅ 로그 제한 제거: 항상 출력
            direction_str = "LONG" if direction == 1 else "SHORT"
            aggressive = abs(leverage) > 5.0  # 레버리지 기반 aggressive 판단
            
            # ✅ f-string 조건부 포맷팅 수정: None 체크 후 포맷팅
            win_mid_str = f"{win_mid:.4f}" if win_mid is not None else "None"
            ev_mid_str = f"{ev_mid:.6f}" if ev_mid is not None else "None"
            log_msg = (
                f"[ENTRY_GATE] 선택방향={direction_str} | "
                f"p_pos_for_gate={win:.4f}, win_floor={win_floor:.4f} | "
                f"ev_for_gate={ev:.6f}, ev_floor={ev_floor:.6f}, cost_p90={cost_floor:.6f}, profit_target={params.profit_target:.6f} | "
                f"cvar={cvar:.6f}, cvar_floor_abs={cvar_floor_abs:.6f} | "
                f"aggressive={aggressive}, win_mid={win_mid_str}, ev_mid={ev_mid_str}, mid_cut={not can_enter and len(blocked_by) > 0 and any('mid_gate' in b for b in blocked_by)} | "
                f"최종 blocked_by={', '.join(blocked_by) if blocked_by else 'PASS'}"
            )
            logger.info(log_msg)
            self._gate_log_count += 1

        # Kelly raw (EV / variance proxy)
        variance_proxy = float(sigma * sigma)
        kelly_raw = max(0.0, ev / max(variance_proxy, 1e-6))

        # CVaR 기반 축소 (레버리지 고려)
        leverage_penalty = max(1.0, abs(leverage) / 5.0)
        cvar_penalty = max(0.05, 1.0 - params.cvar_scale * abs(cvar) * leverage_penalty)

        # 고레버리지일수록 Kelly 상한을 자동 축소
        kelly_cap = params.max_kelly / leverage_penalty

        kelly = min(kelly_raw * cvar_penalty, kelly_cap)

        confidence = float(win)  # hub confidence는 win 기반 유지
        size_frac = float(max(0.0, kelly * confidence))

        # Event-based MC (first passage TP/SL)
        # ✅ 기본값 직접 완화: 0.3%/0.5% (이전: 0.05%/0.08%)
        tp_pct = config.default_tp_pct
        sl_pct = config.default_sl_pct
        # ✅ 격리 테스트: dist를 gaussian으로 고정
        dist_mode = "gaussian" if getattr(self, "_force_gaussian_dist", False) else str(getattr(self, "_tail_mode", self.default_tail_mode))
        event_metrics = self.mc_first_passage_tp_sl(
            s0=price,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            mu=mu_adj,
            sigma=sigma,
            dt=self.dt,
            max_steps=int(max(self.horizons)),
            n_paths=int(params.n_paths),
            cvar_alpha=params.cvar_alpha,
            seed=seed,
            dist_override=dist_mode,
        )

        # event EV/CVaR를 % 수익률 단위로 환산 (R * SL%)
        event_ev_pct = None
        event_cvar_pct = None
        event_unified_score = None
        try:
            if event_metrics.get("event_ev_r") is not None:
                event_ev_pct = float(event_metrics["event_ev_r"]) * sl_pct
        except Exception:
            event_ev_pct = None
        try:
            if event_metrics.get("event_cvar_r") is not None:
                event_cvar_pct = float(event_metrics["event_cvar_r"]) * sl_pct
        except Exception:
            event_cvar_pct = None
        try:
            if event_ev_pct is not None and event_cvar_pct is not None:
                lambda_val = float(ctx.get("unified_lambda", config.unified_risk_lambda))
                rho_val = float(ctx.get("rho", config.unified_rho))
                tau_evt = float(event_metrics.get("event_t_median") or max(self.horizons))
                tau_evt = float(max(1.0, tau_evt))
                event_unified_score = float(event_ev_pct - lambda_val * abs(event_cvar_pct) - rho_val * tau_evt)
        except Exception:
            event_unified_score = None

        # ✅ BTC 상관관계 반영 (리스크 관리)
        btc_corr = _s(ctx.get("btc_corr"), 0.0)
        if btc_corr > 0.7:
            # 상관관계가 높으면 Kelly 비중 축소 (분산투자 효과 감소)
            kelly *= 0.8
            logger.info(f"[RISK] {symbol} High BTC corr={btc_corr:.2f} -> Kelly reduced to {kelly:.2f}")

        # ✅ 역선택 방지 (Adverse Selection Protection)
        # SCORE_ONLY_MODE에서는 우회
        use_score_only = config.score_only_mode
        
        if not use_score_only:
            pmaker_entry = _s(ctx.get("pmaker_entry"), 0.0)
            pmaker_threshold = 0.3  # 임계값 (필요시 환경변수화)
            if pmaker_entry > 0 and pmaker_entry < pmaker_threshold:
                logger.info(f"[ADVERSE_SELECTION] {symbol} pmaker_entry={pmaker_entry:.2f} < {pmaker_threshold} -> Entry blocked")
                can_enter = False

        # [EV_DEBUG] 최종 EV 값 확인
        if MC_VERBOSE_PRINT:
            print(f"[EV_DEBUG] evaluate_entry_metrics: symbol={symbol} ev={ev} win={win} cvar={cvar} can_enter={can_enter} direction={direction}")
            logger.info(f"[EV_DEBUG] evaluate_entry_metrics: symbol={symbol} ev={ev} win={win} cvar={cvar} can_enter={can_enter} direction={direction}")

            # [EV_FINAL_DEBUG] 최종 반환값 확인
            print(
                f"[EV_FINAL_DEBUG] {symbol} | Returning: ev={ev:.6f} policy_ev_mix={policy_ev_mix:.6f} ev_mix_long={ev_mix_long:.6f} ev_mix_short={ev_mix_short:.6f} direction={direction}"
            )

        meta = {
            "can_enter": bool(can_enter),
            "ev": float(ev) if not math.isnan(float(ev)) else 0.0,
            "ev_raw": float(ev_raw) if not math.isnan(float(ev_raw)) else 0.0,
            "win": win,
            "cvar": cvar,
            "best_h": int(best_h),
            "direction": int(direction),
            "kelly": float(kelly),
            "size_frac": float(size_frac),
            "ev_hold": float(ev_hold),
            "p_pos_hold": float(p_pos_hold),
            "cvar_hold": float(cvar_hold),
            "policy_ev_mix": float(policy_ev_mix),
            "policy_p_pos_mix": float(policy_p_pos_mix),
            "policy_cvar_mix": float(policy_cvar_mix),
            "policy_cvar_gate": float(cvar_gate),
            "policy_ev_target": float(policy_ev_target) if "policy_ev_target" in locals() else None,
            "policy_ev_bonus": float(policy_ev_bonus) if "policy_ev_bonus" in locals() else None,
            "policy_ev_penalty": float(policy_ev_penalty) if "policy_ev_penalty" in locals() else None,
            "policy_ev_adjust": float(policy_ev_bonus - policy_ev_penalty)
            if ("policy_ev_bonus" in locals() and "policy_ev_penalty" in locals())
            else None,
            "policy_ev_score_long": float(score_long) if "score_long" in locals() else None,
            "policy_ev_score_short": float(score_short) if "score_short" in locals() else None,
            "policy_best_ev_gap": float(best_ev_gap) if "best_ev_gap" in locals() else None,
            "policy_ev_gap": float(ev_gap) if "ev_gap" in locals() else None,
            "policy_ev_neighbor_veto_abs": float(neighbor_veto_abs) if "neighbor_veto_abs" in locals() else None,
            "policy_ev_mix_long": float(policy_ev_mix_long),
            "policy_ev_mix_short": float(policy_ev_mix_short),
            "policy_p_pos_mix_long": float(policy_p_pos_mix_long),
            "policy_p_pos_mix_short": float(policy_p_pos_mix_short),
            "policy_signal_strength": float(signal_strength),
            "policy_h_eff_sec": float(policy_h_eff_sec),
            "policy_h_eff_sec_prior": float(policy_h_eff_sec_prior),
            "policy_w_short_sum": float(policy_w_short_sum),
            "policy_horizons": [int(h) for h in policy_horizons],
            "policy_w_h": [float(x) for x in w_arr],
            "policy_direction": int(direction_policy),
            "policy_direction_reason": str(policy_direction_reason) if policy_direction_reason else None,
            "mu_alpha": float(mu_alpha_parts.get("mu_alpha") or 0.0)
            if isinstance(mu_alpha_parts, dict)
            else float(signal_mu),
            "mu_alpha_for_ev": float(mu_alpha_for_ev),
            "mu_alpha_raw": float(mu_alpha_parts.get("mu_alpha_raw") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_alpha_pmaker_fill_rate": float(mu_alpha_parts.get("mu_alpha_pmaker_fill_rate"))
            if isinstance(mu_alpha_parts, dict) and mu_alpha_parts.get("mu_alpha_pmaker_fill_rate") is not None
            else None,
            "mu_alpha_pmaker_boost": float(mu_alpha_parts.get("mu_alpha_pmaker_boost") or 0.0)
            if isinstance(mu_alpha_parts, dict)
            else None,
            "mu_dir_edge": float(mu_alpha_parts.get("mu_dir_edge") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_dir_conf": float(mu_alpha_parts.get("mu_dir_conf") or 0.0) if isinstance(mu_alpha_parts, dict) else None,
            "mu_dir_prob_long": float(mu_alpha_parts.get("mu_dir_prob_long") or 0.5) if isinstance(mu_alpha_parts, dict) else None,
            "mu_adjusted": float(mu_adj),
            "mu_adjusted_for_ev": float(mu_adjusted_for_ev),
            "sigma_sim": float(sigma),
            "sigma_annual": float(sigma),
            "execution_cost": float(execution_cost),
            "fee_roundtrip_total": float(execution_cost),
            "slippage_dyn": float(slippage_dyn),
            "spread_pct": float(spread_pct),
            "event_p_tp": event_metrics.get("event_p_tp"),
            "event_p_sl": event_metrics.get("event_p_sl"),
            "event_p_timeout": event_metrics.get("event_p_timeout"),
            "event_ev_r": event_metrics.get("event_ev_r"),
            "event_cvar_r": event_metrics.get("event_cvar_r"),
            "event_unified_score": event_unified_score,
            "event_sl_pct": float(sl_pct),
            "event_tp_pct": float(tp_pct),
            "event_t_median": event_metrics.get("event_t_median"),
            "event_t_mean": event_metrics.get("event_t_mean"),
            "unified_lambda": float(ctx.get("unified_lambda", config.unified_risk_lambda)),
            "unified_rho": float(ctx.get("rho", config.unified_rho)),
            "ev_by_horizon": [float(x) for x in ev_list] if (ev_list is not None and len(ev_list) > 0) else [],
            "cvar_by_horizon": [float(x) for x in cvar_list] if (cvar_list is not None and len(cvar_list) > 0) else [],
            "ev_vector": [float(x) for x in ev_list] if (ev_list is not None and len(ev_list) > 0) else [],
            "cvar_vector": [float(x) for x in cvar_list] if (cvar_list is not None and len(cvar_list) > 0) else [],
            # Directional EV curves for Score_A (vector: LONG/SHORT).
            # These are hold-to-horizon EVs (net ROE) for each side.
            "ev_by_horizon_long": [float(x) for x in (dbg_L[0] if (isinstance(dbg_L, (list, tuple)) and len(dbg_L) > 0 and len(dbg_L[0]) > 0) else [])] if "dbg_L" in locals() else [],
            "ev_by_horizon_short": [float(x) for x in (dbg_S[0] if (isinstance(dbg_S, (list, tuple)) and len(dbg_S) > 0 and len(dbg_S[0]) > 0) else [])] if "dbg_S" in locals() else [],
            "cvar_by_horizon_long": [float(x) for x in (dbg_L[2] if (isinstance(dbg_L, (list, tuple)) and len(dbg_L) > 2 and len(dbg_L[2]) > 0) else [])] if "dbg_L" in locals() else [],
            "cvar_by_horizon_short": [float(x) for x in (dbg_S[2] if (isinstance(dbg_S, (list, tuple)) and len(dbg_S) > 2 and len(dbg_S[2]) > 0) else [])] if "dbg_S" in locals() else [],
            "horizon_seq": [int(h) for h in h_list] if (h_list is not None and len(h_list) > 0) else [],
            "horizon_seq_long": [int(h) for h in (dbg_L[3] if (isinstance(dbg_L, (list, tuple)) and len(dbg_L) > 3 and len(dbg_L[3]) > 0) else [])] if "dbg_L" in locals() else [],
            "horizon_seq_short": [int(h) for h in (dbg_S[3] if (isinstance(dbg_S, (list, tuple)) and len(dbg_S) > 3 and len(dbg_S[3]) > 0) else [])] if "dbg_S" in locals() else [],
            
            # ✅ NEW: Per-horizon EV vectors for Economic NAPV Brain
            "policy_ev_per_h_long": [float(x) for x in ev_long_h] if 'ev_long_h' in locals() else [],
            "policy_ev_per_h_short": [float(x) for x in ev_short_h] if 'ev_short_h' in locals() else [],
        }
        if alpha_hit_features_np is not None:
            try:
                meta["alpha_hit_features"] = [float(x) for x in alpha_hit_features_np.tolist()]
            except Exception:
                meta["alpha_hit_features"] = None

        # ✅ Payload Splitting: Core vs Detail
        meta_core = {
            k: v
            for k, v in meta.items()
            if (not isinstance(v, (np.ndarray, list)))
            or k
            in (
                "policy_horizons",
                "horizon_seq",
                "horizon_seq_long",
                "horizon_seq_short",
                "policy_w_h",
                "ev_by_horizon",
                "ev_by_horizon_long",
                "ev_by_horizon_short",
                "cvar_by_horizon",
                "cvar_by_horizon_long",
                "cvar_by_horizon_short",
                "ev_vector",
                "cvar_vector",
            )
        }
        meta_detail = {k: v for k, v in meta.items() if k not in meta_core}
        
        # Flatten meta_core into the return dict for back-compat with orchestrator/dashboard
        res = {
            **meta_core,
            "meta": dict(meta_core),
            "can_enter": bool(can_enter),
            "ev": float(ev) if not math.isnan(float(ev)) else 0.0,
            "ev_raw": float(ev_raw) if not math.isnan(float(ev_raw)) else 0.0,
            "confidence": float(confidence),
            "reason": f"MC EV {ev*100:.2f}% Win {win*100:.1f}% CVaR {cvar*100:.2f}%",
            "kelly": float(kelly),
            "size_frac": float(size_frac),
            "direction": int(direction),
            "best_h": int(best_h),
            "meta_detail": meta_detail,
        }
        return res
    # ✅ Fixed batch size for JAX JIT stability (Static Batching)
    # CRITICAL: constants.py에서 중앙 관리됨
    from engines.mc.constants import JAX_STATIC_BATCH_SIZE

    def evaluate_entry_metrics_batch(
        self, tasks: List[Dict[str, Any]], n_paths_override: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        GLOBAL BATCHING: 여러 심볼에 대해 병렬로 Monte Carlo 평가를 수행한다.
        - tasks: List of {ctx, params, seed}
        """
        if not tasks: return []
        num_symbols = len(tasks)
        t_start = time.perf_counter()
        
        # 1. Prepare scalar inputs for Batch Gen1
        seeds = [int(t["seed"]) for t in tasks]
        s0s = [float(t["ctx"].get("price", 0.0)) for t in tasks]
        
        # Calculate drift/diffusion/mu for each
        mus = []
        sigmas = []
        leverages_list = [] # Renamed to avoid conflict with numpy array later
        mu_alpha_vals = []
        mu_alpha_raw_vals = []
        mu_dir_prob_long_vals = []
        mu_dir_edge_vals = []
        mu_dir_conf_vals = []
        alpha_hit_preds = [None for _ in range(num_symbols)]
        alpha_hit_features_list = [None for _ in range(num_symbols)]
        from regime import adjust_mu_sigma

        for idx, t in enumerate(tasks):
            ctx, params = t["ctx"], t["params"]
            sym = ctx.get("symbol", "UNKNOWN")
            closes = ctx.get("closes", [])
            bar_seconds = float(ctx.get("bar_seconds", 60.0))
            ofi_score = float(ctx.get("ofi_score", 0.0))
            regime = ctx.get("regime", "chop")
            
            # Base signal alpha
            alpha_parts = self._signal_alpha_mu_annual_parts(closes, bar_seconds, ofi_score, regime)
            mu_alpha = float(alpha_parts.get("mu_alpha", 0.0))
            
            # Base sigma from ctx or closes fallback
            sigma_base = ctx.get("sigma_sim") or ctx.get("sigma")
            if sigma_base is None and len(closes) >= 10:
                # Annualized vol fallback
                rets = np.diff(np.log(np.maximum(closes, 1e-12)))
                sigma_base = float(rets.std()) * math.sqrt(SECONDS_PER_YEAR / bar_seconds)
            
            sigma_base = float(sigma_base or 0.15)

            # Optional advanced alpha mix (batch-safe approximation)
            try:
                use_mlofi = bool(getattr(config, "alpha_use_mlofi", False))
                use_kf = bool(getattr(config, "alpha_use_kf", False))
                use_bayes = bool(getattr(config, "alpha_use_bayes", False))
                use_arima = bool(getattr(config, "alpha_use_arima", False))
                use_pf = bool(getattr(config, "alpha_use_pf", False))
                use_ml = bool(getattr(config, "alpha_use_ml", False))
                use_causal = bool(getattr(config, "alpha_use_causal", False))
                use_hurst = bool(getattr(config, "alpha_use_hurst", False))
                use_vpin = bool(getattr(config, "alpha_use_vpin", False))
                use_hawkes = bool(getattr(config, "alpha_use_hawkes", False))
                use_garch = bool(getattr(config, "alpha_use_garch", False))
            except Exception:
                use_mlofi = use_kf = use_bayes = use_arima = False
                use_pf = use_ml = use_causal = use_hurst = False
                use_vpin = use_hawkes = use_garch = False

            comp_w = 0.0
            comp_sum = 0.0

            def _add_comp(val: Any, w: float) -> None:
                nonlocal comp_w, comp_sum
                try:
                    w = float(w)
                except Exception:
                    return
                if w <= 0:
                    return
                try:
                    v = float(val)
                except Exception:
                    v = 0.0
                comp_w += w
                comp_sum += w * v

            if use_mlofi:
                _add_comp(float(ctx.get("mlofi", 0.0) or 0.0) * float(getattr(config, "mlofi_scale", 8.0)), float(getattr(config, "mlofi_weight", 0.20)))
            if use_kf:
                _add_comp(ctx.get("mu_kf", 0.0), float(getattr(config, "kf_weight", 0.20)))
            if use_bayes:
                _add_comp(ctx.get("mu_bayes", 0.0), float(getattr(config, "bayes_weight", 0.10)))
            if use_arima:
                _add_comp(ctx.get("mu_ar", 0.0), float(getattr(config, "arima_weight", 0.10)))
            if use_pf:
                _add_comp(ctx.get("mu_pf", 0.0), float(getattr(config, "pf_weight", 0.10)))
            if use_ml:
                _add_comp(ctx.get("mu_ml", 0.0), float(getattr(config, "ml_weight", 0.15)))
            if use_causal:
                _add_comp(ctx.get("mu_causal", 0.0), float(getattr(config, "causal_weight", 0.05)))

            if comp_w > 1.0:
                comp_sum = comp_sum / comp_w
                comp_w = 1.0
            mu_alpha = float(max(0.0, 1.0 - comp_w) * mu_alpha + comp_sum)

            if use_hurst:
                try:
                    h = float(ctx.get("hurst")) if ctx.get("hurst") is not None else None
                except Exception:
                    h = None
                if h is not None:
                    low = float(getattr(config, "hurst_low", 0.45))
                    high = float(getattr(config, "hurst_high", 0.55))
                    if h < low:
                        ou_w = float(getattr(config, "ou_weight", 0.7))
                        mu_ou = float(ctx.get("mu_ou", 0.0) or 0.0)
                        mu_alpha = float((1.0 - ou_w) * mu_alpha + ou_w * mu_ou)
                    elif h > high:
                        mu_alpha = float(mu_alpha * float(getattr(config, "hurst_trend_boost", 1.15)))
                    else:
                        mu_alpha = float(mu_alpha * float(getattr(config, "hurst_random_dampen", 0.25)))

            if use_vpin and ctx.get("vpin") is not None:
                try:
                    vpin = max(0.0, min(1.0, float(ctx.get("vpin"))))
                except Exception:
                    vpin = 0.0
                gamma = float(getattr(config, "vpin_gamma", 0.6))
                damp_floor = float(getattr(config, "vpin_damp_floor", 0.10))
                damp = max(damp_floor, 1.0 - gamma * vpin)
                extreme_th = float(getattr(config, "vpin_extreme_threshold", 0.90))
                if vpin >= extreme_th:
                    ou_w = float(getattr(config, "vpin_extreme_ou_weight", 0.85))
                    ou_w = min(1.0, max(0.0, ou_w))
                    mu_ou = float(ctx.get("mu_ou", 0.0) or 0.0)
                    if abs(mu_ou) > 0:
                        mu_alpha = float((1.0 - ou_w) * mu_alpha + ou_w * mu_ou)
                mu_alpha = float(mu_alpha * damp)
                sigma_mult = float(1.0 + float(getattr(config, "vpin_sigma_k", 0.8)) * vpin)
                sigma_mult = min(sigma_mult, max(1.0, float(getattr(config, "vpin_sigma_cap_mult", 2.5))))
                sigma_base = float(sigma_base) * sigma_mult

            if use_hawkes:
                mu_alpha = float(mu_alpha + float(getattr(config, "hawkes_boost_k", 0.3)) * float(ctx.get("hawkes_boost", 0.0) or 0.0))

            if use_garch:
                sigma_g = ctx.get("sigma_garch")
                if sigma_g is not None:
                    try:
                        sigma_g = float(sigma_g)
                    except Exception:
                        sigma_g = 0.0
                    if sigma_g > 0:
                        w_g = float(getattr(config, "garch_sigma_weight", 0.5))
                        sigma_base = float((1.0 - w_g) * float(sigma_base) + w_g * sigma_g)

            # Directional correction in batch path (parity with single-symbol evaluator).
            mu_alpha_raw = float(mu_alpha)
            try:
                use_dir_corr = bool(getattr(config, "alpha_direction_use", False))
            except Exception:
                use_dir_corr = False
            dir_edge = 0.0
            dir_conf = 0.0
            dir_prob_long = 0.5
            if use_dir_corr:
                try:
                    dir_edge = float(ctx.get("mu_dir_edge", 0.0) or 0.0)
                except Exception:
                    dir_edge = 0.0
                try:
                    dir_prob_long = float(ctx.get("mu_dir_prob_long", 0.5) or 0.5)
                except Exception:
                    dir_prob_long = 0.5
                try:
                    dir_conf = float(ctx.get("mu_dir_conf", abs(dir_edge)) or abs(dir_edge))
                except Exception:
                    dir_conf = abs(dir_edge)
                try:
                    dir_strength = float(getattr(config, "alpha_direction_strength", 0.6))
                except Exception:
                    dir_strength = 0.6
                try:
                    dir_conf_th = float(getattr(config, "alpha_direction_min_confidence", 0.55))
                except Exception:
                    dir_conf_th = 0.55
                dir_conf = float(max(0.0, min(1.0, dir_conf)))
                if abs(dir_edge) > 0.0 and dir_conf >= dir_conf_th and dir_strength > 0.0:
                    dir_target = math.copysign(abs(mu_alpha), float(dir_edge))
                    dir_blend = float(min(1.0, max(0.0, dir_strength * dir_conf)))
                    mu_alpha = float((1.0 - dir_blend) * float(mu_alpha) + dir_blend * float(dir_target))

            mu_alpha_vals.append(float(mu_alpha))
            mu_alpha_raw_vals.append(float(mu_alpha_raw))
            mu_dir_prob_long_vals.append(float(dir_prob_long))
            mu_dir_edge_vals.append(float(dir_edge))
            mu_dir_conf_vals.append(float(dir_conf))
            
            # Apply regime-based adjustments
            mu_adj, sigma_adj = adjust_mu_sigma(mu_alpha, sigma_base, regime)
            
            mus.append(mu_adj)
            sigmas.append(sigma_adj)
            leverages_list.append(params.leverage if hasattr(params, "leverage") else 1.0)

            # AlphaHit features/prediction (batch path)
            if self.alpha_hit_enabled and self.alpha_hit_trainer is not None:
                try:
                    ctx["mu_alpha"] = float(mu_alpha)
                    momentum_z = 0.0
                    if closes is not None and len(closes) >= 6:
                        rets = np.diff(np.log(np.asarray(closes, dtype=np.float64)))
                        if rets.size >= 5:
                            window = int(min(20, rets.size))
                            subset = rets[-window:]
                            mean_r = float(subset.mean())
                            std_r = float(subset.std())
                            if std_r > 1e-9:
                                momentum_z = float((subset[-1] - mean_r) / std_r)

                    ofi_z = 0.0
                    try:
                        key = (str(regime), str(ctx.get("session", "OFF")))
                        hist = self._ofi_hist.setdefault(key, [])
                        hist.append(ofi_score)
                        if len(hist) > 500:
                            hist.pop(0)
                        if len(hist) >= 5:
                            arr = np.asarray(hist, dtype=np.float64)
                            mean = float(arr.mean())
                            std = float(arr.std())
                            std = std if std > 1e-6 else 0.05
                            ofi_z = float((ofi_score - mean) / std) if std > 1e-9 else 0.0
                    except Exception:
                        ofi_z = 0.0

                    leverage_val = float(ctx.get("leverage") or 1.0)
                    features = self._extract_alpha_hit_features(
                        symbol=str(sym),
                        price=float(ctx.get("price") or 0.0),
                        mu=float(mu_adj),
                        sigma=float(sigma_adj),
                        momentum_z=float(momentum_z),
                        ofi_z=float(ofi_z),
                        regime=str(regime),
                        leverage=float(leverage_val),
                        ctx=ctx,
                    )
                    if features is not None:
                        features_np = features[0].detach().cpu().numpy().astype(np.float32)
                        alpha_hit_features_list[idx] = features_np
                        pred = self.alpha_hit_trainer.predict(features_np)
                        alpha_hit_preds[idx] = {
                            "p_tp_long": pred["p_tp_long"].detach().cpu().numpy().reshape(-1),
                            "p_sl_long": pred["p_sl_long"].detach().cpu().numpy().reshape(-1),
                            "p_tp_short": pred["p_tp_short"].detach().cpu().numpy().reshape(-1),
                            "p_sl_short": pred["p_sl_short"].detach().cpu().numpy().reshape(-1),
                        }
                except Exception:
                    alpha_hit_preds[idx] = None

        # AlphaHit warmup/beta scaling (batch-level)
        alpha_hit_beta_eff = 0.0
        alpha_hit_buffer_n = 0
        alpha_hit_loss = None
        if self.alpha_hit_enabled and self.alpha_hit_trainer is not None:
            train_stats = {}
            try:
                train_stats = self.alpha_hit_trainer.train_tick()
                alpha_hit_loss = train_stats.get("loss")
            except Exception:
                alpha_hit_loss = None
            try:
                alpha_hit_buffer_n = int(getattr(self.alpha_hit_trainer, "buffer_size", 0))
            except Exception:
                alpha_hit_buffer_n = 0
            if alpha_hit_buffer_n <= 0:
                try:
                    alpha_hit_buffer_n = int(train_stats.get("buffer_n", train_stats.get("n_samples", 0)) or 0)
                except Exception:
                    alpha_hit_buffer_n = 0
            try:
                base_beta = float(getattr(self, "alpha_hit_beta", 1.0))
            except Exception:
                base_beta = 1.0
            base_beta = float(np.clip(base_beta, 0.0, 1.0))
            warm = 1.0
            min_buf = int(getattr(config, "alpha_hit_min_buffer", 0))
            if min_buf > 0:
                warm = min(1.0, float(alpha_hit_buffer_n) / float(min_buf))
                max_loss = float(getattr(config, "alpha_hit_max_loss", 2.0))
                if alpha_hit_loss is None or (not math.isfinite(float(alpha_hit_loss))) or float(alpha_hit_loss) > max_loss:
                    warm = 0.0
            alpha_hit_beta_eff = float(base_beta * warm)

        # Prepare padded inputs for JAX
        num_symbols = len(tasks)
        pad_size = self.JAX_STATIC_BATCH_SIZE
        
        # Original inputs (None 값 방어)
        seeds_orig = np.array([t["seed"] for t in tasks], dtype=np.uint32)
        s0s_orig = np.array(s0s, dtype=np.float32)
        mus_orig = np.array(mus, dtype=np.float32)
        sigmas_orig = np.array(sigmas, dtype=np.float32)
        leverages_orig = np.array([float(t["ctx"].get("leverage") or 1.0) for t in tasks], dtype=np.float32)
        fees_orig = np.array([(float(self.fee_roundtrip_base) + float(self.slippage_perc)) for _ in tasks], dtype=np.float32)

        # Padding
        seeds_jax = np.zeros(pad_size, dtype=np.uint32)
        s0s_jax = np.ones(pad_size, dtype=np.float32) * 100.0
        mus_jax = np.zeros(pad_size, dtype=np.float32)
        sigmas_jax = np.ones(pad_size, dtype=np.float32) * 0.1
        leverages_jax = np.ones(pad_size, dtype=np.float32)
        fees_jax = np.zeros(pad_size, dtype=np.float32)
        
        seeds_jax[:num_symbols] = seeds_orig
        s0s_jax[:num_symbols] = s0s_orig
        mus_jax[:num_symbols] = mus_orig
        sigmas_jax[:num_symbols] = sigmas_orig
        leverages_jax[:num_symbols] = leverages_orig
        fees_jax[:num_symbols] = fees_orig

        t_prep0 = time.perf_counter()

        n_paths = int(n_paths_override) if n_paths_override is not None else int(config.n_paths_live)
        step_sec = int(getattr(self, "time_step_sec", 1) or 1)
        step_sec = int(max(1, step_sec))
        # CRITICAL FIX: dt must account for step_sec (step_sec/SECONDS_PER_YEAR)
        dt = float(step_sec) / float(SECONDS_PER_YEAR)  # step_sec seconds per step, annualized
        max_h_sec = int(max(getattr(self, "horizons", (3600,))))
        max_steps = int(math.ceil(max_h_sec / float(step_sec))) if max_h_sec > 0 else 0
        # Ensure policy_horizons_all is always defined (avoid NameError on batch path)
        policy_horizons_all = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", (60, 300, 600, 1800, 3600)))
        
        # Horizons summary inputs (seconds)
        from engines.mc.constants import HORIZON_SUMMARY_DEFAULT
        h_cols: list[int] = []
        for h in HORIZON_SUMMARY_DEFAULT:
            h_cols.append(min(int(h), int(max_h_sec)))
        # Convert sec horizons to step indices
        h_indices = np.array(
            [min(int(max_steps), int(math.ceil(int(h) / float(step_sec)))) for h in h_cols],
            dtype=np.int32,
        )
        # Exit Policy horizons (shared across batch to keep shapes static)
        policy_horizons_all = list(getattr(self, "POLICY_MULTI_HORIZONS_SEC", h_cols)) or list(h_cols)

        t_prep1 = time.perf_counter()

        price_paths_batch = None
        t_sim0 = time.perf_counter()

        print(f"[BATCH_TIMING] prep={(t_prep1-t_prep0):.3f}s n_paths={n_paths} max_steps={max_steps} step_sec={step_sec} dt={dt:.2e}", flush=True)

        use_torch = bool(getattr(self, "_use_torch", True)) and bool(getattr(jax_backend, "_TORCH_OK", False)) and (not bool(getattr(jax_backend, "DEV_MODE", False)))

        if use_torch:
            print("[BATCH_TIMING] Torch batch path simulation...", flush=True)
            price_paths_batch = self.simulate_paths_price_batch(
                seeds=seeds_jax, s0s=s0s_jax, mus=mus_jax, sigmas=sigmas_jax,
                n_paths=n_paths, n_steps=max_steps, dt=dt, return_torch=True
            )
            t_sim1 = time.perf_counter()
            t_sum0 = time.perf_counter()
            summary_results = summarize_gbm_horizons_multi_symbol_jax(
                price_paths_batch, s0s_jax, leverages_jax, fees_jax * leverages_jax, h_indices, 0.05
            )
            summary_cpu = _summary_to_numpy(summary_results)
            t_sum1 = time.perf_counter()
            print(f"[BATCH_TIMING] torch sim={(t_sim1-t_sim0):.3f}s sum={(t_sum1-t_sum0):.3f}s", flush=True)
        else:
            print("[BATCH_TIMING] NumPy batch fallback...", flush=True)
            from engines.mc.torch_backend import summarize_gbm_horizons_multi_symbol_numpy
            price_paths_batch = self.simulate_paths_price_batch(
                seeds=seeds_jax, s0s=s0s_jax, mus=mus_jax, sigmas=sigmas_jax,
                n_paths=n_paths, n_steps=max_steps, dt=dt, return_torch=False
            )
            t_sim1 = time.perf_counter()
            t_sum0 = time.perf_counter()
            summary_cpu = summarize_gbm_horizons_multi_symbol_numpy(
                np.asarray(price_paths_batch),
                np.asarray(s0s_jax, dtype=np.float64),
                np.asarray(leverages_jax, dtype=np.float64),
                np.asarray(fees_jax * leverages_jax, dtype=np.float64),
                h_indices,
                0.05,
            )
            t_sum1 = time.perf_counter()
            print(f"[BATCH_TIMING] numpy sim={(t_sim1-t_sim0):.3f}s sum={(t_sum1-t_sum0):.3f}s", flush=True)

        # No explicit GPU->CPU transfer stage in this path; keep perf keys stable.
        t_xfer0 = t_sum1
        t_xfer1 = t_sum1

        # Optional control variate adjustment for EV (variance reduction)
        use_cv = str(os.environ.get("MC_USE_CONTROL_VARIATE", "1")).strip().lower() in ("1", "true", "yes", "on")
        if use_cv:
            mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
            for i in range(num_symbols):
                try:
                    prices_np = np.asarray(jax_backend.to_numpy(price_paths_batch[i]), dtype=np.float64)
                    s0_val = float(s0s_jax[i])
                    lev_val = float(leverages_jax[i])
                    fee_roe = float(fees_jax[i]) * lev_val
                    mu_i = float(mus_jax[i])
                    sigma_i = float(sigmas_jax[i])
                    if mode in ("student_t", "bootstrap", "johnson_su"):
                        drift_i = mu_i * dt
                    else:
                        drift_i = (mu_i - 0.5 * sigma_i * sigma_i) * dt
                    diffusion_i = sigma_i * math.sqrt(dt)
                    z_hat = self._infer_standardized_noise(prices_np, drift_i, diffusion_i)
                    control = np.sum(z_hat, axis=1)
                    for k, h_step in enumerate(h_indices):
                        idx = int(min(max_steps, int(h_step)))
                        tp = prices_np[:, idx]
                        gross = (tp - s0_val) / max(s0_val, 1e-12)
                        net_long = gross * lev_val - fee_roe
                        net_short = -gross * lev_val - fee_roe
                        cv_mean_long, _ = self._control_variate_mean(net_long, control)
                        cv_mean_short, _ = self._control_variate_mean(net_short, control)
                        summary_cpu["ev_long"][i, k] = cv_mean_long
                        summary_cpu["ev_short"][i, k] = cv_mean_short
                except Exception:
                    continue
        
        # ✅ GLOBAL BATCH EXIT POLICY (can be skipped with SKIP_EXIT_POLICY=true)
        t_exit_prep0 = time.perf_counter()
        t_exit_prep1 = t_exit_prep0  # default
        t_exit0 = time.perf_counter()
        t_exit1 = t_exit0  # default
        exit_policy_results = None
        tp_targets_by_sym = [None for _ in range(pad_size)]
        sl_targets_by_sym = [None for _ in range(pad_size)]
        
        if SKIP_EXIT_POLICY:
            # Skip Exit Policy - use simplified EV from summary_cpu
            print("[BATCH_TIMING] SKIP_EXIT_POLICY=true, using summary-based EV", flush=True)
            # Debug: Print EV values for first few symbols
            for dbg_i in range(min(3, num_symbols)):
                sym_name = tasks[dbg_i]["ctx"].get("symbol", f"sym{dbg_i}")
                ev_l = summary_cpu["ev_long"][dbg_i]  # shape (n_horizons,)
                ev_s = summary_cpu["ev_short"][dbg_i]
                mu_val = float(mus[dbg_i])
                sigma_val = float(sigmas[dbg_i])
                fee_val = float(fees_jax[dbg_i])
                lev_val = float(leverages_jax[dbg_i])
                print(f"[EV_DEBUG] {sym_name}: mu={mu_val:.4f} sigma={sigma_val:.4f} fee={fee_val:.6f} lev={lev_val:.1f} ev_long={ev_l.max():.6f} ev_short={ev_s.max():.6f}", flush=True)
            # Build simplified results from summary_cpu
            exit_policy_results = []
            h_list = list(h_cols)
            directions = [1, -1]
            for i in range(pad_size):
                # Precompute TP/SL targets (for optional AlphaHit blending)
                if i < num_symbols:
                    sigma_val = float(sigmas_jax[i])
                    leverage = float(leverages_jax[i])
                    tp_targets = []
                    sl_targets = []
                    for h in h_list:
                        for _d in directions:
                            tp_r, sl_r = self.tp_sl_targets_for_horizon(float(h), sigma_val)
                            tp_targets.append(float(tp_r) * leverage)
                            sl_targets.append(float(sl_r) * leverage)
                    tp_targets_by_sym[i] = np.array(tp_targets, dtype=np.float32)
                    sl_targets_by_sym[i] = np.array(sl_targets, dtype=np.float32)
                slot_results = []
                for h_idx, h_val in enumerate(h_list):
                    for d in directions:
                        # Use ev_long or ev_short from summary depending on direction
                        if d == 1:
                            ev_val = float(summary_cpu["ev_long"][i, h_idx]) if i < num_symbols else 0.0
                        else:
                            ev_val = float(summary_cpu["ev_short"][i, h_idx]) if i < num_symbols else 0.0
                        res = {"ev_exit_policy": ev_val}
                        # Approximate TP/SL hit probs using barrier formula (for AlphaHit blend)
                        if i < num_symbols:
                            try:
                                from engines.probability_methods import _prob_max_geq, _prob_min_leq
                                mu_ps = float(mus_jax[i]) / float(SECONDS_PER_YEAR)
                                sigma_ps = float(sigmas_jax[i]) / math.sqrt(float(SECONDS_PER_YEAR))
                                lev = float(leverages_jax[i]) if float(leverages_jax[i]) > 0 else 1.0
                                tp_arr = tp_targets_by_sym[i]
                                sl_arr = sl_targets_by_sym[i]
                                idx = (h_idx * 2) + (0 if d == 1 else 1)
                                tp_under = float(tp_arr[idx]) / lev if tp_arr is not None and idx < len(tp_arr) else 0.0
                                sl_under = float(sl_arr[idx]) / lev if sl_arr is not None and idx < len(sl_arr) else 0.0
                                if d == 1:
                                    p_tp = _prob_max_geq(mu_ps, sigma_ps, float(h_val), float(tp_under))
                                    p_sl = _prob_min_leq(mu_ps, sigma_ps, float(h_val), float(-sl_under))
                                else:
                                    p_tp = _prob_min_leq(mu_ps, sigma_ps, float(h_val), float(-tp_under))
                                    p_sl = _prob_max_geq(mu_ps, sigma_ps, float(h_val), float(sl_under))
                                res["p_tp"] = float(p_tp)
                                res["p_sl"] = float(p_sl)
                            except Exception:
                                pass
                        slot_results.append(res)
                exit_policy_results.append(slot_results)
            t_exit_prep1 = time.perf_counter()
            t_exit1 = t_exit_prep1
        else:
            # Full Exit Policy calculation
            # Prepare arguments for all slots (including padding)
            exit_policy_args = []
            # Fixed horizons and directions for all slots to maximize JIT efficiency
            policy_horizons = list(policy_horizons_all)
            directions = [1, -1] # LONG/SHORT
            batch_h_fixed = []
            batch_d_fixed = []
            for h in policy_horizons:
                for d in directions:
                    batch_h_fixed.append(h)
                    batch_d_fixed.append(d)
            
            batch_h_np = np.array(batch_h_fixed, dtype=np.int32)
            batch_d_np = np.array(batch_d_fixed, dtype=np.int32)
            dd_stop_base = np.array([float(getattr(base_config, "PAPER_EXIT_POLICY_DD_STOP_ROE", -0.02)) for _ in batch_h_fixed], dtype=np.float32)

            for i in range(pad_size):
                if i < num_symbols:
                    t = tasks[i]
                    cost_m = self._get_execution_costs(t["ctx"], t["params"])
                    leverage = float(leverages_jax[i])
                    sigma_val = float(sigmas_jax[i])
                    tp_targets = []
                    sl_targets = []
                    for h in batch_h_fixed:
                        tp_r, sl_r = self.tp_sl_targets_for_horizon(float(h), sigma_val)
                        tp_targets.append(float(tp_r) * leverage)
                        sl_targets.append(float(sl_r) * leverage)
                    tp_targets_arr = np.array(tp_targets, dtype=np.float32)
                    sl_targets_arr = np.array(sl_targets, dtype=np.float32)
                    tp_targets_by_sym[i] = tp_targets_arr
                    sl_targets_by_sym[i] = sl_targets_arr
                    arg = {
                        "symbol": t["ctx"].get("symbol"),
                        "price": s0s_jax[i],
                        "mu": mus_jax[i],
                        "sigma": sigmas_jax[i],
                        "leverage": leverages_jax[i],
                        "fee_roundtrip": cost_m["fee_roundtrip"],
                        "exec_oneway": cost_m["exec_oneway"],
                        "impact_cost": cost_m["impact_cost"],
                        "regime": t["ctx"].get("regime", "chop"),
                        "batch_directions": batch_d_np,
                        "batch_horizons": batch_h_np,
                        "tp_target_roe_batch": tp_targets_arr,
                        "sl_target_roe_batch": sl_targets_arr,
                        "dd_stop_roe_batch": dd_stop_base,
                        "price_paths": price_paths_batch[i]
                    }
                else:
                    # Padding slot
                    tp_targets = []
                    sl_targets = []
                    for h in batch_h_fixed:
                        tp_r, sl_r = self.tp_sl_targets_for_horizon(float(h), None)
                        tp_targets.append(float(tp_r))
                        sl_targets.append(float(sl_r))
                    tp_targets_arr = np.array(tp_targets, dtype=np.float32)
                    sl_targets_arr = np.array(sl_targets, dtype=np.float32)
                    tp_targets_by_sym[i] = tp_targets_arr
                    sl_targets_by_sym[i] = sl_targets_arr
                    arg = {
                        "symbol": "PAD",
                        "price": 100.0,
                        "mu": 0.0,
                        "sigma": 0.1,
                        "leverage": 1.0,
                        "fee_roundtrip": 0.0006,
                        "exec_oneway": 0.0003,
                        "impact_cost": 0.0,
                        "regime": "chop",
                        "batch_directions": batch_d_np,
                        "batch_horizons": batch_h_np,
                        "tp_target_roe_batch": tp_targets_arr,
                        "sl_target_roe_batch": sl_targets_arr,
                        "dd_stop_roe_batch": dd_stop_base,
                        "price_paths": price_paths_batch[i]
                    }
                exit_policy_args.append(arg)

            t_exit_prep1 = time.perf_counter()
                
            t_exit0 = time.perf_counter()
            try:
                exit_policy_results = self.compute_exit_policy_metrics_multi_symbol(exit_policy_args)
            except Exception as e:
                logger.warning(f"⚠️ [EXIT_POLICY_FALLBACK] Exit Policy failed: {e}. Falling back to sequential.")
                exit_policy_results = [self.compute_exit_policy_metrics_batched(**args) for args in exit_policy_args]

            t_exit1 = time.perf_counter()
        
        # 6. Final Assembly (per symbol)
        t_asm0 = time.perf_counter()
        final_outputs = []

        for i in range(num_symbols):
            sym_results = exit_policy_results[i]

            # Aggregate stats from Exit Policy results
            # horizons = [60, 300, 600, 1800, 3600], directions = [1, -1]
            h_list_exit = list(policy_horizons_all)
            n_h = len(h_list_exit)
            ev_long_h = np.zeros(n_h, dtype=np.float64)
            ev_short_h = np.zeros(n_h, dtype=np.float64)
            cvar_long_h = np.zeros(n_h, dtype=np.float64)
            cvar_short_h = np.zeros(n_h, dtype=np.float64)
            ppos_long_h = np.zeros(n_h, dtype=np.float64)
            ppos_short_h = np.zeros(n_h, dtype=np.float64)
            mc_p_tp_long = np.zeros(n_h, dtype=np.float64)
            mc_p_sl_long = np.zeros(n_h, dtype=np.float64)
            mc_p_tp_short = np.zeros(n_h, dtype=np.float64)
            mc_p_sl_short = np.zeros(n_h, dtype=np.float64)

            best_ev_long = -999.0
            best_h_long = h_list_exit[0] if n_h > 0 else 0
            best_ev_short = -999.0
            best_h_short = h_list_exit[0] if n_h > 0 else 0

            # Parse exit policy results into per-horizon vectors
            for j, res in enumerate(sym_results):
                direction = 1 if (j % 2 == 0) else -1
                h_idx = j // 2
                if h_idx >= n_h:
                    continue
                h_val = h_list_exit[h_idx]
                ev_val = float(res.get("ev_exit_policy", 0.0))
                if math.isnan(ev_val):
                    ev_val = 0.0
                cvar_val = res.get("cvar_exit_policy")
                p_pos_val = res.get("p_pos_exit_policy")
                p_tp_val = res.get("p_tp")
                p_sl_val = res.get("p_sl")

                if direction == 1:
                    ev_long_h[h_idx] = ev_val
                    if cvar_val is not None:
                        cvar_long_h[h_idx] = float(cvar_val)
                    if p_pos_val is not None:
                        ppos_long_h[h_idx] = float(p_pos_val)
                    if p_tp_val is not None:
                        mc_p_tp_long[h_idx] = float(p_tp_val)
                    if p_sl_val is not None:
                        mc_p_sl_long[h_idx] = float(p_sl_val)
                    if ev_val > best_ev_long:
                        best_ev_long = ev_val
                        best_h_long = h_val
                else:
                    ev_short_h[h_idx] = ev_val
                    if cvar_val is not None:
                        cvar_short_h[h_idx] = float(cvar_val)
                    if p_pos_val is not None:
                        ppos_short_h[h_idx] = float(p_pos_val)
                    if p_tp_val is not None:
                        mc_p_tp_short[h_idx] = float(p_tp_val)
                    if p_sl_val is not None:
                        mc_p_sl_short[h_idx] = float(p_sl_val)
                    if ev_val > best_ev_short:
                        best_ev_short = ev_val
                        best_h_short = h_val

            # Fallback CVaR from summary (if exit policy didn't populate)
            try:
                if summary_cpu is not None:
                    if not np.any(np.isfinite(cvar_long_h)) or np.all(cvar_long_h == 0.0):
                        cvar_long_h = np.asarray(summary_cpu["cvar_long"][i], dtype=np.float64)
                    if not np.any(np.isfinite(cvar_short_h)) or np.all(cvar_short_h == 0.0):
                        cvar_short_h = np.asarray(summary_cpu["cvar_short"][i], dtype=np.float64)
            except Exception:
                pass

            # Per-horizon TP/SL targets (for AlphaHit blending)
            tp_targets_arr = tp_targets_by_sym[i] if i < len(tp_targets_by_sym) else None
            sl_targets_arr = sl_targets_by_sym[i] if i < len(sl_targets_by_sym) else None

            lev_val = float(leverages_jax[i])
            cost_base = float(fees_jax[i])
            cost_roe = float(cost_base * lev_val)
            cost_entry_roe = float(cost_roe / 2.0)
            rho_val = float(tasks[i]["ctx"].get("rho", config.unified_rho))
            lambda_val = float(tasks[i]["ctx"].get("unified_lambda", config.unified_risk_lambda))

            # AlphaHit blending (batch)
            alpha_pred = alpha_hit_preds[i] if i < len(alpha_hit_preds) else None
            if alpha_pred is not None and tp_targets_arr is not None and sl_targets_arr is not None and alpha_hit_beta_eff > 0.0:
                def _normalize_tp_sl(p_tp: float, p_sl: float) -> tuple[float, float, float]:
                    tp = float(np.clip(float(p_tp), 0.0, 1.0))
                    sl = float(np.clip(float(p_sl), 0.0, 1.0))
                    s = tp + sl
                    if s > 1.0:
                        tp = tp / s
                        sl = sl / s
                        s = 1.0
                    other = float(max(0.0, 1.0 - s))
                    return tp, sl, other

                have_mc_probs = bool(np.any(mc_p_tp_long) or np.any(mc_p_sl_long) or np.any(mc_p_tp_short) or np.any(mc_p_sl_short))
                if have_mc_probs:
                    for h_idx in range(n_h):
                        tp_r = float(tp_targets_arr[2 * h_idx]) if (2 * h_idx) < len(tp_targets_arr) else 0.0
                        sl_r_raw = float(sl_targets_arr[2 * h_idx]) if (2 * h_idx) < len(sl_targets_arr) else 0.0
                        sl_r = -abs(sl_r_raw) if sl_r_raw >= 0 else float(sl_r_raw)

                        p_tp_L_mc = float(mc_p_tp_long[h_idx])
                        p_sl_L_mc = float(mc_p_sl_long[h_idx])
                        p_tp_S_mc = float(mc_p_tp_short[h_idx])
                        p_sl_S_mc = float(mc_p_sl_short[h_idx])

                        tp_pL_pred = float(alpha_pred["p_tp_long"][h_idx]) if h_idx < len(alpha_pred["p_tp_long"]) else 0.0
                        sl_pL_pred = float(alpha_pred["p_sl_long"][h_idx]) if h_idx < len(alpha_pred["p_sl_long"]) else 0.0
                        tp_pS_pred = float(alpha_pred["p_tp_short"][h_idx]) if h_idx < len(alpha_pred["p_tp_short"]) else 0.0
                        sl_pS_pred = float(alpha_pred["p_sl_short"][h_idx]) if h_idx < len(alpha_pred["p_sl_short"]) else 0.0

                        tp_pL_pred, sl_pL_pred, _ = _normalize_tp_sl(tp_pL_pred, sl_pL_pred)
                        tp_pS_pred, sl_pS_pred, _ = _normalize_tp_sl(tp_pS_pred, sl_pS_pred)
                        p_tp_L_mc, p_sl_L_mc, _ = _normalize_tp_sl(p_tp_L_mc, p_sl_L_mc)
                        p_tp_S_mc, p_sl_S_mc, _ = _normalize_tp_sl(p_tp_S_mc, p_sl_S_mc)

                        conf_L = float(self.alpha_hit_confidence(tp_pL_pred, sl_pL_pred))
                        conf_S = float(self.alpha_hit_confidence(tp_pS_pred, sl_pS_pred))
                        beta_eff_L = float(np.clip(alpha_hit_beta_eff * conf_L, 0.0, 1.0))
                        beta_eff_S = float(np.clip(alpha_hit_beta_eff * conf_S, 0.0, 1.0))

                        tp_pL_val = beta_eff_L * tp_pL_pred + (1.0 - beta_eff_L) * p_tp_L_mc
                        sl_pL_val = beta_eff_L * sl_pL_pred + (1.0 - beta_eff_L) * p_sl_L_mc
                        tp_pS_val = beta_eff_S * tp_pS_pred + (1.0 - beta_eff_S) * p_tp_S_mc
                        sl_pS_val = beta_eff_S * sl_pS_pred + (1.0 - beta_eff_S) * p_sl_S_mc

                        tp_pL_val, sl_pL_val, p_other_L = _normalize_tp_sl(tp_pL_val, sl_pL_val)
                        tp_pS_val, sl_pS_val, p_other_S = _normalize_tp_sl(tp_pS_val, sl_pS_val)

                        evL = float(ev_long_h[h_idx])
                        evS = float(ev_short_h[h_idx])
                        denom_L = max(1e-12, p_other_L)
                        denom_S = max(1e-12, p_other_S)
                        other_r_L = (evL + cost_entry_roe - (tp_pL_val * tp_r + sl_pL_val * sl_r)) / denom_L
                        other_r_S = (evS + cost_entry_roe - (tp_pS_val * tp_r + sl_pS_val * sl_r)) / denom_S

                        ev_long_h[h_idx] = tp_pL_val * tp_r + sl_pL_val * sl_r + p_other_L * other_r_L - cost_entry_roe
                        ev_short_h[h_idx] = tp_pS_val * tp_r + sl_pS_val * sl_r + p_other_S * other_r_S - cost_entry_roe
                        ppos_long_h[h_idx] = tp_pL_val
                        ppos_short_h[h_idx] = tp_pS_val

            # Unified score (Psi) from cumulative EV/CVaR vectors
            h_list = list(h_list_exit)
            ev_long_vec = np.asarray(ev_long_h, dtype=np.float64)
            ev_short_vec = np.asarray(ev_short_h, dtype=np.float64)
            cvar_long_vec = np.asarray(cvar_long_h, dtype=np.float64)
            cvar_short_vec = np.asarray(cvar_short_h, dtype=np.float64)
            if "var_long" in summary_cpu:
                var_long_vec = np.asarray(summary_cpu["var_long"][i], dtype=np.float64)
            else:
                var_long_vec = np.zeros_like(ev_long_vec, dtype=np.float64)
            if "var_short" in summary_cpu:
                var_short_vec = np.asarray(summary_cpu["var_short"][i], dtype=np.float64)
            else:
                var_short_vec = np.zeros_like(ev_short_vec, dtype=np.float64)

            # Convert net cumulative vectors to gross (remove one-time cost)
            ev_long_gross = ev_long_vec + cost_roe
            ev_short_gross = ev_short_vec + cost_roe
            cvar_long_gross = cvar_long_vec + cost_roe
            cvar_short_gross = cvar_short_vec + cost_roe

            score_long, t_long = _calc_unified_score_from_cumulative(
                h_list, ev_long_gross, cvar_long_gross,
                cost=cost_roe, rho=rho_val, lambda_param=lambda_val,
            )
            score_short, t_short = _calc_unified_score_from_cumulative(
                h_list, ev_short_gross, cvar_short_gross,
                cost=cost_roe, rho=rho_val, lambda_param=lambda_val,
            )

            if score_long >= score_short:
                direction_best = 1
                policy_score = float(score_long)
                best_h = int(t_long) if t_long > 0 else int(best_h_long)
            else:
                direction_best = -1
                policy_score = float(score_short)
                best_h = int(t_short) if t_short > 0 else int(best_h_short)

            trend_hold_uplift = False
            trend_hold_from_h = int(best_h)
            trend_hold_to_h = int(best_h)
            try:
                trend_hold_on = str(os.environ.get("POLICY_TREND_MIN_HOLD_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                trend_hold_on = True
            regime_tag_batch = str(tasks[i]["ctx"].get("regime") or "").strip().lower()
            is_trend_regime_batch = ("trend" in regime_tag_batch) or ("bull" in regime_tag_batch) or ("bear" in regime_tag_batch)
            if trend_hold_on and is_trend_regime_batch and h_list:
                try:
                    trend_min_hold = int(float(os.environ.get("POLICY_TREND_MIN_HOLD_SEC", 300) or 300))
                except Exception:
                    trend_min_hold = 300
                trend_min_hold = int(max(1, trend_min_hold))
                try:
                    trend_min_dir_conf = float(os.environ.get("POLICY_TREND_MIN_HOLD_DIR_CONF", 0.58) or 0.58)
                except Exception:
                    trend_min_dir_conf = 0.58
                try:
                    trend_max_vpin = float(os.environ.get("POLICY_TREND_MIN_HOLD_MAX_VPIN", 0.80) or 0.80)
                except Exception:
                    trend_max_vpin = 0.80
                dir_conf_batch = float(mu_dir_conf_vals[i]) if i < len(mu_dir_conf_vals) and mu_dir_conf_vals[i] is not None else 0.0
                try:
                    vpin_batch = float(tasks[i]["ctx"].get("vpin") or tasks[i]["ctx"].get("alpha_vpin") or 0.0)
                except Exception:
                    vpin_batch = 0.0
                if float(dir_conf_batch) >= float(trend_min_dir_conf) and float(vpin_batch) <= float(trend_max_vpin):
                    for h_val in sorted(int(h) for h in h_list):
                        if int(h_val) >= int(trend_min_hold):
                            if int(h_val) > int(best_h):
                                best_h = int(h_val)
                                trend_hold_to_h = int(h_val)
                                trend_hold_uplift = True
                            break

            ev_vec = ev_long_vec if direction_best == 1 else ev_short_vec
            cvar_vec = cvar_long_vec if direction_best == 1 else cvar_short_vec
            var_vec = var_long_vec if direction_best == 1 else var_short_vec

            # Use EV at the chosen horizon (or max EV) as the policy EV signal.
            def _ev_at_h(ev_arr: np.ndarray, h_list: list[int], h_target: int, fallback: float) -> float:
                try:
                    if h_target in h_list:
                        return float(ev_arr[h_list.index(h_target)])
                except Exception:
                    pass
                try:
                    # fallback to nearest horizon
                    if h_list:
                        idx = int(np.argmin(np.abs(np.asarray(h_list, dtype=np.float64) - float(h_target))))
                        return float(ev_arr[idx])
                except Exception:
                    pass
                return float(fallback)

            policy_ev_mix_long = float(best_ev_long)
            policy_ev_mix_short = float(best_ev_short)
            if direction_best == 1:
                policy_ev_mix = _ev_at_h(ev_long_vec, h_list, best_h, fallback=policy_ev_mix_long)
            else:
                policy_ev_mix = _ev_at_h(ev_short_vec, h_list, best_h, fallback=policy_ev_mix_short)

            try:
                both_ev_neg_on = str(os.environ.get("ENTRY_BOTH_EV_NEG_FILTER_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                both_ev_neg_on = True
            try:
                both_ev_neg_floor = float(os.environ.get("ENTRY_BOTH_EV_NEG_NET_FLOOR", 0.0) or 0.0)
            except Exception:
                both_ev_neg_floor = 0.0
            both_ev_negative = bool(
                float(policy_ev_mix_long) <= float(both_ev_neg_floor)
                and float(policy_ev_mix_short) <= float(both_ev_neg_floor)
            )

            # Confidence-interval (LCB) gate for entry validity
            ci_enabled = str(os.environ.get("ENTRY_CI_GATE_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
            ci_alpha = float(os.environ.get("ENTRY_CI_ALPHA", "0.05"))
            ci_floor = float(os.environ.get("ENTRY_CI_FLOOR", "0.0"))
            ci_lcb = None
            if ci_enabled and len(h_list) > 0:
                try:
                    idx_best = int(h_list.index(int(best_h))) if int(best_h) in h_list else int(np.argmax(ev_vec))
                    ev_at = float(ev_vec[idx_best])
                    var_at = float(var_vec[idx_best])
                    se = math.sqrt(max(var_at, 0.0) / max(float(n_paths), 1.0))
                    z = float(NormalDist().inv_cdf(1.0 - ci_alpha))
                    ci_lcb = float(ev_at - z * se)
                except Exception:
                    ci_lcb = None

            # [Sizing] Kelly calculation similar to sequential path
            if direction_best == 1:
                win_val = float(summary_cpu["win_long"][i].max())
                try:
                    cvar_val = float(np.asarray(cvar_long_vec, dtype=np.float64).mean())
                except Exception:
                    cvar_val = float(np.asarray(summary_cpu["cvar_long"][i], dtype=np.float64).mean())
            else:
                win_val = float(summary_cpu["win_short"][i].max())
                try:
                    cvar_val = float(np.asarray(cvar_short_vec, dtype=np.float64).mean())
                except Exception:
                    cvar_val = float(np.asarray(summary_cpu["cvar_short"][i], dtype=np.float64).mean())
            sigma_val = float(sigmas[i])

            variance_proxy = float(sigma_val * sigma_val)
            kelly_raw = max(0.0, policy_ev_mix / max(variance_proxy, 1e-6))

            lev_penalty = 1.0
            cvar_penalty = max(0.05, 1.0 - 15.0 * abs(cvar_val) * lev_penalty)
            kelly_cap = 1.0

            kelly = min(kelly_raw * cvar_penalty, kelly_cap)
            size_frac = float(max(0.01, kelly * win_val))

            perf = {
                "prep": float(t_prep1 - t_prep0),
                "sim": float(t_sim1 - t_sim0),
                "sum": float(t_sum1 - t_sum0),
                "xfer": float(t_xfer1 - t_xfer0),
                "exit_prep": float(t_exit_prep1 - t_exit_prep0),
                "exit": float(t_exit1 - t_exit0),
            }

            can_enter_score = bool(policy_ev_mix > 0.0001)
            if both_ev_neg_on and both_ev_negative:
                can_enter_score = False
            can_enter_ci = True
            if ci_enabled:
                can_enter_ci = (ci_lcb is not None) and (float(ci_lcb) >= float(ci_floor))
            can_enter = bool(can_enter_score and can_enter_ci)

            mu_alpha_val = tasks[i]["ctx"].get("mu_alpha")
            if mu_alpha_val is None:
                mu_alpha_val = mu_alpha_vals[i] if i < len(mu_alpha_vals) else mus[i]
            mu_alpha_raw_val = mu_alpha_raw_vals[i] if i < len(mu_alpha_raw_vals) else mu_alpha_val
            mu_dir_prob_long_val = mu_dir_prob_long_vals[i] if i < len(mu_dir_prob_long_vals) else None
            mu_dir_edge_val = mu_dir_edge_vals[i] if i < len(mu_dir_edge_vals) else None
            mu_dir_conf_val = mu_dir_conf_vals[i] if i < len(mu_dir_conf_vals) else None
            out = {
                "ok": True,
                "ev": float(policy_ev_mix),
                "score": float(policy_score),
                "win": float(win_val),
                "cvar": float(cvar_val),
                "can_enter": can_enter,
                "best_h": int(best_h),
                "direction": int(direction_best),
                "kelly": float(kelly),
                "size_frac": float(size_frac),
                "unified_score": float(policy_score),
                "unified_score_long": float(score_long),
                "unified_score_short": float(score_short),
                "unified_t_star": float(best_h),
                "unified_t_star_long": float(t_long),
                "unified_t_star_short": float(t_short),
                "ev_vector": [float(x) for x in ev_vec.tolist()],
                "cvar_vector": [float(x) for x in cvar_vec.tolist()],
                "horizon_seq": [int(h) for h in h_list],
                "mu_alpha": float(mu_alpha_val),
                "mu_alpha_for_ev": float(mu_alpha_val),
                "mu_alpha_raw": float(mu_alpha_raw_val),
                "mu_dir_prob_long": (float(mu_dir_prob_long_val) if mu_dir_prob_long_val is not None else None),
                "mu_dir_edge": (float(mu_dir_edge_val) if mu_dir_edge_val is not None else None),
                "mu_dir_conf": (float(mu_dir_conf_val) if mu_dir_conf_val is not None else None),
                "meta": {
                    "can_enter": can_enter,
                    "ev": float(policy_ev_mix),
                    "ev_raw": float(policy_ev_mix),
                    "win": float(win_val),
                    "cvar": float(cvar_val),
                    "best_h": int(best_h),
                    "direction": int(direction_best),
                    "kelly": float(kelly),
                    "size_frac": float(size_frac),
                    "unified_score": float(policy_score),
                    "unified_score_long": float(score_long),
                    "unified_score_short": float(score_short),
                    "unified_t_star": float(best_h),
                    "unified_t_star_long": float(t_long),
                    "unified_t_star_short": float(t_short),
                    "unified_lambda": float(lambda_val),
                    "unified_rho": float(rho_val),
                    "execution_cost": float(cost_base),
                    "fee_roundtrip_total": float(cost_base),
                    "leverage": float(lev_val),
                    "ev_vector": [float(x) for x in ev_vec.tolist()],
                    "cvar_vector": [float(x) for x in cvar_vec.tolist()],
                    "horizon_seq": [int(h) for h in h_list],
                    "ev_by_horizon_long": [float(x) for x in ev_long_vec.tolist()],
                    "ev_by_horizon_short": [float(x) for x in ev_short_vec.tolist()],
                    "cvar_by_horizon_long": [float(x) for x in cvar_long_vec.tolist()],
                    "cvar_by_horizon_short": [float(x) for x in cvar_short_vec.tolist()],
                    "horizon_seq": [int(h) for h in h_list],
                    "policy_ev_score_long": float(score_long),
                    "policy_ev_score_short": float(score_short),
                    "policy_best_h_long": int(best_h_long),
                    "policy_best_h_short": int(best_h_short),
                    "policy_best_ev_long": float(best_ev_long),
                    "policy_best_ev_short": float(best_ev_short),
                    "policy_horizon_eff_sec": int(best_h),
                    "mu_alpha": float(mu_alpha_val),
                    "mu_alpha_for_ev": float(mu_alpha_val),
                    "mu_alpha_raw": float(mu_alpha_raw_val),
                    "mu_dir_prob_long": (float(mu_dir_prob_long_val) if mu_dir_prob_long_val is not None else None),
                    "mu_dir_edge": (float(mu_dir_edge_val) if mu_dir_edge_val is not None else None),
                    "mu_dir_conf": (float(mu_dir_conf_val) if mu_dir_conf_val is not None else None),
                    "mu_adjusted": float(mus_jax[i]),
                    "sigma_sim": float(sigmas_jax[i]),
                    "sigma_annual": float(sigmas_jax[i]),
                    "policy_ev_mix": float(policy_ev_mix),
                    "policy_ev_mix_long": float(policy_ev_mix_long),
                    "policy_ev_mix_short": float(policy_ev_mix_short),
                    "policy_both_ev_negative": bool(both_ev_negative),
                    "policy_both_ev_neg_floor": float(both_ev_neg_floor),
                    "policy_trend_hold_uplift": bool(trend_hold_uplift),
                    "policy_trend_hold_from_h": int(trend_hold_from_h),
                    "policy_trend_hold_to_h": int(trend_hold_to_h),
                    "policy_signal_strength": float(policy_ev_mix),
                    "ci_enabled": bool(ci_enabled),
                    "ci_alpha": float(ci_alpha),
                    "ci_floor": float(ci_floor),
                    "ci_lcb": float(ci_lcb) if ci_lcb is not None else None,
                    "ci_gate_pass": bool(can_enter_ci),
                    "n_paths": int(n_paths),
                    "perf": perf,
                },
            }
            if alpha_hit_features_list[i] is not None:
                try:
                    out["meta"]["alpha_hit_features"] = [float(x) for x in alpha_hit_features_list[i].tolist()]
                except Exception:
                    out["meta"]["alpha_hit_features"] = None
            final_outputs.append(out)

        t_asm1 = time.perf_counter()

        # Detailed perf logging for batch internals (always emit)
        try:
            print(
                "[BATCH_PIPE] "
                f"prep={(t_prep1-t_prep0):.3f}s "
                f"sim={(t_sim1-t_sim0):.3f}s "
                f"sum={(t_sum1-t_sum0):.3f}s "
                f"xfer={(t_xfer1-t_xfer0):.3f}s "
                f"exit_prep={(t_exit_prep1-t_exit_prep0):.3f}s "
                f"exit={(t_exit1-t_exit0):.3f}s "
                f"asm={(t_asm1-t_asm0):.3f}s "
                f"total={(time.perf_counter()-t_start):.3f}s "
                f"num_symbols={num_symbols} pad={pad_size} n_paths={n_paths} step_sec={step_sec} max_steps={max_steps}"
            )
        except Exception:
            pass

        t_end = time.perf_counter()
        logger.info(f"🚀 [GLOBAL_VMAP] Processed {num_symbols} symbols in {(t_end - t_start)*1000:.1f}ms")

        # ============================================================================
        # MEMORY CLEANUP: 대형 배열 명시적 해제 (RAM 2-3GB 제한 목표)
        # ============================================================================
        try:
            # 1. Exit policy args에서 price_paths 참조 해제
            if exit_policy_results is not None:
                del exit_policy_results
            if 'exit_policy_args' in locals():
                del exit_policy_args
            
            # 2. Summary arrays 해제
            if summary_cpu is not None:
                del summary_cpu
            
            # 3. 대형 텐서 price_paths_batch 해제
            if price_paths_batch is not None:
                del price_paths_batch
            
            # 4. Padded input arrays 해제
            del seeds_jax, s0s_jax, mus_jax, sigmas_jax, leverages_jax, fees_jax
            
            # 5. 강제 가비지 컬렉션
            import gc
            gc.collect()
            
            # 6. PyTorch MPS/CUDA 메모리 캐시 정리 (GPU 메모리 반환)
            if jax_backend._TORCH_OK and jax_backend.torch is not None:
                try:
                    if hasattr(jax_backend.torch, 'mps') and hasattr(jax_backend.torch.mps, 'empty_cache'):
                        jax_backend.torch.mps.empty_cache()
                    elif hasattr(jax_backend.torch, 'cuda') and jax_backend.torch.cuda.is_available():
                        jax_backend.torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception as mem_err:
            logger.warning(f"[MEM_CLEANUP] Warning during cleanup: {mem_err}")

        return final_outputs

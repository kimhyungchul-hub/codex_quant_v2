from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Optional

import numpy as np

from engines.cvar_methods import cvar_ensemble
from engines.exit_policy_methods import simulate_exit_policy_rollforward
from engines.mc.constants import MC_VERBOSE_PRINT, SECONDS_PER_YEAR
from engines.mc.torch_backend import _TORCH_OK, to_numpy
from engines.mc.config import config

logger = logging.getLogger(__name__)

# PyTorch exit-policy backends (JAX replacement)
try:
    from engines.mc.exit_policy_torch import (
        simulate_exit_policy_rollforward_batched_vmap_torch as simulate_exit_policy_rollforward_batched_vmap_jax,
        simulate_exit_policy_multi_symbol_torch as simulate_exit_policy_multi_symbol_jax,
    )
except Exception:  # pragma: no cover
    simulate_exit_policy_rollforward_batched_vmap_jax = None
    simulate_exit_policy_multi_symbol_jax = None



class MonteCarloExitPolicyMixin:
    def compute_exit_policy_metrics(
        self,
        *,
        symbol: str,
        price: float,
        mu: float,
        sigma: float,
        leverage: float,
        direction: int,
        fee_roundtrip: float,
        exec_oneway: float,
        impact_cost: float,
        regime: str,
        horizon_sec: int,
        decision_dt_sec: int = 5,
        seed: int = 0,
        cvar_alpha: float = 0.05,
        price_paths: Optional[np.ndarray] = None,
        decision_meta: Optional[Dict[str, Any]] = None,
        # Explicit target overrides (ROE units)
        tp_target_roe_over: Optional[float] = None,
        sl_target_roe_over: Optional[float] = None,
        # Optional overrides from MCParams
        p_pos_floor_enter: Optional[float] = None,
        p_pos_floor_hold: Optional[float] = None,
        p_sl_enter_ceiling: Optional[float] = None,
        p_sl_hold_ceiling: Optional[float] = None,
        p_tp_floor_enter: Optional[float] = None,
        p_tp_floor_hold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        정책 롤포워드를 단일 horizon에서 돌려 p_pos/ev 기준을 반환한다.
        """
        if MC_VERBOSE_PRINT:
            print(
                f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: decision_meta={decision_meta} keys={list(decision_meta.keys()) if decision_meta else []}"
            )
            logger.info(
                f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: decision_meta={decision_meta} keys={list(decision_meta.keys()) if decision_meta else []}"
            )
        h = int(horizon_sec)
        step_sec = int(getattr(self, "time_step_sec", 1) or 1)
        step_sec = int(max(1, step_sec))
        dt = float(self.dt)
        # decision_dt_sec defaults to 5 seconds per step (used for shift calculation)
        decision_dt_sec = int(getattr(self, "POLICY_DECISION_DT_SEC", 5))
        h_steps = int(max(1, math.ceil(float(h) / float(step_sec))))
        h_pts = int(h_steps + 1)  # t=0 포함

        # ---- NEW: survival-based execution mixing (entry + exit) ----
        entry_mix = None
        exit_mix = None
        sigma_per_sec = float(self._sigma_per_sec(sigma=sigma, dt=dt))

        extra_entry_delay_penalty_r = 0.0
        extra_exit_delay_penalty_r = 0.0
        h_eff = h
        start_shift_steps = 0
        mu_adj = float(mu)

        if decision_meta is not None and ("pmaker_entry" in decision_meta):
            fee_maker = float(decision_meta.get("fee_roundtrip_maker", fee_roundtrip))
            fee_taker = float(decision_meta.get("fee_roundtrip_taker", fee_roundtrip))
            entry_mix = self._execution_mix_from_survival(
                meta=decision_meta,
                fee_maker=fee_maker,
                fee_taker=fee_taker,
                horizon_sec=h,
                sigma_per_sec=sigma_per_sec,
                prefix="pmaker_entry",
                delay_penalty_mult=self.PMAKER_DELAY_PENALTY_MULT,
            )
            # entry는 roundtrip fee의 "진입+청산"이 섞여 들어가면 과중첩될 수 있어.
            # 여기서는 fee_roundtrip를 entry_mix로 덮되, exit는 아래에서 exec_oneway/추가페널티로 반영.
            fee_roundtrip = float(entry_mix["fee_mix"])
            extra_entry_delay_penalty_r = float(entry_mix["delay_penalty_r"])

            # entry delay는 시뮬레이션 시작을 늦추는 방식(가장 현실적)
            # dt is in years (1.0 / SECONDS_PER_YEAR), but we need seconds per step
            # Use decision_dt_sec (default 5 seconds per step) for shift calculation
            if self.PMAKER_ENTRY_DELAY_SHIFT:
                delay_sec_val = float(entry_mix["delay_sec"])
                dt_step_sec = float(step_sec)  # price_paths step size
                if dt_step_sec > 0:
                    start_shift_steps = int(round(delay_sec_val / dt_step_sec))
                    start_shift_steps = max(0, min(start_shift_steps, h_pts - 2))  # Cap at horizon
                    if MC_VERBOSE_PRINT:
                        print(
                            f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: delay_sec={delay_sec_val:.4f} dt_step_sec={dt_step_sec} start_shift_steps={start_shift_steps}"
                        )
                else:
                    start_shift_steps = 0
                if start_shift_steps > 0:
                    # horizon도 그만큼 줄어듦
                    h_eff = max(1, h - int(round(delay_sec_val)))

            # alpha decay: delay 동안 알파가 죽는다 -> mu를 약화
            tau = max(1e-6, float(self.ALPHA_DELAY_DECAY_TAU_SEC))
            decay = math.exp(-float(entry_mix["delay_sec"]) / tau)
            mu_adj = float(mu_adj) * decay

        if decision_meta is not None and ("pmaker_exit" in decision_meta):
            # exit는 one-way 성격이므로 fee/penalty를 별도로 더 보수적으로 반영
            fee_maker_exit = float(decision_meta.get("fee_oneway_maker", exec_oneway))
            fee_taker_exit = float(decision_meta.get("fee_oneway_taker", exec_oneway))
            exit_mix = self._execution_mix_from_survival(
                meta=decision_meta,
                fee_maker=fee_maker_exit,
                fee_taker=fee_taker_exit,
                horizon_sec=h_eff,
                sigma_per_sec=sigma_per_sec,
                prefix="pmaker_exit",
                delay_penalty_mult=self.PMAKER_EXIT_DELAY_PENALTY_MULT,
            )
            # exit one-way 예상비용을 덮어쓰기
            exec_oneway = float(exit_mix["fee_mix"])
            extra_exit_delay_penalty_r = float(exit_mix["delay_penalty_r"])
            # exit delay는 실질적으로 청산이 늦어져 변동성 노출이 늘어난다고 보고 horizon을 추가 감소
            h_eff = max(1, int(round(float(exit_mix["horizon_eff_sec"]))))

        # ✅ [HORIZON_SLICING] 전달된 price_paths를 슬라이싱만 사용 (시뮬레이션 재생성 방지)
        # evaluate_entry_metrics에서 이미 최대 horizon 길이로 생성하여 전달하므로,
        # 여기서는 슬라이싱만 수행하여 Horizon Slicing 기법 적용
        paths = None
        if price_paths is not None:
            try:
                arr = np.asarray(to_numpy(price_paths), dtype=np.float32)
                if arr.ndim != 2:
                    logger.warning(
                        f"[HORIZON_SLICING] {symbol} | price_paths ndim={arr.ndim} (expected 2), ignoring provided paths"
                    )
                elif arr.shape[1] >= h_pts:
                    paths = arr[:, :h_pts]
                else:
                    paths = arr[:, :]
                    logger.warning(
                        f"[HORIZON_SLICING] {symbol} | price_paths shape={arr.shape} < h_pts={h_pts}, using available length"
                    )
            except Exception as e:
                logger.warning(f"[HORIZON_SLICING] {symbol} | invalid price_paths, regenerating: {e}")
                paths = None

        # ✅ [HORIZON_SLICING] price_paths가 전달되지 않았거나 유효하지 않은 경우에만 시뮬레이션 생성
        if paths is None:
            n_paths = int(self.N_PATHS_EXIT_POLICY)
            n_steps = int(h_steps)
            logger.info(
                f"[HORIZON_SLICING] {symbol} | generating new paths for h={h}s (step_sec={step_sec}, n_steps={n_steps})"
            )
            paths = self.simulate_paths_price(
                seed=seed,
                s0=price,
                mu=float(mu_adj),
                sigma=float(sigma),
                n_paths=n_paths,
                n_steps=n_steps,
                dt=float(self.dt),
            )[:, :h_pts]
        else:
            logger.debug(
                f"[HORIZON_SLICING] {symbol} | Using sliced price_paths: original_shape={price_paths.shape}, sliced_shape={paths.shape}, h_pts={h_pts}"
            )

        # ---- apply entry delay as path shift (entry happens later) ----
        pp = paths
        if pp is not None and start_shift_steps > 0:
            # pp shape: (n_paths, n_steps)
            if pp.shape[1] > start_shift_steps + 2:
                pp = pp[:, start_shift_steps:]
            else:
                # 너무 짧으면 그대로 두되, horizon_eff는 최소 1로
                h_eff = max(1, min(h_eff, pp.shape[1] - 1))

        cash_exit_enabled = bool(getattr(config, "policy_cash_exit_enabled", False))
        cash_exit_score = float(getattr(config, "policy_cash_exit_score", 0.0))
        max_hold_sec = int(getattr(config, "policy_max_hold_sec", 0) or 0)
        if max_hold_sec > 0:
            h_eff = int(min(int(h_eff), max_hold_sec))
        max_hold_steps = 0
        if max_hold_sec > 0:
            max_hold_steps = int(math.ceil(float(max_hold_sec) / float(step_sec)))

        h_eff_steps = int(max(1, math.ceil(float(h_eff) / float(step_sec))))
        h_eff_pts = int(h_eff_steps + 1)
        if pp is not None and pp.shape[1] > h_eff_pts:
            pp = pp[:, :h_eff_pts]

        # mu/sigma: annualized -> per-second for probability approximations
        mu_ps = float(mu_adj) / float(SECONDS_PER_YEAR)
        sigma_ps = float(sigma) / math.sqrt(float(SECONDS_PER_YEAR))

        # ---- drawdown stop config (short-term survival) ----
        dd_stop_roe = config.policy_dd_stop_roe

        # ✅ [Refinement] Volatility-Scaled SL
        # Adjust dd_stop_roe based on current sigma. 
        # Ref sigma is 0.4 (40% annual vol).
        # Formula: dd_eff = dd_base * (sigma / 0.4)
        sigma_ref = 0.4
        vol_scale = float(sigma) / sigma_ref
        vol_scale = max(0.5, min(3.0, vol_scale)) # Bound scaling
        dd_stop_roe_eff = dd_stop_roe * vol_scale

        enable_dd_stop = config.policy_enable_dd_stop

        fee_exit_only_override = None
        try:
            if decision_meta is not None and bool(decision_meta.get("fee_roundtrip_is_exit_only")):
                fee_exit_only_override = float(fee_roundtrip)
        except Exception:
            fee_exit_only_override = None

        # Unified score parameters (risk/loss aware)
        meta_src = decision_meta if isinstance(decision_meta, dict) else {}
        nested_meta = meta_src.get("meta") if isinstance(meta_src, dict) else None
        if isinstance(nested_meta, dict):
            meta_src = nested_meta
        try:
            unified_lambda = float(meta_src.get("unified_lambda", config.unified_risk_lambda))
        except Exception:
            unified_lambda = float(config.unified_risk_lambda)
        try:
            unified_rho = float(meta_src.get("unified_rho", config.unified_rho))
        except Exception:
            unified_rho = float(config.unified_rho)

        # Determine TP/SL targets (ROE units)
        # ✅ If explicit overrides are provided, use them. 
        # Otherwise, fallback to TP_R_BY_H and SL_R_FIXED but scale them by leverage
        # (treating the defaults as "price change %" targets).
        tp_r_price, sl_r_price = self.tp_sl_targets_for_horizon(float(h), float(sigma) if sigma is not None else None)
        if tp_target_roe_over is not None:
            tp_target_roe = float(tp_target_roe_over)
        else:
            tp_target_roe = float(tp_r_price) * float(leverage)
            
        if sl_target_roe_over is not None:
            sl_target_roe = float(sl_target_roe_over)
        else:
            sl_target_roe = float(sl_r_price) * float(leverage)

        use_torch = bool(getattr(self, "_use_torch", True)) and _TORCH_OK and simulate_exit_policy_rollforward_batched_vmap_jax is not None

        if use_torch:
            try:
                min_hold_sec_val = int(max(self.MIN_HOLD_SEC_DIRECTIONAL, int(h * config.policy_min_hold_frac)))
                res_batch = simulate_exit_policy_rollforward_batched_vmap_jax(
                    price_paths=np.asarray(pp, dtype=np.float32),
                    s0=float(price),
                    mu_ps=float(mu_ps),
                    sigma_ps=float(sigma_ps),
                    leverage=float(leverage),
                    fee_roundtrip=float(fee_roundtrip),
                    exec_oneway=float(exec_oneway),
                    impact_cost=float(impact_cost),
                    decision_dt_sec=int(decision_dt_sec),
                    step_sec=int(step_sec),
                    max_horizon_sec=int(h_eff),
                    side_now_batch=np.array([int(direction)], dtype=np.int32),
                    horizon_sec_batch=np.array([int(h_eff)], dtype=np.int32),
                    min_hold_sec_batch=np.array([int(min_hold_sec_val)], dtype=np.int32),
                    tp_target_roe_batch=np.array([float(tp_target_roe)], dtype=np.float32),
                    sl_target_roe_batch=np.array([float(sl_target_roe)], dtype=np.float32),
                    p_pos_floor_enter=float(p_pos_floor_enter if p_pos_floor_enter is not None else self.POLICY_P_POS_ENTER_BY_REGIME.get(regime, 0.52)),
                    p_pos_floor_hold=float(p_pos_floor_hold if p_pos_floor_hold is not None else self.POLICY_P_POS_HOLD_BY_REGIME.get(regime, 0.50)),
                    p_sl_enter_ceiling=float(p_sl_enter_ceiling if p_sl_enter_ceiling is not None else self.POLICY_P_SL_ENTER_MAX_BY_REGIME.get(regime, 0.20)),
                    p_sl_hold_ceiling=float(p_sl_hold_ceiling if p_sl_hold_ceiling is not None else self.POLICY_P_SL_HOLD_MAX_BY_REGIME.get(regime, 0.25)),
                    p_sl_emergency=float(self.POLICY_P_SL_EMERGENCY),
                    p_tp_floor_enter=float(p_tp_floor_enter if p_tp_floor_enter is not None else self.POLICY_P_TP_ENTER_MIN_BY_REGIME.get(regime, 0.15)),
                    p_tp_floor_hold=float(p_tp_floor_hold if p_tp_floor_hold is not None else self.POLICY_P_TP_HOLD_MIN_BY_REGIME.get(regime, 0.12)),
                    score_margin=float(self.SCORE_MARGIN_DEFAULT),
                    soft_floor=float(self.POLICY_VALUE_SOFT_FLOOR_AFTER_COST),
                    enable_dd_stop=bool(enable_dd_stop),
                    dd_stop_roe_batch=np.array([float(dd_stop_roe_eff)], dtype=np.float32),
                    flip_confirm_ticks=int(self.FLIP_CONFIRM_TICKS),
                    hold_bad_ticks=int(self.POLICY_HOLD_BAD_TICKS),
                    fee_exit_only_override=float(fee_exit_only_override if fee_exit_only_override is not None else -1.0),
                    cvar_alpha=float(cvar_alpha),
                    unified_lambda=float(unified_lambda),
                    unified_rho=float(unified_rho),
                    cash_exit_score=(cash_exit_score if cash_exit_enabled else None),
                    max_hold_sec=int(max_hold_sec),
                )

                res_batch_cpu = {k: to_numpy(v) for k, v in res_batch.items()}
                res = {
                    "p_pos_exit": float(res_batch_cpu["p_pos_exit"][0]),
                    "ev_exit": float(res_batch_cpu["ev_exit"][0]),
                    "exit_t_mean_sec": float(res_batch_cpu["exit_t_mean_sec"][0]),
                    "net_out": np.asarray(res_batch_cpu.get("net_out", [np.zeros((0,), dtype=np.float64)])[0], dtype=np.float64),
                    "exit_t": np.asarray(res_batch_cpu.get("exit_t", [np.zeros((0,), dtype=np.int64)])[0], dtype=np.int64),
                    "ok": True,
                    "p_tp": float(res_batch_cpu.get("p_tp", [0.0])[0]),
                    "p_sl": float(res_batch_cpu.get("p_sl", [0.0])[0]),
                    "p_other": float(res_batch_cpu.get("p_other", [0.0])[0]),
                }
                reason_idxs = np.asarray(res_batch_cpu["reason_idx"][0], dtype=np.int32)
                reason_map = {
                    0: "time_stop",
                    1: "psl_emergency",
                    2: "score_flip",
                    3: "hold_bad",
                    4: "unrealized_dd",
                    5: "tp_hit",
                    6: "sl_hit",
                    7: "unified_cash",
                    8: "max_hold",
                }
                counts = {}
                for idx in reason_idxs:
                    r = reason_map.get(int(idx), "unknown")
                    counts[r] = counts.get(r, 0) + 1
                res["exit_reason_counts"] = counts
            except Exception as e:
                logger.warning(f"[TORCH_ROLLFORWARD] Failed: {e}, falling back to CPU")
                use_torch = False

        if not use_torch:
            res = simulate_exit_policy_rollforward(
                price_paths=pp,
                s0=float(price),
                mu=mu_ps,  # 초당 단위로 변환된 mu
                sigma=sigma_ps,  # 초당 단위로 변환된 sigma
                leverage=float(leverage),
                fee_roundtrip=float(fee_roundtrip),
                exec_oneway=float(exec_oneway),
                impact_cost=float(impact_cost),
                regime=str(regime),
                decision_dt_sec=int(decision_dt_sec),
                horizon_sec=int(h_eff),
                min_hold_sec=int(max(self.MIN_HOLD_SEC_DIRECTIONAL, int(h * config.policy_min_hold_frac))),
                flip_confirm_ticks=int(self.FLIP_CONFIRM_TICKS),
                hold_bad_ticks=int(self.POLICY_HOLD_BAD_TICKS),
                p_pos_floor_enter=float(p_pos_floor_enter if p_pos_floor_enter is not None else self.POLICY_P_POS_ENTER_BY_REGIME.get(regime, 0.52)),
                p_pos_floor_hold=float(p_pos_floor_hold if p_pos_floor_hold is not None else self.POLICY_P_POS_HOLD_BY_REGIME.get(regime, 0.50)),
                p_sl_enter_ceiling=float(p_sl_enter_ceiling if p_sl_enter_ceiling is not None else self.POLICY_P_SL_ENTER_MAX_BY_REGIME.get(regime, 0.20)),
                p_sl_hold_ceiling=float(p_sl_hold_ceiling if p_sl_hold_ceiling is not None else self.POLICY_P_SL_HOLD_MAX_BY_REGIME.get(regime, 0.25)),
                p_sl_emergency=float(self.POLICY_P_SL_EMERGENCY),
                p_tp_floor_enter=float(p_tp_floor_enter if p_tp_floor_enter is not None else self.POLICY_P_TP_ENTER_MIN_BY_REGIME.get(regime, 0.15)),
                p_tp_floor_hold=float(p_tp_floor_hold if p_tp_floor_hold is not None else self.POLICY_P_TP_HOLD_MIN_BY_REGIME.get(regime, 0.12)),
                score_margin=float(self.SCORE_MARGIN_DEFAULT),
                soft_floor=float(self.POLICY_VALUE_SOFT_FLOOR_AFTER_COST),
                cash_exit_score=(cash_exit_score if cash_exit_enabled else None),
                max_hold_sec=int(max_hold_steps),
                side_now=int(direction),
                enable_dd_stop=bool(enable_dd_stop),
                dd_stop_roe=float(dd_stop_roe_eff),
                fee_exit_only_override=fee_exit_only_override,
                tp_target_roe=float(tp_target_roe),
                sl_target_roe=float(sl_target_roe),
                unified_lambda=float(unified_lambda),
                unified_rho=float(unified_rho),
                cvar_alpha=float(cvar_alpha),
            )

        delay_penalty_r_total = float(extra_entry_delay_penalty_r) + float(extra_exit_delay_penalty_r)

        ev = float(res.get("ev_exit", 0.0))
        # delay penalties: treat as deterministic cost shift on outcomes
        ev_adj = ev - float(delay_penalty_r_total)
        res["ev_exit"] = ev_adj
        if delay_penalty_r_total != 0.0:
            try:
                res["tp_r_actual"] = float(res.get("tp_r_actual", 0.0)) - float(delay_penalty_r_total)
                res["sl_r_actual"] = float(res.get("sl_r_actual", 0.0)) - float(delay_penalty_r_total)
                res["other_r_actual"] = float(res.get("other_r_actual", 0.0)) - float(delay_penalty_r_total)
            except Exception:
                pass

        # strict maker-only: entry fill이 안되면 거래가 성립 안 함.
        if self.PMAKER_STRICT and entry_mix is not None:
            p_fill = float(entry_mix["p_fill"])
            res["ev_exit"] = float(res["ev_exit"]) * p_fill
            res["p_pos_exit"] = float(res.get("p_pos_exit", 0.0)) * p_fill

        net_out_raw = res.get("net_out")
        if net_out_raw is None:
            net_out_raw = np.zeros((0,), dtype=np.float64)
        net_out = np.asarray(net_out_raw, dtype=np.float64)
        if net_out.size and delay_penalty_r_total != 0.0:
            net_out = net_out - float(delay_penalty_r_total)
            res["net_out"] = net_out
        cvar_exit = (
            float(cvar_ensemble(net_out, alpha=float(cvar_alpha))) if net_out.size else 0.0
        )
        var_exit = float(np.var(net_out)) if net_out.size else 0.0

        # ---- drawdown / liquidation proxy metrics (per horizon) ----
        p_liq = 0.0
        dd_min = 0.0
        dd_p05 = 0.0
        dd_p50 = 0.0
        try:
            exit_t_raw = res.get("exit_t")
            if exit_t_raw is not None and pp is not None:
                exit_t = np.asarray(exit_t_raw, dtype=np.int64)
                if pp.ndim == 2 and exit_t.ndim == 1 and pp.shape[0] == exit_t.shape[0] and pp.shape[1] >= 2:
                    n_paths = int(pp.shape[0])
                    logret = np.log(np.maximum(pp / float(price), 1e-12))
                    realized = float(direction) * float(leverage) * logret  # ROE (no fees)
                    dd_min_prefix = np.minimum.accumulate(realized, axis=1)
                    idx = np.clip(exit_t, 0, dd_min_prefix.shape[1] - 1)
                    dd_min_at_exit = dd_min_prefix[np.arange(n_paths), idx]
                    dd_min = float(np.min(dd_min_at_exit)) if n_paths > 0 else 0.0
                    dd_p05 = float(np.quantile(dd_min_at_exit, 0.05)) if n_paths > 0 else 0.0
                    dd_p50 = float(np.quantile(dd_min_at_exit, 0.50)) if n_paths > 0 else 0.0
                    p_liq = float(np.mean(dd_min_at_exit <= -1.0)) if n_paths > 0 else 0.0
        except Exception:
            p_liq = 0.0
            dd_min = 0.0
            dd_p05 = 0.0
            dd_p50 = 0.0

        out_meta = res.get("meta", {}) or {}

        def _opt_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                return float(val)
            except Exception:
                return None

        score_exit_signal = None
        # 1) Current hold time (seconds) from decision meta (if provided)
        current_hold_sec = 0.0
        if decision_meta:
            try:
                current_hold_sec = float(decision_meta.get("hold_sec", 0.0) or 0.0)
            except Exception:
                current_hold_sec = 0.0
        if decision_meta is not None:
            meta_src = decision_meta
            nested = decision_meta.get("meta") if isinstance(decision_meta, dict) else None
            if isinstance(nested, dict):
                meta_src = nested
            score_long = _opt_float(meta_src.get("unified_score_long"))
            if score_long is None:
                score_long = _opt_float(meta_src.get("policy_ev_score_long"))
            score_short = _opt_float(meta_src.get("unified_score_short"))
            if score_short is None:
                score_short = _opt_float(meta_src.get("policy_ev_score_short"))
            if score_long is not None or score_short is not None:
                horizon_scale = math.sqrt(max(1.0, float(horizon_sec)) / 180.0)
                entry_th = max(0.0, float(config.score_entry_threshold)) * horizon_scale
                exit_th_base = max(0.0, float(config.score_exit_threshold))
                # 2) Grace period: ignore score-based exit shortly after entry
                grace_period_sec = max(300.0, float(horizon_sec) * 0.15)
                if current_hold_sec < grace_period_sec:
                    real_exit_th = -999.0
                    if MC_VERBOSE_PRINT:
                        print(
                            f"[EXIT_POLICY] {symbol} | Grace Period Active ({current_hold_sec:.0f}s < {grace_period_sec:.0f}s). Ignoring Score Drop."
                        )
                        logger.info(
                            f"[EXIT_POLICY] {symbol} | Grace Period Active ({current_hold_sec:.0f}s < {grace_period_sec:.0f}s). Ignoring Score Drop."
                        )
                else:
                    real_exit_th = exit_th_base * horizon_scale

                real_exit_th = max(real_exit_th, float(config.score_entry_floor))
                extend_th = max(real_exit_th, entry_th * max(0.0, float(config.score_extend_mult)))
                score_side = score_long if direction > 0 else score_short
                out_meta["score_exit_threshold"] = float(real_exit_th)
                out_meta["score_extend_threshold"] = float(extend_th)
                out_meta["score_side"] = float(score_side) if score_side is not None else None
                if score_side is not None:
                    if score_side <= real_exit_th:
                        score_exit_signal = "EXIT"
                    elif score_side < extend_th:
                        score_exit_signal = "WEAK_HOLD"
                    else:
                        score_exit_signal = "HOLD"
        if entry_mix is not None:
            out_meta.update({
                "pmaker_entry_used": float(entry_mix["p_fill"]),
                "pmaker_entry_delay_used_sec": float(entry_mix["delay_sec"]),
                "pmaker_entry_delay_cond_used_sec": float(entry_mix["delay_cond_sec"]),
                "pmaker_entry_fee_mix_used": float(entry_mix["fee_mix"]),
                "pmaker_entry_delay_penalty_r": float(extra_entry_delay_penalty_r),
                "policy_entry_shift_steps": int(start_shift_steps),
            })
            if MC_VERBOSE_PRINT:
                print(
                    f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: entry_mix added, delay_penalty_r={extra_entry_delay_penalty_r:.6f} shift_steps={start_shift_steps}"
                )
                logger.info(
                    f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: entry_mix added, delay_penalty_r={extra_entry_delay_penalty_r:.6f} shift_steps={start_shift_steps}"
                )
        else:
            if MC_VERBOSE_PRINT:
                print(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: entry_mix is None")
                logger.info(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: entry_mix is None")
        if exit_mix is not None:
            out_meta.update({
                "pmaker_exit_used": float(exit_mix["p_fill"]),
                "pmaker_exit_delay_used_sec": float(exit_mix["delay_sec"]),
                "pmaker_exit_delay_cond_used_sec": float(exit_mix["delay_cond_sec"]),
                "pmaker_exit_fee_mix_used": float(exit_mix["fee_mix"]),
                "pmaker_exit_delay_penalty_r": float(extra_exit_delay_penalty_r),
            })
            if MC_VERBOSE_PRINT:
                print(
                    f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: exit_mix added, delay_penalty_r={extra_exit_delay_penalty_r:.6f}"
                )
                logger.info(
                    f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: exit_mix added, delay_penalty_r={extra_exit_delay_penalty_r:.6f}"
                )
        else:
            if MC_VERBOSE_PRINT:
                print(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: exit_mix is None")
                logger.info(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: exit_mix is None")
        out_meta["policy_horizon_eff_sec"] = int(h_eff)
        out_meta["policy_dd_stop_roe"] = float(dd_stop_roe)
        out_meta["policy_dd_stop_enabled"] = bool(enable_dd_stop)
        out_meta["policy_cash_exit_enabled"] = bool(cash_exit_enabled)
        out_meta["policy_cash_exit_score"] = float(cash_exit_score)
        out_meta["policy_max_hold_sec"] = int(max_hold_sec)
        if MC_VERBOSE_PRINT:
            print(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: out_meta keys={list(out_meta.keys())} horizon_eff={h_eff}")
            logger.info(f"[PMAKER_DEBUG] {symbol} | compute_exit_policy_metrics: out_meta keys={list(out_meta.keys())} horizon_eff={h_eff}")
        res["meta"] = out_meta

        return {
            "ok": True,
            "symbol": symbol,
            "dir": int(direction),
            "horizon_sec": int(horizon_sec),
            "ev_exit_policy": float(res.get("ev_exit", 0.0)),
            "p_pos_exit_policy": float(res.get("p_pos_exit", 0.0)),
            "exit_t_mean_sec": float(res.get("exit_t_mean_sec", 0.0)),
            "exit_t_p50_sec": float(res.get("exit_t_p50_sec", 0.0)),
            "exit_reason_counts": res.get("exit_reason_counts"),
            "cvar_exit_policy": float(cvar_exit),
            "var_exit_policy": float(var_exit),
            "p_liq_exit_policy": float(p_liq),
            "dd_min_exit_policy": float(dd_min),
            "dd_p05_exit_policy": float(dd_p05),
            "dd_p50_exit_policy": float(dd_p50),
            "meta": out_meta,
            "score_exit_signal": score_exit_signal,
            # ✅ TP/SL/Other 직접 집계 결과
            "p_tp": float(res.get("p_tp", 0.0)),
            "p_sl": float(res.get("p_sl", 0.0)),
            "p_other": float(res.get("p_other", 0.0)),
            "tp_r_actual": float(res.get("tp_r_actual", 0.0)),  # 실제 TP 평균 수익률 (net)
            "sl_r_actual": float(res.get("sl_r_actual", 0.0)),  # 실제 SL 평균 수익률 (net)
            "other_r_actual": float(res.get("other_r_actual", 0.0)),  # 실제 other 평균 수익률 (net)
            "prob_sum_check": bool(res.get("prob_sum_check", False)),  # p_tp + p_sl + p_other == 1 검증
        }

    def compute_exit_policy_metrics_batched(
        self,
        *,
        symbol: str,
        price: float,
        mu: float,
        sigma: float,
        leverage: float,
        fee_roundtrip: float,
        exec_oneway: float,
        impact_cost: float,
        regime: str,
        batch_directions: np.ndarray,
        batch_horizons: np.ndarray,
        price_paths: np.ndarray,
        cvar_alpha: float = 0.05,
        tp_target_roe_batch: Optional[np.ndarray] = None,
        sl_target_roe_batch: Optional[np.ndarray] = None,
        enable_dd_stop: bool = True,
        dd_stop_roe_batch: Optional[np.ndarray] = None,
    ) -> list[Dict[str, Any]]:
        """
        Computes exit policy metrics for a batch of (direction, horizon) combinations using Torch vmap.
        """
        cash_exit_enabled = bool(getattr(config, "policy_cash_exit_enabled", False))
        cash_exit_score = float(getattr(config, "policy_cash_exit_score", 0.0))
        max_hold_sec = int(getattr(config, "policy_max_hold_sec", 0) or 0)
        use_torch = bool(getattr(self, "_use_torch", True)) and _TORCH_OK and simulate_exit_policy_rollforward_batched_vmap_jax is not None
        
        if not use_torch:
            # Fallback to sequential CPU processing
            results = []
            for i in range(len(batch_directions)):
                res = self.compute_exit_policy_metrics(
                    symbol=symbol,
                    price=price,
                    mu=mu,
                    sigma=sigma,
                    leverage=leverage,
                    direction=int(batch_directions[i]),
                    fee_roundtrip=fee_roundtrip,
                    exec_oneway=exec_oneway,
                    impact_cost=impact_cost,
                    regime=regime,
                    horizon_sec=int(batch_horizons[i]),
                    cvar_alpha=cvar_alpha,
                    tp_target_roe_over=float(tp_target_roe_batch[i]) if tp_target_roe_batch is not None else None,
                    sl_target_roe_over=float(sl_target_roe_batch[i]) if sl_target_roe_batch is not None else None,
                    price_paths=np.asarray(price_paths) if price_paths is not None else None,
                )
                results.append(res)
            return results

        # Torch Batched execution
        try:
            dt = float(self.dt)
            mu_ps = mu / SECONDS_PER_YEAR
            sigma_ps = sigma / math.sqrt(SECONDS_PER_YEAR)
            decision_dt_sec = int(getattr(self, "POLICY_DECISION_DT_SEC", 5))
            step_sec = int(getattr(self, "time_step_sec", 1) or 1)
            step_sec = int(max(1, step_sec))
            
            # Prepare batched inputs
            # batch_horizons are seconds
            horizon_sec_batch = batch_horizons.astype(np.int32)
            if max_hold_sec > 0:
                horizon_sec_batch = np.minimum(horizon_sec_batch, int(max_hold_sec)).astype(np.int32)
            max_horizon_sec = int(np.max(horizon_sec_batch))
            
            min_hold_sec_batch = np.array([
                max(self.MIN_HOLD_SEC_DIRECTIONAL, int(h * config.policy_min_hold_frac))
                for h in batch_horizons
            ])

            res_jax_batch = simulate_exit_policy_rollforward_batched_vmap_jax(
                price_paths=price_paths,
                s0=float(price),
                mu_ps=float(mu_ps),
                sigma_ps=float(sigma_ps),
                leverage=float(leverage),
                fee_roundtrip=float(fee_roundtrip),
                exec_oneway=float(exec_oneway),
                impact_cost=float(impact_cost),
                decision_dt_sec=int(decision_dt_sec),
                step_sec=int(step_sec),
                max_horizon_sec=int(max_horizon_sec),
                side_now_batch=np.array(batch_directions, dtype=np.int32),
                horizon_sec_batch=np.array(horizon_sec_batch, dtype=np.int32),
                min_hold_sec_batch=np.array(min_hold_sec_batch, dtype=np.int32),
                tp_target_roe_batch=np.array(tp_target_roe_batch, dtype=np.float32),
                sl_target_roe_batch=np.array(sl_target_roe_batch, dtype=np.float32),
                p_pos_floor_enter=float(self.POLICY_P_POS_ENTER_BY_REGIME.get(regime, 0.52)),
                p_pos_floor_hold=float(self.POLICY_P_POS_HOLD_BY_REGIME.get(regime, 0.50)),
                p_sl_enter_ceiling=float(self.POLICY_P_SL_ENTER_MAX_BY_REGIME.get(regime, 0.20)),
                p_sl_hold_ceiling=float(self.POLICY_P_SL_HOLD_MAX_BY_REGIME.get(regime, 0.25)),
                p_sl_emergency=float(self.POLICY_P_SL_EMERGENCY),
                p_tp_floor_enter=float(self.POLICY_P_TP_ENTER_MIN_BY_REGIME.get(regime, 0.15)),
                p_tp_floor_hold=float(self.POLICY_P_TP_HOLD_MIN_BY_REGIME.get(regime, 0.12)),
                score_margin=float(self.SCORE_MARGIN_DEFAULT),
                soft_floor=float(self.POLICY_VALUE_SOFT_FLOOR_AFTER_COST),
                enable_dd_stop=bool(enable_dd_stop),
                dd_stop_roe_batch=np.array(dd_stop_roe_batch, dtype=np.float32),
                flip_confirm_ticks=int(self.FLIP_CONFIRM_TICKS),
                hold_bad_ticks=int(self.POLICY_HOLD_BAD_TICKS),
                cvar_alpha=float(cvar_alpha),
                unified_lambda=float(config.unified_risk_lambda),
                unified_rho=float(config.unified_rho),
                cash_exit_score=(cash_exit_score if cash_exit_enabled else None),
                max_hold_sec=int(max_hold_sec),
            )

            # Unpack results
            results = []
            reason_map = {
                0: "time_stop",
                1: "psl_emergency",
                2: "score_flip",
                3: "hold_bad",
                4: "unrealized_dd",
                5: "tp_hit",
                6: "sl_hit",
                7: "unified_cash",
                8: "max_hold",
            }

            # Convert torch -> numpy once
            res_jax_batch_cpu = {k: to_numpy(v) for k, v in res_jax_batch.items()}

            p_pos_exit_arr = res_jax_batch_cpu["p_pos_exit"]
            ev_exit_arr = res_jax_batch_cpu["ev_exit"]
            exit_t_mean_arr = res_jax_batch_cpu["exit_t_mean_sec"]
            reason_idx_arr = res_jax_batch_cpu["reason_idx"]
            cvar_exit_arr = res_jax_batch_cpu["cvar_exit"]
            p_tp_arr = res_jax_batch_cpu.get("p_tp")
            p_sl_arr = res_jax_batch_cpu.get("p_sl")
            p_other_arr = res_jax_batch_cpu.get("p_other")

            for i in range(len(batch_directions)):
                res = {
                    "ok": True,
                    "p_pos_exit_policy": float(p_pos_exit_arr[i]),
                    "ev_exit_policy": float(ev_exit_arr[i]),
                    "exit_t_mean_sec": float(exit_t_mean_arr[i]),
                    "cvar_exit_policy": float(cvar_exit_arr[i]),
                    "exit_reason_counts": {},
                    "meta": {"policy_horizon_eff_sec": int(batch_horizons[i])}
                }
                if p_tp_arr is not None:
                    res["p_tp"] = float(p_tp_arr[i])
                if p_sl_arr is not None:
                    res["p_sl"] = float(p_sl_arr[i])
                if p_other_arr is not None:
                    res["p_other"] = float(p_other_arr[i])
                
                # Simple reason count estimate
                counts = {}
                for ridx in reason_idx_arr[i]:
                    rnm = reason_map.get(int(ridx), "unknown")
                    counts[rnm] = counts.get(rnm, 0) + 1
                res["exit_reason_counts"] = counts
                
                results.append(res)
                
            return results

        except Exception as e:
            logger.error(f"[TORCH_BATCH_ROLLFORWARD] Failed: {e}. Falling back to sequential.")
            # Fallback
            results = []
            for i in range(len(batch_directions)):
                res = self.compute_exit_policy_metrics(
                    symbol=symbol,
                    price=price,
                    mu=mu,
                    sigma=sigma,
                    leverage=leverage,
                    direction=int(batch_directions[i]),
                    fee_roundtrip=fee_roundtrip,
                    exec_oneway=exec_oneway,
                    impact_cost=impact_cost,
                    regime=regime,
                    horizon_sec=int(batch_horizons[i]),
                    cvar_alpha=cvar_alpha,
                    tp_target_roe_over=float(tp_target_roe_batch[i]) if tp_target_roe_batch is not None else None,
                    sl_target_roe_over=float(sl_target_roe_batch[i]) if sl_target_roe_batch is not None else None,
                    price_paths=np.asarray(price_paths) if price_paths is not None else None,
                )
                results.append(res)
            return results




    @staticmethod
    def _compress_reason_counts(counts: Any, *, top_k: int = 3) -> Dict[str, int]:
        """
        exit_reason_counts는 사유 문자열이 길어질 수 있으니, 상위 K개만 남기고 나머지는 _other로 합친다.
        """
        if not isinstance(counts, dict) or not counts:
            return {}
        items = []
        other = 0
        for k, v in counts.items():
            try:
                key = str(k)
                val = int(v)
            except Exception:
                continue
            if val <= 0:
                continue
            items.append((key, val))
        if not items:
            return {}
        items.sort(key=lambda x: x[1], reverse=True)
        top = items[: max(0, int(top_k))]
        if len(items) > len(top):
            other = sum(v for _, v in items[len(top) :])
        out = {k: int(v) for k, v in top}
        if other > 0:
            out["_other"] = int(other)
        return out
    def compute_exit_policy_metrics_multi_symbol(self, symbols_args: List[Dict]) -> List[List[Dict]]:
        """
        GLOBAL BATCHING: 여러 심볼에 대해 병렬로 Exit Policy Metrics를 계산한다.
        - symbols_args: 각 심볼별 compute_exit_policy_metrics_batched에 필요한 인자 리스트
        - 반환: 각 심볼별 결과 리스트의 리스트
        """
        global simulate_exit_policy_multi_symbol_jax
        if not _TORCH_OK or simulate_exit_policy_multi_symbol_jax is None:
            # Fallback (sequential)
            return [self.compute_exit_policy_metrics_batched(**args) for args in symbols_args]

        try:
            num_symbols = len(symbols_args)
            if num_symbols == 0:
                return []

            dt = float(self.dt)
            decision_dt_sec = int(getattr(self, "POLICY_DECISION_DT_SEC", 5))
            step_sec = int(getattr(self, "time_step_sec", 1) or 1)
            step_sec = int(max(1, step_sec))
            cash_exit_enabled = bool(getattr(config, "policy_cash_exit_enabled", False))
            cash_exit_score = float(getattr(config, "policy_cash_exit_score", 0.0))
            max_hold_sec = int(getattr(config, "policy_max_hold_sec", 0) or 0)
            
            # Prepare flattened batches
            batch_sizes = [len(args["batch_directions"]) for args in symbols_args]
            max_batch_size = max(batch_sizes)
            
            # Stack symbol-level scalar inputs
            s0_v = np.array([float(a["price"]) for a in symbols_args], dtype=np.float32)
            mu_ps_v = (np.array([float(a["mu"]) for a in symbols_args], dtype=np.float32) / SECONDS_PER_YEAR).astype(np.float32)
            sigma_ps_v = (np.array([float(a["sigma"]) for a in symbols_args], dtype=np.float32) / math.sqrt(SECONDS_PER_YEAR)).astype(np.float32)
            leverage_v = np.array([float(a["leverage"]) for a in symbols_args], dtype=np.float32)
            fee_rt_v = np.array([float(a["fee_roundtrip"]) for a in symbols_args], dtype=np.float32)
            exec_ow_v = np.array([float(a["exec_oneway"]) for a in symbols_args], dtype=np.float32)
            impact_v = np.array([float(a["impact_cost"]) for a in symbols_args], dtype=np.float32)
            
            # Policy thresholds
            p_pos_floor_enter_v = np.array([float(self.POLICY_P_POS_ENTER_BY_REGIME.get(a["regime"], 0.52)) for a in symbols_args], dtype=np.float32)
            p_pos_floor_hold_v = np.array([float(self.POLICY_P_POS_HOLD_BY_REGIME.get(a["regime"], 0.50)) for a in symbols_args], dtype=np.float32)
            p_sl_enter_ceiling_v = np.array([float(self.POLICY_P_SL_ENTER_MAX_BY_REGIME.get(a["regime"], 0.20)) for a in symbols_args], dtype=np.float32)
            p_sl_hold_ceiling_v = np.array([float(self.POLICY_P_SL_HOLD_MAX_BY_REGIME.get(a["regime"], 0.25)) for a in symbols_args], dtype=np.float32)
            p_sl_emergency_v = np.array([float(self.POLICY_P_SL_EMERGENCY) for a in symbols_args], dtype=np.float32)
            p_tp_floor_enter_v = np.array([float(self.POLICY_P_TP_ENTER_MIN_BY_REGIME.get(a["regime"], 0.15)) for a in symbols_args], dtype=np.float32)
            p_tp_floor_hold_v = np.array([float(self.POLICY_P_TP_HOLD_MIN_BY_REGIME.get(a["regime"], 0.12)) for a in symbols_args], dtype=np.float32)
            score_margin_v = np.array([float(self.SCORE_MARGIN_DEFAULT) for _ in symbols_args], dtype=np.float32)
            soft_floor_v = np.array([float(self.POLICY_VALUE_SOFT_FLOOR_AFTER_COST) for _ in symbols_args], dtype=np.float32)
            
            # Pad batched arguments to same size (max_batch_size) for vmap
            side_now_b = np.zeros((num_symbols, max_batch_size), dtype=np.int32)
            horizon_sec_b = np.zeros((num_symbols, max_batch_size), dtype=np.int32)
            min_hold_b = np.zeros((num_symbols, max_batch_size), dtype=np.int32)
            tp_target_b = np.zeros((num_symbols, max_batch_size), dtype=np.float32)
            sl_target_b = np.zeros((num_symbols, max_batch_size), dtype=np.float32)
            dd_stop_b = np.zeros((num_symbols, max_batch_size), dtype=np.float32)
            
            max_horizon_pts_overall = 0
            
            for i, args in enumerate(symbols_args):
                bs = batch_sizes[i]
                side_now_b[i, :bs] = args["batch_directions"]
                h_sec = args["batch_horizons"]
                horizon_pts = h_sec.astype(np.int32)
                if max_hold_sec > 0:
                    horizon_pts = np.minimum(horizon_pts, int(max_hold_sec)).astype(np.int32)
                horizon_sec_b[i, :bs] = horizon_pts
                max_horizon_pts_overall = max(max_horizon_pts_overall, int(np.max(horizon_pts)))
                
                min_hold_b[i, :bs] = [
                    max(self.MIN_HOLD_SEC_DIRECTIONAL, int(h * config.policy_min_hold_frac))
                    for h in h_sec
                ]
                tp_target_b[i, :bs] = args["tp_target_roe_batch"]
                sl_target_b[i, :bs] = args["sl_target_roe_batch"]
                dd_stop_b[i, :bs] = args["dd_stop_roe_batch"]

            # Paths batching (price_paths indexed by steps, not seconds)
            # symbols_args["price_paths"] are already DeviceArrays
            # Need to ensure they have the same n_steps
            n_paths_v = symbols_args[0]["price_paths"].shape[0]
            max_horizon_steps_overall = int(math.ceil(float(max_horizon_pts_overall) / float(step_sec)))
            price_paths_b = np.zeros((num_symbols, n_paths_v, max_horizon_steps_overall + 1), dtype=np.float32)
            for i, args in enumerate(symbols_args):
                p_paths_np = to_numpy(args["price_paths"])
                cur_steps = p_paths_np.shape[1]
                steps_to_copy = min(cur_steps, max_horizon_steps_overall + 1)
                # Pad/slice price_paths to global max_horizon_steps
                price_paths_b[i, :, :steps_to_copy] = np.asarray(p_paths_np[:, :steps_to_copy], dtype=np.float32)

            # Torch multi-symbol kernel call
            decision_dt_sec_v = np.full(num_symbols, decision_dt_sec, dtype=np.int64)
            step_sec_v = np.full(num_symbols, step_sec, dtype=np.int64)
            max_horizon_sec_v = np.full(num_symbols, max_horizon_pts_overall, dtype=np.int64)

            res_jax_multi = simulate_exit_policy_multi_symbol_jax(
                price_paths_b,
                s0_v, mu_ps_v, sigma_ps_v, leverage_v, fee_rt_v, exec_ow_v, impact_v,
                decision_dt_sec_v, step_sec_v, max_horizon_sec_v,
                side_now_b, horizon_sec_b, min_hold_b, tp_target_b, sl_target_b,
                p_pos_floor_enter_v, p_pos_floor_hold_v,
                p_sl_enter_ceiling_v, p_sl_hold_ceiling_v, p_sl_emergency_v,
                p_tp_floor_enter_v, p_tp_floor_hold_v,
                score_margin_v, soft_floor_v,
                True, dd_stop_b,
                int(self.FLIP_CONFIRM_TICKS), int(self.POLICY_HOLD_BAD_TICKS),
                -1.0, float(symbols_args[0].get("cvar_alpha", 0.05)),
                float(config.unified_risk_lambda), float(config.unified_rho),
                cash_exit_score=cash_exit_score if cash_exit_enabled else None,
                max_hold_sec=int(max_hold_sec),
            )
            
            # Transfer all results in one go
            res_cpu_multi = {k: to_numpy(v) for k, v in res_jax_multi.items()}
            
            final_results = []
            reason_map = {
                0: "time_stop",
                1: "psl_emergency",
                2: "score_flip",
                3: "hold_bad",
                4: "unrealized_dd",
                5: "tp_hit",
                6: "sl_hit",
                7: "unified_cash",
                8: "max_hold",
            }
            
            for i in range(num_symbols):
                bs = batch_sizes[i]
                sym_res = []
                for j in range(bs):
                    ridx_list = res_cpu_multi["reason_idx"][i, j]
                    counts = {}
                    for ridx in ridx_list:
                        rnm = reason_map.get(int(ridx), "unknown")
                        counts[rnm] = counts.get(rnm, 0) + 1
                        
                    res = {
                        "ok": True,
                        "p_pos_exit_policy": float(res_cpu_multi["p_pos_exit"][i, j]),
                        "ev_exit_policy": float(res_cpu_multi["ev_exit"][i, j]),
                        "exit_t_mean_sec": float(res_cpu_multi["exit_t_mean_sec"][i, j]),
                        "cvar_exit_policy": float(res_cpu_multi["cvar_exit"][i, j]),
                        "exit_reason_counts": counts,
                        "meta": {"policy_horizon_eff_sec": int(symbols_args[i]["batch_horizons"][j])}
                    }
                    if "p_tp" in res_cpu_multi:
                        res["p_tp"] = float(res_cpu_multi["p_tp"][i, j])
                    if "p_sl" in res_cpu_multi:
                        res["p_sl"] = float(res_cpu_multi["p_sl"][i, j])
                    if "p_other" in res_cpu_multi:
                        res["p_other"] = float(res_cpu_multi["p_other"][i, j])
                    sym_res.append(res)
                final_results.append(sym_res)
            return final_results
            
        except Exception as e:
            logger.error(f"[TORCH_MULTI_ERR] {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                # Disable multi-symbol torch path for the rest of the run to avoid repeated failures.
                simulate_exit_policy_multi_symbol_jax = None
            except Exception:
                pass
            # Gracefully fall back to per-symbol evaluation so the worker still responds
            return [self.compute_exit_policy_metrics_batched(**args) for args in symbols_args]
            return [self.compute_exit_policy_metrics_batched(**args) for args in symbols_args]

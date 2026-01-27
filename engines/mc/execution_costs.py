from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from engines.mc.constants import DEFAULT_IMPACT_CONSTANT
from engines.mc.jax_backend import ensure_jax, jax, jnp, _JAX_OK


class ExecutionCostModel:
    """
    동적 시장 충격(Dynamic Market Impact) 모델 기반 실행 비용 계산.

    Square-Root Market Impact Formula 적용:
    Impact Cost (%) = σ * (Order Size / Daily Volume)^0.5 * Constant

    JAX 호환 벡터화 연산 지원.
    """

    def __init__(self, constant: Optional[float] = None):
        """
        Args:
            constant: 자산별 유동성 계수 (None이면 DEFAULT_IMPACT_CONSTANT 사용)
        """
        self.constant = constant if constant is not None else DEFAULT_IMPACT_CONSTANT

    def calculate_cost(
        self,
        order_size: float,
        price: float,
        sigma: float,
        adv: Optional[float] = None,
        base_spread: float = 0.001  # 0.1% 기본 스프레드
    ) -> float:
        """
        주문 실행 비용 계산 (시장 충격 기반).

        Args:
            order_size: 주문 수량 (shares/contracts)
            price: 현재 가격
            sigma: 변동성 (annualized, 예: 0.2 for 20%)
            adv: 일평균 거래량 (없으면 None, fallback 사용)
            base_spread: 기본 스프레드 (%) for fallback

        Returns:
            실행 비용 (금액 단위, impact_pct * price * order_size)
        """
        ensure_jax()
        if not _JAX_OK:
            # Fallback to NumPy if JAX unavailable
            return self._calculate_cost_numpy(order_size, price, sigma, adv, base_spread)

        # JAX 벡터화 연산
        order_size_j = jnp.array(order_size, dtype=jnp.float32)
        price_j = jnp.array(price, dtype=jnp.float32)
        sigma_j = jnp.array(sigma, dtype=jnp.float32)

        if adv is not None and adv > 0:
            # Square-Root Market Impact
            adv_j = jnp.array(adv, dtype=jnp.float32)
            ratio = order_size_j / adv_j
            impact_pct = sigma_j * jnp.sqrt(jnp.maximum(ratio, 1e-12)) * self.constant
        else:
            # Fallback: Tiered Spread (order_size 기반 비용 증가)
            # Thresholds: order_size 단위에 따라 spread 배율 증가
            threshold1 = 1000.0  # 작은 주문
            threshold2 = 10000.0  # 중간 주문
            spread_mult = jnp.where(
                order_size_j < threshold1,
                1.0,  # 기본 spread
                jnp.where(
                    order_size_j < threshold2,
                    2.0,  # 2배
                    3.0   # 3배
                )
            )
            impact_pct = base_spread * spread_mult

        # 비용 계산: impact_pct * price * order_size
        cost = impact_pct * price_j * order_size_j
        return float(cost)

    def _calculate_cost_numpy(
        self,
        order_size: float,
        price: float,
        sigma: float,
        adv: Optional[float],
        base_spread: float
    ) -> float:
        """NumPy fallback for calculate_cost (JAX unavailable 시)."""
        if adv is not None and adv > 0:
            ratio = order_size / adv
            impact_pct = sigma * math.sqrt(max(ratio, 1e-12)) * self.constant
        else:
            threshold1 = 1000.0
            threshold2 = 10000.0
            if order_size < threshold1:
                spread_mult = 1.0
            elif order_size < threshold2:
                spread_mult = 2.0
            else:
                spread_mult = 3.0
            impact_pct = base_spread * spread_mult

        cost = impact_pct * price * order_size
        return cost


# Legacy compatibility: 기존 MonteCarloExecutionCostsMixin 유지 (deprecated)
class MonteCarloExecutionCostsMixin:
    def _estimate_slippage(self, leverage: float, sigma: float, liq_score: float, ofi_z_abs: float = 0.0) -> float:
        base = self.slippage_perc
        vol_term = 1.0 + float(sigma) * 0.5
        liq_term = 1.0 if liq_score <= 0 else min(2.0, 1.0 + 1.0 / max(liq_score, 1.0))
        lev_term = 1.0 + 0.1 * math.log(1.0 + abs(leverage))
        adv_k = 1.0 + 0.6 * min(2.0, max(0.0, ofi_z_abs))
        slip = base * vol_term * liq_term * lev_term * adv_k
        slip_mult = float(os.environ.get("SLIPPAGE_MULT", "0.3"))
        slip_cap = float(os.environ.get("SLIPPAGE_CAP", "0.0003"))
        slip = max(0.0, float(slip) * slip_mult)
        if slip_cap > 0:
            slip = min(slip, slip_cap)
        return slip

    def _estimate_p_maker(self, *, spread_pct: float, liq_score: float, ofi_z_abs: float) -> float:
        """
        post-only maker 시도(짧은 timeout)에서 maker fill 성공 확률(0~1) 근사.
        - 기본: 유동성↑, 스프레드↓, OFI extreme↓일수록 maker 성공 확률↑
        - 너무 과도한 낙관을 막기 위해 [0.05, 0.95]로 클립
        """
        fixed = os.environ.get("P_MAKER_FIXED")
        if fixed is not None and str(fixed).strip() != "":
            try:
                return float(np.clip(float(fixed), 0.0, 1.0))
            except Exception:
                pass

        sp = float(max(0.0, spread_pct))
        liq = float(max(1.0, liq_score))
        ofi = float(max(0.0, ofi_z_abs))

        liq_term = math.log(liq)
        x = 0.35 + 0.12 * liq_term - 900.0 * sp - 0.25 * ofi
        if x >= 0:
            p = 1.0 / (1.0 + math.exp(-x))
        else:
            ex = math.exp(x)
            p = ex / (1.0 + ex)
        return float(np.clip(p, 0.05, 0.95))

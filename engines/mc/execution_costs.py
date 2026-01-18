from __future__ import annotations

import math
import os

import numpy as np


class MonteCarloExecutionCostsMixin:
    def _estimate_slippage(self, leverage: float, sigma: float, liq_score: float, ofi_z_abs: float = 0.0) -> float:
        base = self.slippage_perc
        vol_term = 1.0 + float(sigma) * 0.5
        liq_term = 1.0 if liq_score <= 0 else min(2.0, 1.0 + 1.0 / max(liq_score, 1.0))
        # ✅ 로그 스케일 적용: 레버리지 영향력을 대폭 줄임
        # 기존: lev_term = max(1.0, abs(leverage) / 5.0) (20배 레버리지 시 4배 증가)
        # 수정: 로그 스케일로 변경 (10배 레버리지 시 약 1.2배 정도만 증가)
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

        # liq_score는 scale이 크므로 log로 완만하게
        liq_term = math.log(liq)  # 0~(대략)
        # simple logistic
        x = 0.35 + 0.12 * liq_term - 900.0 * sp - 0.25 * ofi
        # numerical-stable sigmoid
        if x >= 0:
            p = 1.0 / (1.0 + math.exp(-x))
        else:
            ex = math.exp(x)
            p = ex / (1.0 + ex)
        return float(np.clip(p, 0.05, 0.95))

"""Market regime detection (online GMM/HMM-style) for fast inference.

기존 시간대 기반 보정 로직을 유지하면서, 실시간 로그수익률에 대해
JAX 가속이 가능한 경량 Online GMM 업데이트를 추가한다. 결과는
Monte Carlo 초기 파라미터(mu, sigma) 조정에 사용 가능하다.
"""

import datetime
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from engines.mc.jax_backend import ensure_jax, jax, jnp, _JAX_OK

    ensure_jax()
    JAX_AVAILABLE = bool(_JAX_OK and jax is not None)
except Exception:  # pragma: no cover - 안전 가드
    jax = None
    jnp = None
    JAX_AVAILABLE = False


EPS = 1e-8
TWO_PI = 2.0 * np.pi


class MarketRegimeDetector:
    """Online Gaussian mixture 기반의 국면 분류기.

    - 입력: 최근 가격 또는 로그수익률 시퀀스
    - 출력: 현재 국면 ID, 확률, 국면별 추정 sigma
    - 기본 3상태: bull / chop / bear (mu 크기 순으로 매핑)
    - 2상태 선택 시: calm / volatile (분산 순으로 매핑)

    JAX 사용 가능 시 jnp 백엔드를, 그렇지 않으면 numpy를 자동 선택한다.
    온라인 EM 업데이트(지수 가중)를 사용해 빠른 추론을 보장한다.
    """

    def __init__(
        self,
        n_states: int = 3,
        window: int = 256,
        alpha: float = 0.05,
        min_var: float = 1e-6,
        initial_sigma: float = 0.02,
        use_jax: bool = True,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be >= 2")

        self.n_states = n_states
        self.window = window
        self.alpha = alpha
        self.min_var = min_var
        self.use_jax = bool(use_jax and JAX_AVAILABLE)
        self.xp = jnp if self.use_jax else np
        self.initialized = False

        init_var = float(max(initial_sigma**2, min_var))
        self.pi = self._as_xp(np.full(n_states, 1.0 / n_states))
        self.mu = self._as_xp(np.zeros(n_states))
        self.var = self._as_xp(np.full(n_states, init_var))

    # ---------- public API ----------
    def detect_regime(self, prices: Iterable[float], assume_returns: bool = False) -> Dict[str, Any]:
        """가격 또는 로그수익률 배열을 받아 현재 국면을 반환.

        Args:
            prices: 최근 시계열(가격 또는 로그수익률). 길이가 window보다 짧아도 동작.
            assume_returns: True이면 입력을 로그수익률로 간주.

        Returns:
            dict: {
                "regime_id": int,
                "regime": str,
                "probs": {state: prob},
                "mu_per_state": {state: mu},
                "sigma_per_state": {state: sigma},
                "volatility": float,  # 선택된 국면의 sigma
                "latest_return": float,
            }
        """

        returns = self._prepare_returns(prices, assume_returns)
        if returns.size == 0:
            return self._empty_response()

        self._maybe_initialize(returns)
        self._update_online(returns)
        last_resp = self._posterior_for_last(returns[-1:])
        state_id = int(self._to_numpy(self.xp.argmax(last_resp)))

        label_map = self._state_labels()
        state_name = label_map.get(state_id, f"state_{state_id}")

        mu_np = self._to_numpy(self.mu)
        var_np = self._to_numpy(self.var)
        probs = {label_map.get(i, f"state_{i}"): float(last_resp[i]) for i in range(self.n_states)}
        mu_dict = {label_map.get(i, f"state_{i}"): float(mu_np[i]) for i in range(self.n_states)}
        sigma_dict = {
            label_map.get(i, f"state_{i}"): float(np.sqrt(max(var_np[i], self.min_var)))
            for i in range(self.n_states)
        }

        return {
            "regime_id": state_id,
            "regime": state_name,
            "probs": probs,
            "mu_per_state": mu_dict,
            "sigma_per_state": sigma_dict,
            "volatility": sigma_dict[state_name],
            "latest_return": float(self._to_numpy(returns[-1])),
        }

    # ---------- core logic ----------
    def _prepare_returns(self, data: Iterable[float], assume_returns: bool) -> Any:
        arr = self._as_xp(np.asarray(list(data), dtype=float))
        if arr.size == 0:
            return arr
        series = arr[-self.window :]
        if assume_returns:
            return series
        if series.size < 2:
            return self.xp.asarray([])
        return self.xp.diff(self.xp.log(self._clip_positive(series)))

    def _clip_positive(self, series: Any) -> Any:
        return self.xp.clip(series, a_min=EPS, a_max=None)

    def _update_online(self, returns: Any) -> None:
        r = returns.reshape(-1, 1)
        log_resp = self._log_prob(r, self.mu, self.var) + self.xp.log(self.pi)
        resp = self._softmax(log_resp, axis=1)

        r_sum = resp.sum(axis=0) + self.xp.asarray(EPS)
        total = resp.sum()

        weighted_mean = (resp * r).sum(axis=0) / r_sum
        # 분산 업데이트: 기존 mu 대신 최신 weighted_mean 사용
        sq = self.xp.square(r - weighted_mean) * resp
        weighted_var = sq.sum(axis=0) / r_sum

        alpha = self.alpha
        self.mu = (1 - alpha) * self.mu + alpha * weighted_mean
        self.var = (1 - alpha) * self.var + alpha * self.xp.maximum(weighted_var, self.min_var)
        self.pi = (1 - alpha) * self.pi + alpha * (r_sum / (total + EPS))

        # 확률 정규화 안전장치
        self.pi = self.pi / (self.pi.sum() + EPS)

    # ---------- initialization ----------
    def initialize_params(self, returns: Any) -> bool:
        """Quantile 기반 초기화. 충분 데이터 없으면 False 반환."""
        arr = np.asarray(returns, dtype=float)
        required = max(self.window - 1, self.n_states)
        if arr.size < required:
            return False

        n = self.n_states
        mu_list = []
        var_list = []
        pi_list = []

        if n == 3:
            q1, q2 = np.percentile(arr, [33.3333, 66.6667])
            masks = [arr <= q1, (arr > q1) & (arr <= q2), arr > q2]
        elif n == 2:
            abs_arr = np.abs(arr)
            q50 = np.percentile(abs_arr, 50.0)
            masks = [abs_arr <= q50, abs_arr > q50]
        else:
            # 일반화: n등분 분위수
            cuts = np.percentile(arr, np.linspace(0, 100, n + 1))
            masks = []
            for i in range(n):
                lo, hi = cuts[i], cuts[i + 1]
                if i == 0:
                    masks.append(arr <= hi)
                elif i == n - 1:
                    masks.append(arr > lo)
                else:
                    masks.append((arr > lo) & (arr <= hi))

        total = float(arr.size)
        global_mu = float(np.mean(arr))
        global_var = float(np.var(arr))

        for mask in masks:
            segment = arr[mask]
            if segment.size == 0:
                seg_mu = global_mu
                seg_var = max(global_var, self.min_var)
                seg_pi = 1.0 / n
            else:
                seg_mu = float(np.mean(segment))
                seg_var = float(np.var(segment))
                seg_var = max(seg_var, self.min_var)
                seg_pi = float(segment.size / total)
            mu_list.append(seg_mu)
            var_list.append(seg_var)
            pi_list.append(seg_pi)

        # 정규화 및 저장 (xp 백엔드)
        pi_arr = np.asarray(pi_list, dtype=float)
        pi_arr = pi_arr / (pi_arr.sum() + EPS)
        self.pi = self._as_xp(pi_arr)
        self.mu = self._as_xp(np.asarray(mu_list, dtype=float))
        self.var = self._as_xp(np.asarray(var_list, dtype=float))
        self.initialized = True
        return True

    def _maybe_initialize(self, returns: Any) -> None:
        if self.initialized:
            return
        try:
            required = max(self.window - 1, self.n_states)
            self.initialize_params(returns[-required:])
        except Exception:
            return

    def _posterior_for_last(self, last_return: Any) -> Any:
        log_resp = self._log_prob(last_return.reshape(1, 1), self.mu, self.var) + self.xp.log(self.pi)
        resp = self._softmax(log_resp, axis=1)[0]
        return self._to_numpy(resp)

    def _log_prob(self, x: Any, mu: Any, var: Any) -> Any:
        return -0.5 * ((x - mu) ** 2 / (var + EPS) + self.xp.log(TWO_PI * (var + EPS)))

    def _softmax(self, x: Any, axis: int = -1) -> Any:
        x_max = self.xp.max(x, axis=axis, keepdims=True)
        exps = self.xp.exp(x - x_max)
        return exps / (self.xp.sum(exps, axis=axis, keepdims=True) + EPS)

    def _state_labels(self) -> Dict[int, str]:
        mu_np = self._to_numpy(self.mu)
        var_np = self._to_numpy(self.var)

        if self.n_states == 3:
            order = np.argsort(mu_np)  # 낮은 mu= bear, 높은 mu= bull
            return {
                int(order[0]): "bear",
                int(order[1]): "chop",
                int(order[2]): "bull",
            }
        if self.n_states == 2:
            order = np.argsort(var_np)  # 낮은 var= calm, 높은 var= volatile
            return {
                int(order[0]): "calm",
                int(order[1]): "volatile",
            }
        return {i: f"state_{i}" for i in range(self.n_states)}

    # ---------- utils ----------
    def _as_xp(self, arr: Any) -> Any:
        return self.xp.asarray(arr)

    def _to_numpy(self, arr: Any) -> np.ndarray:
        return np.asarray(arr)

    def _empty_response(self) -> Dict[str, Any]:
        label_map = self._state_labels()
        probs = {label: 1.0 / self.n_states for label in label_map.values()}
        zeros = {label: 0.0 for label in label_map.values()}
        return {
            "regime_id": 0,
            "regime": label_map.get(0, "state_0"),
            "probs": probs,
            "mu_per_state": zeros,
            "sigma_per_state": {k: float(np.sqrt(self.min_var)) for k in label_map.values()},
            "volatility": float(np.sqrt(self.min_var)),
            "latest_return": 0.0,
        }


def time_regime() -> str:
    """Return current time-based regime: ASIA, EU, US, or OFF."""
    h = datetime.datetime.utcnow().hour
    if 0 <= h < 6:
        return "ASIA"
    if 6 <= h < 13:
        return "EU"
    if 13 <= h < 21:
        return "US"
    return "OFF"


def adjust_mu_sigma(
    mu: float,
    sigma: float,
    regime_ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """
    Adjust drift (mu) and volatility (sigma) based on regime context.

    Args:
        mu: Base drift (annualized)
        sigma: Base volatility (annualized)
        regime_ctx: Optional dict with 'regime', 'vix', 'session' keys

    Returns:
        (adjusted_mu, adjusted_sigma)

    See docs/MATHEMATICS.md for formula details.
    """
    if regime_ctx is None:
        return mu, sigma

    # 방어적 처리: str이 들어온 경우 dict로 래핑
    if isinstance(regime_ctx, str):
        regime_ctx = {"regime": regime_ctx}

    regime = regime_ctx.get("regime", "neutral")
    session = regime_ctx.get("session", time_regime())

    # Regime-based multipliers
    regime_mult = {
        "bull": (1.2, 0.9),      # Higher drift, lower vol in bull
        "bear": (0.7, 1.3),      # Lower drift, higher vol in bear
        "chop": (0.90, 1.25),    # [FIX 2026-02-14] Chop: mu×0.80→0.90, σ×1.5→1.25 완화
        "volatile": (0.6, 1.8),  # [FIX 2026-02-13] Volatile: drift suppressed, vol amplified
        "neutral": (1.0, 1.0),
    }
    mu_mult, sigma_mult = regime_mult.get(regime, (1.0, 1.0))

    # Session-based adjustments
    session_mult = {
        "ASIA": (0.9, 0.8),   # Lower activity
        "EU": (1.0, 1.0),     # Normal
        "US": (1.1, 1.2),     # Higher activity
        "OFF": (0.7, 0.7),    # Weekend/off-hours
    }
    mu_sess, sigma_sess = session_mult.get(session, (1.0, 1.0))

    # VIX-based vol boost (if provided)
    vix = regime_ctx.get("vix")
    vix_mult = 1.0
    if vix is not None and vix > 25:
        vix_mult = 1.0 + (vix - 25) * 0.02  # 2% per VIX point above 25

    adjusted_mu = mu * mu_mult * mu_sess
    adjusted_sigma = sigma * sigma_mult * sigma_sess * vix_mult

    return adjusted_mu, adjusted_sigma


def get_regime_mu_sigma(
    regime: str,
    session: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Get baseline mu and sigma for a given regime.

    Args:
        regime: Market regime ('bull', 'bear', 'chop', 'neutral')
        session: Optional session override
        symbol: Optional symbol for asset-specific params (e.g., 'BTC/USDT:USDT')

    Returns:
        (mu, sigma) - annualized values
    """
    # Asset-specific base volatility (annualized)
    # BTC is baseline, alts typically have higher vol
    symbol_vol_mult = {
        "BTC": 1.0,
        "ETH": 1.15,
        "SOL": 1.4,
        "BNB": 1.2,
        "XRP": 1.3,
        "ADA": 1.35,
        "DOGE": 1.5,
        "AVAX": 1.4,
        "DOT": 1.35,
    }
    
    base_sigma = 0.80  # ~80% annualized vol for BTC
    
    # Extract base asset from symbol (e.g., 'BTC/USDT:USDT' -> 'BTC')
    vol_mult = 1.0
    if symbol:
        base_asset = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
        vol_mult = symbol_vol_mult.get(base_asset, 1.2)  # default 1.2 for unknown alts
    
    adjusted_base_sigma = base_sigma * vol_mult

    regime_params = {
        "bull": (0.50, adjusted_base_sigma * 0.9),    # Positive drift, lower vol
        "bear": (-0.20, adjusted_base_sigma * 1.3),   # Negative drift, higher vol
        "chop": (0.00, adjusted_base_sigma * 1.25),   # [FIX 2026-02-14] Zero drift, moderate vol
        "neutral": (0.10, adjusted_base_sigma),        # Slight positive drift
    }

    mu, sigma = regime_params.get(regime, (0.10, adjusted_base_sigma))

    # Session adjustment (optional)
    if session:
        ctx = {"regime": regime, "session": session}
        mu, sigma = adjust_mu_sigma(mu, sigma, ctx)

    return mu, sigma

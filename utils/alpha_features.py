from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def calculate_order_flow_imbalance(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int = 50,
    use_jax: bool = False,
    eps: float = 1e-8,
    return_components: bool = False,
):
    """
    Bulk Volume Classification 기반 Order Flow Imbalance 계산.

    Args:
        prices: 1D 가격 시계열
        volumes: 1D 거래량 시계열 (prices와 길이 동일 가정)
        window: 계산에 사용할 최근 샘플 개수 (시간/체결 버킷 대용)
        use_jax: True 시 jax.numpy 사용 (호환 모드)
        eps: 0 나누기 방지 상수
        return_components: True 시 (imbalance, v_buy, v_sell) 반환

    Returns:
        불균형 지표 스칼라 (|V_buy - V_sell| / (V_buy + V_sell))
    """

    xp = np  # default numpy backend
    jax_backend = False

    if use_jax:
        try:
            import jax.numpy as jnp  # type: ignore

            xp = jnp
            jax_backend = True
        except Exception:
            xp = np
            jax_backend = False

    prices_arr = xp.asarray(prices, dtype=xp.float64)
    volumes_arr = xp.asarray(volumes, dtype=xp.float64)

    if prices_arr.size < 2 or volumes_arr.size == 0:
        zero = xp.asarray(0.0) if jax_backend else 0.0
        return (zero, zero, zero) if return_components else zero

    n = int(xp.minimum(prices_arr.size, volumes_arr.size))
    prices_arr = prices_arr[-n:]
    volumes_arr = volumes_arr[-n:]

    delta_p = xp.diff(prices_arr)
    if delta_p.size == 0:
        zero = xp.asarray(0.0) if jax_backend else 0.0
        return (zero, zero, zero) if return_components else zero

    window = max(1, int(window))
    delta_p = delta_p[-window:]
    vol_window = volumes_arr[-delta_p.shape[0]:]

    buy_mask = xp.where(delta_p > 0, 1.0, 0.0)
    sell_mask = xp.where(delta_p < 0, 1.0, 0.0)
    neutral_mask = xp.where(delta_p == 0, 1.0, 0.0)

    buy_vol = vol_window * (buy_mask + 0.5 * neutral_mask)
    sell_vol = vol_window * (sell_mask + 0.5 * neutral_mask)

    v_buy = xp.sum(buy_vol)
    v_sell = xp.sum(sell_vol)

    denom = v_buy + v_sell + eps
    imbalance = xp.abs(v_buy - v_sell) / denom

    if return_components:
        if jax_backend:
            return imbalance, v_buy, v_sell
        return float(imbalance), float(v_buy), float(v_sell)

    if jax_backend:
        return imbalance
    return float(imbalance)


def calculate_vpin(
    prices: np.ndarray,
    volumes: np.ndarray,
    bucket_size: Optional[float] = None,
    bucket_count_hint: int = 50,
    vpin_window: Optional[int] = None,
    use_jax: bool = False,
    eps: float = 1e-8,
    return_components: bool = False,
):
    """
    VPIN (Volume-Synchronized PIN) 계산.

    - Bulk Volume Classification(BVC): ΔP 기준 확률적 분류
      V_buy = V * Φ(ΔP/σ), V_sell = V - V_buy
    - Volume Bucket: 누적 거래량이 bucket_size에 도달할 때마다 버킷 완성
    - VPIN = (최근 N개 버킷 OI 합) / (N * bucket_size)

    Args:
        prices: 가격 시계열
        volumes: 거래량 시계열 (prices와 동일 길이)
        bucket_size: 볼륨 버킷 크기. None이면 총 거래량을 bucket_count_hint로 나눈 값 사용
        bucket_count_hint: bucket_size 자동 산출 시 분모 (기본 50)
        vpin_window: VPIN 계산에 사용할 최근 버킷 수. None이면 bucket_count_hint 사용
        use_jax: True 시 jax.numpy 사용 (JIT 미필요, 호환 모드)
        eps: 0 나누기 방지 상수
        return_components: True 시 (vpin, oi_list, bucket_size) 반환
    """

    xp = np
    jax_backend = False

    if use_jax:
        try:
            import jax.numpy as jnp  # type: ignore

            xp = jnp
            jax_backend = True
        except Exception:
            xp = np
            jax_backend = False

    prices_arr = xp.asarray(prices, dtype=xp.float64)
    volumes_arr = xp.asarray(volumes, dtype=xp.float64)

    if prices_arr.size < 2 or volumes_arr.size == 0:
        zero = xp.asarray(0.0) if jax_backend else 0.0
        empty = [] if not jax_backend else ()
        return (zero, empty, zero) if return_components else zero

    n = int(xp.minimum(prices_arr.size, volumes_arr.size))
    prices_arr = prices_arr[-n:]
    volumes_arr = volumes_arr[-n:]

    # ΔP와 σ 추정
    delta_p = xp.diff(prices_arr)
    if delta_p.size == 0:
        zero = xp.asarray(0.0) if jax_backend else 0.0
        empty = [] if not jax_backend else ()
        return (zero, empty, zero) if return_components else zero

    sigma = xp.std(delta_p) + eps

    # 표준정규 CDF Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    def _phi(x):
        if jax_backend:
            return 0.5 * (1.0 + xp.erf(x / xp.sqrt(2.0)))
        try:
            from scipy.special import erf as sp_erf  # type: ignore

            erf_fn = sp_erf
        except Exception:
            import math

            erf_fn = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf_fn(x / np.sqrt(2.0)))

    buy_prob = _phi(delta_p / sigma)
    vol_effective = volumes_arr[1:][: buy_prob.shape[0]]

    # bucket_size 자동 산출: 총 거래량 / bucket_count_hint
    if bucket_size is None:
        total_vol = xp.sum(vol_effective)
        bucket_size = float(total_vol) / max(bucket_count_hint, 1)
        bucket_size = max(bucket_size, eps)

    bucket_size_scalar = float(bucket_size)
    window = vpin_window if vpin_window is not None else bucket_count_hint
    window = max(1, int(window))

    bucket_oi: List[float] = []
    acc_vol = 0.0
    acc_buy = 0.0
    acc_sell = 0.0

    for v, pb in zip(vol_effective.tolist(), buy_prob.tolist()):
        # 실수 값으로 변환 (jax일 경우 Python float)
        v = float(v)
        pb = float(pb)
        buy_v = v * pb
        sell_v = v - buy_v

        remaining = bucket_size_scalar - acc_vol
        # 현 거래량으로 여러 버킷을 채울 수 있는 경우 while로 처리
        while v > 0:
            if v <= remaining + eps:
                acc_vol += v
                acc_buy += buy_v
                acc_sell += sell_v
                remaining -= v
                v = 0.0
            else:
                # 버킷을 채우고 잔량 carry
                fill_ratio = remaining / v
                acc_vol += remaining
                acc_buy += buy_v * fill_ratio
                acc_sell += sell_v * fill_ratio
                # 버킷 종료
                bucket_oi.append(abs(acc_buy - acc_sell))
                # 잔여 거래량/볼륨을 다음 버킷에 carry
                v -= remaining
                buy_v -= buy_v * fill_ratio
                sell_v -= sell_v * fill_ratio
                acc_vol = 0.0
                acc_buy = 0.0
                acc_sell = 0.0
                remaining = bucket_size_scalar

            if acc_vol >= bucket_size_scalar - eps:
                bucket_oi.append(abs(acc_buy - acc_sell))
                acc_vol = 0.0
                acc_buy = 0.0
                acc_sell = 0.0
                remaining = bucket_size_scalar

    # VPIN: 최근 window 버킷 기준
    if len(bucket_oi) == 0:
        vpin = xp.asarray(0.0) if jax_backend else 0.0
    else:
        recent = bucket_oi[-window:]
        vpin = sum(recent) / (len(recent) * bucket_size_scalar + eps)
        if jax_backend:
            vpin = xp.asarray(vpin)

    if return_components:
        return vpin, bucket_oi[-window:], bucket_size_scalar
    return vpin


def safe_z(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / (s + eps)


# ── Hurst Exponent (R/S method) ──
def calculate_hurst_exponent(
    prices: np.ndarray,
    window: int = 120,
    taus: Optional[List[int]] = None,
) -> float:
    """
    Rescaled Range (R/S) 기반 Hurst Exponent.
    H < 0.5: 강한 평균 회귀 (mean-reversion)
    H ≈ 0.5: Random Walk
    H > 0.5: 추세 지속 (trending)
    """
    if taus is None:
        taus = [2, 4, 8, 16, 32]

    prices_arr = np.asarray(prices, dtype=np.float64)
    if prices_arr.size < max(taus) + 2:
        return 0.5  # insufficient data → neutral

    # Use log returns
    lr = np.diff(np.log(np.clip(prices_arr[-window:], 1e-12, None)))
    if lr.size < max(taus):
        return 0.5

    rs_values = []
    valid_taus = []
    for tau in taus:
        if tau > lr.size:
            continue
        n_chunks = lr.size // tau
        if n_chunks < 1:
            continue
        rs_list = []
        for i in range(n_chunks):
            chunk = lr[i * tau : (i + 1) * tau]
            m = np.mean(chunk)
            cumdev = np.cumsum(chunk - m)
            R = float(np.max(cumdev) - np.min(cumdev))
            S = float(np.std(chunk, ddof=1)) if tau > 1 else 1e-12
            if S > 1e-12:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.log(np.mean(rs_list)))
            valid_taus.append(np.log(tau))

    if len(valid_taus) < 2:
        return 0.5

    # Linear regression: log(R/S) = H * log(tau) + c
    x = np.array(valid_taus)
    y = np.array(rs_values)
    A = np.vstack([x, np.ones(len(x))]).T
    try:
        result = np.linalg.lstsq(A, y, rcond=None)
        H = float(result[0][0])
        return max(0.0, min(1.0, H))
    except Exception:
        return 0.5


# ── Bollinger Band %B ──
def bollinger_pct_b(closes: np.ndarray, window: int = 20, n_std: float = 2.0) -> float:
    """
    현재 가격이 볼린저 밴드 내 어디에 있는지 [0, 1] 범위로 반환.
    %B = (Price - Lower) / (Upper - Lower)
    %B > 1.0: 상단 돌파 (과매수)
    %B < 0.0: 하단 돌파 (과매도)
    """
    if len(closes) < window:
        return 0.5
    recent = closes[-window:]
    ma = float(np.mean(recent))
    std = float(np.std(recent))
    if std < 1e-12:
        return 0.5
    upper = ma + n_std * std
    lower = ma - n_std * std
    band_width = upper - lower
    if band_width < 1e-12:
        return 0.5
    pct_b = (float(closes[-1]) - lower) / band_width
    return float(np.clip(pct_b, -0.5, 1.5))


# ── RSI (Relative Strength Index) ──
def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
    """
    RSI = 100 - 100 / (1 + RS), RS = avg_gain / avg_loss
    RSI < 30: 과매도 (반등 기대)
    RSI > 70: 과매수 (하락 기대)
    반환값: 0~100
    """
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss < 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


# ── 이동평균 이격도 Z-Score ──
def ma_zscore(closes: np.ndarray, ma_window: int = 20, lookback: int = 60) -> float:
    """
    현재 가격과 이동평균의 괴리를 Z-Score로 정규화.
    양수: 이동평균보다 위 (과매수 방향)
    음수: 이동평균보다 아래 (과매도 방향)
    """
    if len(closes) < max(ma_window, lookback):
        return 0.0
    ma = np.convolve(closes[-lookback:], np.ones(ma_window) / ma_window, mode='valid')
    if len(ma) < 2:
        return 0.0
    deviations = closes[-len(ma):] - ma
    std = float(np.std(deviations))
    if std < 1e-12:
        return 0.0
    return float(deviations[-1] / std)


def build_alpha_features(
    closes: np.ndarray,      # [T]
    vols: np.ndarray,        # [T]
    returns: np.ndarray,     # [T-1]
    ofi_z: float,
    spread_pct: float,
    pmaker_entry: float,
    pmaker_delay_sec: float,
    regime_id: float,
) -> np.ndarray:
    """
    최소 피처 세트. (너가 원하는 ticker/체결내역은 아래에 계속 추가하면 됨)

    - momentum multi-scale
    - vol regime
    - ofi
    - spread
    - pmaker
    """
    T = len(closes)
    if T < 60:
        # pad
        # ✅ 빈 배열인 경우 처리
        if T == 0:
            # 빈 배열이면 기본값으로 채움
            closes = np.full(60, closes[0] if len(closes) > 0 and closes[0] > 0 else 1.0, dtype=np.float64)
            vols = np.full(60, vols[0] if len(vols) > 0 and vols[0] > 0 else 0.01, dtype=np.float64)
            returns = np.zeros(59, dtype=np.float64)
        else:
            closes = np.pad(closes, (60 - T, 0), mode="edge")
            vols = np.pad(vols, (60 - T, 0), mode="edge")
            returns = np.diff(closes)

    # log returns
    lr = np.diff(np.log(closes + 1e-12))
    def mom(k):
        k = min(k, len(closes) - 1)
        return float(np.log(closes[-1] / (closes[-1-k] + 1e-12)))

    # realized vol
    rv_30 = float(np.std(lr[-30:]) + 1e-8)
    rv_120 = float(np.std(lr[-120:]) + 1e-8)

    # ── Chop-specific alpha features ──
    bb_pct_b = bollinger_pct_b(closes, window=20, n_std=2.0)
    rsi_14 = calculate_rsi(closes, period=14) / 100.0  # normalize to [0, 1]
    ma_z = ma_zscore(closes, ma_window=20, lookback=60)
    hurst = calculate_hurst_exponent(closes, window=120)

    f = [
        mom(5), mom(15), mom(30), mom(60), mom(120),
        rv_30, rv_120,
        float(ofi_z),
        float(spread_pct),
        float(pmaker_entry),
        float(pmaker_delay_sec),
        float(regime_id),
        # New chop-regime features
        float(bb_pct_b),      # Bollinger %B: 0~1 (박스권 위치)
        float(rsi_14),        # RSI normalized: 0~1
        float(ma_z),          # MA Z-Score: 이격도
        float(hurst),         # Hurst Exponent: 0~1
    ]
    return np.asarray(f, dtype=np.float32)


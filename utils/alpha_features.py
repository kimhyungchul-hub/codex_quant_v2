from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def safe_z(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / (s + eps)


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

    f = [
        mom(5), mom(15), mom(30), mom(60), mom(120),
        rv_30, rv_120,
        float(ofi_z),
        float(spread_pct),
        float(pmaker_entry),
        float(pmaker_delay_sec),
        float(regime_id),
    ]
    return np.asarray(f, dtype=np.float32)


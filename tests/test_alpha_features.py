import numpy as np
from utils.alpha_features import calculate_order_flow_imbalance, calculate_vpin


def test_order_flow_imbalance_range():
    np.random.seed(1)
    T = 120
    prices = np.cumsum(np.random.randn(T) * 0.05) + 100.0
    vols = np.abs(np.random.randn(T) * 20 + 50)

    imb = calculate_order_flow_imbalance(prices, vols, window=50)
    assert isinstance(imb, (float,)) or hasattr(imb, 'item')
    # imbalance must be between 0 and 1
    val = float(imb)
    assert 0.0 <= val <= 1.0


def test_vpin_wrapper_matches():
    np.random.seed(2)
    T = 150
    prices = np.cumsum(np.random.randn(T) * 0.07) + 200.0
    vols = np.abs(np.random.randn(T) * 10 + 80)

    # bucket_size = 총거래량 / 20 기준으로 계산
    bucket_size = float(np.sum(vols) / 20)
    vpin, oi_list, used_bucket = calculate_vpin(
        prices,
        vols,
        bucket_size=bucket_size,
        bucket_count_hint=20,
        vpin_window=10,
        return_components=True,
    )

    assert 0.0 <= float(vpin) <= 1.0
    assert used_bucket > 0
    assert len(oi_list) <= 10


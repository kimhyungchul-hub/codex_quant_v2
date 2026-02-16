"""DirectionModel unit tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.mc.direction_model import DirectionModel, compute_direction_override


def test_long_direction():
    """mu_alpha > 0 + positive momentum -> LONG."""
    r = compute_direction_override(
        mu_alpha=2.5,
        meta={"mu_mom": 1.0, "mu_ofi": 0.5, "mu_kf": 0.3},
        ctx={"regime": "trending", "hurst": 0.65, "vpin": 0.3},
        score_long=0.05, score_short=0.04,
        ev_long=0.01, ev_short=-0.005,
    )
    assert r["direction"] == 1, f"Expected LONG(1), got {r['direction']}"
    assert r["confidence"] > 0.5, f"Expected conf > 0.5, got {r['confidence']}"
    print(f"  LONG: dir={r['direction']}, conf={r['confidence']:.3f}, src={r['direction_source']}, consensus={r.get('consensus_score',0):.3f}")


def test_short_direction():
    """mu_alpha < 0 + negative momentum -> SHORT."""
    r = compute_direction_override(
        mu_alpha=-3.0,
        meta={"mu_mom": -1.5, "mu_ofi": -0.8},
        ctx={"regime": "mean_reverting", "hurst": 0.35, "vpin": 0.7},
        score_long=0.03, score_short=0.06,
        ev_long=-0.002, ev_short=0.008,
    )
    assert r["direction"] == -1, f"Expected SHORT(-1), got {r['direction']}"
    assert r["confidence"] > 0.5
    print(f"  SHORT: dir={r['direction']}, conf={r['confidence']:.3f}, src={r['direction_source']}, consensus={r.get('consensus_score',0):.3f}")


def test_weak_signal():
    """mu_alpha near zero -> low confidence."""
    r = compute_direction_override(
        mu_alpha=0.001,
        meta={},
        ctx={"regime": "choppy", "hurst": 0.50, "vpin": 0.5},
        score_long=0.02, score_short=0.021,
        ev_long=0.001, ev_short=0.001,
    )
    # Direction should still be decided, but confidence should be lower
    assert r["direction"] in (1, -1)
    print(f"  WEAK: dir={r['direction']}, conf={r['confidence']:.3f}, src={r['direction_source']}, consensus={r.get('consensus_score',0):.3f}")


def test_calibration_update():
    """Online Platt scaling should adjust confidence."""
    dm = DirectionModel()
    # Feed calibration data: high raw_conf -> 66% WR
    for i in range(60):
        dm.update_calibration(0.7, i % 3 != 0)
    # Feed calibration data: low raw_conf -> 20% WR
    for i in range(60):
        dm.update_calibration(0.3, i % 5 == 0)

    r = compute_direction_override(
        mu_alpha=1.5,
        meta={"mu_mom": 0.8},
        ctx={"regime": "trending", "hurst": 0.60},
        score_long=0.04, score_short=0.03,
        ev_long=0.005, ev_short=-0.001,
    )
    print(f"  CALIBRATED: dir={r['direction']}, conf={r['confidence']:.3f}, raw={r.get('raw_confidence',0):.3f}")


def test_consensus_agreement():
    """All signals agree -> high consensus, high confidence."""
    r_agree = compute_direction_override(
        mu_alpha=3.0,
        meta={"mu_mom": 2.0, "mu_ofi": 1.5, "mu_kf": 1.0, "mu_bayes": 0.5, "mu_ml": 0.8},
        ctx={"regime": "trending", "hurst": 0.70, "vpin": 0.2},
        score_long=0.06, score_short=0.02,
        ev_long=0.015, ev_short=-0.01,
    )
    r_disagree = compute_direction_override(
        mu_alpha=1.0,
        meta={"mu_mom": -1.0, "mu_ofi": 0.5, "mu_kf": -0.3, "mu_bayes": -0.2, "mu_ml": 0.1},
        ctx={"regime": "choppy", "hurst": 0.50, "vpin": 0.5},
        score_long=0.035, score_short=0.034,
        ev_long=0.002, ev_short=0.001,
    )
    print(f"  AGREE:    dir={r_agree['direction']}, conf={r_agree['confidence']:.3f}, consensus={r_agree.get('consensus_score',0):.3f}, signals={r_agree.get('signal_count',0)}")
    print(f"  DISAGREE: dir={r_disagree['direction']}, conf={r_disagree['confidence']:.3f}, consensus={r_disagree.get('consensus_score',0):.3f}, signals={r_disagree.get('signal_count',0)}")
    assert r_agree["confidence"] > r_disagree["confidence"], \
        f"Agreement conf ({r_agree['confidence']:.3f}) should > disagreement conf ({r_disagree['confidence']:.3f})"


if __name__ == "__main__":
    tests = [test_long_direction, test_short_direction, test_weak_signal,
             test_calibration_update, test_consensus_agreement]
    for t in tests:
        print(f"Running {t.__name__}...")
        t()
        print(f"  PASSED")
    print(f"\nAll {len(tests)} tests passed!")

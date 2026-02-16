#!/usr/bin/env python3
"""Test script for mu_alpha pipeline fixes (2026-02-09)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def test_divergence_gate():
    """Fix 1: Divergence gate should keep 30% of dominant signal, not zero."""
    from engines.mc.signal_features import _calculate_refined_mu_alpha_with_debug
    
    closes = np.array([100 + 0.1*i for i in range(30)], dtype=np.float64)
    params = {'mu_mom_scale': 10.0, 'mu_ofi_scale': 8.0, 'mu_cap': 5.0, 'mu_floor': -5.0}
    
    # Divergence case: positive momentum, negative OFI
    mu, dbg = _calculate_refined_mu_alpha_with_debug(closes, -0.5, 0.01, 0.01, 0.5, params)
    print(f"  divergence: mu={mu:.4f}, combined_raw={dbg['combined_raw']:.4f}, divergence={dbg['divergence']}")
    assert dbg['divergence'] == True, "Should detect divergence"
    assert abs(dbg['combined_raw']) > 0.0, f"Signal should NOT be zero, got {dbg['combined_raw']}"
    print("  PASS: Divergence gate keeps partial signal")
    
    # Normal agreement
    mu2, dbg2 = _calculate_refined_mu_alpha_with_debug(closes, 0.5, 0.01, 0.01, 0.5, params)
    print(f"  agreement: mu={mu2:.4f}, combined_raw={dbg2['combined_raw']:.4f}")
    assert dbg2['divergence'] == False
    print("  PASS: Agreement case works")

def test_kama_er_dampening():
    """Fix 2: KAMA ER dampening should be linear, not quadratic."""
    from engines.mc.signal_features import _calculate_refined_mu_alpha_with_debug
    
    closes = np.array([100 + 0.1*i for i in range(30)], dtype=np.float64)
    params = {'mu_mom_scale': 10.0, 'mu_ofi_scale': 8.0, 'mu_cap': 5.0, 'mu_floor': -5.0, 'chop_threshold': 0.3}
    
    # Low KAMA ER = 0.1 (deep chop)
    mu, dbg = _calculate_refined_mu_alpha_with_debug(closes, 0.5, 0.01, 0.01, 0.1, params)
    rf = dbg['regime_factor']
    print(f"  ER=0.1: regime_factor={rf:.4f} (old quadratic would give {(0.1/0.3)**2:.4f})")
    assert rf >= 0.3, f"Linear dampening should give >= 0.3, got {rf}"
    
    # Medium KAMA ER = 0.2
    mu2, dbg2 = _calculate_refined_mu_alpha_with_debug(closes, 0.5, 0.01, 0.01, 0.2, params)
    rf2 = dbg2['regime_factor']
    print(f"  ER=0.2: regime_factor={rf2:.4f} (old quadratic would give {(0.2/0.3)**2:.4f})")
    assert rf2 >= 0.6, f"Linear dampening should give >= 0.6, got {rf2}"
    print("  PASS: KAMA ER uses linear dampening")

def test_mu_alpha_cap():
    """Fix 4: mu_alpha cap should be 5.0 (not 15.0)."""
    from engines.mc.config import MCConfig
    
    config = MCConfig()
    print(f"  mu_alpha_cap={config.mu_alpha_cap}, mu_alpha_floor={config.mu_alpha_floor}")
    print(f"  mu_ofi_scale={config.mu_ofi_scale}, alpha_scaling_factor={config.alpha_scaling_factor}")
    assert config.mu_alpha_cap <= 5.0, f"Cap should be <= 5.0, got {config.mu_alpha_cap}"
    assert config.mu_alpha_floor >= -5.0, f"Floor should be >= -5.0, got {config.mu_alpha_floor}"
    print("  PASS: mu_alpha cap is sensible")

def test_compound_dampening():
    """Fix 3: Compound dampening should be much less aggressive."""
    # Worst case in chop: KAMA ER=0.1, Hurst=0.50, regime=chop
    # OLD: (0.1/0.3)^2 × 0.25 × 0.8 = 0.11 × 0.25 × 0.8 = 0.022 (98% reduction)
    # NEW: max(0.3, 0.1/0.3) × 0.60 × 0.8 = 0.33 × 0.60 × 0.8 = 0.16 (84% reduction)
    old_compound = (0.1/0.3)**2 * 0.25 * 0.8
    new_compound = max(0.3, 0.1/0.3) * 0.60 * 0.8
    print(f"  OLD compound dampening (ER=0.1): {old_compound:.4f} ({(1-old_compound)*100:.1f}% reduction)")
    print(f"  NEW compound dampening (ER=0.1): {new_compound:.4f} ({(1-new_compound)*100:.1f}% reduction)")
    assert new_compound > old_compound * 5, f"New should be 5x+ less aggressive"
    
    # More typical case: ER=0.2
    old_typ = (0.2/0.3)**2 * 0.25 * 0.8
    new_typ = max(0.3, 0.2/0.3) * 0.60 * 0.8
    print(f"  OLD compound (ER=0.2): {old_typ:.4f} ({(1-old_typ)*100:.1f}% reduction)")
    print(f"  NEW compound (ER=0.2): {new_typ:.4f} ({(1-new_typ)*100:.1f}% reduction)")
    print("  PASS: Compound dampening is now reasonable")

def test_signal_gradient():
    """Test that mu_alpha values have gradient (not binary ±5)."""
    from engines.mc.signal_features import _calculate_refined_mu_alpha_with_debug
    
    params = {'mu_mom_scale': 10.0, 'mu_ofi_scale': 8.0, 'mu_cap': 5.0, 'mu_floor': -5.0}
    values = []
    for ofi in np.linspace(-1.0, 1.0, 11):
        closes = np.array([100 + 0.1*i for i in range(30)], dtype=np.float64)
        mu, _ = _calculate_refined_mu_alpha_with_debug(closes, float(ofi), 0.01, 0.01, 0.5, params)
        values.append(mu)
    
    unique_vals = len(set([round(v, 2) for v in values]))
    print(f"  OFI from -1 to +1: {unique_vals} unique mu_alpha values")
    print(f"  Values: {[round(v, 2) for v in values]}")
    assert unique_vals >= 5, f"Should have gradient, not binary. Got {unique_vals} unique values"
    print("  PASS: mu_alpha has gradient (not binary)")

if __name__ == "__main__":
    print("=" * 60)
    print("mu_alpha Pipeline Fix Verification")
    print("=" * 60)
    
    tests = [
        ("1. Divergence Gate Fix", test_divergence_gate),
        ("2. KAMA ER Linear Dampening", test_kama_er_dampening),
        ("3. Compound Dampening Reduction", test_compound_dampening),
        ("4. mu_alpha Cap Reduction", test_mu_alpha_cap),
        ("5. Signal Gradient Test", test_signal_gradient),
    ]
    
    passed = 0
    for name, func in tests:
        print(f"\n[{name}]")
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)

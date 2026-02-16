#!/usr/bin/env python3
"""Signal chain dampen analysis — trace mu_alpha → unified_score"""
import math

print("=" * 72)
print("Signal Chain Dampen Analysis: mu_alpha → unified_score")
print("=" * 72)

# STAGE 1: signal_features.py
# Raw momentum + OFI → mu_alpha_raw
# Typical raw values: mu_mom ~ ±(0.5 ~ 5), mu_ofi ~ ±(0.5 ~ 5)
# After KAMA ER dampen, combined_raw might be ±(0.3 ~ 3)
mu_raw_example = 1.5  # typical post-KAMA combined value

print(f"\n--- STAGE 1: Raw Signal (signal_features.py) ---")
print(f"mu_alpha_raw (example) = {mu_raw_example}")

# STAGE 2: entry_evaluation.py advanced alpha blending
# base_w ~ 0.30 if all components active, but usually only a few
# Net effect: typically marginal, ±10%
mu_after_blend = mu_raw_example * 0.85  # conservative blend
print(f"\n--- STAGE 2: Alpha Blending ---")
print(f"mu_after_blend = {mu_after_blend:.4f} (×0.85 conservative)")

# STAGE 3: Dampen chain

# 3.1 Hurst random walk dampen (H ≈ 0.5 → ×0.60)
hurst_factor = 0.60  # default for random walk
mu_after_hurst = mu_after_blend * hurst_factor
print(f"\n--- STAGE 3.1: Hurst Dampen (random walk) ---")
print(f"hurst_factor = {hurst_factor}")
print(f"mu_after_hurst = {mu_after_hurst:.4f} (×{hurst_factor})")

# 3.2 VPIN dampen (typical VPIN ~ 0.3-0.7)
vpin_typical = 0.5
vpin_damp = max(0.10, 1 - 0.6 * vpin_typical)  # = 0.70
mu_after_vpin = mu_after_hurst * vpin_damp
print(f"\n--- STAGE 3.2: VPIN Dampen ---")
print(f"vpin = {vpin_typical}, damp = {vpin_damp:.2f}")
print(f"mu_after_vpin = {mu_after_vpin:.4f} (×{vpin_damp:.2f})")

# 3.3 Hawkes/Direction/PMaker — typically small effect, skip

# STAGE 4: regime.py adjust_mu_sigma
# chop: mu × 0.8, sigma × 1.5
chop_mu_mult = 0.8
session_mult = 0.9  # ASIA session
mu_after_regime = mu_after_vpin * chop_mu_mult * session_mult
print(f"\n--- STAGE 4: Regime Adjust (chop + ASIA) ---")
print(f"chop_mu_mult = {chop_mu_mult}, session = {session_mult}")  
print(f"mu_after_regime = {mu_after_regime:.6f} (×{chop_mu_mult*session_mult:.2f})")

sigma = 0.50  # typical annualized vol
sigma_after = sigma * 1.5 * 0.8  # chop × session
print(f"sigma after regime = {sigma_after:.2f}")

# STAGE 5: MC Path Simulation → EV
# Net PnL = direction × (exp(mu_adj * dt - sigma²/2 * dt + sigma * sqrt(dt) * Z) - 1) × lev - fee × lev
# For short horizons (60s = 60/31536000 year), mu is negligible
dt_60s = 60 / 31536000  # ~1.9e-6 year
mu_drift_60s = mu_after_regime * dt_60s
sigma_effect_60s = sigma_after * math.sqrt(dt_60s)

print(f"\n--- STAGE 5: MC Path (60s horizon) ---")
print(f"dt (60s as year) = {dt_60s:.2e}")
print(f"mu × dt = {mu_drift_60s:.2e}  (drift per step)")
print(f"sigma × √dt = {sigma_effect_60s:.6f}  (noise per step)")
print(f"Drift/Noise ratio = {abs(mu_drift_60s/sigma_effect_60s):.2e}")

# For 60 steps (60s horizon, 1s steps):
total_drift = mu_drift_60s * 60
total_noise = sigma_effect_60s * math.sqrt(60)
print(f"\n60 steps (60s horizon):")
print(f"  Total drift = {total_drift:.2e}")
print(f"  Total noise = {total_noise:.6f}")
print(f"  Signal/Noise = {abs(total_drift/total_noise):.2e}")

# Fee impact: 0.07% roundtrip + slippage ≈ 0.08%  
fee_roe = 0.0007 * 1.0  # at 1x leverage
print(f"  Fee (roundtrip 0.07% at 1x lev) = {fee_roe:.6f}")
print(f"  >>> {total_drift:.2e} drift vs {fee_roe:.4f} fee → {'DRIFT < FEE ❌' if abs(total_drift) < fee_roe else 'DRIFT > FEE ✅'}")

# STAGE 6: Unified Score (Ψ)
# Ψ(t) = (NAPV(t) - cost) / t 
# When drift ≈ 0 and fee dominates: Ψ ≈ -cost/t which is negative
# This explains both_ev_neg AND unified < threshold

print(f"\n{'='*72}")
print(f"DIAGNOSIS:")
print(f"{'='*72}")
print(f"""
Root Cause Chain:
1. mu_alpha_raw ~ 1.5 (reasonable)
2. × Hurst 0.60 = {mu_raw_example * 0.60:.2f}
3. × VPIN 0.70 = {mu_raw_example * 0.60 * 0.70:.2f}
4. × Chop 0.80 = {mu_raw_example * 0.60 * 0.70 * 0.80:.2f}
5. × Session 0.90 = {mu_raw_example * 0.60 * 0.70 * 0.80 * 0.90:.4f}
6. Total dampen = {0.60 * 0.70 * 0.80 * 0.90:.4f} (= {0.60 * 0.70 * 0.80 * 0.90 * 100:.1f}% retained)
   mu_final = {mu_raw_example * 0.60 * 0.70 * 0.80 * 0.90:.4f}

But even this mu = {mu_raw_example * 0.60 * 0.70 * 0.80 * 0.90:.4f} annualized is
WAY too small vs the noise+fee in 60s-3600s horizons.

Actual problem: CHOP_ENTRY_FLOOR_ADD = 0.0015
- unified_score (Ψ) Max = 0.000419 
- Effective threshold in chop = 0.001501
- Gap: score needs to be 3.6× larger to pass

Fix options:
A. Remove/reduce CHOP_ENTRY_FLOOR_ADD (0.0015 → 0.0003 or 0)
B. Lower unified_floor globally (already at -0.0003)
C. Combine: set CHOP_ENTRY_FLOOR_ADD=0.0003 (chop penalty but passable)
""")

print(f"\nAlso: both_ev_neg blocks because Long EV AND Short EV both ≤ 0")
print(f"This is EXPECTED when mu is nearly 0 and fee > drift")
print(f"If chop_entry_floor is lowered, both_ev_neg may still block")
print(f"Consider: ENTRY_BOTH_EV_NEG_NET_FLOOR=-0.0003 (allow slightly negative)")

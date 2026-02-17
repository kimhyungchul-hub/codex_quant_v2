# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-17 21:14
> Baseline: 4635 trades, PnL=$-71.47, WR=36.5%, R:R=1.43

## Pipeline Stage Impact Summary

### MC_HYBRID_PATHS — mc_hybrid_paths

**Best Finding:** mc_hybrid_paths: PnL +$848.56
- Improvement: $+848.56
- Confidence: 80%
- Parameters: `{"mc_hybrid_n_paths": 16384, "mc_hybrid_horizon_steps": 300}`

```
[MC_HYBRID_PATHS] 파라미터 변경 제안:
  mc_hybrid_n_paths = 16384
  mc_hybrid_horizon_steps = 300
예상 효과: PnL $+848.56, WR +0.0%, R:R +4.83
신뢰도: 80.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 4635 | +0 |
| pnl | -71.47 | 777.09 | +848.56 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.43 | 6.27 | +4.83 |
| edge | -0.0457 | 0.2277 | +0.2734 |
| sharpe | -1.28 | 5.20 | +6.48 |
| pf | 0.82 | 3.61 | +2.78 |

### TP_SL — TP/SL 타겟

**Best Finding:** tp_sl: PnL +$148.63
- Improvement: $+148.63
- Confidence: 80%
- Parameters: `{"tp_pct": 0.04, "sl_pct": 0.005}`

```
[TP_SL] 파라미터 변경 제안:
  tp_pct = 0.04
  sl_pct = 0.005
예상 효과: PnL $+148.63, WR -0.5%, R:R +1.78
신뢰도: 80.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 4635 | +0 |
| pnl | -71.47 | 77.16 | +148.63 |
| wr | 0.3653 | 0.3601 | -0.0052 |
| rr | 1.43 | 3.22 | +1.78 |
| edge | -0.0457 | 0.1230 | +0.1687 |
| sharpe | -1.28 | 3.71 | +4.99 |
| pf | 0.82 | 1.81 | +0.99 |

### VOLATILITY_GATE — volatility_gate

**Best Finding:** volatility_gate: PnL +$129.03
- Improvement: $+129.03
- Confidence: 83%
- Parameters: `{"scope": "chop_only", "chop_min_sigma": 0.35, "chop_max_sigma": 4.0, "chop_max_vpin": 0.65, "chop_min_dir_conf": 0.68, "chop_min_abs_mu_alpha": 40.0, "chop_max_hold_sec": 600}`

```
[VOLATILITY_GATE] 파라미터 변경 제안:
  scope = chop_only
  chop_min_sigma = 0.35
  chop_max_sigma = 4.0
  chop_max_vpin = 0.65
  chop_min_dir_conf = 0.68
  chop_min_abs_mu_alpha = 40.0
  chop_max_hold_sec = 600
예상 효과: PnL $+129.03, WR +8.4%, R:R +0.30
신뢰도: 82.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 1213 | -3422 |
| pnl | -71.47 | 57.55 | +129.03 |
| wr | 0.3653 | 0.4493 | +0.0840 |
| rr | 1.43 | 1.73 | +0.30 |
| edge | -0.0457 | 0.0834 | +0.1291 |
| sharpe | -1.28 | 1.26 | +2.54 |
| pf | 0.82 | 1.41 | +0.59 |

### CHOP_GUARD — chop_guard

**Best Finding:** chop_guard: PnL +$127.61
- Improvement: $+127.61
- Confidence: 83%
- Parameters: `{"chop_entry_floor_add": 0.003, "chop_entry_min_dir_conf": 0.8}`

```
[CHOP_GUARD] 파라미터 변경 제안:
  chop_entry_floor_add = 0.003
  chop_entry_min_dir_conf = 0.8
예상 효과: PnL $+127.61, WR +7.1%, R:R +0.46
신뢰도: 83.5%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 955 | -3680 |
| pnl | -71.47 | 56.14 | +127.61 |
| wr | 0.3653 | 0.4366 | +0.0713 |
| rr | 1.43 | 1.89 | +0.46 |
| edge | -0.0457 | 0.0911 | +0.1368 |
| sharpe | -1.28 | 1.33 | +2.61 |
| pf | 0.82 | 1.47 | +0.64 |

### REGIME_SIDE_BLOCK — regime_side_block

**Best Finding:** regime_side_block: PnL +$102.91
- Improvement: $+102.91
- Confidence: 77%
- Parameters: `{"regime_side_block_list": "bear_long,bull_short,chop_long"}`

```
[REGIME_SIDE_BLOCK] 파라미터 변경 제안:
  regime_side_block_list = bear_long,bull_short,chop_long
예상 효과: PnL $+102.91, WR +7.1%, R:R +0.15
신뢰도: 77.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 1862 | -2773 |
| pnl | -71.47 | 31.44 | +102.91 |
| wr | 0.3653 | 0.4361 | +0.0708 |
| rr | 1.43 | 1.58 | +0.15 |
| edge | -0.0457 | 0.0487 | +0.0944 |
| sharpe | -1.28 | 0.98 | +2.27 |
| pf | 0.82 | 1.22 | +0.40 |

### DIRECTION_GATE — direction_gate

**Best Finding:** direction_gate: PnL +$92.55
- Improvement: $+92.55
- Confidence: 78%
- Parameters: `{"dir_gate_min_conf": 0.7, "dir_gate_min_edge": 0.0}`

```
[DIRECTION_GATE] 파라미터 변경 제안:
  dir_gate_min_conf = 0.7
  dir_gate_min_edge = 0.0
예상 효과: PnL $+92.55, WR -0.5%, R:R +0.88
신뢰도: 77.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 843 | -3792 |
| pnl | -71.47 | 21.07 | +92.55 |
| wr | 0.3653 | 0.3606 | -0.0047 |
| rr | 1.43 | 2.32 | +0.88 |
| edge | -0.0457 | 0.0590 | +0.1047 |
| sharpe | -1.28 | 0.58 | +1.87 |
| pf | 0.82 | 1.31 | +0.48 |

### HYBRID_LEVERAGE — hybrid_leverage

**Best Finding:** hybrid_leverage: PnL +$90.12
- Improvement: $+90.12
- Confidence: 68%
- Parameters: `{"hybrid_lev_sweep_min": 1.0, "hybrid_lev_sweep_max": 3.0, "hybrid_lev_ev_scale": 100}`

```
[HYBRID_LEVERAGE] 파라미터 변경 제안:
  hybrid_lev_sweep_min = 1.0
  hybrid_lev_sweep_max = 3.0
  hybrid_lev_ev_scale = 100
예상 효과: PnL $+90.12, WR +0.0%, R:R +0.41
신뢰도: 68.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 4635 | +0 |
| pnl | -71.47 | 18.65 | +90.12 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.43 | 1.84 | +0.41 |
| edge | -0.0457 | 0.0131 | +0.0588 |
| sharpe | -1.28 | 0.23 | +1.52 |
| pf | 0.82 | 1.06 | +0.23 |

### ENTRY_FILTER — 진입 필터

**Best Finding:** entry_filter: PnL +$89.81
- Improvement: $+89.81
- Confidence: 68%
- Parameters: `{"min_confidence": 0.55, "min_dir_conf": 0.65, "min_entry_quality": 0.3, "min_ev": 0.02}`

```
[ENTRY_FILTER] 파라미터 변경 제안:
  min_confidence = 0.55
  min_dir_conf = 0.65
  min_entry_quality = 0.3
  min_ev = 0.02
예상 효과: PnL $+89.81, WR +9.2%, R:R +0.28
신뢰도: 68.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 494 | -4141 |
| pnl | -71.47 | 18.34 | +89.81 |
| wr | 0.3653 | 0.4575 | +0.0922 |
| rr | 1.43 | 1.71 | +0.28 |
| edge | -0.0457 | 0.0889 | +0.1346 |
| sharpe | -1.28 | 0.77 | +2.05 |
| pf | 0.82 | 1.44 | +0.62 |

### DIRECTION — 방향 결정

**Best Finding:** direction: PnL +$89.34
- Improvement: $+89.34
- Confidence: 80%
- Parameters: `{"chop_prefer_short": true, "min_dir_conf_for_entry": 0.5, "mu_alpha_sign_override": true}`

```
[DIRECTION] 파라미터 변경 제안:
  chop_prefer_short = True
  min_dir_conf_for_entry = 0.5
  mu_alpha_sign_override = True
예상 효과: PnL $+89.34, WR +19.9%, R:R -0.62
신뢰도: 80.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 3821 | -814 |
| pnl | -71.47 | 17.87 | +89.34 |
| wr | 0.3653 | 0.5643 | +0.1990 |
| rr | 1.43 | 0.81 | -0.62 |
| edge | -0.0457 | 0.0132 | +0.0589 |
| sharpe | -1.28 | 0.33 | +1.61 |
| pf | 0.82 | 1.05 | +0.23 |

### LEVERAGE — 레버리지 결정

**Best Finding:** leverage: PnL +$83.34
- Improvement: $+83.34
- Confidence: 68%
- Parameters: `{"max_leverage": 50, "regime_max_bull": 20, "regime_max_chop": 3, "regime_max_bear": 5}`

```
[LEVERAGE] 파라미터 변경 제안:
  max_leverage = 50
  regime_max_bull = 20
  regime_max_chop = 3
  regime_max_bear = 5
예상 효과: PnL $+83.34, WR +0.0%, R:R +0.39
신뢰도: 67.9%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 4635 | +0 |
| pnl | -71.47 | 11.87 | +83.34 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.43 | 1.83 | +0.39 |
| edge | -0.0457 | 0.0114 | +0.0571 |
| sharpe | -1.28 | 0.26 | +1.54 |
| pf | 0.82 | 1.05 | +0.23 |

### VPIN_FILTER — VPIN 필터

**Best Finding:** vpin_filter: PnL +$78.48
- Improvement: $+78.48
- Confidence: 70%
- Parameters: `{"max_vpin": 0.3}`

```
[VPIN_FILTER] 파라미터 변경 제안:
  max_vpin = 0.3
예상 효과: PnL $+78.48, WR +3.9%, R:R +0.09
신뢰도: 69.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 2109 | -2526 |
| pnl | -71.47 | 7.01 | +78.48 |
| wr | 0.3653 | 0.4040 | +0.0387 |
| rr | 1.43 | 1.53 | +0.09 |
| edge | -0.0457 | 0.0084 | +0.0541 |
| sharpe | -1.28 | 0.15 | +1.43 |
| pf | 0.82 | 1.04 | +0.21 |

### HYBRID_EXIT_TIMING — hybrid_exit_timing

**Best Finding:** hybrid_exit_timing: PnL +$71.40
- Improvement: $+71.40
- Confidence: 66%
- Parameters: `{"hybrid_exit_confirm_shock": 5, "hybrid_exit_confirm_normal": 8, "hybrid_exit_confirm_noise": 12}`

```
[HYBRID_EXIT_TIMING] 파라미터 변경 제안:
  hybrid_exit_confirm_shock = 5
  hybrid_exit_confirm_normal = 8
  hybrid_exit_confirm_noise = 12
예상 효과: PnL $+71.40, WR +0.0%, R:R +0.30
신뢰도: 66.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 4635 | +0 |
| pnl | -71.47 | -0.07 | +71.40 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.43 | 1.74 | +0.30 |
| edge | -0.0457 | -0.0000 | +0.0457 |
| sharpe | -1.28 | -0.00 | +1.28 |
| pf | 0.82 | 1.00 | +0.18 |

### PRE_MC_GATE — pre_mc_gate

**Best Finding:** pre_mc_gate: PnL +$70.61
- Improvement: $+70.61
- Confidence: 44%
- Parameters: `{"pre_mc_min_expected_pnl": 0.0, "pre_mc_max_liq_prob": 0.1}`

```
[PRE_MC_GATE] 파라미터 변경 제안:
  pre_mc_min_expected_pnl = 0.0
  pre_mc_max_liq_prob = 0.1
예상 효과: PnL $+70.61, WR +7.6%, R:R -0.89
신뢰도: 43.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 102 | -4533 |
| pnl | -71.47 | -0.86 | +70.61 |
| wr | 0.3653 | 0.4412 | +0.0759 |
| rr | 1.43 | 0.54 | -0.89 |
| edge | -0.0457 | -0.2083 | -0.1626 |
| sharpe | -1.28 | -1.79 | -0.51 |
| pf | 0.82 | 0.43 | -0.40 |

### PRE_MC_BLOCK_MODE — pre_mc_block_mode

**Best Finding:** pre_mc_block_mode: PnL +$70.00
- Improvement: $+70.00
- Confidence: 45%
- Parameters: `{"pre_mc_block_on_fail": 1, "pre_mc_min_cvar": -0.05}`

```
[PRE_MC_BLOCK_MODE] 파라미터 변경 제안:
  pre_mc_block_on_fail = 1
  pre_mc_min_cvar = -0.05
예상 효과: PnL $+70.00, WR +8.3%, R:R -0.78
신뢰도: 44.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 194 | -4441 |
| pnl | -71.47 | -1.47 | +70.00 |
| wr | 0.3653 | 0.4485 | +0.0832 |
| rr | 1.43 | 0.65 | -0.78 |
| edge | -0.0457 | -0.1568 | -0.1111 |
| sharpe | -1.28 | -1.14 | +0.14 |
| pf | 0.82 | 0.53 | -0.29 |

### DIRECTION_CONFIRM — direction_confirm

**Best Finding:** direction_confirm: PnL +$69.81
- Improvement: $+69.81
- Confidence: 72%
- Parameters: `{"dir_gate_confirm_ticks": 1, "dir_gate_confirm_ticks_chop": 4}`

```
[DIRECTION_CONFIRM] 파라미터 변경 제안:
  dir_gate_confirm_ticks = 1
  dir_gate_confirm_ticks_chop = 4
예상 효과: PnL $+69.81, WR -4.3%, R:R +0.65
신뢰도: 72.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 2307 | -2328 |
| pnl | -71.47 | -1.66 | +69.81 |
| wr | 0.3653 | 0.3225 | -0.0428 |
| rr | 1.43 | 2.08 | +0.65 |
| edge | -0.0457 | -0.0017 | +0.0440 |
| sharpe | -1.28 | -0.03 | +1.25 |
| pf | 0.82 | 0.99 | +0.17 |

### PRE_MC_SCALED_SIZE — pre_mc_scaled_size

**Best Finding:** pre_mc_scaled_size: PnL +$53.60
- Improvement: $+53.60
- Confidence: 30%
- Parameters: `{"pre_mc_size_scale": 0.25, "pre_mc_max_liq_prob": 0.03}`

```
[PRE_MC_SCALED_SIZE] 파라미터 변경 제안:
  pre_mc_size_scale = 0.25
  pre_mc_max_liq_prob = 0.03
예상 효과: PnL $+53.60, WR +0.0%, R:R +0.00
신뢰도: 30.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 4635 | +0 |
| pnl | -71.47 | -17.87 | +53.60 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.43 | 1.43 | +0.00 |
| edge | -0.0457 | -0.0457 | +0.0000 |
| sharpe | -1.28 | -1.28 | +0.00 |
| pf | 0.82 | 0.82 | +0.00 |

### MU_SIGN_FLIP — mu_sign_flip

**Best Finding:** mu_sign_flip: PnL +$47.45
- Improvement: $+47.45
- Confidence: 46%
- Parameters: `{"mu_sign_flip_min_age": 1800, "mu_sign_flip_confirm_ticks": 4}`

```
[MU_SIGN_FLIP] 파라미터 변경 제안:
  mu_sign_flip_min_age = 1800
  mu_sign_flip_confirm_ticks = 4
예상 효과: PnL $+47.45, WR +0.0%, R:R +0.19
신뢰도: 45.5%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4635 | 4635 | +0 |
| pnl | -71.47 | -24.02 | +47.45 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.43 | 1.63 | +0.19 |
| edge | -0.0457 | -0.0155 | +0.0302 |
| sharpe | -1.28 | -0.43 | +0.85 |
| pf | 0.82 | 0.94 | +0.11 |

## Regime Performance Breakdown

| Regime | N | PnL | WR | R:R | Edge |
|--------|---|-----|----|----|------|
| chop | 3853 | $-88.54 | 34.9% | 1.32 | -8.2% |
| bull | 478 | $36.82 | 47.7% | 1.89 | +13.1% |
| bear | 303 | $-19.77 | 39.6% | 0.95 | -11.8% |
| volatile | 1 | $0.02 | 100.0% | 18.80 | +95.0% |

## 🎯 Recommended Actions

1. **mc_hybrid_paths: PnL +$848.56** (ΔPnL: $+848.56, confidence: 80%)
   - `mc_hybrid_n_paths` = `16384`
   - `mc_hybrid_horizon_steps` = `300`

2. **tp_sl: PnL +$148.63** (ΔPnL: $+148.63, confidence: 80%)
   - `tp_pct` = `0.04`
   - `sl_pct` = `0.005`

3. **volatility_gate: PnL +$129.03** (ΔPnL: $+129.03, confidence: 83%)
   - `scope` = `chop_only`
   - `chop_min_sigma` = `0.35`
   - `chop_max_sigma` = `4.0`
   - `chop_max_vpin` = `0.65`
   - `chop_min_dir_conf` = `0.68`
   - `chop_min_abs_mu_alpha` = `40.0`
   - `chop_max_hold_sec` = `600`

4. **chop_guard: PnL +$127.61** (ΔPnL: $+127.61, confidence: 83%)
   - `chop_entry_floor_add` = `0.003`
   - `chop_entry_min_dir_conf` = `0.8`

5. **regime_side_block: PnL +$102.91** (ΔPnL: $+102.91, confidence: 77%)
   - `regime_side_block_list` = `bear_long,bull_short,chop_long`

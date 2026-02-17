# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-17 21:26
> Baseline: 4638 trades, PnL=$-73.36, WR=36.5%, R:R=1.43

## Pipeline Stage Impact Summary

### MC_HYBRID_PATHS — mc_hybrid_paths

**Best Finding:** mc_hybrid_paths: PnL +$849.07
- Improvement: $+849.07
- Confidence: 80%
- Parameters: `{"mc_hybrid_n_paths": 16384, "mc_hybrid_horizon_steps": 300}`

```
[MC_HYBRID_PATHS] 파라미터 변경 제안:
  mc_hybrid_n_paths = 16384
  mc_hybrid_horizon_steps = 300
예상 효과: PnL $+849.07, WR +0.0%, R:R +4.82
신뢰도: 80.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 4638 | +0 |
| pnl | -73.36 | 775.71 | +849.07 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.43 | 6.24 | +4.82 |
| edge | -0.0468 | 0.2270 | +0.2738 |
| sharpe | -1.31 | 5.19 | +6.51 |
| pf | 0.82 | 3.59 | +2.77 |

### TP_SL — TP/SL 타겟

**Best Finding:** tp_sl: PnL +$150.44
- Improvement: $+150.44
- Confidence: 80%
- Parameters: `{"tp_pct": 0.04, "sl_pct": 0.005}`

```
[TP_SL] 파라미터 변경 제안:
  tp_pct = 0.04
  sl_pct = 0.005
예상 효과: PnL $+150.44, WR -0.5%, R:R +1.79
신뢰도: 80.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 4638 | +0 |
| pnl | -73.36 | 77.08 | +150.44 |
| wr | 0.3650 | 0.3599 | -0.0051 |
| rr | 1.43 | 3.22 | +1.79 |
| edge | -0.0468 | 0.1228 | +0.1696 |
| sharpe | -1.31 | 3.71 | +5.02 |
| pf | 0.82 | 1.81 | +0.99 |

### CHOP_GUARD — chop_guard

**Best Finding:** chop_guard: PnL +$129.50
- Improvement: $+129.50
- Confidence: 84%
- Parameters: `{"chop_entry_floor_add": 0.003, "chop_entry_min_dir_conf": 0.8}`

```
[CHOP_GUARD] 파라미터 변경 제안:
  chop_entry_floor_add = 0.003
  chop_entry_min_dir_conf = 0.8
예상 효과: PnL $+129.50, WR +7.2%, R:R +0.47
신뢰도: 83.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 955 | -3683 |
| pnl | -73.36 | 56.14 | +129.50 |
| wr | 0.3650 | 0.4366 | +0.0716 |
| rr | 1.43 | 1.89 | +0.47 |
| edge | -0.0468 | 0.0911 | +0.1379 |
| sharpe | -1.31 | 1.33 | +2.64 |
| pf | 0.82 | 1.47 | +0.65 |

### VOLATILITY_GATE — volatility_gate

**Best Finding:** volatility_gate: PnL +$125.05
- Improvement: $+125.05
- Confidence: 79%
- Parameters: `{"scope": "chop_only", "chop_min_sigma": 0.2, "chop_max_sigma": 2.5, "chop_max_vpin": 0.65, "chop_min_dir_conf": 0.64, "chop_min_abs_mu_alpha": 10.0, "chop_max_hold_sec": 180}`

```
[VOLATILITY_GATE] 파라미터 변경 제안:
  scope = chop_only
  chop_min_sigma = 0.2
  chop_max_sigma = 2.5
  chop_max_vpin = 0.65
  chop_min_dir_conf = 0.64
  chop_min_abs_mu_alpha = 10.0
  chop_max_hold_sec = 180
예상 효과: PnL $+125.05, WR +4.2%, R:R +0.52
신뢰도: 78.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 1452 | -3186 |
| pnl | -73.36 | 51.69 | +125.05 |
| wr | 0.3650 | 0.4070 | +0.0420 |
| rr | 1.43 | 1.95 | +0.52 |
| edge | -0.0468 | 0.0675 | +0.1143 |
| sharpe | -1.31 | 1.12 | +2.44 |
| pf | 0.82 | 1.34 | +0.51 |

### REGIME_SIDE_BLOCK — regime_side_block

**Best Finding:** regime_side_block: PnL +$102.91
- Improvement: $+102.91
- Confidence: 77%
- Parameters: `{"regime_side_block_list": "bear_long,bull_short,chop_long"}`

```
[REGIME_SIDE_BLOCK] 파라미터 변경 제안:
  regime_side_block_list = bear_long,bull_short,chop_long
예상 효과: PnL $+102.91, WR +7.0%, R:R +0.14
신뢰도: 76.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 1865 | -2773 |
| pnl | -73.36 | 29.55 | +102.91 |
| wr | 0.3650 | 0.4354 | +0.0704 |
| rr | 1.43 | 1.56 | +0.14 |
| edge | -0.0468 | 0.0455 | +0.0923 |
| sharpe | -1.31 | 0.92 | +2.24 |
| pf | 0.82 | 1.21 | +0.39 |

### DIRECTION_GATE — direction_gate

**Best Finding:** direction_gate: PnL +$94.44
- Improvement: $+94.44
- Confidence: 78%
- Parameters: `{"dir_gate_min_conf": 0.7, "dir_gate_min_edge": 0.0}`

```
[DIRECTION_GATE] 파라미터 변경 제안:
  dir_gate_min_conf = 0.7
  dir_gate_min_edge = 0.0
예상 효과: PnL $+94.44, WR -0.4%, R:R +0.89
신뢰도: 77.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 843 | -3795 |
| pnl | -73.36 | 21.07 | +94.44 |
| wr | 0.3650 | 0.3606 | -0.0044 |
| rr | 1.43 | 2.32 | +0.89 |
| edge | -0.0468 | 0.0590 | +0.1058 |
| sharpe | -1.31 | 0.58 | +1.90 |
| pf | 0.82 | 1.31 | +0.49 |

### ENTRY_FILTER — 진입 필터

**Best Finding:** entry_filter: PnL +$91.39
- Improvement: $+91.39
- Confidence: 68%
- Parameters: `{"min_confidence": 0.55, "min_dir_conf": 0.65, "min_entry_quality": 0.5, "min_ev": 0.02}`

```
[ENTRY_FILTER] 파라미터 변경 제안:
  min_confidence = 0.55
  min_dir_conf = 0.65
  min_entry_quality = 0.5
  min_ev = 0.02
예상 효과: PnL $+91.39, WR +9.4%, R:R +0.27
신뢰도: 68.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 490 | -4148 |
| pnl | -73.36 | 18.03 | +91.39 |
| wr | 0.3650 | 0.4592 | +0.0942 |
| rr | 1.43 | 1.69 | +0.27 |
| edge | -0.0468 | 0.0879 | +0.1347 |
| sharpe | -1.31 | 0.76 | +2.07 |
| pf | 0.82 | 1.44 | +0.62 |

### HYBRID_LEVERAGE — hybrid_leverage

**Best Finding:** hybrid_leverage: PnL +$91.06
- Improvement: $+91.06
- Confidence: 68%
- Parameters: `{"hybrid_lev_sweep_min": 1.0, "hybrid_lev_sweep_max": 3.0, "hybrid_lev_ev_scale": 100}`

```
[HYBRID_LEVERAGE] 파라미터 변경 제안:
  hybrid_lev_sweep_min = 1.0
  hybrid_lev_sweep_max = 3.0
  hybrid_lev_ev_scale = 100
예상 효과: PnL $+91.06, WR +0.0%, R:R +0.41
신뢰도: 68.2%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 4638 | +0 |
| pnl | -73.36 | 17.70 | +91.06 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.43 | 1.84 | +0.41 |
| edge | -0.0468 | 0.0124 | +0.0592 |
| sharpe | -1.31 | 0.22 | +1.54 |
| pf | 0.82 | 1.06 | +0.23 |

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
| n | 4638 | 3824 | -814 |
| pnl | -73.36 | 15.98 | +89.34 |
| wr | 0.3650 | 0.5638 | +0.1988 |
| rr | 1.43 | 0.81 | -0.62 |
| edge | -0.0468 | 0.0117 | +0.0585 |
| sharpe | -1.31 | 0.29 | +1.61 |
| pf | 0.82 | 1.05 | +0.23 |

### LEVERAGE — 레버리지 결정

**Best Finding:** leverage: PnL +$84.67
- Improvement: $+84.67
- Confidence: 68%
- Parameters: `{"max_leverage": 50, "regime_max_bull": 20, "regime_max_chop": 3, "regime_max_bear": 5}`

```
[LEVERAGE] 파라미터 변경 제안:
  max_leverage = 50
  regime_max_bull = 20
  regime_max_chop = 3
  regime_max_bear = 5
예상 효과: PnL $+84.67, WR +0.0%, R:R +0.40
신뢰도: 67.9%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 4638 | +0 |
| pnl | -73.36 | 11.30 | +84.67 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.43 | 1.82 | +0.40 |
| edge | -0.0468 | 0.0109 | +0.0577 |
| sharpe | -1.31 | 0.24 | +1.56 |
| pf | 0.82 | 1.05 | +0.23 |

### VPIN_FILTER — VPIN 필터

**Best Finding:** vpin_filter: PnL +$80.37
- Improvement: $+80.37
- Confidence: 70%
- Parameters: `{"max_vpin": 0.3}`

```
[VPIN_FILTER] 파라미터 변경 제안:
  max_vpin = 0.3
예상 효과: PnL $+80.37, WR +3.9%, R:R +0.10
신뢰도: 69.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 2109 | -2529 |
| pnl | -73.36 | 7.01 | +80.37 |
| wr | 0.3650 | 0.4040 | +0.0390 |
| rr | 1.43 | 1.53 | +0.10 |
| edge | -0.0468 | 0.0084 | +0.0552 |
| sharpe | -1.31 | 0.15 | +1.47 |
| pf | 0.82 | 1.04 | +0.21 |

### PRE_MC_GATE — pre_mc_gate

**Best Finding:** pre_mc_gate: PnL +$72.50
- Improvement: $+72.50
- Confidence: 44%
- Parameters: `{"pre_mc_min_expected_pnl": 0.0, "pre_mc_max_liq_prob": 0.1}`

```
[PRE_MC_GATE] 파라미터 변경 제안:
  pre_mc_min_expected_pnl = 0.0
  pre_mc_max_liq_prob = 0.1
예상 효과: PnL $+72.50, WR +7.6%, R:R -0.89
신뢰도: 43.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 102 | -4536 |
| pnl | -73.36 | -0.86 | +72.50 |
| wr | 0.3650 | 0.4412 | +0.0762 |
| rr | 1.43 | 0.54 | -0.89 |
| edge | -0.0468 | -0.2083 | -0.1615 |
| sharpe | -1.31 | -1.79 | -0.47 |
| pf | 0.82 | 0.43 | -0.39 |

### PRE_MC_BLOCK_MODE — pre_mc_block_mode

**Best Finding:** pre_mc_block_mode: PnL +$71.89
- Improvement: $+71.89
- Confidence: 45%
- Parameters: `{"pre_mc_block_on_fail": 1, "pre_mc_min_cvar": -0.05}`

```
[PRE_MC_BLOCK_MODE] 파라미터 변경 제안:
  pre_mc_block_on_fail = 1
  pre_mc_min_cvar = -0.05
예상 효과: PnL $+71.89, WR +8.3%, R:R -0.78
신뢰도: 44.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 194 | -4444 |
| pnl | -73.36 | -1.47 | +71.89 |
| wr | 0.3650 | 0.4485 | +0.0835 |
| rr | 1.43 | 0.65 | -0.78 |
| edge | -0.0468 | -0.1568 | -0.1100 |
| sharpe | -1.31 | -1.14 | +0.17 |
| pf | 0.82 | 0.53 | -0.29 |

### DIRECTION_CONFIRM — direction_confirm

**Best Finding:** direction_confirm: PnL +$71.70
- Improvement: $+71.70
- Confidence: 72%
- Parameters: `{"dir_gate_confirm_ticks": 1, "dir_gate_confirm_ticks_chop": 4}`

```
[DIRECTION_CONFIRM] 파라미터 변경 제안:
  dir_gate_confirm_ticks = 1
  dir_gate_confirm_ticks_chop = 4
예상 효과: PnL $+71.70, WR -4.2%, R:R +0.66
신뢰도: 72.2%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 2307 | -2331 |
| pnl | -73.36 | -1.66 | +71.70 |
| wr | 0.3650 | 0.3225 | -0.0425 |
| rr | 1.43 | 2.08 | +0.66 |
| edge | -0.0468 | -0.0017 | +0.0451 |
| sharpe | -1.31 | -0.03 | +1.28 |
| pf | 0.82 | 0.99 | +0.17 |

### HYBRID_EXIT_TIMING — hybrid_exit_timing

**Best Finding:** hybrid_exit_timing: PnL +$71.63
- Improvement: $+71.63
- Confidence: 65%
- Parameters: `{"hybrid_exit_confirm_shock": 5, "hybrid_exit_confirm_normal": 8, "hybrid_exit_confirm_noise": 12}`

```
[HYBRID_EXIT_TIMING] 파라미터 변경 제안:
  hybrid_exit_confirm_shock = 5
  hybrid_exit_confirm_normal = 8
  hybrid_exit_confirm_noise = 12
예상 효과: PnL $+71.63, WR +0.0%, R:R +0.30
신뢰도: 65.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 4638 | +0 |
| pnl | -73.36 | -1.74 | +71.63 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.43 | 1.73 | +0.30 |
| edge | -0.0468 | -0.0011 | +0.0457 |
| sharpe | -1.31 | -0.03 | +1.28 |
| pf | 0.82 | 1.00 | +0.17 |

### MU_SIGN_FLIP — mu_sign_flip

**Best Finding:** mu_sign_flip: PnL +$47.45
- Improvement: $+47.45
- Confidence: 45%
- Parameters: `{"mu_sign_flip_min_age": 1800, "mu_sign_flip_confirm_ticks": 4}`

```
[MU_SIGN_FLIP] 파라미터 변경 제안:
  mu_sign_flip_min_age = 1800
  mu_sign_flip_confirm_ticks = 4
예상 효과: PnL $+47.45, WR +0.0%, R:R +0.19
신뢰도: 44.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 4638 | +0 |
| pnl | -73.36 | -25.91 | +47.45 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.43 | 1.62 | +0.19 |
| edge | -0.0468 | -0.0167 | +0.0301 |
| sharpe | -1.31 | -0.46 | +0.85 |
| pf | 0.82 | 0.93 | +0.11 |

### MTF_IMAGE_DL_GATE — mtf_image_dl_gate

**Best Finding:** mtf_image_dl_gate: PnL +$45.96
- Improvement: $+45.96
- Confidence: 43%
- Parameters: `{"dl_gate_mode": "chop_only", "dl_gate_quantile": 0.7}`

```
[MTF_IMAGE_DL_GATE] 파라미터 변경 제안:
  dl_gate_mode = chop_only
  dl_gate_quantile = 0.7
예상 효과: PnL $+45.96, WR -1.6%, R:R +0.27
신뢰도: 43.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 3606 | -1032 |
| pnl | -73.36 | -27.40 | +45.96 |
| wr | 0.3650 | 0.3489 | -0.0161 |
| rr | 1.43 | 1.70 | +0.27 |
| edge | -0.0468 | -0.0218 | +0.0250 |
| sharpe | -1.31 | -0.53 | +0.79 |
| pf | 0.82 | 0.91 | +0.09 |

## Regime Performance Breakdown

| Regime | N | PnL | WR | R:R | Edge |
|--------|---|-----|----|----|------|
| chop | 3856 | $-90.43 | 34.8% | 1.32 | -8.3% |
| bull | 478 | $36.82 | 47.7% | 1.89 | +13.1% |
| bear | 303 | $-19.77 | 39.6% | 0.95 | -11.8% |
| volatile | 1 | $0.02 | 100.0% | 18.80 | +95.0% |

## 🎯 Recommended Actions

1. **mc_hybrid_paths: PnL +$849.07** (ΔPnL: $+849.07, confidence: 80%)
   - `mc_hybrid_n_paths` = `16384`
   - `mc_hybrid_horizon_steps` = `300`

2. **tp_sl: PnL +$150.44** (ΔPnL: $+150.44, confidence: 80%)
   - `tp_pct` = `0.04`
   - `sl_pct` = `0.005`

3. **chop_guard: PnL +$129.50** (ΔPnL: $+129.50, confidence: 84%)
   - `chop_entry_floor_add` = `0.003`
   - `chop_entry_min_dir_conf` = `0.8`

4. **volatility_gate: PnL +$125.05** (ΔPnL: $+125.05, confidence: 79%)
   - `scope` = `chop_only`
   - `chop_min_sigma` = `0.2`
   - `chop_max_sigma` = `2.5`
   - `chop_max_vpin` = `0.65`
   - `chop_min_dir_conf` = `0.64`
   - `chop_min_abs_mu_alpha` = `10.0`
   - `chop_max_hold_sec` = `180`

5. **regime_side_block: PnL +$102.91** (ΔPnL: $+102.91, confidence: 77%)
   - `regime_side_block_list` = `bear_long,bull_short,chop_long`

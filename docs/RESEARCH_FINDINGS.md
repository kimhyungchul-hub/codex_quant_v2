# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-16 15:07
> Baseline: 4909 trades, PnL=$-59.16, WR=36.5%, R:R=1.48

## Pipeline Stage Impact Summary

### MC_HYBRID_PATHS — mc_hybrid_paths

**Best Finding:** mc_hybrid_paths: PnL +$845.29
- Improvement: $+845.29
- Confidence: 80%
- Parameters: `{"mc_hybrid_n_paths": 16384, "mc_hybrid_horizon_steps": 300}`

```
[MC_HYBRID_PATHS] 파라미터 변경 제안:
  mc_hybrid_n_paths = 16384
  mc_hybrid_horizon_steps = 300
예상 효과: PnL $+845.29, WR +0.0%, R:R +4.99
신뢰도: 80.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 4909 | +0 |
| pnl | -59.16 | 786.13 | +845.29 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.48 | 6.47 | +4.99 |
| edge | -0.0383 | 0.2311 | +0.2694 |
| sharpe | -1.07 | 5.26 | +6.33 |
| pf | 0.85 | 3.72 | +2.87 |

### TP_SL — TP/SL 타겟

**Best Finding:** tp_sl: PnL +$127.47
- Improvement: $+127.47
- Confidence: 80%
- Parameters: `{"tp_pct": 0.04, "sl_pct": 0.005}`

```
[TP_SL] 파라미터 변경 제안:
  tp_pct = 0.04
  sl_pct = 0.005
예상 효과: PnL $+127.47, WR -0.5%, R:R +1.46
신뢰도: 80.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 4909 | +0 |
| pnl | -59.16 | 68.31 | +127.47 |
| wr | 0.3650 | 0.3604 | -0.0046 |
| rr | 1.48 | 2.93 | +1.46 |
| edge | -0.0383 | 0.1062 | +0.1445 |
| sharpe | -1.07 | 3.23 | +4.30 |
| pf | 0.85 | 1.65 | +0.80 |

### CHOP_GUARD — chop_guard

**Best Finding:** chop_guard: PnL +$110.09
- Improvement: $+110.09
- Confidence: 81%
- Parameters: `{"chop_entry_floor_add": 0.003, "chop_entry_min_dir_conf": 0.8}`

```
[CHOP_GUARD] 파라미터 변경 제안:
  chop_entry_floor_add = 0.003
  chop_entry_min_dir_conf = 0.8
예상 효과: PnL $+110.09, WR +6.4%, R:R +0.40
신뢰도: 80.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 1045 | -3864 |
| pnl | -59.16 | 50.93 | +110.09 |
| wr | 0.3650 | 0.4287 | +0.0637 |
| rr | 1.48 | 1.88 | +0.40 |
| edge | -0.0383 | 0.0810 | +0.1193 |
| sharpe | -1.07 | 1.20 | +2.27 |
| pf | 0.85 | 1.41 | +0.56 |

### REGIME_SIDE_BLOCK — regime_side_block

**Best Finding:** regime_side_block: PnL +$93.67
- Improvement: $+93.67
- Confidence: 77%
- Parameters: `{"regime_side_block_list": "bear_long,bull_short,chop_long"}`

```
[REGIME_SIDE_BLOCK] 파라미터 변경 제안:
  regime_side_block_list = bear_long,bull_short,chop_long
예상 효과: PnL $+93.67, WR +6.9%, R:R +0.15
신뢰도: 76.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 1998 | -2911 |
| pnl | -59.16 | 34.51 | +93.67 |
| wr | 0.3650 | 0.4339 | +0.0689 |
| rr | 1.48 | 1.63 | +0.15 |
| edge | -0.0383 | 0.0532 | +0.0915 |
| sharpe | -1.07 | 1.08 | +2.15 |
| pf | 0.85 | 1.25 | +0.40 |

### DIRECTION_GATE — direction_gate

**Best Finding:** direction_gate: PnL +$83.04
- Improvement: $+83.04
- Confidence: 78%
- Parameters: `{"dir_gate_min_conf": 0.7, "dir_gate_min_edge": 0.0}`

```
[DIRECTION_GATE] 파라미터 변경 제안:
  dir_gate_min_conf = 0.7
  dir_gate_min_edge = 0.0
예상 효과: PnL $+83.04, WR -0.3%, R:R +0.92
신뢰도: 78.3%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 883 | -4026 |
| pnl | -59.16 | 23.89 | +83.04 |
| wr | 0.3650 | 0.3624 | -0.0026 |
| rr | 1.48 | 2.40 | +0.92 |
| edge | -0.0383 | 0.0679 | +0.1062 |
| sharpe | -1.07 | 0.67 | +1.74 |
| pf | 0.85 | 1.36 | +0.51 |

### HYBRID_LEVERAGE — hybrid_leverage

**Best Finding:** hybrid_leverage: PnL +$79.52
- Improvement: $+79.52
- Confidence: 67%
- Parameters: `{"hybrid_lev_sweep_min": 1.0, "hybrid_lev_sweep_max": 3.0, "hybrid_lev_ev_scale": 100}`

```
[HYBRID_LEVERAGE] 파라미터 변경 제안:
  hybrid_lev_sweep_min = 1.0
  hybrid_lev_sweep_max = 3.0
  hybrid_lev_ev_scale = 100
예상 효과: PnL $+79.52, WR +0.0%, R:R +0.37
신뢰도: 67.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 4909 | +0 |
| pnl | -59.16 | 20.36 | +79.52 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.48 | 1.85 | +0.37 |
| edge | -0.0383 | 0.0143 | +0.0526 |
| sharpe | -1.07 | 0.26 | +1.33 |
| pf | 0.85 | 1.06 | +0.21 |

### ENTRY_FILTER — 진입 필터

**Best Finding:** entry_filter: PnL +$77.83
- Improvement: $+77.83
- Confidence: 69%
- Parameters: `{"min_confidence": 0.55, "min_dir_conf": 0.65, "min_entry_quality": 0.2, "min_ev": 0.03}`

```
[ENTRY_FILTER] 파라미터 변경 제안:
  min_confidence = 0.55
  min_dir_conf = 0.65
  min_entry_quality = 0.2
  min_ev = 0.03
예상 효과: PnL $+77.83, WR +8.6%, R:R +0.37
신뢰도: 69.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 495 | -4414 |
| pnl | -59.16 | 18.68 | +77.83 |
| wr | 0.3650 | 0.4505 | +0.0855 |
| rr | 1.48 | 1.85 | +0.37 |
| edge | -0.0383 | 0.0994 | +0.1377 |
| sharpe | -1.07 | 0.80 | +1.87 |
| pf | 0.85 | 1.52 | +0.66 |

### DIRECTION — 방향 결정

**Best Finding:** direction: PnL +$75.15
- Improvement: $+75.15
- Confidence: 80%
- Parameters: `{"chop_prefer_short": true, "min_dir_conf_for_entry": 0.6, "mu_alpha_sign_override": true}`

```
[DIRECTION] 파라미터 변경 제안:
  chop_prefer_short = True
  min_dir_conf_for_entry = 0.6
  mu_alpha_sign_override = True
예상 효과: PnL $+75.15, WR +26.6%, R:R -0.83
신뢰도: 80.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 2153 | -2756 |
| pnl | -59.16 | 16.00 | +75.15 |
| wr | 0.3650 | 0.6307 | +0.2657 |
| rr | 1.48 | 0.65 | -0.83 |
| edge | -0.0383 | 0.0231 | +0.0614 |
| sharpe | -1.07 | 0.36 | +1.43 |
| pf | 0.85 | 1.10 | +0.25 |

### LEVERAGE — 레버리지 결정

**Best Finding:** leverage: PnL +$73.33
- Improvement: $+73.33
- Confidence: 67%
- Parameters: `{"max_leverage": 50, "regime_max_bull": 20, "regime_max_chop": 3, "regime_max_bear": 5}`

```
[LEVERAGE] 파라미터 변경 제안:
  max_leverage = 50
  regime_max_bull = 20
  regime_max_chop = 3
  regime_max_bear = 5
예상 효과: PnL $+73.33, WR +0.0%, R:R +0.37
신뢰도: 67.3%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 4909 | +0 |
| pnl | -59.16 | 14.17 | +73.33 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.48 | 1.84 | +0.37 |
| edge | -0.0383 | 0.0135 | +0.0518 |
| sharpe | -1.07 | 0.31 | +1.38 |
| pf | 0.85 | 1.06 | +0.21 |

### VPIN_FILTER — VPIN 필터

**Best Finding:** vpin_filter: PnL +$70.19
- Improvement: $+70.19
- Confidence: 69%
- Parameters: `{"max_vpin": 0.3}`

```
[VPIN_FILTER] 파라미터 변경 제안:
  max_vpin = 0.3
예상 효과: PnL $+70.19, WR +3.7%, R:R +0.09
신뢰도: 69.3%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 2136 | -2773 |
| pnl | -59.16 | 11.03 | +70.19 |
| wr | 0.3650 | 0.4022 | +0.0372 |
| rr | 1.48 | 1.57 | +0.09 |
| edge | -0.0383 | 0.0137 | +0.0520 |
| sharpe | -1.07 | 0.24 | +1.31 |
| pf | 0.85 | 1.06 | +0.21 |

### HYBRID_EXIT_TIMING — hybrid_exit_timing

**Best Finding:** hybrid_exit_timing: PnL +$69.36
- Improvement: $+69.36
- Confidence: 66%
- Parameters: `{"hybrid_exit_confirm_shock": 4, "hybrid_exit_confirm_normal": 8, "hybrid_exit_confirm_noise": 12}`

```
[HYBRID_EXIT_TIMING] 파라미터 변경 제안:
  hybrid_exit_confirm_shock = 4
  hybrid_exit_confirm_normal = 8
  hybrid_exit_confirm_noise = 12
예상 효과: PnL $+69.36, WR +0.0%, R:R +0.31
신뢰도: 66.2%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 4909 | +0 |
| pnl | -59.16 | 10.20 | +69.36 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.48 | 1.79 | +0.31 |
| edge | -0.0383 | 0.0065 | +0.0448 |
| sharpe | -1.07 | 0.18 | +1.25 |
| pf | 0.85 | 1.03 | +0.18 |

### PRE_MC_GATE — pre_mc_gate

**Best Finding:** pre_mc_gate: PnL +$58.29
- Improvement: $+58.29
- Confidence: 42%
- Parameters: `{"pre_mc_min_expected_pnl": 0.0, "pre_mc_max_liq_prob": 0.1}`

```
[PRE_MC_GATE] 파라미터 변경 제안:
  pre_mc_min_expected_pnl = 0.0
  pre_mc_max_liq_prob = 0.1
예상 효과: PnL $+58.29, WR +6.8%, R:R -0.92
신뢰도: 42.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 104 | -4805 |
| pnl | -59.16 | -0.86 | +58.29 |
| wr | 0.3650 | 0.4327 | +0.0677 |
| rr | 1.48 | 0.56 | -0.92 |
| edge | -0.0383 | -0.2089 | -0.1706 |
| sharpe | -1.07 | -1.79 | -0.72 |
| pf | 0.85 | 0.43 | -0.42 |

### PRE_MC_BLOCK_MODE — pre_mc_block_mode

**Best Finding:** pre_mc_block_mode: PnL +$57.69
- Improvement: $+57.69
- Confidence: 44%
- Parameters: `{"pre_mc_block_on_fail": 1, "pre_mc_min_cvar": -0.05}`

```
[PRE_MC_BLOCK_MODE] 파라미터 변경 제안:
  pre_mc_block_on_fail = 1
  pre_mc_min_cvar = -0.05
예상 효과: PnL $+57.69, WR +8.0%, R:R -0.82
신뢰도: 44.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 200 | -4709 |
| pnl | -59.16 | -1.46 | +57.69 |
| wr | 0.3650 | 0.4450 | +0.0800 |
| rr | 1.48 | 0.66 | -0.82 |
| edge | -0.0383 | -0.1560 | -0.1177 |
| sharpe | -1.07 | -1.14 | -0.07 |
| pf | 0.85 | 0.53 | -0.32 |

### DIRECTION_CONFIRM — direction_confirm

**Best Finding:** direction_confirm: PnL +$56.88
- Improvement: $+56.88
- Confidence: 69%
- Parameters: `{"dir_gate_confirm_ticks": 1, "dir_gate_confirm_ticks_chop": 4}`

```
[DIRECTION_CONFIRM] 파라미터 변경 제안:
  dir_gate_confirm_ticks = 1
  dir_gate_confirm_ticks_chop = 4
예상 효과: PnL $+56.88, WR -4.3%, R:R +0.61
신뢰도: 68.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 2449 | -2460 |
| pnl | -59.16 | -2.27 | +56.88 |
| wr | 0.3650 | 0.3218 | -0.0432 |
| rr | 1.48 | 2.09 | +0.61 |
| edge | -0.0383 | -0.0023 | +0.0360 |
| sharpe | -1.07 | -0.05 | +1.02 |
| pf | 0.85 | 0.99 | +0.14 |

### MU_SIGN_FLIP — mu_sign_flip

**Best Finding:** mu_sign_flip: PnL +$48.82
- Improvement: $+48.82
- Confidence: 53%
- Parameters: `{"mu_sign_flip_min_age": 1800, "mu_sign_flip_confirm_ticks": 4}`

```
[MU_SIGN_FLIP] 파라미터 변경 제안:
  mu_sign_flip_min_age = 1800
  mu_sign_flip_confirm_ticks = 4
예상 효과: PnL $+48.82, WR +0.0%, R:R +0.21
신뢰도: 53.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 4909 | +0 |
| pnl | -59.16 | -10.33 | +48.82 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.48 | 1.69 | +0.21 |
| edge | -0.0383 | -0.0068 | +0.0315 |
| sharpe | -1.07 | -0.19 | +0.88 |
| pf | 0.85 | 0.97 | +0.12 |

### PRE_MC_SCALED_SIZE — pre_mc_scaled_size

**Best Finding:** pre_mc_scaled_size: PnL +$44.37
- Improvement: $+44.37
- Confidence: 30%
- Parameters: `{"pre_mc_size_scale": 0.25, "pre_mc_max_liq_prob": 0.03}`

```
[PRE_MC_SCALED_SIZE] 파라미터 변경 제안:
  pre_mc_size_scale = 0.25
  pre_mc_max_liq_prob = 0.03
예상 효과: PnL $+44.37, WR +0.0%, R:R +0.00
신뢰도: 30.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 4909 | +0 |
| pnl | -59.16 | -14.79 | +44.37 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.48 | 1.48 | +0.00 |
| edge | -0.0383 | -0.0383 | +0.0000 |
| sharpe | -1.07 | -1.07 | +0.00 |
| pf | 0.85 | 0.85 | +0.00 |

### SYMBOL_QUALITY_TIME — symbol_quality_time

**Best Finding:** symbol_quality_time: PnL +$36.06
- Improvement: $+36.06
- Confidence: 31%
- Parameters: `{"sq_time_window_hours": 1, "sq_time_weight": 0.5}`

```
[SYMBOL_QUALITY_TIME] 파라미터 변경 제안:
  sq_time_window_hours = 1
  sq_time_weight = 0.5
예상 효과: PnL $+36.06, WR +0.0%, R:R +0.07
신뢰도: 31.2%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 4909 | +0 |
| pnl | -59.16 | -23.09 | +36.06 |
| wr | 0.3650 | 0.3650 | +0.0000 |
| rr | 1.48 | 1.55 | +0.07 |
| edge | -0.0383 | -0.0275 | +0.0108 |
| sharpe | -1.07 | -0.71 | +0.36 |
| pf | 0.85 | 0.89 | +0.04 |

### TOP_N — top_n

**Best Finding:** top_n: PnL +$34.59
- Improvement: $+34.59
- Confidence: 35%
- Parameters: `{"top_n_symbols": 4}`

```
[TOP_N] 파라미터 변경 제안:
  top_n_symbols = 4
예상 효과: PnL $+34.59, WR +4.7%, R:R -0.24
신뢰도: 35.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 1203 | -3706 |
| pnl | -59.16 | -24.57 | +34.59 |
| wr | 0.3650 | 0.4123 | +0.0473 |
| rr | 1.48 | 1.24 | -0.24 |
| edge | -0.0383 | -0.0337 | +0.0046 |
| sharpe | -1.07 | -0.63 | +0.44 |
| pf | 0.85 | 0.87 | +0.02 |

### UNIFIED_FLOOR — unified_floor

**Best Finding:** unified_floor: PnL +$27.39
- Improvement: $+27.39
- Confidence: 34%
- Parameters: `{"min_ev": 0.003}`

```
[UNIFIED_FLOOR] 파라미터 변경 제안:
  min_ev = 0.003
예상 효과: PnL $+27.39, WR +4.5%, R:R -0.18
신뢰도: 33.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 3261 | -1648 |
| pnl | -59.16 | -31.76 | +27.40 |
| wr | 0.3650 | 0.4097 | +0.0447 |
| rr | 1.48 | 1.30 | -0.18 |
| edge | -0.0383 | -0.0260 | +0.0123 |
| sharpe | -1.07 | -0.59 | +0.48 |
| pf | 0.85 | 0.90 | +0.05 |

## Regime Performance Breakdown

| Regime | N | PnL | WR | R:R | Edge |
|--------|---|-----|----|----|------|
| chop | 4039 | $-73.54 | 35.0% | 1.38 | -7.0% |
| bull | 558 | $39.43 | 47.1% | 1.93 | +13.0% |
| bear | 311 | $-25.07 | 37.3% | 0.89 | -15.6% |
| volatile | 1 | $0.02 | 100.0% | 18.80 | +95.0% |

## 🎯 Recommended Actions

1. **mc_hybrid_paths: PnL +$845.29** (ΔPnL: $+845.29, confidence: 80%)
   - `mc_hybrid_n_paths` = `16384`
   - `mc_hybrid_horizon_steps` = `300`

2. **tp_sl: PnL +$127.47** (ΔPnL: $+127.47, confidence: 80%)
   - `tp_pct` = `0.04`
   - `sl_pct` = `0.005`

3. **chop_guard: PnL +$110.09** (ΔPnL: $+110.09, confidence: 81%)
   - `chop_entry_floor_add` = `0.003`
   - `chop_entry_min_dir_conf` = `0.8`

4. **regime_side_block: PnL +$93.67** (ΔPnL: $+93.67, confidence: 77%)
   - `regime_side_block_list` = `bear_long,bull_short,chop_long`

5. **direction_gate: PnL +$83.04** (ΔPnL: $+83.04, confidence: 78%)
   - `dir_gate_min_conf` = `0.7`
   - `dir_gate_min_edge` = `0.0`

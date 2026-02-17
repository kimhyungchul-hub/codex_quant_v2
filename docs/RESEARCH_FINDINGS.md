# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-17 20:06
> Baseline: 4634 trades, PnL=$-70.00, WR=36.5%, R:R=1.44

## Pipeline Stage Impact Summary

### MC_HYBRID_PATHS — mc_hybrid_paths

**Best Finding:** mc_hybrid_paths: PnL +$848.17
- Improvement: $+848.17
- Confidence: 80%
- Parameters: `{"mc_hybrid_n_paths": 16384, "mc_hybrid_horizon_steps": 300}`

```
[MC_HYBRID_PATHS] 파라미터 변경 제안:
  mc_hybrid_n_paths = 16384
  mc_hybrid_horizon_steps = 300
예상 효과: PnL $+848.17, WR +0.0%, R:R +4.85
신뢰도: 80.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 4634 | +0 |
| pnl | -70.00 | 778.16 | +848.17 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.44 | 6.29 | +4.85 |
| edge | -0.0449 | 0.2281 | +0.2730 |
| sharpe | -1.26 | 5.21 | +6.46 |
| pf | 0.83 | 3.62 | +2.79 |

### TP_SL — TP/SL 타겟

**Best Finding:** tp_sl: PnL +$147.18
- Improvement: $+147.18
- Confidence: 80%
- Parameters: `{"tp_pct": 0.04, "sl_pct": 0.005}`

```
[TP_SL] 파라미터 변경 제안:
  tp_pct = 0.04
  sl_pct = 0.005
예상 효과: PnL $+147.18, WR -0.5%, R:R +1.78
신뢰도: 80.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 4634 | +0 |
| pnl | -70.00 | 77.18 | +147.18 |
| wr | 0.3653 | 0.3602 | -0.0051 |
| rr | 1.44 | 3.22 | +1.78 |
| edge | -0.0449 | 0.1230 | +0.1679 |
| sharpe | -1.26 | 3.71 | +4.97 |
| pf | 0.83 | 1.81 | +0.98 |

### CHOP_GUARD — chop_guard

**Best Finding:** chop_guard: PnL +$126.14
- Improvement: $+126.14
- Confidence: 83%
- Parameters: `{"chop_entry_floor_add": 0.003, "chop_entry_min_dir_conf": 0.8}`

```
[CHOP_GUARD] 파라미터 변경 제안:
  chop_entry_floor_add = 0.003
  chop_entry_min_dir_conf = 0.8
예상 효과: PnL $+126.14, WR +7.1%, R:R +0.46
신뢰도: 83.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 955 | -3679 |
| pnl | -70.00 | 56.14 | +126.14 |
| wr | 0.3653 | 0.4366 | +0.0713 |
| rr | 1.44 | 1.89 | +0.46 |
| edge | -0.0449 | 0.0911 | +0.1360 |
| sharpe | -1.26 | 1.33 | +2.58 |
| pf | 0.83 | 1.47 | +0.64 |

### REGIME_SIDE_BLOCK — regime_side_block

**Best Finding:** regime_side_block: PnL +$102.91
- Improvement: $+102.91
- Confidence: 77%
- Parameters: `{"regime_side_block_list": "bear_long,bull_short,chop_long"}`

```
[REGIME_SIDE_BLOCK] 파라미터 변경 제안:
  regime_side_block_list = bear_long,bull_short,chop_long
예상 효과: PnL $+102.91, WR +7.1%, R:R +0.16
신뢰도: 77.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 1861 | -2773 |
| pnl | -70.00 | 32.91 | +102.91 |
| wr | 0.3653 | 0.4363 | +0.0710 |
| rr | 1.44 | 1.60 | +0.16 |
| edge | -0.0449 | 0.0512 | +0.0961 |
| sharpe | -1.26 | 1.03 | +2.29 |
| pf | 0.83 | 1.24 | +0.41 |

### DIRECTION_GATE — direction_gate

**Best Finding:** direction_gate: PnL +$91.08
- Improvement: $+91.08
- Confidence: 78%
- Parameters: `{"dir_gate_min_conf": 0.7, "dir_gate_min_edge": 0.0}`

```
[DIRECTION_GATE] 파라미터 변경 제안:
  dir_gate_min_conf = 0.7
  dir_gate_min_edge = 0.0
예상 효과: PnL $+91.08, WR -0.5%, R:R +0.88
신뢰도: 77.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 843 | -3791 |
| pnl | -70.00 | 21.07 | +91.08 |
| wr | 0.3653 | 0.3606 | -0.0047 |
| rr | 1.44 | 2.32 | +0.88 |
| edge | -0.0449 | 0.0590 | +0.1039 |
| sharpe | -1.26 | 0.58 | +1.84 |
| pf | 0.83 | 1.31 | +0.48 |

### HYBRID_LEVERAGE — hybrid_leverage

**Best Finding:** hybrid_leverage: PnL +$89.39
- Improvement: $+89.39
- Confidence: 68%
- Parameters: `{"hybrid_lev_sweep_min": 1.0, "hybrid_lev_sweep_max": 3.0, "hybrid_lev_ev_scale": 100}`

```
[HYBRID_LEVERAGE] 파라미터 변경 제안:
  hybrid_lev_sweep_min = 1.0
  hybrid_lev_sweep_max = 3.0
  hybrid_lev_ev_scale = 100
예상 효과: PnL $+89.39, WR +0.0%, R:R +0.41
신뢰도: 68.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 4634 | +0 |
| pnl | -70.00 | 19.38 | +89.39 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.44 | 1.84 | +0.41 |
| edge | -0.0449 | 0.0136 | +0.0585 |
| sharpe | -1.26 | 0.24 | +1.50 |
| pf | 0.83 | 1.06 | +0.23 |

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
| n | 4634 | 3820 | -814 |
| pnl | -70.00 | 19.34 | +89.34 |
| wr | 0.3653 | 0.5644 | +0.1991 |
| rr | 1.44 | 0.82 | -0.62 |
| edge | -0.0449 | 0.0143 | +0.0592 |
| sharpe | -1.26 | 0.35 | +1.61 |
| pf | 0.83 | 1.06 | +0.23 |

### ENTRY_FILTER — 진입 필터

**Best Finding:** entry_filter: PnL +$88.34
- Improvement: $+88.34
- Confidence: 68%
- Parameters: `{"min_confidence": 0.55, "min_dir_conf": 0.65, "min_entry_quality": 0.3, "min_ev": 0.02}`

```
[ENTRY_FILTER] 파라미터 변경 제안:
  min_confidence = 0.55
  min_dir_conf = 0.65
  min_entry_quality = 0.3
  min_ev = 0.02
예상 효과: PnL $+88.34, WR +9.2%, R:R +0.28
신뢰도: 68.3%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 494 | -4140 |
| pnl | -70.00 | 18.34 | +88.34 |
| wr | 0.3653 | 0.4575 | +0.0922 |
| rr | 1.44 | 1.71 | +0.28 |
| edge | -0.0449 | 0.0889 | +0.1338 |
| sharpe | -1.26 | 0.77 | +2.03 |
| pf | 0.83 | 1.44 | +0.62 |

### LEVERAGE — 레버리지 결정

**Best Finding:** leverage: PnL +$82.31
- Improvement: $+82.31
- Confidence: 68%
- Parameters: `{"max_leverage": 50, "regime_max_bull": 20, "regime_max_chop": 3, "regime_max_bear": 5}`

```
[LEVERAGE] 파라미터 변경 제안:
  max_leverage = 50
  regime_max_bull = 20
  regime_max_chop = 3
  regime_max_bear = 5
예상 효과: PnL $+82.31, WR +0.0%, R:R +0.39
신뢰도: 67.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 4634 | +0 |
| pnl | -70.00 | 12.31 | +82.31 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.44 | 1.83 | +0.39 |
| edge | -0.0449 | 0.0119 | +0.0568 |
| sharpe | -1.26 | 0.27 | +1.52 |
| pf | 0.83 | 1.05 | +0.23 |

### VPIN_FILTER — VPIN 필터

**Best Finding:** vpin_filter: PnL +$77.01
- Improvement: $+77.01
- Confidence: 70%
- Parameters: `{"max_vpin": 0.3}`

```
[VPIN_FILTER] 파라미터 변경 제안:
  max_vpin = 0.3
예상 효과: PnL $+77.01, WR +3.9%, R:R +0.09
신뢰도: 69.5%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 2109 | -2525 |
| pnl | -70.00 | 7.01 | +77.01 |
| wr | 0.3653 | 0.4040 | +0.0387 |
| rr | 1.44 | 1.53 | +0.09 |
| edge | -0.0449 | 0.0084 | +0.0533 |
| sharpe | -1.26 | 0.15 | +1.41 |
| pf | 0.83 | 1.04 | +0.21 |

### HYBRID_EXIT_TIMING — hybrid_exit_timing

**Best Finding:** hybrid_exit_timing: PnL +$71.22
- Improvement: $+71.22
- Confidence: 66%
- Parameters: `{"hybrid_exit_confirm_shock": 5, "hybrid_exit_confirm_normal": 8, "hybrid_exit_confirm_noise": 12}`

```
[HYBRID_EXIT_TIMING] 파라미터 변경 제안:
  hybrid_exit_confirm_shock = 5
  hybrid_exit_confirm_normal = 8
  hybrid_exit_confirm_noise = 12
예상 효과: PnL $+71.22, WR +0.0%, R:R +0.31
신뢰도: 66.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 4634 | +0 |
| pnl | -70.00 | 1.22 | +71.22 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.44 | 1.74 | +0.31 |
| edge | -0.0449 | 0.0008 | +0.0457 |
| sharpe | -1.26 | 0.02 | +1.28 |
| pf | 0.83 | 1.00 | +0.18 |

### PRE_MC_GATE — pre_mc_gate

**Best Finding:** pre_mc_gate: PnL +$69.14
- Improvement: $+69.14
- Confidence: 44%
- Parameters: `{"pre_mc_min_expected_pnl": 0.0, "pre_mc_max_liq_prob": 0.1}`

```
[PRE_MC_GATE] 파라미터 변경 제안:
  pre_mc_min_expected_pnl = 0.0
  pre_mc_max_liq_prob = 0.1
예상 효과: PnL $+69.14, WR +7.6%, R:R -0.90
신뢰도: 43.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 102 | -4532 |
| pnl | -70.00 | -0.86 | +69.14 |
| wr | 0.3653 | 0.4412 | +0.0759 |
| rr | 1.44 | 0.54 | -0.90 |
| edge | -0.0449 | -0.2083 | -0.1634 |
| sharpe | -1.26 | -1.79 | -0.53 |
| pf | 0.83 | 0.43 | -0.40 |

### PRE_MC_BLOCK_MODE — pre_mc_block_mode

**Best Finding:** pre_mc_block_mode: PnL +$68.53
- Improvement: $+68.53
- Confidence: 45%
- Parameters: `{"pre_mc_block_on_fail": 1, "pre_mc_min_cvar": -0.05}`

```
[PRE_MC_BLOCK_MODE] 파라미터 변경 제안:
  pre_mc_block_on_fail = 1
  pre_mc_min_cvar = -0.05
예상 효과: PnL $+68.53, WR +8.3%, R:R -0.79
신뢰도: 44.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 194 | -4440 |
| pnl | -70.00 | -1.47 | +68.53 |
| wr | 0.3653 | 0.4485 | +0.0832 |
| rr | 1.44 | 0.65 | -0.79 |
| edge | -0.0449 | -0.1568 | -0.1119 |
| sharpe | -1.26 | -1.14 | +0.11 |
| pf | 0.83 | 0.53 | -0.30 |

### DIRECTION_CONFIRM — direction_confirm

**Best Finding:** direction_confirm: PnL +$68.34
- Improvement: $+68.34
- Confidence: 72%
- Parameters: `{"dir_gate_confirm_ticks": 1, "dir_gate_confirm_ticks_chop": 4}`

```
[DIRECTION_CONFIRM] 파라미터 변경 제안:
  dir_gate_confirm_ticks = 1
  dir_gate_confirm_ticks_chop = 4
예상 효과: PnL $+68.34, WR -4.3%, R:R +0.65
신뢰도: 72.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 2307 | -2327 |
| pnl | -70.00 | -1.66 | +68.34 |
| wr | 0.3653 | 0.3225 | -0.0428 |
| rr | 1.44 | 2.08 | +0.65 |
| edge | -0.0449 | -0.0017 | +0.0432 |
| sharpe | -1.26 | -0.03 | +1.22 |
| pf | 0.83 | 0.99 | +0.16 |

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
신뢰도: 46.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 4634 | +0 |
| pnl | -70.00 | -22.55 | +47.45 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.44 | 1.63 | +0.19 |
| edge | -0.0449 | -0.0146 | +0.0303 |
| sharpe | -1.26 | -0.40 | +0.85 |
| pf | 0.83 | 0.94 | +0.11 |

### SYMBOL_QUALITY_TIME — symbol_quality_time

**Best Finding:** symbol_quality_time: PnL +$41.55
- Improvement: $+41.55
- Confidence: 31%
- Parameters: `{"sq_time_window_hours": 1, "sq_time_weight": 0.5}`

```
[SYMBOL_QUALITY_TIME] 파라미터 변경 제안:
  sq_time_window_hours = 1
  sq_time_weight = 0.5
예상 효과: PnL $+41.55, WR +0.0%, R:R +0.07
신뢰도: 31.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4634 | 4634 | +0 |
| pnl | -70.00 | -28.45 | +41.55 |
| wr | 0.3653 | 0.3653 | +0.0000 |
| rr | 1.44 | 1.51 | +0.07 |
| edge | -0.0449 | -0.0333 | +0.0116 |
| sharpe | -1.26 | -0.86 | +0.39 |
| pf | 0.83 | 0.87 | +0.04 |

## Regime Performance Breakdown

| Regime | N | PnL | WR | R:R | Edge |
|--------|---|-----|----|----|------|
| chop | 3852 | $-87.08 | 34.9% | 1.33 | -8.0% |
| bull | 478 | $36.82 | 47.7% | 1.89 | +13.1% |
| bear | 303 | $-19.77 | 39.6% | 0.95 | -11.8% |
| volatile | 1 | $0.02 | 100.0% | 18.80 | +95.0% |

## 🎯 Recommended Actions

1. **mc_hybrid_paths: PnL +$848.17** (ΔPnL: $+848.17, confidence: 80%)
   - `mc_hybrid_n_paths` = `16384`
   - `mc_hybrid_horizon_steps` = `300`

2. **tp_sl: PnL +$147.18** (ΔPnL: $+147.18, confidence: 80%)
   - `tp_pct` = `0.04`
   - `sl_pct` = `0.005`

3. **chop_guard: PnL +$126.14** (ΔPnL: $+126.14, confidence: 83%)
   - `chop_entry_floor_add` = `0.003`
   - `chop_entry_min_dir_conf` = `0.8`

4. **regime_side_block: PnL +$102.91** (ΔPnL: $+102.91, confidence: 77%)
   - `regime_side_block_list` = `bear_long,bull_short,chop_long`

5. **direction_gate: PnL +$91.08** (ΔPnL: $+91.08, confidence: 78%)
   - `dir_gate_min_conf` = `0.7`
   - `dir_gate_min_edge` = `0.0`

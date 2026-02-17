# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-18 02:29
> Baseline: 4638 trades, PnL=$-369.07, WR=19.3%, R:R=1.78

## Pipeline Stage Impact Summary

### VPIN_FILTER — VPIN 필터

**Best Finding:** vpin_filter: OOS-adjusted PnL +$276.06
- Improvement: $+276.06
- Confidence: 65%
- Parameters: `{"max_vpin": 0.3}`

```
[VPIN_FILTER] 파라미터 변경 제안:
  max_vpin = 0.3
예상 효과: PnL $+276.06, WR +4.6%, R:R +0.30
신뢰도: 65.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 2109 | -2529 |
| pnl | -369.07 | -93.02 | +276.06 |
| wr | 0.1932 | 0.2390 | +0.0458 |
| rr | 1.78 | 2.08 | +0.30 |
| edge | -0.1660 | -0.0858 | +0.0802 |
| sharpe | -6.53 | -2.01 | +4.53 |
| pf | 0.43 | 0.65 | +0.23 |

### VOLATILITY_GATE — volatility_gate

**Best Finding:** volatility_gate: OOS-adjusted PnL +$274.38
- Improvement: $+274.38
- Confidence: 65%
- Parameters: `{"scope": "all_regimes", "chop_min_sigma": 0.5, "chop_max_sigma": 1.8, "chop_max_vpin": 0.65, "chop_min_dir_conf": 0.56, "chop_min_abs_mu_alpha": 10.0, "chop_max_hold_sec": 300}`

```
[VOLATILITY_GATE] 파라미터 변경 제안:
  scope = all_regimes
  chop_min_sigma = 0.5
  chop_max_sigma = 1.8
  chop_max_vpin = 0.65
  chop_min_dir_conf = 0.56
  chop_min_abs_mu_alpha = 10.0
  chop_max_hold_sec = 300
예상 효과: PnL $+331.32, WR +2.9%, R:R +0.86
신뢰도: 65.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 1102 | -3536 |
| pnl | -369.07 | -37.76 | +331.32 |
| wr | 0.1932 | 0.2223 | +0.0291 |
| rr | 1.78 | 2.65 | +0.86 |
| edge | -0.1660 | -0.0520 | +0.1140 |
| sharpe | -6.53 | -0.89 | +5.64 |
| pf | 0.43 | 0.76 | +0.33 |

### REGIME_SIDE_BLOCK — regime_side_block

**Best Finding:** regime_side_block: OOS-adjusted PnL +$261.09
- Improvement: $+261.09
- Confidence: 58%
- Parameters: `{"regime_side_block_list": "bear_long,bull_short,chop_long"}`

```
[REGIME_SIDE_BLOCK] 파라미터 변경 제안:
  regime_side_block_list = bear_long,bull_short,chop_long
예상 효과: PnL $+277.70, WR +4.2%, R:R +0.19
신뢰도: 58.5%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 1865 | -2773 |
| pnl | -369.07 | -91.38 | +277.70 |
| wr | 0.1932 | 0.2349 | +0.0417 |
| rr | 1.78 | 1.97 | +0.19 |
| edge | -0.1660 | -0.1016 | +0.0644 |
| sharpe | -6.53 | -2.83 | +3.70 |
| pf | 0.43 | 0.61 | +0.18 |

### CHOP_GUARD — chop_guard

**Best Finding:** chop_guard: OOS-adjusted PnL +$244.45
- Improvement: $+244.45
- Confidence: 59%
- Parameters: `{"chop_entry_floor_add": 0.003, "chop_entry_min_dir_conf": 0.8}`

```
[CHOP_GUARD] 파라미터 변경 제안:
  chop_entry_floor_add = 0.003
  chop_entry_min_dir_conf = 0.8
예상 효과: PnL $+324.92, WR +4.3%, R:R +0.69
신뢰도: 58.5%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 955 | -3683 |
| pnl | -369.07 | -44.16 | +324.92 |
| wr | 0.1932 | 0.2366 | +0.0434 |
| rr | 1.78 | 2.48 | +0.69 |
| edge | -0.1660 | -0.0508 | +0.1152 |
| sharpe | -6.53 | -1.04 | +5.50 |
| pf | 0.43 | 0.77 | +0.34 |

### DIRECTION_GATE — direction_gate

**Best Finding:** direction_gate: OOS-adjusted PnL +$232.79
- Improvement: $+232.79
- Confidence: 53%
- Parameters: `{"dir_gate_min_conf": 0.65, "dir_gate_min_edge": 0.1, "dir_gate_min_side_prob": 0.4975}`

```
[DIRECTION_GATE] 파라미터 변경 제안:
  dir_gate_min_conf = 0.65
  dir_gate_min_edge = 0.1
  dir_gate_min_side_prob = 0.4975
예상 효과: PnL $+337.12, WR -0.1%, R:R +1.43
신뢰도: 52.9%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 1003 | -3635 |
| pnl | -369.07 | -31.95 | +337.12 |
| wr | 0.1932 | 0.1924 | -0.0008 |
| rr | 1.78 | 3.22 | +1.43 |
| edge | -0.1660 | -0.0446 | +0.1214 |
| sharpe | -6.53 | -0.77 | +5.76 |
| pf | 0.43 | 0.77 | +0.34 |

### DIRECTION — 방향 결정

**Best Finding:** direction: OOS-adjusted PnL +$179.58
- Improvement: $+179.58
- Confidence: 44%
- Parameters: `{"chop_prefer_short": false, "min_dir_conf_for_entry": 0.65, "mu_alpha_sign_override": true}`

```
[DIRECTION] 파라미터 변경 제안:
  chop_prefer_short = False
  min_dir_conf_for_entry = 0.65
  mu_alpha_sign_override = True
예상 효과: PnL $+293.15, WR -3.5%, R:R +1.39
신뢰도: 44.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 1359 | -3279 |
| pnl | -369.07 | -75.92 | +293.15 |
| wr | 0.1932 | 0.1582 | -0.0350 |
| rr | 1.78 | 3.18 | +1.39 |
| edge | -0.1660 | -0.0811 | +0.0849 |
| sharpe | -6.53 | -1.80 | +4.74 |
| pf | 0.43 | 0.60 | +0.17 |

### DIRECTION_CONFIRM — direction_confirm

**Best Finding:** direction_confirm: OOS-adjusted PnL +$123.00
- Improvement: $+123.00
- Confidence: 31%
- Parameters: `{"dir_gate_confirm_ticks": 4, "dir_gate_confirm_ticks_chop": 4}`

```
[DIRECTION_CONFIRM] 파라미터 변경 제안:
  dir_gate_confirm_ticks = 4
  dir_gate_confirm_ticks_chop = 4
예상 효과: PnL $+220.20, WR -4.5%, R:R +0.77
신뢰도: 31.2%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 1884 | -2754 |
| pnl | -369.07 | -148.88 | +220.20 |
| wr | 0.1932 | 0.1481 | -0.0451 |
| rr | 1.78 | 2.56 | +0.77 |
| edge | -0.1660 | -0.1330 | +0.0330 |
| sharpe | -6.53 | -3.36 | +3.18 |
| pf | 0.43 | 0.44 | +0.02 |

### CAPITAL_ALLOCATION — 자본 분배

**Best Finding:** capital_allocation: OOS-adjusted PnL +$118.48
- Improvement: $+118.48
- Confidence: 33%
- Parameters: `{"notional_hard_cap": 50, "max_pos_frac": 0.15}`

```
[CAPITAL_ALLOCATION] 파라미터 변경 제안:
  notional_hard_cap = 50
  max_pos_frac = 0.15
예상 효과: PnL $+158.89, WR +0.0%, R:R +0.34
신뢰도: 32.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 4638 | +0 |
| pnl | -369.07 | -210.19 | +158.89 |
| wr | 0.1932 | 0.1932 | +0.0000 |
| rr | 1.78 | 2.13 | +0.34 |
| edge | -0.1660 | -0.1265 | +0.0395 |
| sharpe | -6.53 | -4.35 | +2.18 |
| pf | 0.43 | 0.51 | +0.08 |

### LEVERAGE — 레버리지 결정

**Best Finding:** leverage: OOS-adjusted PnL +$108.95
- Improvement: $+108.95
- Confidence: 43%
- Parameters: `{"max_leverage": 20, "regime_max_bull": 20, "regime_max_chop": 3, "regime_max_bear": 12}`

```
[LEVERAGE] 파라미터 변경 제안:
  max_leverage = 20
  regime_max_bull = 20
  regime_max_chop = 3
  regime_max_bear = 12
예상 효과: PnL $+122.17, WR -4.5%, R:R +0.83
신뢰도: 43.3%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4638 | 4638 | +0 |
| pnl | -369.07 | -246.91 | +122.17 |
| wr | 0.1932 | 0.1477 | -0.0455 |
| rr | 1.78 | 2.62 | +0.83 |
| edge | -0.1660 | -0.1287 | +0.0373 |
| sharpe | -6.53 | -5.18 | +1.36 |
| pf | 0.43 | 0.45 | +0.03 |

## Regime Performance Breakdown

| Regime | N | PnL | WR | R:R | Edge |
|--------|---|-----|----|----|------|
| chop | 3856 | $-293.38 | 18.7% | 1.66 | -18.8% |
| bull | 478 | $-24.90 | 27.0% | 1.98 | -6.6% |
| bear | 303 | $-50.72 | 14.8% | 1.87 | -20.0% |
| volatile | 1 | $-0.08 | 0.0% | 0.00 | -100.0% |

## 🎯 Recommended Actions

1. **vpin_filter: OOS-adjusted PnL +$276.06** (ΔPnL: $+276.06, confidence: 65%)
   - `max_vpin` = `0.3`

2. **volatility_gate: OOS-adjusted PnL +$274.38** (ΔPnL: $+274.38, confidence: 65%)
   - `scope` = `all_regimes`
   - `chop_min_sigma` = `0.5`
   - `chop_max_sigma` = `1.8`
   - `chop_max_vpin` = `0.65`
   - `chop_min_dir_conf` = `0.56`
   - `chop_min_abs_mu_alpha` = `10.0`
   - `chop_max_hold_sec` = `300`

3. **volatility_gate: OOS-adjusted PnL +$263.42** (ΔPnL: $+263.42, confidence: 60%)
   - `scope` = `all_regimes`
   - `chop_min_sigma` = `0.1`
   - `chop_max_sigma` = `1.2`
   - `chop_max_vpin` = `0.65`
   - `chop_min_dir_conf` = `0.6`
   - `chop_min_abs_mu_alpha` = `5.0`
   - `chop_max_hold_sec` = `300`

4. **regime_side_block: OOS-adjusted PnL +$261.09** (ΔPnL: $+261.09, confidence: 58%)
   - `regime_side_block_list` = `bear_long,bull_short,chop_long`

5. **chop_guard: OOS-adjusted PnL +$244.45** (ΔPnL: $+244.45, confidence: 59%)
   - `chop_entry_floor_add` = `0.003`
   - `chop_entry_min_dir_conf` = `0.8`

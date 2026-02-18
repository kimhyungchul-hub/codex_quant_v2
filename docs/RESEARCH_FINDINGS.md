# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-18 11:48
> Baseline: 4642 trades, PnL=$-368.57, WR=19.3%, R:R=1.79

## Pipeline Stage Impact Summary

### VPIN_FILTER — VPIN 필터

**Best Finding:** vpin_filter: OOS-adjusted PnL +$276.06
- Improvement: $+276.06
- Confidence: 65%
- Parameters: `{"max_vpin": 0.3}`

```
[VPIN_FILTER] 파라미터 변경 제안:
  max_vpin = 0.3
예상 효과: PnL $+276.06 (OOS 보정 $+276.06), WR +4.6%, R:R +0.30
OOS 검증: pass=True rate=75% trainΔ=+62.25 testΔ=+63.65 penalty=1.00
신뢰도: 65.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4642 | 2113 | -2529 |
| pnl | -368.57 | -92.51 | +276.06 |
| wr | 0.1932 | 0.2390 | +0.0458 |
| rr | 1.79 | 2.09 | +0.30 |
| edge | -0.1650 | -0.0846 | +0.0804 |
| sharpe | -6.52 | -2.00 | +4.52 |
| pf | 0.43 | 0.66 | +0.23 |

### REGIME_SIDE_BLOCK — regime_side_block

**Best Finding:** regime_side_block: OOS-adjusted PnL +$258.25
- Improvement: $+258.25
- Confidence: 58%
- Parameters: `{"regime_side_block_list": "bear_long,bull_short,chop_long"}`

```
[REGIME_SIDE_BLOCK] 파라미터 변경 제안:
  regime_side_block_list = bear_long,bull_short,chop_long
예상 효과: PnL $+276.32 (OOS 보정 $+258.25), WR +4.1%, R:R +0.18
OOS 검증: pass=True rate=75% trainΔ=+70.70 testΔ=+66.07 penalty=0.93
신뢰도: 57.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4642 | 1867 | -2775 |
| pnl | -368.57 | -92.25 | +276.32 |
| wr | 0.1932 | 0.2346 | +0.0414 |
| rr | 1.79 | 1.97 | +0.18 |
| edge | -0.1650 | -0.1024 | +0.0626 |
| sharpe | -6.52 | -2.86 | +3.66 |
| pf | 0.43 | 0.60 | +0.17 |

### CHOP_GUARD — chop_guard

**Best Finding:** chop_guard: OOS-adjusted PnL +$243.08
- Improvement: $+243.08
- Confidence: 58%
- Parameters: `{"chop_entry_floor_add": 0.003, "chop_entry_min_dir_conf": 0.8}`

```
[CHOP_GUARD] 파라미터 변경 제안:
  chop_entry_floor_add = 0.003
  chop_entry_min_dir_conf = 0.8
예상 효과: PnL $+324.23 (OOS 보정 $+243.08), WR +4.3%, R:R +0.69
OOS 검증: pass=True rate=75% trainΔ=+92.20 testΔ=+69.13 penalty=0.75
신뢰도: 58.2%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4642 | 956 | -3686 |
| pnl | -368.57 | -44.34 | +324.23 |
| wr | 0.1932 | 0.2364 | +0.0432 |
| rr | 1.79 | 2.48 | +0.69 |
| edge | -0.1650 | -0.0510 | +0.1140 |
| sharpe | -6.52 | -1.04 | +5.48 |
| pf | 0.43 | 0.77 | +0.34 |

### VOLATILITY_GATE — volatility_gate

**Best Finding:** volatility_gate: OOS-adjusted PnL +$240.11
- Improvement: $+240.11
- Confidence: 55%
- Parameters: `{"scope": "all_regimes", "chop_min_sigma": 0.35, "chop_max_sigma": 1.2, "chop_max_vpin": 0.5, "chop_min_dir_conf": 0.64, "chop_min_abs_mu_alpha": 0.0, "chop_max_hold_sec": 450}`

```
[VOLATILITY_GATE] 파라미터 변경 제안:
  scope = all_regimes
  chop_min_sigma = 0.35
  chop_max_sigma = 1.2
  chop_max_vpin = 0.5
  chop_min_dir_conf = 0.64
  chop_min_abs_mu_alpha = 0.0
  chop_max_hold_sec = 450
예상 효과: PnL $+325.59 (OOS 보정 $+240.11), WR -0.3%, R:R +0.99
OOS 검증: pass=True rate=75% trainΔ=+97.34 testΔ=+71.78 penalty=0.74
신뢰도: 55.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4642 | 895 | -3747 |
| pnl | -368.57 | -42.97 | +325.59 |
| wr | 0.1932 | 0.1899 | -0.0033 |
| rr | 1.79 | 2.78 | +0.99 |
| edge | -0.1650 | -0.0746 | +0.0904 |
| sharpe | -6.52 | -1.19 | +5.33 |
| pf | 0.43 | 0.65 | +0.22 |

### DIRECTION_GATE — direction_gate

**Best Finding:** direction_gate: OOS-adjusted PnL +$221.39
- Improvement: $+221.39
- Confidence: 51%
- Parameters: `{"dir_gate_min_conf": 0.65, "dir_gate_min_edge": 0.08, "dir_gate_min_side_prob": 0.56}`

```
[DIRECTION_GATE] 파라미터 변경 제안:
  dir_gate_min_conf = 0.65
  dir_gate_min_edge = 0.08
  dir_gate_min_side_prob = 0.56
예상 효과: PnL $+329.09 (OOS 보정 $+221.39), WR -0.4%, R:R +1.32
OOS 검증: pass=True rate=75% trainΔ=+106.08 testΔ=+71.36 penalty=0.67
신뢰도: 50.9%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4642 | 1029 | -3613 |
| pnl | -368.57 | -39.48 | +329.09 |
| wr | 0.1932 | 0.1895 | -0.0037 |
| rr | 1.79 | 3.11 | +1.32 |
| edge | -0.1650 | -0.0538 | +0.1112 |
| sharpe | -6.52 | -0.94 | +5.58 |
| pf | 0.43 | 0.73 | +0.30 |

### DIRECTION — 방향 결정

**Best Finding:** direction: OOS-adjusted PnL +$178.75
- Improvement: $+178.75
- Confidence: 44%
- Parameters: `{"chop_prefer_short": false, "min_dir_conf_for_entry": 0.65, "mu_alpha_sign_override": true}`

```
[DIRECTION] 파라미터 변경 제안:
  chop_prefer_short = False
  min_dir_conf_for_entry = 0.65
  mu_alpha_sign_override = True
예상 효과: PnL $+292.65 (OOS 보정 $+178.75), WR -3.5%, R:R +1.39
OOS 검증: pass=True rate=100% trainΔ=+101.92 testΔ=+62.25 penalty=0.61
신뢰도: 43.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4642 | 1359 | -3283 |
| pnl | -368.57 | -75.92 | +292.65 |
| wr | 0.1932 | 0.1582 | -0.0350 |
| rr | 1.79 | 3.18 | +1.39 |
| edge | -0.1650 | -0.0811 | +0.0839 |
| sharpe | -6.52 | -1.80 | +4.72 |
| pf | 0.43 | 0.60 | +0.17 |

### DIRECTION_CONFIRM — direction_confirm

**Best Finding:** direction_confirm: OOS-adjusted PnL +$122.32
- Improvement: $+122.32
- Confidence: 31%
- Parameters: `{"dir_gate_confirm_ticks": 4, "dir_gate_confirm_ticks_chop": 4}`

```
[DIRECTION_CONFIRM] 파라미터 변경 제안:
  dir_gate_confirm_ticks = 4
  dir_gate_confirm_ticks_chop = 4
예상 효과: PnL $+219.69 (OOS 보정 $+122.32), WR -4.5%, R:R +0.77
OOS 검증: pass=True rate=75% trainΔ=+81.42 testΔ=+45.33 penalty=0.56
신뢰도: 30.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4642 | 1884 | -2758 |
| pnl | -368.57 | -148.88 | +219.69 |
| wr | 0.1932 | 0.1481 | -0.0451 |
| rr | 1.79 | 2.56 | +0.77 |
| edge | -0.1650 | -0.1330 | +0.0320 |
| sharpe | -6.52 | -3.36 | +3.17 |
| pf | 0.43 | 0.44 | +0.02 |

### CAPITAL_ALLOCATION — 자본 분배

**Best Finding:** capital_allocation: OOS-adjusted PnL +$118.45
- Improvement: $+118.45
- Confidence: 33%
- Parameters: `{"notional_hard_cap": 50, "max_pos_frac": 0.15}`

```
[CAPITAL_ALLOCATION] 파라미터 변경 제안:
  notional_hard_cap = 50
  max_pos_frac = 0.15
예상 효과: PnL $+158.89 (OOS 보정 $+118.45), WR +0.0%, R:R +0.35
OOS 검증: pass=True rate=100% trainΔ=+44.24 testΔ=+32.98 penalty=0.75
신뢰도: 32.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4642 | 4642 | +0 |
| pnl | -368.57 | -209.68 | +158.89 |
| wr | 0.1932 | 0.1932 | +0.0000 |
| rr | 1.79 | 2.14 | +0.35 |
| edge | -0.1650 | -0.1254 | +0.0396 |
| sharpe | -6.52 | -4.34 | +2.18 |
| pf | 0.43 | 0.51 | +0.08 |

### LEVERAGE — 레버리지 결정

**Best Finding:** leverage: OOS-adjusted PnL +$107.83
- Improvement: $+107.83
- Confidence: 43%
- Parameters: `{"max_leverage": 50, "regime_max_bull": 20, "regime_max_chop": 3, "regime_max_bear": 12}`

```
[LEVERAGE] 파라미터 변경 제안:
  max_leverage = 50
  regime_max_bull = 20
  regime_max_chop = 3
  regime_max_bear = 12
예상 효과: PnL $+121.61 (OOS 보정 $+107.83), WR -4.5%, R:R +0.83
OOS 검증: pass=True rate=75% trainΔ=+31.38 testΔ=+27.82 penalty=0.89
신뢰도: 42.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4642 | 4642 | +0 |
| pnl | -368.57 | -246.96 | +121.61 |
| wr | 0.1932 | 0.1478 | -0.0454 |
| rr | 1.79 | 2.62 | +0.83 |
| edge | -0.1650 | -0.1285 | +0.0365 |
| sharpe | -6.52 | -5.18 | +1.34 |
| pf | 0.43 | 0.45 | +0.03 |

## Regime Performance Breakdown

| Regime | N | PnL | WR | R:R | Edge |
|--------|---|-----|----|----|------|
| chop | 3859 | $-292.68 | 18.7% | 1.68 | -18.6% |
| bull | 478 | $-24.90 | 27.0% | 1.98 | -6.6% |
| bear | 303 | $-50.72 | 14.8% | 1.87 | -20.0% |
| volatile | 2 | $-0.27 | 0.0% | 0.00 | -100.0% |

## 🎯 Recommended Actions

1. **vpin_filter: OOS-adjusted PnL +$276.06** (ΔPnL: $+276.06, confidence: 65%)
   - `max_vpin` = `0.3`

2. **regime_side_block: OOS-adjusted PnL +$258.25** (ΔPnL: $+258.25, confidence: 58%)
   - `regime_side_block_list` = `bear_long,bull_short,chop_long`

3. **chop_guard: OOS-adjusted PnL +$243.08** (ΔPnL: $+243.08, confidence: 58%)
   - `chop_entry_floor_add` = `0.003`
   - `chop_entry_min_dir_conf` = `0.8`

4. **volatility_gate: OOS-adjusted PnL +$240.11** (ΔPnL: $+240.11, confidence: 55%)
   - `scope` = `all_regimes`
   - `chop_min_sigma` = `0.35`
   - `chop_max_sigma` = `1.2`
   - `chop_max_vpin` = `0.5`
   - `chop_min_dir_conf` = `0.64`
   - `chop_min_abs_mu_alpha` = `0.0`
   - `chop_max_hold_sec` = `450`

5. **direction_gate: OOS-adjusted PnL +$221.39** (ΔPnL: $+221.39, confidence: 51%)
   - `dir_gate_min_conf` = `0.65`
   - `dir_gate_min_edge` = `0.08`
   - `dir_gate_min_side_prob` = `0.56`

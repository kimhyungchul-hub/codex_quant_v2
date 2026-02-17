# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-18 03:20
> Baseline: 4640 trades, PnL=$-369.94, WR=19.3%, R:R=1.78

## Pipeline Stage Impact Summary

### VPIN_FILTER — VPIN 필터

**Best Finding:** vpin_filter: OOS-adjusted PnL +$276.06
- Improvement: $+276.06
- Confidence: 65%
- Parameters: `{"max_vpin": 0.3}`

```
[VPIN_FILTER] 파라미터 변경 제안:
  max_vpin = 0.3
예상 효과: PnL $+276.06, WR +4.6%, R:R +0.29
신뢰도: 64.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4640 | 2111 | -2529 |
| pnl | -369.94 | -93.89 | +276.06 |
| wr | 0.1931 | 0.2387 | +0.0456 |
| rr | 1.78 | 2.08 | +0.29 |
| edge | -0.1663 | -0.0864 | +0.0799 |
| sharpe | -6.55 | -2.03 | +4.52 |
| pf | 0.43 | 0.65 | +0.22 |

### VOLATILITY_GATE — volatility_gate

**Best Finding:** volatility_gate: OOS-adjusted PnL +$272.63
- Improvement: $+272.63
- Confidence: 60%
- Parameters: `{"scope": "all_regimes", "chop_min_sigma": 0.1, "chop_max_sigma": 2.5, "chop_max_vpin": 0.65, "chop_min_dir_conf": 0.64, "chop_min_abs_mu_alpha": 5.0, "chop_max_hold_sec": 180}`

```
[VOLATILITY_GATE] 파라미터 변경 제안:
  scope = all_regimes
  chop_min_sigma = 0.1
  chop_max_sigma = 2.5
  chop_max_vpin = 0.65
  chop_min_dir_conf = 0.64
  chop_min_abs_mu_alpha = 5.0
  chop_max_hold_sec = 180
예상 효과: PnL $+356.63, WR -0.6%, R:R +2.01
신뢰도: 60.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4640 | 919 | -3721 |
| pnl | -369.94 | -13.31 | +356.63 |
| wr | 0.1931 | 0.1872 | -0.0059 |
| rr | 1.78 | 3.79 | +2.01 |
| edge | -0.1663 | -0.0214 | +0.1449 |
| sharpe | -6.55 | -0.35 | +6.20 |
| pf | 0.43 | 0.87 | +0.45 |

### REGIME_SIDE_BLOCK — regime_side_block

**Best Finding:** regime_side_block: OOS-adjusted PnL +$262.04
- Improvement: $+262.04
- Confidence: 59%
- Parameters: `{"regime_side_block_list": "bear_long,bull_short,chop_long"}`

```
[REGIME_SIDE_BLOCK] 파라미터 변경 제안:
  regime_side_block_list = bear_long,bull_short,chop_long
예상 효과: PnL $+277.70, WR +4.2%, R:R +0.18
신뢰도: 58.5%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4640 | 1867 | -2773 |
| pnl | -369.94 | -92.25 | +277.70 |
| wr | 0.1931 | 0.2346 | +0.0415 |
| rr | 1.78 | 1.97 | +0.19 |
| edge | -0.1663 | -0.1024 | +0.0639 |
| sharpe | -6.55 | -2.86 | +3.69 |
| pf | 0.43 | 0.60 | +0.18 |

### CHOP_GUARD — chop_guard

**Best Finding:** chop_guard: OOS-adjusted PnL +$246.11
- Improvement: $+246.11
- Confidence: 59%
- Parameters: `{"chop_entry_floor_add": 0.003, "chop_entry_min_dir_conf": 0.8}`

```
[CHOP_GUARD] 파라미터 변경 제안:
  chop_entry_floor_add = 0.003
  chop_entry_min_dir_conf = 0.8
예상 효과: PnL $+325.60, WR +4.3%, R:R +0.70
신뢰도: 58.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4640 | 956 | -3684 |
| pnl | -369.94 | -44.34 | +325.60 |
| wr | 0.1931 | 0.2364 | +0.0433 |
| rr | 1.78 | 2.48 | +0.70 |
| edge | -0.1663 | -0.0510 | +0.1153 |
| sharpe | -6.55 | -1.04 | +5.51 |
| pf | 0.43 | 0.77 | +0.34 |

### DIRECTION_GATE — direction_gate

**Best Finding:** direction_gate: OOS-adjusted PnL +$234.67
- Improvement: $+234.67
- Confidence: 53%
- Parameters: `{"dir_gate_min_conf": 0.65, "dir_gate_min_edge": 0.1, "dir_gate_min_side_prob": 0.4975}`

```
[DIRECTION_GATE] 파라미터 변경 제안:
  dir_gate_min_conf = 0.65
  dir_gate_min_edge = 0.1
  dir_gate_min_side_prob = 0.4975
예상 효과: PnL $+337.99, WR -0.1%, R:R +1.44
신뢰도: 53.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4640 | 1003 | -3637 |
| pnl | -369.94 | -31.95 | +337.99 |
| wr | 0.1931 | 0.1924 | -0.0007 |
| rr | 1.78 | 3.22 | +1.44 |
| edge | -0.1663 | -0.0446 | +0.1217 |
| sharpe | -6.55 | -0.77 | +5.78 |
| pf | 0.43 | 0.77 | +0.34 |

### DIRECTION — 방향 결정

**Best Finding:** direction: OOS-adjusted PnL +$181.25
- Improvement: $+181.25
- Confidence: 44%
- Parameters: `{"chop_prefer_short": false, "min_dir_conf_for_entry": 0.65, "mu_alpha_sign_override": true}`

```
[DIRECTION] 파라미터 변경 제안:
  chop_prefer_short = False
  min_dir_conf_for_entry = 0.65
  mu_alpha_sign_override = True
예상 효과: PnL $+294.02, WR -3.5%, R:R +1.40
신뢰도: 44.3%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4640 | 1359 | -3281 |
| pnl | -369.94 | -75.92 | +294.02 |
| wr | 0.1931 | 0.1582 | -0.0349 |
| rr | 1.78 | 3.18 | +1.40 |
| edge | -0.1663 | -0.0811 | +0.0852 |
| sharpe | -6.55 | -1.80 | +4.75 |
| pf | 0.43 | 0.60 | +0.17 |

### DIRECTION_CONFIRM — direction_confirm

**Best Finding:** direction_confirm: OOS-adjusted PnL +$124.54
- Improvement: $+124.54
- Confidence: 32%
- Parameters: `{"dir_gate_confirm_ticks": 4, "dir_gate_confirm_ticks_chop": 4}`

```
[DIRECTION_CONFIRM] 파라미터 변경 제안:
  dir_gate_confirm_ticks = 4
  dir_gate_confirm_ticks_chop = 4
예상 효과: PnL $+221.07, WR -4.5%, R:R +0.78
신뢰도: 31.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4640 | 1884 | -2756 |
| pnl | -369.94 | -148.88 | +221.07 |
| wr | 0.1931 | 0.1481 | -0.0450 |
| rr | 1.78 | 2.56 | +0.78 |
| edge | -0.1663 | -0.1330 | +0.0333 |
| sharpe | -6.55 | -3.36 | +3.19 |
| pf | 0.43 | 0.44 | +0.02 |

### CAPITAL_ALLOCATION — 자본 분배

**Best Finding:** capital_allocation: OOS-adjusted PnL +$118.52
- Improvement: $+118.52
- Confidence: 33%
- Parameters: `{"notional_hard_cap": 50, "max_pos_frac": 0.15}`

```
[CAPITAL_ALLOCATION] 파라미터 변경 제안:
  notional_hard_cap = 50
  max_pos_frac = 0.15
예상 효과: PnL $+158.89, WR +0.0%, R:R +0.34
신뢰도: 32.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4640 | 4640 | +0 |
| pnl | -369.94 | -211.06 | +158.89 |
| wr | 0.1931 | 0.1931 | +0.0000 |
| rr | 1.78 | 2.12 | +0.34 |
| edge | -0.1663 | -0.1269 | +0.0394 |
| sharpe | -6.55 | -4.37 | +2.18 |
| pf | 0.43 | 0.51 | +0.08 |

### LEVERAGE — 레버리지 결정

**Best Finding:** leverage: OOS-adjusted PnL +$110.51
- Improvement: $+110.51
- Confidence: 44%
- Parameters: `{"max_leverage": 20, "regime_max_bull": 20, "regime_max_chop": 3, "regime_max_bear": 12}`

```
[LEVERAGE] 파라미터 변경 제안:
  max_leverage = 20
  regime_max_bull = 20
  regime_max_chop = 3
  regime_max_bear = 12
예상 효과: PnL $+122.68, WR -4.5%, R:R +0.84
신뢰도: 43.9%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4640 | 4640 | +0 |
| pnl | -369.94 | -247.26 | +122.68 |
| wr | 0.1931 | 0.1476 | -0.0455 |
| rr | 1.78 | 2.62 | +0.84 |
| edge | -0.1663 | -0.1288 | +0.0375 |
| sharpe | -6.55 | -5.18 | +1.37 |
| pf | 0.43 | 0.45 | +0.03 |

## Regime Performance Breakdown

| Regime | N | PnL | WR | R:R | Edge |
|--------|---|-----|----|----|------|
| chop | 3857 | $-294.06 | 18.7% | 1.66 | -18.8% |
| bull | 478 | $-24.90 | 27.0% | 1.98 | -6.6% |
| bear | 303 | $-50.72 | 14.8% | 1.87 | -20.0% |
| volatile | 2 | $-0.27 | 0.0% | 0.00 | -100.0% |

## 🎯 Recommended Actions

1. **vpin_filter: OOS-adjusted PnL +$276.06** (ΔPnL: $+276.06, confidence: 65%)
   - `max_vpin` = `0.3`

2. **volatility_gate: OOS-adjusted PnL +$272.63** (ΔPnL: $+272.63, confidence: 60%)
   - `scope` = `all_regimes`
   - `chop_min_sigma` = `0.1`
   - `chop_max_sigma` = `2.5`
   - `chop_max_vpin` = `0.65`
   - `chop_min_dir_conf` = `0.64`
   - `chop_min_abs_mu_alpha` = `5.0`
   - `chop_max_hold_sec` = `180`

3. **volatility_gate: OOS-adjusted PnL +$271.63** (ΔPnL: $+271.63, confidence: 60%)
   - `scope` = `all_regimes`
   - `chop_min_sigma` = `0.2`
   - `chop_max_sigma` = `1.8`
   - `chop_max_vpin` = `0.65`
   - `chop_min_dir_conf` = `0.64`
   - `chop_min_abs_mu_alpha` = `5.0`
   - `chop_max_hold_sec` = `900`

4. **regime_side_block: OOS-adjusted PnL +$262.04** (ΔPnL: $+262.04, confidence: 59%)
   - `regime_side_block_list` = `bear_long,bull_short,chop_long`

5. **chop_guard: OOS-adjusted PnL +$246.11** (ΔPnL: $+246.11, confidence: 59%)
   - `chop_entry_floor_add` = `0.003`
   - `chop_entry_min_dir_conf` = `0.8`

# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-18 09:23
> Baseline: 4970 trades, PnL=$-390.23, WR=18.8%, R:R=1.81

## Pipeline Stage Impact Summary

### VPIN_FILTER — VPIN 필터

**Best Finding:** vpin_filter: OOS-adjusted PnL +$292.27
- Improvement: $+292.27
- Confidence: 65%
- Parameters: `{"max_vpin": 0.3}`

```
[VPIN_FILTER] 파라미터 변경 제안:
  max_vpin = 0.3
예상 효과: PnL $+292.27 (OOS 보정 $+292.27), WR +5.0%, R:R +0.24
OOS 검증: pass=True rate=75% trainΔ=+66.33 testΔ=+67.30 penalty=1.00
신뢰도: 64.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4970 | 2160 | -2810 |
| pnl | -390.23 | -97.96 | +292.27 |
| wr | 0.1883 | 0.2384 | +0.0501 |
| rr | 1.81 | 2.06 | +0.24 |
| edge | -0.1672 | -0.0888 | +0.0784 |
| sharpe | -6.88 | -2.11 | +4.77 |
| pf | 0.42 | 0.64 | +0.22 |

### VOLATILITY_GATE — volatility_gate

**Best Finding:** volatility_gate: OOS-adjusted PnL +$274.37
- Improvement: $+274.37
- Confidence: 59%
- Parameters: `{"scope": "all_regimes", "chop_min_sigma": 0.5, "chop_max_sigma": 2.5, "chop_max_vpin": 0.4, "chop_min_dir_conf": 0.56, "chop_min_abs_mu_alpha": 0.0, "chop_max_hold_sec": 180}`

```
[VOLATILITY_GATE] 파라미터 변경 제안:
  scope = all_regimes
  chop_min_sigma = 0.5
  chop_max_sigma = 2.5
  chop_max_vpin = 0.4
  chop_min_dir_conf = 0.56
  chop_min_abs_mu_alpha = 0.0
  chop_max_hold_sec = 180
예상 효과: PnL $+342.97 (OOS 보정 $+274.37), WR +3.4%, R:R +0.60
OOS 검증: pass=True rate=75% trainΔ=+95.15 testΔ=+76.12 penalty=0.80
신뢰도: 59.3%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4970 | 1141 | -3829 |
| pnl | -390.23 | -47.26 | +342.97 |
| wr | 0.1883 | 0.2226 | +0.0343 |
| rr | 1.81 | 2.42 | +0.60 |
| edge | -0.1672 | -0.0701 | +0.0971 |
| sharpe | -6.88 | -1.23 | +5.66 |
| pf | 0.42 | 0.69 | +0.27 |

### REGIME_SIDE_BLOCK — regime_side_block

**Best Finding:** regime_side_block: OOS-adjusted PnL +$262.40
- Improvement: $+262.40
- Confidence: 56%
- Parameters: `{"regime_side_block_list": "bear_long,bull_short,chop_long"}`

```
[REGIME_SIDE_BLOCK] 파라미터 변경 제안:
  regime_side_block_list = bear_long,bull_short,chop_long
예상 효과: PnL $+289.03 (OOS 보정 $+262.40), WR +4.2%, R:R +0.16
OOS 검증: pass=True rate=75% trainΔ=+75.66 testΔ=+68.69 penalty=0.91
신뢰도: 55.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4970 | 2037 | -2933 |
| pnl | -390.23 | -101.20 | +289.03 |
| wr | 0.1883 | 0.2302 | +0.0419 |
| rr | 1.81 | 1.97 | +0.16 |
| edge | -0.1672 | -0.1061 | +0.0611 |
| sharpe | -6.88 | -3.12 | +3.77 |
| pf | 0.42 | 0.59 | +0.17 |

### CHOP_GUARD — chop_guard

**Best Finding:** chop_guard: OOS-adjusted PnL +$246.94
- Improvement: $+246.94
- Confidence: 56%
- Parameters: `{"chop_entry_floor_add": 0.005, "chop_entry_min_dir_conf": 0.8}`

```
[CHOP_GUARD] 파라미터 변경 제안:
  chop_entry_floor_add = 0.005
  chop_entry_min_dir_conf = 0.8
예상 효과: PnL $+339.70 (OOS 보정 $+246.94), WR +3.9%, R:R +0.73
OOS 검증: pass=True rate=75% trainΔ=+99.16 testΔ=+72.08 penalty=0.73
신뢰도: 56.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4970 | 1046 | -3924 |
| pnl | -390.23 | -50.52 | +339.70 |
| wr | 0.1883 | 0.2275 | +0.0392 |
| rr | 1.81 | 2.54 | +0.73 |
| edge | -0.1672 | -0.0551 | +0.1121 |
| sharpe | -6.88 | -1.18 | +5.70 |
| pf | 0.42 | 0.75 | +0.33 |

### DIRECTION_GATE — direction_gate

**Best Finding:** direction_gate: OOS-adjusted PnL +$242.82
- Improvement: $+242.82
- Confidence: 52%
- Parameters: `{"dir_gate_min_conf": 0.58, "dir_gate_min_edge": 0.1, "dir_gate_min_side_prob": 0.65}`

```
[DIRECTION_GATE] 파라미터 변경 제안:
  dir_gate_min_conf = 0.58
  dir_gate_min_edge = 0.1
  dir_gate_min_side_prob = 0.65
예상 효과: PnL $+356.45 (OOS 보정 $+242.82), WR +0.2%, R:R +1.42
OOS 검증: pass=True rate=75% trainΔ=+113.32 testΔ=+77.20 penalty=0.68
신뢰도: 52.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4970 | 1068 | -3902 |
| pnl | -390.23 | -33.78 | +356.45 |
| wr | 0.1883 | 0.1901 | +0.0018 |
| rr | 1.81 | 3.24 | +1.42 |
| edge | -0.1672 | -0.0459 | +0.1213 |
| sharpe | -6.88 | -0.81 | +6.07 |
| pf | 0.42 | 0.76 | +0.34 |

### DIRECTION — 방향 결정

**Best Finding:** direction: OOS-adjusted PnL +$188.36
- Improvement: $+188.36
- Confidence: 44%
- Parameters: `{"chop_prefer_short": false, "min_dir_conf_for_entry": 0.65, "mu_alpha_sign_override": true}`

```
[DIRECTION] 파라미터 변경 제안:
  chop_prefer_short = False
  min_dir_conf_for_entry = 0.65
  mu_alpha_sign_override = True
예상 효과: PnL $+311.07 (OOS 보정 $+188.36), WR -3.2%, R:R +1.37
OOS 검증: pass=True rate=100% trainΔ=+108.76 testΔ=+65.85 penalty=0.61
신뢰도: 43.5%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4970 | 1437 | -3533 |
| pnl | -390.23 | -79.15 | +311.07 |
| wr | 0.1883 | 0.1566 | -0.0317 |
| rr | 1.81 | 3.18 | +1.37 |
| edge | -0.1672 | -0.0824 | +0.0848 |
| sharpe | -6.88 | -1.87 | +5.01 |
| pf | 0.42 | 0.59 | +0.17 |

### DIRECTION_CONFIRM — direction_confirm

**Best Finding:** direction_confirm: OOS-adjusted PnL +$129.88
- Improvement: $+129.88
- Confidence: 31%
- Parameters: `{"dir_gate_confirm_ticks": 4, "dir_gate_confirm_ticks_chop": 4}`

```
[DIRECTION_CONFIRM] 파라미터 변경 제안:
  dir_gate_confirm_ticks = 4
  dir_gate_confirm_ticks_chop = 4
예상 효과: PnL $+235.11 (OOS 보정 $+129.88), WR -4.3%, R:R +0.76
OOS 검증: pass=True rate=75% trainΔ=+87.22 testΔ=+48.18 penalty=0.55
신뢰도: 30.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4970 | 1990 | -2980 |
| pnl | -390.23 | -155.12 | +235.11 |
| wr | 0.1883 | 0.1457 | -0.0426 |
| rr | 1.81 | 2.57 | +0.76 |
| edge | -0.1672 | -0.1345 | +0.0327 |
| sharpe | -6.88 | -3.49 | +3.39 |
| pf | 0.42 | 0.44 | +0.02 |

### CAPITAL_ALLOCATION — 자본 분배

**Best Finding:** capital_allocation: OOS-adjusted PnL +$120.29
- Improvement: $+120.29
- Confidence: 31%
- Parameters: `{"notional_hard_cap": 50, "max_pos_frac": 0.15}`

```
[CAPITAL_ALLOCATION] 파라미터 변경 제안:
  notional_hard_cap = 50
  max_pos_frac = 0.15
예상 효과: PnL $+166.20 (OOS 보정 $+120.29), WR +0.0%, R:R +0.35
OOS 검증: pass=True rate=100% trainΔ=+47.50 testΔ=+34.38 penalty=0.72
신뢰도: 31.5%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4970 | 4970 | +0 |
| pnl | -390.23 | -224.02 | +166.20 |
| wr | 0.1883 | 0.1883 | +0.0000 |
| rr | 1.81 | 2.16 | +0.35 |
| edge | -0.1672 | -0.1282 | +0.0390 |
| sharpe | -6.88 | -4.62 | +2.26 |
| pf | 0.42 | 0.50 | +0.08 |

## Regime Performance Breakdown

| Regime | N | PnL | WR | R:R | Edge |
|--------|---|-----|----|----|------|
| chop | 4086 | $-307.83 | 18.3% | 1.69 | -19.0% |
| bull | 561 | $-29.49 | 25.7% | 2.06 | -7.0% |
| bear | 321 | $-52.64 | 14.3% | 1.90 | -20.1% |
| volatile | 2 | $-0.27 | 0.0% | 0.00 | -100.0% |

## 🎯 Recommended Actions

1. **vpin_filter: OOS-adjusted PnL +$292.27** (ΔPnL: $+292.27, confidence: 65%)
   - `max_vpin` = `0.3`

2. **volatility_gate: OOS-adjusted PnL +$274.37** (ΔPnL: $+274.37, confidence: 59%)
   - `scope` = `all_regimes`
   - `chop_min_sigma` = `0.5`
   - `chop_max_sigma` = `2.5`
   - `chop_max_vpin` = `0.4`
   - `chop_min_dir_conf` = `0.56`
   - `chop_min_abs_mu_alpha` = `0.0`
   - `chop_max_hold_sec` = `180`

3. **regime_side_block: OOS-adjusted PnL +$262.40** (ΔPnL: $+262.40, confidence: 56%)
   - `regime_side_block_list` = `bear_long,bull_short,chop_long`

4. **chop_guard: OOS-adjusted PnL +$246.94** (ΔPnL: $+246.94, confidence: 56%)
   - `chop_entry_floor_add` = `0.005`
   - `chop_entry_min_dir_conf` = `0.8`

5. **direction_gate: OOS-adjusted PnL +$242.82** (ΔPnL: $+242.82, confidence: 52%)
   - `dir_gate_min_conf` = `0.58`
   - `dir_gate_min_edge` = `0.1`
   - `dir_gate_min_side_prob` = `0.65`

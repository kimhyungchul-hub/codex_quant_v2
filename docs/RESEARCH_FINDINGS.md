# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-18 06:44
> Baseline: 4641 trades, PnL=$-368.18, WR=19.3%, R:R=1.79

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
OOS 검증: pass=True rate=75% trainΔ=+62.22 testΔ=+63.65 penalty=1.00
신뢰도: 65.2%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4641 | 2112 | -2529 |
| pnl | -368.18 | -92.12 | +276.06 |
| wr | 0.1933 | 0.2391 | +0.0458 |
| rr | 1.79 | 2.09 | +0.30 |
| edge | -0.1649 | -0.0843 | +0.0806 |
| sharpe | -6.51 | -1.99 | +4.53 |
| pf | 0.43 | 0.66 | +0.23 |

### VOLATILITY_GATE — volatility_gate

**Best Finding:** volatility_gate: OOS-adjusted PnL +$263.51
- Improvement: $+263.51
- Confidence: 60%
- Parameters: `{"scope": "all_regimes", "chop_min_sigma": 0.5, "chop_max_sigma": 1.2, "chop_max_vpin": 0.65, "chop_min_dir_conf": 0.6, "chop_min_abs_mu_alpha": 5.0, "chop_max_hold_sec": 180}`

```
[VOLATILITY_GATE] 파라미터 변경 제안:
  scope = all_regimes
  chop_min_sigma = 0.5
  chop_max_sigma = 1.2
  chop_max_vpin = 0.65
  chop_min_dir_conf = 0.6
  chop_min_abs_mu_alpha = 5.0
  chop_max_hold_sec = 180
예상 효과: PnL $+339.88 (OOS 보정 $+263.51), WR +0.3%, R:R +1.41
OOS 검증: pass=True rate=75% trainΔ=+97.71 testΔ=+75.75 penalty=0.78
신뢰도: 60.1%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4641 | 1059 | -3582 |
| pnl | -368.18 | -28.30 | +339.88 |
| wr | 0.1933 | 0.1964 | +0.0031 |
| rr | 1.79 | 3.20 | +1.41 |
| edge | -0.1649 | -0.0417 | +0.1232 |
| sharpe | -6.51 | -0.72 | +5.79 |
| pf | 0.43 | 0.78 | +0.35 |

### REGIME_SIDE_BLOCK — regime_side_block

**Best Finding:** regime_side_block: OOS-adjusted PnL +$258.40
- Improvement: $+258.40
- Confidence: 58%
- Parameters: `{"regime_side_block_list": "bear_long,bull_short,chop_long"}`

```
[REGIME_SIDE_BLOCK] 파라미터 변경 제안:
  regime_side_block_list = bear_long,bull_short,chop_long
예상 효과: PnL $+275.93 (OOS 보정 $+258.40), WR +4.1%, R:R +0.18
OOS 검증: pass=True rate=75% trainΔ=+70.56 testΔ=+66.08 penalty=0.94
신뢰도: 57.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4641 | 1867 | -2774 |
| pnl | -368.18 | -92.25 | +275.93 |
| wr | 0.1933 | 0.2346 | +0.0413 |
| rr | 1.79 | 1.97 | +0.18 |
| edge | -0.1649 | -0.1024 | +0.0625 |
| sharpe | -6.51 | -2.86 | +3.65 |
| pf | 0.43 | 0.60 | +0.17 |

### CHOP_GUARD — chop_guard

**Best Finding:** chop_guard: OOS-adjusted PnL +$243.16
- Improvement: $+243.16
- Confidence: 58%
- Parameters: `{"chop_entry_floor_add": 0.003, "chop_entry_min_dir_conf": 0.8}`

```
[CHOP_GUARD] 파라미터 변경 제안:
  chop_entry_floor_add = 0.003
  chop_entry_min_dir_conf = 0.8
예상 효과: PnL $+323.83 (OOS 보정 $+243.16), WR +4.3%, R:R +0.69
OOS 검증: pass=True rate=75% trainΔ=+92.07 testΔ=+69.13 penalty=0.75
신뢰도: 58.2%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4641 | 956 | -3685 |
| pnl | -368.18 | -44.34 | +323.83 |
| wr | 0.1933 | 0.2364 | +0.0431 |
| rr | 1.79 | 2.48 | +0.69 |
| edge | -0.1649 | -0.0510 | +0.1139 |
| sharpe | -6.51 | -1.04 | +5.47 |
| pf | 0.43 | 0.77 | +0.34 |

### DIRECTION_GATE — direction_gate

**Best Finding:** direction_gate: OOS-adjusted PnL +$231.89
- Improvement: $+231.89
- Confidence: 53%
- Parameters: `{"dir_gate_min_conf": 0.58, "dir_gate_min_edge": 0.1, "dir_gate_min_side_prob": 0.65}`

```
[DIRECTION_GATE] 파라미터 변경 제안:
  dir_gate_min_conf = 0.58
  dir_gate_min_edge = 0.1
  dir_gate_min_side_prob = 0.65
예상 효과: PnL $+336.22 (OOS 보정 $+231.89), WR -0.1%, R:R +1.43
OOS 검증: pass=True rate=75% trainΔ=+106.21 testΔ=+73.25 penalty=0.69
신뢰도: 52.8%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4641 | 1003 | -3638 |
| pnl | -368.18 | -31.95 | +336.22 |
| wr | 0.1933 | 0.1924 | -0.0009 |
| rr | 1.79 | 3.22 | +1.43 |
| edge | -0.1649 | -0.0446 | +0.1203 |
| sharpe | -6.51 | -0.77 | +5.74 |
| pf | 0.43 | 0.77 | +0.34 |

### DIRECTION — 방향 결정

**Best Finding:** direction: OOS-adjusted PnL +$178.76
- Improvement: $+178.76
- Confidence: 44%
- Parameters: `{"chop_prefer_short": false, "min_dir_conf_for_entry": 0.65, "mu_alpha_sign_override": true}`

```
[DIRECTION] 파라미터 변경 제안:
  chop_prefer_short = False
  min_dir_conf_for_entry = 0.65
  mu_alpha_sign_override = True
예상 효과: PnL $+292.25 (OOS 보정 $+178.76), WR -3.5%, R:R +1.39
OOS 검증: pass=True rate=100% trainΔ=+101.79 testΔ=+62.26 penalty=0.61
신뢰도: 43.9%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4641 | 1359 | -3282 |
| pnl | -368.18 | -75.92 | +292.25 |
| wr | 0.1933 | 0.1582 | -0.0351 |
| rr | 1.79 | 3.18 | +1.39 |
| edge | -0.1649 | -0.0811 | +0.0838 |
| sharpe | -6.51 | -1.80 | +4.72 |
| pf | 0.43 | 0.60 | +0.17 |

### DIRECTION_CONFIRM — direction_confirm

**Best Finding:** direction_confirm: OOS-adjusted PnL +$122.27
- Improvement: $+122.27
- Confidence: 31%
- Parameters: `{"dir_gate_confirm_ticks": 4, "dir_gate_confirm_ticks_chop": 4}`

```
[DIRECTION_CONFIRM] 파라미터 변경 제안:
  dir_gate_confirm_ticks = 4
  dir_gate_confirm_ticks_chop = 4
예상 효과: PnL $+219.30 (OOS 보정 $+122.27), WR -4.5%, R:R +0.77
OOS 검증: pass=True rate=75% trainΔ=+81.31 testΔ=+45.34 penalty=0.56
신뢰도: 30.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4641 | 1884 | -2757 |
| pnl | -368.18 | -148.88 | +219.30 |
| wr | 0.1933 | 0.1481 | -0.0452 |
| rr | 1.79 | 2.56 | +0.77 |
| edge | -0.1649 | -0.1330 | +0.0319 |
| sharpe | -6.51 | -3.36 | +3.16 |
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
예상 효과: PnL $+158.89 (OOS 보정 $+118.52), WR +0.0%, R:R +0.35
OOS 검증: pass=True rate=100% trainΔ=+44.23 testΔ=+32.99 penalty=0.75
신뢰도: 32.9%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4641 | 4641 | +0 |
| pnl | -368.18 | -209.29 | +158.89 |
| wr | 0.1933 | 0.1933 | +0.0000 |
| rr | 1.79 | 2.14 | +0.35 |
| edge | -0.1649 | -0.1252 | +0.0397 |
| sharpe | -6.51 | -4.33 | +2.18 |
| pf | 0.43 | 0.51 | +0.08 |

## Regime Performance Breakdown

| Regime | N | PnL | WR | R:R | Edge |
|--------|---|-----|----|----|------|
| chop | 3858 | $-292.29 | 18.7% | 1.68 | -18.6% |
| bull | 478 | $-24.90 | 27.0% | 1.98 | -6.6% |
| bear | 303 | $-50.72 | 14.8% | 1.87 | -20.0% |
| volatile | 2 | $-0.27 | 0.0% | 0.00 | -100.0% |

## 🎯 Recommended Actions

1. **vpin_filter: OOS-adjusted PnL +$276.06** (ΔPnL: $+276.06, confidence: 65%)
   - `max_vpin` = `0.3`

2. **volatility_gate: OOS-adjusted PnL +$263.51** (ΔPnL: $+263.51, confidence: 60%)
   - `scope` = `all_regimes`
   - `chop_min_sigma` = `0.5`
   - `chop_max_sigma` = `1.2`
   - `chop_max_vpin` = `0.65`
   - `chop_min_dir_conf` = `0.6`
   - `chop_min_abs_mu_alpha` = `5.0`
   - `chop_max_hold_sec` = `180`

3. **regime_side_block: OOS-adjusted PnL +$258.40** (ΔPnL: $+258.40, confidence: 58%)
   - `regime_side_block_list` = `bear_long,bull_short,chop_long`

4. **chop_guard: OOS-adjusted PnL +$243.16** (ΔPnL: $+243.16, confidence: 58%)
   - `chop_entry_floor_add` = `0.003`
   - `chop_entry_min_dir_conf` = `0.8`

5. **volatility_gate: OOS-adjusted PnL +$240.62** (ΔPnL: $+240.62, confidence: 56%)
   - `scope` = `all_regimes`
   - `chop_min_sigma` = `0.1`
   - `chop_max_sigma` = `0.8`
   - `chop_max_vpin` = `0.65`
   - `chop_min_dir_conf` = `0.64`
   - `chop_min_abs_mu_alpha` = `0.0`
   - `chop_max_hold_sec` = `180`

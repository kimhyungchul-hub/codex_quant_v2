# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-18 04:58
> Baseline: 4969 trades, PnL=$-389.84, WR=18.8%, R:R=1.81

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
신뢰도: 64.9%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4969 | 2159 | -2810 |
| pnl | -389.84 | -97.57 | +292.27 |
| wr | 0.1884 | 0.2385 | +0.0501 |
| rr | 1.81 | 2.06 | +0.24 |
| edge | -0.1671 | -0.0885 | +0.0786 |
| sharpe | -6.88 | -2.10 | +4.77 |
| pf | 0.42 | 0.64 | +0.22 |

### VOLATILITY_GATE — volatility_gate

**Best Finding:** volatility_gate: OOS-adjusted PnL +$286.32
- Improvement: $+286.32
- Confidence: 64%
- Parameters: `{"scope": "all_regimes", "chop_min_sigma": 0.5, "chop_max_sigma": 1.8, "chop_max_vpin": 0.65, "chop_min_dir_conf": 0.6, "chop_min_abs_mu_alpha": 10.0, "chop_max_hold_sec": 300}`

```
[VOLATILITY_GATE] 파라미터 변경 제안:
  scope = all_regimes
  chop_min_sigma = 0.5
  chop_max_sigma = 1.8
  chop_max_vpin = 0.65
  chop_min_dir_conf = 0.6
  chop_min_abs_mu_alpha = 10.0
  chop_max_hold_sec = 300
예상 효과: PnL $+370.93 (OOS 보정 $+286.32), WR +2.2%, R:R +1.34
OOS 검증: pass=True rate=75% trainΔ=+106.70 testΔ=+82.36 penalty=0.77
신뢰도: 63.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4969 | 960 | -4009 |
| pnl | -389.84 | -18.90 | +370.93 |
| wr | 0.1884 | 0.2104 | +0.0220 |
| rr | 1.81 | 3.16 | +1.34 |
| edge | -0.1671 | -0.0302 | +0.1369 |
| sharpe | -6.88 | -0.48 | +6.40 |
| pf | 0.42 | 0.84 | +0.42 |

### REGIME_SIDE_BLOCK — regime_side_block

**Best Finding:** regime_side_block: OOS-adjusted PnL +$261.52
- Improvement: $+261.52
- Confidence: 55%
- Parameters: `{"regime_side_block_list": "bear_long,bull_short,chop_long"}`

```
[REGIME_SIDE_BLOCK] 파라미터 변경 제안:
  regime_side_block_list = bear_long,bull_short,chop_long
예상 효과: PnL $+288.64 (OOS 보정 $+261.52), WR +4.2%, R:R +0.16
OOS 검증: pass=True rate=75% trainΔ=+75.69 testΔ=+68.58 penalty=0.91
신뢰도: 55.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4969 | 2037 | -2932 |
| pnl | -389.84 | -101.20 | +288.64 |
| wr | 0.1884 | 0.2302 | +0.0418 |
| rr | 1.81 | 1.97 | +0.16 |
| edge | -0.1671 | -0.1061 | +0.0610 |
| sharpe | -6.88 | -3.12 | +3.76 |
| pf | 0.42 | 0.59 | +0.17 |

### CHOP_GUARD — chop_guard

**Best Finding:** chop_guard: OOS-adjusted PnL +$246.17
- Improvement: $+246.17
- Confidence: 56%
- Parameters: `{"chop_entry_floor_add": 0.005, "chop_entry_min_dir_conf": 0.8}`

```
[CHOP_GUARD] 파라미터 변경 제안:
  chop_entry_floor_add = 0.005
  chop_entry_min_dir_conf = 0.8
예상 효과: PnL $+339.31 (OOS 보정 $+246.17), WR +3.9%, R:R +0.73
OOS 검증: pass=True rate=75% trainΔ=+99.20 testΔ=+71.97 penalty=0.73
신뢰도: 56.0%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4969 | 1046 | -3923 |
| pnl | -389.84 | -50.52 | +339.31 |
| wr | 0.1884 | 0.2275 | +0.0391 |
| rr | 1.81 | 2.54 | +0.73 |
| edge | -0.1671 | -0.0551 | +0.1120 |
| sharpe | -6.88 | -1.18 | +5.69 |
| pf | 0.42 | 0.75 | +0.33 |

### DIRECTION_GATE — direction_gate

**Best Finding:** direction_gate: OOS-adjusted PnL +$242.03
- Improvement: $+242.03
- Confidence: 52%
- Parameters: `{"dir_gate_min_conf": 0.65, "dir_gate_min_edge": 0.1, "dir_gate_min_side_prob": 0.6}`

```
[DIRECTION_GATE] 파라미터 변경 제안:
  dir_gate_min_conf = 0.65
  dir_gate_min_edge = 0.1
  dir_gate_min_side_prob = 0.6
예상 효과: PnL $+356.05 (OOS 보정 $+242.03), WR +0.2%, R:R +1.42
OOS 검증: pass=True rate=75% trainΔ=+113.37 testΔ=+77.06 penalty=0.68
신뢰도: 52.3%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4969 | 1068 | -3901 |
| pnl | -389.84 | -33.78 | +356.05 |
| wr | 0.1884 | 0.1901 | +0.0017 |
| rr | 1.81 | 3.24 | +1.42 |
| edge | -0.1671 | -0.0459 | +0.1212 |
| sharpe | -6.88 | -0.81 | +6.06 |
| pf | 0.42 | 0.76 | +0.34 |

### DIRECTION — 방향 결정

**Best Finding:** direction: OOS-adjusted PnL +$187.66
- Improvement: $+187.66
- Confidence: 43%
- Parameters: `{"chop_prefer_short": false, "min_dir_conf_for_entry": 0.65, "mu_alpha_sign_override": true}`

```
[DIRECTION] 파라미터 변경 제안:
  chop_prefer_short = False
  min_dir_conf_for_entry = 0.65
  mu_alpha_sign_override = True
예상 효과: PnL $+310.68 (OOS 보정 $+187.66), WR -3.2%, R:R +1.37
OOS 검증: pass=True rate=100% trainΔ=+108.80 testΔ=+65.72 penalty=0.60
신뢰도: 43.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4969 | 1437 | -3532 |
| pnl | -389.84 | -79.15 | +310.68 |
| wr | 0.1884 | 0.1566 | -0.0318 |
| rr | 1.81 | 3.18 | +1.37 |
| edge | -0.1671 | -0.0824 | +0.0847 |
| sharpe | -6.88 | -1.87 | +5.00 |
| pf | 0.42 | 0.59 | +0.17 |

### DIRECTION_CONFIRM — direction_confirm

**Best Finding:** direction_confirm: OOS-adjusted PnL +$129.23
- Improvement: $+129.23
- Confidence: 31%
- Parameters: `{"dir_gate_confirm_ticks": 4, "dir_gate_confirm_ticks_chop": 4}`

```
[DIRECTION_CONFIRM] 파라미터 변경 제안:
  dir_gate_confirm_ticks = 4
  dir_gate_confirm_ticks_chop = 4
예상 효과: PnL $+234.72 (OOS 보정 $+129.23), WR -4.3%, R:R +0.76
OOS 검증: pass=True rate=75% trainΔ=+87.26 testΔ=+48.05 penalty=0.55
신뢰도: 30.6%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4969 | 1990 | -2979 |
| pnl | -389.84 | -155.12 | +234.72 |
| wr | 0.1884 | 0.1457 | -0.0427 |
| rr | 1.81 | 2.57 | +0.76 |
| edge | -0.1671 | -0.1345 | +0.0326 |
| sharpe | -6.88 | -3.49 | +3.39 |
| pf | 0.42 | 0.44 | +0.02 |

### CAPITAL_ALLOCATION — 자본 분배

**Best Finding:** capital_allocation: OOS-adjusted PnL +$120.36
- Improvement: $+120.36
- Confidence: 31%
- Parameters: `{"notional_hard_cap": 50, "max_pos_frac": 0.15}`

```
[CAPITAL_ALLOCATION] 파라미터 변경 제안:
  notional_hard_cap = 50
  max_pos_frac = 0.15
예상 효과: PnL $+166.20 (OOS 보정 $+120.36), WR +0.0%, R:R +0.35
OOS 검증: pass=True rate=100% trainΔ=+47.49 testΔ=+34.39 penalty=0.72
신뢰도: 31.5%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4969 | 4969 | +0 |
| pnl | -389.84 | -223.63 | +166.20 |
| wr | 0.1884 | 0.1884 | +0.0000 |
| rr | 1.81 | 2.16 | +0.35 |
| edge | -0.1671 | -0.1281 | +0.0390 |
| sharpe | -6.88 | -4.62 | +2.26 |
| pf | 0.42 | 0.50 | +0.08 |

### LEVERAGE — 레버리지 결정

**Best Finding:** leverage: OOS-adjusted PnL +$106.25
- Improvement: $+106.25
- Confidence: 39%
- Parameters: `{"max_leverage": 50, "regime_max_bull": 20, "regime_max_chop": 3, "regime_max_bear": 5}`

```
[LEVERAGE] 파라미터 변경 제안:
  max_leverage = 50
  regime_max_bull = 20
  regime_max_chop = 3
  regime_max_bear = 5
예상 효과: PnL $+136.77 (OOS 보정 $+106.25), WR -4.5%, R:R +0.87
OOS 검증: pass=True rate=75% trainΔ=+40.50 testΔ=+31.46 penalty=0.78
신뢰도: 39.4%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4969 | 4969 | +0 |
| pnl | -389.84 | -253.07 | +136.77 |
| wr | 0.1884 | 0.1431 | -0.0453 |
| rr | 1.81 | 2.68 | +0.87 |
| edge | -0.1671 | -0.1284 | +0.0387 |
| sharpe | -6.88 | -5.42 | +1.46 |
| pf | 0.42 | 0.45 | +0.03 |

## Regime Performance Breakdown

| Regime | N | PnL | WR | R:R | Edge |
|--------|---|-----|----|----|------|
| chop | 4085 | $-307.44 | 18.3% | 1.69 | -19.0% |
| bull | 561 | $-29.49 | 25.7% | 2.06 | -7.0% |
| bear | 321 | $-52.64 | 14.3% | 1.90 | -20.1% |
| volatile | 2 | $-0.27 | 0.0% | 0.00 | -100.0% |

## 🎯 Recommended Actions

1. **vpin_filter: OOS-adjusted PnL +$292.27** (ΔPnL: $+292.27, confidence: 65%)
   - `max_vpin` = `0.3`

2. **volatility_gate: OOS-adjusted PnL +$286.32** (ΔPnL: $+286.32, confidence: 64%)
   - `scope` = `all_regimes`
   - `chop_min_sigma` = `0.5`
   - `chop_max_sigma` = `1.8`
   - `chop_max_vpin` = `0.65`
   - `chop_min_dir_conf` = `0.6`
   - `chop_min_abs_mu_alpha` = `10.0`
   - `chop_max_hold_sec` = `300`

3. **regime_side_block: OOS-adjusted PnL +$261.52** (ΔPnL: $+261.52, confidence: 55%)
   - `regime_side_block_list` = `bear_long,bull_short,chop_long`

4. **volatility_gate: OOS-adjusted PnL +$257.07** (ΔPnL: $+257.07, confidence: 56%)
   - `scope` = `all_regimes`
   - `chop_min_sigma` = `0.35`
   - `chop_max_sigma` = `1.8`
   - `chop_max_vpin` = `0.65`
   - `chop_min_dir_conf` = `0.6`
   - `chop_min_abs_mu_alpha` = `5.0`
   - `chop_max_hold_sec` = `900`

5. **chop_guard: OOS-adjusted PnL +$246.17** (ΔPnL: $+246.17, confidence: 56%)
   - `chop_entry_floor_add` = `0.005`
   - `chop_entry_min_dir_conf` = `0.8`

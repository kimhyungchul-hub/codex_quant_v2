# Research Findings — Counterfactual Analysis

> Auto-generated: 2026-02-16 12:48
> Baseline: 4909 trades, PnL=$-59.16, WR=36.5%, R:R=1.48

## Pipeline Stage Impact Summary

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

### ENTRY_FILTER — 진입 필터

**Best Finding:** entry_filter: PnL +$78.70
- Improvement: $+78.70
- Confidence: 85%
- Parameters: `{"min_confidence": 0.55, "min_dir_conf": 0.65, "min_entry_quality": 0.2, "min_ev": 0.02}`

```
[ENTRY_FILTER] 파라미터 변경 제안:
  min_confidence = 0.55
  min_dir_conf = 0.65
  min_entry_quality = 0.2
  min_ev = 0.02
예상 효과: PnL $+78.70, WR +8.6%, R:R +0.37
신뢰도: 84.7%
```

| Metric | Baseline | CF | Delta |
|--------|----------|----|----|
| n | 4909 | 519 | -4390 |
| pnl | -59.16 | 19.54 | +78.70 |
| wr | 0.3650 | 0.4509 | +0.0859 |
| rr | 1.48 | 1.85 | +0.37 |
| edge | -0.0383 | 0.1003 | +0.1386 |
| sharpe | -1.07 | 0.84 | +1.91 |
| pf | 0.85 | 1.52 | +0.67 |

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
- Parameters: `{"max_leverage": 20, "regime_max_bull": 20, "regime_max_chop": 3, "regime_max_bear": 5}`

```
[LEVERAGE] 파라미터 변경 제안:
  max_leverage = 20
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

1. **tp_sl: PnL +$127.47** (ΔPnL: $+127.47, confidence: 80%)
   - `tp_pct` = `0.04`
   - `sl_pct` = `0.005`

2. **chop_guard: PnL +$110.09** (ΔPnL: $+110.09, confidence: 81%)
   - `chop_entry_floor_add` = `0.003`
   - `chop_entry_min_dir_conf` = `0.8`

3. **entry_filter: PnL +$78.70** (ΔPnL: $+78.70, confidence: 85%)
   - `min_confidence` = `0.55`
   - `min_dir_conf` = `0.65`
   - `min_entry_quality` = `0.2`
   - `min_ev` = `0.02`

4. **direction: PnL +$75.15** (ΔPnL: $+75.15, confidence: 80%)
   - `chop_prefer_short` = `True`
   - `min_dir_conf_for_entry` = `0.6`
   - `mu_alpha_sign_override` = `True`

5. **leverage: PnL +$73.33** (ΔPnL: $+73.33, confidence: 67%)
   - `max_leverage` = `20`
   - `regime_max_bull` = `20`
   - `regime_max_chop` = `3`
   - `regime_max_bear` = `5`

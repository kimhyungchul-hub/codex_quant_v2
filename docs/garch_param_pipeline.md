# GARCH Parameter Pipeline (mu_alpha)

## 1) `GARCH_PARAM_PATH` JSON format

`scripts/fit_garch_params.py` writes the following structure:

```json
{
  "version": 1,
  "fitted_at": "2026-02-08T13:10:00+00:00",
  "fit_mode": "arch_garch_11",
  "source_glob": "data/*.csv",
  "lookback_bars": 4000,
  "bar_seconds": 60.0,
  "garch": {
    "omega": 1.1e-06,
    "alpha": 0.061,
    "beta": 0.912,
    "var0": 2.4e-05,
    "n_obs": 25000
  },
  "symbols": {
    "BTC/USDT:USDT": {
      "omega": 9.8e-07,
      "alpha": 0.055,
      "beta": 0.921,
      "var0": 2.1e-05,
      "n_obs": 6200
    }
  }
}
```

Notes:
- `garch` is the global fallback used for all symbols.
- `symbols.<symbol>` is optional and overrides the global params for matching symbols.
- `var0` is optional. If present, it is used as initial variance warm-start.

## 2) Daily auto-fit pipeline

The orchestrator now runs a non-blocking daily fit process and auto-reloads JSON:

- Script: `scripts/fit_garch_params.py`
- Triggered from decision loop (throttled by interval)
- Reloaded into live alpha state immediately after successful fit

## 3) Required env vars

```bash
GARCH_PARAM_PATH=state/garch_params.json
GARCH_DAILY_FIT_ENABLED=1
GARCH_FIT_INTERVAL_SEC=86400
GARCH_FIT_DATA_GLOB=data/*.csv
GARCH_FIT_LOOKBACK=4000
GARCH_FIT_MIN_OBS=300
GARCH_FIT_TIMEOUT_SEC=120
GARCH_FIT_ALLOW_FALLBACK=1
GARCH_PARAM_RELOAD_SEC=60
GARCH_FIT_QUOTE=USDT
GARCH_FIT_EXCHANGE_SUFFIX=:USDT
GARCH_FIT_STRIP_TIMEFRAME=1
GARCH_FIT_STRIP_SUFFIXES=PERP,SWAP,FUTURES,LINEAR
```

## 4) Manual run (one-shot)

```bash
python scripts/fit_garch_params.py \
  --input-glob "data/*.csv" \
  --out "state/garch_params.json" \
  --lookback 4000 \
  --min-obs 300
```

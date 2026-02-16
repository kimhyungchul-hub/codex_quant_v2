#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np


def _extract_param(params, prefix: str, default: float = np.nan) -> float:
    try:
        if hasattr(params, "items"):
            for k, v in params.items():
                if str(k).lower().startswith(prefix):
                    return float(v)
        if hasattr(params, "index") and hasattr(params, "__getitem__"):
            for k in params.index:
                if str(k).lower().startswith(prefix):
                    return float(params[k])
        vals = np.asarray(params, dtype=np.float64).reshape(-1)
        if vals.size >= 3:
            if prefix == "omega":
                return float(vals[0])
            if prefix == "alpha":
                return float(vals[1])
            if prefix == "beta":
                return float(vals[2])
    except Exception:
        return float(default)
    return float(default)


def _sanitize_garch_params(omega: float, alpha: float, beta: float, var0: float, n_obs: int) -> Optional[Dict[str, float]]:
    try:
        omega = float(omega)
        alpha = float(alpha)
        beta = float(beta)
        var0 = float(var0)
        n_obs = int(n_obs)
    except Exception:
        return None
    if not np.isfinite(omega) or not np.isfinite(alpha) or not np.isfinite(beta):
        return None
    omega = max(omega, 1e-12)
    alpha = min(max(alpha, 1e-6), 0.999)
    beta = min(max(beta, 1e-6), 0.999)
    ab = alpha + beta
    if ab >= 0.999:
        scale = 0.999 / max(ab, 1e-12)
        alpha *= scale
        beta *= scale
    if not np.isfinite(var0) or var0 <= 0.0:
        var0 = max(omega / max(1.0 - alpha - beta, 1e-6), 1e-10)
    else:
        var0 = max(var0, 1e-10)
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "var0": float(var0),
        "n_obs": int(max(n_obs, 0)),
    }


def _normalize_raw_symbol(raw: str, quote: str, exchange_suffix: str, strip_timeframe: bool, strip_suffixes: set[str]) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if "/" in s:
        if ":" in s:
            return s
        if exchange_suffix:
            return f"{s}{exchange_suffix}"
        return s

    u = s.upper()
    if strip_timeframe:
        # Examples: BTCUSDT_1M, BTCUSDT-1H, BTCUSDT_15MIN, BTCUSDT_240
        u = re.sub(r"([_\-])(\d+[MHDW]|\d+MIN|\d+HOUR|\d+DAY|\d+)$", "", u)
    for suf in strip_suffixes:
        if suf and u.endswith(suf) and len(u) > len(suf):
            u = u[: -len(suf)]
            break
    compact = re.sub(r"[^A-Z0-9]", "", u)
    if not compact:
        return s
    q = quote.upper().strip()
    if q and compact.endswith(q) and len(compact) > len(q):
        base = compact[: -len(q)]
        return f"{base}/{q}{exchange_suffix}"
    # Fallback: keep as-is if no quote suffix match.
    return s


def _infer_symbol(
    path: str,
    symbol_hint: str | None,
    *,
    quote: str = "USDT",
    exchange_suffix: str = ":USDT",
    strip_timeframe: bool = True,
    strip_suffixes: set[str] | None = None,
) -> str:
    strip_suffixes = strip_suffixes or set()
    if symbol_hint and str(symbol_hint).strip():
        parsed = _normalize_raw_symbol(
            str(symbol_hint),
            quote=quote,
            exchange_suffix=exchange_suffix,
            strip_timeframe=strip_timeframe,
            strip_suffixes=strip_suffixes,
        )
        if parsed:
            return parsed
    base = os.path.splitext(os.path.basename(path))[0].strip()
    parsed = _normalize_raw_symbol(
        base,
        quote=quote,
        exchange_suffix=exchange_suffix,
        strip_timeframe=strip_timeframe,
        strip_suffixes=strip_suffixes,
    )
    return parsed or base


def _load_close_series(
    path: str,
    symbol_col: str = "symbol",
    *,
    quote: str = "USDT",
    exchange_suffix: str = ":USDT",
    strip_timeframe: bool = True,
    strip_suffixes: set[str] | None = None,
) -> Optional[Tuple[str, np.ndarray]]:
    closes = []
    symbol_hint = None
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None
        cols = {str(c).strip().lower(): i for i, c in enumerate(header)}
        close_idx = None
        for name in ("close", "c", "closing_price"):
            if name in cols:
                close_idx = cols[name]
                break
        if close_idx is None:
            return None
        symbol_idx = cols.get(str(symbol_col).strip().lower())
        for row in reader:
            if close_idx >= len(row):
                continue
            try:
                closes.append(float(row[close_idx]))
            except Exception:
                continue
            if symbol_idx is not None and symbol_idx < len(row) and symbol_hint is None:
                sv = str(row[symbol_idx]).strip()
                if sv:
                    symbol_hint = sv
    if len(closes) < 20:
        return None
    arr = np.asarray(closes, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    if arr.size < 20:
        return None
    symbol = _infer_symbol(
        path,
        symbol_hint,
        quote=quote,
        exchange_suffix=exchange_suffix,
        strip_timeframe=strip_timeframe,
        strip_suffixes=strip_suffixes or set(),
    )
    return symbol, arr


def _fit_one_garch(log_returns: np.ndarray) -> Optional[Dict[str, float]]:
    try:
        from arch import arch_model
    except Exception:
        return None
    if log_returns is None or len(log_returns) < 20:
        return None
    y = np.asarray(log_returns, dtype=np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size < 20:
        return None
    y_pct = y * 100.0
    try:
        model = arch_model(y_pct, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
        res = model.fit(disp="off", show_warning=False, update_freq=0)
    except Exception:
        return None
    params = getattr(res, "params", None)
    if params is None:
        return None
    omega_pct2 = _extract_param(params, "omega")
    alpha = _extract_param(params, "alpha")
    beta = _extract_param(params, "beta")
    try:
        cond_vol = np.asarray(getattr(res, "conditional_volatility"), dtype=np.float64).reshape(-1)
        var0 = float(max((cond_vol[-1] / 100.0) ** 2, 1e-12)) if cond_vol.size else np.nan
    except Exception:
        var0 = np.nan
    omega = float(omega_pct2) / (100.0 * 100.0)
    return _sanitize_garch_params(omega, alpha, beta, var0, int(y.size))


def _weighted_average(params_by_symbol: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not params_by_symbol:
        return None
    items = list(params_by_symbol.values())
    weights = np.asarray([max(float(it.get("n_obs", 0)), 1.0) for it in items], dtype=np.float64)
    sw = float(np.sum(weights))
    if sw <= 0:
        return None
    omega = float(np.sum([float(it["omega"]) * w for it, w in zip(items, weights)]) / sw)
    alpha = float(np.sum([float(it["alpha"]) * w for it, w in zip(items, weights)]) / sw)
    beta = float(np.sum([float(it["beta"]) * w for it, w in zip(items, weights)]) / sw)
    var0 = float(np.sum([float(it.get("var0", 1e-6)) * w for it, w in zip(items, weights)]) / sw)
    n_obs = int(np.sum([int(it.get("n_obs", 0)) for it in items]))
    return _sanitize_garch_params(omega, alpha, beta, var0, n_obs)


def _write_json(path: str, payload: Dict) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _fallback_payload(args, reason: str) -> Dict:
    g = _sanitize_garch_params(
        args.fallback_omega,
        args.fallback_alpha,
        args.fallback_beta,
        args.fallback_var0,
        0,
    )
    return {
        "version": 1,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "fit_mode": "fallback",
        "fit_reason": str(reason),
        "source_glob": str(args.input_glob),
        "lookback_bars": int(args.lookback),
        "bar_seconds": float(args.bar_seconds),
        "garch": g,
        "symbols": {},
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Fit GARCH(1,1) parameters from OHLCV CSV files.")
    p.add_argument("--input-glob", default=os.environ.get("GARCH_FIT_DATA_GLOB", "data/*.csv"))
    p.add_argument("--out", default=os.environ.get("GARCH_PARAM_PATH", "state/garch_params.json"))
    p.add_argument("--lookback", type=int, default=int(os.environ.get("GARCH_FIT_LOOKBACK", 4000)))
    p.add_argument("--min-obs", type=int, default=int(os.environ.get("GARCH_FIT_MIN_OBS", 300)))
    p.add_argument("--bar-seconds", type=float, default=float(os.environ.get("GARCH_FIT_BAR_SECONDS", 60.0)))
    p.add_argument("--symbol-col", default=os.environ.get("GARCH_FIT_SYMBOL_COL", "symbol"))
    p.add_argument("--quote", default=os.environ.get("GARCH_FIT_QUOTE", "USDT"))
    p.add_argument("--exchange-suffix", default=os.environ.get("GARCH_FIT_EXCHANGE_SUFFIX", ":USDT"))
    p.add_argument(
        "--strip-suffixes",
        default=os.environ.get("GARCH_FIT_STRIP_SUFFIXES", "PERP,SWAP,FUTURES,LINEAR"),
        help="Comma-separated trailing tokens stripped before mapping, e.g. PERP,SWAP",
    )
    p.add_argument(
        "--strip-timeframe",
        type=int,
        default=1 if str(os.environ.get("GARCH_FIT_STRIP_TIMEFRAME", "1")).strip().lower() in ("1", "true", "yes", "on") else 0,
        help="Strip timeframe suffix from filename/symbol tokens (1=on, 0=off)",
    )
    p.add_argument("--allow-fallback", type=int, default=1 if str(os.environ.get("GARCH_FIT_ALLOW_FALLBACK", "1")).strip().lower() in ("1", "true", "yes", "on") else 0)
    p.add_argument("--fallback-omega", type=float, default=float(os.environ.get("GARCH_OMEGA", 1e-6)))
    p.add_argument("--fallback-alpha", type=float, default=float(os.environ.get("GARCH_ALPHA", 0.05)))
    p.add_argument("--fallback-beta", type=float, default=float(os.environ.get("GARCH_BETA", 0.90)))
    p.add_argument("--fallback-var0", type=float, default=float(os.environ.get("GARCH_VAR0", 1e-6)))
    args = p.parse_args()

    paths = sorted(glob.glob(str(args.input_glob)))
    by_symbol: Dict[str, Dict[str, float]] = {}
    used_files = 0

    arch_ok = True
    try:
        import arch  # noqa: F401
    except Exception:
        arch_ok = False

    if arch_ok:
        strip_suffixes = {x.strip().upper() for x in str(args.strip_suffixes or "").split(",") if x.strip()}
        quote = str(args.quote or "USDT").strip().upper()
        exchange_suffix = str(args.exchange_suffix or ":USDT").strip()
        strip_timeframe = bool(int(args.strip_timeframe))
        for path in paths:
            loaded = _load_close_series(
                path,
                symbol_col=args.symbol_col,
                quote=quote,
                exchange_suffix=exchange_suffix,
                strip_timeframe=strip_timeframe,
                strip_suffixes=strip_suffixes,
            )
            if loaded is None:
                continue
            symbol, closes = loaded
            lookback = int(max(args.lookback, 20))
            closes = closes[-(lookback + 1) :]
            if closes.size < int(max(args.min_obs, 20)):
                continue
            log_returns = np.diff(np.log(np.maximum(closes, 1e-12)))
            if log_returns.size < int(max(args.min_obs, 20)):
                continue
            fitted = _fit_one_garch(log_returns)
            if fitted is None:
                continue
            by_symbol[symbol] = fitted
            used_files += 1

    if not by_symbol:
        if int(args.allow_fallback) != 1:
            print(f"[garch_fit] failed: no fitted symbols from {args.input_glob}")
            if not arch_ok:
                print("[garch_fit] arch library is missing. Install with `pip install arch`.")
            return 1
        reason = "no_arch" if not arch_ok else "no_usable_data"
        payload = _fallback_payload(args, reason=reason)
        _write_json(args.out, payload)
        print(f"[garch_fit] fallback written -> {args.out} reason={reason}")
        return 0

    garch_global = _weighted_average(by_symbol)
    if garch_global is None:
        if int(args.allow_fallback) != 1:
            return 1
        payload = _fallback_payload(args, reason="aggregate_failed")
        _write_json(args.out, payload)
        print(f"[garch_fit] fallback written -> {args.out} reason=aggregate_failed")
        return 0

    payload = {
        "version": 1,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "fit_mode": "arch_garch_11",
        "source_glob": str(args.input_glob),
        "lookback_bars": int(args.lookback),
        "bar_seconds": float(args.bar_seconds),
        "file_count": int(len(paths)),
        "used_file_count": int(used_files),
        "symbol_count": int(len(by_symbol)),
        "garch": garch_global,
        "symbols": {k: by_symbol[k] for k in sorted(by_symbol.keys())},
    }
    _write_json(args.out, payload)
    print(
        "[garch_fit] ok "
        f"symbols={len(by_symbol)} used_files={used_files} out={args.out} "
        f"omega={garch_global['omega']:.3e} alpha={garch_global['alpha']:.4f} beta={garch_global['beta']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

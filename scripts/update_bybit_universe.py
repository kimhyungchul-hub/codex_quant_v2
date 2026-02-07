#!/usr/bin/env python3
"""
Update SYMBOLS_CSV to Bybit top-N 24h quote-volume (USDT linear perpetual).
Writes to state/bybit.env (preserves existing keys).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt


def _safe_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _quote_volume(ticker: dict) -> float:
    # Prefer unified fields
    for k in ("quoteVolume", "quote_volume", "quoteVol"):
        if k in ticker and ticker[k] is not None:
            return _safe_float(ticker[k])
    info = ticker.get("info") or {}
    for k in ("turnover24h", "turnover_24h", "quote_volume_24h", "quoteVolume", "quoteVol"):
        if k in info and info[k] is not None:
            return _safe_float(info[k])
    # Fallback: baseVolume * last
    base = ticker.get("baseVolume")
    last = ticker.get("last")
    if base is not None and last is not None:
        return _safe_float(base) * _safe_float(last)
    return 0.0


def _load_env(path: Path) -> Tuple[Dict[str, str], List[str]]:
    if not path.exists():
        return {}, []
    lines = path.read_text(encoding="utf-8").splitlines()
    env: Dict[str, str] = {}
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        env[k.strip()] = v.strip()
    return env, lines


def _write_env(path: Path, lines: List[str], key: str, value: str) -> None:
    if not lines:
        path.write_text(f"{key}={value}\n", encoding="utf-8")
        return
    replaced = False
    out: List[str] = []
    for line in lines:
        raw = line.strip()
        if raw and not raw.startswith("#") and raw.split("=", 1)[0].strip() == key:
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=30, help="Top N symbols by 24h quote volume.")
    ap.add_argument("--out", type=str, default="state/bybit.env", help="Env file to write SYMBOLS_CSV.")
    ap.add_argument("--include-inactive", action="store_true", help="Include inactive markets.")
    args = ap.parse_args()

    ex = ccxt.bybit({"enableRateLimit": True})
    ex.load_markets()

    # USDT linear perpetual only
    candidates: List[str] = []
    for sym, m in ex.markets.items():
        if not m:
            continue
        if not m.get("swap"):
            continue
        if not m.get("linear"):
            continue
        if m.get("quote") != "USDT":
            continue
        if m.get("settle") != "USDT":
            continue
        if not args.include_inactive and (m.get("active") is False):
            continue
        candidates.append(sym)

    tickers = ex.fetch_tickers(candidates)
    ranked: List[Tuple[str, float]] = []
    for sym, t in (tickers or {}).items():
        vol = _quote_volume(t or {})
        if vol <= 0:
            continue
        ranked.append((sym, vol))

    ranked.sort(key=lambda x: x[1], reverse=True)
    top_syms = [s for s, _ in ranked[: max(1, int(args.top_n))]]

    out_path = Path(args.out)
    env, lines = _load_env(out_path)
    _write_env(out_path, lines, "SYMBOLS_CSV", ",".join(top_syms))

    print(f"Wrote {len(top_syms)} symbols to {out_path}")
    for s in top_syms:
        print(s)


if __name__ == "__main__":
    main()

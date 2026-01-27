#!/usr/bin/env python3
"""Fetch OHLCV for symbols in config.SYMBOLS and save CSVs into `data/`.

Behavior:
- Uses ccxt (public) to fetch OHLCV per symbol
- Symbol conversion: takes left of ':' in config.SYMBOLS (e.g. 'BTC/USDT:USDT' -> 'BTC/USDT')
- Saves CSV as `data/{BASEQUOTE}.csv` (e.g. BTCUSDT.csv) with columns timestamp, open, high, low, close, volume
"""
import os
import time
import math
import pandas as pd

try:
    import ccxt
except Exception:
    raise SystemExit("ccxt not installed. Run `pip install ccxt` in the virtualenv.")

from config import SYMBOLS, TIMEFRAME

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

exchange = ccxt.binance({
    'enableRateLimit': True,
    # public requests only
})

LIMIT = 1000


def normalize_symbol(s: str) -> str:
    """Convert config symbol to exchange symbol. e.g. 'BTC/USDT:USDT' -> 'BTC/USDT'"""
    if ':' in s:
        s = s.split(':', 1)[0]
    return s


def filename_for_symbol(s: str) -> str:
    return s.replace('/', '') + '.csv'


def fetch_and_save(sym: str) -> bool:
    ex_sym = normalize_symbol(sym)
    out_path = os.path.join(OUT_DIR, filename_for_symbol(ex_sym))
    try:
        print(f"Fetching {ex_sym} from {exchange.id} timeframe={TIMEFRAME} limit={LIMIT}")
        ohlcv = exchange.fetch_ohlcv(ex_sym, timeframe=TIMEFRAME, limit=LIMIT)
        if not ohlcv:
            print(f"  No OHLCV returned for {ex_sym}")
            return False
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # convert ms -> datetime string optional
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows -> {out_path}")
        return True
    except ccxt.BaseError as e:
        print(f"  Exchange error for {ex_sym}: {e}")
        return False
    except Exception as e:
        print(f"  Error fetching {ex_sym}: {e}")
        return False


if __name__ == '__main__':
    results = {}
    for s in SYMBOLS:
        ok = fetch_and_save(s)
        results[s] = ok
        time.sleep(1.0)

    print('\nFetch summary:')
    for s, ok in results.items():
        print(f"  {s} -> {'OK' if ok else 'FAIL'}")

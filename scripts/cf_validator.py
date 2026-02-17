#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.cf_engine import TradeLoader, compute_metrics


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _slice_oos(trades: list[Any], oos_ratio: float) -> tuple[list[Any], list[Any]]:
    n = len(trades)
    if n <= 1:
        return list(trades), []
    ratio = max(0.05, min(0.50, float(oos_ratio)))
    cut = int(round(n * (1.0 - ratio)))
    cut = max(1, min(n - 1, cut))
    return trades[:cut], trades[cut:]


def main() -> int:
    ap = argparse.ArgumentParser(description="CF-engine based validator metrics (overall + IS/OOS)")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--limit", type=int, default=0, help="0 means all trades")
    ap.add_argument("--since-ms", type=int, default=0, help="filter trades by exit timestamp_ms")
    ap.add_argument("--oos-ratio", type=float, default=0.20, help="tail ratio used as out-of-sample")
    ap.add_argument("--exclude-symbols", default="", help="CSV symbols to exclude")
    args = ap.parse_args()

    t0 = time.time()
    db_path = str(Path(args.db))

    try:
        loader = TradeLoader(db_path=db_path)
        trades = loader.load_trades(
            limit=int(args.limit),
            since_ms=int(args.since_ms),
            exclude_symbols=args.exclude_symbols or None,
        )
        if not trades:
            out = {
                "status": "no_trades",
                "source": "cf_engine_metrics",
                "timestamp": int(time.time() * 1000),
                "config": {
                    "db": db_path,
                    "limit": int(args.limit),
                    "since_ms": int(args.since_ms),
                    "oos_ratio": float(args.oos_ratio),
                    "exclude_symbols": args.exclude_symbols or "",
                },
                "overall": compute_metrics([]),
                "in_sample": compute_metrics([]),
                "out_of_sample": compute_metrics([]),
                "elapsed_sec": round(time.time() - t0, 4),
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0

        trades = sorted(trades, key=lambda x: int(getattr(x, "timestamp_ms", 0) or 0))
        in_sample, out_of_sample = _slice_oos(trades, float(args.oos_ratio))

        overall_metrics = compute_metrics(trades)
        in_sample_metrics = compute_metrics(in_sample)
        out_of_sample_metrics = compute_metrics(out_of_sample)
        out = {
            "status": "ok",
            "source": "cf_engine_metrics",
            "timestamp": int(time.time() * 1000),
            "config": {
                "db": db_path,
                "limit": int(args.limit),
                "since_ms": int(args.since_ms),
                "oos_ratio": float(max(0.05, min(0.50, float(args.oos_ratio)))),
                "exclude_symbols": args.exclude_symbols or "",
            },
            "n_trades": int(len(trades)),
            "split": {
                "in_sample_n": int(len(in_sample)),
                "out_of_sample_n": int(len(out_of_sample)),
                "out_of_sample_ratio": float(len(out_of_sample) / max(1, len(trades))),
                "ts_min_ms": int(getattr(trades[0], "timestamp_ms", 0) or 0),
                "ts_max_ms": int(getattr(trades[-1], "timestamp_ms", 0) or 0),
            },
            "overall": overall_metrics,
            "in_sample": in_sample_metrics,
            "out_of_sample": out_of_sample_metrics,
            "summary": {
                "wr": _safe_float(overall_metrics.get("wr")),
                "sharpe": _safe_float(overall_metrics.get("sharpe")),
                "max_dd": _safe_float(overall_metrics.get("max_dd")),
                "avg_pnl": _safe_float(overall_metrics.get("avg_pnl")),
                "pnl": _safe_float(overall_metrics.get("pnl")),
                "oos_wr": _safe_float(out_of_sample_metrics.get("wr")),
                "oos_sharpe": _safe_float(out_of_sample_metrics.get("sharpe")),
                "oos_max_dd": _safe_float(out_of_sample_metrics.get("max_dd")),
                "oos_avg_pnl": _safe_float(out_of_sample_metrics.get("avg_pnl")),
                "oos_pnl": _safe_float(out_of_sample_metrics.get("pnl")),
            },
            "elapsed_sec": round(time.time() - t0, 4),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    except Exception as e:
        out = {
            "status": "error",
            "source": "cf_engine_metrics",
            "timestamp": int(time.time() * 1000),
            "error": str(e),
            "config": {
                "db": db_path,
                "limit": int(args.limit),
                "since_ms": int(args.since_ms),
                "oos_ratio": float(args.oos_ratio),
                "exclude_symbols": args.exclude_symbols or "",
            },
            "elapsed_sec": round(time.time() - t0, 4),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

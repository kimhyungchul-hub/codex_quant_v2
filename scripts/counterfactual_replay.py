#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
ONE_MIN_MS = 60_000


@dataclass
class EntryRow:
    id: int
    symbol: str
    side: str
    fill_price: float
    ts_ms: int
    entry_ev: float | None
    entry_link_id: str | None = None


@dataclass
class ExitRow:
    id: int
    symbol: str
    side: str
    fill_price: float
    ts_ms: int
    hold_sec: float
    reason: str
    roe: float | None
    action: str
    entry_link_id: str | None = None


@dataclass
class SymbolSeries:
    times_ms: list[int]
    close: list[float]


def _to_bybit_symbol(ccxt_symbol: str) -> str:
    # e.g. BTC/USDT:USDT -> BTCUSDT
    base = ccxt_symbol.split("/")[0].strip().upper()
    return f"{base}USDT"


def _fetch_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _fetch_klines_1m(bybit_symbol: str, start_ms: int, end_ms: int) -> SymbolSeries:
    all_rows: list[tuple[int, float]] = []
    cursor = int(start_ms)
    while cursor < end_ms:
        chunk_end = min(end_ms, cursor + 999 * ONE_MIN_MS)
        params = {
            "category": "linear",
            "symbol": bybit_symbol,
            "interval": "1",
            "start": str(cursor),
            "end": str(chunk_end),
            "limit": "1000",
        }
        url = f"{BYBIT_KLINE_URL}?{urllib.parse.urlencode(params)}"
        data = _fetch_json(url)
        if str(data.get("retCode", -1)) != "0":
            raise RuntimeError(f"Bybit API error for {bybit_symbol}: {data.get('retMsg')}")
        rows = (data.get("result") or {}).get("list") or []
        if not rows:
            cursor = chunk_end + ONE_MIN_MS
            time.sleep(0.02)
            continue
        for row in rows:
            # [startTime, open, high, low, close, volume, turnover]
            t = int(row[0])
            c = float(row[4])
            all_rows.append((t, c))
        cursor = chunk_end + ONE_MIN_MS
        time.sleep(0.02)

    if not all_rows:
        return SymbolSeries(times_ms=[], close=[])

    # Bybit list ordering can vary; normalize ascending unique timestamps.
    uniq = {}
    for t, c in all_rows:
        uniq[t] = c
    times = sorted(uniq.keys())
    close = [uniq[t] for t in times]
    return SymbolSeries(times_ms=times, close=close)


def _load_entries(conn: sqlite3.Connection) -> list[EntryRow]:
    rows = conn.execute(
        """
        SELECT id, symbol, side, fill_price, timestamp_ms, entry_ev, entry_link_id
        FROM trades
        WHERE action IN ('ENTER','SPREAD') AND fill_price IS NOT NULL AND fill_price > 0
        ORDER BY timestamp_ms ASC, id ASC
        """
    ).fetchall()
    out: list[EntryRow] = []
    for r in rows:
        out.append(
            EntryRow(
                id=int(r[0]),
                symbol=str(r[1]),
                side=str(r[2]).upper(),
                fill_price=float(r[3]),
                ts_ms=int(r[4]),
                entry_ev=None if r[5] is None else float(r[5]),
                entry_link_id=str(r[6]).strip() if r[6] is not None and str(r[6]).strip() else None,
            )
        )
    return out


def _load_exits(conn: sqlite3.Connection, since_id: int = 0) -> list[ExitRow]:
    if int(since_id) > 0:
        rows = conn.execute(
            """
            SELECT id, symbol, side, fill_price, timestamp_ms, hold_duration_sec, entry_reason, roe, action, entry_link_id
            FROM trades
            WHERE action IN ('EXIT','REBAL_EXIT')
              AND fill_price IS NOT NULL
              AND fill_price > 0
              AND hold_duration_sec IS NOT NULL
              AND hold_duration_sec > 0
              AND id > ?
            ORDER BY timestamp_ms ASC, id ASC
            """,
            (int(since_id),),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, symbol, side, fill_price, timestamp_ms, hold_duration_sec, entry_reason, roe, action, entry_link_id
            FROM trades
            WHERE action IN ('EXIT','REBAL_EXIT')
              AND fill_price IS NOT NULL
              AND fill_price > 0
              AND hold_duration_sec IS NOT NULL
              AND hold_duration_sec > 0
            ORDER BY timestamp_ms ASC, id ASC
            """
        ).fetchall()
    out: list[ExitRow] = []
    for r in rows:
        out.append(
            ExitRow(
                id=int(r[0]),
                symbol=str(r[1]),
                side=str(r[2]).upper(),
                fill_price=float(r[3]),
                ts_ms=int(r[4]),
                hold_sec=float(r[5]),
                reason=str(r[6] or ""),
                roe=None if r[7] is None else float(r[7]),
                action=str(r[8]),
                entry_link_id=str(r[9]).strip() if r[9] is not None and str(r[9]).strip() else None,
            )
        )
    return out


def _bucket_index(times_ms: list[int], ts_ms: int) -> int:
    idx = bisect.bisect_right(times_ms, ts_ms) - 1
    return idx


def _side_sign(side: str) -> float:
    return 1.0 if side.upper() == "LONG" else -1.0


def _signed_ret(side: str, entry_px: float, next_px: float) -> float:
    raw = (next_px / max(1e-12, entry_px)) - 1.0
    return raw * _side_sign(side)


def _get_return_at_h(
    series: SymbolSeries, entry_ts_ms: int, entry_px: float, side: str, h_min: int
) -> float | None:
    if h_min < 1:
        h_min = 1
    idx0 = _bucket_index(series.times_ms, entry_ts_ms)
    if idx0 < 0:
        return None
    idx1 = idx0 + int(h_min)
    if idx1 >= len(series.close):
        return None
    return _signed_ret(side, entry_px, series.close[idx1])


def _best_hold_ret(
    series: SymbolSeries,
    entry_ts_ms: int,
    entry_px: float,
    side: str,
    horizons_min: list[int],
) -> tuple[float | None, int | None, dict[int, float]]:
    out_h: dict[int, float] = {}
    best_r = None
    best_h = None
    for h in horizons_min:
        r = _get_return_at_h(series, entry_ts_ms, entry_px, side, h)
        if r is None:
            continue
        out_h[int(h)] = float(r)
        if best_r is None or r > best_r:
            best_r = float(r)
            best_h = int(h)
    return best_r, best_h, out_h


def _safe_mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = int(round((len(ys) - 1) * q))
    i = max(0, min(len(ys) - 1, i))
    return float(ys[i])


def _build_entry_index(entries: list[EntryRow]) -> dict[tuple[str, str], list[EntryRow]]:
    mp: dict[tuple[str, str], list[EntryRow]] = defaultdict(list)
    for e in entries:
        mp[(e.symbol, e.side)].append(e)
    for k in list(mp.keys()):
        mp[k].sort(key=lambda x: (x.ts_ms, x.id))
    return mp


def _build_entry_link_index(entries: list[EntryRow]) -> dict[str, EntryRow]:
    out: dict[str, EntryRow] = {}
    for e in entries:
        if not e.entry_link_id:
            continue
        out[e.entry_link_id] = e
    return out


def _match_exits_to_entries(
    exits: list[ExitRow],
    entry_index: dict[tuple[str, str], list[EntryRow]],
    entry_link_index: dict[str, EntryRow],
    tolerance_sec: int = 240,
    fallback_lookback_sec: int = 14_400,
) -> list[tuple[ExitRow, EntryRow, str]]:
    used_ids: set[int] = set()
    out: list[tuple[ExitRow, EntryRow, str]] = []
    tol_ms = int(tolerance_sec * 1000)
    fb_ms = int(fallback_lookback_sec * 1000)
    for ex in exits:
        # Best path: direct link-based match from persisted entry_link_id.
        if ex.entry_link_id:
            linked = entry_link_index.get(ex.entry_link_id)
            if linked is not None and linked.id not in used_ids:
                used_ids.add(linked.id)
                out.append((ex, linked, "entry_link_id"))
                continue
        key = (ex.symbol, ex.side)
        arr = entry_index.get(key) or []
        if not arr:
            continue
        est_entry_ts = int(ex.ts_ms - max(1.0, ex.hold_sec) * 1000.0)
        best = None
        best_dist = None
        match_mode = None
        # Linear scan is acceptable for current data size.
        for en in arr:
            if en.id in used_ids:
                continue
            if en.ts_ms > ex.ts_ms:
                break
            d = abs(en.ts_ms - est_entry_ts)
            if d > tol_ms:
                continue
            if best is None or d < best_dist:
                best = en
                best_dist = d
                match_mode = "strict_hold_time"
        if best is None:
            # Fallback: nearest previous unmatched entry in a wider window.
            for en in arr:
                if en.id in used_ids:
                    continue
                if en.ts_ms > ex.ts_ms:
                    break
                if (ex.ts_ms - en.ts_ms) > fb_ms:
                    continue
                d = abs(en.ts_ms - est_entry_ts)
                if best is None or d < best_dist:
                    best = en
                    best_dist = d
                    match_mode = "fallback_prev_unmatched"
        if best is None:
            continue
        used_ids.add(best.id)
        out.append((ex, best, str(match_mode or "unknown")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Counterfactual replay for entry/exit diagnosis on live trade history.")
    ap.add_argument("--db", default="state/bot_data_live.db", help="SQLite DB path")
    ap.add_argument("--out", default="state/counterfactual_replay_report.json", help="Output report path")
    ap.add_argument("--max-hold-min", type=int, default=60, help="Max hold horizon (minutes)")
    ap.add_argument("--entry-sample-limit", type=int, default=0, help="Optional cap on number of entries (0=all)")
    ap.add_argument("--since-id", type=int, default=0, help="Only include EXIT rows with id > since_id")
    ap.add_argument("--exit-match-tolerance-sec", type=int, default=240, help="Tolerance for matching EXIT to ENTER")
    ap.add_argument(
        "--exit-match-fallback-lookback-sec",
        type=int,
        default=14_400,
        help="Fallback lookback for unmatched EXIT->ENTER pairing (seconds)",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    horizons = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120]
    horizons = [h for h in horizons if h <= int(args.max_hold_min)]
    if not horizons:
        horizons = [1, 3, 5, 10, 15, 30, 60]

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    entries = _load_entries(conn)
    exits = _load_exits(conn, since_id=int(args.since_id))
    conn.close()

    if args.entry_sample_limit and args.entry_sample_limit > 0:
        entries = entries[-int(args.entry_sample_limit) :]

    if not entries:
        raise SystemExit("no entries found")

    by_symbol: dict[str, list[EntryRow]] = defaultdict(list)
    for e in entries:
        by_symbol[e.symbol].append(e)

    # Fetch all needed symbol series once (max sample).
    series_map: dict[str, SymbolSeries] = {}
    fetch_errors: dict[str, str] = {}
    for symbol, arr in sorted(by_symbol.items()):
        bybit_symbol = _to_bybit_symbol(symbol)
        min_ts = min(e.ts_ms for e in arr) - 5 * ONE_MIN_MS
        max_ts = max(e.ts_ms for e in arr) + (max(horizons) + 2) * ONE_MIN_MS
        try:
            s = _fetch_klines_1m(bybit_symbol, min_ts, max_ts)
            series_map[symbol] = s
        except Exception as exc:
            fetch_errors[symbol] = str(exc)

    # Entry-only counterfactual: direction + best hold potential.
    entry_eval_n = 0
    entry_side_profitable_n = 0
    entry_opp_better_n = 0
    best_ret_vals: list[float] = []
    side_ret_15m_vals: list[float] = []
    dir_regret_vals: list[float] = []

    for e in entries:
        s = series_map.get(e.symbol)
        if not s or not s.times_ms:
            continue
        best_side_ret, best_h, hmap_side = _best_hold_ret(s, e.ts_ms, e.fill_price, e.side, horizons)
        if best_side_ret is None:
            continue
        opp_side = "SHORT" if e.side == "LONG" else "LONG"
        best_opp_ret, _, _ = _best_hold_ret(s, e.ts_ms, e.fill_price, opp_side, horizons)
        if best_opp_ret is None:
            continue

        entry_eval_n += 1
        best_ret_vals.append(float(best_side_ret))
        if best_side_ret > 0:
            entry_side_profitable_n += 1
        if best_opp_ret > best_side_ret:
            entry_opp_better_n += 1
        dir_regret_vals.append(float(best_opp_ret - best_side_ret))

        r15 = hmap_side.get(15)
        if r15 is not None:
            side_ret_15m_vals.append(float(r15))

    # Exit-timing counterfactual: matched exits vs same-entry best hold.
    entry_index = _build_entry_index(entries)
    entry_link_index = _build_entry_link_index(entries)
    matched = _match_exits_to_entries(
        exits,
        entry_index,
        entry_link_index,
        tolerance_sec=int(args.exit_match_tolerance_sec),
        fallback_lookback_sec=int(args.exit_match_fallback_lookback_sec),
    )

    exit_eval_n = 0
    exit_improvable_n = 0
    exit_actual_ret_vals: list[float] = []
    exit_best_ret_vals: list[float] = []
    exit_regret_vals: list[float] = []
    early_like_n = 0
    late_like_n = 0
    by_reason: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "avg_regret": 0.0})
    by_reason_buf: dict[str, list[float]] = defaultdict(list)
    match_mode_counter: dict[str, int] = defaultdict(int)

    for ex, en, match_mode in matched:
        s = series_map.get(en.symbol)
        if not s or not s.times_ms:
            continue
        match_mode_counter[match_mode] += 1
        best_ret, best_h, _ = _best_hold_ret(s, en.ts_ms, en.fill_price, en.side, horizons)
        if best_ret is None or best_h is None:
            continue
        h_actual = int(max(1, round(float(ex.hold_sec) / 60.0)))
        actual_ret = _get_return_at_h(s, en.ts_ms, en.fill_price, en.side, h_actual)
        if actual_ret is None:
            continue
        exit_eval_n += 1
        regret = float(best_ret - actual_ret)
        exit_actual_ret_vals.append(float(actual_ret))
        exit_best_ret_vals.append(float(best_ret))
        exit_regret_vals.append(regret)
        if regret > 0.0010:
            exit_improvable_n += 1
        if h_actual < int(0.8 * best_h):
            early_like_n += 1
        elif h_actual > int(1.2 * best_h):
            late_like_n += 1

        rs = ex.reason or ex.action
        by_reason_buf[rs].append(regret)

    for rs, vals in by_reason_buf.items():
        by_reason[rs]["n"] = float(len(vals))
        by_reason[rs]["avg_regret"] = _safe_mean(vals)

    reason_top = sorted(
        (
            {
                "reason": rs,
                "n": int(v["n"]),
                "avg_regret": float(v["avg_regret"]),
            }
            for rs, v in by_reason.items()
        ),
        key=lambda x: x["avg_regret"],
        reverse=True,
    )[:15]

    report: dict[str, Any] = {
        "timestamp_ms": int(time.time() * 1000),
        "config": {
            "db": str(db_path),
            "out": str(args.out),
            "since_id": int(args.since_id),
            "max_hold_min": int(args.max_hold_min),
            "horizons_min": horizons,
            "entry_sample_limit": int(args.entry_sample_limit),
            "exit_match_tolerance_sec": int(args.exit_match_tolerance_sec),
            "exit_match_fallback_lookback_sec": int(args.exit_match_fallback_lookback_sec),
        },
        "coverage": {
            "entries_total": int(len(entries)),
            "exits_total": int(len(exits)),
            "symbols_total": int(len(by_symbol)),
            "symbols_fetched": int(len(series_map)),
            "symbols_fetch_error": int(len(fetch_errors)),
            "entry_eval_n": int(entry_eval_n),
            "matched_exit_pairs": int(len(matched)),
            "exit_eval_n": int(exit_eval_n),
            "matched_exit_pair_modes": {k: int(v) for k, v in sorted(match_mode_counter.items())},
        },
        "entry_counterfactual": {
            "side_profitable_rate_best_h": float(entry_side_profitable_n / entry_eval_n) if entry_eval_n else None,
            "opp_side_better_rate": float(entry_opp_better_n / entry_eval_n) if entry_eval_n else None,
            "avg_best_side_ret": _safe_mean(best_ret_vals),
            "avg_side_ret_15m": _safe_mean(side_ret_15m_vals),
            "avg_direction_regret": _safe_mean(dir_regret_vals),
            "p50_direction_regret": _quantile(dir_regret_vals, 0.50),
            "p90_direction_regret": _quantile(dir_regret_vals, 0.90),
        },
        "exit_counterfactual": {
            "avg_actual_ret_at_real_hold": _safe_mean(exit_actual_ret_vals),
            "avg_best_ret_within_horizon": _safe_mean(exit_best_ret_vals),
            "avg_exit_regret": _safe_mean(exit_regret_vals),
            "p50_exit_regret": _quantile(exit_regret_vals, 0.50),
            "p90_exit_regret": _quantile(exit_regret_vals, 0.90),
            "improvable_rate_regret_gt_10bps": float(exit_improvable_n / exit_eval_n) if exit_eval_n else None,
            "early_like_rate": float(early_like_n / exit_eval_n) if exit_eval_n else None,
            "late_like_rate": float(late_like_n / exit_eval_n) if exit_eval_n else None,
            "top_regret_reasons": reason_top,
        },
        "fetch_errors": fetch_errors,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

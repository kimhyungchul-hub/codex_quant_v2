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
    entry_confidence: float | None = None
    regime: str | None = None
    fee: float | None = None
    notional: float | None = None
    pred_mu_alpha: float | None = None
    pred_mu_dir_conf: float | None = None
    alpha_vpin: float | None = None
    alpha_hurst: float | None = None
    entry_quality_score: float | None = None
    one_way_move_score: float | None = None
    leverage_signal_score: float | None = None
    policy_score_threshold: float | None = None
    policy_event_exit_min_score: float | None = None
    policy_unrealized_dd_floor: float | None = None


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


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        out = float(v)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _canonical_regime(raw: str | None) -> str:
    txt = str(raw or "").strip().lower()
    if ("trend" in txt) or (txt in ("bull", "bear")):
        return "trend"
    if ("vol" in txt) or ("noise" in txt) or ("random" in txt):
        return "volatile"
    return "chop"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys, strict=False):
        dx = float(x - mx)
        dy = float(y - my)
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    den = math.sqrt(max(1e-18, sxx) * max(1e-18, syy))
    if den <= 0:
        return None
    return float(sxy / den)


def _load_entries(conn: sqlite3.Connection) -> list[EntryRow]:
    rows = conn.execute(
        """
        SELECT
            id,
            symbol,
            side,
            fill_price,
            timestamp_ms,
            entry_ev,
            entry_link_id,
            entry_confidence,
            regime,
            fee,
            notional,
            pred_mu_alpha,
            pred_mu_dir_conf,
            alpha_vpin,
            alpha_hurst,
            entry_quality_score,
            one_way_move_score,
            leverage_signal_score,
            policy_score_threshold,
            policy_event_exit_min_score,
            policy_unrealized_dd_floor
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
                entry_confidence=_safe_float(r[7], None),
                regime=str(r[8]).strip() if r[8] is not None and str(r[8]).strip() else None,
                fee=_safe_float(r[9], None),
                notional=_safe_float(r[10], None),
                pred_mu_alpha=_safe_float(r[11], None),
                pred_mu_dir_conf=_safe_float(r[12], None),
                alpha_vpin=_safe_float(r[13], None),
                alpha_hurst=_safe_float(r[14], None),
                entry_quality_score=_safe_float(r[15], None),
                one_way_move_score=_safe_float(r[16], None),
                leverage_signal_score=_safe_float(r[17], None),
                policy_score_threshold=_safe_float(r[18], None),
                policy_event_exit_min_score=_safe_float(r[19], None),
                policy_unrealized_dd_floor=_safe_float(r[20], None),
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


def _signed_ret_and_ratio(side: str, entry_px: float, next_px: float) -> tuple[float, float]:
    ratio = next_px / max(1e-12, entry_px)
    return ((ratio - 1.0) * _side_sign(side), ratio)


def _entry_fee_rate(entry: EntryRow, default_fee_rate: float) -> float:
    fee = _safe_float(entry.fee, None)
    notional = _safe_float(entry.notional, None)
    if fee is not None and notional is not None and notional > 0:
        return float(max(0.0, abs(float(fee)) / float(notional)))
    return float(max(0.0, default_fee_rate))


def _return_cost_rate(
    *,
    entry_fee_rate: float,
    exit_fee_rate: float,
    side_slippage_bps: float,
    carry_bps_per_hour: float,
    price_ratio: float,
    h_min: int,
) -> float:
    slip_rate = max(0.0, float(side_slippage_bps)) / 10_000.0
    carry_rate_per_min = (max(0.0, float(carry_bps_per_hour)) / 10_000.0) / 60.0
    return float(
        max(0.0, float(entry_fee_rate))
        + max(0.0, float(exit_fee_rate)) * max(0.0, float(price_ratio))
        + slip_rate
        + slip_rate * max(0.0, float(price_ratio))
        + carry_rate_per_min * max(1.0, float(h_min))
    )


def _get_return_components_at_h(
    series: SymbolSeries, entry_ts_ms: int, entry_px: float, side: str, h_min: int
) -> tuple[float, float] | None:
    if h_min < 1:
        h_min = 1
    idx0 = _bucket_index(series.times_ms, entry_ts_ms)
    if idx0 < 0:
        return None
    idx1 = idx0 + int(h_min)
    if idx1 >= len(series.close):
        return None
    return _signed_ret_and_ratio(side, entry_px, series.close[idx1])


def _get_return_at_h(
    series: SymbolSeries, entry_ts_ms: int, entry_px: float, side: str, h_min: int
) -> float | None:
    comp = _get_return_components_at_h(series, entry_ts_ms, entry_px, side, h_min)
    if comp is None:
        return None
    return float(comp[0])


def _get_net_return_at_h(
    series: SymbolSeries,
    entry_ts_ms: int,
    entry_px: float,
    side: str,
    h_min: int,
    *,
    entry_fee_rate: float,
    exit_fee_rate: float,
    side_slippage_bps: float,
    carry_bps_per_hour: float,
) -> float | None:
    comp = _get_return_components_at_h(series, entry_ts_ms, entry_px, side, h_min)
    if comp is None:
        return None
    gross_ret, ratio = comp
    cost_rate = _return_cost_rate(
        entry_fee_rate=entry_fee_rate,
        exit_fee_rate=exit_fee_rate,
        side_slippage_bps=side_slippage_bps,
        carry_bps_per_hour=carry_bps_per_hour,
        price_ratio=ratio,
        h_min=h_min,
    )
    return float(gross_ret - cost_rate)


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


def _best_hold_ret_net(
    series: SymbolSeries,
    entry_ts_ms: int,
    entry_px: float,
    side: str,
    horizons_min: list[int],
    *,
    entry_fee_rate: float,
    exit_fee_rate: float,
    side_slippage_bps: float,
    carry_bps_per_hour: float,
) -> tuple[float | None, int | None, dict[int, float]]:
    out_h: dict[int, float] = {}
    best_r = None
    best_h = None
    for h in horizons_min:
        r = _get_net_return_at_h(
            series,
            entry_ts_ms,
            entry_px,
            side,
            h,
            entry_fee_rate=entry_fee_rate,
            exit_fee_rate=exit_fee_rate,
            side_slippage_bps=side_slippage_bps,
            carry_bps_per_hour=carry_bps_per_hour,
        )
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


def _describe_distribution(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {
            "n": 0,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    return {
        "n": int(len(xs)),
        "mean": _safe_mean(xs),
        "p10": _quantile(xs, 0.10),
        "p50": _quantile(xs, 0.50),
        "p90": _quantile(xs, 0.90),
    }


def _hist_int(xs: list[int]) -> dict[str, int]:
    out: dict[int, int] = defaultdict(int)
    for v in xs:
        try:
            out[int(v)] += 1
        except Exception:
            continue
    return {str(k): int(out[k]) for k in sorted(out.keys())}


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
    ap.add_argument("--default-fee-rate", type=float, default=0.0001, help="Fallback per-side fee rate (fraction)")
    ap.add_argument(
        "--default-exit-fee-rate",
        type=float,
        default=0.0,
        help="Fallback exit-side fee rate (fraction); <=0 means use entry-side fee rate",
    )
    ap.add_argument(
        "--default-slippage-bps-side",
        type=float,
        default=0.0,
        help="Fallback per-side slippage cost in bps",
    )
    ap.add_argument(
        "--carry-bps-per-hour",
        type=float,
        default=0.0,
        help="Time-decay/funding proxy in bps per hour for net counterfactual",
    )
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
    entry_side_profitable_n_net = 0
    entry_opp_better_n_net = 0
    best_ret_vals_net: list[float] = []
    side_ret_15m_vals_net: list[float] = []
    dir_regret_vals_net: list[float] = []
    tstar_gross_vals: list[int] = []
    tstar_net_vals: list[int] = []
    tstar_net_minus_gross_vals: list[float] = []
    tstar_best_ret_net_vals: list[float] = []
    regime_buf: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "tstar_gross": [],
            "tstar_net": [],
            "best_ret_gross": [],
            "best_ret_net": [],
        }
    )
    tstar_rows: list[dict[str, Any]] = []

    for e in entries:
        s = series_map.get(e.symbol)
        if not s or not s.times_ms:
            continue
        fee_entry = _entry_fee_rate(e, default_fee_rate=float(max(0.0, args.default_fee_rate)))
        fee_exit_cfg = float(args.default_exit_fee_rate or 0.0)
        fee_exit = float(fee_entry if fee_exit_cfg <= 0 else max(0.0, fee_exit_cfg))
        best_side_ret, best_h_gross, hmap_side = _best_hold_ret(s, e.ts_ms, e.fill_price, e.side, horizons)
        best_side_ret_net, best_h_net, hmap_side_net = _best_hold_ret_net(
            s,
            e.ts_ms,
            e.fill_price,
            e.side,
            horizons,
            entry_fee_rate=fee_entry,
            exit_fee_rate=fee_exit,
            side_slippage_bps=float(max(0.0, args.default_slippage_bps_side)),
            carry_bps_per_hour=float(max(0.0, args.carry_bps_per_hour)),
        )
        if best_side_ret is None or best_h_gross is None or best_side_ret_net is None or best_h_net is None:
            continue
        opp_side = "SHORT" if e.side == "LONG" else "LONG"
        best_opp_ret, _, _ = _best_hold_ret(s, e.ts_ms, e.fill_price, opp_side, horizons)
        best_opp_ret_net, _, _ = _best_hold_ret_net(
            s,
            e.ts_ms,
            e.fill_price,
            opp_side,
            horizons,
            entry_fee_rate=fee_entry,
            exit_fee_rate=fee_exit,
            side_slippage_bps=float(max(0.0, args.default_slippage_bps_side)),
            carry_bps_per_hour=float(max(0.0, args.carry_bps_per_hour)),
        )
        if best_opp_ret is None or best_opp_ret_net is None:
            continue

        entry_eval_n += 1
        best_ret_vals.append(float(best_side_ret))
        if best_side_ret > 0:
            entry_side_profitable_n += 1
        if best_opp_ret > best_side_ret:
            entry_opp_better_n += 1
        dir_regret_vals.append(float(best_opp_ret - best_side_ret))
        best_ret_vals_net.append(float(best_side_ret_net))
        if best_side_ret_net > 0:
            entry_side_profitable_n_net += 1
        if best_opp_ret_net > best_side_ret_net:
            entry_opp_better_n_net += 1
        dir_regret_vals_net.append(float(best_opp_ret_net - best_side_ret_net))
        tstar_gross_vals.append(int(best_h_gross))
        tstar_net_vals.append(int(best_h_net))
        tstar_net_minus_gross_vals.append(float(best_h_net - best_h_gross))
        tstar_best_ret_net_vals.append(float(best_side_ret_net))
        reg_key = _canonical_regime(e.regime)
        regime_buf[reg_key]["tstar_gross"].append(float(best_h_gross))
        regime_buf[reg_key]["tstar_net"].append(float(best_h_net))
        regime_buf[reg_key]["best_ret_gross"].append(float(best_side_ret))
        regime_buf[reg_key]["best_ret_net"].append(float(best_side_ret_net))
        tstar_rows.append(
            {
                "entry_id": int(e.id),
                "symbol": e.symbol,
                "side": e.side,
                "regime": reg_key,
                "entry_ts_ms": int(e.ts_ms),
                "tstar_gross_min": int(best_h_gross),
                "tstar_net_min": int(best_h_net),
                "best_ret_gross": float(best_side_ret),
                "best_ret_net": float(best_side_ret_net),
                "entry_ev": _safe_float(e.entry_ev, None),
                "entry_confidence": _safe_float(e.entry_confidence, None),
                "pred_mu_alpha": _safe_float(e.pred_mu_alpha, None),
                "pred_mu_dir_conf": _safe_float(e.pred_mu_dir_conf, None),
                "alpha_vpin": _safe_float(e.alpha_vpin, None),
                "alpha_hurst": _safe_float(e.alpha_hurst, None),
                "entry_quality_score": _safe_float(e.entry_quality_score, None),
                "one_way_move_score": _safe_float(e.one_way_move_score, None),
                "leverage_signal_score": _safe_float(e.leverage_signal_score, None),
                "policy_score_threshold": _safe_float(e.policy_score_threshold, None),
                "policy_event_exit_min_score": _safe_float(e.policy_event_exit_min_score, None),
                "policy_unrealized_dd_floor": _safe_float(e.policy_unrealized_dd_floor, None),
                "entry_fee_rate": float(fee_entry),
                "exit_fee_rate": float(fee_exit),
            }
        )

        r15 = hmap_side.get(15)
        if r15 is not None:
            side_ret_15m_vals.append(float(r15))
        r15_net = hmap_side_net.get(15)
        if r15_net is not None:
            side_ret_15m_vals_net.append(float(r15_net))

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
    exit_actual_ret_vals_net: list[float] = []
    exit_best_ret_vals_net: list[float] = []
    exit_regret_vals_net: list[float] = []
    early_like_n = 0
    late_like_n = 0
    early_like_n_net = 0
    late_like_n_net = 0
    by_reason: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "avg_regret": 0.0})
    by_reason_buf: dict[str, list[float]] = defaultdict(list)
    by_reason_net: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "avg_regret": 0.0})
    by_reason_buf_net: dict[str, list[float]] = defaultdict(list)
    match_mode_counter: dict[str, int] = defaultdict(int)

    for ex, en, match_mode in matched:
        s = series_map.get(en.symbol)
        if not s or not s.times_ms:
            continue
        match_mode_counter[match_mode] += 1
        fee_entry = _entry_fee_rate(en, default_fee_rate=float(max(0.0, args.default_fee_rate)))
        fee_exit_cfg = float(args.default_exit_fee_rate or 0.0)
        fee_exit = float(fee_entry if fee_exit_cfg <= 0 else max(0.0, fee_exit_cfg))
        best_ret, best_h, _ = _best_hold_ret(s, en.ts_ms, en.fill_price, en.side, horizons)
        best_ret_net, best_h_net, _ = _best_hold_ret_net(
            s,
            en.ts_ms,
            en.fill_price,
            en.side,
            horizons,
            entry_fee_rate=fee_entry,
            exit_fee_rate=fee_exit,
            side_slippage_bps=float(max(0.0, args.default_slippage_bps_side)),
            carry_bps_per_hour=float(max(0.0, args.carry_bps_per_hour)),
        )
        if best_ret is None or best_h is None or best_ret_net is None or best_h_net is None:
            continue
        h_actual = int(max(1, round(float(ex.hold_sec) / 60.0)))
        actual_ret = _get_return_at_h(s, en.ts_ms, en.fill_price, en.side, h_actual)
        actual_ret_net = _get_net_return_at_h(
            s,
            en.ts_ms,
            en.fill_price,
            en.side,
            h_actual,
            entry_fee_rate=fee_entry,
            exit_fee_rate=fee_exit,
            side_slippage_bps=float(max(0.0, args.default_slippage_bps_side)),
            carry_bps_per_hour=float(max(0.0, args.carry_bps_per_hour)),
        )
        if actual_ret is None or actual_ret_net is None:
            continue
        exit_eval_n += 1
        regret = float(best_ret - actual_ret)
        regret_net = float(best_ret_net - actual_ret_net)
        exit_actual_ret_vals.append(float(actual_ret))
        exit_best_ret_vals.append(float(best_ret))
        exit_regret_vals.append(regret)
        exit_actual_ret_vals_net.append(float(actual_ret_net))
        exit_best_ret_vals_net.append(float(best_ret_net))
        exit_regret_vals_net.append(float(regret_net))
        if regret > 0.0010:
            exit_improvable_n += 1
        if h_actual < int(0.8 * best_h):
            early_like_n += 1
        elif h_actual > int(1.2 * best_h):
            late_like_n += 1
        if h_actual < int(0.8 * best_h_net):
            early_like_n_net += 1
        elif h_actual > int(1.2 * best_h_net):
            late_like_n_net += 1

        rs = ex.reason or ex.action
        by_reason_buf[rs].append(regret)
        by_reason_buf_net[rs].append(regret_net)

    for rs, vals in by_reason_buf.items():
        by_reason[rs]["n"] = float(len(vals))
        by_reason[rs]["avg_regret"] = _safe_mean(vals)
    for rs, vals in by_reason_buf_net.items():
        by_reason_net[rs]["n"] = float(len(vals))
        by_reason_net[rs]["avg_regret"] = _safe_mean(vals)

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
    reason_top_net = sorted(
        (
            {
                "reason": rs,
                "n": int(v["n"]),
                "avg_regret": float(v["avg_regret"]),
            }
            for rs, v in by_reason_net.items()
        ),
        key=lambda x: x["avg_regret"],
        reverse=True,
    )[:15]

    feature_names = [
        "entry_ev",
        "entry_confidence",
        "pred_mu_alpha",
        "pred_mu_dir_conf",
        "alpha_vpin",
        "alpha_hurst",
        "entry_quality_score",
        "one_way_move_score",
        "leverage_signal_score",
        "policy_score_threshold",
        "policy_event_exit_min_score",
        "policy_unrealized_dd_floor",
        "entry_fee_rate",
    ]
    feature_corr: list[dict[str, Any]] = []
    for fn in feature_names:
        x_t: list[float] = []
        y_tstar_net: list[float] = []
        y_best_net: list[float] = []
        y_tdelta: list[float] = []
        for row in tstar_rows:
            x = _safe_float(row.get(fn), None)
            if x is None:
                continue
            t_net = _safe_float(row.get("tstar_net_min"), None)
            t_gross = _safe_float(row.get("tstar_gross_min"), None)
            best_net = _safe_float(row.get("best_ret_net"), None)
            if t_net is None or t_gross is None or best_net is None:
                continue
            x_t.append(float(x))
            y_tstar_net.append(float(t_net))
            y_best_net.append(float(best_net))
            y_tdelta.append(float(t_net - t_gross))
        if len(x_t) < 20:
            continue
        feature_corr.append(
            {
                "feature": fn,
                "n": int(len(x_t)),
                "pearson_tstar_net_min": _pearson(x_t, y_tstar_net),
                "pearson_best_ret_net": _pearson(x_t, y_best_net),
                "pearson_tstar_net_minus_gross": _pearson(x_t, y_tdelta),
            }
        )
    feature_corr.sort(
        key=lambda x: abs(_safe_float(x.get("pearson_best_ret_net"), 0.0) or 0.0),
        reverse=True,
    )

    by_regime = []
    for reg, vals in sorted(regime_buf.items()):
        by_regime.append(
            {
                "regime": reg,
                "n": int(len(vals["tstar_net"])),
                "tstar_gross": _describe_distribution(vals["tstar_gross"]),
                "tstar_net": _describe_distribution(vals["tstar_net"]),
                "best_ret_gross": _describe_distribution(vals["best_ret_gross"]),
                "best_ret_net": _describe_distribution(vals["best_ret_net"]),
            }
        )

    tstar_top_shift = sorted(
        tstar_rows,
        key=lambda row: abs(float((_safe_float(row.get("tstar_net_min"), 0.0) or 0.0) - (_safe_float(row.get("tstar_gross_min"), 0.0) or 0.0))),
        reverse=True,
    )[:30]

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
            "default_fee_rate": float(max(0.0, args.default_fee_rate)),
            "default_exit_fee_rate": float(args.default_exit_fee_rate),
            "default_slippage_bps_side": float(max(0.0, args.default_slippage_bps_side)),
            "carry_bps_per_hour": float(max(0.0, args.carry_bps_per_hour)),
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
            "side_profitable_rate_best_h_net": float(entry_side_profitable_n_net / entry_eval_n) if entry_eval_n else None,
            "opp_side_better_rate_net": float(entry_opp_better_n_net / entry_eval_n) if entry_eval_n else None,
            "avg_best_side_ret_net": _safe_mean(best_ret_vals_net),
            "avg_side_ret_15m_net": _safe_mean(side_ret_15m_vals_net),
            "avg_direction_regret_net": _safe_mean(dir_regret_vals_net),
            "p50_direction_regret_net": _quantile(dir_regret_vals_net, 0.50),
            "p90_direction_regret_net": _quantile(dir_regret_vals_net, 0.90),
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
            "avg_actual_ret_at_real_hold_net": _safe_mean(exit_actual_ret_vals_net),
            "avg_best_ret_within_horizon_net": _safe_mean(exit_best_ret_vals_net),
            "avg_exit_regret_net": _safe_mean(exit_regret_vals_net),
            "p50_exit_regret_net": _quantile(exit_regret_vals_net, 0.50),
            "p90_exit_regret_net": _quantile(exit_regret_vals_net, 0.90),
            "early_like_rate_net": float(early_like_n_net / exit_eval_n) if exit_eval_n else None,
            "late_like_rate_net": float(late_like_n_net / exit_eval_n) if exit_eval_n else None,
            "top_regret_reasons_net": reason_top_net,
        },
        "tstar_counterfactual": {
            "entry_eval_n": int(entry_eval_n),
            "tstar_gross_min": {
                **_describe_distribution([float(v) for v in tstar_gross_vals]),
                "hist": _hist_int(tstar_gross_vals),
            },
            "tstar_net_min": {
                **_describe_distribution([float(v) for v in tstar_net_vals]),
                "hist": _hist_int(tstar_net_vals),
            },
            "tstar_net_minus_gross_min": _describe_distribution(tstar_net_minus_gross_vals),
            "best_ret_net": _describe_distribution(tstar_best_ret_net_vals),
            "by_regime": by_regime,
            "feature_correlations": feature_corr[:20],
            "top_shift_rows": tstar_top_shift,
        },
        "fetch_errors": fetch_errors,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import os
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
    entry_reason: str
    pred_mu_alpha: float | None
    pred_mu_dir_conf: float | None
    regime: str | None
    alpha_vpin: float | None
    alpha_hurst: float | None
    raw_data: dict[str, Any]
    entry_link_id: str | None


@dataclass
class ExitRow:
    id: int
    symbol: str
    side: str
    fill_price: float
    ts_ms: int
    hold_sec: float
    exit_reason: str
    roe: float | None
    action: str
    entry_link_id: str | None


@dataclass
class SymbolSeries:
    times_ms: list[int]
    close: list[float]


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _parse_excluded_symbols(raw: str | None) -> set[str]:
    txt = str(raw or "").strip()
    out: set[str] = set()
    if not txt:
        return out
    for tok in txt.replace(";", ",").split(","):
        sym = str(tok or "").strip().upper()
        if sym:
            out.add(sym)
    return out


def _symbol_exclusions_from_args(raw: str | None) -> set[str]:
    if str(raw or "").strip():
        return _parse_excluded_symbols(raw)
    return _parse_excluded_symbols(
        os.environ.get("AUTO_REVAL_EXCLUDE_SYMBOLS")
        or os.environ.get("RESEARCH_EXCLUDE_SYMBOLS")
        or ""
    )


def _safe_rate(n: int, d: int) -> float | None:
    return float(n / d) if d > 0 else None


def _sign(v: float | None) -> int:
    if v is None:
        return 0
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _to_bybit_symbol(ccxt_symbol: str) -> str:
    base = (ccxt_symbol.split("/")[0] if "/" in ccxt_symbol else ccxt_symbol).strip().upper()
    return f"{base}USDT"


def _fetch_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _fetch_klines_1m(bybit_symbol: str, start_ms: int, end_ms: int) -> SymbolSeries:
    rows_all: list[tuple[int, float]] = []
    cursor = int(start_ms)
    while cursor < int(end_ms):
        chunk_end = min(int(end_ms), cursor + 999 * ONE_MIN_MS)
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
        for row in rows:
            t = int(row[0])
            c = float(row[4])
            rows_all.append((t, c))
        cursor = chunk_end + ONE_MIN_MS
        time.sleep(0.01)
    if not rows_all:
        return SymbolSeries(times_ms=[], close=[])
    uniq = {}
    for t, c in rows_all:
        uniq[t] = c
    times = sorted(uniq.keys())
    close = [uniq[t] for t in times]
    return SymbolSeries(times_ms=times, close=close)


def _bucket_index(times_ms: list[int], ts_ms: int) -> int:
    return bisect.bisect_right(times_ms, int(ts_ms)) - 1


def _signed_ret(side: str, entry_px: float, next_px: float) -> float:
    raw = (float(next_px) / max(1e-12, float(entry_px))) - 1.0
    return raw if str(side).upper() == "LONG" else -raw


def _get_return_at_h(series: SymbolSeries, entry_ts_ms: int, entry_px: float, side: str, h_min: int) -> float | None:
    idx0 = _bucket_index(series.times_ms, int(entry_ts_ms))
    if idx0 < 0:
        return None
    idx1 = idx0 + max(1, int(h_min))
    if idx1 >= len(series.close):
        return None
    return _signed_ret(side, entry_px, series.close[idx1])


def _best_hold_ret(
    series: SymbolSeries, entry_ts_ms: int, entry_px: float, side: str, horizons_min: list[int]
) -> tuple[float | None, int | None]:
    best_r = None
    best_h = None
    for h in horizons_min:
        r = _get_return_at_h(series, entry_ts_ms, entry_px, side, h)
        if r is None:
            continue
        if best_r is None or r > best_r:
            best_r = float(r)
            best_h = int(h)
    return best_r, best_h


def _canonical_reason(reason: str, kind: str) -> str:
    txt = str(reason or "").strip().lower()
    if not txt:
        return "unknown"
    txt = txt.split("|")[0].strip()
    keywords = (
        "event_mc_exit",
        "unrealized_dd",
        "hold_vs_exit",
        "unified_flip",
        "unified_cash",
        "hybrid_exit",
        "ev_drop",
        "take_profit",
        "stop_loss",
        "timeout",
        "liquidation",
        "exchange_close_external_sync",
        "exchange_close_manual_cleanup",
        "exchange_close_risk_forced",
        "exchange_manual_close",
        "rebalance",
        "entry_net_expectancy",
        "dir_gate",
        "event_exit",
        "fee",
    )
    for k in keywords:
        if k in txt:
            return k
    if kind == "entry":
        return txt[:48]
    return txt[:64]


def _conf_band(v: float | None) -> str:
    x = _safe_float(v)
    if x is None:
        return "na"
    if x < 0.56:
        return "low"
    if x < 0.62:
        return "mid"
    return "high"


def _tox_band(vpin: float | None) -> str:
    x = _safe_float(vpin)
    if x is None:
        return "na"
    if x >= 0.80:
        return "high"
    if x >= 0.50:
        return "mid"
    return "low"


def _score_band(v: float | None, t1: float, t2: float) -> str:
    x = _safe_float(v)
    if x is None:
        return "na"
    if x < t1:
        return "low"
    if x < t2:
        return "mid"
    return "high"


def _derive_entry_basis(en: EntryRow) -> str:
    base = _canonical_reason(en.entry_reason, "entry")
    if base != "unknown":
        return base
    regime = str(en.regime or "na").lower()
    conf = _conf_band(en.pred_mu_dir_conf)
    tox = _tox_band(en.alpha_vpin)
    eq = _score_band(_safe_float((en.raw_data or {}).get("entry_quality_score")), 0.58, 0.72)
    psl = _score_band(_safe_float((en.raw_data or {}).get("event_p_sl")), 0.30, 0.60)
    mu_s = _sign(en.pred_mu_alpha)
    mu_tag = "mu_pos" if mu_s > 0 else "mu_neg" if mu_s < 0 else "mu_zero"
    return f"reg:{regime}|conf:{conf}|tox:{tox}|eq:{eq}|psl:{psl}|{mu_tag}"


def _load_rows(
    db_path: Path,
    since_id: int,
    recent_exits: int,
    excluded_symbols: set[str] | None = None,
) -> tuple[list[EntryRow], list[ExitRow]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    entry_rows = cur.execute(
        """
        SELECT id, symbol, side, fill_price, timestamp_ms, entry_reason, pred_mu_alpha, pred_mu_dir_conf,
               regime, alpha_vpin, alpha_hurst, raw_data, entry_link_id
        FROM trades
        WHERE action IN ('ENTER','SPREAD')
          AND fill_price IS NOT NULL
          AND fill_price > 0
        ORDER BY timestamp_ms ASC, id ASC
        """
    ).fetchall()

    exits_sql = (
        """
        SELECT id, symbol, side, fill_price, timestamp_ms, hold_duration_sec, entry_reason, roe, action, entry_link_id
        FROM trades
        WHERE action IN ('EXIT','REBAL_EXIT')
          AND fill_price IS NOT NULL
          AND fill_price > 0
          AND hold_duration_sec IS NOT NULL
          AND hold_duration_sec > 0
          AND id > ?
        ORDER BY timestamp_ms DESC, id DESC
        LIMIT ?
        """
    )
    exit_rows_desc = cur.execute(exits_sql, (int(since_id), int(recent_exits))).fetchall()
    conn.close()

    excluded = excluded_symbols or set()
    entries: list[EntryRow] = []
    for r in entry_rows:
        symbol = str(r["symbol"] or "")
        if excluded and symbol.strip().upper() in excluded:
            continue
        raw_obj: dict[str, Any] = {}
        raw_txt = r["raw_data"]
        if raw_txt:
            try:
                dec = json.loads(raw_txt)
                if isinstance(dec, dict):
                    raw_obj = dec
            except Exception:
                raw_obj = {}
        entries.append(
            EntryRow(
                id=int(r["id"]),
                symbol=symbol,
                side=str(r["side"]).upper(),
                fill_price=float(r["fill_price"]),
                ts_ms=int(r["timestamp_ms"]),
                entry_reason=str(r["entry_reason"] or ""),
                pred_mu_alpha=_safe_float(r["pred_mu_alpha"]),
                pred_mu_dir_conf=_safe_float(r["pred_mu_dir_conf"]),
                regime=str(r["regime"]).strip() if r["regime"] is not None and str(r["regime"]).strip() else None,
                alpha_vpin=_safe_float(r["alpha_vpin"]),
                alpha_hurst=_safe_float(r["alpha_hurst"]),
                raw_data=raw_obj,
                entry_link_id=str(r["entry_link_id"]).strip()
                if r["entry_link_id"] is not None and str(r["entry_link_id"]).strip()
                else None,
            )
        )

    exits: list[ExitRow] = []
    for r in list(exit_rows_desc)[::-1]:
        symbol = str(r["symbol"] or "")
        if excluded and symbol.strip().upper() in excluded:
            continue
        exits.append(
            ExitRow(
                id=int(r["id"]),
                symbol=symbol,
                side=str(r["side"]).upper(),
                fill_price=float(r["fill_price"]),
                ts_ms=int(r["timestamp_ms"]),
                hold_sec=float(r["hold_duration_sec"]),
                exit_reason=str(r["entry_reason"] or ""),
                roe=_safe_float(r["roe"]),
                action=str(r["action"] or ""),
                entry_link_id=str(r["entry_link_id"]).strip() if r["entry_link_id"] is not None and str(r["entry_link_id"]).strip() else None,
            )
        )
    return entries, exits


def _match_exits(entries: list[EntryRow], exits: list[ExitRow], fallback_lookback_sec: int) -> list[tuple[ExitRow, EntryRow, str]]:
    by_link: dict[str, EntryRow] = {}
    by_sym_side: dict[tuple[str, str], list[EntryRow]] = defaultdict(list)
    for e in entries:
        if e.entry_link_id:
            by_link[e.entry_link_id] = e
        by_sym_side[(e.symbol, e.side)].append(e)
    for arr in by_sym_side.values():
        arr.sort(key=lambda x: (x.ts_ms, x.id))

    matched: list[tuple[ExitRow, EntryRow, str]] = []
    lookback_ms = int(fallback_lookback_sec) * 1000
    for ex in exits:
        if ex.entry_link_id and ex.entry_link_id in by_link:
            matched.append((ex, by_link[ex.entry_link_id], "entry_link_id"))
            continue
        arr = by_sym_side.get((ex.symbol, ex.side))
        if not arr:
            continue
        idx = bisect.bisect_right([x.ts_ms for x in arr], ex.ts_ms) - 1
        if idx < 0:
            continue
        en = arr[idx]
        if en.ts_ms <= ex.ts_ms and (ex.ts_ms - en.ts_ms) <= lookback_ms:
            matched.append((ex, en, "fallback_symbol_side"))
    return matched


def _sorted_top(table: dict[str, dict[str, Any]], key: str, n: int, reverse: bool = True) -> list[dict[str, Any]]:
    vals = list(table.values())
    vals.sort(key=lambda x: float(x.get(key) or 0.0), reverse=bool(reverse))
    return vals[: int(max(1, n))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Entry-vs-exit reason matrix + counterfactual mismatch analyzer.")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--since-id", type=int, default=0)
    ap.add_argument("--recent-exits", type=int, default=1500)
    ap.add_argument("--fallback-lookback-sec", type=int, default=6 * 3600)
    ap.add_argument("--max-symbols", type=int, default=60)
    ap.add_argument("--max-cell-top", type=int, default=20)
    ap.add_argument("--horizons-min", default="5,10,15,20,30,45,60")
    ap.add_argument("--out", default="state/entry_exit_reason_matrix_report.json")
    ap.add_argument(
        "--exclude-symbols",
        default="",
        help="Comma-separated symbols to exclude. Empty -> AUTO_REVAL_EXCLUDE_SYMBOLS/RESEARCH_EXCLUDE_SYMBOLS.",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    horizons = sorted({int(x) for x in str(args.horizons_min).split(",") if str(x).strip()})
    if not horizons:
        horizons = [5, 10, 15, 20, 30, 45, 60]

    excluded_symbols = _symbol_exclusions_from_args(args.exclude_symbols)
    entries, exits = _load_rows(
        db_path,
        int(args.since_id),
        int(args.recent_exits),
        excluded_symbols=excluded_symbols,
    )
    matched = _match_exits(entries, exits, int(args.fallback_lookback_sec))
    if not matched:
        out = {
            "timestamp_ms": int(time.time() * 1000),
            "db": str(db_path),
            "message": "no matched exits",
            "counts": {"entries": len(entries), "exits": len(exits), "matched": 0},
        }
        Path(args.out).write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=True, indent=2))
        return

    symbol_bounds: dict[str, list[int]] = defaultdict(lambda: [10**18, 0])
    for ex, en, _ in matched:
        b = symbol_bounds[en.symbol]
        b[0] = min(b[0], en.ts_ms - 5 * ONE_MIN_MS)
        b[1] = max(b[1], ex.ts_ms + (max(horizons) + 2) * ONE_MIN_MS)
    symbol_rank = sorted(symbol_bounds.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True)
    selected_symbols = {k for k, _ in symbol_rank[: int(max(1, args.max_symbols))]}

    series_map: dict[str, SymbolSeries] = {}
    fetch_errors: dict[str, str] = {}
    for sym in sorted(selected_symbols):
        s0, s1 = symbol_bounds[sym]
        try:
            series_map[sym] = _fetch_klines_1m(_to_bybit_symbol(sym), int(s0), int(s1))
        except Exception as exc:
            fetch_errors[sym] = str(exc)

    reason_stats: dict[str, dict[str, Any]] = {}
    pair_stats: dict[str, dict[str, Any]] = {}
    cause_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "reasons": defaultdict(int)})
    eval_n = 0

    for ex, en, match_mode in matched:
        if en.symbol not in series_map or not series_map[en.symbol].times_ms:
            continue
        s = series_map[en.symbol]
        best_same, best_h = _best_hold_ret(s, en.ts_ms, en.fill_price, en.side, horizons)
        opp_side = "SHORT" if en.side == "LONG" else "LONG"
        best_opp, _ = _best_hold_ret(s, en.ts_ms, en.fill_price, opp_side, horizons)
        h_actual = max(1, int(round(float(ex.hold_sec) / 60.0)))
        actual_ret = _get_return_at_h(s, en.ts_ms, en.fill_price, en.side, h_actual)
        if best_same is None or best_opp is None or actual_ret is None:
            continue

        eval_n += 1
        entry_key = _derive_entry_basis(en)
        exit_key = _canonical_reason(ex.exit_reason, "exit")
        pair_key = f"{entry_key} -> {exit_key}"

        pred_sign = _sign(en.pred_mu_alpha)
        real_sign = _sign(ex.roe if ex.roe is not None else actual_ret)
        dir_miss = int(pred_sign != 0 and real_sign != 0 and pred_sign != real_sign)

        opp_better = int(float(best_opp) > float(best_same))
        regret = float(best_same - actual_ret)
        early_like = int(best_h is not None and h_actual < int(0.8 * int(best_h)))
        late_like = int(best_h is not None and h_actual > int(1.2 * int(best_h)))
        profitable_side_loss = int(float(best_same) > 0 and ((ex.roe if ex.roe is not None else actual_ret) < 0))
        win = int((ex.roe if ex.roe is not None else actual_ret) > 0)

        if dir_miss:
            if opp_better:
                cause = "entry_direction_mismatch"
            elif profitable_side_loss:
                cause = "exit_timing_or_cost_flip"
            elif regret > 0.001:
                cause = "exit_timing_regret"
            else:
                cause = "mixed_or_noise"
            cause_stats[cause]["n"] += 1
            cause_stats[cause]["reasons"][exit_key] += 1

        if exit_key not in reason_stats:
            reason_stats[exit_key] = {
                "exit_reason": exit_key,
                "n": 0,
                "win_n": 0,
                "sum_roe": 0.0,
                "roe_n": 0,
                "dir_eval_n": 0,
                "dir_miss_n": 0,
                "opp_better_n": 0,
                "profitable_side_loss_n": 0,
                "early_like_n": 0,
                "late_like_n": 0,
                "sum_regret": 0.0,
                "sum_best_same": 0.0,
                "sum_actual_ret": 0.0,
                "match_fallback_n": 0,
            }
        rs = reason_stats[exit_key]
        rs["n"] += 1
        rs["win_n"] += win
        if ex.roe is not None:
            rs["sum_roe"] += float(ex.roe)
            rs["roe_n"] += 1
        if pred_sign != 0 and real_sign != 0:
            rs["dir_eval_n"] += 1
            rs["dir_miss_n"] += dir_miss
        rs["opp_better_n"] += opp_better
        rs["profitable_side_loss_n"] += profitable_side_loss
        rs["early_like_n"] += early_like
        rs["late_like_n"] += late_like
        rs["sum_regret"] += regret
        rs["sum_best_same"] += float(best_same)
        rs["sum_actual_ret"] += float(actual_ret)
        if match_mode != "entry_link_id":
            rs["match_fallback_n"] += 1

        if pair_key not in pair_stats:
            pair_stats[pair_key] = {
                "pair": pair_key,
                "entry_reason": entry_key,
                "exit_reason": exit_key,
                "n": 0,
                "sum_roe": 0.0,
                "roe_n": 0,
                "win_n": 0,
                "dir_eval_n": 0,
                "dir_miss_n": 0,
                "opp_better_n": 0,
                "profitable_side_loss_n": 0,
                "sum_regret": 0.0,
            }
        ps = pair_stats[pair_key]
        ps["n"] += 1
        ps["win_n"] += win
        if ex.roe is not None:
            ps["sum_roe"] += float(ex.roe)
            ps["roe_n"] += 1
        if pred_sign != 0 and real_sign != 0:
            ps["dir_eval_n"] += 1
            ps["dir_miss_n"] += dir_miss
        ps["opp_better_n"] += opp_better
        ps["profitable_side_loss_n"] += profitable_side_loss
        ps["sum_regret"] += regret

    reason_rows: list[dict[str, Any]] = []
    for v in reason_stats.values():
        n = int(v["n"])
        roe_n = int(v["roe_n"])
        dir_eval_n = int(v["dir_eval_n"])
        reason_rows.append(
            {
                "exit_reason": v["exit_reason"],
                "n": n,
                "win_rate": _safe_rate(int(v["win_n"]), n),
                "avg_roe": (float(v["sum_roe"] / roe_n) if roe_n > 0 else None),
                "direction_miss_rate": _safe_rate(int(v["dir_miss_n"]), dir_eval_n),
                "opp_side_better_rate": _safe_rate(int(v["opp_better_n"]), n),
                "profitable_side_loss_rate": _safe_rate(int(v["profitable_side_loss_n"]), n),
                "early_like_rate": _safe_rate(int(v["early_like_n"]), n),
                "late_like_rate": _safe_rate(int(v["late_like_n"]), n),
                "avg_exit_regret": float(v["sum_regret"] / n) if n > 0 else None,
                "avg_best_same_side_ret": float(v["sum_best_same"] / n) if n > 0 else None,
                "avg_actual_ret_at_real_hold": float(v["sum_actual_ret"] / n) if n > 0 else None,
                "fallback_match_rate": _safe_rate(int(v["match_fallback_n"]), n),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    for v in pair_stats.values():
        n = int(v["n"])
        roe_n = int(v["roe_n"])
        dir_eval_n = int(v["dir_eval_n"])
        pair_rows.append(
            {
                "pair": v["pair"],
                "entry_reason": v["entry_reason"],
                "exit_reason": v["exit_reason"],
                "n": n,
                "win_rate": _safe_rate(int(v["win_n"]), n),
                "avg_roe": (float(v["sum_roe"] / roe_n) if roe_n > 0 else None),
                "direction_miss_rate": _safe_rate(int(v["dir_miss_n"]), dir_eval_n),
                "opp_side_better_rate": _safe_rate(int(v["opp_better_n"]), n),
                "profitable_side_loss_rate": _safe_rate(int(v["profitable_side_loss_n"]), n),
                "avg_exit_regret": float(v["sum_regret"] / n) if n > 0 else None,
            }
        )

    reason_rows.sort(key=lambda x: (int(x.get("n") or 0), float(x.get("avg_roe") or 0.0)))
    reason_rows = list(reversed(reason_rows))
    pair_rows_by_n = sorted(pair_rows, key=lambda x: int(x.get("n") or 0), reverse=True)[: int(args.max_cell_top)]
    pair_rows_worst = sorted(pair_rows, key=lambda x: float(x.get("avg_roe") or 0.0))[: int(args.max_cell_top)]

    dir_miss_total = sum(int(v["n"]) for v in cause_stats.values())
    causes: list[dict[str, Any]] = []
    for k, v in sorted(cause_stats.items(), key=lambda kv: kv[1]["n"], reverse=True):
        reasons_sorted = sorted(v["reasons"].items(), key=lambda kv: kv[1], reverse=True)[:8]
        causes.append(
            {
                "cause": k,
                "n": int(v["n"]),
                "share_over_direction_miss": _safe_rate(int(v["n"]), int(dir_miss_total)),
                "top_exit_reasons": [{"exit_reason": rk, "n": int(rv)} for rk, rv in reasons_sorted],
            }
        )

    out = {
        "timestamp_ms": int(time.time() * 1000),
        "db": str(db_path),
        "config": {
            "since_id": int(args.since_id),
            "recent_exits": int(args.recent_exits),
            "horizons_min": horizons,
            "fallback_lookback_sec": int(args.fallback_lookback_sec),
            "max_symbols": int(args.max_symbols),
            "exclude_symbols": sorted(excluded_symbols),
        },
        "coverage": {
            "entries_total": int(len(entries)),
            "exits_total_recent": int(len(exits)),
            "matched_pairs": int(len(matched)),
            "counterfactual_eval_n": int(eval_n),
            "symbols_requested": int(len(selected_symbols)),
            "symbols_fetched": int(len(series_map)),
            "symbols_fetch_error": int(len(fetch_errors)),
            "direction_miss_cause_eval_n": int(dir_miss_total),
        },
        "root_cause_direction_miss": causes,
        "by_exit_reason": reason_rows,
        "entry_exit_matrix_top_by_n": pair_rows_by_n,
        "entry_exit_matrix_worst_avg_roe": pair_rows_worst,
        "fetch_errors": fetch_errors,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

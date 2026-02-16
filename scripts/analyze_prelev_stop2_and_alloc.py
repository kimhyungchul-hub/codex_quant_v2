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
from pathlib import Path
from typing import Any


ONE_MIN_MS = 60_000
BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"


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


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = min(len(xs), len(ys))
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    for x, y in zip(xs, ys, strict=False):
        dx = float(x - mx)
        dy = float(y - my)
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    den = math.sqrt(max(1e-18, sxx) * max(1e-18, syy))
    if den <= 0:
        return None
    return float(sxy / den)


def _rank_average(vals: list[float]) -> list[float]:
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(idx):
        j = i + 1
        while j < len(idx) and vals[idx[j]] == vals[idx[i]]:
            j += 1
        rr = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[idx[k]] = rr
        i = j
    return out


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if min(len(xs), len(ys)) < 3:
        return None
    return _pearson(_rank_average(xs), _rank_average(ys))


def _to_bybit_symbol(ccxt_symbol: str) -> str:
    base = str(ccxt_symbol).split("/")[0].strip().upper()
    return f"{base}USDT"


def _fetch_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _fetch_klines_1m(bybit_symbol: str, start_ms: int, end_ms: int) -> tuple[list[int], list[float]]:
    rows: list[tuple[int, float]] = []
    cursor = int(start_ms)
    while cursor < int(end_ms):
        chunk_end = min(int(end_ms), int(cursor + 999 * ONE_MIN_MS))
        params = {
            "category": "linear",
            "symbol": bybit_symbol,
            "interval": "1",
            "start": str(int(cursor)),
            "end": str(int(chunk_end)),
            "limit": "1000",
        }
        url = f"{BYBIT_KLINE_URL}?{urllib.parse.urlencode(params)}"
        try:
            obj = _fetch_json(url)
        except Exception:
            obj = {}
        if str(obj.get("retCode", "-1")) != "0":
            cursor = int(chunk_end + ONE_MIN_MS)
            time.sleep(0.02)
            continue
        items = (obj.get("result") or {}).get("list") or []
        if items:
            for it in items:
                try:
                    t = int(it[0])
                    c = float(it[4])
                except Exception:
                    continue
                rows.append((t, c))
        cursor = int(chunk_end + ONE_MIN_MS)
        time.sleep(0.01)
    if not rows:
        return [], []
    uniq = {}
    for t, c in rows:
        uniq[int(t)] = float(c)
    times = sorted(uniq.keys())
    close = [float(uniq[t]) for t in times]
    return times, close


def _bucket_idx_le(times_ms: list[int], ts_ms: int) -> int:
    return bisect.bisect_right(times_ms, int(ts_ms)) - 1


def _load_pairs(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], int, int, int]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    entries = cur.execute(
        """
        SELECT id, entry_link_id, symbol, side, fill_price, qty, notional, fee, timestamp_ms, raw_data
               , entry_ev, entry_confidence, pred_mu_dir_conf, alpha_vpin, alpha_hurst
               , entry_quality_score, leverage_signal_score
        FROM trades
        WHERE action='ENTER'
          AND entry_link_id IS NOT NULL
          AND TRIM(entry_link_id) <> ''
          AND fill_price IS NOT NULL
          AND fill_price > 0
        ORDER BY timestamp_ms ASC, id ASC
        """
    ).fetchall()
    exits = cur.execute(
        """
        SELECT id, entry_link_id, symbol, side, fill_price, qty, fee, timestamp_ms, action, entry_reason
        FROM trades
        WHERE action IN ('EXIT','REBAL_EXIT')
          AND entry_link_id IS NOT NULL
          AND TRIM(entry_link_id) <> ''
          AND fill_price IS NOT NULL
          AND fill_price > 0
        ORDER BY timestamp_ms ASC, id ASC
        """
    ).fetchall()
    eq_rows = cur.execute(
        """
        SELECT timestamp_ms, total_equity
        FROM equity_history
        WHERE trading_mode='live'
          AND total_equity IS NOT NULL
          AND total_equity > 0
        ORDER BY timestamp_ms ASC
        """
    ).fetchall()
    eq_t = [int(r["timestamp_ms"]) for r in eq_rows]
    eq_v = [float(r["total_equity"]) for r in eq_rows]

    def equity_at(ts_ms: int) -> float | None:
        if not eq_t:
            return None
        i = bisect.bisect_right(eq_t, int(ts_ms)) - 1
        if i < 0:
            return float(eq_v[0])
        return float(eq_v[i])

    ex_by_link: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for ex in exits:
        ex_by_link[str(ex["entry_link_id"])].append(ex)

    def parse_lev(raw: Any) -> float | None:
        if raw is None:
            return None
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        for k in ("leverage_effective", "entry_leverage", "leverage"):
            v = _safe_float(obj.get(k), None)
            if v is not None and float(v) > 0:
                return float(v)
        return None

    def parse_raw(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        try:
            obj = json.loads(raw)
        except Exception:
            return {}
        return obj if isinstance(obj, dict) else {}

    def raw_float(obj: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
        for k in keys:
            v = _safe_float(obj.get(k), None)
            if v is not None:
                return float(v)
        return default

    unmatched_entries = 0
    partial_exit_pairs = 0
    pairs: list[dict[str, Any]] = []
    for en in entries:
        link = str(en["entry_link_id"])
        arr = ex_by_link.get(link) or []
        if not arr:
            unmatched_entries += 1
            continue
        arr2 = [
            ex
            for ex in arr
            if int(ex["timestamp_ms"]) >= int(en["timestamp_ms"])
            and str(ex["symbol"]) == str(en["symbol"])
            and str(ex["side"]).upper() == str(en["side"]).upper()
        ]
        if not arr2:
            unmatched_entries += 1
            continue

        q_entry = float(_safe_float(en["qty"], 0.0) or 0.0)
        q_sum = float(sum(max(0.0, float(_safe_float(ex["qty"], 0.0) or 0.0)) for ex in arr2))
        q_eff = min(q_entry if q_entry > 0 else q_sum, q_sum)
        if q_eff <= 0:
            continue
        if q_entry > 0 and q_sum < 0.98 * q_entry:
            partial_exit_pairs += 1

        rem = float(q_eff)
        px_num = 0.0
        fee_exit = 0.0
        last_ts = int(arr2[0]["timestamp_ms"])
        for ex in arr2:
            q = max(0.0, float(_safe_float(ex["qty"], 0.0) or 0.0))
            if q <= 0 or rem <= 0:
                continue
            used = min(rem, q)
            px_num += float(used * float(ex["fill_price"]))
            f = float(_safe_float(ex["fee"], 0.0) or 0.0)
            fee_exit += float(f * (used / q if q > 0 else 0.0))
            rem -= used
            last_ts = max(last_ts, int(ex["timestamp_ms"]))
            if rem <= 1e-12:
                break
        if px_num <= 0:
            continue

        raw_obj = parse_raw(en["raw_data"])
        entry_px = float(en["fill_price"])
        exit_vwap = float(px_num / q_eff)
        side = 1.0 if str(en["side"]).upper() == "LONG" else -1.0
        gross_ret = float(side * ((exit_vwap / max(1e-12, entry_px)) - 1.0))
        notional = float(_safe_float(en["notional"], None) or (q_entry * entry_px))
        fee_entry = float(_safe_float(en["fee"], 0.0) or 0.0)
        fee_rate = float((abs(fee_entry) + abs(fee_exit)) / max(1e-12, abs(notional)))
        net_ret = float(gross_ret - fee_rate)
        eq = equity_at(int(en["timestamp_ms"]))
        alloc_ratio = float(notional / eq) if (eq is not None and eq > 0) else None
        hold_min = float(max(0.0, (float(last_ts) - float(en["timestamp_ms"])) / 60_000.0))

        pairs.append(
            {
                "entry_id": int(en["id"]),
                "entry_link_id": link,
                "symbol": str(en["symbol"]),
                "side": str(en["side"]).upper(),
                "entry_ts_ms": int(en["timestamp_ms"]),
                "exit_ts_ms": int(last_ts),
                "entry_price": float(entry_px),
                "exit_vwap": float(exit_vwap),
                "entry_qty": float(q_entry),
                "exit_qty_used": float(q_eff),
                "entry_notional": float(notional),
                "gross_ret_prelev": float(gross_ret),
                "net_ret_prelev": float(net_ret),
                "fee_rate_roundtrip": float(fee_rate),
                "alloc_ratio": float(alloc_ratio) if alloc_ratio is not None else None,
                "hold_min": float(hold_min),
                "leverage_effective": parse_lev(en["raw_data"]),
                "entry_ev": _safe_float(en["entry_ev"], None),
                "entry_confidence": _safe_float(en["entry_confidence"], None),
                "pred_mu_dir_conf": _safe_float(en["pred_mu_dir_conf"], None),
                "entry_quality_score": _safe_float(en["entry_quality_score"], None),
                "leverage_signal_score": _safe_float(en["leverage_signal_score"], None),
                "alpha_vpin": _safe_float(en["alpha_vpin"], None),
                "alpha_hurst": _safe_float(en["alpha_hurst"], None),
                "capital_signal_score": raw_float(raw_obj, "capital_signal_score"),
                "capital_signal_pre_roe": raw_float(raw_obj, "capital_signal_pre_roe"),
                "capital_signal_lev_sig": raw_float(raw_obj, "capital_signal_lev_sig"),
                "capital_signal_entry_q": raw_float(raw_obj, "capital_signal_entry_q"),
                "capital_signal_dir_conf": raw_float(raw_obj, "capital_signal_dir_conf"),
                "capital_signal_net": raw_float(raw_obj, "capital_signal_net"),
                "pre_roe_proxy_score": raw_float(raw_obj, "pre_roe_proxy_score"),
                "leverage_signal_score_legacy": raw_float(raw_obj, "leverage_signal_score_legacy"),
            }
        )

    return pairs, int(len(entries)), int(len(exits)), int(unmatched_entries + partial_exit_pairs)


def _alloc_correlation(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in pairs if r.get("alloc_ratio") is not None]
    x = [float(r["alloc_ratio"]) for r in rows]
    y_gross = [float(r["gross_ret_prelev"]) for r in rows]
    y_net = [float(r["net_ret_prelev"]) for r in rows]
    y_lev_scaled = [
        float(r["gross_ret_prelev"]) * max(1.0, float(_safe_float(r.get("leverage_effective"), 1.0) or 1.0))
        for r in rows
    ]
    quantiles: list[dict[str, Any]] = []
    if rows:
        s = sorted(rows, key=lambda z: float(z["alloc_ratio"]))
        qn = 5
        for qi in range(qn):
            lo = qi * len(s) // qn
            hi = (qi + 1) * len(s) // qn
            chunk = s[lo:hi]
            if not chunk:
                continue
            n = len(chunk)
            gross = [float(z["gross_ret_prelev"]) for z in chunk]
            net = [float(z["net_ret_prelev"]) for z in chunk]
            quantiles.append(
                {
                    "bucket": int(qi + 1),
                    "n": int(n),
                    "alloc_ratio_min": float(chunk[0]["alloc_ratio"]),
                    "alloc_ratio_max": float(chunk[-1]["alloc_ratio"]),
                    "gross_ret_avg": float(sum(gross) / n),
                    "net_ret_avg": float(sum(net) / n),
                    "win_rate_gross": float(sum(1 for v in gross if v > 0) / n),
                    "win_rate_net": float(sum(1 for v in net if v > 0) / n),
                }
            )
    return {
        "sample_n": int(len(rows)),
        "pearson_alloc_vs_prelev_gross_ret": _pearson(x, y_gross),
        "spearman_alloc_vs_prelev_gross_ret": _spearman(x, y_gross),
        "pearson_alloc_vs_prelev_net_ret": _pearson(x, y_net),
        "spearman_alloc_vs_prelev_net_ret": _spearman(x, y_net),
        "pearson_alloc_vs_lev_scaled_ret": _pearson(x, y_lev_scaled),
        "spearman_alloc_vs_lev_scaled_ret": _spearman(x, y_lev_scaled),
        "alloc_quantile_stats": quantiles,
    }


def _allocation_driver_report(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in pairs if r.get("alloc_ratio") is not None]
    if not rows:
        return {"sample_n": 0, "drivers": [], "same_as_leverage_check": {}}
    feature_keys = [
        "entry_ev",
        "entry_confidence",
        "pred_mu_dir_conf",
        "entry_quality_score",
        "leverage_signal_score",
        "leverage_signal_score_legacy",
        "pre_roe_proxy_score",
        "alpha_vpin",
        "alpha_hurst",
        "capital_signal_score",
        "capital_signal_pre_roe",
        "capital_signal_lev_sig",
        "capital_signal_entry_q",
        "capital_signal_dir_conf",
        "capital_signal_net",
    ]
    drivers: list[dict[str, Any]] = []
    for key in feature_keys:
        xs: list[float] = []
        ys_alloc: list[float] = []
        ys_net: list[float] = []
        for r in rows:
            xv = _safe_float(r.get(key), None)
            av = _safe_float(r.get("alloc_ratio"), None)
            nv = _safe_float(r.get("net_ret_prelev"), None)
            if xv is None or av is None or nv is None:
                continue
            xs.append(float(xv))
            ys_alloc.append(float(av))
            ys_net.append(float(nv))
        if len(xs) < 30:
            continue
        pa = _pearson(xs, ys_alloc)
        sa = _spearman(xs, ys_alloc)
        pn = _pearson(xs, ys_net)
        sn = _spearman(xs, ys_net)
        if pa is None and sa is None and pn is None and sn is None:
            continue
        drivers.append(
            {
                "feature": key,
                "sample_n": int(len(xs)),
                "corr_alloc_pearson": pa,
                "corr_alloc_spearman": sa,
                "corr_netret_pearson": pn,
                "corr_netret_spearman": sn,
            }
        )
    drivers.sort(
        key=lambda d: (
            abs(float(d.get("corr_alloc_spearman") or 0.0)),
            abs(float(d.get("corr_alloc_pearson") or 0.0)),
        ),
        reverse=True,
    )

    lev_vs_cap: dict[str, Any] = {}
    diffs = []
    lev_vals = []
    cap_vals = []
    alloc_vals = []
    for r in rows:
        lev = _safe_float(r.get("leverage_signal_score"), None)
        cap = _safe_float(r.get("capital_signal_score"), None)
        alloc = _safe_float(r.get("alloc_ratio"), None)
        if lev is None or cap is None or alloc is None:
            continue
        lev_vals.append(float(lev))
        cap_vals.append(float(cap))
        alloc_vals.append(float(alloc))
        diffs.append(abs(float(cap) - float(lev)))
    if diffs:
        close_thr = 0.02
        lev_vs_cap = {
            "sample_n": int(len(diffs)),
            "mean_abs_diff_capital_vs_leverage_signal": float(sum(diffs) / len(diffs)),
            "p50_abs_diff_capital_vs_leverage_signal": float(sorted(diffs)[len(diffs) // 2]),
            "share_diff_le_0_02": float(sum(1 for d in diffs if d <= close_thr) / len(diffs)),
            "corr_capital_vs_leverage_signal": _pearson(cap_vals, lev_vals),
            "corr_alloc_vs_leverage_signal": _pearson(alloc_vals, lev_vals),
            "corr_alloc_vs_capital_signal": _pearson(alloc_vals, cap_vals),
        }
    else:
        lev_only = []
        alloc_only = []
        for r in rows:
            lev = _safe_float(r.get("leverage_signal_score"), None)
            alloc = _safe_float(r.get("alloc_ratio"), None)
            if lev is None or alloc is None:
                continue
            lev_only.append(float(lev))
            alloc_only.append(float(alloc))
        lev_vs_cap = {
            "sample_n": 0,
            "capital_signal_missing": True,
            "note": "capital_signal_score is not available in historical raw_data for this slice.",
            "corr_alloc_vs_leverage_signal": _pearson(alloc_only, lev_only) if len(lev_only) >= 3 else None,
            "corr_alloc_vs_leverage_signal_spearman": _spearman(alloc_only, lev_only) if len(lev_only) >= 3 else None,
        }
    return {
        "sample_n": int(len(rows)),
        "drivers": drivers[:20],
        "same_as_leverage_check": lev_vs_cap,
    }


def _stop_minus2_counterfactual(
    pairs: list[dict[str, Any]],
    *,
    stop_threshold: float = -0.02,
    max_hold_min: int = 90,
) -> dict[str, Any]:
    ranges: dict[str, list[int]] = defaultdict(lambda: [10**18, 0])
    for p in pairs:
        st = int(p["entry_ts_ms"])
        en = max(int(p["exit_ts_ms"]), int(st + max_hold_min * ONE_MIN_MS))
        rr = ranges[str(p["symbol"])]
        rr[0] = min(rr[0], st)
        rr[1] = max(rr[1], en)

    series: dict[str, tuple[list[int], list[float]]] = {}
    failed: list[str] = []
    items = sorted(ranges.items(), key=lambda kv: kv[0])
    for i, (sym, (start_ms, end_ms)) in enumerate(items, start=1):
        bybit_symbol = _to_bybit_symbol(sym)
        t, c = _fetch_klines_1m(bybit_symbol, start_ms - ONE_MIN_MS, end_ms + ONE_MIN_MS)
        if not t:
            failed.append(sym)
        series[sym] = (t, c)
        if i % 10 == 0:
            print(f"[stop2] fetched {i}/{len(items)} symbols")

    stop_rows: list[dict[str, Any]] = []
    for p in pairs:
        sym = str(p["symbol"])
        times, close = series.get(sym, ([], []))
        if not times:
            continue
        i0 = _bucket_idx_le(times, int(p["entry_ts_ms"]))
        i1 = _bucket_idx_le(times, int(p["exit_ts_ms"]))
        if i0 < 0 or i1 <= i0:
            continue
        entry_px = float(p["entry_price"])
        side = 1.0 if str(p["side"]).upper() == "LONG" else -1.0
        rets = [float(side * ((close[i] / max(1e-12, entry_px)) - 1.0)) for i in range(i0, i1 + 1)]
        hit_j = None
        for j, r in enumerate(rets):
            if float(r) <= float(stop_threshold):
                hit_j = j
                break
        if hit_j is None:
            continue
        r_hit = float(rets[hit_j])
        after = rets[hit_j + 1 :] if hit_j + 1 < len(rets) else []
        recovered_before_exit = any(float(r) >= 0.0 for r in after)
        best_after = max(after) if after else r_hit
        actual_ret = float(p["gross_ret_prelev"])
        benefit = float(actual_ret - r_hit)
        stop_rows.append(
            {
                "entry_id": int(p["entry_id"]),
                "symbol": sym,
                "side": str(p["side"]),
                "hit_ret": float(r_hit),
                "actual_ret": float(actual_ret),
                "best_after_hit_to_exit": float(best_after),
                "recovered_before_exit": bool(recovered_before_exit),
                "benefit_hold_vs_stop": float(benefit),
                "hold_min": float(p["hold_min"]),
                "alloc_ratio": p.get("alloc_ratio"),
                "leverage_effective": p.get("leverage_effective"),
            }
        )

    n = len(stop_rows)
    rec_n = sum(1 for r in stop_rows if bool(r.get("recovered_before_exit")))
    hold_better_n = sum(1 for r in stop_rows if float(r.get("benefit_hold_vs_stop") or 0.0) > 0.0)
    avg_benefit = (sum(float(r["benefit_hold_vs_stop"]) for r in stop_rows) / n) if n else None
    p50_benefit = (sorted(float(r["benefit_hold_vs_stop"]) for r in stop_rows)[n // 2]) if n else None
    avg_actual = (sum(float(r["actual_ret"]) for r in stop_rows) / n) if n else None

    by_symbol: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in stop_rows:
        ss = by_symbol[str(r["symbol"])]
        ss[0] += 1.0
        if bool(r["recovered_before_exit"]):
            ss[1] += 1.0
        ss[2] += float(r["benefit_hold_vs_stop"])
    top_symbols: list[dict[str, Any]] = []
    for sym, (cnt, rec, ben) in by_symbol.items():
        n_sym = max(1.0, cnt)
        top_symbols.append(
            {
                "symbol": sym,
                "n": int(cnt),
                "recover_rate": float(rec / n_sym),
                "avg_benefit_hold_vs_stop": float(ben / n_sym),
            }
        )
    top_symbols.sort(key=lambda z: (int(z["n"]), float(z["recover_rate"])), reverse=True)

    return {
        "stop_threshold_prelev_ret": float(stop_threshold),
        "horizon_note": "entry->actual_exit using 1m bybit kline path",
        "sample_n_hit_stop": int(n),
        "recover_to_non_negative_before_exit_n": int(rec_n),
        "recover_to_non_negative_before_exit_rate": float(rec_n / n) if n else None,
        "hold_better_than_stop_n": int(hold_better_n),
        "hold_better_than_stop_rate": float(hold_better_n / n) if n else None,
        "avg_benefit_hold_vs_stop": avg_benefit,
        "p50_benefit_hold_vs_stop": p50_benefit,
        "avg_actual_prelev_ret_for_stop_hits": avg_actual,
        "top_symbols": top_symbols[:15],
        "fetch_failed_symbols": failed[:30],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze pre-leverage -2% stop validity and capital allocation correlation.")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--out", default="state/prelev_stop2_cap_alloc_report.json")
    ap.add_argument("--stop-threshold", type=float, default=-0.02)
    ap.add_argument("--max-hold-min", type=int, default=90)
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    out_path = Path(args.out).resolve()
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        pairs, n_entries, n_exits, n_unmatched_or_partial = _load_pairs(conn)

    alloc_report = _alloc_correlation(pairs)
    stop_report = _stop_minus2_counterfactual(
        pairs,
        stop_threshold=float(args.stop_threshold),
        max_hold_min=int(max(1, args.max_hold_min)),
    )

    report = {
        "generated_at_ms": int(time.time() * 1000),
        "db_path": str(db_path),
        "summary": {
            "entries_with_link": int(n_entries),
            "exits_with_link": int(n_exits),
            "matched_pairs": int(len(pairs)),
            "unmatched_or_partial_count": int(n_unmatched_or_partial),
            "symbols_in_pairs": int(len({str(p['symbol']) for p in pairs})),
        },
        "allocation_correlation": alloc_report,
        "allocation_driver_correlation": _allocation_driver_report(pairs),
        "stop_minus2_counterfactual": stop_report,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_path}")
    print(
        json.dumps(
            {
                "matched_pairs": int(len(pairs)),
                "alloc_corr_gross": alloc_report.get("pearson_alloc_vs_prelev_gross_ret"),
                "stop_hit_n": stop_report.get("sample_n_hit_stop"),
                "stop_recover_rate": stop_report.get("recover_to_non_negative_before_exit_rate"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

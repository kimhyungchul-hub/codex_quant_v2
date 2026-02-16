#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ENTRY_ACTIONS = {"ENTER", "SPREAD"}
EXIT_ACTIONS = {"EXIT", "REBAL_EXIT", "EXTERNAL", "MANUAL", "KILL"}


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


def _safe_int(v: Any, default: int | None = None) -> int | None:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _sign(v: float | None, eps: float = 1e-12) -> int:
    if v is None:
        return 0
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


def _ts_local(ms: int | None) -> str | None:
    if ms is None:
        return None
    try:
        return dt.datetime.fromtimestamp(int(ms) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


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


def _rank_avg(vals: list[float]) -> list[float]:
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
    n = min(len(xs), len(ys))
    if n < 3:
        return None
    return _pearson(_rank_avg(xs[:n]), _rank_avg(ys[:n]))


def _bucket(v: float, edges: list[float], labels: list[str]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return labels[i]
    return labels[-1]


def _json_obj(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _extract_lev(obj: dict[str, Any]) -> float | None:
    for k in ("leverage_effective", "entry_leverage", "leverage"):
        v = _safe_float(obj.get(k), None)
        if v is not None and v > 0:
            return float(v)
    return None


def _top_k_sum(rows: list[dict[str, Any]], key: str, value_key: str, k: int = 15) -> list[dict[str, Any]]:
    agg = defaultdict(float)
    cnt = defaultdict(int)
    for r in rows:
        kx = str(r.get(key) or "")
        if not kx:
            continue
        v = float(_safe_float(r.get(value_key), 0.0) or 0.0)
        agg[kx] += v
        cnt[kx] += 1
    out = []
    for kx, sv in agg.items():
        out.append({"key": kx, "n": int(cnt[kx]), "sum": float(sv), "avg": float(sv / max(1, cnt[kx]))})
    out.sort(key=lambda z: z["sum"])
    return out[:k]


def _load_equity(conn: sqlite3.Connection) -> tuple[list[int], list[float], list[float | None], list[dict[str, Any]]]:
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT timestamp_ms, total_equity, wallet_balance
        FROM equity_history
        WHERE trading_mode='live'
        ORDER BY timestamp_ms ASC
        """
    ).fetchall()
    ts = [int(r[0]) for r in rows]
    eq = [float(r[1]) for r in rows]
    wallet = [_safe_float(r[2], None) for r in rows]
    jumps = []
    prev_eq = None
    prev_w = None
    for t, e, w in zip(ts, eq, wallet, strict=False):
        de = None if prev_eq is None else float(e - prev_eq)
        dw = None if (w is None or prev_w is None) else float(w - prev_w)
        jumps.append({"ts_ms": int(t), "eq": float(e), "wallet": w, "d_eq": de, "d_wallet": dw})
        prev_eq = float(e)
        prev_w = w
    return ts, eq, wallet, jumps


def _equity_at(ts_ms: int, eq_ts: list[int], eq_val: list[float]) -> float | None:
    if not eq_ts:
        return None
    idx = bisect.bisect_right(eq_ts, int(ts_ms)) - 1
    if idx < 0:
        return float(eq_val[0])
    return float(eq_val[idx])


def _find_deposit_ts(
    jumps: list[dict[str, Any]],
    target_amount: float,
    tolerance: float,
    min_delta: float,
) -> tuple[int | None, dict[str, Any] | None]:
    near = []
    broad = []
    for j in jumps:
        dw = _safe_float(j.get("d_wallet"), None)
        de = _safe_float(j.get("d_eq"), None)
        d = dw if dw is not None else de
        if d is None:
            continue
        if d >= float(min_delta):
            broad.append(j)
        if abs(float(d) - float(target_amount)) <= float(tolerance):
            near.append(j)
    cand = near[-1] if near else (broad[-1] if broad else None)
    if cand is None:
        return None, None
    return int(cand["ts_ms"]), cand


def _load_trades(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT
            id, symbol, side, action, fill_price, qty, notional,
            timestamp_ms, entry_link_id, entry_reason, realized_pnl, roe, hold_duration_sec,
            entry_confidence, entry_ev, pred_mu_alpha, pred_mu_dir_conf,
            alpha_vpin, alpha_hurst, entry_quality_score, one_way_move_score, leverage_signal_score,
            raw_data
        FROM trades
        WHERE trading_mode='live'
        ORDER BY timestamp_ms ASC, id ASC
        """
    ).fetchall()
    out = []
    for r in rows:
        out.append({k: r[k] for k in r.keys()})
    return out


def _enrich_exits(
    trades: list[dict[str, Any]],
    deposit_ts_ms: int,
    eq_ts: list[int],
    eq_val: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    entries = [t for t in trades if str(t.get("action") or "").upper() in ENTRY_ACTIONS]
    exits = [t for t in trades if str(t.get("action") or "").upper() in EXIT_ACTIONS]
    post_exits = [t for t in exits if int(t.get("timestamp_ms") or 0) >= int(deposit_ts_ms)]

    by_link: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sym_side: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        link = str(e.get("entry_link_id") or "").strip()
        if link:
            by_link[link].append(e)
        key = (str(e.get("symbol") or ""), str(e.get("side") or "").upper())
        by_sym_side[key].append(e)
    for arr in by_link.values():
        arr.sort(key=lambda z: (int(z.get("timestamp_ms") or 0), int(z.get("id") or 0)))
    for arr in by_sym_side.values():
        arr.sort(key=lambda z: (int(z.get("timestamp_ms") or 0), int(z.get("id") or 0)))

    used_fallback_entries: set[int] = set()
    enriched = []
    match_counts = defaultdict(int)
    for ex in post_exits:
        ex_ts = int(ex.get("timestamp_ms") or 0)
        sym = str(ex.get("symbol") or "")
        side = str(ex.get("side") or "").upper()
        link = str(ex.get("entry_link_id") or "").strip()
        ex_raw = _json_obj(ex.get("raw_data"))

        entry = None
        match_mode = "none"
        if link and link in by_link:
            cand = [e for e in by_link[link] if int(e.get("timestamp_ms") or 0) <= ex_ts]
            if cand:
                entry = cand[-1]
                match_mode = "entry_link_id"

        if entry is None:
            arr = by_sym_side.get((sym, side), [])
            cand2 = [
                e
                for e in arr
                if int(e.get("timestamp_ms") or 0) <= ex_ts
                and int(e.get("id") or 0) not in used_fallback_entries
            ]
            if cand2:
                entry = cand2[-1]
                used_fallback_entries.add(int(entry.get("id") or 0))
                match_mode = "fallback_symbol_side"

        en_raw = _json_obj(entry.get("raw_data")) if entry else {}
        en_ts = int(entry.get("timestamp_ms") or 0) if entry else ex_ts

        pnl = _safe_float(ex.get("realized_pnl"), None)
        roe = _safe_float(ex.get("roe"), None)
        if roe is None and pnl is not None:
            # Fallback rough ROE on exit notional.
            ntn = _safe_float(ex.get("notional"), None)
            if ntn and ntn > 0:
                roe = float(pnl / ntn)

        lev_entry = _extract_lev(en_raw) if entry else None
        lev_exit = _extract_lev(ex_raw)
        lev_eff = lev_entry
        if lev_eff is None:
            lev_eff = lev_exit

        entry_notional = _safe_float(entry.get("notional"), None) if entry else None
        if entry_notional is None:
            entry_notional = _safe_float(en_raw.get("entry_notional"), None)
        if entry_notional is None:
            entry_notional = _safe_float(ex.get("notional"), None)

        eq_at_entry = _equity_at(en_ts, eq_ts, eq_val)
        eq_at_exit = _equity_at(ex_ts, eq_ts, eq_val)
        cap_ratio_entry = None
        if entry_notional is not None and eq_at_entry and eq_at_entry > 0:
            cap_ratio_entry = float(entry_notional / eq_at_entry)

        mu_alpha = None
        for src in (
            (entry or {}).get("pred_mu_alpha") if entry else None,
            ex.get("pred_mu_alpha"),
            en_raw.get("pred_mu_alpha"),
            ex_raw.get("pred_mu_alpha"),
        ):
            mu_alpha = _safe_float(src, mu_alpha)
            if mu_alpha is not None:
                break
        dir_conf = None
        for src in (
            (entry or {}).get("pred_mu_dir_conf") if entry else None,
            ex.get("pred_mu_dir_conf"),
            en_raw.get("pred_mu_dir_conf"),
            ex_raw.get("pred_mu_dir_conf"),
            (entry or {}).get("entry_confidence") if entry else None,
            ex.get("entry_confidence"),
        ):
            dir_conf = _safe_float(src, dir_conf)
            if dir_conf is not None:
                break

        entry_q = None
        for src in (
            (entry or {}).get("entry_quality_score") if entry else None,
            ex.get("entry_quality_score"),
            en_raw.get("entry_quality_score"),
            ex_raw.get("entry_quality_score"),
        ):
            entry_q = _safe_float(src, entry_q)
            if entry_q is not None:
                break

        lev_sig = None
        for src in (
            (entry or {}).get("leverage_signal_score") if entry else None,
            ex.get("leverage_signal_score"),
            en_raw.get("leverage_signal_score"),
            ex_raw.get("leverage_signal_score"),
        ):
            lev_sig = _safe_float(src, lev_sig)
            if lev_sig is not None:
                break

        alpha_vpin = None
        for src in (
            (entry or {}).get("alpha_vpin") if entry else None,
            ex.get("alpha_vpin"),
            en_raw.get("alpha_vpin"),
            ex_raw.get("alpha_vpin"),
        ):
            alpha_vpin = _safe_float(src, alpha_vpin)
            if alpha_vpin is not None:
                break

        alpha_hurst = None
        for src in (
            (entry or {}).get("alpha_hurst") if entry else None,
            ex.get("alpha_hurst"),
            en_raw.get("alpha_hurst"),
            ex_raw.get("alpha_hurst"),
        ):
            alpha_hurst = _safe_float(src, alpha_hurst)
            if alpha_hurst is not None:
                break

        hit = None
        if _sign(mu_alpha) != 0 and _sign(roe if roe is not None else pnl) != 0:
            hit = bool(_sign(mu_alpha) == _sign(roe if roe is not None else pnl))

        reason = str(ex.get("entry_reason") or "").strip()
        reason_main = reason.split(",")[0].strip() if reason else ""
        row = {
            "exit_id": int(ex.get("id") or 0),
            "entry_id": int(entry.get("id") or 0) if entry else None,
            "entry_link_id": link or (str(entry.get("entry_link_id") or "").strip() if entry else ""),
            "match_mode": match_mode,
            "symbol": sym,
            "side": side,
            "exit_ts_ms": int(ex_ts),
            "entry_ts_ms": int(en_ts) if entry else None,
            "hold_sec": _safe_float(ex.get("hold_duration_sec"), None),
            "reason": reason,
            "reason_main": reason_main,
            "pnl": pnl,
            "roe": roe,
            "leverage": lev_eff,
            "entry_notional": entry_notional,
            "cap_ratio_entry": cap_ratio_entry,
            "eq_entry": eq_at_entry,
            "eq_exit": eq_at_exit,
            "pnl_on_entry_equity": (float(pnl / eq_at_entry) if pnl is not None and eq_at_entry and eq_at_entry > 0 else None),
            "pred_mu_alpha": mu_alpha,
            "pred_mu_dir_conf": dir_conf,
            "entry_quality_score": entry_q,
            "leverage_signal_score": lev_sig,
            "alpha_vpin": alpha_vpin,
            "alpha_hurst": alpha_hurst,
            "entry_ev": _safe_float((entry or {}).get("entry_ev") if entry else ex.get("entry_ev"), None),
            "entry_confidence": _safe_float((entry or {}).get("entry_confidence") if entry else ex.get("entry_confidence"), None),
            "direction_hit": hit,
        }
        match_counts[match_mode] += 1
        enriched.append(row)

    # previous same-size exits for baseline comparison
    n_post = len(enriched)
    prev_exits = [t for t in exits if int(t.get("timestamp_ms") or 0) < int(deposit_ts_ms)]
    prev_exits = sorted(prev_exits, key=lambda z: int(z.get("timestamp_ms") or 0), reverse=True)[:n_post]
    prev_exits = sorted(prev_exits, key=lambda z: int(z.get("timestamp_ms") or 0))
    prev_simple = []
    for ex in prev_exits:
        prev_simple.append(
            {
                "id": int(ex.get("id") or 0),
                "pnl": _safe_float(ex.get("realized_pnl"), None),
                "roe": _safe_float(ex.get("roe"), None),
                "pred_mu_alpha": _safe_float(ex.get("pred_mu_alpha"), None),
                "reason": str(ex.get("entry_reason") or "").strip(),
            }
        )
    return enriched, prev_simple, {"match_mode_counts": dict(match_counts), "post_exit_count": len(enriched)}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [float(r["pnl"]) for r in rows if _safe_float(r.get("pnl"), None) is not None]
    roes = [float(r["roe"]) for r in rows if _safe_float(r.get("roe"), None) is not None]
    hits = [1.0 if bool(r["direction_hit"]) else 0.0 for r in rows if r.get("direction_hit") is not None]
    holds = [float(r["hold_sec"]) for r in rows if _safe_float(r.get("hold_sec"), None) is not None]
    levs = [float(r["leverage"]) for r in rows if _safe_float(r.get("leverage"), None) is not None]
    caps = [float(r["cap_ratio_entry"]) for r in rows if _safe_float(r.get("cap_ratio_entry"), None) is not None]
    return {
        "n": int(len(rows)),
        "n_with_pnl": int(len(pnls)),
        "win_rate": float(sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else None,
        "sum_pnl": float(sum(pnls)) if pnls else None,
        "avg_pnl": float(mean(pnls)) if pnls else None,
        "p50_pnl": float(median(pnls)) if pnls else None,
        "avg_roe": float(mean(roes)) if roes else None,
        "direction_hit": float(mean(hits)) if hits else None,
        "avg_hold_sec": float(mean(holds)) if holds else None,
        "avg_leverage": float(mean(levs)) if levs else None,
        "p90_leverage": float(sorted(levs)[int(0.9 * (len(levs) - 1))]) if levs else None,
        "avg_cap_ratio_entry": float(mean(caps)) if caps else None,
        "p90_cap_ratio_entry": float(sorted(caps)[int(0.9 * (len(caps) - 1))]) if caps else None,
    }


def _bin_report(rows: list[dict[str, Any]], feature: str, edges: list[float], labels: list[str]) -> list[dict[str, Any]]:
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        v = _safe_float(r.get(feature), None)
        if v is None:
            continue
        b = _bucket(float(v), edges, labels)
        bucket_rows[b].append(r)
    out = []
    for b in labels:
        arr = bucket_rows.get(b, [])
        pnls = [float(x["pnl"]) for x in arr if _safe_float(x.get("pnl"), None) is not None]
        caps = [float(x["cap_ratio_entry"]) for x in arr if _safe_float(x.get("cap_ratio_entry"), None) is not None]
        out.append(
            {
                "bucket": b,
                "n": int(len(arr)),
                "win_rate": float(sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else None,
                "sum_pnl": float(sum(pnls)) if pnls else None,
                "avg_pnl": float(mean(pnls)) if pnls else None,
                "avg_cap_ratio_entry": float(mean(caps)) if caps else None,
            }
        )
    return out


def _drawdown_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arr = sorted(rows, key=lambda z: int(z.get("exit_ts_ms") or 0))
    cum = 0.0
    peak = 0.0
    peak_idx = -1
    max_dd = 0.0
    trough_idx = -1
    curve = []
    for i, r in enumerate(arr):
        pnl = float(_safe_float(r.get("pnl"), 0.0) or 0.0)
        cum += pnl
        if cum > peak:
            peak = cum
            peak_idx = i
        dd = cum - peak
        if dd < max_dd:
            max_dd = dd
            trough_idx = i
        curve.append({"i": i, "ts_ms": int(r.get("exit_ts_ms") or 0), "cum_pnl": float(cum), "drawdown": float(dd)})
    seg = []
    if trough_idx >= 0:
        start = max(0, peak_idx + 1)
        seg = arr[start : trough_idx + 1]
    contrib = []
    by_link = defaultdict(float)
    by_symbol = defaultdict(float)
    for r in seg:
        pnl = float(_safe_float(r.get("pnl"), 0.0) or 0.0)
        if pnl >= 0:
            continue
        lk = str(r.get("entry_link_id") or f"exit:{r.get('exit_id')}")
        by_link[lk] += pnl
        by_symbol[str(r.get("symbol") or "")] += pnl
    for lk, v in by_link.items():
        contrib.append({"entry_link_id": lk, "sum_pnl": float(v)})
    contrib.sort(key=lambda z: z["sum_pnl"])
    sym_contrib = [{"symbol": s, "sum_pnl": float(v)} for s, v in by_symbol.items()]
    sym_contrib.sort(key=lambda z: z["sum_pnl"])
    return {
        "max_drawdown_realized_pnl": float(max_dd),
        "peak_ts_local": _ts_local(curve[peak_idx]["ts_ms"]) if peak_idx >= 0 and peak_idx < len(curve) else None,
        "trough_ts_local": _ts_local(curve[trough_idx]["ts_ms"]) if trough_idx >= 0 and trough_idx < len(curve) else None,
        "segment_n": int(len(seg)),
        "top_loss_links_in_dd_segment": contrib[:15],
        "top_loss_symbols_in_dd_segment": sym_contrib[:15],
    }


def _corr_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feats = [
        "leverage",
        "cap_ratio_entry",
        "pred_mu_dir_conf",
        "entry_quality_score",
        "leverage_signal_score",
        "alpha_vpin",
        "alpha_hurst",
        "hold_sec",
        "entry_ev",
    ]
    out = []
    for f in feats:
        xs = []
        ys = []
        ys_loss_mag = []
        for r in rows:
            x = _safe_float(r.get(f), None)
            y = _safe_float(r.get("pnl"), None)
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
            ys_loss_mag.append(float(-y) if y < 0 else 0.0)
        if len(xs) < 8:
            continue
        out.append(
            {
                "feature": f,
                "n": int(len(xs)),
                "corr_pnl_pearson": _pearson(xs, ys),
                "corr_pnl_spearman": _spearman(xs, ys),
                "corr_lossmag_pearson": _pearson(xs, ys_loss_mag),
                "corr_lossmag_spearman": _spearman(xs, ys_loss_mag),
            }
        )
    out.sort(key=lambda z: abs(float(z.get("corr_lossmag_spearman") or 0.0)), reverse=True)
    return {"feature_correlation": out}


def _write_top_losses_csv(rows: list[dict[str, Any]], out_csv: Path, top_n: int = 80) -> None:
    losses = [r for r in rows if _safe_float(r.get("pnl"), 0.0) is not None and float(r.get("pnl") or 0.0) < 0.0]
    losses.sort(key=lambda z: float(z.get("pnl") or 0.0))
    cols = [
        "exit_id",
        "entry_id",
        "entry_link_id",
        "symbol",
        "side",
        "exit_ts_ms",
        "entry_ts_ms",
        "hold_sec",
        "reason",
        "pnl",
        "roe",
        "leverage",
        "entry_notional",
        "eq_entry",
        "cap_ratio_entry",
        "pnl_on_entry_equity",
        "pred_mu_alpha",
        "pred_mu_dir_conf",
        "entry_quality_score",
        "leverage_signal_score",
        "alpha_vpin",
        "alpha_hurst",
        "direction_hit",
        "match_mode",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in losses[:top_n]:
            w.writerow({c: r.get(c) for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze loss drivers after a deposit event.")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--deposit-ts-ms", type=int, default=0, help="Explicit deposit timestamp ms (0=auto)")
    ap.add_argument("--deposit-amount", type=float, default=500.0)
    ap.add_argument("--deposit-tolerance", type=float, default=220.0)
    ap.add_argument("--deposit-min-delta", type=float, default=300.0)
    ap.add_argument("--out", default="state/post_500_loss_driver_report.json")
    ap.add_argument("--top-csv", default="state/post_500_top_loss_positions.csv")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    out_path = Path(args.out).resolve()
    top_csv = Path(args.top_csv).resolve()
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        eq_ts, eq_val, _wallet, jumps = _load_equity(conn)
        dep_ts = int(args.deposit_ts_ms) if int(args.deposit_ts_ms or 0) > 0 else None
        dep_info = None
        if dep_ts is None:
            dep_ts, dep_info = _find_deposit_ts(
                jumps=jumps,
                target_amount=float(args.deposit_amount),
                tolerance=float(args.deposit_tolerance),
                min_delta=float(args.deposit_min_delta),
            )
        else:
            dep_info = next((j for j in jumps if int(j["ts_ms"]) == int(dep_ts)), None)
        if dep_ts is None:
            raise SystemExit("deposit timestamp not found; provide --deposit-ts-ms")

        trades = _load_trades(conn)

    post_rows, prev_rows_simple, cover = _enrich_exits(trades, dep_ts, eq_ts, eq_val)
    if not post_rows:
        raise SystemExit("no post-deposit exits found")

    post_summary = _summary(post_rows)
    prev_proxy_rows = [
        {
            "pnl": r.get("pnl"),
            "roe": r.get("roe"),
            "direction_hit": (_sign(_safe_float(r.get("pred_mu_alpha"), None)) == _sign(_safe_float(r.get("roe"), _safe_float(r.get("pnl"), None))))
            if _sign(_safe_float(r.get("pred_mu_alpha"), None)) != 0 and _sign(_safe_float(r.get("roe"), _safe_float(r.get("pnl"), None))) != 0
            else None,
            "hold_sec": None,
            "leverage": None,
            "cap_ratio_entry": None,
        }
        for r in prev_rows_simple
    ]
    prev_summary = _summary(prev_proxy_rows)

    losses = [r for r in post_rows if _safe_float(r.get("pnl"), None) is not None and float(r.get("pnl") or 0.0) < 0.0]
    total_loss_abs = float(sum(-float(r["pnl"]) for r in losses)) if losses else 0.0

    by_symbol = defaultdict(lambda: {"n": 0, "sum_pnl": 0.0, "loss_n": 0, "loss_sum_abs": 0.0})
    by_reason = defaultdict(lambda: {"n": 0, "sum_pnl": 0.0, "loss_n": 0, "loss_sum_abs": 0.0})
    for r in post_rows:
        sym = str(r.get("symbol") or "")
        rs = str(r.get("reason_main") or "")
        pnl = float(_safe_float(r.get("pnl"), 0.0) or 0.0)
        for agg, key in ((by_symbol, sym), (by_reason, rs)):
            agg[key]["n"] += 1
            agg[key]["sum_pnl"] += pnl
            if pnl < 0:
                agg[key]["loss_n"] += 1
                agg[key]["loss_sum_abs"] += -pnl

    def _pack_agg(agg: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for k, v in agg.items():
            n = int(v["n"])
            out.append(
                {
                    "key": k,
                    "n": n,
                    "sum_pnl": float(v["sum_pnl"]),
                    "avg_pnl": float(v["sum_pnl"] / max(1, n)),
                    "loss_n": int(v["loss_n"]),
                    "loss_share_abs": float(v["loss_sum_abs"] / max(1e-12, total_loss_abs)) if total_loss_abs > 0 else None,
                }
            )
        out.sort(key=lambda z: z["sum_pnl"])
        return out

    lev_edges = [0.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 1e9]
    lev_labels = ["<=3x", "3-5x", "5-8x", "8-12x", "12-20x", "20-50x", "50x+"]
    cap_edges = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 2.0, 1e9]
    cap_labels = ["<=5%", "5-10%", "10-20%", "20-40%", "40-80%", "80-200%", "200%+"]

    # 2D loss heatmap by leverage/capital ratio.
    heat = defaultdict(lambda: {"n": 0, "sum_pnl": 0.0, "loss_n": 0})
    for r in post_rows:
        lv = _safe_float(r.get("leverage"), None)
        cp = _safe_float(r.get("cap_ratio_entry"), None)
        pnl = float(_safe_float(r.get("pnl"), 0.0) or 0.0)
        if lv is None or cp is None:
            continue
        lb = _bucket(float(lv), lev_edges, lev_labels)
        cb = _bucket(float(cp), cap_edges, cap_labels)
        key = (lb, cb)
        heat[key]["n"] += 1
        heat[key]["sum_pnl"] += pnl
        if pnl < 0:
            heat[key]["loss_n"] += 1
    heat_rows = []
    for (lb, cb), v in heat.items():
        n = int(v["n"])
        heat_rows.append(
            {
                "leverage_bucket": lb,
                "cap_bucket": cb,
                "n": n,
                "sum_pnl": float(v["sum_pnl"]),
                "avg_pnl": float(v["sum_pnl"] / max(1, n)),
                "loss_rate": float(v["loss_n"] / max(1, n)),
            }
        )
    heat_rows.sort(key=lambda z: z["sum_pnl"])

    report = {
        "generated_at_ms": int(dt.datetime.now().timestamp() * 1000),
        "db_path": str(db_path),
        "detected_deposit": {
            "deposit_ts_ms": int(dep_ts),
            "deposit_ts_local": _ts_local(dep_ts),
            "deposit_jump": dep_info,
            "detection_config": {
                "target_amount": float(args.deposit_amount),
                "tolerance": float(args.deposit_tolerance),
                "min_delta": float(args.deposit_min_delta),
            },
        },
        "coverage": cover,
        "post_period_summary": post_summary,
        "prev_same_exit_count_summary": prev_summary,
        "delta_vs_prev_same_exit_count": {
            "sum_pnl_delta": (post_summary.get("sum_pnl") or 0.0) - (prev_summary.get("sum_pnl") or 0.0),
            "win_rate_delta": (post_summary.get("win_rate") or 0.0) - (prev_summary.get("win_rate") or 0.0),
            "avg_pnl_delta": (post_summary.get("avg_pnl") or 0.0) - (prev_summary.get("avg_pnl") or 0.0),
            "direction_hit_delta": (post_summary.get("direction_hit") or 0.0) - (prev_summary.get("direction_hit") or 0.0),
        },
        "loss_concentration": {
            "loss_rows_n": int(len(losses)),
            "total_loss_abs": float(total_loss_abs),
            "top_loss_exits": sorted(losses, key=lambda z: float(z.get("pnl") or 0.0))[:25],
            "top_loss_symbols": _pack_agg(by_symbol)[:20],
            "top_loss_reasons": _pack_agg(by_reason)[:20],
            "top_symbol_reason_pairs": _top_k_sum(
                [
                    {
                        "pair": f"{r.get('symbol')}::{r.get('reason_main')}",
                        "pnl": r.get("pnl"),
                    }
                    for r in post_rows
                ],
                key="pair",
                value_key="pnl",
                k=20,
            ),
        },
        "leverage_bins": _bin_report(post_rows, "leverage", lev_edges, lev_labels),
        "capital_ratio_bins": _bin_report(post_rows, "cap_ratio_entry", cap_edges, cap_labels),
        "leverage_capital_heatmap": heat_rows[:40],
        "drawdown_linked_losses": _drawdown_report(post_rows),
        "correlation": _corr_report(post_rows),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_top_losses_csv(post_rows, top_csv, top_n=120)

    print(json.dumps(
        {
            "deposit_ts_local": report["detected_deposit"]["deposit_ts_local"],
            "post_exit_n": report["post_period_summary"]["n"],
            "post_sum_pnl": report["post_period_summary"]["sum_pnl"],
            "post_win_rate": report["post_period_summary"]["win_rate"],
            "prev_sum_pnl": report["prev_same_exit_count_summary"]["sum_pnl"],
            "delta_sum_pnl": report["delta_vs_prev_same_exit_count"]["sum_pnl_delta"],
            "report": str(out_path),
            "top_csv": str(top_csv),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()

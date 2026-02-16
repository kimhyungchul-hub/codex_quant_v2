#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import math
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ENTRY_ACTIONS = {"ENTER", "SPREAD"}
EXIT_ACTIONS = {"EXIT", "REBAL_EXIT", "KILL", "MANUAL", "EXTERNAL"}


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


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


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


def _json_obj(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    txt = str(raw or "").strip()
    if not txt:
        return out
    for tok in txt.split(","):
        fv = _safe_float(tok.strip(), None)
        if fv is not None:
            out.append(float(fv))
    return out


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = str(raw).strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = str(k).strip()
            if not key:
                continue
            val = str(v).strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            os.environ.setdefault(key, val)
    except Exception:
        return


def _extract_lev(obj: dict[str, Any]) -> float | None:
    for k in ("leverage_effective", "entry_leverage", "leverage"):
        v = _safe_float(obj.get(k), None)
        if v is not None and v > 0:
            return float(v)
    return None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = min(len(xs), len(ys))
    if n < 3:
        return None
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    for x, y in zip(xs[:n], ys[:n], strict=False):
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


def _load_equity(conn: sqlite3.Connection) -> tuple[list[int], list[float]]:
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT timestamp_ms, total_equity
        FROM equity_history
        WHERE trading_mode='live'
        ORDER BY timestamp_ms ASC
        """
    ).fetchall()
    ts = [int(r[0]) for r in rows]
    eq = [float(r[1]) for r in rows]
    return ts, eq


def _equity_at(ts_ms: int, eq_ts: list[int], eq_val: list[float]) -> float | None:
    if not eq_ts:
        return None
    idx = bisect.bisect_right(eq_ts, int(ts_ms)) - 1
    if idx < 0:
        return float(eq_val[0])
    return float(eq_val[idx])


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


def _build_enriched_exits(
    trades: list[dict[str, Any]],
    eq_ts: list[int],
    eq_val: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = [t for t in trades if str(t.get("action") or "").upper() in ENTRY_ACTIONS]
    exits = [t for t in trades if str(t.get("action") or "").upper() in EXIT_ACTIONS]

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
    enriched: list[dict[str, Any]] = []
    match_counts = defaultdict(int)
    for ex in exits:
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
            ntn = _safe_float(ex.get("notional"), None)
            if ntn and ntn > 0:
                roe = float(pnl / ntn)

        lev_entry = _extract_lev(en_raw) if entry else None
        lev_exit = _extract_lev(ex_raw)
        lev_eff = lev_entry if lev_entry is not None else lev_exit

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

        direction_hit = None
        if _sign(mu_alpha) != 0 and _sign(roe if roe is not None else pnl) != 0:
            direction_hit = bool(_sign(mu_alpha) == _sign(roe if roe is not None else pnl))

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
            "direction_hit": direction_hit,
        }
        match_counts[match_mode] += 1
        enriched.append(row)
    return enriched, {"match_mode_counts": dict(match_counts), "all_exit_count": len(enriched)}


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
        "p50_hold_sec": float(median(holds)) if holds else None,
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


def _pack_loss_agg(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    losses = [r for r in rows if _safe_float(r.get("pnl"), None) is not None and float(r.get("pnl") or 0.0) < 0.0]
    total_loss_abs = float(sum(-float(r["pnl"]) for r in losses)) if losses else 0.0
    agg = defaultdict(lambda: {"n": 0, "sum_pnl": 0.0, "loss_n": 0, "loss_sum_abs": 0.0})
    for r in rows:
        k = str(r.get(key_name) or "")
        if not k:
            continue
        pnl = float(_safe_float(r.get("pnl"), 0.0) or 0.0)
        st = agg[k]
        st["n"] += 1
        st["sum_pnl"] += pnl
        if pnl < 0:
            st["loss_n"] += 1
            st["loss_sum_abs"] += -pnl
    out = []
    for k, st in agg.items():
        n = int(st["n"])
        out.append(
            {
                "key": k,
                "n": n,
                "sum_pnl": float(st["sum_pnl"]),
                "avg_pnl": float(st["sum_pnl"] / max(1, n)),
                "loss_n": int(st["loss_n"]),
                "loss_share_abs": float(st["loss_sum_abs"] / max(1e-12, total_loss_abs)) if total_loss_abs > 0 else None,
            }
        )
    out.sort(key=lambda z: z["sum_pnl"])
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
    by_link = defaultdict(float)
    by_symbol = defaultdict(float)
    for r in seg:
        pnl = float(_safe_float(r.get("pnl"), 0.0) or 0.0)
        if pnl >= 0:
            continue
        lk = str(r.get("entry_link_id") or f"exit:{r.get('exit_id')}")
        by_link[lk] += pnl
        by_symbol[str(r.get("symbol") or "")] += pnl
    contrib = [{"entry_link_id": lk, "sum_pnl": float(v)} for lk, v in by_link.items()]
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


def _write_top_losses_csv(rows: list[dict[str, Any]], out_csv: Path, top_n: int = 120) -> None:
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


def _pick_reason_row(reason_matrix: dict[str, Any], token: str) -> dict[str, Any]:
    rows = reason_matrix.get("by_exit_reason")
    if not isinstance(rows, list):
        return {}
    tok = str(token).strip().lower()
    best = {}
    best_n = -1
    for r in rows:
        if not isinstance(r, dict):
            continue
        reason = str(r.get("exit_reason") or "").strip().lower()
        if tok not in reason:
            continue
        n = int(_safe_int(r.get("n"), 0))
        if n > best_n:
            best = r
            best_n = n
    return best if isinstance(best, dict) else {}


def _reason_window_stats(rows: list[dict[str, Any]], token: str) -> dict[str, Any]:
    tok = str(token).strip().lower()
    arr = [r for r in rows if tok in str(r.get("reason") or "").lower()]
    pnls = [float(r["pnl"]) for r in arr if _safe_float(r.get("pnl"), None) is not None]
    roes = [float(r["roe"]) for r in arr if _safe_float(r.get("roe"), None) is not None]
    holds = [float(r["hold_sec"]) for r in arr if _safe_float(r.get("hold_sec"), None) is not None]
    return {
        "n": int(len(arr)),
        "avg_pnl": float(mean(pnls)) if pnls else None,
        "avg_roe": float(mean(roes)) if roes else None,
        "loss_rate": float(sum(1 for p in pnls if p < 0) / len(pnls)) if pnls else None,
        "avg_hold_sec": float(mean(holds)) if holds else None,
        "p50_hold_sec": float(median(holds)) if holds else None,
        "p90_hold_sec": float(sorted(holds)[int(0.9 * (len(holds) - 1))]) if holds else None,
    }


def _timing_inc_ratio(timing_gap: float | None) -> float:
    g = float(timing_gap or 0.0)
    if g <= 0.0015:
        return 0.0
    if g < 0.0050:
        return 0.15
    if g < 0.0100:
        return 0.25
    return 0.35


def _timing_validation(
    *,
    cf: dict[str, Any],
    reason_matrix: dict[str, Any],
    post_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exit_cf = (cf or {}).get("exit_counterfactual") if isinstance((cf or {}).get("exit_counterfactual"), dict) else {}
    avg_exit_regret = _safe_float(exit_cf.get("avg_exit_regret"), None)
    early_like_rate = _safe_float(exit_cf.get("early_like_rate"), None)
    improvable_rate = _safe_float(exit_cf.get("improvable_rate_regret_gt_10bps"), None)

    top_regret_reasons = exit_cf.get("top_regret_reasons") if isinstance(exit_cf.get("top_regret_reasons"), list) else []
    # Map for tuning: reason -> avg_regret (top-K only).
    exit_regret_by_reason: dict[str, float] = {}
    for row in top_regret_reasons:
        if not isinstance(row, dict):
            continue
        rs = str(row.get("reason") or "").strip().lower()
        if not rs:
            continue
        ar = _safe_float(row.get("avg_regret"), None)
        if ar is None:
            continue
        exit_regret_by_reason[rs] = float(ar)

    axis_tokens = {
        "unified_flip": "unified_flip",
        "hold_vs_exit": "hold_vs_exit",
        "exchange_close_external_sync": "exchange_close_external_sync",
    }
    by_reason = {}
    any_issue = False
    for axis, tok in axis_tokens.items():
        rm = _pick_reason_row(reason_matrix, tok)
        win = _reason_window_stats(post_rows, tok)
        rm_n = int(_safe_int(rm.get("n"), 0))
        rm_early = _safe_float(rm.get("early_like_rate"), None)
        rm_best = _safe_float(rm.get("avg_best_same_side_ret"), None)
        rm_actual = _safe_float(rm.get("avg_actual_ret_at_real_hold"), None)
        timing_gap = None
        if rm_best is not None and rm_actual is not None:
            timing_gap = float(rm_best - rm_actual)
        inc = _timing_inc_ratio(timing_gap)
        timing_issue = bool(
            rm_n >= 8
            and timing_gap is not None
            and float(timing_gap) > 0.0015
            and (rm_early is None or float(rm_early) >= 0.55)
        )
        if timing_issue:
            any_issue = True
        rec_params: dict[str, Any] = {}
        if timing_issue:
            if axis == "unified_flip":
                rec_params = {
                    "UNIFIED_FLIP_CONFIRM_TICKS_NORMAL": +1,
                    "UNIFIED_FLIP_CONFIRM_TICKS_NOISE": +1,
                    "UNIFIED_FLIP_MIN_PROGRESS_RANDOM": +0.03,
                    "UNIFIED_FLIP_MIN_PROGRESS_MEAN_REVERT": +0.03,
                }
            elif axis == "hold_vs_exit":
                rec_params = {
                    "HOLD_EVAL_MIN_PROGRESS_TO_EXIT": +0.05,
                    "HOLD_EVAL_EXIT_CONFIRM_TICKS_NORMAL": +1,
                    "HOLD_EVAL_EXIT_CONFIRM_TICKS_NOISE": +1,
                    "HOLD_EVAL_EXIT_MARGIN": -0.0001,
                }
            elif axis == "exchange_close_external_sync":
                rec_params = {
                    "LIVE_LIQUIDATION_MISS_COUNT": +1,
                    "EXTERNAL_CLOSE_LIQ_BUFFER_PCT": -0.0003,
                }
        rec_hold_sec = None
        base_hold = _safe_float(win.get("avg_hold_sec"), None)
        if base_hold is not None and inc > 0:
            rec_hold_sec = float(base_hold * (1.0 + float(inc)))
        by_reason[axis] = {
            "token": tok,
            "timing_issue": bool(timing_issue),
            "timing_gap_ret": timing_gap,
            "rm_n": int(rm_n),
            "rm_avg_roe": _safe_float(rm.get("avg_roe"), None),
            "rm_early_like_rate": rm_early,
            "rm_avg_exit_regret": _safe_float(rm.get("avg_exit_regret"), None),
            "window_n": int(win.get("n", 0)),
            "window_avg_roe": _safe_float(win.get("avg_roe"), None),
            "window_loss_rate": _safe_float(win.get("loss_rate"), None),
            "window_avg_hold_sec": _safe_float(win.get("avg_hold_sec"), None),
            "recommended_hold_increase_ratio": float(inc),
            "recommended_hold_sec": rec_hold_sec,
            "recommended_param_deltas": rec_params,
        }
    global_issue = bool(
        any_issue
        or (
            avg_exit_regret is not None
            and early_like_rate is not None
            and float(avg_exit_regret) >= 0.008
            and float(early_like_rate) >= 0.55
        )
    )
    return {
        "avg_exit_regret": avg_exit_regret,
        "early_like_rate": early_like_rate,
        "improvable_rate_regret_gt_10bps": improvable_rate,
        "timing_issue_validated": bool(global_issue),
        "by_reason": by_reason,
        "top_regret_reasons": top_regret_reasons,
        "exit_regret_by_reason": exit_regret_by_reason,
    }


def _capital_scale_plan(
    *,
    post_summary: dict[str, Any],
    timing_validation: dict[str, Any],
    capitals: list[float],
) -> list[dict[str, Any]]:
    tiers = _parse_float_list(os.environ.get("CAPITAL_TIER_USDT", ""))
    if not tiers:
        tiers = [500.0, 1500.0, 3000.0, 6000.0, 9000.0]
    tiers = sorted(float(max(0.0, x)) for x in tiers)
    n_stage = int(len(tiers) + 1)
    total_cap = _parse_float_list(os.environ.get("CAPITAL_TIER_TOTAL_CAP", ""))
    if not total_cap:
        total_cap = [2.0, 2.8, 4.0, 5.5, 7.5, 9.5]
    lev_cap = _parse_float_list(os.environ.get("AUTO_TUNE_CAPITAL_STAGE_LEVERAGE_MAX", ""))
    if not lev_cap:
        lev_cap = [8.0, 12.0, 20.0, 28.0, 38.0, 50.0]
    min_notional = _parse_float_list(os.environ.get("CAPITAL_TIER_MIN_NOTIONAL", ""))
    if not min_notional:
        min_notional = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]

    def _fit(vals: list[float], expected: int, dflt: float) -> list[float]:
        out = [float(v) for v in vals]
        if not out:
            out = [float(dflt)]
        if len(out) < expected:
            out.extend([float(out[-1])] * (expected - len(out)))
        return out[:expected]

    total_cap = _fit(total_cap, n_stage, 2.0)
    lev_cap = _fit(lev_cap, n_stage, 10.0)
    min_notional = _fit(min_notional, n_stage, 0.1)

    timing_issue = bool(timing_validation.get("timing_issue_validated") is True)
    avg_roe = _safe_float(post_summary.get("avg_roe"), 0.0) or 0.0
    sum_pnl = _safe_float(post_summary.get("sum_pnl"), 0.0) or 0.0
    risk_scale = 1.0
    if timing_issue:
        risk_scale *= 0.85
    if avg_roe < 0:
        risk_scale *= 0.88
    if sum_pnl < 0:
        risk_scale *= 0.92
    risk_scale = float(max(0.65, min(1.0, risk_scale)))

    out = []
    for cap in capitals:
        c = float(cap)
        idx = 0
        for th in tiers:
            if c < float(th):
                break
            idx += 1
        idx = int(max(0, min(idx, n_stage - 1)))
        base_lev = float(lev_cap[idx])
        tuned_lev = float(max(2.0, min(50.0, base_lev * risk_scale)))
        base_total_cap = float(total_cap[idx])
        tuned_total_cap = float(max(1.5, base_total_cap * (0.92 if timing_issue else 1.0)))
        base_min_notional = float(min_notional[idx])
        tuned_min_notional = float(max(0.1, base_min_notional * (1.15 if avg_roe < 0 else 1.0)))
        out.append(
            {
                "capital_usdt": c,
                "tier_index": idx,
                "tier_thresholds": tiers,
                "base_leverage_cap": base_lev,
                "suggested_leverage_cap": tuned_lev,
                "base_total_cap": base_total_cap,
                "suggested_total_cap": tuned_total_cap,
                "base_min_notional": base_min_notional,
                "suggested_min_notional": tuned_min_notional,
                "risk_scale": risk_scale,
                "timing_issue": timing_issue,
            }
        )
    return out


def _append_history(history_path: Path | None, snapshot: dict[str, Any]) -> dict[str, Any]:
    if history_path is None:
        return {"enabled": False, "total_snapshots": 0}
    hist = {"snapshots": []}
    if history_path.exists():
        try:
            obj = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and isinstance(obj.get("snapshots"), list):
                hist = obj
        except Exception:
            hist = {"snapshots": []}
    snaps = list(hist.get("snapshots") or [])
    prev = snaps[-1] if snaps else None
    snaps.append(snapshot)
    if len(snaps) > 240:
        snaps = snaps[-240:]
    payload = {
        "updated_at_ms": int(dt.datetime.now().timestamp() * 1000),
        "snapshots": snaps,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    delta_prev = None
    if isinstance(prev, dict):
        curr_s = snapshot.get("post_period_summary") if isinstance(snapshot.get("post_period_summary"), dict) else {}
        prev_s = prev.get("post_period_summary") if isinstance(prev.get("post_period_summary"), dict) else {}
        curr_tv = snapshot.get("timing_validation") if isinstance(snapshot.get("timing_validation"), dict) else {}
        prev_tv = prev.get("timing_validation") if isinstance(prev.get("timing_validation"), dict) else {}
        delta_prev = {
            "sum_pnl_delta": (float(curr_s.get("sum_pnl") or 0.0) - float(prev_s.get("sum_pnl") or 0.0)),
            "win_rate_delta": (float(curr_s.get("win_rate") or 0.0) - float(prev_s.get("win_rate") or 0.0)),
            "avg_roe_delta": (float(curr_s.get("avg_roe") or 0.0) - float(prev_s.get("avg_roe") or 0.0)),
            "avg_exit_regret_delta": (
                float(curr_tv.get("avg_exit_regret") or 0.0)
                - float(prev_tv.get("avg_exit_regret") or 0.0)
            ),
        }

        # Reason-level regret delta (top reasons only).
        curr_map = curr_tv.get("exit_regret_by_reason") if isinstance(curr_tv.get("exit_regret_by_reason"), dict) else {}
        prev_map = prev_tv.get("exit_regret_by_reason") if isinstance(prev_tv.get("exit_regret_by_reason"), dict) else {}
        if curr_map or prev_map:
            deltas: dict[str, float] = {}
            keys = set(str(k) for k in list(curr_map.keys()) + list(prev_map.keys()))
            for k in keys:
                a = _safe_float(curr_map.get(k), None)
                b = _safe_float(prev_map.get(k), None)
                if a is None or b is None:
                    continue
                deltas[str(k)] = float(a) - float(b)
            # Keep only the largest movers for readability.
            if deltas:
                top = sorted(deltas.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:12]
                delta_prev["exit_regret_by_reason_delta"] = {k: float(v) for k, v in top}
    return {
        "enabled": True,
        "path": str(history_path),
        "total_snapshots": int(len(snaps)),
        "delta_vs_prev_snapshot": delta_prev,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze batch loss drivers for auto-reval window (since_id based).")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--since-id", type=int, required=True)
    ap.add_argument("--batch-id", type=int, default=0)
    ap.add_argument("--target-new", type=int, default=120)
    ap.add_argument("--out", default="state/reval_loss_driver_report.json")
    ap.add_argument("--top-csv", default="state/reval_loss_driver_top_loss_positions.csv")
    ap.add_argument("--history-out", default="state/reval_loss_driver_history.json")
    ap.add_argument("--cf-file", default="state/counterfactual_replay_report_new_window.json")
    ap.add_argument("--reason-matrix-file", default="state/entry_exit_reason_matrix_report.json")
    ap.add_argument("--env-file", default="state/bybit.env")
    ap.add_argument("--capital-steps", default="1500,3000,6000,9000")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    out_path = Path(args.out).resolve()
    top_csv = Path(args.top_csv).resolve()
    history_path = Path(args.history_out).resolve() if str(args.history_out).strip() else None
    cf_path = Path(args.cf_file).resolve() if str(args.cf_file).strip() else None
    rm_path = Path(args.reason_matrix_file).resolve() if str(args.reason_matrix_file).strip() else None
    env_path = Path(args.env_file).resolve() if str(args.env_file).strip() else None
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")
    if env_path is not None:
        _load_env_file(env_path)

    since_id = int(max(0, args.since_id))
    capitals = _parse_float_list(args.capital_steps)
    if not capitals:
        capitals = [1500.0, 3000.0, 6000.0, 9000.0]

    with sqlite3.connect(str(db_path)) as conn:
        eq_ts, eq_val = _load_equity(conn)
        trades = _load_trades(conn)

    all_rows, cover_all = _build_enriched_exits(trades, eq_ts, eq_val)
    post_rows = [r for r in all_rows if int(r.get("exit_id") or 0) > since_id]
    if not post_rows:
        raise SystemExit(f"no exits after since_id={since_id}")
    prev_pool = [r for r in all_rows if int(r.get("exit_id") or 0) <= since_id]
    prev_rows = prev_pool[-len(post_rows) :] if prev_pool else []
    cover = {
        "match_mode_counts": dict(
            defaultdict(int, {k: int(v) for k, v in defaultdict(int, {
                r.get("match_mode") or "none": 1 for r in post_rows
            }).items()})
        ),
        "post_exit_count": int(len(post_rows)),
        "all_exit_count": int(cover_all.get("all_exit_count", len(all_rows))),
    }
    # Rebuild match counts correctly.
    mm = defaultdict(int)
    for r in post_rows:
        mm[str(r.get("match_mode") or "none")] += 1
    cover["match_mode_counts"] = dict(mm)

    post_summary = _summary(post_rows)
    prev_summary = _summary(prev_rows)

    losses = [r for r in post_rows if _safe_float(r.get("pnl"), None) is not None and float(r.get("pnl") or 0.0) < 0.0]
    total_loss_abs = float(sum(-float(r["pnl"]) for r in losses)) if losses else 0.0
    top_loss_exits = sorted(losses, key=lambda z: float(z.get("pnl") or 0.0))[:25]

    lev_edges = [0.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 1e9]
    lev_labels = ["<=3x", "3-5x", "5-8x", "8-12x", "12-20x", "20-50x", "50x+"]
    cap_edges = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 2.0, 1e9]
    cap_labels = ["<=5%", "5-10%", "10-20%", "20-40%", "40-80%", "80-200%", "200%+"]

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

    cf = _load_json(cf_path)
    reason_matrix = _load_json(rm_path)
    timing_validation = _timing_validation(cf=cf, reason_matrix=reason_matrix, post_rows=post_rows)
    scale_plan = _capital_scale_plan(post_summary=post_summary, timing_validation=timing_validation, capitals=capitals)

    report = {
        "generated_at_ms": int(dt.datetime.now().timestamp() * 1000),
        "mode": "since_id",
        "db_path": str(db_path),
        "batch_window": {
            "since_id": int(since_id),
            "batch_id": int(max(0, args.batch_id)),
            "target_new": int(max(1, args.target_new)),
            "window_exit_n": int(len(post_rows)),
        },
        "coverage": cover,
        "post_period_summary": post_summary,
        "prev_same_exit_count_summary": prev_summary,
        "delta_vs_prev_same_exit_count": {
            "sum_pnl_delta": (post_summary.get("sum_pnl") or 0.0) - (prev_summary.get("sum_pnl") or 0.0),
            "win_rate_delta": (post_summary.get("win_rate") or 0.0) - (prev_summary.get("win_rate") or 0.0),
            "avg_pnl_delta": (post_summary.get("avg_pnl") or 0.0) - (prev_summary.get("avg_pnl") or 0.0),
            "avg_roe_delta": (post_summary.get("avg_roe") or 0.0) - (prev_summary.get("avg_roe") or 0.0),
            "direction_hit_delta": (post_summary.get("direction_hit") or 0.0) - (prev_summary.get("direction_hit") or 0.0),
            "avg_hold_sec_delta": (post_summary.get("avg_hold_sec") or 0.0) - (prev_summary.get("avg_hold_sec") or 0.0),
        },
        "loss_concentration": {
            "loss_rows_n": int(len(losses)),
            "total_loss_abs": float(total_loss_abs),
            "top_loss_exits": top_loss_exits,
            "top_loss_symbols": _pack_loss_agg(post_rows, "symbol")[:20],
            "top_loss_reasons": _pack_loss_agg(post_rows, "reason_main")[:20],
            "top_symbol_reason_pairs": _top_k_sum(
                [{"pair": f"{r.get('symbol')}::{r.get('reason_main')}", "pnl": r.get("pnl")} for r in post_rows],
                key="pair",
                value_key="pnl",
                k=20,
            ),
            "top5_sum_pnl": float(sum(float(r.get("pnl") or 0.0) for r in top_loss_exits[:5])),
            "top10_sum_pnl": float(sum(float(r.get("pnl") or 0.0) for r in top_loss_exits[:10])),
        },
        "leverage_bins": _bin_report(post_rows, "leverage", lev_edges, lev_labels),
        "capital_ratio_bins": _bin_report(post_rows, "cap_ratio_entry", cap_edges, cap_labels),
        "leverage_capital_heatmap": heat_rows[:40],
        "drawdown_linked_losses": _drawdown_report(post_rows),
        "correlation": _corr_report(post_rows),
        "timing_validation": timing_validation,
        "capital_scale_plan": scale_plan,
    }

    snapshot = {
        "timestamp_ms": int(report["generated_at_ms"]),
        "batch_window": dict(report["batch_window"]),
        "post_period_summary": dict(post_summary),
        "delta_vs_prev_same_exit_count": dict(report["delta_vs_prev_same_exit_count"]),
        "timing_validation": {
            "timing_issue_validated": bool((timing_validation or {}).get("timing_issue_validated") is True),
            "avg_exit_regret": _safe_float((timing_validation or {}).get("avg_exit_regret"), None),
            "early_like_rate": _safe_float((timing_validation or {}).get("early_like_rate"), None),
            "exit_regret_by_reason": (timing_validation or {}).get("exit_regret_by_reason") if isinstance((timing_validation or {}).get("exit_regret_by_reason"), dict) else {},
        },
    }
    history_meta = _append_history(history_path, snapshot)
    report["cumulative_compare"] = history_meta

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_top_losses_csv(post_rows, top_csv, top_n=120)

    print(
        json.dumps(
            {
                "mode": report["mode"],
                "since_id": since_id,
                "batch_id": report["batch_window"]["batch_id"],
                "post_exit_n": report["post_period_summary"]["n"],
                "post_sum_pnl": report["post_period_summary"]["sum_pnl"],
                "post_win_rate": report["post_period_summary"]["win_rate"],
                "prev_sum_pnl": report["prev_same_exit_count_summary"]["sum_pnl"],
                "delta_sum_pnl": report["delta_vs_prev_same_exit_count"]["sum_pnl_delta"],
                "timing_issue_validated": bool((timing_validation or {}).get("timing_issue_validated") is True),
                "history_snapshots": int((history_meta or {}).get("total_snapshots") or 0),
                "report": str(out_path),
                "top_csv": str(top_csv),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


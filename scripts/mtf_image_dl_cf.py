#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ONE_MIN_MS = 60_000


@dataclass
class TradeRow:
    trade_uid: str
    symbol: str
    ts_exit_ms: int
    ts_entry_ms: int
    hold_sec: float
    pnl: float
    regime: str


@dataclass
class TFSeries:
    times: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


class TinyMTFConv(nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 48, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(48, 1)
        self._h = h
        self._w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = z.view(z.size(0), -1)
        return self.fc(z).squeeze(-1)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if not math.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _parse_symbols(raw: str | None) -> set[str]:
    txt = str(raw or "").strip()
    out: set[str] = set()
    if not txt:
        return out
    for tok in txt.replace(";", ",").split(","):
        sym = str(tok or "").strip().upper()
        if sym:
            out.add(sym)
    return out


def _load_trades(
    db_path: Path,
    *,
    max_symbols: int,
    max_trades: int,
    excluded_symbols: set[str],
    min_hold_sec: float,
) -> list[TradeRow]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT
            trade_uid,
            symbol,
            timestamp_ms,
            hold_duration_sec,
            realized_pnl,
            COALESCE(regime, 'chop')
        FROM trades
        WHERE action='EXIT' AND realized_pnl IS NOT NULL
        ORDER BY timestamp_ms ASC
        """
    ).fetchall()
    conn.close()

    parsed: list[TradeRow] = []
    for trade_uid, symbol, ts_ms, hold_sec, pnl, regime in rows:
        symbol = str(symbol or "")
        if not symbol:
            continue
        if symbol.strip().upper() in excluded_symbols:
            continue
        ts_exit_ms = int(_safe_float(ts_ms, 0))
        if ts_exit_ms <= 0:
            continue
        hold = max(0.0, _safe_float(hold_sec, 0.0))
        if hold < float(min_hold_sec):
            continue
        ts_entry_ms = int(ts_exit_ms - hold * 1000.0)
        if ts_entry_ms <= 0:
            continue
        parsed.append(
            TradeRow(
                trade_uid=str(trade_uid or ""),
                symbol=symbol,
                ts_exit_ms=ts_exit_ms,
                ts_entry_ms=ts_entry_ms,
                hold_sec=hold,
                pnl=_safe_float(pnl, 0.0),
                regime=str(regime or "chop"),
            )
        )

    if not parsed:
        return []

    # Use symbols with sufficient samples first.
    sym_count: dict[str, int] = {}
    for t in parsed:
        sym_count[t.symbol] = sym_count.get(t.symbol, 0) + 1
    selected_symbols = {s for s, _ in sorted(sym_count.items(), key=lambda kv: kv[1], reverse=True)[:max_symbols]}
    filtered = [t for t in parsed if t.symbol in selected_symbols]
    if len(filtered) > max_trades:
        filtered = filtered[-max_trades:]
    return filtered


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    tag = symbol.replace("/", "_").replace(":", "_")
    return cache_dir / f"{tag}_1m.csv"


def _load_cache(cache_file: Path) -> pd.DataFrame | None:
    if not cache_file.exists():
        return None
    try:
        df = pd.read_csv(cache_file)
        if not {"ts_ms", "open", "high", "low", "close", "volume"}.issubset(df.columns):
            return None
        df = df.sort_values("ts_ms").drop_duplicates(subset=["ts_ms"], keep="last").reset_index(drop=True)
        return df
    except Exception:
        return None


def _save_cache(cache_file: Path, df: pd.DataFrame) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("ts_ms").drop_duplicates(subset=["ts_ms"], keep="last").to_csv(cache_file, index=False)


def _fetch_ohlcv_1m(
    ex: ccxt.Exchange,
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    retries: int = 5,
) -> pd.DataFrame:
    since = int(start_ms)
    out: list[list[float]] = []
    while since < int(end_ms):
        last_err: Exception | None = None
        batch: list[list[float]] = []
        for attempt in range(1, retries + 1):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe="1m", since=since, limit=1000)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(min(5.0, 0.4 * attempt))
        if last_err is not None:
            raise RuntimeError(f"fetch_ohlcv failed symbol={symbol} since={since}: {last_err}") from last_err
        if not batch:
            since += 1000 * ONE_MIN_MS
            continue
        out.extend(batch)
        since = int(batch[-1][0]) + ONE_MIN_MS

    if not out:
        return pd.DataFrame(columns=["ts_ms", "open", "high", "low", "close", "volume"])
    arr = np.asarray(out, dtype=np.float64)
    df = pd.DataFrame(
        {
            "ts_ms": arr[:, 0].astype(np.int64),
            "open": arr[:, 1],
            "high": arr[:, 2],
            "low": arr[:, 3],
            "close": arr[:, 4],
            "volume": arr[:, 5],
        }
    )
    df = df.sort_values("ts_ms").drop_duplicates(subset=["ts_ms"], keep="last").reset_index(drop=True)
    return df


def _ensure_symbol_ohlcv(
    ex: ccxt.Exchange,
    cache_dir: Path,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    cache_file = _cache_path(cache_dir, symbol)
    cached = _load_cache(cache_file)
    need_fetch = cached is None
    if cached is not None and not cached.empty:
        c_min = int(cached["ts_ms"].min())
        c_max = int(cached["ts_ms"].max())
        if c_min > start_ms + ONE_MIN_MS or c_max < end_ms - ONE_MIN_MS:
            need_fetch = True
    if need_fetch:
        fresh = _fetch_ohlcv_1m(ex, symbol, start_ms, end_ms)
        if fresh.empty:
            raise RuntimeError(f"No ohlcv fetched for {symbol}")
        _save_cache(cache_file, fresh)
        return fresh
    return cached if cached is not None else pd.DataFrame()


def _resample_tf(df_1m: pd.DataFrame, tf_min: int) -> TFSeries:
    if df_1m.empty:
        z = np.zeros(0, dtype=np.float64)
        return TFSeries(times=z.astype(np.int64), open=z, high=z, low=z, close=z, volume=z)
    if int(tf_min) <= 1:
        d = df_1m.copy()
    else:
        d = df_1m.copy()
        d["ts"] = pd.to_datetime(d["ts_ms"], unit="ms", utc=True)
        d = d.set_index("ts")
        rule = f"{int(tf_min)}min"
        d = d.resample(rule, label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        d = d.dropna().reset_index()
        # pandas>=2.0 removed Series.view; use astype for epoch-ms conversion.
        d["ts_ms"] = (d["ts"].astype("int64") // 1_000_000).astype(np.int64)
    d = d.sort_values("ts_ms").drop_duplicates(subset=["ts_ms"], keep="last")
    return TFSeries(
        times=d["ts_ms"].to_numpy(dtype=np.int64),
        open=d["open"].to_numpy(dtype=np.float64),
        high=d["high"].to_numpy(dtype=np.float64),
        low=d["low"].to_numpy(dtype=np.float64),
        close=d["close"].to_numpy(dtype=np.float64),
        volume=d["volume"].to_numpy(dtype=np.float64),
    )


def _build_image(
    tf_map: dict[int, TFSeries],
    entry_ms: int,
    tf_list: list[int],
    lookback: int,
) -> np.ndarray | None:
    mats: list[np.ndarray] = []
    min_required = max(10, lookback // 3)
    for tf in tf_list:
        s = tf_map.get(int(tf))
        if s is None or s.times.size == 0:
            return None
        idx = int(np.searchsorted(s.times, int(entry_ms), side="right"))
        if idx <= 0:
            return None
        st = max(0, idx - int(lookback))
        if (idx - st) < min_required:
            return None
        c = s.close[st:idx]
        h = s.high[st:idx]
        l = s.low[st:idx]
        v = s.volume[st:idx]
        if c.size == 0:
            return None
        c = np.maximum(c, 1e-12)
        logc = np.log(c)
        ret = np.zeros_like(logc)
        if ret.size > 1:
            ret[1:] = np.diff(logc)
        hl = (h - l) / c
        v_mean = float(np.mean(v))
        v_std = float(np.std(v)) + 1e-6
        vz = (v - v_mean) / v_std
        feat = np.vstack([ret, hl, vz]).astype(np.float32)
        if feat.shape[1] < lookback:
            pad = np.zeros((feat.shape[0], int(lookback - feat.shape[1])), dtype=np.float32)
            feat = np.hstack([pad, feat])
        feat = np.clip(feat, -5.0, 5.0)
        mats.append(feat)
    if not mats:
        return None
    return np.vstack(mats).astype(np.float32)


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(np.int64)
    s = y_score.astype(np.float64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    sum_ranks_pos = float(ranks[pos].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _metrics(pnls: np.ndarray) -> dict[str, float]:
    if pnls.size == 0:
        return {
            "n": 0,
            "pnl": 0.0,
            "wr": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "pf": 0.0,
        }
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = float(np.sum(wins)) if wins.size else 0.0
    gross_loss_abs = float(np.abs(np.sum(losses))) if losses.size else 0.0
    return {
        "n": int(pnls.size),
        "pnl": float(np.sum(pnls)),
        "wr": float(np.mean(pnls > 0)),
        "avg_pnl": float(np.mean(pnls)),
        "avg_win": float(np.mean(wins)) if wins.size else 0.0,
        "avg_loss": float(np.mean(np.abs(losses))) if losses.size else 0.0,
        "pf": float(gross_win / gross_loss_abs) if gross_loss_abs > 1e-12 else 0.0,
    }


def _pick_best(rows: list[dict[str, Any]], min_n: int) -> dict[str, Any] | None:
    cands = [r for r in rows if int(r.get("n", 0)) >= int(min_n)]
    if not cands:
        return None
    cands.sort(key=lambda r: (float(r.get("pnl", 0.0)), float(r.get("wr", 0.0))), reverse=True)
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-timeframe image DL + counterfactual gate")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--out", default="state/mtf_image_dl_cf_report.json")
    ap.add_argument("--scores-out", default="state/mtf_image_trade_scores.json")
    ap.add_argument("--model-out", default="state/mtf_image_model.pt")
    ap.add_argument("--cache-dir", default="state/mtf_ohlcv_cache")
    ap.add_argument("--max-symbols", type=int, default=12)
    ap.add_argument("--max-trades", type=int, default=2200)
    ap.add_argument("--min-hold-sec", type=float, default=30.0)
    ap.add_argument("--lookback-bars", type=int, default=48)
    ap.add_argument("--timeframes", default="1,3,5,15,30,60,240")
    ap.add_argument("--train-ratio", type=float, default=0.75)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-n", type=int, default=80)
    ap.add_argument("--exclude-symbol", action="append", default=[])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    excluded = set()
    excluded.update(_parse_symbols(os.environ.get("RESEARCH_EXCLUDE_SYMBOLS")))
    excluded.update(_parse_symbols(os.environ.get("AUTO_REVAL_EXCLUDE_SYMBOLS")))
    for raw in args.exclude_symbol or []:
        excluded.update(_parse_symbols(raw))

    db_path = Path(args.db)
    trades = _load_trades(
        db_path,
        max_symbols=max(1, int(args.max_symbols)),
        max_trades=max(200, int(args.max_trades)),
        excluded_symbols=excluded,
        min_hold_sec=float(args.min_hold_sec),
    )
    if not trades:
        raise SystemExit("No trades available for training")

    tf_list = [max(1, int(x.strip())) for x in str(args.timeframes).split(",") if x.strip()]
    tf_list = sorted(set(tf_list))
    max_tf = max(tf_list)
    lookback = max(24, int(args.lookback_bars))

    symbols = sorted({t.symbol for t in trades})
    print(f"[INFO] trades={len(trades)} symbols={len(symbols)} tf={tf_list} lookback={lookback}")

    ex = ccxt.bybit({"enableRateLimit": True, "timeout": 30000})
    try:
        ex.load_markets()
    except Exception:
        pass

    # Build per-symbol timeframe maps.
    sym_tf_map: dict[str, dict[int, TFSeries]] = {}
    cache_dir = Path(args.cache_dir)
    needed_pad_ms = int((lookback + 4) * max_tf * ONE_MIN_MS)

    by_symbol: dict[str, list[TradeRow]] = {}
    for t in trades:
        by_symbol.setdefault(t.symbol, []).append(t)

    for sym, rows in by_symbol.items():
        s_min = min(r.ts_entry_ms for r in rows) - needed_pad_ms
        s_max = max(r.ts_entry_ms for r in rows) + ONE_MIN_MS * 3
        try:
            df_1m = _ensure_symbol_ohlcv(ex, cache_dir, sym, s_min, s_max)
        except Exception as e:
            print(f"[WARN] ohlcv fetch failed symbol={sym}: {e}")
            continue
        if df_1m.empty:
            print(f"[WARN] empty ohlcv symbol={sym}")
            continue
        tf_map: dict[int, TFSeries] = {}
        for tf in tf_list:
            tf_map[int(tf)] = _resample_tf(df_1m, int(tf))
        sym_tf_map[sym] = tf_map

    try:
        ex.close()
    except Exception:
        pass

    # Build training arrays.
    imgs: list[np.ndarray] = []
    ys: list[int] = []
    pnls: list[float] = []
    regimes: list[str] = []
    trade_ids: list[str] = []
    ts_entries: list[int] = []
    symbols_used: list[str] = []
    skipped = 0

    for t in trades:
        tf_map = sym_tf_map.get(t.symbol)
        if not tf_map:
            skipped += 1
            continue
        img = _build_image(tf_map, t.ts_entry_ms, tf_list, lookback)
        if img is None:
            skipped += 1
            continue
        imgs.append(img)
        ys.append(1 if t.pnl > 0 else 0)
        pnls.append(float(t.pnl))
        regimes.append(str(t.regime or "chop"))
        trade_ids.append(str(t.trade_uid or ""))
        ts_entries.append(int(t.ts_entry_ms))
        symbols_used.append(t.symbol)

    if len(imgs) < 200:
        raise SystemExit(f"Insufficient samples after image build: {len(imgs)}")

    X = np.stack(imgs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    pnl_arr = np.asarray(pnls, dtype=np.float64)
    ts_arr = np.asarray(ts_entries, dtype=np.int64)
    regimes_arr = np.asarray(regimes, dtype=object)

    # Chronological split.
    order = np.argsort(ts_arr)
    X = X[order]
    y = y[order]
    pnl_arr = pnl_arr[order]
    ts_arr = ts_arr[order]
    regimes_arr = regimes_arr[order]
    trade_ids = [trade_ids[i] for i in order]
    symbols_used = [symbols_used[i] for i in order]

    split = int(max(100, min(len(X) - 50, int(len(X) * float(args.train_ratio)))))
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    h, w = int(X.shape[1]), int(X.shape[2])
    model = TinyMTFConv(h, w).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    pos = float(np.sum(y_tr > 0.5))
    neg = float(np.sum(y_tr <= 0.5))
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    bs = max(16, int(args.batch_size))
    n_epochs = max(1, int(args.epochs))
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_auc: list[float] = []

    X_va_t = torch.from_numpy(X_va[:, None, :, :]).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    for ep in range(1, n_epochs + 1):
        model.train()
        idx = np.random.permutation(len(X_tr))
        loss_sum = 0.0
        n_seen = 0
        for st in range(0, len(idx), bs):
            bi = idx[st : st + bs]
            xb = torch.from_numpy(X_tr[bi][:, None, :, :]).to(device)
            yb = torch.from_numpy(y_tr[bi]).to(device)
            opt.zero_grad(set_to_none=True)
            lg = model(xb)
            loss = loss_fn(lg, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * int(len(bi))
            n_seen += int(len(bi))
        tr_loss = float(loss_sum / max(1, n_seen))
        train_losses.append(tr_loss)

        model.eval()
        with torch.no_grad():
            lg_va = model(X_va_t)
            va_loss = float(loss_fn(lg_va, y_va_t).item())
            pr_va = torch.sigmoid(lg_va).detach().cpu().numpy()
        val_losses.append(va_loss)
        val_auc.append(_auc(y_va.astype(np.int64), pr_va.astype(np.float64)))
        print(
            f"[TRAIN] epoch={ep:02d}/{n_epochs} "
            f"loss={tr_loss:.5f} val_loss={va_loss:.5f} val_auc={val_auc[-1]:.4f}"
        )

    # Full predictions.
    model.eval()
    with torch.no_grad():
        X_all_t = torch.from_numpy(X[:, None, :, :]).to(device)
        prob = torch.sigmoid(model(X_all_t)).detach().cpu().numpy().astype(np.float64)

    # Save model and scores.
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "timeframes": tf_list,
            "lookback_bars": lookback,
            "height": h,
            "width": w,
            "train_samples": int(len(X_tr)),
            "val_samples": int(len(X_va)),
            "val_auc_last": float(val_auc[-1] if val_auc else 0.0),
        },
        args.model_out,
    )

    score_rows = []
    for i in range(len(prob)):
        score_rows.append(
            {
                "trade_uid": trade_ids[i],
                "symbol": symbols_used[i],
                "entry_ts_ms": int(ts_arr[i]),
                "regime": str(regimes_arr[i]),
                "mtf_dl_prob": float(prob[i]),
                "pnl": float(pnl_arr[i]),
            }
        )
    Path(args.scores_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores_out).write_text(json.dumps(score_rows, ensure_ascii=False, indent=2))

    # Counterfactual evaluation is computed on validation-only slice to reduce in-sample bias.
    prob_eval = prob[split:]
    pnl_eval = pnl_arr[split:]
    regimes_eval = regimes_arr[split:]
    baseline_all = _metrics(pnl_eval)
    is_chop = (regimes_eval.astype(str) == "chop")
    baseline_chop = _metrics(pnl_eval[is_chop])
    thresholds = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]
    quantiles = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]

    rows_global: list[dict[str, Any]] = []
    rows_chop_only: list[dict[str, Any]] = []
    rows_global_quantile: list[dict[str, Any]] = []
    rows_chop_only_quantile: list[dict[str, Any]] = []
    for thr in thresholds:
        keep_global = prob_eval >= float(thr)
        m_global = _metrics(pnl_eval[keep_global])
        m_global["threshold"] = float(thr)
        rows_global.append(m_global)

        keep_chop_only = np.ones(len(prob_eval), dtype=bool)
        keep_chop_only[is_chop] = prob_eval[is_chop] >= float(thr)
        m_chop_only = _metrics(pnl_eval[keep_chop_only])
        m_chop_only["threshold"] = float(thr)
        # extra metric: chop-only subset after gate
        m_chop_gate = _metrics(pnl_eval[np.logical_and(is_chop, prob_eval >= float(thr))])
        m_chop_only["chop_subset_n"] = int(m_chop_gate["n"])
        m_chop_only["chop_subset_pnl"] = float(m_chop_gate["pnl"])
        m_chop_only["chop_subset_wr"] = float(m_chop_gate["wr"])
        rows_chop_only.append(m_chop_only)

    chop_prob = prob_eval[is_chop]
    for q in quantiles:
        thr_global_q = float(np.quantile(prob_eval, float(q)))
        keep_global_q = prob_eval >= thr_global_q
        m_global_q = _metrics(pnl_eval[keep_global_q])
        m_global_q["quantile"] = float(q)
        m_global_q["threshold"] = float(thr_global_q)
        rows_global_quantile.append(m_global_q)

        keep_chop_q = np.ones(len(prob_eval), dtype=bool)
        if chop_prob.size > 0:
            thr_chop_q = float(np.quantile(chop_prob, float(q)))
            keep_chop_q[is_chop] = prob_eval[is_chop] >= thr_chop_q
            chop_gate_q = _metrics(pnl_eval[np.logical_and(is_chop, prob_eval >= thr_chop_q)])
        else:
            thr_chop_q = 1.0
            keep_chop_q[is_chop] = False
            chop_gate_q = _metrics(np.asarray([], dtype=np.float64))
        m_chop_q = _metrics(pnl_eval[keep_chop_q])
        m_chop_q["quantile"] = float(q)
        m_chop_q["threshold"] = float(thr_chop_q)
        m_chop_q["chop_subset_n"] = int(chop_gate_q["n"])
        m_chop_q["chop_subset_pnl"] = float(chop_gate_q["pnl"])
        m_chop_q["chop_subset_wr"] = float(chop_gate_q["wr"])
        rows_chop_only_quantile.append(m_chop_q)

    best_global = _pick_best(rows_global, min_n=max(40, int(args.min_n)))
    best_chop_only = _pick_best(rows_chop_only, min_n=max(40, int(args.min_n)))
    best_global_quantile = _pick_best(rows_global_quantile, min_n=max(40, int(args.min_n)))
    best_chop_only_quantile = _pick_best(rows_chop_only_quantile, min_n=max(40, int(args.min_n)))

    report = {
        "timestamp": int(time.time()),
        "input": {
            "db": str(args.db),
            "excluded_symbols": sorted(excluded),
            "max_symbols": int(args.max_symbols),
            "max_trades": int(args.max_trades),
            "timeframes": tf_list,
            "lookback_bars": int(lookback),
            "epochs": int(n_epochs),
            "batch_size": int(bs),
            "train_ratio": float(args.train_ratio),
            "min_n": int(args.min_n),
        },
        "dataset": {
            "n_trades_loaded": int(len(trades)),
            "n_samples_built": int(len(X)),
            "n_skipped": int(skipped),
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_cf_eval": int(len(prob_eval)),
            "cf_eval_start_ts_ms": int(ts_arr[split]) if split < len(ts_arr) else int(ts_arr[-1]),
            "symbols_used": sorted(set(symbols_used)),
        },
        "training": {
            "device": str(device),
            "train_loss_last": float(train_losses[-1] if train_losses else 0.0),
            "val_loss_last": float(val_losses[-1] if val_losses else 0.0),
            "val_auc_last": float(val_auc[-1] if val_auc else 0.0),
            "train_loss_curve": [float(x) for x in train_losses],
            "val_loss_curve": [float(x) for x in val_losses],
            "val_auc_curve": [float(x) for x in val_auc],
        },
        "baseline": {"all": baseline_all, "chop": baseline_chop},
        "threshold_sweep": {
            "global_gate": rows_global,
            "chop_only_gate": rows_chop_only,
            "global_gate_quantile": rows_global_quantile,
            "chop_only_gate_quantile": rows_chop_only_quantile,
        },
        "best": {
            "global_gate": best_global,
            "chop_only_gate": best_chop_only,
            "global_gate_quantile": best_global_quantile,
            "chop_only_gate_quantile": best_chop_only_quantile,
        },
        "artifacts": {
            "scores_out": str(args.scores_out),
            "model_out": str(args.model_out),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(
        "[DONE] "
        f"samples={len(X)} val_auc={report['training']['val_auc_last']:.4f} "
        f"baseline_pnl={baseline_all['pnl']:.2f} "
        f"best_global={None if best_global is None else best_global.get('pnl')} "
        f"best_chop_only={None if best_chop_only is None else best_chop_only.get('pnl')} "
        f"best_global_q={None if best_global_quantile is None else best_global_quantile.get('pnl')} "
        f"best_chop_q={None if best_chop_only_quantile is None else best_chop_only_quantile.get('pnl')}"
    )
    print(f"[DONE] report={args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mtf_multitask import (
    EXIT_REASON_LABELS,
    REGIME_LABELS,
    SYMBOL_GROUP_LABELS,
    MTFMultiTaskNet,
    binary_auc,
    build_mtf_image_from_1m,
    infer_symbol_group,
    normalize_exit_reason,
    normalize_regime,
)


ONE_MIN_MS = 60_000


@dataclass
class TradeRow:
    trade_uid: str
    symbol: str
    side: str
    ts_entry_ms: int
    ts_exit_ms: int
    entry_price: float
    exit_price: float
    hold_sec: float
    pnl: float
    regime: str
    exit_reason: str
    symbol_group: str


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if not math.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


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


def _resolve_link_id(d: dict[str, Any]) -> str:
    return str(d.get("entry_link_id") or d.get("entry_id") or "").strip()


def _load_trades(
    db_path: Path,
    *,
    max_symbols: int,
    max_trades: int,
    excluded_symbols: set[str],
    min_hold_sec: float,
) -> list[TradeRow]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            id,
            trade_uid,
            symbol,
            action,
            side,
            fill_price,
            timestamp_ms,
            hold_duration_sec,
            realized_pnl,
            COALESCE(regime, ''),
            COALESCE(entry_reason, ''),
            entry_link_id,
            entry_id
        FROM trades
        ORDER BY id ASC
        """
    ).fetchall()
    conn.close()

    enters: dict[str, dict[str, Any]] = {}
    parsed: list[TradeRow] = []

    for rr in rows:
        d = dict(rr)
        action = str(d.get("action") or "").upper().strip()
        link_id = _resolve_link_id(d)
        if not link_id:
            continue

        if action == "ENTER":
            enters[link_id] = d
            continue

        if action != "EXIT":
            continue

        entry = enters.get(link_id)
        if not isinstance(entry, dict):
            continue

        symbol = str(d.get("symbol") or entry.get("symbol") or "").strip()
        if not symbol:
            continue
        if symbol.upper() in excluded_symbols:
            continue

        side = str(entry.get("side") or d.get("side") or "LONG").upper().strip()
        if side not in ("LONG", "SHORT"):
            side = "LONG"

        ts_exit_ms = _safe_int(d.get("timestamp_ms"), 0)
        if ts_exit_ms <= 0:
            continue

        hold_sec = max(0.0, _safe_float(d.get("hold_duration_sec"), 0.0))
        if hold_sec < float(min_hold_sec):
            continue

        ts_entry_ms = _safe_int(entry.get("timestamp_ms"), 0)
        if ts_entry_ms <= 0:
            ts_entry_ms = int(ts_exit_ms - hold_sec * 1000.0)
        if ts_entry_ms <= 0:
            continue

        entry_px = _safe_float(entry.get("fill_price"), 0.0)
        exit_px = _safe_float(d.get("fill_price"), 0.0)
        if entry_px <= 0 or exit_px <= 0:
            continue

        pnl = _safe_float(d.get("realized_pnl"), 0.0)
        regime = normalize_regime(d.get("regime") or entry.get("regime") or "")
        exit_reason = normalize_exit_reason(d.get("entry_reason"))
        sym_group = infer_symbol_group(symbol)
        trade_uid = str(d.get("trade_uid") or entry.get("trade_uid") or link_id)

        parsed.append(
            TradeRow(
                trade_uid=trade_uid,
                symbol=symbol,
                side=side,
                ts_entry_ms=int(ts_entry_ms),
                ts_exit_ms=int(ts_exit_ms),
                entry_price=float(entry_px),
                exit_price=float(exit_px),
                hold_sec=float(hold_sec),
                pnl=float(pnl),
                regime=str(regime),
                exit_reason=str(exit_reason),
                symbol_group=str(sym_group),
            )
        )

    if not parsed:
        return []

    sym_count: dict[str, int] = {}
    for t in parsed:
        sym_count[t.symbol] = sym_count.get(t.symbol, 0) + 1
    selected = {s for s, _ in sorted(sym_count.items(), key=lambda kv: kv[1], reverse=True)[:max_symbols]}
    filtered = [t for t in parsed if t.symbol in selected]
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
        need_cols = {"ts_ms", "open", "high", "low", "close", "volume"}
        if not need_cols.issubset(df.columns):
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


def _bce_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = float(np.sum(y > 0.5))
    neg = float(np.sum(y <= 0.5))
    return torch.tensor([max(1.0, neg / max(1.0, pos))], dtype=torch.float32)


def _env_excluded_symbols(extra: list[str]) -> set[str]:
    out = set()
    out.update(_parse_symbols(os.environ.get("RESEARCH_EXCLUDE_SYMBOLS")))
    out.update(_parse_symbols(os.environ.get("AUTO_REVAL_EXCLUDE_SYMBOLS")))
    for raw in extra:
        out.update(_parse_symbols(raw))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-timeframe multitask DL + counterfactual gate")
    ap.add_argument("--db", default="state/bot_data_live.db")
    ap.add_argument("--out", default="state/mtf_image_dl_cf_report.json")
    ap.add_argument("--scores-out", default="state/mtf_image_trade_scores.json")
    ap.add_argument("--model-out", default="state/mtf_image_model.pt")
    ap.add_argument("--cache-dir", default="state/mtf_ohlcv_cache")
    ap.add_argument("--max-symbols", type=int, default=12)
    ap.add_argument("--max-trades", type=int, default=2200)
    ap.add_argument("--min-hold-sec", type=float, default=30.0)
    ap.add_argument("--lookback-bars", type=int, default=16)
    ap.add_argument("--timeframes", default="1,3,5,15")
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

    excluded = _env_excluded_symbols(args.exclude_symbol or [])
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
    lookback = max(12, int(args.lookback_bars))
    max_tf = max(tf_list)

    symbols = sorted({t.symbol for t in trades})
    print(f"[INFO] trades={len(trades)} symbols={len(symbols)} tf={tf_list} lookback={lookback}")

    ex = ccxt.bybit({"enableRateLimit": True, "timeout": 30000})
    try:
        ex.load_markets()
    except Exception:
        pass

    by_symbol: dict[str, list[TradeRow]] = {}
    for t in trades:
        by_symbol.setdefault(t.symbol, []).append(t)

    sym_ohlcv: dict[str, pd.DataFrame] = {}
    pad_ms = int((lookback + 6) * max_tf * ONE_MIN_MS)
    cache_dir = Path(args.cache_dir)
    for sym, rows in by_symbol.items():
        s_min = min(r.ts_entry_ms for r in rows) - pad_ms
        s_max = max(r.ts_entry_ms for r in rows) + ONE_MIN_MS * 5
        try:
            df_1m = _ensure_symbol_ohlcv(ex, cache_dir, sym, s_min, s_max)
        except Exception as e:
            print(f"[WARN] ohlcv fetch failed symbol={sym}: {e}")
            continue
        if df_1m.empty:
            print(f"[WARN] empty ohlcv symbol={sym}")
            continue
        sym_ohlcv[sym] = df_1m

    try:
        ex.close()
    except Exception:
        pass

    regime_to_id = {k: i for i, k in enumerate([x.lower() for x in REGIME_LABELS])}
    group_to_id = {k: i for i, k in enumerate([x.lower() for x in SYMBOL_GROUP_LABELS])}
    exit_to_id = {k: i for i, k in enumerate([x.lower() for x in EXIT_REASON_LABELS])}

    imgs: list[np.ndarray] = []
    y_win: list[int] = []
    y_long_fit: list[int] = []
    y_short_fit: list[int] = []
    y_hold_sec: list[float] = []
    y_exit_cls: list[int] = []

    pnls: list[float] = []
    sides: list[str] = []
    symbols_used: list[str] = []
    regimes_used: list[str] = []
    groups_used: list[str] = []
    trade_ids: list[str] = []
    ts_entries: list[int] = []

    skipped = 0
    for t in trades:
        df = sym_ohlcv.get(t.symbol)
        if df is None or df.empty:
            skipped += 1
            continue
        img = build_mtf_image_from_1m(
            ts_ms=df["ts_ms"].to_numpy(dtype=np.int64),
            open_=df["open"].to_numpy(dtype=np.float64),
            high=df["high"].to_numpy(dtype=np.float64),
            low=df["low"].to_numpy(dtype=np.float64),
            close=df["close"].to_numpy(dtype=np.float64),
            volume=df["volume"].to_numpy(dtype=np.float64),
            entry_ts_ms=int(t.ts_entry_ms),
            tf_list=tf_list,
            lookback=lookback,
        )
        if img is None:
            skipped += 1
            continue

        raw_ret = float((t.exit_price / max(t.entry_price, 1e-12)) - 1.0)
        long_fit = 1 if raw_ret > 0 else 0
        short_fit = 1 if raw_ret < 0 else 0
        win = 1 if float(t.pnl) > 0 else 0

        regime_norm = normalize_regime(t.regime)
        group_norm = infer_symbol_group(t.symbol)
        exit_norm = normalize_exit_reason(t.exit_reason)

        imgs.append(img)
        y_win.append(win)
        y_long_fit.append(long_fit)
        y_short_fit.append(short_fit)
        y_hold_sec.append(float(max(0.0, t.hold_sec)))
        y_exit_cls.append(int(exit_to_id.get(exit_norm, exit_to_id["other"])))

        pnls.append(float(t.pnl))
        sides.append(str(t.side))
        symbols_used.append(str(t.symbol))
        regimes_used.append(regime_norm)
        groups_used.append(group_norm)
        trade_ids.append(str(t.trade_uid))
        ts_entries.append(int(t.ts_entry_ms))

    if len(imgs) < 200:
        raise SystemExit(f"Insufficient samples after image build: {len(imgs)}")

    X = np.stack(imgs, axis=0).astype(np.float32)
    y_win_arr = np.asarray(y_win, dtype=np.float32)
    y_long_arr = np.asarray(y_long_fit, dtype=np.float32)
    y_short_arr = np.asarray(y_short_fit, dtype=np.float32)
    y_hold_arr = np.asarray(y_hold_sec, dtype=np.float32)
    y_exit_arr = np.asarray(y_exit_cls, dtype=np.int64)

    pnl_arr = np.asarray(pnls, dtype=np.float64)
    ts_arr = np.asarray(ts_entries, dtype=np.int64)
    side_arr = np.asarray(sides, dtype=object)
    regime_arr = np.asarray(regimes_used, dtype=object)
    group_arr = np.asarray(groups_used, dtype=object)

    order = np.argsort(ts_arr)
    X = X[order]
    y_win_arr = y_win_arr[order]
    y_long_arr = y_long_arr[order]
    y_short_arr = y_short_arr[order]
    y_hold_arr = y_hold_arr[order]
    y_exit_arr = y_exit_arr[order]
    pnl_arr = pnl_arr[order]
    ts_arr = ts_arr[order]
    side_arr = side_arr[order]
    regime_arr = regime_arr[order]
    group_arr = group_arr[order]
    trade_ids = [trade_ids[i] for i in order]
    symbols_used = [symbols_used[i] for i in order]

    regime_id_arr = np.asarray([regime_to_id.get(str(r).lower(), regime_to_id["unknown"]) for r in regime_arr], dtype=np.int64)
    group_id_arr = np.asarray([group_to_id.get(str(g).lower(), group_to_id["other"]) for g in group_arr], dtype=np.int64)

    split = int(max(100, min(len(X) - 50, int(len(X) * float(args.train_ratio)))))

    hold_log = np.log1p(np.maximum(y_hold_arr, 0.0))
    hold_mean = float(np.mean(hold_log[:split]))
    hold_std = float(np.std(hold_log[:split])) + 1e-6
    y_hold_norm = ((hold_log - hold_mean) / hold_std).astype(np.float32)

    X_tr, X_va = X[:split], X[split:]
    rg_tr, rg_va = regime_id_arr[:split], regime_id_arr[split:]
    gp_tr, gp_va = group_id_arr[:split], group_id_arr[split:]
    y_win_tr, y_win_va = y_win_arr[:split], y_win_arr[split:]
    y_long_tr, y_long_va = y_long_arr[:split], y_long_arr[split:]
    y_short_tr, y_short_va = y_short_arr[:split], y_short_arr[split:]
    y_hold_tr, y_hold_va = y_hold_norm[:split], y_hold_norm[split:]
    y_exit_tr, y_exit_va = y_exit_arr[:split], y_exit_arr[split:]

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    h, w = int(X.shape[1]), int(X.shape[2])
    model = MTFMultiTaskNet(
        h=h,
        w=w,
        n_regimes=len(regime_to_id),
        n_groups=len(group_to_id),
        n_exit_reasons=len(exit_to_id),
        embed_dim=8,
        hidden_dim=96,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    bce_win = nn.BCEWithLogitsLoss(pos_weight=_bce_pos_weight(y_win_tr).to(device))
    bce_long = nn.BCEWithLogitsLoss(pos_weight=_bce_pos_weight(y_long_tr).to(device))
    bce_short = nn.BCEWithLogitsLoss(pos_weight=_bce_pos_weight(y_short_tr).to(device))
    mse_hold = nn.SmoothL1Loss()
    ce_exit = nn.CrossEntropyLoss()

    bs = max(16, int(args.batch_size))
    n_epochs = max(1, int(args.epochs))

    tr_loss_curve: list[float] = []
    va_loss_curve: list[float] = []
    va_win_auc_curve: list[float] = []
    va_side_auc_curve: list[float] = []
    va_hold_mae_curve: list[float] = []
    va_exit_acc_curve: list[float] = []

    X_va_t = torch.from_numpy(X_va[:, None, :, :]).to(device)
    rg_va_t = torch.from_numpy(rg_va).to(device)
    gp_va_t = torch.from_numpy(gp_va).to(device)
    y_win_va_t = torch.from_numpy(y_win_va).to(device)
    y_long_va_t = torch.from_numpy(y_long_va).to(device)
    y_short_va_t = torch.from_numpy(y_short_va).to(device)
    y_hold_va_t = torch.from_numpy(y_hold_va).to(device)
    y_exit_va_t = torch.from_numpy(y_exit_va).to(device)

    for ep in range(1, n_epochs + 1):
        model.train()
        idx = np.random.permutation(len(X_tr))
        loss_sum = 0.0
        n_seen = 0

        for st in range(0, len(idx), bs):
            bi = idx[st: st + bs]
            xb = torch.from_numpy(X_tr[bi][:, None, :, :]).to(device)
            rb = torch.from_numpy(rg_tr[bi]).to(device)
            gb = torch.from_numpy(gp_tr[bi]).to(device)
            ywb = torch.from_numpy(y_win_tr[bi]).to(device)
            ylb = torch.from_numpy(y_long_tr[bi]).to(device)
            ysb = torch.from_numpy(y_short_tr[bi]).to(device)
            yhb = torch.from_numpy(y_hold_tr[bi]).to(device)
            yeb = torch.from_numpy(y_exit_tr[bi]).to(device)

            opt.zero_grad(set_to_none=True)
            out = model(xb, rb, gb)
            loss = (
                bce_win(out["win"], ywb)
                + 0.60 * bce_long(out["long"], ylb)
                + 0.60 * bce_short(out["short"], ysb)
                + 0.25 * mse_hold(out["hold"], yhb)
                + 0.35 * ce_exit(out["exit"], yeb)
            )
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * int(len(bi))
            n_seen += int(len(bi))

        tr_loss = float(loss_sum / max(1, n_seen))
        tr_loss_curve.append(tr_loss)

        model.eval()
        with torch.no_grad():
            out_va = model(X_va_t, rg_va_t, gp_va_t)
            loss_va = (
                bce_win(out_va["win"], y_win_va_t)
                + 0.60 * bce_long(out_va["long"], y_long_va_t)
                + 0.60 * bce_short(out_va["short"], y_short_va_t)
                + 0.25 * mse_hold(out_va["hold"], y_hold_va_t)
                + 0.35 * ce_exit(out_va["exit"], y_exit_va_t)
            )
            p_win_va = torch.sigmoid(out_va["win"]).detach().cpu().numpy()
            p_long_va = torch.sigmoid(out_va["long"]).detach().cpu().numpy()
            p_short_va = torch.sigmoid(out_va["short"]).detach().cpu().numpy()
            hold_va_pred_norm = out_va["hold"].detach().cpu().numpy()
            exit_logits_va = out_va["exit"].detach().cpu().numpy()

        side_is_long = (side_arr[split:] == "LONG")
        p_side_va = np.where(side_is_long, p_long_va, p_short_va)
        y_side_va = np.where(side_is_long, y_long_va, y_short_va).astype(np.int64)

        hold_va_pred_sec = np.expm1(hold_va_pred_norm * hold_std + hold_mean)
        hold_va_true_sec = np.expm1(y_hold_va * hold_std + hold_mean)
        hold_mae = float(np.mean(np.abs(hold_va_pred_sec - hold_va_true_sec)))
        exit_pred_cls = np.argmax(exit_logits_va, axis=1)
        exit_acc = float(np.mean(exit_pred_cls == y_exit_va.astype(np.int64)))

        va_loss_curve.append(float(loss_va.item()))
        va_win_auc_curve.append(binary_auc(y_win_va.astype(np.int64), p_win_va.astype(np.float64)))
        va_side_auc_curve.append(binary_auc(y_side_va.astype(np.int64), p_side_va.astype(np.float64)))
        va_hold_mae_curve.append(hold_mae)
        va_exit_acc_curve.append(exit_acc)

        print(
            f"[TRAIN] epoch={ep:02d}/{n_epochs} "
            f"loss={tr_loss:.5f} val_loss={va_loss_curve[-1]:.5f} "
            f"val_win_auc={va_win_auc_curve[-1]:.4f} val_side_auc={va_side_auc_curve[-1]:.4f} "
            f"val_hold_mae={hold_mae:.1f}s val_exit_acc={exit_acc:.3f}"
        )

    model.eval()
    with torch.no_grad():
        X_all_t = torch.from_numpy(X[:, None, :, :]).to(device)
        rg_all_t = torch.from_numpy(regime_id_arr).to(device)
        gp_all_t = torch.from_numpy(group_id_arr).to(device)
        out_all = model(X_all_t, rg_all_t, gp_all_t)
        p_win = torch.sigmoid(out_all["win"]).detach().cpu().numpy().astype(np.float64)
        p_long = torch.sigmoid(out_all["long"]).detach().cpu().numpy().astype(np.float64)
        p_short = torch.sigmoid(out_all["short"]).detach().cpu().numpy().astype(np.float64)
        hold_pred_norm = out_all["hold"].detach().cpu().numpy().astype(np.float64)
        exit_logits = out_all["exit"].detach().cpu().numpy().astype(np.float64)

    side_is_long_all = (side_arr == "LONG")
    p_side = np.where(side_is_long_all, p_long, p_short)
    hold_pred_sec = np.expm1(hold_pred_norm * hold_std + hold_mean)
    exit_prob = torch.softmax(torch.from_numpy(exit_logits), dim=1).numpy().astype(np.float64)
    exit_top_idx = np.argmax(exit_prob, axis=1)
    id_to_exit = {v: k for k, v in exit_to_id.items()}

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "mtf_multitask_v1",
            "state_dict": model.state_dict(),
            "timeframes": [int(x) for x in tf_list],
            "lookback_bars": int(lookback),
            "height": int(h),
            "width": int(w),
            "regime_to_id": regime_to_id,
            "group_to_id": group_to_id,
            "exit_to_id": exit_to_id,
            "hold_norm_mean": float(hold_mean),
            "hold_norm_std": float(hold_std),
            "train_samples": int(len(X_tr)),
            "val_samples": int(len(X_va)),
            "val_win_auc_last": float(va_win_auc_curve[-1] if va_win_auc_curve else 0.0),
            "val_side_auc_last": float(va_side_auc_curve[-1] if va_side_auc_curve else 0.0),
            "val_hold_mae_last": float(va_hold_mae_curve[-1] if va_hold_mae_curve else 0.0),
            "val_exit_acc_last": float(va_exit_acc_curve[-1] if va_exit_acc_curve else 0.0),
        },
        args.model_out,
    )

    score_rows: list[dict[str, Any]] = []
    for i in range(len(p_win)):
        exit_top = str(id_to_exit.get(int(exit_top_idx[i]), "other"))
        exit_conf = float(exit_prob[i, int(exit_top_idx[i])]) if exit_prob.ndim == 2 else 0.0
        score_rows.append(
            {
                "trade_uid": trade_ids[i],
                "symbol": symbols_used[i],
                "entry_ts_ms": int(ts_arr[i]),
                "side": str(side_arr[i]),
                "regime": str(regime_arr[i]),
                "symbol_group": str(group_arr[i]),
                "mtf_dl_prob": float(p_side[i]),
                "mtf_dl_prob_win": float(p_win[i]),
                "mtf_dl_prob_long": float(p_long[i]),
                "mtf_dl_prob_short": float(p_short[i]),
                "mtf_dl_hold_sec_pred": float(max(0.0, hold_pred_sec[i])),
                "mtf_dl_exit_reason_top": exit_top,
                "mtf_dl_exit_reason_conf": float(max(0.0, min(1.0, exit_conf))),
                "pnl": float(pnl_arr[i]),
            }
        )
    Path(args.scores_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores_out).write_text(json.dumps(score_rows, ensure_ascii=False, indent=2))

    # validation-only CF to reduce in-sample bias
    prob_eval = p_side[split:]
    pnl_eval = pnl_arr[split:]
    regimes_eval = regime_arr[split:]

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

    side_eval_true = np.where(side_arr[split:] == "LONG", y_long_arr[split:], y_short_arr[split:]).astype(np.int64)

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
            "regimes_used": sorted(set([str(x) for x in regime_arr.tolist()])),
            "symbol_groups_used": sorted(set([str(x) for x in group_arr.tolist()])),
        },
        "training": {
            "device": str(device),
            "train_loss_last": float(tr_loss_curve[-1] if tr_loss_curve else 0.0),
            "val_loss_last": float(va_loss_curve[-1] if va_loss_curve else 0.0),
            "val_win_auc_last": float(va_win_auc_curve[-1] if va_win_auc_curve else 0.0),
            "val_side_auc_last": float(va_side_auc_curve[-1] if va_side_auc_curve else 0.0),
            "val_hold_mae_last": float(va_hold_mae_curve[-1] if va_hold_mae_curve else 0.0),
            "val_exit_acc_last": float(va_exit_acc_curve[-1] if va_exit_acc_curve else 0.0),
            "train_loss_curve": [float(x) for x in tr_loss_curve],
            "val_loss_curve": [float(x) for x in va_loss_curve],
            "val_win_auc_curve": [float(x) for x in va_win_auc_curve],
            "val_side_auc_curve": [float(x) for x in va_side_auc_curve],
            "val_hold_mae_curve": [float(x) for x in va_hold_mae_curve],
            "val_exit_acc_curve": [float(x) for x in va_exit_acc_curve],
            "eval_side_auc": float(binary_auc(side_eval_true, prob_eval.astype(np.float64))) if len(prob_eval) > 0 else 0.5,
            "eval_win_auc": float(binary_auc(y_win_arr[split:].astype(np.int64), p_win[split:].astype(np.float64))) if len(p_win[split:]) > 0 else 0.5,
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
        f"samples={len(X)} val_side_auc={report['training']['val_side_auc_last']:.4f} "
        f"val_win_auc={report['training']['val_win_auc_last']:.4f} "
        f"baseline_pnl={baseline_all['pnl']:.2f} "
        f"best_global={None if best_global is None else best_global.get('pnl')} "
        f"best_chop_only={None if best_chop_only is None else best_chop_only.get('pnl')} "
        f"best_global_q={None if best_global_quantile is None else best_global_quantile.get('pnl')} "
        f"best_chop_q={None if best_chop_only_quantile is None else best_chop_only_quantile.get('pnl')}"
    )
    print(f"[DONE] report={args.out}")


if __name__ == "__main__":
    main()

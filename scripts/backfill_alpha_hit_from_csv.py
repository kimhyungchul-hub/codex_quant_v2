#!/usr/bin/env python3
"""
Backfill AlphaHit replay buffer from OHLCV CSVs.

Usage:
  python scripts/backfill_alpha_hit_from_csv.py --data-dir ./data --symbols BTCUSDT,ETHUSDT
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as base_config
from engines.mc.config import config as mc_config
from engines.mc.constants import SECONDS_PER_YEAR
from engines.mc.signal_features import MonteCarloSignalFeaturesMixin as Sig
from regime import adjust_mu_sigma
from trainers.online_alpha_trainer import AlphaTrainerConfig, OnlineAlphaTrainer


def _parse_symbols(raw: str | None) -> List[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def _infer_bar_seconds(ts: pd.Series) -> float:
    if ts.size < 2:
        return 60.0
    diffs = ts.diff().dropna().dt.total_seconds().values
    if diffs.size == 0:
        return 60.0
    return float(np.median(diffs))


def _momentum_z(closes: np.ndarray) -> float:
    if closes.size < 6:
        return 0.0
    rets = np.diff(np.log(closes))
    if rets.size < 5:
        return 0.0
    window = int(min(20, rets.size))
    subset = rets[-window:]
    mean_r = float(subset.mean())
    std_r = float(subset.std())
    if std_r <= 1e-9:
        return 0.0
    return float((subset[-1] - mean_r) / std_r)


def _tp_sl_targets(h: float, sigma: float) -> Tuple[float, float]:
    if not bool(getattr(mc_config, "tpsl_autoscale", True)):
        tp_base = float(getattr(mc_config, "tp_base_roe", 0.0015))
        sl_base = float(getattr(mc_config, "sl_base_roe", 0.0020))
        return tp_base, sl_base

    h = float(max(1.0, h))
    sigma_ref = float(max(1e-3, getattr(mc_config, "tpsl_sigma_ref", 0.5)))
    sigma_val = float(sigma) if math.isfinite(float(sigma)) and float(sigma) > 0 else sigma_ref
    if sigma_val <= 0:
        sigma_val = sigma_ref
    vol_scale = sigma_val / sigma_ref
    vol_scale = float(
        max(
            float(getattr(mc_config, "tpsl_sigma_min_scale", 0.6)),
            min(float(getattr(mc_config, "tpsl_sigma_max_scale", 2.5)), vol_scale),
        )
    )
    horizon_scale = math.sqrt(h / max(1.0, float(getattr(mc_config, "tpsl_h_scale_base", 60.0))))
    tp_r = float(getattr(mc_config, "tp_base_roe", 0.0015)) * horizon_scale * vol_scale
    sl_r = float(getattr(mc_config, "sl_base_roe", 0.0020)) * horizon_scale * vol_scale
    return float(tp_r), float(sl_r)


def _hit_type_long(entry: float, highs: Iterable[float], lows: Iterable[float], closes: Iterable[float], tp_r: float, sl_r: float) -> str | None:
    tp_px = entry * (1.0 + tp_r)
    sl_px = entry * (1.0 - sl_r)
    for h, l, c in zip(highs, lows, closes):
        hit_tp = float(h) >= tp_px
        hit_sl = float(l) <= sl_px
        if hit_tp and hit_sl:
            return "TP" if float(c) >= entry else "SL"
        if hit_tp:
            return "TP"
        if hit_sl:
            return "SL"
    return None


def _hit_type_short(entry: float, highs: Iterable[float], lows: Iterable[float], closes: Iterable[float], tp_r: float, sl_r: float) -> str | None:
    tp_px = entry * (1.0 - tp_r)
    sl_px = entry * (1.0 + sl_r)
    for h, l, c in zip(highs, lows, closes):
        hit_tp = float(l) <= tp_px
        hit_sl = float(h) >= sl_px
        if hit_tp and hit_sl:
            return "TP" if float(c) <= entry else "SL"
        if hit_tp:
            return "TP"
        if hit_sl:
            return "SL"
    return None


def _build_features(
    *,
    mu_adj: float,
    sigma_adj: float,
    momentum_z: float,
    ofi_z: float,
    regime: str,
    leverage: float,
    price: float,
    mu_alpha: float,
    n_features: int,
) -> np.ndarray:
    features = [
        float(mu_adj) * SECONDS_PER_YEAR,
        float(sigma_adj) * math.sqrt(SECONDS_PER_YEAR),
        float(momentum_z),
        float(ofi_z),
        float(leverage),
        float(price),
        1.0 if regime == "bull" else 0.0,
        1.0 if regime == "bear" else 0.0,
        1.0 if regime == "chop" else 0.0,
        1.0 if regime == "volatile" else 0.0,
        0.0,  # spread_pct
        0.0,  # kelly
        0.0,  # confidence
        0.0,  # ev
        float(mu_alpha),
        0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    features = features[:n_features] + [0.0] * max(0, n_features - len(features))
    return np.asarray(features, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill AlphaHit replay buffer from OHLCV CSVs.")
    ap.add_argument("--data-dir", type=str, default="./data", help="CSV directory")
    ap.add_argument("--symbols", type=str, default="", help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)")
    ap.add_argument("--max-samples", type=int, default=0, help="Max samples per symbol (0 = all)")
    ap.add_argument("--stride", type=int, default=1, help="Bar stride (>=1)")
    ap.add_argument("--train-steps", type=int, default=200, help="Train ticks after buffer fill")
    ap.add_argument("--device", type=str, default=str(getattr(mc_config, "alpha_hit_device", "cpu")), help="Trainer device")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        symbols = [p.stem for p in data_dir.glob("*.csv")]

    horizons = []
    raw = str(os.environ.get("POLICY_MULTI_HORIZONS_SEC", "60,300,600,1800,3600"))
    horizons = [int(x) for x in raw.split(",") if x.strip().isdigit()]
    if not horizons:
        horizons = [60, 300, 600, 1800, 3600]

    device = str(args.device)
    try:
        import torch
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    cfg = AlphaTrainerConfig(
        horizons_sec=horizons,
        n_features=20,
        device=device,
        lr=float(getattr(mc_config, "alpha_hit_lr", 2e-4)),
        batch_size=int(getattr(mc_config, "alpha_hit_batch_size", 256)),
        steps_per_tick=int(getattr(mc_config, "alpha_hit_steps_per_tick", 2)),
        max_buffer=int(getattr(mc_config, "alpha_hit_max_buffer", 200000)),
        min_buffer=int(getattr(mc_config, "alpha_hit_min_buffer", 1024)),
        warmup_samples=int(getattr(mc_config, "alpha_hit_warmup_samples", 512)),
        data_half_life_sec=float(getattr(mc_config, "alpha_hit_data_half_life_sec", 3600.0)),
        ckpt_path=str(getattr(mc_config, "alpha_hit_model_path", "state/alpha_hit_mlp.pt")),
        replay_path=str(getattr(mc_config, "alpha_hit_replay_path", "state/alpha_hit_replay.npz")),
        replay_save_every=int(getattr(mc_config, "alpha_hit_replay_save_every", 2000)),
    )
    trainer = OnlineAlphaTrainer(cfg)

    total_added = 0
    stride = max(1, int(args.stride))
    for sym in symbols:
        csv_path = data_dir / f"{sym}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        bar_seconds = _infer_bar_seconds(df["timestamp"])
        closes = df["close"].astype(float).values
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values

        max_h = int(max(horizons))
        max_h_bars = max(1, int(math.ceil(max_h / bar_seconds)))
        min_lookback = 200
        start = min_lookback
        end = len(df) - max_h_bars - 1
        if end <= start:
            continue

        n_added = 0
        for i in range(start, end, stride):
            if args.max_samples > 0 and n_added >= int(args.max_samples):
                break
            entry_price = float(closes[i])
            if entry_price <= 0:
                continue
            window = closes[max(0, i - 400) : i + 1]
            if window.size < 30:
                continue

            rets = np.diff(np.log(window))
            if rets.size < 10:
                continue
            sigma_bar = float(np.std(rets[-120:])) if rets.size >= 20 else float(np.std(rets))
            sigma_annual = sigma_bar * math.sqrt(SECONDS_PER_YEAR / float(bar_seconds))

            regime = Sig._cluster_regime(window.tolist())
            mu_alpha_parts = Sig._signal_alpha_mu_annual_parts(window.tolist(), bar_seconds, 0.0, regime)
            mu_alpha = float(mu_alpha_parts.get("mu_alpha") or 0.0)
            mu_adj, sigma_adj = adjust_mu_sigma(mu_alpha, sigma_annual, {"regime": regime})

            momentum_z = _momentum_z(window)
            leverage = float(getattr(base_config, "DEFAULT_LEVERAGE", 5.0))
            features = _build_features(
                mu_adj=mu_adj,
                sigma_adj=sigma_adj,
                momentum_z=momentum_z,
                ofi_z=0.0,
                regime=regime,
                leverage=leverage,
                price=entry_price,
                mu_alpha=mu_alpha,
                n_features=cfg.n_features,
            )

            y_tp_long = np.zeros(len(horizons), dtype=np.float32)
            y_sl_long = np.zeros(len(horizons), dtype=np.float32)
            y_tp_short = np.zeros(len(horizons), dtype=np.float32)
            y_sl_short = np.zeros(len(horizons), dtype=np.float32)

            for j, h in enumerate(horizons):
                h_bars = max(1, int(math.ceil(float(h) / bar_seconds)))
                hi = highs[i + 1 : i + 1 + h_bars]
                lo = lows[i + 1 : i + 1 + h_bars]
                cl = closes[i + 1 : i + 1 + h_bars]
                tp_r, sl_r = _tp_sl_targets(float(h), float(sigma_adj))
                hit_long = _hit_type_long(entry_price, hi, lo, cl, tp_r, sl_r)
                hit_short = _hit_type_short(entry_price, hi, lo, cl, tp_r, sl_r)
                if hit_long == "TP":
                    y_tp_long[j] = 1.0
                elif hit_long == "SL":
                    y_sl_long[j] = 1.0
                if hit_short == "TP":
                    y_tp_short[j] = 1.0
                elif hit_short == "SL":
                    y_sl_short[j] = 1.0

            ts_ms = int(df["timestamp"].iloc[i].value // 1_000_000)
            trainer.add_sample(
                x=features,
                y={
                    "tp_long": y_tp_long,
                    "sl_long": y_sl_long,
                    "tp_short": y_tp_short,
                    "sl_short": y_sl_short,
                },
                ts_ms=ts_ms,
                symbol=str(sym),
            )
            n_added += 1
            total_added += 1

        print(f"[BACKFILL] {sym}: added={n_added}")

    for _ in range(max(0, int(args.train_steps))):
        trainer.train_tick()

    trainer._save_checkpoint()
    trainer._save_replay()
    print(f"[BACKFILL] total_added={total_added} buffer={trainer.buffer_size}")


if __name__ == "__main__":
    main()

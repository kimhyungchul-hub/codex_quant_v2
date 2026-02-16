#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import csv
from typing import List, Tuple

import numpy as np


def _load_ohlcv_from_csv(path: str) -> dict:
    cols = {"open": None, "high": None, "low": None, "close": None, "volume": None}
    data = {k: [] for k in cols}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return {}
        for i, col in enumerate(header):
            key = str(col).strip().lower()
            if key in cols:
                cols[key] = i
        if cols["close"] is None:
            return {}
        for row in reader:
            if cols["close"] is not None and cols["close"] < len(row):
                try:
                    data["close"].append(float(row[cols["close"]]))
                except Exception:
                    continue
            for k in ("open", "high", "low", "volume"):
                idx = cols[k]
                if idx is not None and idx < len(row):
                    try:
                        data[k].append(float(row[idx]))
                    except Exception:
                        data[k].append(np.nan)
    out = {}
    for k, vals in data.items():
        if vals:
            out[k] = np.asarray(vals, dtype=np.float32)
    return out


def _make_dataset(seq_len: int, paths: List[str], max_samples: int = 200000) -> Tuple[np.ndarray, np.ndarray, int]:
    series = []
    for p in paths:
        ohlcv = _load_ohlcv_from_csv(p)
        closes = ohlcv.get("close")
        if closes is not None and closes.size >= seq_len + 2:
            series.append(ohlcv)

    if not series:
        # synthetic lognormal walk
        rng = np.random.default_rng(7)
        rets = rng.normal(0.0001, 0.003, size=8192)
        logp = np.cumsum(rets)
        closes = 100.0 * np.exp(logp).astype(np.float32)
        series = [{"close": closes}]

    X = []
    y = []
    n_features = 1
    for ohlcv in series:
        closes = ohlcv.get("close")
        if closes is None:
            continue
        rets = np.diff(np.log(np.maximum(closes, 1e-12))).astype(np.float32)
        if rets.size < seq_len + 1:
            continue
        vol_rets = None
        volumes = ohlcv.get("volume")
        if volumes is not None and volumes.size == closes.size:
            v = np.where(np.isfinite(volumes), volumes, 0.0)
            v = np.maximum(v, 1e-12)
            vol_rets = np.diff(np.log(v)).astype(np.float32)
        abs_rets = np.abs(rets).astype(np.float32)
        if vol_rets is None:
            vol_rets = np.zeros_like(rets)
        feats = np.stack([rets, vol_rets, abs_rets], axis=1)
        n_features = feats.shape[1]
        for i in range(seq_len, rets.size):
            seq = feats[i - seq_len : i].astype(np.float32)
            mean = seq.mean(axis=0, keepdims=True)
            std = seq.std(axis=0, keepdims=True) + 1e-6
            seq = (seq - mean) / std
            X.append(seq)
            y.append(rets[i])
            if max_samples and len(X) >= max_samples:
                break
        if max_samples and len(X) >= max_samples:
            break
    if not X:
        return np.zeros((0, seq_len, 1), dtype=np.float32), np.zeros((0,), dtype=np.float32), 1
    return np.stack(X, axis=0), np.asarray(y, dtype=np.float32), int(n_features)


def build_mu_alpha_model(
    out_path: str,
    seq_len: int = 64,
    hidden: int = 32,
    layers: int = 2,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    dropout: float = 0.1,
    max_samples: int = 200000,
) -> bool:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        return False

    csv_paths = glob.glob(os.path.join("data", "*.csv"))
    X, y, n_features = _make_dataset(seq_len, csv_paths, max_samples=max_samples)
    if X.shape[0] < 64:
        return False

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"

    class MuAlphaNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_features = int(n_features)
            self.seq_len = int(seq_len)
            self.conv = nn.Conv1d(self.n_features, hidden, kernel_size=3, padding=1)
            self.act = nn.GELU()
            self.gru = nn.GRU(
                hidden,
                hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            mid = max(4, hidden // 2)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden, mid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid, 1),
            )
        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.act(self.conv(x))
            x = x.transpose(1, 2)
            out, _ = self.gru(x)
            h = self.norm(out[:, -1, :])
            return self.head(h)

    model = MuAlphaNet().to(device)
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(-1))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    try:
        scripted = torch.jit.script(model)
        scripted.save(out_path)
    except Exception:
        torch.save(model, out_path)
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.environ.get("ML_MODEL_PATH", "state/mu_alpha_model.pt"))
    p.add_argument("--seq-len", type=int, default=int(os.environ.get("ML_SEQ_LEN", 64)))
    p.add_argument("--hidden", type=int, default=int(os.environ.get("ML_HIDDEN", 32)))
    p.add_argument("--layers", type=int, default=int(os.environ.get("ML_LAYERS", 2)))
    p.add_argument("--epochs", type=int, default=int(os.environ.get("ML_EPOCHS", 3)))
    p.add_argument("--batch", type=int, default=int(os.environ.get("ML_BATCH", 128)))
    p.add_argument("--lr", type=float, default=float(os.environ.get("ML_LR", 1e-3)))
    p.add_argument("--dropout", type=float, default=float(os.environ.get("ML_DROPOUT", 0.1)))
    p.add_argument("--max-samples", type=int, default=int(os.environ.get("ML_MAX_SAMPLES", 200000)))
    args = p.parse_args()
    ok = build_mu_alpha_model(
        args.out,
        args.seq_len,
        args.hidden,
        args.layers,
        args.epochs,
        args.batch,
        args.lr,
        args.dropout,
        args.max_samples,
    )
    print(f"[mu_alpha_model] build={ok} out={args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

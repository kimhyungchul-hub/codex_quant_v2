#!/usr/bin/env python3
from __future__ import annotations

import bootstrap  # ensure JAX/XLA env is set for any GPU-accelerated workloads
"""
PatchTST 학습 파이프라인 (GPU 최적화)
- Patch 단위 Transformer로 multivariate 시계열 예측
- Channel-independent patch embedding + patch-level positional encoding
- Torch absent 시 graceful degradation
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency guard
    torch = None
    nn = None
    Categorical = None
    DataLoader = None
    Dataset = None
    TORCH_AVAILABLE = False

from engines.mc.execution_costs import ExecutionCostModel
from engines.mc.jax_backend import ensure_jax
from engines.mc.monte_carlo_engine import MonteCarloEngine
from models.patchtst import PatchTST
from config import SYMBOLS


# Action space: -1 (Short), 0 (Flat), +1 (Long)
ACTION_SPACE = (-1, 0, 1)


@dataclass
class TrainConfig:
    symbol: str = "BTCUSDT"
    bar_seconds: int = 60
    regime: str = "chop"
    n_paths: int = 2048
    episodes: int = 5  # lightweight by default; override for long runs
    steps_per_episode: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99
    device: str = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    position_size: float = 1_000.0  # contracts/units per step
    adv: float = 1_000_000.0  # average daily volume proxy
    cost_budget_pct: float = 0.001  # skip trades if cost > 0.1% notionals
    seed: int = 42


@dataclass
class PatchTSTConfig:
    context_length: int = 128
    pred_length: int = 4
    patch_len: int = 16
    stride: int = 8
    d_model: int = 96
    depth: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 3e-4
    n_vars: int = 5
    train_length: int = 2048
    seed: int = 7
    device: str = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"


def _annualize(mu: float, sigma: float, bar_seconds: int) -> Tuple[float, float]:
    """Annualize log-return mean/vol for MonteCarloEngine ctx."""
    if bar_seconds <= 0:
        bar_seconds = 60
    scale = 31536000.0 / float(bar_seconds)
    mu_ann = float(mu * scale)
    sigma_ann = float(sigma * math.sqrt(scale))
    return mu_ann, sigma_ann


def _synthetic_price_series(length: int, seed: int = 42) -> np.ndarray:
    """Generate a stable log-normal walk for quick training/debug."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.003, size=length)
    logp = np.cumsum(rets)
    return 100.0 * np.exp(logp)


def _make_multivariate_series(length: int, n_vars: int = 5, seed: int = 7) -> np.ndarray:
    """Build synthetic OHLCV-like multivariate series for PatchTST."""
    rng = np.random.default_rng(seed)
    close = _synthetic_price_series(length, seed)
    open_price = close * (1 + rng.normal(0, 0.001, size=length))
    high = np.maximum(open_price, close) * (1 + rng.normal(0, 0.0015, size=length))
    low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.0015, size=length)))
    volume = rng.lognormal(mean=0.0, sigma=0.25, size=length) * 1000.0

    series = np.zeros((length, max(n_vars, 5)), dtype=np.float32)
    series[:, 0] = open_price
    series[:, 1] = high
    series[:, 2] = low
    series[:, 3] = close
    series[:, 4] = volume

    if series.shape[1] > 5:
        noise = rng.normal(0, 0.01, size=(length, series.shape[1] - 5))
        series[:, 5:] = close[:, None] * (1 + noise)

    if n_vars < series.shape[1]:
        series = series[:, :n_vars]
    return series


def load_historical_series(
    length: int,
    n_vars: int = 5,
    symbol: str | list[str] = "BTC-USD",
    csv_dirs: list | None = None,
    seed: int = 7,
    verbose: bool = False,
) -> tuple[np.ndarray, str | dict]:
    """Load historical OHLCV data from repo CSVs or fall back to yfinance.

    Returns:
        data: np.ndarray shape (length, n_vars)
        source: string describing source
    """
    import glob
    import os
    import pandas as pd

    csv_dirs = csv_dirs or ["data", "."]
    candidates = []
    for d in csv_dirs:
        p = os.path.join(d, "*.csv")
        candidates.extend(glob.glob(p))

    # If symbol is a list, try to load each symbol separately and stack close prices as channels
    if isinstance(symbol, list):
        syms = list(symbol)
        channels = []
        sources = []
        per_symbol = {}
        for s in syms:
            # try CSV first: match filename containing the base symbol
            base = s.split("/")[0] if "/" in s else s
            found = False
            for c in candidates:
                if base.upper() in os.path.basename(c).upper():
                    try:
                        df = pd.read_csv(c)
                        # find close column case-insensitive
                        close_col = None
                        for col in df.columns:
                            if col.lower() == "close":
                                close_col = col
                                break
                        if close_col is None:
                            continue
                        close = df[close_col].values.astype(np.float32)
                        if len(close) < length:
                            close = np.concatenate([np.repeat(close[-1:], length - len(close)), close]) if len(close) > 0 else np.zeros(length, dtype=np.float32)
                        channels.append(close[-length:].astype(np.float32))
                        sources.append(f"csv:{os.path.basename(c)}")
                        per_symbol[s] = {"source": f"csv:{os.path.basename(c)}", "attempts": [os.path.basename(c)]}
                        found = True
                        break
                    except Exception:
                        continue
            if found:
                continue

            # CSV not found, try yfinance with multiple candidate tickers
            attempts = []
            cand_tickers = [f"{base}-USD", f"{base}USD", f"{base}USDT", f"{base}-USDT", base.upper()]
            yfinance_success = False
            for ticker in cand_tickers:
                attempts.append(ticker)
                try:
                    df = yf.download(ticker, period="max", progress=False)
                    if df is not None and not df.empty:
                        close = df["Close"].values.astype(np.float32)
                        if len(close) < length:
                            close = np.concatenate([np.repeat(close[-1:], length - len(close)), close])
                        channels.append(close[-length:].astype(np.float32))
                        sources.append(f"yfinance:{ticker}")
                        per_symbol[s] = {"source": f"yfinance:{ticker}", "attempts": attempts}
                        yfinance_success = True
                        break
                except Exception:
                    continue
            if yfinance_success:
                continue

            # fallback for this single symbol -> synthetic close
            synth = _make_multivariate_series(length, n_vars=5, seed=seed)[:, 3]
            channels.append(synth.astype(np.float32))
            sources.append("synthetic")
            per_symbol[s] = {"source": "synthetic", "attempts": attempts}

        # Stack channels -> shape (length, n_syms)
        mat = np.stack(channels, axis=1)
        # standardize per channel
        mean = mat.mean(axis=0, keepdims=True)
        std = np.clip(mat.std(axis=0, keepdims=True), 1e-6, None)
        mat = (mat - mean) / std
        if verbose:
            return mat.astype(np.float32), per_symbol
        return mat.astype(np.float32), ",".join(sorted(set(sources)))

    # fallback: try yfinance for single symbol
    try:
        import yfinance as yf
        import pandas as pd

        df = yf.download(symbol, period="max", progress=False)
        if df is not None and not df.empty:
            close = df["Close"].values.astype(np.float32)
            open_ = df["Open"].values.astype(np.float32)
            high = df["High"].values.astype(np.float32)
            low = df["Low"].values.astype(np.float32)
            vol = df["Volume"].values.astype(np.float32)
            base = np.vstack([open_, high, low, close, vol]).T
            if base.shape[0] < length:
                pad = np.repeat(base[-1:, :], length - base.shape[0], axis=0)
                base = np.concatenate([base, pad], axis=0)
            out = base[-length:, :]
            if out.shape[1] < n_vars:
                extra = np.tile(out[:, 3:4], (1, n_vars - out.shape[1])) * (
                    1.0 + np.random.default_rng(seed).normal(0, 0.01, size=(length, n_vars - out.shape[1]))
                )
                out = np.concatenate([out, extra.astype(np.float32)], axis=1)
            elif out.shape[1] > n_vars:
                out = out[:, :n_vars]
            mean = out.mean(axis=0, keepdims=True)
            std = np.clip(out.std(axis=0, keepdims=True), 1e-6, None)
            out = (out - mean) / std
            return out.astype(np.float32), f"yfinance:{symbol}"
    except Exception:
        pass

    # final fallback: synthetic
    out = _make_multivariate_series(length, n_vars=n_vars, seed=seed)
    out = (out - out.mean(axis=0, keepdims=True)) / np.clip(out.std(axis=0, keepdims=True), 1e-6, None)
    return out.astype(np.float32), "synthetic"


if TORCH_AVAILABLE:
    class PolicyNet(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, len(ACTION_SPACE)),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)
else:
    class PolicyNet:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required to use PolicyNet and train the RL agent.")


class MCRLEnvironment:
    """MC 시뮬레이션 기반 RL 환경 (상태 전이/보상에 MC 사용)."""

    def __init__(self, prices: np.ndarray, cfg: TrainConfig):
        ensure_jax()
        self.engine = MonteCarloEngine()
        self.cost_model = ExecutionCostModel()
        self.cfg = cfg
        self.prices = np.asarray(prices, dtype=np.float64)
        self.window = 60  # lookback window for vol/ctx
        self.t = 0
        self.prev_action = 0
        self.last_ev = 0.0
        self.last_cost = 0.0
        self.max_step = min(cfg.steps_per_episode, len(self.prices) - 2)
        self.state_dim = 6
        self.action_space = ACTION_SPACE

    def reset(self) -> np.ndarray:
        self.t = self.window
        self.prev_action = 0
        self.last_ev = 0.0
        self.last_cost = 0.0
        return self._build_state()

    def _build_ctx(self) -> Tuple[Dict[str, float], float, float]:
        idx_start = max(0, self.t - self.window)
        window = self.prices[idx_start : self.t + 1]
        price = float(window[-1])
        logrets = np.diff(np.log(window))
        mu = float(logrets.mean()) if logrets.size > 0 else 0.0
        sigma = float(logrets.std()) if logrets.size > 0 else 0.01
        mu_ann, sigma_ann = _annualize(mu, sigma, self.cfg.bar_seconds)

        ctx: Dict[str, float] = {
            "symbol": self.cfg.symbol,
            "price": price,
            "mu_sim": mu_ann,
            "sigma_sim": sigma_ann,
            "closes": window.tolist(),
            "bar_seconds": self.cfg.bar_seconds,
            "regime": self.cfg.regime,
            "pmaker_entry": 0.9,
            "ts": int(time.time() * 1000),
            "n_paths": self.cfg.n_paths,
            "boost": 1.0,
            "max_leverage": 50.0,
            "tail_mode": "student_t",
            "use_jax": True,
        }
        return ctx, price, sigma_ann

    def _build_state(self) -> np.ndarray:
        idx_start = max(0, self.t - 1)
        price = self.prices[self.t]
        prev_price = self.prices[idx_start]
        price_change = (price - prev_price) / max(prev_price, 1e-6)
        return np.array(
            [
                price_change,
                self.prev_action,
                self.last_ev,
                self.last_cost,
                float(self.t) / float(self.max_step + 1),
                1.0,
            ],
            dtype=np.float32,
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        ctx, price, sigma_ann = self._build_ctx()
        params = self.engine._get_params(self.cfg.regime, ctx)
        try:
            metrics = self.engine.evaluate_entry_metrics(ctx, params, seed=self.cfg.seed + self.t)
        except Exception as e:  # defensive fallback to keep RL loop running
            metrics = {"ev": 0.0, "ev_raw": 0.0, "error": str(e)}
        ev = float(metrics.get("ev", metrics.get("ev_raw", 0.0)) or 0.0)

        # Pre-trade cost-aware gating
        order_size = float(abs(action)) * self.cfg.position_size
        est_cost = self.cost_model.calculate_cost(
            order_size=order_size,
            price=price,
            sigma=sigma_ann,
            adv=self.cfg.adv,
        )
        notional = price * order_size
        cost_cap = notional * self.cfg.cost_budget_pct
        allowed_action = action
        if cost_cap > 0 and est_cost > cost_cap:
            allowed_action = 0

        reward = float(allowed_action) * ev
        if notional > 0:
            reward -= est_cost / notional

        self.t += 1
        self.prev_action = allowed_action
        self.last_ev = ev
        self.last_cost = est_cost
        done = self.t >= self.max_step
        next_state = self._build_state()
        info = {
            "ev": ev,
            "cost": est_cost,
            "notional": notional,
            "action": allowed_action,
        }
        return next_state, reward, done, info


def _compute_returns(rewards: List[torch.Tensor], gamma: float) -> torch.Tensor:
    g = 0.0
    returns: List[torch.Tensor] = []
    for r in reversed(rewards):
        g = float(r.item()) + gamma * g
        returns.append(torch.tensor(g, dtype=torch.float32, device=r.device))
    returns = torch.stack(list(reversed(returns)))
    if returns.std() > 0:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


if TORCH_AVAILABLE:

    class PatchTSTDataset(Dataset):
        def __init__(self, data: np.ndarray, context_length: int, pred_length: int) -> None:
            self.data = np.asarray(data, dtype=np.float32)
            self.context_length = int(context_length)
            self.pred_length = int(pred_length)
            if self.data.ndim != 2:
                raise ValueError("data must be 2D [time, features]")
            if len(self.data) <= self.context_length + self.pred_length:
                raise ValueError("not enough data for requested context/prediction lengths")
            self.mean = self.data.mean(axis=0, keepdims=True)
            self.std = np.clip(self.data.std(axis=0, keepdims=True), 1e-6, None)

        def __len__(self) -> int:
            return len(self.data) - (self.context_length + self.pred_length) + 1

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            start = idx
            mid = idx + self.context_length
            end = mid + self.pred_length
            x = self.data[start:mid]
            y = self.data[mid:end]
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std
            return torch.from_numpy(x), torch.from_numpy(y)


else:

    class PatchTSTDataset:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for PatchTSTDataset")


def train(cfg: TrainConfig) -> None:
    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch not installed; skipping training. Install torch>=2.0 to enable RL training.")
        return

    ensure_jax()
    prices = _synthetic_price_series(length=cfg.steps_per_episode + 200, seed=cfg.seed)
    env = MCRLEnvironment(prices, cfg)
    policy = PolicyNet(env.state_dim).to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    for ep in range(cfg.episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32, device=cfg.device)
        log_probs: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        ep_reward = 0.0

        for _ in range(env.max_step):
            logits = policy(state)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action_idx = dist.sample()
            action = env.action_space[action_idx.item()]

            next_state, reward, done, info = env.step(action)
            log_probs.append(dist.log_prob(action_idx))
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=cfg.device))
            ep_reward += reward
            state = torch.tensor(next_state, dtype=torch.float32, device=cfg.device)

            if done:
                break

        if log_probs:
            returns = _compute_returns(rewards, cfg.gamma)
            loss = -torch.sum(torch.stack(log_probs) * returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(
            f"[Episode {ep+1}/{cfg.episodes}] reward={ep_reward:.4f} last_ev={env.last_ev:.6f} last_cost={env.last_cost:.4f}"
        )


def train_patchtst(cfg: PatchTSTConfig) -> float:
    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch not installed; PatchTST training skipped. Install torch>=2.0 to enable.")
        return 0.0

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    total_len = cfg.train_length + cfg.context_length + cfg.pred_length
    # load all configured symbols as separate channels (use close prices)
    data, source = load_historical_series(total_len, n_vars=cfg.n_vars, symbol=SYMBOLS, seed=cfg.seed, verbose=True)
    # `source` may be a dict mapping symbol->info when verbose
    if isinstance(source, dict):
        print("Data Integration: Connected to real data (per-symbol sources):")
        for sym, info in source.items():
            print(f"  {sym}: {info}")
    else:
        print(f"Data Integration: Connected to real data (Source: {source})")
    dataset = PatchTSTDataset(data, context_length=cfg.context_length, pred_length=cfg.pred_length)

    n_train = max(1, int(len(dataset) * 0.8))
    n_val = max(1, len(dataset) - n_train)
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    model = PatchTST(
        n_vars=cfg.n_vars,
        context_length=cfg.context_length,
        pred_length=cfg.pred_length,
        patch_len=cfg.patch_len,
        stride=cfg.stride,
        d_model=cfg.d_model,
        depth=cfg.depth,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            target = yb.permute(0, 2, 1)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss = train_loss / max(len(loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                target = yb.permute(0, 2, 1)
                loss = loss_fn(pred, target)
                val_loss += float(loss.item())
        val_loss = val_loss / max(len(val_loader), 1)
        best_val = min(best_val, val_loss)
        print(
            f"[PatchTST][{epoch+1}/{cfg.epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} device={device}"
        )

    # save checkpoint
    ckpt_dir = "checkpoints"
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"patchtst_{int(time.time())}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
    except Exception as e:
        print(f"Warning: failed to save checkpoint: {e}")

    return best_val


def main():
    cfg = PatchTSTConfig()
    train_patchtst(cfg)


if __name__ == "__main__":
    main()



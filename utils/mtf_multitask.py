from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn


ONE_MIN_MS = 60_000

REGIME_LABELS = [
    "bull",
    "bear",
    "chop",
    "volatile",
    "mean_revert",
    "unknown",
]

SYMBOL_GROUP_LABELS = [
    "major",
    "large_alt",
    "mid_alt",
    "meme",
    "new_listing",
    "other",
]

EXIT_REASON_LABELS = [
    "event_mc_exit",
    "unrealized_dd",
    "unified_flip",
    "hybrid_exit",
    "prelev_stop2",
    "unified_cash",
    "take_profit",
    "stop_or_liq",
    "other",
]


def normalize_regime(raw: Any) -> str:
    txt = str(raw or "").strip().lower()
    if not txt:
        return "unknown"
    if ("bull" in txt) or ("trend" in txt):
        return "bull"
    if "bear" in txt:
        return "bear"
    if ("volatile" in txt) or ("random" in txt) or ("noise" in txt):
        return "volatile"
    if ("mean" in txt) or ("revert" in txt):
        return "mean_revert"
    if "chop" in txt:
        return "chop"
    return "unknown"


def infer_symbol_group(symbol: Any) -> str:
    sym = str(symbol or "").upper()
    base = sym.split("/")[0].split(":")[0].replace("USDT", "").replace("USDC", "").strip()
    if base in {"BTC", "ETH", "SOL", "BNB"}:
        return "major"
    if base in {"XRP", "ADA", "DOGE", "LINK", "AVAX", "DOT", "LTC", "SUI"}:
        return "large_alt"
    if any(tok in base for tok in ("PEPE", "FART", "BONK", "SHIB", "FLOKI", "MEME")):
        return "meme"
    if base.startswith("1000") or len(base) >= 9:
        return "new_listing"
    if base:
        return "mid_alt"
    return "other"


def normalize_exit_reason(raw: Any) -> str:
    txt = str(raw or "").strip().lower()
    if not txt:
        return "other"
    if "event" in txt:
        return "event_mc_exit"
    if "unrealized_dd" in txt:
        return "unrealized_dd"
    if "unified_flip" in txt or "flip" in txt:
        return "unified_flip"
    if "hybrid_exit" in txt:
        return "hybrid_exit"
    if "prelev_stop2" in txt:
        return "prelev_stop2"
    if "unified_cash" in txt:
        return "unified_cash"
    if ("tp" in txt) or ("take_profit" in txt):
        return "take_profit"
    if ("sl" in txt) or ("liq" in txt) or ("stop" in txt):
        return "stop_or_liq"
    return "other"


def _to_float_array(v: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(v), dtype=np.float64)


def _resample_ohlcv(
    ts_ms: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    tf_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(tf_min) <= 1:
        return ts_ms, open_, high, low, close, volume
    n = int(ts_ms.size)
    if n <= 0:
        zf = np.zeros(0, dtype=np.float64)
        zi = np.zeros(0, dtype=np.int64)
        return zi, zf, zf, zf, zf, zf
    tf_ms = int(max(1, int(tf_min)) * ONE_MIN_MS)
    bucket = (ts_ms // tf_ms).astype(np.int64)
    out_ts: list[int] = []
    out_o: list[float] = []
    out_h: list[float] = []
    out_l: list[float] = []
    out_c: list[float] = []
    out_v: list[float] = []

    i = 0
    while i < n:
        b = int(bucket[i])
        j = i + 1
        o = float(open_[i])
        h = float(high[i])
        l = float(low[i])
        c = float(close[i])
        v = float(volume[i])
        while j < n and int(bucket[j]) == b:
            h = max(h, float(high[j]))
            l = min(l, float(low[j]))
            c = float(close[j])
            v += float(volume[j])
            j += 1
        out_ts.append(int((b + 1) * tf_ms))
        out_o.append(o)
        out_h.append(h)
        out_l.append(l)
        out_c.append(c)
        out_v.append(v)
        i = j

    return (
        np.asarray(out_ts, dtype=np.int64),
        np.asarray(out_o, dtype=np.float64),
        np.asarray(out_h, dtype=np.float64),
        np.asarray(out_l, dtype=np.float64),
        np.asarray(out_c, dtype=np.float64),
        np.asarray(out_v, dtype=np.float64),
    )


def build_mtf_image_from_1m(
    ts_ms: Iterable[Any],
    open_: Iterable[Any],
    high: Iterable[Any],
    low: Iterable[Any],
    close: Iterable[Any],
    volume: Iterable[Any],
    entry_ts_ms: int,
    tf_list: list[int],
    lookback: int,
) -> np.ndarray | None:
    t = np.asarray(list(ts_ms), dtype=np.int64)
    o = _to_float_array(open_)
    h = _to_float_array(high)
    l = _to_float_array(low)
    c = _to_float_array(close)
    v = _to_float_array(volume)
    if t.size == 0 or c.size == 0:
        return None
    n = int(min(t.size, o.size, h.size, l.size, c.size, v.size))
    t, o, h, l, c, v = t[:n], o[:n], h[:n], l[:n], c[:n], v[:n]
    order = np.argsort(t)
    t, o, h, l, c, v = t[order], o[order], h[order], l[order], c[order], v[order]

    mats: list[np.ndarray] = []
    min_required = max(8, int(lookback // 3))
    for tf in [max(1, int(x)) for x in tf_list]:
        tt, oo, hh, ll, cc, vv = _resample_ohlcv(t, o, h, l, c, v, int(tf))
        if tt.size == 0:
            return None
        idx = int(np.searchsorted(tt, int(entry_ts_ms), side="right"))
        if idx <= 0:
            return None
        st = max(0, idx - int(lookback))
        if (idx - st) < min_required:
            return None
        c_win = np.maximum(cc[st:idx], 1e-12)
        h_win = hh[st:idx]
        l_win = ll[st:idx]
        v_win = vv[st:idx]
        if c_win.size == 0:
            return None

        logc = np.log(c_win)
        ret = np.zeros_like(logc)
        if ret.size > 1:
            ret[1:] = np.diff(logc)
        hl = (h_win - l_win) / c_win
        v_mean = float(np.mean(v_win))
        v_std = float(np.std(v_win)) + 1e-6
        vz = (v_win - v_mean) / v_std
        feat = np.vstack([ret, hl, vz]).astype(np.float32)
        if feat.shape[1] < int(lookback):
            pad = np.zeros((feat.shape[0], int(lookback - feat.shape[1])), dtype=np.float32)
            feat = np.hstack([pad, feat])
        feat = np.clip(feat, -5.0, 5.0)
        mats.append(feat)
    if not mats:
        return None
    return np.vstack(mats).astype(np.float32)


def safe_label_id(vocab: dict[str, int], key: Any, unknown_key: str) -> int:
    k = str(key or "").strip().lower()
    if k in vocab:
        return int(vocab[k])
    return int(vocab.get(unknown_key, 0))


class MTFMultiTaskNet(nn.Module):
    def __init__(
        self,
        *,
        h: int,
        w: int,
        n_regimes: int,
        n_groups: int,
        n_exit_reasons: int,
        embed_dim: int = 8,
        hidden_dim: int = 96,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 48, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.reg_emb = nn.Embedding(max(1, int(n_regimes)), int(embed_dim))
        self.grp_emb = nn.Embedding(max(1, int(n_groups)), int(embed_dim))
        self.body = nn.Sequential(
            nn.Linear(48 + int(embed_dim) * 2, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.head_win = nn.Linear(int(hidden_dim), 1)
        self.head_long = nn.Linear(int(hidden_dim), 1)
        self.head_short = nn.Linear(int(hidden_dim), 1)
        self.head_hold = nn.Linear(int(hidden_dim), 1)
        self.head_exit = nn.Linear(int(hidden_dim), max(2, int(n_exit_reasons)))
        self._h = int(h)
        self._w = int(w)

    def forward(
        self,
        x: torch.Tensor,
        regime_id: torch.Tensor,
        group_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        er = self.reg_emb(regime_id.long())
        eg = self.grp_emb(group_id.long())
        h = self.body(torch.cat([z, er, eg], dim=-1))
        return {
            "win": self.head_win(h).squeeze(-1),
            "long": self.head_long(h).squeeze(-1),
            "short": self.head_short(h).squeeze(-1),
            "hold": self.head_hold(h).squeeze(-1),
            "exit": self.head_exit(h),
        }


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
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

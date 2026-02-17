#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mtf_signal_attention import MTFSignalAttentionNet

try:
    from captum.attr import IntegratedGradients, NoiseTunnel
except Exception:
    IntegratedGradients = None
    NoiseTunnel = None


ONE_MIN_MS = 60_000


@dataclass
class AlignedDataset:
    mtf_latent: np.ndarray  # (N, 96)
    signal_seq: np.ndarray  # (N, L, 2)
    signal_mask: np.ndarray  # (N, L)
    signal_ts_ms: np.ndarray  # (N, L)
    sample_ids: list[str]
    end_ts_ms: np.ndarray  # (N,)
    pnl: np.ndarray | None


def _require_plotting() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib")


def _first_existing(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"missing required columns among: {names}")


def _detect_latent_columns(df: pd.DataFrame, latent_dim: int) -> list[str]:
    patt = re.compile(r"^(latent|mtf_latent|h)_(\d+)$", flags=re.IGNORECASE)
    indexed: dict[int, str] = {}
    for c in df.columns:
        m = patt.match(str(c))
        if not m:
            continue
        idx = int(m.group(2))
        indexed[idx] = str(c)
    if len(indexed) < latent_dim:
        raise ValueError(
            f"could not detect enough latent columns; found={len(indexed)} required={latent_dim}. "
            "Expected columns like latent_0..latent_95."
        )
    expected = list(range(int(latent_dim)))
    missing = [i for i in expected if i not in indexed]
    if missing:
        raise ValueError(
            f"missing latent columns for indices: {missing[:8]} "
            f"(and {max(0, len(missing) - 8)} more)"
        )
    cols = [indexed[i] for i in expected]
    if len(cols) != latent_dim:
        raise ValueError(f"latent column count mismatch: {len(cols)} != {latent_dim}")
    return cols


def _coerce_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if default is not None:
        out = out.fillna(float(default))
    return out


def _align_signal_window(
    signal_slice: pd.DataFrame,
    *,
    end_ts_ms: int,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Leakage guard: only use timestamps <= end_ts_ms.
    src = signal_slice[signal_slice["ts_ms"] <= int(end_ts_ms)].copy()
    src = src.sort_values("ts_ms")
    if src.empty:
        expected = np.arange(
            int(end_ts_ms) - (int(lookback) - 1) * ONE_MIN_MS,
            int(end_ts_ms) + ONE_MIN_MS,
            ONE_MIN_MS,
            dtype=np.int64,
        )
        seq = np.zeros((int(lookback), 2), dtype=np.float32)
        mask = np.zeros(int(lookback), dtype=bool)
        return seq, mask, expected

    diffs = src["ts_ms"].diff().dropna().to_numpy(dtype=np.float64)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    step_ms = int(np.median(diffs)) if diffs.size > 0 else ONE_MIN_MS
    step_ms = max(1, int(step_ms))

    src["bucket_ts_ms"] = (src["ts_ms"].astype(np.int64) // step_ms) * step_ms
    src = src.sort_values("bucket_ts_ms").drop_duplicates(subset=["bucket_ts_ms"], keep="last")

    end_bucket = (int(end_ts_ms) // step_ms) * step_ms
    expected = np.arange(
        end_bucket - (int(lookback) - 1) * step_ms,
        end_bucket + step_ms,
        step_ms,
        dtype=np.int64,
    )
    aligned = (
        src.set_index("bucket_ts_ms")[["ofi", "vpin"]]
        .reindex(pd.Index(expected, name="bucket_ts_ms"))
        .sort_index()
    )
    valid_mask = aligned.notna().all(axis=1).to_numpy(dtype=bool)
    aligned = aligned.fillna(0.0)
    seq = aligned.to_numpy(dtype=np.float32)
    return seq, valid_mask, expected


def _pick_join_key(samples_df: pd.DataFrame, signal_df: pd.DataFrame) -> str | None:
    candidates = ["sample_id", "trade_uid", "entry_link_id", "symbol"]
    for c in candidates:
        if c in samples_df.columns and c in signal_df.columns:
            return c
    return None


def load_aligned_dataset(
    *,
    samples_csv: Path,
    signal_csv: Path,
    lookback: int,
    latent_dim: int,
) -> AlignedDataset:
    samples_df = pd.read_csv(samples_csv)
    signal_df = pd.read_csv(signal_csv)

    samples_end_col = _first_existing(samples_df, ["mtf_end_ts_ms", "entry_ts_ms", "ts_ms", "timestamp_ms"])
    signal_ts_col = _first_existing(signal_df, ["ts_ms", "timestamp_ms", "time_ms"])
    ofi_col = _first_existing(signal_df, ["ofi", "OFI", "alpha_ofi"])
    vpin_col = _first_existing(signal_df, ["vpin", "VPIN", "alpha_vpin"])

    latent_cols = _detect_latent_columns(samples_df, latent_dim=latent_dim)
    samples_df[samples_end_col] = _coerce_numeric(samples_df[samples_end_col], default=0).astype(np.int64)
    signal_df[signal_ts_col] = _coerce_numeric(signal_df[signal_ts_col], default=0).astype(np.int64)
    signal_df["ofi"] = _coerce_numeric(signal_df[ofi_col], default=0.0)
    signal_df["vpin"] = _coerce_numeric(signal_df[vpin_col], default=0.0)

    join_key = _pick_join_key(samples_df, signal_df)
    if join_key is None:
        signal_df["_sample_id"] = "global"
        join_key = "_sample_id"
        samples_df[join_key] = "global"

    signal_df = signal_df.rename(columns={signal_ts_col: "ts_ms"})
    samples_df = samples_df.rename(columns={samples_end_col: "end_ts_ms"})

    grouped = {str(k): g for k, g in signal_df.groupby(join_key, sort=False)}
    empty_signal = pd.DataFrame(columns=["ts_ms", "ofi", "vpin"])

    latents = samples_df[latent_cols].fillna(0.0).to_numpy(dtype=np.float32)
    signal_seq_list: list[np.ndarray] = []
    signal_mask_list: list[np.ndarray] = []
    signal_ts_list: list[np.ndarray] = []
    sample_ids: list[str] = []

    mismatch_count = 0
    for _, row in samples_df.iterrows():
        sid = str(row.get(join_key))
        end_ts = int(row["end_ts_ms"])
        sample_signal = grouped.get(sid, empty_signal)
        seq, mask, ts_arr = _align_signal_window(sample_signal, end_ts_ms=end_ts, lookback=lookback)
        signal_seq_list.append(seq)
        signal_mask_list.append(mask)
        signal_ts_list.append(ts_arr)
        sample_ids.append(sid)

        if "mtf_start_ts_ms" in samples_df.columns:
            mtf_start = int(float(row.get("mtf_start_ts_ms", 0.0) or 0.0))
            if mtf_start > 0 and abs(int(ts_arr[0]) - mtf_start) > ONE_MIN_MS:
                mismatch_count += 1

    pnl_col = None
    for candidate in ("pnl", "realized_pnl", "profit", "target_pnl"):
        if candidate in samples_df.columns:
            pnl_col = candidate
            break
    pnl = None
    if pnl_col is not None:
        pnl = _coerce_numeric(samples_df[pnl_col], default=0.0).to_numpy(dtype=np.float32)

    if mismatch_count > 0:
        print(
            f"[warn] lookback sync mismatches detected (count={mismatch_count}). "
            "Check mtf_start_ts_ms and signal timestamps for exact alignment."
        )

    return AlignedDataset(
        mtf_latent=latents,
        signal_seq=np.asarray(signal_seq_list, dtype=np.float32),
        signal_mask=np.asarray(signal_mask_list, dtype=bool),
        signal_ts_ms=np.asarray(signal_ts_list, dtype=np.int64),
        sample_ids=sample_ids,
        end_ts_ms=samples_df["end_ts_ms"].to_numpy(dtype=np.int64),
        pnl=pnl,
    )


def load_model(
    *,
    device: torch.device,
    latent_dim: int,
    num_heads: int,
    weights: Path | None,
) -> MTFSignalAttentionNet:
    model = MTFSignalAttentionNet(
        latent_dim=int(latent_dim),
        signal_dim=2,
        num_heads=int(num_heads),
    ).to(device)
    if weights is not None and weights.exists():
        ckpt = torch.load(weights, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            remap = {}
            for k, v in ckpt.items():
                kk = str(k)
                if kk.startswith("module."):
                    kk = kk[len("module.") :]
                remap[kk] = v
            missing, unexpected = model.load_state_dict(remap, strict=False)
            if missing:
                print(f"[warn] missing keys while loading weights: {len(missing)}")
            if unexpected:
                print(f"[warn] unexpected keys while loading weights: {len(unexpected)}")
    model.eval()
    return model


def batched_win_prob(
    model: MTFSignalAttentionNet,
    *,
    mtf_latent: np.ndarray,
    signal_seq: np.ndarray,
    signal_mask: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    n = int(mtf_latent.shape[0])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for st in range(0, n, int(batch_size)):
            ed = min(n, st + int(batch_size))
            lat = torch.from_numpy(mtf_latent[st:ed]).to(device)
            sig = torch.from_numpy(signal_seq[st:ed]).to(device)
            msk = torch.from_numpy(signal_mask[st:ed]).to(device)
            pred = model(lat, sig, signal_mask=msk, return_attention_weights=False)["win"]
            out.append(torch.sigmoid(pred).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def choose_sample_index(pnl: np.ndarray | None, requested: int, total: int) -> int:
    if total <= 0:
        raise ValueError("empty dataset")
    if requested >= 0:
        if requested >= total:
            raise ValueError(f"sample-index out of range: {requested} >= {total}")
        return int(requested)
    if pnl is not None and pnl.size == total:
        return int(np.argmax(pnl))
    return 0


def compute_attention_for_sample(
    model: MTFSignalAttentionNet,
    *,
    mtf_latent: np.ndarray,
    signal_seq: np.ndarray,
    signal_mask: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    with torch.no_grad():
        lat = torch.from_numpy(mtf_latent[None, ...]).to(device)
        sig = torch.from_numpy(signal_seq[None, ...]).to(device)
        msk = torch.from_numpy(signal_mask[None, ...]).to(device)
        out = model(lat, sig, signal_mask=msk, return_attention_weights=True)
        win_prob = float(torch.sigmoid(out["win"]).item())
        attn_w = out["attention_weights"]
        if attn_w is None:
            raise RuntimeError("attention weights are missing")
        attn = model.summarize_attention(attn_w).squeeze(0).detach().cpu().numpy()
    return win_prob, attn


def compute_captum_attribution(
    model: MTFSignalAttentionNet,
    *,
    mtf_latent: np.ndarray,
    signal_seq: np.ndarray,
    signal_mask: np.ndarray,
    device: torch.device,
    ig_steps: int,
    smoothgrad_samples: int,
    smoothgrad_stdev: float,
    use_smoothgrad: bool,
) -> np.ndarray:
    if IntegratedGradients is None:
        raise RuntimeError("captum is required. Install with: pip install captum")
    if use_smoothgrad and NoiseTunnel is None:
        raise RuntimeError("captum NoiseTunnel is unavailable in current environment")

    lat = torch.from_numpy(mtf_latent[None, ...]).to(device)
    msk = torch.from_numpy(signal_mask[None, ...]).to(device)
    sig = torch.from_numpy(signal_seq[None, ...]).to(device).requires_grad_(True)
    baseline = torch.zeros_like(sig)

    def forward_fn(signal_input: torch.Tensor) -> torch.Tensor:
        bsz = int(signal_input.size(0))
        lat_b = lat.expand(bsz, -1)
        msk_b = msk.expand(bsz, -1)
        return model(lat_b, signal_input, signal_mask=msk_b, return_attention_weights=False)["win"]

    ig = IntegratedGradients(forward_fn)
    if use_smoothgrad:
        nt = NoiseTunnel(ig)
        attrs = nt.attribute(
            sig,
            baselines=baseline,
            nt_type="smoothgrad_sq",
            nt_samples=max(1, int(smoothgrad_samples)),
            stdevs=float(smoothgrad_stdev),
            n_steps=max(8, int(ig_steps)),
        )
    else:
        attrs = ig.attribute(sig, baselines=baseline, n_steps=max(8, int(ig_steps)))
    return attrs.squeeze(0).detach().cpu().numpy()


def save_attention_overlay_plot(
    *,
    ts_ms: np.ndarray,
    ofi: np.ndarray,
    vpin: np.ndarray,
    attention: np.ndarray,
    out_path: Path,
) -> None:
    _require_plotting()
    x = np.arange(len(ts_ms), dtype=np.int64)
    attn = np.asarray(attention, dtype=np.float64)
    attn = np.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(attn) > np.min(attn):
        attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn))
    else:
        attn = np.zeros_like(attn)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, values, label in (
        (axes[0], ofi, "OFI"),
        (axes[1], vpin, "VPIN"),
    ):
        ax.plot(x, values, color="#444444", linewidth=1.1, alpha=0.9)
        sc = ax.scatter(x, values, c=attn, cmap="magma", s=58, edgecolors="none")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
    cbar = fig.colorbar(sc, ax=axes, location="right", pad=0.02)
    cbar.set_label("Cross-Attention Weight (normalized)")
    axes[-1].set_xlabel("Lookback Step (old -> recent)")
    axes[0].set_title("OFI/VPIN with Attention Heat Overlay")

    if len(ts_ms) > 0:
        tick_count = min(6, len(ts_ms))
        tick_idx = np.linspace(0, len(ts_ms) - 1, tick_count, dtype=int)
        tick_lbl = [pd.to_datetime(int(ts_ms[i]), unit="ms", utc=True).strftime("%H:%M") for i in tick_idx]
        axes[-1].set_xticks(tick_idx)
        axes[-1].set_xticklabels(tick_lbl, rotation=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_threshold_scatter3d(
    *,
    ofi_vals: np.ndarray,
    vpin_vals: np.ndarray,
    win_prob: np.ndarray,
    selected_index: int,
    out_path: Path,
) -> None:
    _require_plotting()
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        ofi_vals,
        win_prob,
        vpin_vals,
        c=vpin_vals,
        cmap="viridis",
        s=26,
        alpha=0.85,
    )
    if 0 <= selected_index < len(ofi_vals):
        ax.scatter(
            [ofi_vals[selected_index]],
            [win_prob[selected_index]],
            [vpin_vals[selected_index]],
            c="red",
            marker="*",
            s=180,
            label="selected sample",
        )
        ax.legend(loc="best")
    ax.set_xlabel("OFI (last lookback step)")
    ax.set_ylabel("Predicted Win Probability")
    ax.set_zlabel("VPIN (last lookback step)")
    ax.set_title("OFI vs Predicted Win Probability (color/height = VPIN)")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label("VPIN")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def derive_threshold_candidates(
    *,
    ofi_vals: np.ndarray,
    vpin_vals: np.ndarray,
    win_prob: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q in (0.70, 0.80, 0.90):
        thr = float(np.quantile(win_prob, q))
        m = win_prob >= thr
        count = int(np.sum(m))
        if count <= 0:
            rows.append(
                {
                    "win_prob_quantile": q,
                    "win_prob_threshold": thr,
                    "count": 0,
                    "ofi_median": math.nan,
                    "vpin_median": math.nan,
                    "ofi_p25": math.nan,
                    "ofi_p75": math.nan,
                    "vpin_p25": math.nan,
                    "vpin_p75": math.nan,
                }
            )
            continue
        rows.append(
            {
                "win_prob_quantile": q,
                "win_prob_threshold": thr,
                "count": count,
                "ofi_median": float(np.median(ofi_vals[m])),
                "vpin_median": float(np.median(vpin_vals[m])),
                "ofi_p25": float(np.quantile(ofi_vals[m], 0.25)),
                "ofi_p75": float(np.quantile(ofi_vals[m], 0.75)),
                "vpin_p25": float(np.quantile(vpin_vals[m], 0.25)),
                "vpin_p75": float(np.quantile(vpin_vals[m], 0.75)),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MTF latent / OFI-VPIN cross-attention and Captum attributions."
    )
    parser.add_argument("--samples-csv", type=Path, required=True, help="Per-sample table with latent_0..95")
    parser.add_argument("--signal-csv", type=Path, required=True, help="Signal table with ts_ms, ofi, vpin")
    parser.add_argument("--weights", type=Path, default=None, help="Optional model checkpoint for attention net")
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--sample-index", type=int, default=-1, help="-1 => max pnl sample (if pnl exists)")
    parser.add_argument("--ig-steps", type=int, default=64)
    parser.add_argument("--smoothgrad-samples", type=int, default=8)
    parser.add_argument("--smoothgrad-stdev", type=float, default=0.02)
    parser.add_argument("--no-smoothgrad", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mtf_signal_xai"))
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.lookback) <= 0:
        raise ValueError("lookback must be positive")
    if int(args.latent_dim) != 96:
        print("[warn] Current MTF hidden_dim is 96. Consider using --latent-dim 96 for exact matching.")

    dataset = load_aligned_dataset(
        samples_csv=args.samples_csv,
        signal_csv=args.signal_csv,
        lookback=int(args.lookback),
        latent_dim=int(args.latent_dim),
    )
    n_samples = int(dataset.mtf_latent.shape[0])
    if n_samples <= 0:
        raise RuntimeError("no samples loaded")
    print(f"[info] loaded samples={n_samples} lookback={args.lookback}")

    device = torch.device(str(args.device))
    model = load_model(
        device=device,
        latent_dim=int(args.latent_dim),
        num_heads=int(args.num_heads),
        weights=args.weights,
    )

    win_prob_all = batched_win_prob(
        model,
        mtf_latent=dataset.mtf_latent,
        signal_seq=dataset.signal_seq,
        signal_mask=dataset.signal_mask,
        device=device,
        batch_size=int(args.batch_size),
    )

    sample_index = choose_sample_index(dataset.pnl, int(args.sample_index), n_samples)
    sample_win_prob, sample_attn = compute_attention_for_sample(
        model,
        mtf_latent=dataset.mtf_latent[sample_index],
        signal_seq=dataset.signal_seq[sample_index],
        signal_mask=dataset.signal_mask[sample_index],
        device=device,
    )

    attrs = compute_captum_attribution(
        model,
        mtf_latent=dataset.mtf_latent[sample_index],
        signal_seq=dataset.signal_seq[sample_index],
        signal_mask=dataset.signal_mask[sample_index],
        device=device,
        ig_steps=int(args.ig_steps),
        smoothgrad_samples=int(args.smoothgrad_samples),
        smoothgrad_stdev=float(args.smoothgrad_stdev),
        use_smoothgrad=not bool(args.no_smoothgrad),
    )
    attr_abs = np.abs(attrs).sum(axis=-1)
    top_k = max(1, min(int(args.top_k), int(len(attr_abs))))
    top_idx = np.argsort(-attr_abs)[:top_k]

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_ms = dataset.signal_ts_ms[sample_index]
    ofi_seq = dataset.signal_seq[sample_index, :, 0]
    vpin_seq = dataset.signal_seq[sample_index, :, 1]

    attr_df = pd.DataFrame(
        {
            "ts_ms": ts_ms.astype(np.int64),
            "ofi": ofi_seq.astype(np.float64),
            "vpin": vpin_seq.astype(np.float64),
            "attention_weight": sample_attn.astype(np.float64),
            "attr_ofi": attrs[:, 0].astype(np.float64),
            "attr_vpin": attrs[:, 1].astype(np.float64),
            "attr_abs_total": attr_abs.astype(np.float64),
        }
    )
    attr_csv = out_dir / "sample_attribution.csv"
    attr_df.to_csv(attr_csv, index=False)

    attn_plot = out_dir / "attention_overlay.png"
    save_attention_overlay_plot(
        ts_ms=ts_ms,
        ofi=ofi_seq,
        vpin=vpin_seq,
        attention=sample_attn,
        out_path=attn_plot,
    )

    ofi_last = dataset.signal_seq[:, -1, 0]
    vpin_last = dataset.signal_seq[:, -1, 1]
    scatter_plot = out_dir / "ofi_winprob_vpin_scatter3d.png"
    save_threshold_scatter3d(
        ofi_vals=ofi_last,
        vpin_vals=vpin_last,
        win_prob=win_prob_all,
        selected_index=sample_index,
        out_path=scatter_plot,
    )

    threshold_df = derive_threshold_candidates(ofi_vals=ofi_last, vpin_vals=vpin_last, win_prob=win_prob_all)
    threshold_csv = out_dir / "threshold_candidates.csv"
    threshold_df.to_csv(threshold_csv, index=False)

    top_rows: list[dict[str, Any]] = []
    for i in top_idx:
        top_rows.append(
            {
                "rank": len(top_rows) + 1,
                "step_index": int(i),
                "ts_ms": int(ts_ms[i]),
                "ofi": float(ofi_seq[i]),
                "vpin": float(vpin_seq[i]),
                "attr_abs_total": float(attr_abs[i]),
            }
        )

    summary = {
        "selected_sample_index": int(sample_index),
        "selected_sample_id": dataset.sample_ids[sample_index],
        "selected_end_ts_ms": int(dataset.end_ts_ms[sample_index]),
        "selected_win_prob": float(sample_win_prob),
        "top_contributing_steps": top_rows,
        "output_files": {
            "sample_attribution_csv": str(attr_csv),
            "attention_overlay_plot": str(attn_plot),
            "threshold_scatter_plot": str(scatter_plot),
            "threshold_candidates_csv": str(threshold_csv),
        },
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] selected_sample_index={sample_index} win_prob={sample_win_prob:.6f}")
    for row in top_rows:
        print(
            "[top] "
            f"rank={row['rank']} step={row['step_index']} ts_ms={row['ts_ms']} "
            f"ofi={row['ofi']:.6f} vpin={row['vpin']:.6f} attr={row['attr_abs_total']:.6f}"
        )
    print(f"[done] saved: {summary_json}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MTFSignalAttentionNet(nn.Module):
    """Cross-attention fusion between MTF latent vector and OFI/VPIN sequence."""

    def __init__(
        self,
        *,
        latent_dim: int = 96,
        signal_dim: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
        mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        if int(latent_dim) <= 0:
            raise ValueError("latent_dim must be positive")
        if int(signal_dim) <= 0:
            raise ValueError("signal_dim must be positive")
        if int(num_heads) <= 0:
            raise ValueError("num_heads must be positive")
        if int(latent_dim) % int(num_heads) != 0:
            raise ValueError("latent_dim must be divisible by num_heads")

        self.latent_dim = int(latent_dim)
        self.signal_dim = int(signal_dim)

        self.signal_encoder = nn.Sequential(
            nn.Conv1d(self.signal_dim, self.latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.signal_norm = nn.LayerNorm(self.latent_dim)

        self.query_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.query_norm = nn.LayerNorm(self.latent_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(self.latent_dim)
        self.dropout = nn.Dropout(float(dropout))

        self.win_head = nn.Sequential(
            nn.Linear(self.latent_dim, int(mlp_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden), 1),
        )

    def _sanitize_signals(
        self,
        signal_seq: torch.Tensor,
        signal_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # signal_seq: (B, L, F), signal_mask: (B, L) where True means valid.
        finite_valid = torch.isfinite(signal_seq).all(dim=-1)
        x = torch.nan_to_num(signal_seq, nan=0.0, posinf=0.0, neginf=0.0)

        if signal_mask is None:
            valid_mask = finite_valid
        else:
            if signal_mask.dim() != 2:
                raise ValueError("signal_mask must be 2D [batch, seq_len]")
            if signal_mask.shape != finite_valid.shape:
                raise ValueError("signal_mask shape must match signal_seq[:, :, 0]")
            valid_mask = finite_valid & signal_mask.to(dtype=torch.bool)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return x, valid_mask

    def forward(
        self,
        mtf_latent: torch.Tensor,
        signal_seq: torch.Tensor,
        *,
        signal_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        if mtf_latent.dim() != 2:
            raise ValueError("mtf_latent must be 2D [batch, latent_dim]")
        if signal_seq.dim() != 3:
            raise ValueError("signal_seq must be 3D [batch, seq_len, features]")
        if mtf_latent.size(0) != signal_seq.size(0):
            raise ValueError("batch size mismatch between mtf_latent and signal_seq")
        if mtf_latent.size(-1) != self.latent_dim:
            raise ValueError(f"mtf_latent dim must be {self.latent_dim}")
        if signal_seq.size(-1) != self.signal_dim:
            raise ValueError(f"signal_seq feature dim must be {self.signal_dim}")

        mtf_latent = torch.nan_to_num(mtf_latent, nan=0.0, posinf=0.0, neginf=0.0)
        x, valid_mask = self._sanitize_signals(signal_seq, signal_mask)

        # (B, L, F) -> (B, F, L) -> CNN -> (B, L, 96)
        sig_embed = self.signal_encoder(x.transpose(1, 2)).transpose(1, 2)
        sig_embed = self.signal_norm(sig_embed)

        query = self.query_norm(self.query_proj(mtf_latent)).unsqueeze(1)  # (B, 1, 96)
        key_padding_mask = ~valid_mask  # True positions are ignored by attention.

        # Prevent all-masked rows that can produce NaNs in attention.
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, -1] = False

        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=sig_embed,
            value=sig_embed,
            key_padding_mask=key_padding_mask,
            need_weights=bool(return_attention_weights),
            average_attn_weights=False,
        )
        fused = self.post_attn_norm(query + self.dropout(attn_out)).squeeze(1)
        win_logit = self.win_head(fused).squeeze(-1)

        out: dict[str, torch.Tensor | None] = {
            "win": win_logit,
            "fused": fused,
            "attention_weights": None,
        }
        if bool(return_attention_weights):
            # Shape: (B, num_heads, target_len=1, seq_len)
            out["attention_weights"] = attn_weights
        return out

    @staticmethod
    def summarize_attention(attn_weights: torch.Tensor) -> torch.Tensor:
        """Return per-time-step attention by averaging over heads."""
        if attn_weights.dim() != 4:
            raise ValueError("attention_weights must be 4D [B, H, T, S]")
        return attn_weights.mean(dim=1).squeeze(1)  # (B, S)


def build_signal_tensor(ofi: Any, vpin: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Utility: create (B, L, 2) tensor from OFI/VPIN arrays with NaN handling."""
    ofi_t = torch.as_tensor(ofi, dtype=dtype)
    vpin_t = torch.as_tensor(vpin, dtype=dtype)
    if ofi_t.shape != vpin_t.shape:
        raise ValueError("ofi and vpin must have the same shape")
    stacked = torch.stack([ofi_t, vpin_t], dim=-1)
    return torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

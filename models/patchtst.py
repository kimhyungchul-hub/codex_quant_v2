"""PatchTST (Patch Time Series Transformer) implementation.

Channel-independent patch embeddings for multivariate time series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class PatchTSTShape:
    """Container for patching metadata."""

    n_patches: int
    patch_len: int
    stride: int


class PatchTST(nn.Module):
    """Patch Time Series Transformer (channel-independent).

    Args:
        n_vars: number of variables (channels).
        context_length: input sequence length.
        pred_length: number of steps to forecast per variable.
        patch_len: window size per patch.
        stride: hop length between patches.
        d_model: transformer hidden size.
        depth: number of encoder layers.
        n_heads: attention heads.
        dropout: dropout probability.
    """

    def __init__(
        self,
        n_vars: int,
        context_length: int,
        pred_length: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        depth: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        self.n_vars = n_vars
        self.context_length = context_length
        self.pred_length = pred_length
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        n_patches = self._calc_n_patches(context_length)
        if n_patches <= 0:
            raise ValueError("context_length too short for given patch_len/stride")
        self.n_patches = n_patches

        self.patch_embed = nn.ModuleList([nn.Linear(patch_len, d_model) for _ in range(n_vars)])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.positional = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.positional, std=0.02)
        self.head = nn.ModuleList([nn.Linear(d_model, pred_length) for _ in range(n_vars)])
        self.dropout = nn.Dropout(dropout)

    def _calc_n_patches(self, length: int) -> int:
        if length < self.patch_len:
            return 0
        return 1 + (length - self.patch_len) // self.stride

    def describe_shape(self, length: int | None = None) -> PatchTSTShape:
        length = length or self.context_length
        return PatchTSTShape(
            n_patches=self._calc_n_patches(length),
            patch_len=self.patch_len,
            stride=self.stride,
        )

    def _unfold(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> patches: (B, C, Np, patch_len)
        if x.dim() != 3:
            raise ValueError("input must be 3D [batch, length, channels]")
        if x.size(1) < self.patch_len:
            raise ValueError("sequence shorter than patch_len")
        x_perm = x.transpose(1, 2)
        patches = x_perm.unfold(dimension=2, size=self.patch_len, step=self.stride)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        patches = self._unfold(x)
        b, c, n_patches, p_len = patches.shape
        if n_patches > self.n_patches:
            raise ValueError("input produces more patches than model capacity; increase context_length")
        pos = self.positional[:, :n_patches, :]

        tokens = []
        for ch in range(c):
            patch = patches[:, ch, :, :].reshape(b * n_patches, p_len)
            emb = self.patch_embed[ch](patch)
            emb = self.dropout(emb)
            emb = emb.view(b, n_patches, self.d_model)
            emb = emb + pos
            tokens.append(emb)

        tokens_cat = torch.stack(tokens, dim=1)  # (B, C, Np, d_model)
        tokens_cat = tokens_cat.view(b * c, n_patches, self.d_model).transpose(0, 1)

        encoded = self.encoder(tokens_cat).transpose(0, 1)
        encoded = encoded.view(b, c, n_patches, self.d_model)
        last_token = encoded[:, :, -1, :]

        preds = []
        for ch in range(c):
            preds.append(self.head[ch](last_token[:, ch, :]))
        out = torch.stack(preds, dim=1)  # (B, C, pred_length)
        return out


def make_patch_tensor(
    series: torch.Tensor, patch_len: int, stride: int
) -> Tuple[torch.Tensor, PatchTSTShape]:
    """Utility for patching outside the model.

    Args:
        series: tensor of shape (B, L, C)
        patch_len: patch length
        stride: hop length
    Returns:
        patches: tensor of shape (B, C, Np, patch_len)
        shape: PatchTSTShape metadata
    """

    if series.dim() != 3:
        raise ValueError("series must be 3D [batch, length, channels]")
    if patch_len <= 0 or stride <= 0:
        raise ValueError("patch_len and stride must be positive")
    if series.size(1) < patch_len:
        raise ValueError("sequence shorter than patch_len")

    perm = series.transpose(1, 2)
    patches = perm.unfold(dimension=2, size=patch_len, step=stride)
    n_patches = 1 + (series.size(1) - patch_len) // stride
    shape = PatchTSTShape(n_patches=n_patches, patch_len=patch_len, stride=stride)
    return patches, shape

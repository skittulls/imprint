"""
Multi-Scale Gram Matrix Attention (MSGMAtt).

This module is the core novelty of Architecture v2.  It computes Gram
matrices at multiple CNN depths and uses learned self-attention to weight
each scale's contribution to the final style embedding.

**Why Gram matrices?**
    Gatys et al. (2016) showed that the Gram matrix of a CNN feature map
    captures the *statistical correlations* between filter responses — i.e.
    which textures co-occur.  These correlations are a powerful signature
    of artistic style that is largely invariant to spatial layout.

**Why multi-scale?**
    Style manifests at different spatial frequencies:
      • Fine scale (layer2): individual brushstrokes, edge quality
      • Mid scale  (layer3): recurring motifs, texture patterns
      • Coarse scale (layer4): compositional rhythm, colour palette

    A single-scale Gram matrix misses cross-frequency style cues.

**Why attention?**
    Not all scales matter equally for every artist pair.  Two Impressionists
    may differ mainly at the brushstroke level (fine scale), while a
    Cubist and a Realist diverge at the composition level (coarse scale).
    Learned self-attention lets the model dynamically weight scales.

Architecture::

    Feature maps {F₂, F₃, F₄}
        │
        ├─ 1×1 Conv (channel reduction to d_reduce)
        ├─ Gram matrix G = FF^T / N  (texture correlations)
        ├─ Upper-triangle flatten
        ├─ Linear projection → d_proj
        │
        └─→ Stack [g₂, g₃, g₄] ∈ ℝ^{B × 3 × d_proj}
              │
              └─ Multi-Head Self-Attention
              │
              └─ Learnable weighted pool → ℝ^{B × d_proj}

References:
    Gatys et al., "A Neural Algorithm of Artistic Style", arXiv 1508.06576.
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramAttentionOutput(NamedTuple):
    """Container for MSGMAtt outputs."""
    style_features: torch.Tensor       # (B, gram_proj_dim)
    attention_weights: torch.Tensor    # (B, num_scales)
    per_scale_grams: torch.Tensor      # (B, num_scales, gram_proj_dim)


class GramMatrix(nn.Module):
    """
    Compute the normalised Gram matrix of a feature map.

    Given feature map F ∈ ℝ^{B × C × H × W}, the Gram matrix is:

        G = (1 / N) · F̃ · F̃ᵀ    where F̃ ∈ ℝ^{B × C × HW},  N = H × W

    Only the **upper triangle** (including diagonal) is returned as a
    flattened vector, since G is symmetric.  This halves dimensionality
    without information loss.

    For C = 64:  upper-triangle size = 64 × 65 / 2 = 2 080.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: ``[B, C, H, W]``

        Returns:
            Flattened upper-triangle Gram vectors ``[B, C*(C+1)/2]``
        """
        B, C, H, W = features.shape
        N = H * W

        # Reshape to (B, C, N) for batch matmul
        F_flat = features.view(B, C, N)

        # G = (1/N) * F @ F^T  →  (B, C, C)
        gram = torch.bmm(F_flat, F_flat.transpose(1, 2)) / N

        # Extract upper triangle (including diagonal) → (B, C*(C+1)/2)
        triu_indices = torch.triu_indices(C, C, device=features.device)
        gram_vec = gram[:, triu_indices[0], triu_indices[1]]

        return gram_vec


class MultiScaleGramAttention(nn.Module):
    """
    MSGMAtt:  Multi-Scale Gram Matrix Attention module.

    Processes feature maps from multiple backbone scales, computes a Gram
    matrix at each, and uses self-attention to produce a single, scale-aware
    style embedding.

    Args:
        scale_channels: Dict mapping scale names to their channel count.
                        E.g. ``{"layer2": 128, "layer3": 256, "layer4": 512}``.
        reduce_dim:     Channel count after 1×1 conv reduction (default 64).
        proj_dim:       Projection dimension for each Gram vector (default 256).
        num_heads:      Attention heads for multi-head self-attention.
        dropout:        Dropout rate inside attention and projection.
    """

    def __init__(
        self,
        scale_channels: Dict[str, int],
        reduce_dim: int = 64,
        proj_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.scale_names = sorted(scale_channels.keys())
        self.num_scales = len(self.scale_names)
        self.proj_dim = proj_dim

        gram_flat_dim = reduce_dim * (reduce_dim + 1) // 2  # upper triangle

        # Per-scale modules
        self.channel_reducers = nn.ModuleDict()
        self.gram_projectors = nn.ModuleDict()

        for name in self.scale_names:
            in_ch = scale_channels[name]
            # 1×1 conv to reduce channels before Gram computation
            self.channel_reducers[name] = nn.Sequential(
                nn.Conv2d(in_ch, reduce_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduce_dim),
                nn.ReLU(inplace=True),
            )
            # Project flattened Gram vector to common dimension
            self.gram_projectors[name] = nn.Sequential(
                nn.Linear(gram_flat_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, proj_dim),
            )

        self.gram = GramMatrix()

        # Self-attention across scales
        self.layer_norm = nn.LayerNorm(proj_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(proj_dim)

        # Learnable scale importance weights for final pooling
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))

    def forward(
        self, feature_maps: Dict[str, torch.Tensor]
    ) -> GramAttentionOutput:
        """
        Args:
            feature_maps: Dict from backbone, e.g.
                          ``{"layer2": (B,128,28,28), ...}``.

        Returns:
            ``GramAttentionOutput`` with style features, attention weights,
            and per-scale Gram projections.
        """
        gram_projections = []

        for name in self.scale_names:
            fm = feature_maps[name]                         # (B, C_in, H, W)
            fm_reduced = self.channel_reducers[name](fm)    # (B, d_reduce, H, W)
            gram_vec = self.gram(fm_reduced)                # (B, d_reduce*(d_reduce+1)/2)
            proj = self.gram_projectors[name](gram_vec)     # (B, proj_dim)
            gram_projections.append(proj)

        # Stack: (B, num_scales, proj_dim)
        scale_tokens = torch.stack(gram_projections, dim=1)
        scale_tokens = self.layer_norm(scale_tokens)

        # Self-attention across the scale dimension
        attended, _ = self.self_attention(
            scale_tokens, scale_tokens, scale_tokens
        )   # (B, num_scales, proj_dim)
        attended = self.post_attn_norm(attended + scale_tokens)  # residual

        # Weighted pooling over scales
        weights = F.softmax(self.scale_weights, dim=0)     # (num_scales,)
        style_vec = (attended * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        # → (B, proj_dim)

        return GramAttentionOutput(
            style_features=style_vec,
            attention_weights=weights.detach().expand(scale_tokens.size(0), -1),
            per_scale_grams=attended.detach(),
        )

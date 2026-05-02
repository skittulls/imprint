"""
Full style encoder: backbone → MSGMAtt → projection → L2-normalised embedding.

This module composes the ``MultiScaleResNet`` backbone with the
``MultiScaleGramAttention`` module and an optional global average-pooling
branch.  The two branches capture complementary signals:

    • **Gram branch** — texture correlations (scale-aware, style-specific)
    • **Global branch** — holistic scene statistics (colour palette, layout)

The fusion head concatenates both and projects to a compact, L2-normalised
embedding suitable for cosine-similarity based retrieval and verification.

Architecture diagram::

    Input image [B, 3, 224, 224]
         │
         ▼
    ┌────────────────────────┐
    │  MultiScaleResNet-18   │
    │  (pretrained, partial  │
    │   freeze)              │
    └────┬───────┬───────┬───┘
         │       │       │
       layer2  layer3  layer4
         │       │       │
         ▼       ▼       ▼
    ┌────────────────────────┐      ┌──────────────────┐
    │  MultiScaleGramAttention│      │  AdaptiveAvgPool  │
    │  (MSGMAtt)              │      │  + Linear(512→256)│
    └────────┬───────────────┘      └────────┬─────────┘
             │ (B, 256)                      │ (B, 256)
             └──────────┬───────────────────┘
                        │ concat → (B, 512)
                        ▼
                ┌───────────────┐
                │ Projection    │
                │ 512→128, ReLU │
                │ + L2 normalise│
                └───────┬───────┘
                        │ (B, 128)
                        ▼
                   Style embedding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import MultiScaleResNet, RESNET18_CHANNELS
from src.models.gram_attention import MultiScaleGramAttention, GramAttentionOutput


@dataclass
class EncoderOutput:
    """Typed container for encoder outputs."""
    embedding: torch.Tensor           # (B, embedding_dim)  — L2-normalised
    attention_weights: torch.Tensor   # (B, num_scales)     — scale importance
    gram_features: torch.Tensor       # (B, num_scales, gram_proj_dim)


class StyleEncoder(nn.Module):
    """
    End-to-end style encoder.

    Args:
        pretrained:       Use ImageNet-pretrained backbone.
        freeze_until:     Freeze backbone up to this layer (inclusive).
        gram_reduce_dim:  Channel reduction before Gram matrix.
        gram_proj_dim:    Projection dim per scale.
        num_heads:        Attention heads in MSGMAtt.
        embedding_dim:    Final embedding dimensionality.
        dropout:          Dropout rate.
        use_global_branch: Whether to include the GAP branch.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_until: str | None = "layer2",
        gram_reduce_dim: int = 64,
        gram_proj_dim: int = 256,
        num_heads: int = 4,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        use_global_branch: bool = True,
    ) -> None:
        super().__init__()
        self.use_global_branch = use_global_branch

        # 1. Backbone
        self.backbone = MultiScaleResNet(
            pretrained=pretrained,
            freeze_until=freeze_until,
        )

        # 2. Multi-Scale Gram Attention (Novelty 1)
        self.gram_attention = MultiScaleGramAttention(
            scale_channels=RESNET18_CHANNELS,
            reduce_dim=gram_reduce_dim,
            proj_dim=gram_proj_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 3. Optional global branch (GAP on layer4)
        fusion_dim = gram_proj_dim
        if use_global_branch:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.global_proj = nn.Sequential(
                nn.Linear(RESNET18_CHANNELS["layer4"], gram_proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            fusion_dim = gram_proj_dim * 2  # concat of gram + global

        # 4. Projection head → final embedding
        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Args:
            x: Batch of images ``[B, 3, 224, 224]``.

        Returns:
            ``EncoderOutput`` with L2-normalised embedding, attention
            weights, and per-scale Gram features.
        """
        # Extract multi-scale features
        feature_maps: Dict[str, torch.Tensor] = self.backbone(x)

        # Gram attention
        gram_out: GramAttentionOutput = self.gram_attention(feature_maps)
        style_vec = gram_out.style_features   # (B, gram_proj_dim)

        # Optional global branch
        if self.use_global_branch:
            f4 = feature_maps["layer4"]
            global_vec = self.global_pool(f4).flatten(1)       # (B, 512)
            global_vec = self.global_proj(global_vec)           # (B, gram_proj_dim)
            style_vec = torch.cat([style_vec, global_vec], dim=1)  # (B, 2*gram_proj_dim)

        # Project and normalise
        embedding = self.projection(style_vec)                 # (B, embedding_dim)
        embedding = F.normalize(embedding, p=2, dim=1)

        return EncoderOutput(
            embedding=embedding,
            attention_weights=gram_out.attention_weights,
            gram_features=gram_out.per_scale_grams,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "StyleEncoder":
        """Construct encoder from a config dictionary (``cfg['model']``)."""
        m = cfg["model"]
        return cls(
            pretrained=m.get("pretrained", True),
            freeze_until=m.get("freeze_backbone_until", "layer2"),
            gram_reduce_dim=m.get("gram_reduce_dim", 64),
            gram_proj_dim=m.get("gram_proj_dim", 256),
            num_heads=m.get("attention_heads", 4),
            embedding_dim=m.get("embedding_dim", 128),
            dropout=m.get("dropout", 0.3),
            use_global_branch=m.get("use_global_branch", True),
        )

"""
Attentive Prototypical Head for few-shot style verification (Novelty 4).

Standard prototypical networks (Snell et al., NeurIPS 2017) compute a class
prototype as the **mean** of support embeddings.  This treats every support
image as equally informative, which is suboptimal for artistic style:
an artist's oeuvre may contain outlier works (commissioned portraits by a
landscape painter, experimental pieces) that are not representative.

Our ``AttentivePrototype`` replaces the mean with a **cross-attention
aggregation**: a learnable query token attends over the support set,
learning to up-weight canonical works and down-weight outliers.

The attention weights are interpretable — during inference you can show
*which* reference works most influenced the prototype.

Architecture::

    Support embeddings {z₁, …, zₖ} ∈ ℝ^d
              │
              ▼
    ┌──────────────────────────┐
    │  Cross-Attention         │
    │  Q = learnable query     │
    │  K = V = support set     │
    └──────────┬───────────────┘
               │ (B, 1, d)
               ▼
    LayerNorm + squeeze → prototype ∈ ℝ^d
    + attention weights ∈ ℝ^K

References:
    Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentivePrototype(nn.Module):
    """
    Compute an attention-weighted prototype from a support set.

    Args:
        embedding_dim: Dimensionality of input embeddings.
        num_heads:     Number of attention heads.
        dropout:       Dropout inside attention.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Learnable query token — represents "what is the canonical style?"
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, support_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            support_embeddings: ``(B, K, D)`` — *K* support embeddings per
                                example in the batch.

        Returns:
            prototype:         ``(B, D)`` — attention-weighted prototype.
            attention_weights: ``(B, K)`` — weight assigned to each support
                               image (sums to 1).
        """
        B, K, D = support_embeddings.shape

        # Expand query for batch
        query = self.query.expand(B, -1, -1)              # (B, 1, D)

        # Cross-attention: query attends to support set
        attended, attn_weights = self.cross_attention(
            query, support_embeddings, support_embeddings
        )   # attended: (B, 1, D),  attn_weights: (B, 1, K)

        prototype = self.norm(attended.squeeze(1))         # (B, D)
        attn_weights = attn_weights.squeeze(1)             # (B, K)

        return prototype, attn_weights

    def compute_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for episodic training.

        Given a flat tensor of support embeddings and their labels, group
        by class and compute one prototype per class.

        Args:
            embeddings: ``(N_way * K_shot, D)``
            labels:     ``(N_way * K_shot,)`` — class indices in ``[0, N_way)``
            n_way:      Number of classes.
            k_shot:     Number of support samples per class.

        Returns:
            prototypes:  ``(N_way, D)``
            all_weights: ``(N_way, K_shot)``
        """
        D = embeddings.size(1)
        device = embeddings.device

        # Group embeddings by class
        support_sets = torch.zeros(n_way, k_shot, D, device=device)
        for c in range(n_way):
            mask = labels == c
            support_sets[c] = embeddings[mask][:k_shot]

        # Compute prototypes via attention
        prototypes, weights = self.forward(support_sets)   # (N_way, D), (N_way, K)
        return prototypes, weights

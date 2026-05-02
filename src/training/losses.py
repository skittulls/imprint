"""
Loss functions for joint metric + prototypical training.

Three losses are provided:

1. ``TripletLoss`` — standard triplet margin loss that works with
   pre-mined triplets from ``miners.py``.

2. ``PrototypicalLoss`` — computes prototypes from a support set within
   the batch and classifies query samples via distance to prototypes.
   Uses the ``AttentivePrototype`` head (Novelty 4).

3. ``JointLoss`` — weighted combination of triplet and prototypical losses,
   enabling multi-task training from a single balanced batch.

References:
    Schroff et al., "FaceNet: A Unified Embedding for Face Recognition
    and Clustering", CVPR 2015.
    Snell et al., "Prototypical Networks for Few-shot Learning",
    NeurIPS 2017.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.prototypical import AttentivePrototype
from src.training.miners import mine_hard_triplets, mine_semi_hard_triplets


class TripletLoss(nn.Module):
    """
    Triplet margin loss with integrated online mining.

    The loss for a single triplet (a, p, n) is::

        L = max(0, ||a - p||² - ||a - n||² + margin)

    Args:
        margin:   Margin for the triplet loss.
        mining:   Mining strategy — ``"hard"`` or ``"semi_hard"``.
    """

    def __init__(self, margin: float = 0.3, mining: str = "semi_hard") -> None:
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: ``(N, D)`` — batch of L2-normalised embeddings.
            labels:     ``(N,)``   — class labels.

        Returns:
            Scalar loss.  Returns ``0`` if no valid triplets are found.
        """
        if self.mining == "hard":
            a, p, n = mine_hard_triplets(embeddings, labels)
        else:
            a, p, n = mine_semi_hard_triplets(embeddings, labels, self.margin)

        if a.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return self.loss_fn(a, p, n)


class PrototypicalLoss(nn.Module):
    """
    Few-shot prototypical loss computed within a balanced batch.

    Given a batch of P classes × K samples, splits each class into
    *support* (first ``k_shot`` samples) and *query* (remaining samples).
    Prototypes are computed from supports via ``AttentivePrototype``,
    and queries are classified by nearest-prototype.

    Args:
        prototype_head: ``AttentivePrototype`` module.
        k_shot:         Number of support images per class.
        temperature:    Softmax temperature for distance-based logits.
                        Lower values → sharper distributions.
    """

    def __init__(
        self,
        prototype_head: AttentivePrototype,
        k_shot: int = 5,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.prototype_head = prototype_head
        self.k_shot = k_shot
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            embeddings: ``(P * K, D)`` — balanced batch.
            labels:     ``(P * K,)``   — class labels.

        Returns:
            loss:     Scalar cross-entropy loss.
            accuracy: Float, classification accuracy on query set.
        """
        device = embeddings.device
        unique_classes = torch.unique(labels)
        n_way = len(unique_classes)

        if n_way < 2:
            return (
                torch.tensor(0.0, device=device, requires_grad=True),
                0.0,
            )

        D = embeddings.size(1)
        k = self.k_shot

        support_list = []
        query_embs = []
        query_labels = []

        for i, cls in enumerate(unique_classes):
            mask = labels == cls
            cls_embs = embeddings[mask]
            n_samples = cls_embs.size(0)

            if n_samples <= k:
                # Not enough for support + query; use all as support,
                # duplicate one as query
                support_list.append(cls_embs)
                query_embs.append(cls_embs[-1:])
                query_labels.append(i)
            else:
                support_list.append(cls_embs[:k])
                query_embs.append(cls_embs[k:])
                query_labels.extend([i] * (n_samples - k))

        # Pad supports to same K and stack → (n_way, k, D)
        max_k = max(s.size(0) for s in support_list)
        support_padded = torch.zeros(n_way, max_k, D, device=device)
        for i, s in enumerate(support_list):
            support_padded[i, : s.size(0)] = s
            # Repeat last embedding to fill padding (better than zeros)
            if s.size(0) < max_k:
                support_padded[i, s.size(0) :] = s[-1:].expand(
                    max_k - s.size(0), -1
                )

        # Compute prototypes via attentive aggregation
        prototypes, _ = self.prototype_head(support_padded)  # (n_way, D)

        # Stack queries
        queries = torch.cat(query_embs, dim=0)               # (Q_total, D)
        q_labels = torch.tensor(query_labels, device=device)  # (Q_total,)

        # Distance-based logits: negative squared Euclidean distance
        # queries: (Q, D),  prototypes: (N, D)
        dists = torch.cdist(queries, prototypes, p=2).pow(2)  # (Q, N)
        logits = -dists / self.temperature

        loss = F.cross_entropy(logits, q_labels)

        # Accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == q_labels).float().mean().item()

        return loss, accuracy


class JointLoss(nn.Module):
    """
    Weighted combination of triplet and prototypical losses.

    Enables simultaneous metric learning (global embedding quality) and
    few-shot learning (prototype-based generalisation).

    Args:
        triplet_loss:  ``TripletLoss`` module.
        proto_loss:    ``PrototypicalLoss`` module.
        triplet_weight: Weight for triplet loss term.
        proto_weight:   Weight for prototypical loss term.
    """

    def __init__(
        self,
        triplet_loss: TripletLoss,
        proto_loss: PrototypicalLoss,
        triplet_weight: float = 1.0,
        proto_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.triplet_loss = triplet_loss
        self.proto_loss = proto_loss
        self.triplet_weight = triplet_weight
        self.proto_weight = proto_weight

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor | float]:
        """
        Args:
            embeddings: ``(N, D)``
            labels:     ``(N,)``

        Returns:
            Dictionary with ``total_loss``, ``triplet_loss``,
            ``proto_loss``, and ``proto_accuracy``.
        """
        t_loss = self.triplet_loss(embeddings, labels)
        p_loss, p_acc = self.proto_loss(embeddings, labels)

        total = self.triplet_weight * t_loss + self.proto_weight * p_loss

        return {
            "total_loss": total,
            "triplet_loss": t_loss,
            "proto_loss": p_loss,
            "proto_accuracy": p_acc,
        }

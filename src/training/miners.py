"""
Online triplet mining strategies.

In metric learning, the quality of training depends heavily on **which**
triplets the model sees.  Random sampling produces mostly trivial triplets
that contribute near-zero gradient, wasting compute.

We implement two mining strategies from Hermans et al. (2017):

    • **Hard mining** — for each anchor, pick the hardest positive (farthest
      same-class) and hardest negative (closest different-class).  Maximises
      gradient signal but can cause training collapse early on.

    • **Semi-hard mining** — pick negatives that are *closer than the
      positive but still outside the margin*.  This is the sweet spot:
      informative gradients without collapse risk.

All miners operate **online** within a batch structured as P classes × K
samples (from ``BalancedBatchSampler``).

Reference:
    Hermans et al., "In Defense of the Triplet Loss for Person
    Re-Identification", arXiv 1703.07737, 2017.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distance matrix.

    Args:
        embeddings: ``(N, D)`` — L2-normalised embeddings.

    Returns:
        ``(N, N)`` distance matrix.
    """
    # For L2-normalised vectors: ||a-b||^2 = 2 - 2*a·b
    dot = embeddings @ embeddings.t()
    sq_norms = torch.diag(dot)
    distances = sq_norms.unsqueeze(0) - 2.0 * dot + sq_norms.unsqueeze(1)
    return torch.clamp(distances, min=0.0)


def mine_hard_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mine the single hardest positive and hardest negative for each anchor.

    Args:
        embeddings: ``(N, D)``
        labels:     ``(N,)``

    Returns:
        ``(anchors, positives, negatives)`` — each ``(num_triplets, D)``
    """
    dist_mat = _pairwise_distances(embeddings)
    N = embeddings.size(0)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N) bool
    labels_ne = ~labels_eq

    anchors, positives, negatives = [], [], []

    for i in range(N):
        # Hardest positive: same class, maximum distance
        pos_mask = labels_eq[i].clone()
        pos_mask[i] = False  # exclude self
        if not pos_mask.any():
            continue
        hardest_pos_idx = torch.argmax(dist_mat[i] * pos_mask.float())

        # Hardest negative: different class, minimum distance
        neg_mask = labels_ne[i]
        if not neg_mask.any():
            continue
        # Set same-class distances to infinity so argmin ignores them
        neg_dists = dist_mat[i].clone()
        neg_dists[~neg_mask] = float("inf")
        hardest_neg_idx = torch.argmin(neg_dists)

        anchors.append(embeddings[i])
        positives.append(embeddings[hardest_pos_idx])
        negatives.append(embeddings[hardest_neg_idx])

    if len(anchors) == 0:
        # Fallback: return empty tensors with correct shape
        D = embeddings.size(1)
        empty = torch.zeros(0, D, device=embeddings.device)
        return empty, empty, empty

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def mine_semi_hard_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mine semi-hard negatives: negatives that are farther than the positive
    but closer than positive + margin.

    This produces the most informative gradients without the instability
    of pure hard mining.

    If no semi-hard negative exists for an anchor, falls back to the
    hardest negative (closest different-class sample).

    Args:
        embeddings: ``(N, D)``
        labels:     ``(N,)``
        margin:     Triplet margin.

    Returns:
        ``(anchors, positives, negatives)`` — each ``(num_triplets, D)``
    """
    dist_mat = _pairwise_distances(embeddings)
    N = embeddings.size(0)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_ne = ~labels_eq

    anchors, positives, negatives = [], [], []

    for i in range(N):
        pos_mask = labels_eq[i].clone()
        pos_mask[i] = False
        if not pos_mask.any():
            continue

        neg_mask = labels_ne[i]
        if not neg_mask.any():
            continue

        # Pick a random positive
        pos_indices = torch.where(pos_mask)[0]
        pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
        d_ap = dist_mat[i, pos_idx]

        # Semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin
        neg_dists = dist_mat[i]
        semi_hard_mask = neg_mask & (neg_dists > d_ap) & (neg_dists < d_ap + margin)

        if semi_hard_mask.any():
            # Pick the closest semi-hard negative
            sh_dists = neg_dists.clone()
            sh_dists[~semi_hard_mask] = float("inf")
            neg_idx = torch.argmin(sh_dists)
        else:
            # Fallback to hardest negative
            neg_dists_masked = neg_dists.clone()
            neg_dists_masked[~neg_mask] = float("inf")
            neg_idx = torch.argmin(neg_dists_masked)

        anchors.append(embeddings[i])
        positives.append(embeddings[pos_idx.squeeze()])
        negatives.append(embeddings[neg_idx])

    if len(anchors) == 0:
        D = embeddings.size(1)
        empty = torch.zeros(0, D, device=embeddings.device)
        return empty, empty, empty

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

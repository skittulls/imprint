"""
Evaluation metrics for style attribution.

Provides three categories of metrics:

1. **Verification** (pairwise) — ROC-AUC, Average Precision, EER.
   "Given two images, are they by the same artist?"

2. **Retrieval** — Recall@K, Mean Average Precision (mAP).
   "Given a query, are the K nearest neighbours from the same artist?"

3. **Few-shot** — N-way K-shot classification accuracy.
   "Given K examples per artist, can we classify a new work?"

All functions are designed to operate on pre-computed embeddings so that
evaluation is fast and decoupled from the model forward pass.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# 1. Verification metrics (pairwise)
# ---------------------------------------------------------------------------

def compute_verification_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_pairs: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Sample random pairs, compute cosine similarity, and evaluate
    binary verification performance.

    Args:
        embeddings: ``(N, D)`` — pre-computed embeddings.
        labels:     ``(N,)``   — artist labels.
        num_pairs:  Number of pairs to sample.
        seed:       Random seed for reproducibility.

    Returns:
        Dictionary with ``roc_auc``, ``average_precision``, and ``eer``.
    """
    rng = np.random.RandomState(seed)
    N = embeddings.size(0)
    emb_np = embeddings.cpu().numpy()
    lbl_np = labels.cpu().numpy()

    idx_a = rng.randint(0, N, size=num_pairs)
    idx_b = rng.randint(0, N, size=num_pairs)

    # Avoid self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    # Cosine similarity (embeddings are L2-normalised → dot product)
    sims = np.sum(emb_np[idx_a] * emb_np[idx_b], axis=1)
    ground_truth = (lbl_np[idx_a] == lbl_np[idx_b]).astype(int)

    # Metrics
    auc = roc_auc_score(ground_truth, sims)
    ap = average_precision_score(ground_truth, sims)

    # Equal Error Rate
    fpr, tpr, _ = roc_curve(ground_truth, sims)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])

    return {"roc_auc": auc, "average_precision": ap, "eer": eer}


# ---------------------------------------------------------------------------
# 2. Retrieval metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Compute Recall@K and mean Average Precision for retrieval.

    For each image as a query, rank all other images by embedding distance
    and check if the top-K results share the same artist.

    Args:
        embeddings: ``(N, D)``
        labels:     ``(N,)``
        k_values:   List of K values for Recall@K.

    Returns:
        Dictionary with ``recall@K`` for each K and ``mAP``.
    """
    N = embeddings.size(0)
    device = embeddings.device

    # Similarity matrix (cosine, since embeddings are L2-normalised)
    sim_matrix = embeddings @ embeddings.t()                  # (N, N)
    sim_matrix.fill_diagonal_(-float("inf"))                  # exclude self

    label_match = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N) bool

    # Sort by descending similarity
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)

    results: Dict[str, float] = {}

    # Recall@K
    for k in k_values:
        top_k = sorted_indices[:, :k]
        # For each query, check if any of top-K are same class
        hits = torch.gather(label_match, 1, top_k).any(dim=1).float()
        results[f"recall@{k}"] = hits.mean().item()

    # Mean Average Precision
    sorted_matches = torch.gather(
        label_match, 1, sorted_indices
    ).float()  # (N, N)
    
    # Drop the self-item which is at the very end (due to -inf diagonal)
    sorted_matches = sorted_matches[:, :-1]  # (N, N-1)
    
    cum_correct = sorted_matches.cumsum(dim=1)
    ranks = torch.arange(1, N, device=device).float().unsqueeze(0)
    precisions = cum_correct / ranks
    ap_per_query = (precisions * sorted_matches).sum(dim=1) / sorted_matches.sum(
        dim=1
    ).clamp(min=1)
    results["mAP"] = ap_per_query.mean().item()

    return results


# ---------------------------------------------------------------------------
# 3. Few-shot evaluation
# ---------------------------------------------------------------------------

def compute_fewshot_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototype_head,
    n_way: int = 5,
    k_shot: int = 5,
    num_episodes: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate few-shot classification accuracy via episodic evaluation.

    Each episode:
      1. Sample ``n_way`` classes.
      2. For each class, pick ``k_shot`` support + 1 query.
      3. Compute prototype from supports (via ``prototype_head``).
      4. Classify query by nearest prototype.

    Args:
        embeddings:     ``(N, D)``
        labels:         ``(N,)``
        prototype_head: ``AttentivePrototype`` module.
        n_way:          Number of classes per episode.
        k_shot:         Number of support samples per class.
        num_episodes:   Number of evaluation episodes.
        seed:           Random seed.

    Returns:
        Dictionary with ``accuracy``, ``accuracy_std``, and
        ``accuracy_95ci``.
    """
    rng = np.random.RandomState(seed)
    unique_classes = torch.unique(labels).cpu().numpy()
    D = embeddings.size(1)
    device = embeddings.device

    # Pre-build index
    class_indices = {}
    labels_np = labels.cpu().numpy()
    for cls in unique_classes:
        class_indices[cls] = np.where(labels_np == cls)[0]

    # Filter to classes with enough samples
    eligible = [c for c in unique_classes if len(class_indices[c]) >= k_shot + 1]
    if len(eligible) < n_way:
        return {"accuracy": 0.0, "accuracy_std": 0.0, "accuracy_95ci": 0.0}

    accs = []
    prototype_head.eval()

    with torch.no_grad():
        for _ in range(num_episodes):
            chosen = rng.choice(eligible, n_way, replace=False)
            support_embs = torch.zeros(n_way, k_shot, D, device=device)
            query_embs = []
            query_labels = []

            for i, cls in enumerate(chosen):
                pool = class_indices[cls]
                selected = rng.choice(pool, k_shot + 1, replace=False)
                s_idx = selected[:k_shot]
                q_idx = selected[k_shot:]
                support_embs[i] = embeddings[s_idx]
                query_embs.append(embeddings[q_idx])
                query_labels.extend([i] * len(q_idx))

            # Prototypes
            prototypes, _ = prototype_head(support_embs)   # (n_way, D)

            queries = torch.cat(query_embs, dim=0)          # (Q, D)
            q_labels = torch.tensor(query_labels, device=device)

            dists = torch.cdist(queries, prototypes, p=2)
            preds = dists.argmin(dim=1)
            acc = (preds == q_labels).float().mean().item()
            accs.append(acc)

    accs = np.array(accs)
    mean = accs.mean()
    std = accs.std()
    ci95 = 1.96 * std / np.sqrt(len(accs))

    return {"accuracy": mean, "accuracy_std": std, "accuracy_95ci": ci95}

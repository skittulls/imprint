"""
Visualisation utilities for embedding spaces and attention analysis.

Provides:
    - UMAP embedding projections coloured by artist
    - Gram attention weight bar charts
    - Training curve plots

All functions save figures to disk and return the matplotlib Figure
for optional interactive display.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_umap_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    label_names: Optional[Dict[int, str]] = None,
    save_path: str = "outputs/umap_embeddings.png",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "Style Embedding Space (UMAP)",
) -> plt.Figure:
    """
    Project embeddings to 2D with UMAP and plot coloured by artist.

    Args:
        embeddings: ``(N, D)``
        labels:     ``(N,)``
        label_names: Optional mapping from label int to artist name.
        save_path:  Where to save the figure.
        n_neighbors: UMAP parameter.
        min_dist:    UMAP parameter.
        title:       Plot title.

    Returns:
        matplotlib Figure.
    """
    from umap import UMAP

    emb_np = embeddings.cpu().numpy()
    lbl_np = labels.cpu().numpy()

    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    proj = reducer.fit_transform(emb_np)

    fig, ax = plt.subplots(figsize=(12, 10))
    unique = np.unique(lbl_np)
    cmap = plt.cm.get_cmap("tab20", len(unique))

    for i, cls in enumerate(unique):
        mask = lbl_np == cls
        name = label_names.get(cls, str(cls)) if label_names else str(cls)
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[cmap(i)], label=name,
                   s=15, alpha=0.7)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if len(unique) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_attention_weights(
    weights: torch.Tensor,
    scale_names: List[str] = ["layer2\n(texture)", "layer3\n(motifs)", "layer4\n(composition)"],
    save_path: str = "outputs/attention_weights.png",
    title: str = "Gram Scale Attention Weights",
) -> plt.Figure:
    """
    Bar chart of learned attention weights across scales.

    Args:
        weights: ``(num_scales,)`` — softmaxed scale weights.
        scale_names: Human-readable names for each scale.
        save_path: Where to save the figure.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    w = weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(scale_names, w, color=["#4fc3f7", "#7986cb", "#e57373"],
                  edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, w):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, max(w) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: str = "outputs/training_curves.png",
    title: str = "Training Curves",
) -> plt.Figure:
    """
    Plot loss and accuracy curves over epochs.

    Args:
        metrics: Dictionary mapping metric names to per-epoch values.
                 e.g. ``{"train_loss": [...], "val_loss": [...], ...}``
        save_path: Where to save the figure.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    loss_keys = [k for k in metrics if "loss" in k]
    acc_keys = [k for k in metrics if "acc" in k]
    has_acc = len(acc_keys) > 0

    fig, axes = plt.subplots(1, 1 + int(has_acc), figsize=(7 * (1 + int(has_acc)), 5))
    if not has_acc:
        axes = [axes]

    # Loss curves
    for key in loss_keys:
        axes[0].plot(metrics[key], label=key, linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    if has_acc:
        for key in acc_keys:
            axes[1].plot(metrics[key], label=key, linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curves")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig

"""
Custom batch samplers for metric learning.

``BalancedBatchSampler`` ensures that every mini-batch contains exactly
*P* classes with *K* samples each.  This structure is **required** for
online triplet mining (Hermans et al., "In Defense of the Triplet Loss
for Person Re-Identification", arXiv 1703.07737) and also allows us to
construct few-shot episodes within the same batch for prototypical training.

Usage with ``DataLoader``::

    sampler = BalancedBatchSampler(dataset.labels, P=10, K=8)
    loader  = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[List[int]]):
    """
    Yields batches of size ``P × K`` where every batch contains exactly
    *P* distinct classes, each represented by *K* samples.

    If a class has fewer than *K* images, samples are drawn **with
    replacement**.  Classes are shuffled each epoch so that different
    combinations appear across epochs.

    Args:
        labels:  Per-sample class labels (contiguous integers).
        P:       Number of classes per batch.
        K:       Number of samples per class per batch.
    """

    def __init__(self, labels: list[int], P: int = 10, K: int = 8) -> None:
        super().__init__()
        self.P = P
        self.K = K
        self.batch_size = P * K

        # Build class → sample-index mapping
        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.label_to_indices[lbl].append(idx)

        # Only keep classes with at least 2 samples (minimum for positive pairs)
        self.classes = [c for c, idxs in self.label_to_indices.items() if len(idxs) >= 2]
        if len(self.classes) < P:
            raise ValueError(
                f"Need at least P={P} classes with ≥2 samples, "
                f"but only found {len(self.classes)}."
            )

        self.n_batches = len(labels) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_batches):
            selected_classes = random.sample(self.classes, self.P)
            batch_indices: List[int] = []
            for cls in selected_classes:
                pool = self.label_to_indices[cls]
                if len(pool) >= self.K:
                    chosen = random.sample(pool, self.K)
                else:
                    # Sample with replacement for small classes
                    chosen = [random.choice(pool) for _ in range(self.K)]
                batch_indices.extend(chosen)
            yield batch_indices

    def __len__(self) -> int:
        return self.n_batches

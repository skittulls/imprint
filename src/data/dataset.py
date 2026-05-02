"""
Dataset and data-preparation utilities.

``StyleDataset`` is intentionally simple — it returns ``(image, artist_id)``
tuples.  All pair/triplet construction happens *online* inside the loss
functions and miners, which is the modern best-practice for metric learning
(see ``pytorch-metric-learning`` library design).

Stochastic pair generation (as in v1) is replaced by
``BalancedBatchSampler`` + online mining, which guarantees that every batch
contains enough same-class examples for effective triplet construction.

Data preparation (CSV loading, splitting) is also provided here so that the
full pipeline lives in one place.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import Compose


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class StyleDataset(Dataset):
    """
    Minimal dataset: returns ``(image_tensor, artist_id)`` pairs.

    The ``artist_id`` is a **contiguous** integer in ``[0, num_classes)``
    regardless of the original IDs in the CSV.  This is critical for the
    balanced batch sampler and prototypical loss.

    Attributes:
        labels:  List[int] — contiguous label for every sample (needed by
                 ``BalancedBatchSampler``).
        label_to_indices: Dict[int, List[int]] — fast lookup of sample
                          indices per class.
        num_classes: int — total number of unique artists.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Compose] = None,
    ) -> None:
        """
        Args:
            df: DataFrame with columns ``['path', 'artist_name', 'artist_id']``.
            transform: torchvision transforms to apply to each image.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # Build contiguous label mapping: original_id → 0..C-1
        unique_ids = sorted(self.df["artist_id"].unique())
        self._id_map = {orig: new for new, orig in enumerate(unique_ids)}
        self.df["label"] = self.df["artist_id"].map(self._id_map)

        # Public attributes consumed by sampler / evaluation
        self.labels: list[int] = self.df["label"].tolist()
        self.num_classes: int = len(unique_ids)

        self.label_to_indices: dict[int, list[int]] = {}
        for idx, lbl in enumerate(self.labels):
            self.label_to_indices.setdefault(lbl, []).append(idx)

    # ---- torch Dataset interface -----------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


# ---------------------------------------------------------------------------
# Data preparation helpers (reused from v1, cleaned up)
# ---------------------------------------------------------------------------

def load_artist_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load ``artists.csv`` from the *Best Artworks of All Time* Kaggle dataset.

    Returns:
        DataFrame with columns ``['name', 'artist_id', 'paintings']``.
    """
    df = pd.read_csv(csv_path)
    if "id" in df.columns:
        df["artist_id"] = df["id"]
    else:
        df = df.reset_index()
        df["artist_id"] = df["index"]
    return df[["name", "artist_id", "paintings"]]


def build_image_index(
    images_dir: str,
    df_artists: pd.DataFrame,
    min_paintings: int = 0,
) -> pd.DataFrame:
    """
    Walk *images_dir*, match each subfolder to an artist, and return an
    image-level DataFrame with columns ``['path', 'artist_name', 'artist_id']``.
    """
    records: list[dict] = []
    images_path = Path(images_dir)

    for folder in sorted(images_path.iterdir()):
        if not folder.is_dir():
            continue
        artist_name = folder.name.replace("_", " ")
        match = df_artists[
            df_artists["name"].str.strip().str.lower() == artist_name.strip().lower()
        ]
        if match.empty:
            token = folder.name.split("_")[0]
            match = df_artists[
                df_artists["name"].str.contains(token, case=False, na=False)
            ]
            if match.empty:
                continue

        artist_id = int(match.iloc[0]["artist_id"])
        artist_name_clean = match.iloc[0]["name"]
        img_files = (
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.png"))
            + list(folder.glob("*.jpeg"))
        )
        if len(img_files) < min_paintings:
            continue
        for p in img_files:
            records.append(
                {"path": str(p), "artist_name": artist_name_clean, "artist_id": artist_id}
            )

    df = pd.DataFrame(records)
    print(f"✓ Found {len(df)} images from {df['artist_id'].nunique()} artists")
    return df


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Artist-stratified train / val / test split.

    Artists with fewer than 3 images are removed because ``train_test_split``
    with ``stratify`` requires at least 2 members in every class per split.
    """
    counts = df["artist_id"].value_counts()
    valid = counts[counts >= 3].index
    df_f = df[df["artist_id"].isin(valid)].copy()
    if len(df_f) < len(df):
        print(f"  ⚠ Removed {len(df) - len(df_f)} images from tiny artists")

    train_df, temp_df = train_test_split(
        df_f,
        test_size=val_size + test_size,
        stratify=df_f["artist_id"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df["artist_id"],
        random_state=seed,
    )
    print(f"✓ Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df

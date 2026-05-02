"""
Data preparation script for Imprint v2.

Reads the raw Kaggle dataset, builds an image index, creates stratified
splits, and saves them as CSVs.

Usage::

    cd imprint_v2
    python scripts/prepare_data.py
    # or with custom config:
    python scripts/prepare_data.py --config configs/default.yaml
"""

import argparse
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import build_image_index, load_artist_metadata, split_dataset
from src.utils.helpers import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data splits.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    d = cfg["data"]
    output_dir = d["splits_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1: Loading artist metadata...")
    df_artists = load_artist_metadata(d["artists_csv"])

    print("\nStep 2: Scanning images and building index...")
    df_full = build_image_index(
        d["images_dir"], df_artists, min_paintings=d.get("min_paintings", 50)
    )

    print("\nStep 3: Creating train/val/test splits...")
    train_df, val_df, test_df = split_dataset(
        df_full,
        val_size=d.get("val_size", 0.15),
        test_size=d.get("test_size", 0.15),
        seed=cfg.get("seed", 42),
    )

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\n✓ Data preparation complete! Files saved to {output_dir}")


if __name__ == "__main__":
    main()

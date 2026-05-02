"""
Full evaluation pipeline for Imprint v2.

Loads a trained checkpoint, computes embeddings for the test set, and
reports verification, retrieval, and few-shot metrics. Also generates
UMAP and attention visualisations.

Usage::

    cd imprint_v2
    python scripts/evaluate.py --checkpoint checkpoints/last.ckpt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import StyleDataset
from src.data.transforms import get_eval_transforms
from src.evaluation.metrics import (
    compute_fewshot_accuracy,
    compute_retrieval_metrics,
    compute_verification_metrics,
)
from src.evaluation.visualize import plot_attention_weights, plot_umap_embeddings
from src.training.lightning_module import ImprintModule
from src.utils.helpers import load_config, set_seed


@torch.no_grad()
def extract_embeddings(
    model: ImprintModule,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract embeddings and attention weights for all images.

    Returns:
        embeddings:        ``(N, D)``
        labels:            ``(N,)``
        attention_weights: ``(N, num_scales)``
    """
    model.eval()
    all_emb, all_lbl, all_attn = [], [], []

    for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
        images = images.to(device)
        out = model.encoder(images)
        all_emb.append(out.embedding.cpu())
        all_lbl.append(labels)
        all_attn.append(out.attention_weights.cpu())

    return torch.cat(all_emb), torch.cat(all_lbl), torch.cat(all_attn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Imprint v2.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load model ----------------------------------------------------
    model = ImprintModule.from_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded from checkpoint.")

    # ---- Load test data ------------------------------------------------
    d_cfg = cfg["data"]
    image_size = d_cfg.get("image_size", 224)
    test_df = pd.read_csv(os.path.join(d_cfg["splits_dir"], "test.csv"))
    test_ds = StyleDataset(test_df, transform=get_eval_transforms(image_size))
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"Test set: {len(test_ds)} images, {test_ds.num_classes} classes")

    # ---- Extract embeddings --------------------------------------------
    embeddings, labels, attn_weights = extract_embeddings(model, test_loader, device)

    # ---- 1. Verification metrics ---------------------------------------
    print("\n" + "=" * 50)
    print("VERIFICATION METRICS (pairwise)")
    print("=" * 50)
    verif = compute_verification_metrics(embeddings, labels)
    for k, v in verif.items():
        print(f"  {k:25s}: {v:.4f}")

    # ---- 2. Retrieval metrics ------------------------------------------
    print("\n" + "=" * 50)
    print("RETRIEVAL METRICS")
    print("=" * 50)
    e_cfg = cfg.get("evaluation", {})
    retrieval = compute_retrieval_metrics(
        embeddings, labels, k_values=e_cfg.get("recall_k", [1, 3, 5, 10])
    )
    for k, v in retrieval.items():
        print(f"  {k:25s}: {v:.4f}")

    # ---- 3. Few-shot metrics -------------------------------------------
    print("\n" + "=" * 50)
    print("FEW-SHOT METRICS")
    print("=" * 50)
    proto_head = model.joint_loss.proto_loss.prototype_head.to(device)
    embeddings_dev = embeddings.to(device)
    labels_dev = labels.to(device)

    for k in e_cfg.get("few_shot_k", [1, 3, 5]):
        fs = compute_fewshot_accuracy(
            embeddings_dev, labels_dev, proto_head,
            n_way=e_cfg.get("eval_n_way", 5),
            k_shot=k,
            num_episodes=e_cfg.get("num_eval_episodes", 500),
        )
        print(f"  {k}-shot accuracy: {fs['accuracy']:.4f} ± {fs['accuracy_95ci']:.4f}")

    # ---- Visualisations ------------------------------------------------
    print("\nGenerating visualisations...")

    # Build label name mapping
    label_names = {}
    for _, row in test_df.drop_duplicates("artist_id").iterrows():
        ds_label = test_ds._id_map.get(row["artist_id"])
        if ds_label is not None:
            label_names[ds_label] = row["artist_name"]

    plot_umap_embeddings(
        embeddings, labels, label_names=label_names,
        save_path=os.path.join(args.output_dir, "umap_embeddings.png"),
    )

    # Average attention weights across test set
    mean_attn = attn_weights.mean(dim=0)
    plot_attention_weights(
        mean_attn,
        save_path=os.path.join(args.output_dir, "attention_weights.png"),
    )

    # ---- Save results JSON ---------------------------------------------
    all_results = {"verification": verif, "retrieval": retrieval}
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    print(f"✓ Visualisations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

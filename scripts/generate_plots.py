"""
Generates the ROC Curve (Figure 5.3) and Few-Shot Accuracy vs. K-Shot (Figure 5.4) plots.
Saves them to outputs/roc_curve.png and outputs/few_shot_accuracy.png.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import StyleDataset
from src.data.transforms import get_eval_transforms
from src.training.lightning_module import ImprintModule
from src.utils.helpers import load_config, set_seed

@torch.no_grad()
def main():
    cfg = load_config("configs/default.yaml")
    set_seed(cfg.get("seed", 42))
    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = ImprintModule.from_config(cfg)
    checkpoint_path = "checkpoints/best.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint {checkpoint_path} not found.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    # Load test data
    d_cfg = cfg["data"]
    image_size = d_cfg.get("image_size", 224)
    test_df = pd.read_csv(os.path.join(d_cfg["splits_dir"], "test.csv"))
    test_ds = StyleDataset(test_df, transform=get_eval_transforms(image_size))
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    # Extract embeddings
    all_emb, all_lbl = [], []
    for images, labels in test_loader:
        images = images.to(device)
        out = model.encoder(images)
        all_emb.append(out.embedding.cpu())
        all_lbl.append(labels)
    embeddings = torch.cat(all_emb).numpy()
    labels = torch.cat(all_lbl).numpy()

    # ---- Plot 1: ROC Curve ----
    print("Generating ROC Curve...")
    rng = np.random.RandomState(42)
    num_pairs = 10000
    N = len(embeddings)
    idx_a = rng.randint(0, N, size=num_pairs)
    idx_b = rng.randint(0, N, size=num_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    sims = np.sum(embeddings[idx_a] * embeddings[idx_b], axis=1)
    ground_truth = (labels[idx_a] == labels[idx_b]).astype(int)

    fpr, tpr, _ = roc_curve(ground_truth, sims)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#7986cb', lw=2, label=f'Imprint ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#e57373', lw=1, linestyle='--', label='Random Baseline (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved outputs/roc_curve.png")

    # ---- Plot 2: Few-Shot Accuracy ----
    print("Generating Few-Shot Accuracy chart...")
    # Accuracy values from evaluate.py run
    shots = ['1-Shot', '3-Shot', '5-Shot']
    accuracies = [61.44, 67.72, 68.92]
    cis = [1.97, 1.86, 1.69]  # 95% confidence intervals

    plt.figure(figsize=(7, 5))
    bars = plt.bar(shots, accuracies, yerr=cis, color=['#81c784', '#64b5f6', '#ffd54f'], 
                   edgecolor='grey', capsize=8, width=0.5)
    
    # Add values on top of bars
    for bar, acc, ci in zip(bars, accuracies, cis):
        plt.text(bar.get_x() + bar.get_width()/2, acc + ci + 1, 
                 f"{acc:.2f}% ± {ci:.2f}%", ha='center', fontsize=10, fontweight='bold')

    plt.ylabel('Classification Accuracy (%)', fontsize=11)
    plt.title('5-Way Few-Shot Classification Accuracy vs. K-Shot', fontsize=13, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("outputs/few_shot_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved outputs/few_shot_accuracy.png")

if __name__ == "__main__":
    main()

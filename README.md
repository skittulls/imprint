# Imprint v2 — Multi-Scale Gram Attention + Few-Shot Prototypical Style Attribution

A research-grade style attribution system that measures **stylistic similarity** between artworks using two novel contributions:

1. **MSGMAtt (Multi-Scale Gram Matrix Attention)** — Extracts texture correlations at three CNN depths via Gram matrices, then uses learned self-attention to weight each scale's contribution. Produces an interpretable style fingerprint that reveals *whether similarity is driven by brushstroke texture (fine scale) or compositional structure (coarse scale)*.

2. **Attentive Prototypical Verification** — Replaces naive mean-prototype with cross-attention aggregation over support embeddings. Works with just 3–5 reference images from artists **never seen during training**, enabling open-set style verification.

## Architecture

```
Input image [B, 3, 224, 224]
     │
     ▼
┌─────────────────────────┐
│  ResNet-18 (pretrained)  │  ← Partial freeze (conv1 through layer2)
└────┬───────┬────────┬───┘
   layer2  layer3   layer4
     │       │        │
     ▼       ▼        ▼
┌─────────────────────────┐    ┌───────────────┐
│  Multi-Scale Gram        │    │  Global Avg    │
│  Attention (MSGMAtt)     │    │  Pool + FC     │
└──────────┬──────────────┘    └───────┬───────┘
           │ (B, 256)                  │ (B, 256)
           └──────────┬───────────────┘
                      │ concat → (B, 512)
                      ▼
              ┌───────────────┐
              │ Projection    │
              │ → L2 normalise│
              └───────┬───────┘
                      │ (B, 128)
                      ▼
                 Style Embedding
```

## Training Strategy

- **Balanced Batches**: P=10 classes × K=8 samples for effective online mining
- **Joint Loss**: Triplet loss (semi-hard mining) + Prototypical loss (attentive aggregation)
- **Optimiser**: AdamW + linear warmup + cosine annealing
- **Mixed Precision**: FP16 for T4 GPU efficiency

## Quick Start

### 1. Prepare data (reuses v1 Kaggle dataset)
```bash
cd imprint_v2
pip install -r requirements.txt
python scripts/prepare_data.py
```

### 2. Train
```bash
python scripts/train.py
```

### 3. Evaluate
```bash
python scripts/evaluate.py --checkpoint checkpoints/last.ckpt
```

## Project Structure

```
imprint_v2/
├── configs/
│   └── default.yaml            # All hyperparameters
├── src/
│   ├── data/
│   │   ├── dataset.py          # StyleDataset with contiguous labels
│   │   ├── transforms.py       # Style-preserving augmentations
│   │   └── sampler.py          # BalancedBatchSampler (P×K)
│   ├── models/
│   │   ├── backbone.py         # Multi-scale ResNet-18
│   │   ├── gram_attention.py   # MSGMAtt (Novelty 1)
│   │   ├── encoder.py          # Full encoder pipeline
│   │   └── prototypical.py     # AttentivePrototype (Novelty 4)
│   ├── training/
│   │   ├── losses.py           # Triplet + Prototypical + Joint
│   │   ├── miners.py           # Hard / Semi-hard mining
│   │   └── lightning_module.py # LightningModule orchestrator
│   ├── evaluation/
│   │   ├── metrics.py          # AUC, Recall@K, Few-shot accuracy
│   │   └── visualize.py        # UMAP, attention plots
│   └── utils/
│       └── helpers.py          # Seed, config, logging
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
└── requirements.txt
```

## Evaluation Metrics

| Category       | Metrics                                  |
|---------------|------------------------------------------|
| Verification  | ROC-AUC, Average Precision, EER          |
| Retrieval     | Recall@1/3/5/10, mAP                     |
| Few-shot      | N-way K-shot accuracy (± 95% CI)         |

## Training on Lightning.ai

1. Create a Studio with a T4 GPU
2. Upload the `imprint_v2/` folder and `data/` directory
3. Install deps: `pip install -r requirements.txt`
4. Run: `python scripts/train.py`

The trainer automatically detects the GPU, enables mixed precision, and logs to `logs/imprint_v2/`.

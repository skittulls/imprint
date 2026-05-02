"""
Main training entry point for Imprint v2.

Uses PyTorch Lightning with:
    - BalancedBatchSampler for effective online triplet mining
    - Joint triplet + prototypical loss
    - Cosine LR schedule with linear warmup
    - ModelCheckpoint + EarlyStopping callbacks
    - CSV logger (works without MLFlow server)

Usage on Lightning.ai::

    cd imprint_v2
    python scripts/train.py
    # or with custom config:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from src.data.dataset import StyleDataset
from src.data.sampler import BalancedBatchSampler
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.training.lightning_module import ImprintModule
from src.utils.helpers import load_config, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Imprint v2.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    logger = setup_logging()

    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    image_size = d_cfg.get("image_size", 224)
    P = t_cfg.get("classes_per_batch", 10)
    K = t_cfg.get("batch_size_per_class", 8)

    # ---- Data ----------------------------------------------------------
    logger.info("Loading data splits...")
    train_df = pd.read_csv(os.path.join(d_cfg["splits_dir"], "train.csv"))
    val_df = pd.read_csv(os.path.join(d_cfg["splits_dir"], "val.csv"))

    train_ds = StyleDataset(train_df, transform=get_train_transforms(image_size))
    val_ds = StyleDataset(val_df, transform=get_eval_transforms(image_size))

    train_sampler = BalancedBatchSampler(train_ds.labels, P=P, K=K)
    val_sampler = BalancedBatchSampler(val_ds.labels, P=P, K=K)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(
        f"Train: {len(train_ds)} images, {train_ds.num_classes} classes | "
        f"Batch: {P} classes × {K} images = {P * K}"
    )

    # ---- Model ---------------------------------------------------------
    model = ImprintModule.from_config(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable:,} trainable")

    # ---- Callbacks -----------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="imprint-v2-{epoch:02d}-{val/total_loss:.4f}",
        save_top_k=2,
        monitor="val/total_loss",
        mode="min",
        verbose=True,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/total_loss",
        patience=10,
        mode="min",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    csv_logger = CSVLogger(save_dir="logs", name="imprint_v2")

    # ---- Trainer -------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=t_cfg.get("epochs", 50),
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=csv_logger,
        log_every_n_steps=10,
        precision="16-mixed",            # AMP for faster T4 training
        gradient_clip_val=1.0,           # Prevent gradient explosion
        deterministic=True,
    )

    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    logger.info(f"✓ Best checkpoint: {checkpoint_cb.best_model_path}")
    logger.info(f"  Best val loss:   {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()

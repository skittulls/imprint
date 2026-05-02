"""
PyTorch Lightning module for Imprint v2.

Orchestrates the full training pipeline:
    1. Forward pass through ``StyleEncoder`` (backbone + MSGMAtt).
    2. Joint loss computation (triplet + prototypical) via ``JointLoss``.
    3. Optimisation with AdamW + cosine annealing + linear warmup.
    4. Structured logging of all loss components and metrics.

The module is designed for ``BalancedBatchSampler`` batches (P × K),
which feed both the triplet miner and the prototypical episode
simultaneously — no separate dataloaders needed.

Usage::

    model = ImprintModule.from_config(cfg)
    trainer = L.Trainer(...)
    trainer.fit(model, train_loader, val_loader)
"""

from __future__ import annotations

from typing import Any, Dict

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.models.encoder import StyleEncoder
from src.models.prototypical import AttentivePrototype
from src.training.losses import JointLoss, PrototypicalLoss, TripletLoss


class ImprintModule(L.LightningModule):
    """
    Lightning module for multi-scale Gram attention + few-shot prototypical
    style attribution.

    Args:
        encoder:    ``StyleEncoder`` instance.
        joint_loss: ``JointLoss`` instance.
        lr:         Base learning rate.
        weight_decay: L2 regularisation strength.
        warmup_epochs: Number of linear warmup epochs.
        max_epochs:    Total training epochs (for cosine schedule).
    """

    def __init__(
        self,
        encoder: StyleEncoder,
        joint_loss: JointLoss,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "joint_loss"])

        self.encoder = encoder
        self.joint_loss = joint_loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        out = self.encoder(images)

        loss_dict = self.joint_loss(out.embedding, labels)

        # Log all components
        self.log("train/total_loss", loss_dict["total_loss"], prog_bar=True)
        self.log("train/triplet_loss", loss_dict["triplet_loss"])
        self.log("train/proto_loss", loss_dict["proto_loss"])
        self.log("train/proto_acc", loss_dict["proto_accuracy"], prog_bar=True)

        # Log scale attention weights for monitoring
        attn = out.attention_weights[0]  # first sample in batch
        for i, name in enumerate(["layer2", "layer3", "layer4"]):
            self.log(f"train/attn_{name}", attn[i])

        return loss_dict["total_loss"]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        images, labels = batch
        out = self.encoder(images)

        loss_dict = self.joint_loss(out.embedding, labels)

        self.log("val/total_loss", loss_dict["total_loss"], prog_bar=True, sync_dist=True)
        self.log("val/triplet_loss", loss_dict["triplet_loss"], sync_dist=True)
        self.log("val/proto_loss", loss_dict["proto_loss"], sync_dist=True)
        self.log("val/proto_acc", loss_dict["proto_accuracy"], prog_bar=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimiser & Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        AdamW with linear warmup followed by cosine annealing.

        Warmup avoids destabilising pretrained backbone weights in early
        epochs.  Cosine annealing provides smooth learning rate decay
        that empirically outperforms step decay for metric learning
        (Musgrave et al., ECCV 2020).
        """
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )

        # Linear warmup
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs,
        )
        # Cosine decay for remaining epochs
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=1e-6,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "ImprintModule":
        """Build the full module from a config dictionary."""
        encoder = StyleEncoder.from_config(cfg)

        t_cfg = cfg["training"]
        m_cfg = cfg["model"]

        triplet_loss = TripletLoss(
            margin=t_cfg.get("triplet_margin", 0.3),
            mining=t_cfg.get("mining_strategy", "semi_hard"),
        )
        proto_head = AttentivePrototype(
            embedding_dim=m_cfg.get("embedding_dim", 128),
            num_heads=m_cfg.get("attention_heads", 4),
        )
        proto_loss = PrototypicalLoss(
            prototype_head=proto_head,
            k_shot=t_cfg.get("proto_k_shot", 5),
            temperature=t_cfg.get("proto_temperature", 0.1),
        )
        joint_loss = JointLoss(
            triplet_loss=triplet_loss,
            proto_loss=proto_loss,
            triplet_weight=t_cfg.get("triplet_weight", 1.0),
            proto_weight=t_cfg.get("proto_weight", 0.5),
        )

        return cls(
            encoder=encoder,
            joint_loss=joint_loss,
            lr=t_cfg.get("lr", 3e-4),
            weight_decay=t_cfg.get("weight_decay", 1e-5),
            warmup_epochs=t_cfg.get("warmup_epochs", 5),
            max_epochs=t_cfg.get("epochs", 50),
        )

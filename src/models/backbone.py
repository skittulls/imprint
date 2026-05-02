"""
Multi-scale feature extraction backbone.

Wraps a pretrained ResNet-18 to return intermediate feature maps from
``layer2``, ``layer3``, and ``layer4``.  These three scales capture
progressively more abstract visual patterns:

    ┌──────────┬───────────┬────────────────────────────────┐
    │ Scale    │ Channels  │ Captures                       │
    ├──────────┼───────────┼────────────────────────────────┤
    │ layer2   │ 128       │ Edges, micro-textures          │
    │ layer3   │ 256       │ Brushstroke patterns, motifs   │
    │ layer4   │ 512       │ Compositional structure        │
    └──────────┴───────────┴────────────────────────────────┘

Freezing strategy:
    Early layers (conv1, bn1, layer1, optionally layer2) are frozen to
    preserve the general low-level features learned from ImageNet.
    Later layers are fine-tuned to specialise for style.

Reference:
    He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# Output channel counts for each ResNet-18 layer.
RESNET18_CHANNELS = {
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
}


class MultiScaleResNet(nn.Module):
    """
    ResNet-18 backbone that outputs intermediate feature maps.

    Returns a dictionary::

        {
            "layer2": Tensor[B, 128, H/8,  W/8],
            "layer3": Tensor[B, 256, H/16, W/16],
            "layer4": Tensor[B, 512, H/32, W/32],
        }

    For a 224 × 224 input, spatial sizes are 28 × 28, 14 × 14, 7 × 7.

    Args:
        pretrained:          Load ImageNet-pretrained weights.
        freeze_until:        Freeze all layers up to (and including) this
                             stage.  One of ``{"layer1", "layer2", "layer3",
                             None}``.  ``None`` means nothing is frozen.
    """

    # Ordered list of stages for freezing logic.
    _STAGES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]

    def __init__(
        self,
        pretrained: bool = True,
        freeze_until: str | None = "layer2",
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Keep everything except the final avgpool + fc.
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Freeze early layers
        if freeze_until is not None:
            self._freeze_until(freeze_until)

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def _freeze_until(self, stage_name: str) -> None:
        """Freeze parameters from ``conv1`` through ``stage_name`` (inclusive)."""
        if stage_name not in self._STAGES:
            raise ValueError(
                f"Unknown stage '{stage_name}'. Choose from {self._STAGES}"
            )
        freeze_idx = self._STAGES.index(stage_name)
        for name in self._STAGES[: freeze_idx + 1]:
            module = getattr(self, name)
            for param in module.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input images ``[B, 3, H, W]``.

        Returns:
            Dictionary mapping scale names to feature tensors.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return {"layer2": f2, "layer3": f3, "layer4": f4}

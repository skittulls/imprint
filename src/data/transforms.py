"""
Image transforms for training and evaluation.

Training transforms include random augmentations chosen for style-preservation:
heavy geometric augmentation is avoided because it can destroy compositional
style cues, while colour jitter and random erasing simulate the variability
seen across an artist's body of work.

All transforms output tensors normalised with ImageNet statistics because
the backbone (ResNet-18) was pretrained on ImageNet.

Reference:
    He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
"""

from torchvision import transforms

# ImageNet normalisation statistics (used because backbone is pretrained).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Return augmentation pipeline for training.

    Augmentations are deliberately *style-preserving*:
      - RandomResizedCrop:  simulates different crops / zoom levels.
      - RandomHorizontalFlip: mirrors are style-invariant.
      - ColorJitter (mild): simulates lighting / digitisation variation.
      - RandomGrayscale:  forces the network to rely on texture, not colour.
      - RandomErasing:  occlusion robustness.

    Heavy geometric transforms (large rotations, shear) are intentionally
    omitted because they can destroy composition — a valid style signal.

    Args:
        image_size: Target spatial resolution (square).

    Returns:
        ``torchvision.transforms.Compose`` pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15,
            saturation=0.15, hue=0.05
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Return deterministic pipeline for validation / test / inference.

    Args:
        image_size: Target spatial resolution (square).

    Returns:
        ``torchvision.transforms.Compose`` pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

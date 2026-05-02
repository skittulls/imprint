"""
Utility functions for reproducibility, configuration, and logging.
"""

import os
import random
import yaml
import logging
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers Python stdlib, NumPy, PyTorch (CPU + CUDA), and cuDNN.
    Setting ``torch.backends.cudnn.deterministic`` trades a small amount
    of performance for bitwise-reproducible convolution results.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load a YAML configuration file into a nested dictionary.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Configure a project-wide logger with both console and file handlers.

    Args:
        log_dir: Directory to store log files.
        level: Logging level (e.g. ``logging.INFO``).

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("imprint_v2")
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Console
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # File
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

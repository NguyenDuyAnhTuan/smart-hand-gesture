"""
Helper utilities: logging, checkpoint saving/loading, seed, etc.
"""

import os
import logging
import random
import numpy as np
import torch


def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """Tạo logger ghi ra console và file"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def set_seed(seed: int = 42):
    """Đặt seed cho tính tái lập"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, path: str):
    """Lưu state_dict của model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device=None) -> torch.nn.Module:
    """Load state_dict vào model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Đếm số tham số trainable"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

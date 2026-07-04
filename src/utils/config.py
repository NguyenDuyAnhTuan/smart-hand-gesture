"""
Central configuration file for the hand gesture recognition project.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Đường dẫn ──────────────────────────────────────────
    root_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_raw_dir: str = "dataset/raw"
    dataset_processed_dir: str = "dataset/processed"
    csv_path: str = "outputs/results/gesture_dataset.csv"
    save_dir: str = "saved_models"
    log_dir: str = "outputs/logs"
    plot_dir: str = "outputs/plots"

    # ── Nhãn cử chỉ ────────────────────────────────────────
    class_names: List[str] = field(default_factory=lambda: [
        "call", "dislike", "like", "ok", "one",
        "punch", "rock", "sayhi", "stop", "tym"
    ])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    # ── Model ───────────────────────────────────────────────
    model_name: str = "mobilenet_v3_small"
    pretrained: bool = True
    freeze_backbone: bool = True   # Giữ pretrained features, chỉ train classifier
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12

    # ── Training ────────────────────────────────────────────
    epochs: int = 50              # Giới hạn trên; early stopping sẽ dừng sớm hơn
    batch_size: int = 32
    learning_rate: float = 3e-4   # Cao hơn khi chỉ train classifier
    weight_decay: float = 1e-2
    early_stopping_patience: int = 8   # Dừng nếu val_acc không cải thiện sau 8 epochs
    train_split: float = 0.7
    val_split: float = 0.15
    # test_split = 1 - train_split - val_split

    # ── Augmentation ────────────────────────────────────────
    use_augmentation: bool = True

    # ── Device ──────────────────────────────────────────────
    seed: int = 42

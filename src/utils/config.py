"""
Central configuration file for the hand gesture recognition project.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Đường dẫn ──────────────────────────────────────────
    dataset_processed_cropped_dir: str = "dataset/processed_cropped"
    save_dir: str = "saved_models"
    log_dir: str = "outputs/logs"

    # ── Nhãn cử chỉ ────────────────────────────────────────
    class_names: List[str] = field(default_factory=lambda: [
        "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "L",
        "M", "N", "O", "P", "R",
        "S", "T", "U", "V", "W",
        "Y"
    ])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    # ── Model ───────────────────────────────────────────────
    # Tùy chọn: "mobilenet_v3_small" (nhanh, ít overfit) | "mobilenet_v3_large" (cần dataset lớn hơn)
    model_name: str = "mobilenet_v3_small"
    pretrained: bool = True
    freeze_backbone: bool = True        # Fine-tune các block cuối và classifier head
    fine_tune_last_blocks: int = 2      # Số block cuối MobileNetV3 được mở để fine-tune
    backbone_lr_multiplier: float = 0.1 # LR backbone = learning_rate * multiplier
    img_size: int = 224

    # ── Training ────────────────────────────────────────────
    epochs: int = 50                    # Giới hạn trên; early stopping sẽ dừng sớm hơn
    batch_size: int = 32
    learning_rate: float = 3e-4         # LR cho classifier head
    weight_decay: float = 1e-2
    early_stopping_patience: int = 8    # Dừng nếu val_acc không cải thiện sau N epochs

    # ── Augmentation ────────────────────────────────────────
    use_augmentation: bool = True
    # Các lớp dễ nhầm lẫn được áp dụng augmentation nhẹ hơn để giữ đặc trưng phân biệt
    targeted_augmentation_classes: List[str] = field(default_factory=lambda: ["R", "U"])

    # ── Reproducibility ─────────────────────────────────────
    seed: int = 42

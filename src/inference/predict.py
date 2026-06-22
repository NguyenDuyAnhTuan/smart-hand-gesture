"""
Inference module - predict gesture from a single image or landmark vector.
"""

import torch
import numpy as np

from src.models.vit_model import build_model
from src.utils.config import Config


class GesturePredictor:
    """Load model đã train và dự đoán cử chỉ"""

    def __init__(self, model_path: str, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(num_classes=cfg.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.class_names = cfg.class_names

    @torch.no_grad()
    def predict_from_image(self, image_tensor) -> str:
        """
        Dự đoán từ tensor ảnh (C, H, W).
        
        Returns:
            tên nhãn dự đoán
        """
        x = image_tensor.unsqueeze(0).to(self.device)
        logits = self.model(x)
        idx = logits.argmax(1).item()
        return self.class_names[idx]

    @torch.no_grad()
    def predict_from_landmarks(self, landmarks: list) -> str:
        """
        Dự đoán từ vector landmark (63 giá trị float).

        Returns:
            tên nhãn dự đoán
        """
        # TODO: thay thế bằng model riêng cho landmarks nếu cần
        raise NotImplementedError("Chưa implement landmark-based prediction")

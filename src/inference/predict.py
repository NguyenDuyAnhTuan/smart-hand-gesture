"""
Inference helpers for predicting hand gesture labels from images.
"""

import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.models.model_factory import build_model
from src.utils.config import Config


class GesturePredictor:
    """Load a trained model and predict gesture labels."""

    def __init__(self, model_path: str, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model = self._build_and_load(state_dict).to(self.device)
        self.model.eval()

        self.class_names = cfg.class_names
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _build_and_load(self, state_dict):
        """
        Tự phát hiện kiến trúc từ keys của state_dict thay vì tin vào cfg.model_name.
        Điều này tránh load nhầm model khi checkpoint không khớp với config hiện tại.
        """
        keys = set(state_dict.keys())

        # Phân biệt Large vs Small qua kích thước classifier[-1].weight
        classifier_weight_key = next((k for k in keys if k.endswith("classifier.3.weight")), None)
        if classifier_weight_key and state_dict[classifier_weight_key].shape[1] == 1280:
            detected_name = "mobilenet_v3_large"
        else:
            detected_name = getattr(self.cfg, "model_name", "mobilenet_v3_small")

        cfg_name = getattr(self.cfg, "model_name", "mobilenet_v3_small")
        if detected_name != cfg_name:
            import warnings
            warnings.warn(
                f"[GesturePredictor] Checkpoint co kien truc '{detected_name}' "
                f"nhung cfg.model_name='{cfg_name}'. "
                f"Su dung kien truc tu checkpoint: '{detected_name}'."
            )

        model = build_model(
            num_classes=self.cfg.num_classes,
            model_name=detected_name,
            pretrained=False,
        )
        model.load_state_dict(state_dict, strict=True)
        return model

    def _image_to_tensor(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input image is empty")

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return self.transform(pil_image)

    @torch.no_grad()
    def predict_from_image(self, image_tensor) -> str:
        x = image_tensor.unsqueeze(0).to(self.device)
        logits = self.model(x)
        idx = logits.argmax(1).item()
        return self.class_names[idx]

    @torch.no_grad()
    def predict_topk_from_image(self, image_tensor, k: int = 3):
        x = image_tensor.unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        scores, indices = probs.topk(min(k, len(self.class_names)), dim=1)

        return [
            (self.class_names[idx.item()], score.item())
            for score, idx in zip(scores[0], indices[0])
        ]

    def predict_raw_image(self, bgr_image) -> str:
        image_tensor = self._image_to_tensor(bgr_image)
        return self.predict_from_image(image_tensor)

    def predict_topk_raw_image(self, bgr_image, k: int = 3):
        image_tensor = self._image_to_tensor(bgr_image)
        return self.predict_topk_from_image(image_tensor, k=k)

    @torch.no_grad()
    def predict_from_landmarks(self, landmarks: list) -> str:
        raise NotImplementedError(
            "Landmark-based prediction is not implemented for the current image model. "
            "Use predict_raw_image instead."
        )

"""
Model builders for hand gesture recognition.

Hỗ trợ hai kiến trúc MobileNetV3 (Transfer Learning từ ImageNet):
  - mobilenet_v3_small  : nhẹ, nhanh, phù hợp với dataset nhỏ → accuracy cao, ít overfit.
  - mobilenet_v3_large  : sâu hơn, cần nhiều data hơn để phát huy lợi thế.
"""

import warnings

import torch.nn as nn
from torchvision import models


def _build_mobilenet_v3_small(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False, **kwargs):
    weights = None
    if pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except AttributeError:
            weights = None

    try:
        model = models.mobilenet_v3_small(weights=weights)
    except Exception as exc:
        warnings.warn(f"Could not load pretrained MobileNetV3 Small weights ({exc}). Using random weights instead.")
        model = models.mobilenet_v3_small(weights=None)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_mobilenet_v3_large(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False, **kwargs):
    weights = None
    if pretrained:
        try:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        except AttributeError:
            weights = None

    try:
        model = models.mobilenet_v3_large(weights=weights)
    except Exception as exc:
        warnings.warn(f"Could not load pretrained MobileNetV3 Large weights ({exc}). Using random weights instead.")
        model = models.mobilenet_v3_large(weights=None)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(
    num_classes: int,
    model_name: str = "mobilenet_v3_small",
    pretrained: bool = False,
    freeze_backbone: bool = False,
    **kwargs,
):
    """Factory function — hỗ trợ MobileNetV3 Small và Large."""
    name = model_name.lower()

    if name in {"mobilenet_v3_small", "mobilenet", "mobilenet_v3"}:
        return _build_mobilenet_v3_small(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name == "mobilenet_v3_large":
        return _build_mobilenet_v3_large(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    raise ValueError(
        f"Unsupported model_name: '{model_name}'. "
        "Các giá trị hợp lệ: 'mobilenet_v3_small', 'mobilenet_v3_large'."
    )

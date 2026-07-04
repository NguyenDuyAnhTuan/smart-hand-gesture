"""
Model builders for hand gesture recognition.

The original project used a ViT trained from scratch. That is kept for
compatibility, but the default training path can use MobileNetV3 transfer
learning, which is usually a better fit for a small image dataset and webcam
input.
"""

import warnings

import torch
import torch.nn as nn
from torchvision import models


class PatchEmbedding(nn.Module):
    """Split an image into patches and project them to embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTGestureClassifier(nn.Module):
    """Vision Transformer classifier retained for backward compatibility."""

    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def _build_mobilenet_v3_small(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    weights = None
    if pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except AttributeError:
            weights = None

    try:
        model = models.mobilenet_v3_small(weights=weights)
    except Exception as exc:
        warnings.warn(f"Could not load pretrained MobileNetV3 weights ({exc}). Using random weights instead.")
        model = models.mobilenet_v3_small(weights=None)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(
    num_classes: int,
    model_name: str = "vit",
    pretrained: bool = False,
    freeze_backbone: bool = False,
    **kwargs,
):
    """Factory function for supported gesture classifiers."""
    name = model_name.lower()

    if name in {"mobilenet", "mobilenet_v3", "mobilenet_v3_small"}:
        return _build_mobilenet_v3_small(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name in {"vit", "vit_scratch"}:
        return ViTGestureClassifier(num_classes=num_classes, **kwargs)

    raise ValueError(f"Unsupported model_name: {model_name}")

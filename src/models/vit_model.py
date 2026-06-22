"""
Vision Transformer (ViT) model for hand gesture recognition.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Chia ảnh thành các patch và embed"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                   # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)   # (B, num_patches, embed_dim)
        return x


class ViTGestureClassifier(nn.Module):
    """
    Vision Transformer cho bài toán phân loại cử chỉ tay.

    Args:
        num_classes: số lớp cử chỉ
        img_size: kích thước ảnh đầu vào (mặc định 224)
        patch_size: kích thước mỗi patch (mặc định 16)
        embed_dim: số chiều embedding
        depth: số lớp Transformer
        num_heads: số attention heads
    """

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
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x[:, 0])   # lấy CLS token
        return self.head(x)


def build_model(num_classes: int, **kwargs) -> ViTGestureClassifier:
    """Factory function tạo model"""
    return ViTGestureClassifier(num_classes=num_classes, **kwargs)

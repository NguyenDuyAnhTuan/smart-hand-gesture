"""
Training script for ViT-based hand gesture recognition model.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.model_factory import build_model
from src.utils.config import Config
from src.utils.helper import save_checkpoint, get_logger

logger = get_logger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train(cfg: Config, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    model_name = getattr(cfg, "model_name", "vit")
    checkpoint_prefix = model_name.replace("/", "_").replace("\\", "_")

    model = build_model(
        num_classes=cfg.num_classes,
        model_name=model_name,
        pretrained=getattr(cfg, "pretrained", False),
        freeze_backbone=getattr(cfg, "freeze_backbone", False),
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # ── Early Stopping ───────────────────────────────────────────────────────
    patience = getattr(cfg, "early_stopping_patience", 0)  # 0 = disabled
    best_acc = 0.0
    epochs_no_improve = 0
    best_state = None
    # ────────────────────────────────────────────────────────────────────────

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch [{epoch}/{cfg.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        save_checkpoint(model, os.path.join(cfg.save_dir, f"{checkpoint_prefix}_last.pth"))
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(model, os.path.join(cfg.save_dir, f"{checkpoint_prefix}_best.pth"))
            logger.info(f"  --> Best model saved! (val_acc={best_acc:.4f})")
        else:
            epochs_no_improve += 1
            if patience > 0:
                logger.info(
                    f"  No improvement for {epochs_no_improve}/{patience} epochs."
                )

        # Early stopping check
        if patience > 0 and epochs_no_improve >= patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(no val_acc improvement for {patience} epochs). "
                f"Best val_acc={best_acc:.4f}"
            )
            break

    # Restore best weights before saving final
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, os.path.join(cfg.save_dir, f"{checkpoint_prefix}_final.pth"))
    logger.info("Training hoàn tất!")
    return model


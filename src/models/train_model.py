"""
Training script for MobileNetV3 hand gesture recognition model.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.model_factory import build_model
from src.utils.config import Config
from src.utils.helper import save_checkpoint, get_logger

logger = get_logger(__name__)


def _is_mobilenet_model(model_name: str) -> bool:
    return model_name.lower() in {"mobilenet", "mobilenet_v3", "mobilenet_v3_small", "mobilenet_v3_large"}


def _configure_mobilenet_finetuning(model, freeze_backbone: bool, fine_tune_last_blocks: int):
    if not hasattr(model, "features") or not hasattr(model, "classifier"):
        return

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    feature_blocks = list(model.features.children())
    if freeze_backbone:
        blocks_to_unfreeze = feature_blocks[-max(fine_tune_last_blocks, 0):] if fine_tune_last_blocks > 0 else []
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
    else:
        for param in model.features.parameters():
            param.requires_grad = True


def _build_optimizer(model, cfg: Config, model_name: str):
    if _is_mobilenet_model(model_name):
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("classifier"):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": cfg.learning_rate * getattr(cfg, "backbone_lr_multiplier", 0.1),
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": cfg.learning_rate,
            })

        return AdamW(param_groups, weight_decay=cfg.weight_decay)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


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
    model_name = getattr(cfg, "model_name", "mobilenet_v3_small")
    checkpoint_prefix = model_name.replace("/", "_").replace("\\", "_")

    model = build_model(
        num_classes=cfg.num_classes,
        model_name=model_name,
        pretrained=getattr(cfg, "pretrained", True),
        freeze_backbone=getattr(cfg, "freeze_backbone", True),
    ).to(device)

    if _is_mobilenet_model(model_name):
        _configure_mobilenet_finetuning(
            model,
            freeze_backbone=getattr(cfg, "freeze_backbone", False),
            fine_tune_last_blocks=getattr(cfg, "fine_tune_last_blocks", 0),
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, cfg, model_name)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,}/{total_params:,} "
        f"({trainable_params / max(total_params, 1):.2%})"
    )
    for group_index, param_group in enumerate(optimizer.param_groups):
        logger.info(f"Optimizer group {group_index}: lr={param_group['lr']:.2e}, params={len(param_group['params'])}")

    # ── Early Stopping ───────────────────────────────────────────────────────
    patience = getattr(cfg, "early_stopping_patience", 0)  # 0 = disabled
    best_acc = 0.0
    epochs_no_improve = 0
    best_state = None
    # ────────────────────────────────────────────────────────────────────────

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

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


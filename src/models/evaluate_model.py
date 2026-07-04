"""
Evaluate a trained checkpoint on dataset/processed_cropped/test.
"""

from collections import Counter
import os

import torch

from src.data.dataloader import HandGestureImageDataset
from src.models.model_factory import build_model
from src.utils.config import Config


def _default_checkpoint_path(cfg: Config):
    model_name = getattr(cfg, "model_name", "vit")
    checkpoint_prefix = model_name.replace("/", "_").replace("\\", "_")
    candidates = [
        os.path.join(cfg.save_dir, f"{checkpoint_prefix}_best.pth"),
        os.path.join(cfg.save_dir, f"{checkpoint_prefix}_last.pth"),
        os.path.join(cfg.save_dir, f"{checkpoint_prefix}_final.pth"),
        os.path.join(cfg.save_dir, "vit_best.pth"),
        os.path.join(cfg.save_dir, "vit_last.pth"),
        os.path.join(cfg.save_dir, "vit_final.pth"),
    ]
    return next((path for path in candidates if os.path.exists(path)), candidates[0])


@torch.no_grad()
def evaluate_checkpoint(model_path=None):
    cfg = Config()
    if model_path is None:
        model_path = _default_checkpoint_path(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    test_dataset = HandGestureImageDataset(
        root_dir="dataset/processed_cropped/test",
        class_names=cfg.class_names,
        use_augment=False,
        img_size=cfg.img_size,
    )
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    state_dict = torch.load(model_path, map_location=device)
    model = build_model(
        num_classes=cfg.num_classes,
        model_name=cfg.model_name,
        pretrained=False,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
    )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model = build_model(
            num_classes=cfg.num_classes,
            model_name="vit",
            pretrained=False,
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
        )
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    total = 0
    correct = 0
    pred_counts = Counter()
    class_correct = Counter()
    class_total = Counter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)

        total += labels.numel()
        correct += (preds == labels).sum().item()

        for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            pred_name = cfg.class_names[pred]
            label_name = cfg.class_names[label]
            pred_counts[pred_name] += 1
            class_total[label_name] += 1
            if pred == label:
                class_correct[label_name] += 1

    print(f"Checkpoint: {model_path}")
    print(f"Accuracy: {correct / max(total, 1):.4f} ({correct}/{total})")
    print("\nPrediction distribution:")
    for label in cfg.class_names:
        print(f"  {label:8s}: {pred_counts[label]}")

    print("\nPer-class accuracy:")
    for label in cfg.class_names:
        acc = class_correct[label] / max(class_total[label], 1)
        print(f"  {label:8s}: {acc:.4f} ({class_correct[label]}/{class_total[label]})")


if __name__ == "__main__":
    evaluate_checkpoint()

"""
Evaluate a trained checkpoint on dataset/processed_cropped/test.
"""

from collections import Counter
import csv
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch

from src.data.dataloader import HandGestureImageDataset
from src.models.model_factory import build_model
from src.utils.config import Config


def _default_checkpoint_path(cfg: Config):
    model_name = getattr(cfg, "model_name", "mobilenet_v3_small")
    checkpoint_prefix = model_name.replace("/", "_").replace("\\", "_")
    candidates = [
        os.path.join(cfg.save_dir, f"{checkpoint_prefix}_best.pth"),
        os.path.join(cfg.save_dir, f"{checkpoint_prefix}_last.pth"),
        os.path.join(cfg.save_dir, f"{checkpoint_prefix}_final.pth"),
    ]
    return next((path for path in candidates if os.path.exists(path)), candidates[0])


def _save_confusion_matrix(conf_matrix, class_names, output_path: str):
    fig_width = max(10, len(class_names) * 0.45)
    fig_height = max(8, len(class_names) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = conf_matrix.max() / 2.0 if conf_matrix.size else 0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                j,
                i,
                format(conf_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_confidence_histogram(correct_confidences, incorrect_confidences, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 21)

    if correct_confidences:
        ax.hist(
            correct_confidences,
            bins=bins,
            alpha=0.7,
            color="#2ca02c",
            label=f"Correct ({len(correct_confidences)})",
            edgecolor="black",
        )
    if incorrect_confidences:
        ax.hist(
            incorrect_confidences,
            bins=bins,
            alpha=0.7,
            color="#d62728",
            label=f"Incorrect ({len(incorrect_confidences)})",
            edgecolor="black",
        )

    ax.set_xlabel("Prediction confidence (max softmax probability)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Histogram")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_misclassified_image(source_path: str, true_label: str, pred_label: str, output_root: str):
    file_name = os.path.basename(source_path)
    destination_dir = os.path.join(output_root, true_label, pred_label)
    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, file_name)

    copied = False
    try:
        shutil.copy2(source_path, destination_path)
        copied = True
    except Exception:
        image = cv2.imread(source_path)
        if image is not None:
            copied = cv2.imwrite(destination_path, image)

    return copied, destination_path


def _compute_classification_metrics(conf_matrix, class_names):
    rows = []
    total_correct = int(np.trace(conf_matrix))
    total_samples = int(conf_matrix.sum())

    for index, class_name in enumerate(class_names):
        tp = int(conf_matrix[index, index])
        fp = int(conf_matrix[:, index].sum() - tp)
        fn = int(conf_matrix[index, :].sum() - tp)
        support = int(conf_matrix[index, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = tp / support if support else 0.0

        rows.append({
            "label": class_name,
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": support,
            "accuracy": accuracy,
        })

    precisions = [row["precision"] for row in rows]
    recalls = [row["recall"] for row in rows]
    f1_scores = [row["f1-score"] for row in rows]
    supports = [row["support"] for row in rows]
    support_sum = sum(supports)

    macro_avg = {
        "label": "macro avg",
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1-score": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "support": support_sum,
        "accuracy": float(total_correct / total_samples) if total_samples else 0.0,
    }
    weighted_avg = {
        "label": "weighted avg",
        "precision": float(np.average(precisions, weights=supports)) if support_sum else 0.0,
        "recall": float(np.average(recalls, weights=supports)) if support_sum else 0.0,
        "f1-score": float(np.average(f1_scores, weights=supports)) if support_sum else 0.0,
        "support": support_sum,
        "accuracy": float(total_correct / total_samples) if total_samples else 0.0,
    }
    accuracy_row = {
        "label": "accuracy",
        "precision": "",
        "recall": "",
        "f1-score": float(total_correct / total_samples) if total_samples else 0.0,
        "support": support_sum,
        "accuracy": float(total_correct / total_samples) if total_samples else 0.0,
    }
    return rows, accuracy_row, macro_avg, weighted_avg


def _write_csv(path: str, header: list, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_checkpoint(model_path=None, output_dir: str = "outputs/evaluation"):
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
        return_paths=True,
    )
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    state_dict = torch.load(model_path, map_location=device)
    # Tự phát hiện Large vs Small qua kích thước classifier[-1].weight
    keys = set(state_dict.keys())
    classifier_weight_key = next((k for k in keys if k.endswith("classifier.3.weight")), None)
    if classifier_weight_key and state_dict[classifier_weight_key].shape[1] == 1280:
        detected_name = "mobilenet_v3_large"
    else:
        detected_name = getattr(cfg, "model_name", "mobilenet_v3_small")

    model = build_model(
        num_classes=cfg.num_classes,
        model_name=detected_name,
        pretrained=False,
    )
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    total = 0
    correct = 0
    top2_correct = 0
    pred_counts = Counter()
    class_correct = Counter()
    class_total = Counter()
    all_labels = []
    all_preds = []
    correct_confidences = []
    incorrect_confidences = []

    os.makedirs(output_dir, exist_ok=True)
    misclassified_root = os.path.join(output_dir, "misclassified")
    os.makedirs(misclassified_root, exist_ok=True)

    for images, labels, paths in loader:
        batch_paths = list(paths)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        probs = torch.softmax(outputs, dim=1)
        confidences = probs.max(dim=1).values
        topk = outputs.topk(min(2, outputs.size(1)), dim=1).indices

        total += labels.numel()
        batch_correct = (preds == labels)
        correct += batch_correct.sum().item()
        top2_correct += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        correct_confidences.extend(confidences[batch_correct].cpu().tolist())
        incorrect_confidences.extend(confidences[~batch_correct].cpu().tolist())

        for pred, label, source_path in zip(preds.cpu().tolist(), labels.cpu().tolist(), batch_paths):
            pred_name = cfg.class_names[pred]
            label_name = cfg.class_names[label]
            pred_counts[pred_name] += 1
            class_total[label_name] += 1
            if pred == label:
                class_correct[label_name] += 1
            else:
                _save_misclassified_image(
                    source_path=source_path,
                    true_label=label_name,
                    pred_label=pred_name,
                    output_root=misclassified_root,
                )

    report_csv_path = os.path.join(output_dir, "classification_report.csv")

    conf_matrix = np.zeros((len(cfg.class_names), len(cfg.class_names)), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        conf_matrix[true_label, pred_label] += 1
    conf_matrix_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    _write_csv(
        conf_matrix_csv_path,
        header=["true_label"] + cfg.class_names,
        rows=[{"true_label": label, **{pred_label: int(conf_matrix[i, j]) for j, pred_label in enumerate(cfg.class_names)}} for i, label in enumerate(cfg.class_names)],
    )
    _save_confusion_matrix(conf_matrix, cfg.class_names, os.path.join(output_dir, "confusion_matrix.png"))
    confidence_histogram_path = os.path.join(output_dir, "confidence_histogram.png")
    _save_confidence_histogram(correct_confidences, incorrect_confidences, confidence_histogram_path)

    class_rows, accuracy_row, macro_avg, weighted_avg = _compute_classification_metrics(conf_matrix, cfg.class_names)
    _write_csv(report_csv_path, header=["label", "precision", "recall", "f1-score", "support", "accuracy"], rows=class_rows + [accuracy_row, macro_avg, weighted_avg])

    print(f"Checkpoint: {model_path}")
    print(f"Accuracy: {correct / max(total, 1):.4f} ({correct}/{total})")
    print(f"Top-2 Accuracy: {top2_correct / max(total, 1):.4f} ({top2_correct}/{total})")
    print(
        f"Confidence mean: correct={np.mean(correct_confidences) if correct_confidences else 0.0:.4f}, "
        f"incorrect={np.mean(incorrect_confidences) if incorrect_confidences else 0.0:.4f}"
    )
    print(f"\nSaved confusion matrix to: {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"Saved confidence histogram to: {confidence_histogram_path}")
    print(f"Saved classification report to: {report_csv_path}")
    print(f"Saved misclassified images to: {misclassified_root}")
    print("\nPrediction distribution:")
    for label in cfg.class_names:
        print(f"  {label:8s}: {pred_counts[label]}")

    print("\nPer-class metrics:")
    for label in cfg.class_names:
        metric_row = next(row for row in class_rows if row["label"] == label)
        precision = metric_row["precision"]
        recall = metric_row["recall"]
        f1_score = metric_row["f1-score"]
        acc = class_correct[label] / max(class_total[label], 1)
        print(
            f"  {label:8s}: "
            f"P={precision:.4f} R={recall:.4f} F1={f1_score:.4f} "
            f"Acc={acc:.4f} ({class_correct[label]}/{class_total[label]})"
        )


if __name__ == "__main__":
    evaluate_checkpoint()

"""
main.py - Entry point for the Hand Gesture Recognition project.

Usage:
    python main.py --mode train
    python main.py --mode webcam
    python main.py --mode predict --image path/to/image.jpg
"""

import argparse
import os

import cv2

from src.utils.config import Config
from src.utils.helper import set_seed, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition")
    parser.add_argument(
        "--mode",
        choices=["train", "webcam", "predict"],
        default="webcam",
        help="Chon che do: train | webcam | predict",
    )
    parser.add_argument("--image", type=str, default=None, help="Duong dan anh (dung khi mode=predict)")
    parser.add_argument("--model", type=str, default=None, help="Duong dan model (mac dinh: tu dong tim checkpoint moi nhat trong saved_models/)")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi de so luong epoch huan luyen")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)

    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.mode == "train":
        logger.info("=== Bat dau training ===")
        from src.data.dataloader import get_dataloaders
        from src.models.train_model import train

        cropped_dir = "dataset/processed_cropped"
        if not os.path.exists(cropped_dir) or len(os.listdir(cropped_dir)) == 0:
            logger.warning("Thu muc dataset/processed_cropped trong. Dang tu dong chay crop_dataset.py...")
            from src.data.crop_dataset import main as run_cropping
            run_cropping()

        logger.info("Dang nap du lieu...")
        train_loader, val_loader = get_dataloaders(cfg)

        logger.info("Dang khoi dong qua trinh huan luyen ViT...")
        train(cfg, train_loader, val_loader)

    elif args.mode == "webcam":
        logger.info("=== Chay webcam demo ===")
        from src.inference.webcam import run_webcam
        run_webcam(cfg)

    elif args.mode == "predict":
        if args.image is None:
            raise ValueError("Can truyen --image khi dung mode=predict")

        from src.inference.predict import GesturePredictor
        from src.inference.mediapipe_hand import HandLandmarkDetector

        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Khong tim thay anh tai: {args.image}")

        # ── Tự tìm checkpoint thông minh (giống webcam) ──────────────────────
        if args.model is not None:
            model_path = args.model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Khong tim thay model tai: {model_path}. Vui long train truoc!")
        else:
            model_name = getattr(cfg, "model_name", "vit")
            checkpoint_prefix = model_name.replace("/", "_").replace("\\", "_")
            candidate_paths = [
                os.path.join(cfg.save_dir, f"{checkpoint_prefix}_best.pth"),
                os.path.join(cfg.save_dir, f"{checkpoint_prefix}_last.pth"),
                os.path.join(cfg.save_dir, f"{checkpoint_prefix}_final.pth"),
                os.path.join(cfg.save_dir, "vit_best.pth"),
                os.path.join(cfg.save_dir, "vit_last.pth"),
                os.path.join(cfg.save_dir, "vit_final.pth"),
            ]
            model_path = next((p for p in candidate_paths if os.path.exists(p)), None)
            if model_path is None:
                raise FileNotFoundError(
                    f"Khong tim thay file model da train trong '{cfg.save_dir}'. "
                    "Vui long huan luyen truoc: python main.py --mode train"
                )
        logger.info(f"Su dung checkpoint: {model_path}")
        # ─────────────────────────────────────────────────────────────────────

        logger.info(f"=== Dang nap anh: {args.image} ===")
        img = cv2.imread(args.image)
        if img is None:
            raise ValueError(f"Khong the doc anh: {args.image}")

        detector = HandLandmarkDetector(static_image_mode=True)
        landmarks = detector.detect(img)

        h, w, _ = img.shape
        cropped_hand = img

        if landmarks is not None:
            logger.info("Da phat hien ban tay trong anh. Dang tien hanh cat...")
            x_coords = landmarks[0::3]
            y_coords = landmarks[1::3]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)

            xmin_px = int(xmin * w)
            xmax_px = int(xmax * w)
            ymin_px = int(ymin * h)
            ymax_px = int(ymax * h)

            box_w = xmax_px - xmin_px
            box_h = ymax_px - ymin_px
            margin_x = int(box_w * 0.15)
            margin_y = int(box_h * 0.15)

            xmin_px = max(0, xmin_px - margin_x)
            ymin_px = max(0, ymin_px - margin_y)
            xmax_px = min(w, xmax_px + margin_x)
            ymax_px = min(h, ymax_px + margin_y)

            if (xmax_px - xmin_px) > 10 and (ymax_px - ymin_px) > 10:
                cropped_hand = img[ymin_px:ymax_px, xmin_px:xmax_px]
        else:
            logger.warning("Khong phat hien ban tay qua MediaPipe. Du doan tren toan bo anh.")

        detector.close()

        logger.info("Dang nap model...")
        predictor = GesturePredictor(model_path, cfg)

        label = predictor.predict_raw_image(cropped_hand)
        logger.info(f"Ket qua du doan: cu chi cua ban la '{label}'")


if __name__ == "__main__":
    main()

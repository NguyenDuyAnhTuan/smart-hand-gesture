"""
main.py - Entry point for the Hand Gesture Recognition project.

Usage:
    python main.py --mode train
    python main.py --mode webcam
    python main.py --mode predict --image path/to/image.jpg
"""

import argparse

from src.utils.config import Config
from src.utils.helper import set_seed, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition")
    parser.add_argument(
        "--mode",
        choices=["train", "webcam", "predict"],
        default="webcam",
        help="Chế độ chạy: train | webcam | predict",
    )
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn ảnh (dùng khi mode=predict)")
    parser.add_argument("--model", type=str, default="saved_models/vit_best.pth", help="Đường dẫn model")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)

    if args.mode == "train":
        logger.info("=== Bắt đầu Training ===")
        # TODO: import DataLoader và gọi train()
        raise NotImplementedError("Training pipeline chưa hoàn thiện")

    elif args.mode == "webcam":
        logger.info("=== Chạy Webcam Demo ===")
        from src.inference.webcam import run_webcam
        run_webcam(cfg)

    elif args.mode == "predict":
        if args.image is None:
            raise ValueError("Cần truyền --image khi dùng mode=predict")
        logger.info(f"=== Dự đoán ảnh: {args.image} ===")
        # TODO: load ảnh và gọi GesturePredictor
        raise NotImplementedError("Predict pipeline chưa hoàn thiện")


if __name__ == "__main__":
    main()

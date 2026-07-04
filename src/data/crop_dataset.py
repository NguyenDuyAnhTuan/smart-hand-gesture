"""
crop_dataset.py - Preprocess hand crops from the raw dataset and save them
into dataset/processed_cropped.
"""

import os

import cv2
from tqdm import tqdm

try:
    import mediapipe as mp
    _hands_factory = mp.solutions.hands.Hands
except Exception:
    mp = None
    _hands_factory = None


INPUT_DIR = "dataset/processed"
OUTPUT_DIR = "dataset/processed_cropped"
SPLITS = ["train", "val", "test"]

hands = None
if _hands_factory is not None:
    hands = _hands_factory(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )


def crop_hand(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    if hands is None:
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    hand_landmarks = results.multi_hand_landmarks[0]
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

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

    if (xmax_px - xmin_px) < 10 or (ymax_px - ymin_px) < 10:
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    cropped = img[ymin_px:ymax_px, xmin_px:xmax_px]
    return cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)


def main():
    print("=== Bat dau tien xu ly cat anh ban tay ===")
    if hands is None:
        print("Canh bao: khong khoi tao duoc MediaPipe Hands. Se dung resize toan bo anh lam fallback.")

    if _hands_factory is None:
        print("\n" + "="*70)
        print("LỖI NGHIÊM TRỌNG: MediaPipe không khả dụng (thiếu module solutions).")
        print("Không thể tự động crop dữ liệu (crop_dataset.py cần MediaPipe).")
        print("Gợi ý: Hãy dùng môi trường ảo (venv) với Python 3.10 hoặc 3.11 để")
        print("chạy script này, do MediaPipe hiện không hỗ trợ tốt Python 3.12+.")
        print("="*70 + "\n")
        return

    total_processed = 0
    total_skipped = 0

    if not os.path.exists(INPUT_DIR):
        print(f"Loi: Thu muc nguon {INPUT_DIR} khong ton tai!")
        return

    for split in SPLITS:
        split_in_dir = os.path.join(INPUT_DIR, split)
        if not os.path.exists(split_in_dir):
            continue

        print(f"\nDang xu ly tap: {split}")
        labels = os.listdir(split_in_dir)

        for label in labels:
            label_in_dir = os.path.join(split_in_dir, label)
            if not os.path.isdir(label_in_dir):
                continue

            label_out_dir = os.path.join(OUTPUT_DIR, split, label)
            os.makedirs(label_out_dir, exist_ok=True)

            img_names = [f for f in os.listdir(label_in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            print(f" - Nhan '{label}': {len(img_names)} anh")

            for img_name in tqdm(img_names, desc=f"  Cat nhan {label}"):
                img_path = os.path.join(label_in_dir, img_name)
                cropped_img = crop_hand(img_path)

                if cropped_img is not None:
                    out_path = os.path.join(label_out_dir, img_name)
                    cv2.imwrite(out_path, cropped_img)
                    total_processed += 1
                else:
                    total_skipped += 1

    print("\n=== Hoan thanh tien xu ly ===")
    print(f"Tong so anh da cat thanh cong: {total_processed}")
    print(f"Tong so anh bi bo qua (khong phat hien duoc tay): {total_skipped}")

    if hands is not None:
        hands.close()


if __name__ == "__main__":
    main()

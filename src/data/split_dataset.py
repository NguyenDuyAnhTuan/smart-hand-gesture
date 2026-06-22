import os
import random
import shutil

RAW_DIR = "dataset/raw"
OUTPUT_DIR = "dataset/processed"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

for label in os.listdir(RAW_DIR):

    label_path = os.path.join(RAW_DIR, label)

    if not os.path.isdir(label_path):
        continue

    images = [
        f for f in os.listdir(label_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images)
    ]:

        save_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            label
        )

        os.makedirs(save_dir, exist_ok=True)

        for img in split_images:
            shutil.copy(
                os.path.join(label_path, img),
                os.path.join(save_dir, img)
            )

    print(
        f"{label}: "
        f"Train={len(train_images)}, "
        f"Val={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("Chia dữ liệu hoàn tất!")
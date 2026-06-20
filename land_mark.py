import os
import cv2
import mediapipe as mp
import pandas as pd

# ==========================
# Đường dẫn dataset
# ==========================
DATASET_DIR = "dataset"
OUTPUT_CSV = "gesture_dataset.csv"

# ==========================
# MediaPipe Hands
# ==========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

data = []

# ==========================
# Duyệt từng thư mục emoji
# ==========================
for label in os.listdir(DATASET_DIR):

    label_path = os.path.join(DATASET_DIR, label)

    if not os.path.isdir(label_path):
        continue

    print(f"\nĐang xử lý: {label}")

    count = 0

    for img_name in os.listdir(label_path):

        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            continue

        hand = results.multi_hand_landmarks[0]

        row = []

        for lm in hand.landmark:
            row.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        row.append(label)

        data.append(row)

        count += 1

    print(f"Đã lấy {count} ảnh")

# ==========================
# Tạo tên cột
# ==========================
columns = []

for i in range(21):
    columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

columns.append("label")

# ==========================
# Lưu CSV
# ==========================
df = pd.DataFrame(data, columns=columns)

df.to_csv(OUTPUT_CSV, index=False)

print("\n====================")
print("Hoàn thành!")
print("Số mẫu:", len(df))
print("File:", OUTPUT_CSV)
print("====================")
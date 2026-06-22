import cv2
import os

video_path = r"C:\Users\Admin\Desktop\btl-tgmt\smart-hand-gesture\video\51679690-21cd-4fde-8629-a512cdc037c4.mp4"      # Video đầu vào
output_folder = r"C:\Users\Admin\Desktop\btl-tgmt\smart-hand-gesture\dataset\call"    # Thư mục lưu ảnh

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Lưu mỗi 10 frame
    if frame_count % 10 == 0:
        filename = os.path.join(
            output_folder,
            f"{saved_count:04d}.jpg"
        )

        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Tổng frame: {frame_count}")
print(f"Đã lưu: {saved_count} ảnh")
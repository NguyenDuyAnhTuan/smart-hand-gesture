"""
Real-time webcam inference for hand gesture recognition.

- Lược bỏ giao diện phức tạp để tăng FPS.
- Tích hợp module HandDetectorWrapper an toàn.
- Bỏ qua vài frame model inference (frame skip) để đảm bảo FPS mượt.
"""

import os
import time
from collections import deque, Counter
import cv2

from src.utils.config import Config
from src.inference.mediapipe_hand import HandDetectorWrapper


def run_webcam(cfg: Config, camera_id: int = 0):
    from src.inference.predict import GesturePredictor

    # ── Tìm checkpoint
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
        print(f"Loi: Khong tim thay file model trong {cfg.save_dir}")
        return

    print(f"Dang tai model: {model_path}")
    try:
        predictor = GesturePredictor(model_path, cfg)
    except Exception as e:
        print(f"Loi khi load model: {e}")
        return

    # ── Khởi tạo Hand Detector
    detector = HandDetectorWrapper(use_mediapipe=True)

    # ── Khởi tạo camera
    import platform
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Khong the mo camera {camera_id}")
        return

    # ── State
    recent_labels = deque(maxlen=7)
    stable_label = "---"
    stable_conf = 0.0

    fps_counter = deque(maxlen=30)
    t_prev = time.time()

    # Để tăng FPS: bỏ qua số frame inference
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 2

    print(">>> Nhan Q de thoat <<<")

    # Warmup camera (bo qua 10 frame dau de tranh bi den)
    for _ in range(10):
        cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_count += 1

        t_now = time.time()
        fps_counter.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        fps = sum(fps_counter) / len(fps_counter)

        # Trích xuất ảnh crop và bbox qua module bọc
        crop_img, bbox, is_fallback = detector.get_crop(frame)

        if crop_img is not None and bbox is not None:
            # Chỉ dự đoán mỗi N frame
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                try:
                    topk = predictor.predict_topk_raw_image(crop_img, k=1)
                    label = topk[0][0]
                    conf = topk[0][1]

                    recent_labels.append(label)
                    stable_label = Counter(recent_labels).most_common(1)[0][0]
                    stable_conf = conf
                except Exception:
                    pass

            # Vẽ UI cơ bản (không bóng đổ, không alpha blending)
            x1, y1, x2, y2 = bbox
            color = (0, 165, 255) if is_fallback else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            text = f"{stable_label} {stable_conf*100:.1f}%"
            cv2.putText(frame, text, (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Vẽ FPS và thông báo Fallback
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if is_fallback:
            cv2.putText(frame, "MediaPipe Error - Dung Center Crop", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.imshow("Hand Gesture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config()
    run_webcam(cfg)

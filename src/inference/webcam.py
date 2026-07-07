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


def _open_camera(camera_id: int):
    """Open the camera with a backend fallback on Windows."""
    import platform

    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    else:
        backends = [None]

    for backend in backends:
        cap = cv2.VideoCapture(camera_id) if backend is None else cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()

    return None


def _warmup_camera(cap, warmup_frames: int = 15, max_attempts: int = 45):
    """Flush stale frames and wait for a non-empty frame before starting the loop."""
    non_black_frame = None
    attempts = 0

    while attempts < max_attempts:
        ret, frame = cap.read()
        attempts += 1
        if not ret or frame is None:
            continue

        if frame.size == 0:
            continue

        if attempts <= warmup_frames:
            continue

        if frame.mean() <= 1.0:
            continue

        non_black_frame = frame
        break

    return non_black_frame


def run_webcam(cfg: Config, camera_id: int = 0):
    from src.inference.predict import GesturePredictor

    # ── Tìm checkpoint: ưu tiên file khớp model_name, sau đó tìm file _best.pth mới nhất
    model_name = getattr(cfg, "model_name", "mobilenet_v3_small")
    checkpoint_prefix = model_name.replace("/", "_").replace("\\", "_")
    candidate_paths = [
        os.path.join(cfg.save_dir, f"{checkpoint_prefix}_best.pth"),
        os.path.join(cfg.save_dir, f"{checkpoint_prefix}_last.pth"),
        os.path.join(cfg.save_dir, f"{checkpoint_prefix}_final.pth"),
    ]
    model_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    # Fallback: quét thư mục, ưu tiên file _best.pth mới nhất (tránh dùng checkpoint sai num_classes)
    if model_path is None and os.path.isdir(cfg.save_dir):
        best_files = sorted(
            [os.path.join(cfg.save_dir, f) for f in os.listdir(cfg.save_dir)
             if f.endswith("_best.pth")],
            key=os.path.getmtime,
            reverse=True,
        )
        if best_files:
            model_path = best_files[0]

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
    cv2.destroyAllWindows()
    cap = _open_camera(camera_id)
    if cap is None:
        print(f"Khong the mo camera {camera_id}")
        detector.close()
        return

    try:
        # ── State
        recent_labels = deque(maxlen=7)
        stable_label = "---"
        stable_conf = 0.0

        fps_counter = deque(maxlen=30)
        t_prev = time.time()

        # Để tăng FPS: bỏ qua số frame inference
        frame_count = 0
        PROCESS_EVERY_N_FRAMES = 2
        MIN_CONFIDENCE = 0.50  # Chỉ chấp nhận dự đoán khi model đủ tự tin

        print(">>> Nhan Q de thoat <<<")

        first_frame = _warmup_camera(cap)
        if first_frame is None:
            print("Loi: Camera khong tra ve frame hop le (co the bi den hoac bi chiem giu boi ung dung khac)")
            return

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
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

                        # Chỉ cập nhật khi model đủ tự tin (tránh nhiễu khi chuyển cử chỉ)
                        if conf >= MIN_CONFIDENCE:
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
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config()
    run_webcam(cfg)

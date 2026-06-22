"""
Real-time webcam inference for hand gesture recognition.
"""

import cv2
from src.inference.mediapipe_hand import HandLandmarkDetector
from src.utils.config import Config


def run_webcam(cfg: Config, camera_id: int = 0):
    """
    Chạy nhận diện cử chỉ tay theo thời gian thực qua webcam.

    Args:
        cfg: cấu hình project
        camera_id: ID camera (thường là 0)
    """
    detector = HandLandmarkDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Không thể mở camera {camera_id}")
        return

    print("Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detector.detect(frame)
        frame = detector.draw(frame)

        if landmarks is not None:
            # TODO: gọi predict từ landmarks
            label = "Detected"
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    cfg = Config()
    run_webcam(cfg)

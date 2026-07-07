"""
MediaPipe hand landmark detection module and robust fallback logic.
"""

import cv2
import numpy as np

# Thử import MediaPipe một cách an toàn
try:
    import mediapipe as mp
    _hands_factory = mp.solutions.hands.Hands
    _drawing_utils = mp.solutions.drawing_utils
    _hand_connections = mp.solutions.hands.HAND_CONNECTIONS
except Exception:
    mp = None
    _hands_factory = None
    _drawing_utils = None
    _hand_connections = None


class HandDetectorWrapper:
    """
    Bọc MediaPipe Hands. Nếu MediaPipe không khả dụng (ví dụ lỗi phiên bản Python),
    cung cấp cơ chế Fallback Center Crop.
    """

    def __init__(self, use_mediapipe=True):
        self.use_mediapipe = use_mediapipe and _hands_factory is not None
        self.hands = None
        
        if self.use_mediapipe:
            try:
                self.hands = _hands_factory(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.45
                )
            except Exception:
                self.use_mediapipe = False
                self.hands = None

    def get_crop(self, frame):
        """
        Lấy crop bàn tay và bounding box từ frame.
        Trả về: crop_img, (x1, y1, x2, y2), is_fallback_mode
        """
        h, w = frame.shape[:2]

        if self.use_mediapipe and self.hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]

                xmin_px = int(min(xs) * w)
                xmax_px = int(max(xs) * w)
                ymin_px = int(min(ys) * h)
                ymax_px = int(max(ys) * h)

                bw = xmax_px - xmin_px
                bh = ymax_px - ymin_px
                mx = int(bw * 0.15)  # Đồng nhất với crop_dataset.py (15%)
                my = int(bh * 0.15)

                x1 = max(0, xmin_px - mx)
                y1 = max(0, ymin_px - my)
                x2 = min(w, xmax_px + mx)
                y2 = min(h, ymax_px + my)

                if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                    return frame[y1:y2, x1:x2], (x1, y1, x2, y2), False

        # Fallback Mode: Crop 45% ở giữa khung hình
        cw, ch = int(w * 0.45), int(h * 0.45)
        x1 = max(0, w // 2 - cw // 2)
        y1 = max(0, h // 2 - ch // 2)
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)
        
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2), True

    def close(self):
        if self.hands is not None:
            self.hands.close()

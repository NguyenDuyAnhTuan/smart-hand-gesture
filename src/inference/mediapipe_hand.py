"""
MediaPipe hand landmark detection module.
Provides utilities for extracting hand landmarks from images/frames.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class HandLandmarkDetector:
    """
    Wrapper cho MediaPipe Hands, dùng để extract landmarks từ ảnh hoặc webcam frame.
    """

    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, bgr_image):
        """
        Phát hiện hand landmarks từ ảnh BGR.

        Returns:
            landmarks (list of float): 21*3=63 giá trị [x0,y0,z0,...,x20,y20,z20],
                                       hoặc None nếu không tìm thấy bàn tay.
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        row = []
        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])
        return row

    def draw(self, bgr_image):
        """Vẽ landmarks lên ảnh (để debug/hiển thị)"""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(bgr_image, hand_lm, mp_hands.HAND_CONNECTIONS)
        return bgr_image

    def close(self):
        self.hands.close()

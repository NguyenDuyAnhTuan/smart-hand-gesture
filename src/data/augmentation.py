"""
Data Augmentation module for hand gesture dataset.
Applies random transformations to increase dataset diversity.
"""

import cv2
import numpy as np
import random


def flip_horizontal(image):
    """Lật ảnh ngang"""
    return cv2.flip(image, 1)


def rotate_image(image, angle_range=(-15, 15)):
    """Xoay ảnh ngẫu nhiên trong khoảng góc cho trước"""
    angle = random.uniform(*angle_range)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def adjust_brightness(image, factor_range=(0.7, 1.3)):
    """Điều chỉnh độ sáng ngẫu nhiên"""
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_gaussian_noise(image, mean=0, std=10):
    """Thêm nhiễu Gaussian"""
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def augment(image, augmentations=None):
    """
    Áp dụng tập hợp các augmentation lên ảnh.
    
    Args:
        image: ảnh BGR (numpy array)
        augmentations: list các hàm augmentation, mặc định dùng tất cả
    
    Returns:
        ảnh đã được augment
    """
    if augmentations is None:
        augmentations = [flip_horizontal, rotate_image, adjust_brightness, add_gaussian_noise]

    for aug_fn in augmentations:
        if random.random() > 0.5:
            image = aug_fn(image)
    return image

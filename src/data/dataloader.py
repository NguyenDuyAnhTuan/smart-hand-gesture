"""
dataloader.py - Định nghĩa PyTorch Dataset và DataLoader để tải ảnh bàn tay đã cắt thô
cho việc huấn luyện mô hình Vision Transformer (ViT).
"""

import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Optional

from src.utils.config import Config


class HandGestureImageDataset(Dataset):
    """
    Dataset tùy chỉnh để tải ảnh cử chỉ tay đã được tiền xử lý cắt thô.
    """

    def __init__(self, root_dir: str, class_names: list, use_augment: bool = False, img_size: int = 224, return_paths: bool = False, targeted_augment_classes: Optional[List[str]] = None):
        """
        Args:
            root_dir: Thư mục chứa các nhãn cử chỉ (ví dụ: dataset/processed_cropped/train)
            class_names: Danh sách tên nhãn cử chỉ để lấy index làm nhãn số
            use_augment: Có áp dụng Data Augmentation (torchvision) hay không
            img_size: Kích thước ảnh đầu ra
            return_paths: Có trả thêm đường dẫn ảnh gốc hay không
            targeted_augment_classes: Danh sách lớp sẽ được augment mạnh hơn
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.use_augment = use_augment
        self.img_size = img_size
        self.return_paths = return_paths
        self.targeted_augment_labels = set(targeted_augment_classes or [])

        self.img_paths = []
        self.labels = []

        # Ánh xạ tên nhãn sang số nguyên
        self.label_to_idx = {name: i for i, name in enumerate(class_names)}

        # Quét các thư mục nhãn cử chỉ
        for label in class_names:
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for img_name in os.listdir(label_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.img_paths.append(os.path.join(label_dir, img_name))
                    self.labels.append(self.label_to_idx[label])

        # Định nghĩa các PyTorch transforms cơ bản (chuyển sang tensor và chuẩn hóa ImageNet)
        self.targeted_torch_transform = None
        if use_augment:
            self.targeted_torch_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),  # Cần thiết vì webcam lật gương frame
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.25,
                    hue=0.05,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                    shear=4,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        if use_augment:
            self.torch_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                # ── Augmentation mạnh cho webcam generalization ──
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                ),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            ])
        else:
            self.torch_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


        print(f"Loaded {len(self.img_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            # Fallback nếu lỗi đọc ảnh
            img = Image.new("RGB", (self.img_size, self.img_size))
        else:
            # Chuyển BGR → RGB → PIL; augmentation xử lý bởi torchvision transforms
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        if self.use_augment and self.targeted_torch_transform is not None and self.class_names[label] in self.targeted_augment_labels:
            img_tensor = self.targeted_torch_transform(img)
        else:
            img_tensor = self.torch_transform(img)
        if self.return_paths:
            return img_tensor, label, img_path
        return img_tensor, label


def get_dataloaders(cfg: Config):
    """
    Tạo train_loader và val_loader cho quá trình huấn luyện từ thư mục dataset/processed_cropped.
    """
    # Trỏ đến thư mục ảnh đã cắt
    cropped_base_dir = cfg.dataset_processed_cropped_dir
    train_dir = os.path.join(cropped_base_dir, "train")
    val_dir = os.path.join(cropped_base_dir, "val")

    # Tạo train dataset (có augmentation) và val dataset (không augmentation)
    train_dataset = HandGestureImageDataset(
        root_dir=train_dir,
        class_names=cfg.class_names,
        use_augment=cfg.use_augmentation,
        img_size=cfg.img_size,
        targeted_augment_classes=getattr(cfg, "targeted_augmentation_classes", []),
    )

    val_dataset = HandGestureImageDataset(
        root_dir=val_dir,
        class_names=cfg.class_names,
        use_augment=False,
        img_size=cfg.img_size
    )

    # Khởi tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # Đặt bằng 0 để tránh lỗi multiprocessing trên Windows
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader
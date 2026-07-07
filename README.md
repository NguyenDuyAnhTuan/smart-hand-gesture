# Smart Hand Gesture Recognition

Hệ thống nhận diện **21 cử chỉ tay theo bảng chữ cái ngôn ngữ ký hiệu (ASL)** theo thời gian thực, sử dụng **MediaPipe Hands** để phát hiện và cắt vùng bàn tay, kết hợp **MobileNetV3** (Transfer Learning) để phân loại.

## Kiến trúc Pipeline

```
Video/Webcam/Ảnh
    │
    ▼
MediaPipe Hands ──► Phát hiện bàn tay & crop vùng bàn tay (224×224)
    │                   └─ Fallback: Center Crop 45% nếu không detect được
    ▼
MobileNetV3 (Small/Large) ──► Phân loại cử chỉ (21 nhãn)
    │
    ▼
Nhãn dự đoán + Độ tin cậy (%)
```

## Cấu trúc thư mục

```
smart-hand-gesture/
├── dataset/
│   ├── raw/                    Ảnh gốc phân theo thư mục nhãn (A/, B/, ...)
│   ├── processed/              Ảnh đã chia train/ val/ test/
│   └── processed_cropped/      Ảnh bàn tay đã cắt — đầu vào cho training
├── outputs/
│   ├── logs/                   Log huấn luyện (train.log)
│   └── evaluation/             Kết quả đánh giá model
│       ├── confusion_matrix.png
│       ├── confusion_matrix.csv
│       ├── confidence_histogram.png
│       ├── classification_report.csv
│       └── misclassified/      Ảnh dự đoán sai, phân theo true_label/pred_label/
├── saved_models/               Checkpoint model (.pth)
├── src/
│   ├── data/
│   │   ├── crop_dataset.py     Cắt vùng bàn tay từ dataset/processed → processed_cropped
│   │   └── dataloader.py       PyTorch Dataset + DataLoader (có Targeted Augmentation)
│   ├── inference/
│   │   ├── mediapipe_hand.py   HandDetectorWrapper (MediaPipe + Fallback Center Crop)
│   │   ├── predict.py          GesturePredictor — tự phát hiện kiến trúc từ checkpoint
│   │   └── webcam.py           Demo webcam thời gian thực
│   ├── models/
│   │   ├── model_factory.py    Factory: MobileNetV3 Small/Large (Transfer Learning)
│   │   ├── train_model.py      Training loop: AdamW, ReduceLROnPlateau, Early Stopping
│   │   └── evaluate_model.py   Đánh giá trên tập test, xuất báo cáo và biểu đồ
│   └── utils/
│       ├── config.py           Cấu hình tập trung toàn dự án
│       └── helper.py           Logger, seed, save_checkpoint
├── tests/
│   └── test_smoke.py           Kiểm thử cơ bản (import, config, forward pass)
├── main.py                     Entry point
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

> **Lưu ý phiên bản Python**: MediaPipe hoạt động tốt nhất với Python **3.10 hoặc 3.11**. Nếu bạn dùng Python 3.12+, hãy tạo virtualenv với phiên bản phù hợp.

## Chuẩn bị dữ liệu

### 1. Đặt ảnh gốc vào `dataset/raw/`
Cấu trúc bên trong `raw/` phải theo từng nhãn:
```
dataset/raw/
    A/   0001.jpg  0002.jpg  ...
    B/   0001.jpg  0002.jpg  ...
    ...
```

### 2. Chia tập dữ liệu (train/val/test)
Tự chia thủ công theo tỉ lệ **70% Train - 20% Val - 10% Test** vào:
```
dataset/processed/train/  val/  test/
```

### 3. Crop vùng bàn tay
Bước này được **tự động thực hiện** khi bạn chạy lệnh `train`. Script sẽ dùng MediaPipe để cắt vùng bàn tay từ `dataset/processed` và lưu vào `dataset/processed_cropped`. Ảnh không phát hiện được bàn tay sẽ bị bỏ qua để đảm bảo dữ liệu sạch.

Bạn cũng có thể chạy thủ công:
```bash
python -m src.data.crop_dataset
```

## Huấn luyện

```bash
python main.py --mode train
```

Lệnh này sẽ:
1. Kiểm tra `dataset/processed_cropped`. Nếu trống, tự động chạy crop.
2. Tạo `DataLoader` với **Targeted Augmentation** cho các lớp khó phân biệt (`R`, `U`).
3. Fine-tune **MobileNetV3** (mặc định: `mobilenet_v3_small` dùng pretrained ImageNet): mở 2 block cuối và classifier head.
4. Lưu checkpoint sau mỗi epoch cải thiện vào thư mục `saved_models/` (ví dụ: `mobilenet_v3_small_best.pth`).
5. Dừng sớm (Early Stopping) nếu `val_acc` không cải thiện sau **8 epochs** liên tiếp.

### Tùy chọn dòng lệnh

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--epochs N` | 50 | Ghi đè số epoch tối đa |

Ví dụ:
```bash
python main.py --mode train --epochs 30
```

### Thay đổi model

Sửa `model_name` trong [`src/utils/config.py`](src/utils/config.py):
```python
model_name: str = "mobilenet_v3_small"   # hoặc "mobilenet_v3_large"
```

## Demo Webcam (Thời gian thực)

```bash
python main.py --mode webcam
```

Hệ thống sẽ:
- Tự động tìm checkpoint tốt nhất trong `saved_models/` tương ứng với `model_name` cấu hình.
- Dùng MediaPipe để detect và crop bàn tay theo từng frame.
- Lọc nhiễu nhãn bằng hàng đợi 7 frame + **ngưỡng tin cậy tối thiểu 50%**.
- Bỏ qua frame inference (`PROCESS_EVERY_N_FRAMES = 2`) để giữ FPS mượt.
- Nhấn **`Q`** để thoát.

## Dự đoán từ ảnh tĩnh

```bash
python main.py --mode predict --image path/to/image.jpg
```

Tùy chọn chỉ định model cụ thể:
```bash
python main.py --mode predict --image path/to/image.jpg --model saved_models/mobilenet_v3_small_best.pth
```

## Đánh giá mô hình

```bash
python -m src.models.evaluate_model
```

Lệnh này đánh giá trên tập `dataset/processed_cropped/test/` và tạo ra trong `outputs/evaluation/`:

| File | Nội dung |
|---|---|
| `confusion_matrix.png` | Ma trận nhầm lẫn trực quan |
| `confusion_matrix.csv` | Ma trận nhầm lẫn dạng CSV |
| `confidence_histogram.png` | Biểu đồ phân phối độ tin cậy (đúng vs sai) |
| `classification_report.csv` | Precision / Recall / F1 theo từng nhãn |
| `misclassified/` | Ảnh dự đoán sai, phân theo `true_label/pred_label/` |

Kết quả in ra: **Accuracy**, **Top-2 Accuracy**, Confidence trung bình.

## Kết quả thực nghiệm (MobileNetV3 Small)

| Metric | Giá trị |
|---|---|
| **Accuracy (Test)** | **97.82%** |
| Top-2 Accuracy | ~99%+ |
| Số lớp | 21 |
| Số ảnh test | 965 |

Các nhãn yếu nhất (do tương đồng thị giác cao):

| Cặp nhãn | Lý do dễ nhầm |
|---|---|
| **R ↔ U** | Ngón trỏ + ngón giữa, chỉ khác chỗ hai ngón đan chéo hay không |
| **M ↔ N** | Nắm đấm, khác ở ngón cái kẹp dưới 3 hay 2 ngón |
| **D ↔ U** | Góc nhìn nhất định trông rất giống nhau |

## Các nhãn cử chỉ (21 lớp)

```
A  B  C  D  E  F  G  H  I  L  M  N  O  P  R  S  T  U  V  W  Y
```

*(Bảng chữ cái ASL tĩnh — không bao gồm các ký tự yêu cầu chuyển động: J, K, Q, X, Z)*

## Kiểm thử

```bash
python -m unittest tests/test_smoke.py -v
```

## Cấu hình nhanh (`src/utils/config.py`)

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `model_name` | `"mobilenet_v3_small"` | Kiến trúc model (`mobilenet_v3_small` hoặc `mobilenet_v3_large`) |
| `epochs` | `50` | Số epoch tối đa |
| `batch_size` | `32` | Kích thước batch |
| `learning_rate` | `3e-4` | LR cho classifier head |
| `fine_tune_last_blocks` | `2` | Số block cuối backbone mở fine-tune |
| `early_stopping_patience` | `8` | Số epoch chịu đựng không cải thiện |
| `targeted_augmentation_classes` | `["R", "U"]` | Nhãn áp dụng augmentation nhẹ |

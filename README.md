# 🤚 Smart Hand Gesture Recognition

Nhận diện cử chỉ tay theo thời gian thực sử dụng **MediaPipe** + **Vision Transformer (ViT)**.

## 📁 Cấu trúc thư mục

```
smart-hand-gesture/
├── dataset/
│   ├── raw/              ← Ảnh gốc theo từng nhãn (call, like, stop, ...)
│   └── processed/        ← Ảnh đã xử lý chia train/val/test
├── notebooks/            ← Jupyter notebooks thử nghiệm
├── src/
│   ├── data/
│   │   ├── preprocessing.py   ← Trích xuất frame từ video
│   │   ├── augmentation.py    ← Data augmentation
│   │   └── dataloader.py      ← Extract landmarks & tạo CSV
│   ├── models/
│   │   ├── vit_model.py       ← Kiến trúc Vision Transformer
│   │   └── train_model.py     ← Training loop
│   ├── inference/
│   │   ├── mediapipe_hand.py  ← Wrapper MediaPipe Hands
│   │   ├── predict.py         ← Dự đoán từ ảnh/landmark
│   │   └── webcam.py          ← Demo real-time qua webcam
│   └── utils/
│       ├── config.py          ← Cấu hình toàn project
│       └── helper.py          ← Logging, checkpoint, seed
├── saved_models/         ← Model đã train (.pth)
├── outputs/
│   ├── logs/             ← Log training
│   ├── plots/            ← Biểu đồ loss/accuracy
│   └── results/          ← gesture_dataset.csv
├── requirements.txt
└── main.py               ← Entry point
```

## 🚀 Cài đặt

```bash
pip install -r requirements.txt
```

## 🎯 Các nhãn cử chỉ (10 classes)

| Nhãn | Mô tả |
|------|--------|
| `call` | Điện thoại |
| `dislike` | Không thích |
| `like` | Thích |
| `ok` | OK |
| `one` | Số 1 |
| `punch` | Đấm |
| `rock` | Rock |
| `sayhi` | Chào |
| `stop` | Dừng |
| `tym` | Tim |

## 🧠 Quy trình

```
Video → frame.py → Ảnh thô → land_mark.py → CSV landmarks → ViT training → Inference
```

## ▶️ Chạy Demo Webcam

```bash
python main.py --mode webcam
```

## 🏋️ Training

```bash
python main.py --mode train
```

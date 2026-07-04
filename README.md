# Smart Hand Gesture Recognition

Dự án nhận diện cử chỉ tay theo thời gian thực bằng `MediaPipe` để phát hiện tay và `Vision Transformer (ViT)` để phân loại ảnh bàn tay đã được cắt vùng.

## Cấu trúc chính

```text
dataset/
  raw/                Ảnh gốc theo từng lớp
  processed/          Dữ liệu đã chia `train/val/test`
  processed_cropped/   Ảnh tay đã cắt từ `processed`
outputs/
  logs/               Log huấn luyện
  results/            CSV thống kê dữ liệu
saved_models/         Checkpoint model `.pth`
src/
  data/               Tiền xử lý, augmentation, dataloader
  inference/          Detect tay, dự đoán, demo webcam
  models/             ViT và training loop
  utils/              Config, logger, checkpoint
main.py               Entry point
```

## Luồng chạy thực tế

Pipeline hiện tại là:

`Video/Image -> MediaPipe detect tay -> crop vùng tay -> ViT classify -> nhãn dự đoán`

Phần dự đoán landmark trực tiếp đã được giữ lại ở mức khung kỹ thuật, nhưng hiện chưa dùng trong luồng chạy chính.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Dữ liệu và checkpoint cần có

Để chạy được đầy đủ, bạn cần:

1. Dữ liệu ảnh trong `dataset/raw` hoặc `dataset/processed`.
2. Ảnh đã cắt tay trong `dataset/processed_cropped` cho training.
3. Checkpoint sau khi train, thường là `saved_models/vit_best.pth`.

Nếu chưa có checkpoint, hãy train trước.

## Train

```bash
python main.py --mode train
```

Lệnh này sẽ:

1. Kiểm tra `dataset/processed_cropped`.
2. Nếu thư mục cắt tay chưa có dữ liệu, tự chạy `src/data/crop_dataset.py`.
3. Tạo `DataLoader`.
4. Train ViT và lưu:
   - `saved_models/vit_last.pth`
   - `saved_models/vit_best.pth`

## Webcam demo

```bash
python main.py --mode webcam
```

Demo này cần có `saved_models/vit_best.pth`.

## Dự đoán từ ảnh

```bash
python main.py --mode predict --image path/to/image.jpg --model saved_models/vit_best.pth
```

Luồng này sẽ:

1. Đọc ảnh bằng OpenCV.
2. Dùng MediaPipe tìm vùng tay.
3. Crop bàn tay nếu phát hiện được.
4. Đưa ảnh crop vào ViT để dự đoán.

## Các lớp nhãn

`call`, `dislike`, `like`, `ok`, `one`, `punch`, `rock`, `sayhi`, `stop`, `tym`

## Ghi chú

`main.py` đã được sửa để import `os` đúng cách. Trước đó, chế độ `train` và `predict` có thể lỗi ngay ở bước kiểm tra file/thư mục.

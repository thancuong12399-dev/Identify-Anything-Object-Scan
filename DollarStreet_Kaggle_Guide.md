# 📖 Hướng Dẫn Toàn Diện: Huấn Luyện YOLOv8 Trên Kaggle GPU & Tích Hợp Phần Mềm

Tài liệu này hướng dẫn chi tiết từng bước cách huấn luyện mô hình nhận diện đồ vật **Dollar Street** trên Kaggle miễn phí bằng GPU và nạp vào phần mềm Desktop.

---

## 📑 Mục lục
1. [Bước 1: Chuẩn bị tệp dữ liệu YOLO (.zip)](#bước-1-chuẩn-bị-tệp-dữ-liệu-yolo-zip)
2. [Bước 2: Tạo Dataset trên Kaggle](#bước-2-tạo-dataset-trên-kaggle)
3. [Bước 3: Tạo Notebook và Kích hoạt GPU](#bước-3-tạo-notebook-và-kích-hoạt-gpu)
4. [Bước 4: Huấn luyện và Theo dõi Kết quả](#bước-4-huấn-luyện-và-theo-dõi-kết-quả)
5. [Bước 5: Tải trọng số `best.pt` và Chạy trên Phần mềm Desktop](#bước-5-tải-trọng-số-bestpt-và-chạy-trên-phần-mềm-desktop)

---

## Bước 1: Chuẩn bị tệp dữ liệu YOLO (.zip)

Bạn có thể xuất file zip trực tiếp từ phần mềm **Dollar Street AI Studio** (Tab **Xuất dữ liệu Kaggle**) hoặc chạy lệnh sau trong terminal:

```bash
python -c "
from dataset_utils import export_to_yolo_dataset, TOP_20_CLASSES
zip_p = export_to_yolo_dataset(
    dataset_dir=r'd:\AI\Dollar street trial',
    output_dir=r'd:\AI\dollar_street_yolo',
    target_classes=TOP_20_CLASSES,
    train_ratio=0.8,
    max_images_per_class=100,
    create_zip=True
)
print('Đã tạo file zip:', zip_p)
"
```
Kết quả sẽ tạo ra file: `d:\AI\dollar_street_yolo.zip` dung lượng khoảng 100 - 150 MB.

---

## Bước 2: Tạo Dataset trên Kaggle

1. Mở trình duyệt và truy cập: [https://www.kaggle.com](https://www.kaggle.com) (Đăng nhập tài khoản Kaggle của bạn).
2. Ở thanh menu bên trái, chọn **Datasets** -> Bấm nút **+ New Dataset** (hoặc truy cập: `https://www.kaggle.com/datasets/new`).
3. Đặt tiêu đề cho Dataset: `dollar-street-yolo`.
4. Kéo thả file `dollar_street_yolo.zip` vào khung tải lên (hoặc bấm **Browse Files**).
5. Sau khi thanh tải lên đạt 100%, bấm nút **Create** ở góc dưới bên phải. Chờ vài giây để Kaggle xử lý xong.

---

## Bước 3: Tạo Notebook và Kích hoạt GPU

1. Trên menu trái Kaggle, chọn **Code** -> Bấm nút **+ New Notebook**.
2. **Kích hoạt GPU (Rất quan trọng):**
   - Ở thanh điều khiển bên phải (Notebook options):
     - **Accelerator**: Chọn **GPU T4 x2** (hoặc GPU P100).
     - **Internet**: Bật **On** (để cài đặt thư viện `ultralytics`).
3. **Nạp Notebook có sẵn:**
   - Trên thanh menu của Notebook: Chọn **File** -> **Import Notebook** -> Chọn file:
     `d:\AI\DollarStreet_YOLO_Training_Kaggle.ipynb`
4. **Liên kết Dataset vừa tạo:**
   - Ở bảng bên phải, bấm nút **+ Add Input** (hoặc **Add Data**).
   - Chọn tab **Your Datasets** -> Bấm dấu `+` cạnh dataset `dollar-street-yolo`.

---

## Bước 4: Huấn luyện và Theo dõi Kết quả

1. Bấm nút **Run All** (hoặc tổ hợp phím `Shift + Enter` chạy từng ô mã).
2. Quy trình sẽ tự động:
   - Kiểm tra GPU T4 và cài đặt Ultralytics.
   - Giải nén dataset vào `/kaggle/working/dataset/`.
   - Cập nhật file `data.yaml`.
   - Huấn luyện YOLOv8s trong 35 epochs (mất khoảng **5 đến 10 phút** trên GPU).
   - Xuất biểu đồ mAP50, Loss và ma trận nhầm lẫn (Confusion Matrix).
   - Thử nghiệm nhận diện trên ảnh kiểm thử validation.
   - Tự động sao chép file mô hình tốt nhất ra `/kaggle/working/best.pt`.

---

## Bước 5: Tải trọng số `best.pt` và Chạy trên Phần mềm Desktop

1. **Tải file về máy tính:**
   - Ở cột bên phải của Kaggle, mở mục **Output** (`/kaggle/working/`).
   - Tìm file `best.pt` -> Bấm vào dấu **3 chấm (...)** cạnh file -> Chọn **Download**.
   - Lưu file `best.pt` vào thư mục dự án của bạn (ví dụ `d:\AI\best.pt`).

2. **Khởi động phần mềm Desktop:**
   - Chạy lệnh: `python app.py` (hoặc nhấp đúp file `run_app.bat`).
   - Vào Tab **🎯 Nhận diện AI / Kiểm thử Mô hình**.
   - Bấm nút **📂 Chọn mô hình (.pt)** và chọn file `best.pt` vừa tải về.
   - Thử nghiệm:
     - **Upload ảnh**: Chọn bất kỳ ảnh nào để xem bounding box và độ tin cậy.
     - **Bật Webcam**: Bấm nút **🎥 Bật Webcam** để trải nghiệm nhận diện thời gian thực với FPS cao!

---
💡 *Mẹo: Nếu bạn muốn kiểm thử phần mềm ngay lập tức mà chưa kịp train trên Kaggle, phần mềm có sẵn tùy chọn dùng mô hình YOLOv8 pretrained (COCO) để trải nghiệm giao diện và webcam trước!*

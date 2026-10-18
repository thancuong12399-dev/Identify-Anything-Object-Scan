# 🌍 Identify Anything - Dollar Street Object Scan (YOLOv8)

Ứng dụng nhận diện và phát hiện vật thể thông minh sử dụng **YOLOv8** được huấn luyện trên tập dữ liệu **Dollar Street** (vật thể sinh hoạt gia đình từ khắp nơi trên thế giới) với giao diện desktop hiện đại viết bằng **CustomTkinter** hoàn toàn bằng **Tiếng Việt**.

---

## 📌 Tính Năng Nổi Bật

- **🎯 Nhận diện hình ảnh & Webcam thời gian thực**:
  - Nhận diện trực tiếp qua Webcam máy tính với tốc độ khung hình cao (FPS cao, độ trễ thấp).
  - Tải ảnh từ máy tính hoặc thử nhanh ảnh ngẫu nhiên từ tập dữ liệu.
  - Vẽ khung Bounding Box sắc nét, hiển thị tên vật thể bằng **Tiếng Việt** và độ tin cậy `%`.
  - Hỗ trợ lưu ảnh chụp kết quả nhận diện ra ổ đĩa.
- **📊 Khám phá Dataset (Dataset Explorer)**:
  - Duyệt và trực quan hóa toàn bộ ảnh cùng nhãn chú thích Bounding Box gốc từ tập dữ liệu Dollar Street.
- **⚡ Xuất dữ liệu YOLO & Đóng gói ZIP**:
  - Tự động chuyển đổi định dạng Supervisely sang chuẩn YOLO (`images/`, `labels/`, `data.yaml`).
  - Lọc Top 20 nhãn phổ biến nhất để huấn luyện cân bằng, nhanh chóng.
- **☁️ Tích hợp Huấn luyện Kaggle GPU**:
  - Kèm sẵn Notebook Jupyter (`DollarStreet_YOLO_Training_Kaggle.ipynb`) tối ưu cho GPU Kaggle (T4 x2 / P100).

---

## 🏷️ Danh Sách 20 Đối Tượng Nhận Diện (Top 20 Classes)

| STT | Tên Tiếng Việt | Tên Tiếng Anh | STT | Tên Tiếng Việt | Tên Tiếng Anh |
|---|---|---|---|---|---|
| 1 | **Điện thoại** | Phone | 11 | **Xà phòng** | Soap |
| 2 | **Máy tính** | Computer | 12 | **Tủ quần áo** | Wardrobe |
| 3 | **Ô tô** | Car | 13 | **Giá để bát đĩa** | Dish rack |
| 4 | **Xe đạp** | Bike | 14 | **Bồn rửa bát** | Kitchen sink |
| 5 | **Giường ngủ** | Bed | 15 | **Đĩa ăn** | Plate |
| 6 | **Bồn cầu** | Toilet | 16 | **Vòi hoa sen** | Shower |
| 7 | **Giày dép** | Shoe | 17 | **Bát ăn** | Bowl |
| 8 | **Bếp nấu** | Stove | 18 | **Cốc nước** | Cup |
| 9 | **Bàn chải đánh răng** | Toothbrush | 19 | **Tivi** | TV |
| 10 | **Nồi nấu** | Cooking pot | 20 | **Bàn tay** | Hand |

---

## 🚀 Cài Đặt & Khởi Chạy

### 1. Yêu cầu hệ thống
- Python 3.10 trở lên
- Windows 10/11

### 2. Cài đặt các thư viện cần thiết
```bash
pip install ultralytics customtkinter opencv-python pillow pyyaml
```

### 3. Khởi chạy phần mềm
- **Cách 1**: Nhấp đúp chuột vào tệp `run_app.bat`.
- **Cách 2**: Chạy lệnh trong Terminal:
  ```bash
  python app.py
  ```

---

## ☁️ Hướng Dẫn Huấn Luyện Lại Bằng Kaggle GPU

1. Đưa tập dữ liệu lên Kaggle hoặc liên kết với tập dữ liệu *Daily objects around the world*.
2. Nạp file notebook `DollarStreet_YOLO_Training_Kaggle.ipynb` vào Kaggle Code.
3. Bật **GPU T4 x2** và bật **Internet: On**.
4. Bấm **Run All**, sau khi hoàn thành tải file trọng số `best.pt` về thư mục ứng dụng.

---

## 📁 Cấu Trúc Dự Án

```
├── app.py                                   # Phần mềm giao diện Desktop chính (CustomTkinter)
├── run_app.bat                              # File nhấp đúp 1-click khởi chạy phần mềm
├── best.pt                                  # Trọng số mô hình YOLOv8s đã huấn luyện
├── dataset_utils.py                         # Module xử lý dữ liệu và chuyển đổi YOLO
├── export_dataset.py                        # Script xuất dữ liệu dòng lệnh
├── DollarStreet_YOLO_Training_Kaggle.ipynb  # Notebook huấn luyện tự động trên Kaggle
├── DollarStreet_Kaggle_Guide.md             # Hướng dẫn chi tiết từng bước cho Kaggle
└── .gitignore                               # Loại trừ dữ liệu lớn và file tạm
```

---

*Phát triển bởi [thancuong12399-dev](https://github.com/thancuong12399-dev).*

import sys
import os
import time

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dataset_utils import export_to_yolo_dataset, TOP_20_CLASSES

if __name__ == "__main__":
    t0 = time.time()
    print("=== BẮT ĐẦU XUẤT DỮ LIỆU SANG YOLO CHO KAGGLE ===")
    print(f"Danh sách Top 20 nhãn: {TOP_20_CLASSES}")
    
    def progress_callback(cur, total, msg):
        pct = int(cur / total * 100)
        print(f"[{pct}%] {msg}")

    zip_res = export_to_yolo_dataset(
        dataset_dir=r"d:\AI\Dollar street trial",
        output_dir=r"d:\AI\dollar_street_yolo",
        target_classes=TOP_20_CLASSES,
        train_ratio=0.8,
        max_images_per_class=100,
        progress_callback=progress_callback,
        create_zip=True
    )
    t1 = time.time()
    size_mb = os.path.getsize(zip_res) / (1024 * 1024)
    print(f"\n🎉 HOÀN TẤT! File ZIP: {zip_res} ({size_mb:.2f} MB)")
    print(f"Thời gian xử lý: {t1 - t0:.2f} giây.")

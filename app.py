"""
Dollar Street AI Studio - Phần Mềm Nhận Diện Vật Thể & Khám Phá Dataset
Tích hợp nhận diện trực tiếp qua Camera (Webcam / IP Camera) thời gian thực
Giao diện và nhãn hiển thị 100% Tiếng Việt
"""

import os
import sys
import glob
import time
import json
import random
import queue
import datetime
import threading
import webbrowser
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Import custom dataset utilities
from dataset_utils import (
    TOP_20_CLASSES,
    TOP_10_CLASSES,
    get_all_dataset_classes,
    parse_supervisely_annotation,
    export_to_yolo_dataset,
    CLASS_TO_FOLDERS
)

# Đường dẫn cấu hình mặc định
BASE_DIR = r"d:\AI"
DATASET_DEFAULT_DIR = os.path.join(BASE_DIR, "Dollar street trial")
DEFAULT_OUTPUT_ZIP = os.path.join(BASE_DIR, "dollar_street_yolo.zip")
KAGGLE_NOTEBOOK_PATH = os.path.join(BASE_DIR, "DollarStreet_YOLO_Training_Kaggle.ipynb")
GUIDE_MD_PATH = os.path.join(BASE_DIR, "DollarStreet_Kaggle_Guide.md")
DEFAULT_WEIGHTS_PATH = os.path.join(BASE_DIR, "best.pt")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)

# Từ điển Việt hóa toàn diện 94 danh mục Dollar Street và 80 lớp COCO
VIETNAMESE_CLASSES = {
    # Dollar Street 20 Classes chính
    "Phone": "Điện thoại",
    "Computer": "Máy tính",
    "Car": "Ô tô",
    "Bike": "Xe đạp",
    "Bed": "Giường ngủ",
    "Toilet": "Bồn cầu",
    "Shoe": "Giày dép",
    "Stove": "Bếp nấu",
    "Toothbrush": "Bàn chải đánh răng",
    "Cooking pot": "Nồi nấu",
    "Soap": "Xà phòng",
    "Wardrobe": "Tủ quần áo",
    "Dish rack": "Giá để bát đĩa",
    "Kitchen sink": "Bồn rửa bát",
    "Plate": "Đĩa ăn",
    "Shower": "Vòi hoa sen",
    "Bowl": "Bát ăn",
    "Cup": "Cốc nước",
    "TV": "Tivi",
    "Hand": "Bàn tay",

    # Các lớp Dollar Street khác
    "Armchair": "Ghế bành",
    "Sofa": "Ghế sofa",
    "Book": "Sách",
    "Radio": "Đài radio",
    "Wall clock": "Đồng hồ treo tường",
    "Refrigerator": "Tủ lạnh",
    "Freezers": "Tủ đông",
    "Oven": "Lò nướng",
    "Dishwasher": "Máy rửa bát",
    "Pet": "Thú cưng",
    "Meat": "Thịt",
    "Vegetable": "Rau củ",
    "Fruits": "Hoa quả",
    "Front door": "Cửa chính",
    "Bathroom door": "Cửa phòng tắm",
    "Power outlet": "Ổ cắm điện",
    "Power switch": "Công tắc điện",
    "Toothpaste": "Kem đánh răng",
    "Shampoo": "Dầu gội đầu",
    "Washing detergent": "Bột giặt",
    "Toilet paper": "Giấy vệ sinh",
    "Cutlery": "Dao thìa dĩa",
    "Trash": "Thùng rác",
    "Toy": "Đồ chơi",
    "Jacket": "Áo khoác",
    "Glasses": "Kính mắt",
    "Pen": "Bút viết",
    "Wrist watch": "Đồng hồ đeo tay",
    "Jewelry": "Trang sức",
    "Motorcycle": "Xe máy",
    "Wheel barrow": "Xe cút kít",
    "Car key": "Chìa khóa xe",
    "Frontdoor keys": "Chìa khóa cửa",
    "Lock on front door": "Khóa cửa",
    "Rug": "Thảm trải sàn",
    "Water outlet": "Vòi nước",
    "Light sources": "Nguồn sáng / Đèn",
    "Light sources kitchen": "Đèn bếp",
    "Light sources in living room": "Đèn phòng khách",
    "Light sources by bed, reading light": "Đèn ngủ",
    "Cleaning equipment": "Dụng cụ vệ sinh",
    "Cleaning floor": "Lau dọn sàn",
    "Cooking utensils": "Dụng cụ nấu ăn",
    "Plates of food": "Đĩa thức ăn",
    "Social drinks": "Đồ uống giao lưu",
    "Alcoholic drink": "Đồ uống có cồn",
    "Spices": "Gia vị",
    "Salt": "Muối",
    "Grains": "Ngũ cốc",
    "Make up": "Đồ trang điểm",
    "Earings": "Khuyên tai",
    "Necklace": "Vòng cổ",
    "Pet food": "Thức ăn thú cưng",
    "Chicken": "Gà",
    "Goat": "Dê",
    "Fruit trees": "Cây ăn quả",
    "Palms": "Cây cọ / dừa",
    "Home": "Ngôi nhà",
    "Roof": "Mái nhà",
    "Dish brush": "Bàn chải rửa bát",
    "Dish washing soap": "Nước rửa chén",
    "Hair brush": "Lược chải tóc",
    "Teeth": "Răng",
    "Tools": "Công cụ dụng cụ",
    "Medication": "Thuốc y tế",
    "Menstruation pads": "Băng vệ sinh",
    "Diaper": "Tã lót trẻ em",
    "Mosquito protections": "Chống muỗi",
    "Music equipment": "Thiết bị âm nhạc",
    "Instrument": "Nhạc cụ",
    "Paper": "Giấy tờ",
    "Kids bed": "Giường trẻ em",
    "Guest beds": "Giường khách",

    # Hỗ trợ 80 lớp chuẩn COCO (khi dùng các mô hình yolov8n, yolov8s chung)
    "person": "Người",
    "bicycle": "Xe đạp",
    "car": "Ô tô",
    "motorcycle": "Xe máy",
    "airplane": "Máy bay",
    "bus": "Xe buýt",
    "train": "Tàu hỏa",
    "truck": "Xe tải",
    "boat": "Thuyền",
    "traffic light": "Đèn giao thông",
    "fire hydrant": "Trụ cứu hỏa",
    "stop sign": "Biển dừng",
    "bench": "Ghế dài",
    "bird": "Chim",
    "cat": "Mèo",
    "dog": "Chó",
    "horse": "Ngựa",
    "sheep": "Cừu",
    "cow": "Bò",
    "elephant": "Voi",
    "bear": "Gấu",
    "zebra": "Ngựa vằn",
    "giraffe": "Hươu cao cổ",
    "backpack": "Ba lô",
    "umbrella": "Cái ô / Dù",
    "handbag": "Túi xách",
    "tie": "Cà vạt",
    "suitcase": "Va li",
    "bottle": "Chai / Lọ",
    "wine glass": "Ly rượu",
    "cup": "Cốc nước",
    "fork": "Dĩa ăn",
    "knife": "Dao",
    "spoon": "Thìa ăn",
    "bowl": "Bát ăn",
    "banana": "Quả chuối",
    "apple": "Quả táo",
    "sandwich": "Bánh mì kẹp",
    "orange": "Quả cam",
    "broccoli": "Súp lơ",
    "carrot": "Cà rốt",
    "hot dog": "Bánh mì xúc xích",
    "pizza": "Bánh Pizza",
    "donut": "Bánh Donut",
    "cake": "Bánh ngọt",
    "chair": "Cái ghế",
    "couch": "Ghế sofa",
    "potted plant": "Chậu cây",
    "bed": "Giường ngủ",
    "dining table": "Bàn ăn",
    "toilet": "Bồn cầu",
    "tv": "Tivi",
    "laptop": "Máy tính xách tay",
    "mouse": "Chuột máy tính",
    "remote": "Điều khiển",
    "keyboard": "Bàn phím",
    "cell phone": "Điện thoại",
    "microwave": "Lò vi sóng",
    "oven": "Lò nướng",
    "toaster": "Máy nướng bánh mì",
    "sink": "Bồn rửa",
    "refrigerator": "Tủ lạnh",
    "book": "Cuốn sách",
    "clock": "Đồng hồ",
    "vase": "Lọ hoa",
    "scissors": "Cây kéo",
    "teddy bear": "Gấu bông",
    "hair drier": "Máy sấy tóc",
    "toothbrush": "Bàn chải đánh răng"
}

# Biểu tượng Emoji tương ứng cho từng lớp vật thể
CLASS_EMOJIS = {
    "Phone": "📱", "cell phone": "📱",
    "Computer": "💻", "laptop": "💻",
    "Car": "🚗", "car": "🚗",
    "Bike": "🚲", "bicycle": "🚲",
    "Bed": "🛏️", "bed": "🛏️",
    "Toilet": "🚽", "toilet": "🚽",
    "Shoe": "👟",
    "Stove": "🍳",
    "Toothbrush": "🪥", "toothbrush": "🪥",
    "Cooking pot": "🍲",
    "Soap": "🧼",
    "Wardrobe": "🚪",
    "Dish rack": "🍽️",
    "Kitchen sink": "🚰", "sink": "🚰",
    "Plate": "🍽️",
    "Shower": "🚿",
    "Bowl": "🥣", "bowl": "🥣",
    "Cup": "☕", "cup": "☕",
    "TV": "📺", "tv": "📺",
    "Hand": "✋",
    "Book": "📚", "book": "📚",
    "Radio": "📻",
    "Wall clock": "⏰", "clock": "⏰",
    "Refrigerator": "🧊", "refrigerator": "🧊",
    "Freezers": "❄️",
    "Oven": "♨️", "oven": "♨️",
    "Dishwasher": "🧼",
    "Pet": "🐾", "cat": "🐱", "dog": "🐶",
    "Meat": "🥩",
    "Vegetable": "🥦",
    "Fruits": "🍎", "apple": "🍎", "banana": "🍌",
    "Power outlet": "🔌",
    "Power switch": "🔘",
    "Toothpaste": "🪥",
    "Cutlery": "🍴", "fork": "🍴", "knife": "🔪", "spoon": "🥄",
    "Trash": "🗑️",
    "Toy": "🧸", "teddy bear": "🧸",
    "Glasses": "👓",
    "Pen": "🖊️",
    "Wrist watch": "⌚",
    "Motorcycle": "🏍️", "motorcycle": "🏍️",
    "bottle": "🍾",
    "chair": "🪑",
    "couch": "🛋️",
    "person": "👤",
    "backpack": "🎒",
    "mouse": "🖱️",
    "keyboard": "⌨️",
    "remote": "📱"
}

def get_vietnamese_name(eng_name: str) -> str:
    """Chuyển tên đối tượng tiếng Anh sang tiếng Việt linh hoạt."""
    if eng_name in VIETNAMESE_CLASSES:
        return VIETNAMESE_CLASSES[eng_name]
    lower_name = eng_name.lower()
    if lower_name in VIETNAMESE_CLASSES:
        return VIETNAMESE_CLASSES[lower_name]
    title_name = eng_name.title()
    if title_name in VIETNAMESE_CLASSES:
        return VIETNAMESE_CLASSES[title_name]
    return eng_name

# Bảng màu sắc phân biệt cho các nhãn
PALETTE = [
    (59, 130, 246), (16, 185, 129), (239, 68, 68), (245, 158, 11),
    (139, 92, 246), (236, 72, 153), (6, 182, 212), (132, 204, 22),
    (249, 115, 22), (168, 85, 247), (20, 184, 166), (244, 63, 94),
    (14, 165, 233), (101, 163, 13), (217, 70, 239), (234, 88, 12),
    (99, 102, 241), (45, 212, 191), (251, 146, 60), (192, 132, 252)
]

def get_class_color(cls_name: str) -> tuple:
    idx = abs(hash(cls_name)) % len(PALETTE)
    return PALETTE[idx]

# Nạp font chữ tiếng Việt cho PIL
FONT_PATHS = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf"
]
DEFAULT_FONT_PATH = None
for fp in FONT_PATHS:
    if os.path.exists(fp):
        DEFAULT_FONT_PATH = fp
        break


class VoiceAnnouncer:
    """Hệ thống phát âm thanh / đọc tên đồ vật bằng giọng nói không block luồng xử lý."""
    def __init__(self):
        self.queue = queue.Queue()
        self.last_spoken = {}
        self.enabled = False
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def set_enabled(self, val: bool):
        self.enabled = val

    def announce(self, text: str, cooldown: float = 4.0):
        if not self.enabled:
            return
        now = time.time()
        if text in self.last_spoken and (now - self.last_spoken[text]) < cooldown:
            return
        self.last_spoken[text] = now
        if self.queue.qsize() < 3:
            self.queue.put(text)

    def stop(self):
        self.running = False

    def _worker(self):
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 0.9)
        except Exception:
            engine = None

        while self.running:
            try:
                text = self.queue.get(timeout=0.5)
                if not self.running:
                    break
                if engine and self.enabled:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception:
                        pass
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass


class DollarStreetApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Cài đặt cửa sổ chính
        self.title("Dollar Street AI Studio - Phần Mềm Nhận Diện Vật Thể (YOLOv8)")
        self.geometry("1360x880")
        self.minsize(1120, 740)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Trạng thái mô hình
        self.dataset_dir = DATASET_DEFAULT_DIR
        self.current_model = None
        self.model_name = "Chưa nạp mô hình"

        # Trạng thái Camera
        self.camera_running = False
        self.cap = None
        self.last_camera_annotated_frame = None
        self.last_saved_snapshot_path = None
        self.last_detected_classes = set()
        self.available_cam_indices = [0]
        self.voice_announcer = VoiceAnnouncer()

        # Biến điều khiển Camera
        self.camera_source_var = ctk.StringVar(value="Camera 0 (Mặc định)")
        self.camera_res_var = ctk.StringVar(value="1280x720 (HD sắc nét)")
        self.camera_mirror_var = ctk.BooleanVar(value=True)
        self.camera_voice_var = ctk.BooleanVar(value=False)
        self.camera_conf_var = ctk.DoubleVar(value=0.25)

        # Trạng thái nhận diện từ ảnh
        self.last_annotated_img = None

        # Trạng thái xem dữ liệu
        self.explorer_categories = []
        self.current_category = ""
        self.category_images = []
        self.current_img_idx = 0
        self.show_boxes_var = ctk.BooleanVar(value=True)

        # Xây dựng giao diện
        self._build_sidebar()
        self._build_main_views()

        # Nạp mô hình ngầm & chuyển sang màn hình Camera mặc định
        has_best_pt = os.path.exists(DEFAULT_WEIGHTS_PATH)
        self.switch_view("camera")
        if has_best_pt:
            threading.Thread(target=self._auto_load_default_model, daemon=True).start()

        # Quét danh mục dataset ở luồng ngầm
        threading.Thread(target=self._init_dataset_categories, daemon=True).start()

    def _auto_load_default_model(self):
        time.sleep(0.3)
        self._load_model_weights(DEFAULT_WEIGHTS_PATH, silent=True)

    # =========================================================================
    # THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR)
    # =========================================================================
    def _build_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar_frame.pack_propagate(False)

        # Tiêu đề và Logo ứng dụng
        title_lbl = ctk.CTkLabel(
            self.sidebar_frame,
            text="🌍 Dollar Street AI",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_lbl.pack(padx=20, pady=(25, 4))

        subtitle_lbl = ctk.CTkLabel(
            self.sidebar_frame,
            text="Nhận Diện Vật Thể Đời Sống",
            font=ctk.CTkFont(size=12),
            text_color="#9CA3AF"
        )
        subtitle_lbl.pack(padx=20, pady=(0, 20))

        # Các nút điều hướng - Đặt Camera lên vị trí ưu tiên hàng đầu
        self.nav_btns = {}
        nav_items = [
            ("camera", "📷 Nhận diện qua Camera"),
            ("inference", "🖼️ Nhận diện từ Ảnh"),
            ("explorer", "📊 Khám phá Dataset"),
            ("exporter", "⚡ Xuất dữ liệu Kaggle"),
            ("kaggle", "☁️ Hướng dẫn Kaggle")
        ]

        for key, text in nav_items:
            btn = ctk.CTkButton(
                self.sidebar_frame,
                text=text,
                height=44,
                corner_radius=8,
                anchor="w",
                font=ctk.CTkFont(size=13, weight="bold"),
                command=lambda k=key: self.switch_view(k)
            )
            btn.pack(padx=16, pady=5, fill="x")
            self.nav_btns[key] = btn

        # Khoảng trống dãn cách
        spacer = ctk.CTkLabel(self.sidebar_frame, text="")
        spacer.pack(expand=True, fill="both")

        # Thẻ thông báo trạng thái mô hình
        self.model_status_box = ctk.CTkFrame(self.sidebar_frame, corner_radius=8, fg_color="#1E293B")
        self.model_status_box.pack(padx=16, pady=6, fill="x")

        self.sidebar_model_lbl = ctk.CTkLabel(
            self.model_status_box,
            text="🤖 Mô hình: Đang kiểm tra...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#F59E0B",
            wraplength=220,
            justify="left"
        )
        self.sidebar_model_lbl.pack(padx=10, pady=8)

        # Thẻ thông báo trạng thái dữ liệu
        self.status_box = ctk.CTkFrame(self.sidebar_frame, corner_radius=8, fg_color="#1E293B")
        self.status_box.pack(padx=16, pady=6, fill="x")

        self.ds_status_lbl = ctk.CTkLabel(
            self.status_box,
            text="📁 Dữ liệu: Đang nạp danh mục...",
            font=ctk.CTkFont(size=11),
            text_color="#38BDF8",
            wraplength=220,
            justify="left"
        )
        self.ds_status_lbl.pack(padx=10, pady=8)

        # Chuyển đổi giao diện Sáng / Tối
        mode_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        mode_frame.pack(padx=16, pady=(10, 20), fill="x")

        theme_lbl = ctk.CTkLabel(mode_frame, text="Giao diện:", font=ctk.CTkFont(size=12))
        theme_lbl.pack(side="left", padx=5)

        self.theme_menu = ctk.CTkOptionMenu(
            mode_frame,
            values=["Tối (Dark)", "Sáng (Light)"],
            width=110,
            height=28,
            command=self._change_theme
        )
        self.theme_menu.set("Tối (Dark)")
        self.theme_menu.pack(side="right", padx=5)

    def _change_theme(self, mode: str):
        if "Sáng" in mode:
            ctk.set_appearance_mode("Light")
        else:
            ctk.set_appearance_mode("Dark")

    # =========================================================================
    # KHUNG HIỂN THỊ CHÍNH (CONTAINER)
    # =========================================================================
    def _build_main_views(self):
        self.container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.container.pack(side="right", fill="both", expand=True, padx=16, pady=16)

        # Khởi tạo các khung màn hình chức năng
        self.view_camera = ctk.CTkFrame(self.container, corner_radius=12)
        self.view_inference = ctk.CTkFrame(self.container, corner_radius=12)
        self.view_explorer = ctk.CTkFrame(self.container, corner_radius=12)
        self.view_exporter = ctk.CTkFrame(self.container, corner_radius=12)
        self.view_kaggle = ctk.CTkFrame(self.container, corner_radius=12)

        self._setup_view_camera()
        self._setup_view_inference()
        self._setup_view_explorer()
        self._setup_view_exporter()
        self._setup_view_kaggle()

    def switch_view(self, view_name: str):
        # Tự động dừng Camera nếu người dùng chuyển sang tab khác
        if view_name != "camera" and self.camera_running:
            self._stop_camera()

        for v in [self.view_camera, self.view_inference, self.view_explorer, self.view_exporter, self.view_kaggle]:
            v.pack_forget()

        for k, btn in self.nav_btns.items():
            if k == view_name:
                btn.configure(fg_color=["#3B82F6", "#2563EB"], text_color="#FFFFFF")
            else:
                btn.configure(fg_color="transparent", text_color=["#E5E7EB", "#9CA3AF"])

        if view_name == "camera":
            self.view_camera.pack(fill="both", expand=True)
        elif view_name == "inference":
            self.view_inference.pack(fill="both", expand=True)
        elif view_name == "explorer":
            self.view_explorer.pack(fill="both", expand=True)
        elif view_name == "exporter":
            self.view_exporter.pack(fill="both", expand=True)
        elif view_name == "kaggle":
            self.view_kaggle.pack(fill="both", expand=True)

    # =========================================================================
    # CHỨC NĂNG 1: NHẬN DIỆN QUA CAMERA THỜI GIAN THỰC (CAMERA STUDIO)
    # =========================================================================
    def _setup_view_camera(self):
        # 1. Thanh trạng thái Model & Độ tin cậy phía trên
        model_bar = ctk.CTkFrame(self.view_camera, corner_radius=10, fg_color="#1E293B")
        model_bar.pack(fill="x", padx=14, pady=(12, 6))

        self.lbl_cam_model_badge = ctk.CTkLabel(
            model_bar,
            text="🤖 Mô hình: Đang kiểm tra best.pt...",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#F59E0B"
        )
        self.lbl_cam_model_badge.pack(side="left", padx=14, pady=10)

        btn_cam_load_pt = ctk.CTkButton(
            model_bar,
            text="📂 Đổi mô hình (.pt)",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#3B82F6",
            hover_color="#2563EB",
            height=32,
            command=self._load_custom_model_dialog
        )
        btn_cam_load_pt.pack(side="left", padx=6)

        btn_cam_reload_best = ctk.CTkButton(
            model_bar,
            text="⚡ Nạp lại best.pt",
            font=ctk.CTkFont(size=12),
            fg_color="#10B981",
            hover_color="#059669",
            height=32,
            command=lambda: self._load_model_weights(DEFAULT_WEIGHTS_PATH)
        )
        btn_cam_reload_best.pack(side="left", padx=6)

        # Thanh trượt độ tin cậy
        thresh_box = ctk.CTkFrame(model_bar, fg_color="transparent")
        thresh_box.pack(side="right", padx=14)

        self.lbl_cam_conf_disp = ctk.CTkLabel(thresh_box, text="Độ tin cậy: 25%", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_cam_conf_disp.pack(side="left", padx=6)

        self.slider_cam_conf = ctk.CTkSlider(
            thresh_box,
            from_=0.10,
            to=0.90,
            number_of_steps=16,
            width=130,
            command=self._on_cam_conf_change
        )
        self.slider_cam_conf.set(0.25)
        self.slider_cam_conf.pack(side="left", padx=4)

        # 2. Thanh điều khiển Camera chuyên dụng
        ctrl_bar = ctk.CTkFrame(self.view_camera, corner_radius=10, fg_color="#1E293B")
        ctrl_bar.pack(fill="x", padx=14, pady=6)

        # Nút Bắt đầu / Dừng Camera
        self.btn_main_cam_toggle = ctk.CTkButton(
            ctrl_bar,
            text="▶ BẮT ĐẦU CAMERA",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669",
            height=36,
            command=self._toggle_camera
        )
        self.btn_main_cam_toggle.pack(side="left", padx=(12, 8), pady=8)

        # Nút Chụp ảnh
        self.btn_main_cam_snap = ctk.CTkButton(
            ctrl_bar,
            text="📸 Chụp ảnh",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0284C7",
            hover_color="#0369A1",
            height=36,
            command=self._take_camera_snapshot
        )
        self.btn_main_cam_snap.pack(side="left", padx=6, pady=8)

        # Chọn Nguồn Camera
        ctk.CTkLabel(ctrl_bar, text="Nguồn:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(8, 4))
        self.cb_cam_source = ctk.CTkComboBox(
            ctrl_bar,
            values=["Camera 0 (Mặc định)", "Camera 1", "Camera 2", "🌐 URL IP Camera/Điện thoại"],
            width=175,
            height=32,
            command=self._on_cam_source_selected
        )
        self.cb_cam_source.set("Camera 0 (Mặc định)")
        self.cb_cam_source.pack(side="left", padx=4)

        self.btn_cam_scan = ctk.CTkButton(
            ctrl_bar,
            text="🔄 Quét Camera",
            font=ctk.CTkFont(size=11),
            fg_color="#475569",
            hover_color="#334155",
            width=90,
            height=32,
            command=self._scan_available_cameras
        )
        self.btn_cam_scan.pack(side="left", padx=4)

        # Ô nhập URL IP Camera (mặc định ẩn)
        self.entry_cam_url = ctk.CTkEntry(
            ctrl_bar,
            placeholder_text="URL (ví dụ http://192.168.1.15:4747/video)",
            width=230,
            height=32
        )

        # Chọn độ phân giải
        ctk.CTkLabel(ctrl_bar, text="Độ phân giải:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(10, 4))
        self.cb_cam_res = ctk.CTkComboBox(
            ctrl_bar,
            values=["1280x720 (HD sắc nét)", "640x480 (Mượt/Nhanh)"],
            width=160,
            height=32,
            command=self._on_cam_res_changed
        )
        self.cb_cam_res.set("1280x720 (HD sắc nét)")
        self.cb_cam_res.pack(side="left", padx=4)

        # Checkbox Lật gương
        self.chk_cam_mirror = ctk.CTkCheckBox(
            ctrl_bar,
            text="🪞 Lật gương",
            variable=self.camera_mirror_var,
            font=ctk.CTkFont(size=12)
        )
        self.chk_cam_mirror.pack(side="left", padx=10)

        # Checkbox Giọng đọc tên vật thể
        self.chk_cam_voice = ctk.CTkCheckBox(
            ctrl_bar,
            text="🔊 Giọng đọc",
            variable=self.camera_voice_var,
            font=ctk.CTkFont(size=12),
            command=self._on_voice_toggle
        )
        self.chk_cam_voice.pack(side="left", padx=8)

        # Mở thư mục ảnh chụp
        self.btn_open_captures = ctk.CTkButton(
            ctrl_bar,
            text="📁 Ảnh chụp",
            font=ctk.CTkFont(size=12),
            fg_color="#334155",
            hover_color="#1E293B",
            width=85,
            height=32,
            command=self._open_captures_folder
        )
        self.btn_open_captures.pack(side="right", padx=(6, 12))

        self.btn_view_last_snap = ctk.CTkButton(
            ctrl_bar,
            text="🔍 Xem ảnh",
            font=ctk.CTkFont(size=12),
            fg_color="#0369A1",
            hover_color="#0284C7",
            width=85,
            height=32,
            state="disabled",
            command=self._view_last_snapshot
        )
        self.btn_view_last_snap.pack(side="right", padx=4)

        # 3. Khu vực hiển thị trung tâm (Chia 2 cột: Màn hình Camera bên trái, Bảng thống kê bên phải)
        main_content = ctk.CTkFrame(self.view_camera, fg_color="transparent")
        main_content.pack(fill="both", expand=True, padx=14, pady=(6, 12))

        # CỘT TRÁI: Khung hiển thị Video Camera
        self.cam_screen_frame = ctk.CTkFrame(main_content, corner_radius=10, fg_color="#0F172A")
        self.cam_screen_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Nhãn hiển thị hình ảnh từ Camera
        self.lbl_camera_video = ctk.CTkLabel(
            self.cam_screen_frame,
            text=(
                "📷\n\n"
                "CAMERA ĐANG Ở CHẾ ĐỘ CHỜ\n\n"
                "Nhấn nút '▶ BẮT ĐẦU CAMERA' phía trên để kích hoạt nhận diện thời gian thực\n\n"
                "💡 Gợi ý đưa các vật thể sinh hoạt trước camera:\n"
                "📱 Điện thoại   •   💻 Máy tính   •   ☕ Cốc nước   •   🥣 Bát ăn   •   🍽️ Đĩa\n"
                "👟 Giày dép    •   🪥 Bàn chải   •   ✋ Bàn tay    •   🚪 Tủ       •   🧼 Xà phòng..."
            ),
            font=ctk.CTkFont(size=14),
            text_color="#94A3B8",
            justify="center"
        )
        self.lbl_camera_video.pack(fill="both", expand=True, padx=10, pady=10)

        # Dải thanh trạng thái HUD bên dưới khung video
        hud_bar = ctk.CTkFrame(self.cam_screen_frame, height=36, corner_radius=8, fg_color="#1E293B")
        hud_bar.pack(fill="x", padx=10, pady=(0, 10))
        hud_bar.pack_propagate(False)

        self.lbl_cam_live_badge = ctk.CTkLabel(
            hud_bar,
            text="⚫ CAMERA ĐANG TẮT",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#94A3B8"
        )
        self.lbl_cam_live_badge.pack(side="left", padx=12, pady=6)

        self.lbl_cam_fps_info = ctk.CTkLabel(
            hud_bar,
            text="⚡ FPS: 0.0  |  Trễ AI: -- ms",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38BDF8"
        )
        self.lbl_cam_fps_info.pack(side="left", padx=16, pady=6)

        self.lbl_cam_snap_toast = ctk.CTkLabel(
            hud_bar,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#10B981"
        )
        self.lbl_cam_snap_toast.pack(side="left", padx=10, pady=6)

        self.lbl_cam_res_info = ctk.CTkLabel(
            hud_bar,
            text="Độ phân giải: 1280×720",
            font=ctk.CTkFont(size=11),
            text_color="#9CA3AF"
        )
        self.lbl_cam_res_info.pack(side="right", padx=12, pady=6)

        # CỘT PHẢI: Bảng Phân Tích & Danh Sách Vật Thể Thời Gian Thực
        side_panel = ctk.CTkFrame(main_content, width=360, corner_radius=10)
        side_panel.pack(side="right", fill="y")
        side_panel.pack_propagate(False)

        panel_title = ctk.CTkLabel(
            side_panel,
            text="📊 Bảng Vật Thể Thời Gian Thực",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        panel_title.pack(padx=14, pady=(12, 4), anchor="w")

        # Thẻ đếm số lượng vật thể đang thấy
        self.lbl_cam_detected_summary = ctk.CTkLabel(
            side_panel,
            text="🎯 Đang phát hiện: 0 vật thể",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#38BDF8",
            anchor="w"
        )
        self.lbl_cam_detected_summary.pack(padx=14, pady=2, fill="x")

        # Thẻ tóm tắt các nhãn đang xuất hiện
        self.lbl_cam_badges = ctk.CTkLabel(
            side_panel,
            text="Chưa phát hiện vật thể nào",
            font=ctk.CTkFont(size=12),
            text_color="#94A3B8",
            wraplength=330,
            justify="left",
            anchor="w"
        )
        self.lbl_cam_badges.pack(padx=14, pady=(2, 8), fill="x")

        sep1 = ctk.CTkFrame(side_panel, height=2, fg_color="#334155")
        sep1.pack(fill="x", padx=14, pady=4)

        # Danh sách chi tiết vật thể trong khung hình hiện tại
        ctk.CTkLabel(
            side_panel,
            text="📦 Chi tiết vật thể trong khung hình:",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        ).pack(padx=14, pady=4, fill="x")

        self.txt_cam_live_objects = ctk.CTkTextbox(
            side_panel,
            height=260,
            corner_radius=8,
            font=ctk.CTkFont(family="Segoe UI", size=12)
        )
        self.txt_cam_live_objects.pack(fill="x", padx=14, pady=(2, 6))
        self.txt_cam_live_objects.insert("end", "Khởi động Camera để xem danh sách chi tiết các vật thể nhận diện được.")

        sep2 = ctk.CTkFrame(side_panel, height=2, fg_color="#334155")
        sep2.pack(fill="x", padx=14, pady=4)

        # Nhật ký phát hiện gần nhất
        log_header = ctk.CTkFrame(side_panel, fg_color="transparent")
        log_header.pack(fill="x", padx=14, pady=4)

        ctk.CTkLabel(
            log_header,
            text="🕒 Nhật ký phát hiện mới:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left")

        btn_clear_log = ctk.CTkButton(
            log_header,
            text="🗑️ Xóa",
            width=55,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color="#475569",
            hover_color="#334155",
            command=self._clear_cam_history
        )
        btn_clear_log.pack(side="right")

        self.txt_cam_history_log = ctk.CTkTextbox(
            side_panel,
            corner_radius=8,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.txt_cam_history_log.pack(fill="both", expand=True, padx=14, pady=(2, 12))

    # --- Các hàm xử lý Camera ---
    def _on_cam_conf_change(self, val):
        self.lbl_cam_conf_disp.configure(text=f"Độ tin cậy: {int(val*100)}%")

    def _on_voice_toggle(self):
        self.voice_announcer.set_enabled(self.camera_voice_var.get())

    def _clear_cam_history(self):
        self.txt_cam_history_log.delete("1.0", "end")
        self.last_detected_classes.clear()

    def _on_cam_source_selected(self, choice: str):
        if "URL" in choice or "IP" in choice:
            self.entry_cam_url.pack(side="left", padx=4, after=self.btn_cam_scan)
        else:
            self.entry_cam_url.pack_forget()

        # Nếu đang chạy thì khởi động lại với nguồn mới
        if self.camera_running:
            self._stop_camera()
            self._start_camera()

    def _on_cam_res_changed(self, choice: str):
        if self.camera_running:
            self._stop_camera()
            self._start_camera()

    def _scan_available_cameras(self):
        """Quét và phát hiện các Camera đang cắm vào máy tính."""
        self.btn_cam_scan.configure(text="⏳ Đang quét...", state="disabled")
        self.update_idletasks()

        def do_scan():
            available = []
            for idx in range(4):
                try:
                    test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if test_cap.isOpened():
                        ret, _ = test_cap.read()
                        if ret:
                            available.append(idx)
                        test_cap.release()
                except Exception:
                    pass

            if not available:
                available = [0]

            options = [f"Camera {i} (Mặc định)" if i == 0 else f"Camera {i}" for i in available]
            options.append("🌐 URL IP Camera/Điện thoại")

            def update_ui():
                self.available_cam_indices = available
                self.cb_cam_source.configure(values=options)
                if options:
                    self.cb_cam_source.set(options[0])
                self.btn_cam_scan.configure(text="🔄 Quét Camera", state="normal")
                messagebox.showinfo(
                    "Quét Camera Hoàn Tất",
                    f"Đã phát hiện {len(available)} camera khả dụng trên hệ thống:\n" +
                    "\n".join([f"• Camera index {i}" for i in available])
                )

            self.after(0, update_ui)

        threading.Thread(target=do_scan, daemon=True).start()

    def _toggle_camera(self):
        if self.camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        # Đảm bảo mô hình đã được nạp
        if not self.current_model:
            if os.path.exists(DEFAULT_WEIGHTS_PATH):
                self._load_model_weights(DEFAULT_WEIGHTS_PATH, silent=True)
            if not self.current_model:
                messagebox.showwarning("Chưa nạp mô hình", "Vui lòng chọn hoặc nạp tệp mô hình YOLO (.pt) trước khi bắt đầu camera!")
                return

        choice = self.cb_cam_source.get()
        if "URL" in choice or "IP" in choice:
            url = self.entry_cam_url.get().strip()
            if not url:
                messagebox.showwarning("Thiếu URL", "Vui lòng nhập đường dẫn URL IP Camera (ví dụ: http://192.168.1.15:4747/video)!")
                return
            cam_source = url
            is_dshow = False
        else:
            try:
                cam_idx = int(choice.split()[1])
            except Exception:
                cam_idx = 0
            cam_source = cam_idx
            is_dshow = True

        try:
            if is_dshow:
                self.cap = cv2.VideoCapture(cam_source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(cam_source)

            if not self.cap.isOpened():
                messagebox.showerror(
                    "Lỗi Mở Camera",
                    f"Không thể kết nối tới nguồn Camera: {cam_source}\n\n"
                    "Gợi ý khắc phục:\n"
                    "1. Kiểm tra camera có bị ứng dụng khác (Zoom, Teams, Chrome...) sử dụng không.\n"
                    "2. Nhấn '🔄 Quét Camera' để tìm cổng camera chính xác.\n"
                    "3. Đảm bảo quyền truy cập Camera được cấp phép trên Windows."
                )
                return

            # Cài đặt độ phân giải
            res_choice = self.cb_cam_res.get()
            if "1280" in res_choice:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.lbl_cam_res_info.configure(text="Độ phân giải: 1280×720 (HD)")
            elif "1920" in res_choice:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.lbl_cam_res_info.configure(text="Độ phân giải: 1920×1080 (FHD)")
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.lbl_cam_res_info.configure(text="Độ phân giải: 640×480")

            self.camera_running = True
            self.btn_main_cam_toggle.configure(text="⏹ DỪNG CAMERA", fg_color="#EF4444", hover_color="#DC2626")
            self.lbl_cam_live_badge.configure(text="🟢 ĐANG PHÁT TRỰC TIẾP", text_color="#10B981")
            self.lbl_cam_snap_toast.configure(text="")

            threading.Thread(target=self._camera_worker_loop, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Lỗi Khởi Động", f"Không thể khởi động camera:\n{e}")

    def _stop_camera(self):
        self.camera_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.btn_main_cam_toggle.configure(text="▶ BẮT ĐẦU CAMERA", fg_color="#10B981", hover_color="#059669")
        self.lbl_cam_live_badge.configure(text="⚫ CAMERA ĐANG TẮT", text_color="#94A3B8")
        self.lbl_cam_fps_info.configure(text="⚡ FPS: 0.0  |  Trễ AI: -- ms")
        self.lbl_cam_detected_summary.configure(text="🎯 Đang phát hiện: 0 vật thể")
        self.lbl_cam_badges.configure(text="Camera đã dừng.")

        self.lbl_camera_video.configure(
            image=None,
            text=(
                "📷\n\n"
                "CAMERA ĐÃ DỪNG\n\n"
                "Nhấn nút '▶ BẮT ĐẦU CAMERA' phía trên để tiếp tục nhận diện thời gian thực"
            )
        )

    def _camera_worker_loop(self):
        """Vòng lặp đọc khung hình và nhận diện YOLO thời gian thực."""
        prev_time = time.time()
        frame_idx = 0
        fps = 0.0

        while self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            frame_idx += 1
            if frame_idx % 4 == 0 or (now - prev_time) > 0.05:
                fps = 1.0 / max(0.001, now - prev_time)
            prev_time = now

            # Lật gương nếu người dùng tích chọn
            if self.camera_mirror_var.get():
                frame = cv2.flip(frame, 1)

            conf_val = float(self.slider_cam_conf.get())

            t_infer_start = time.time()
            detected_items = []
            try:
                results = self.current_model.predict(source=frame, conf=conf_val, verbose=False)
                res = results[0]
                boxes = res.boxes

                # Vẽ bounding box và tên nhãn Tiếng Việt
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                annotated_pil, detected_items = self._draw_vietnamese_boxes(pil_frame, boxes, res.names, conf_val)
                self.last_camera_annotated_frame = annotated_pil.copy()
            except Exception:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_pil = Image.fromarray(rgb_frame)

            latency_ms = (time.time() - t_infer_start) * 1000

            # Đọc tên vật thể bằng giọng nói nếu được kích hoạt
            if self.camera_voice_var.get() and detected_items:
                for it in detected_items:
                    self.voice_announcer.announce(it["vi_name"])

            # Cập nhật GUI
            self.after(0, lambda img=annotated_pil, items=detected_items, f=fps, lat=latency_ms:
                       self._update_camera_frame(img, items, f, lat))

            time.sleep(0.005)

    def _update_camera_frame(self, pil_img: Image.Image, detected_items: List[Dict], fps: float, latency_ms: float):
        if not self.camera_running:
            return

        # Tính kích thước co giãn tối ưu cho khung hiển thị
        display_w = max(400, self.lbl_camera_video.winfo_width())
        display_h = max(300, self.lbl_camera_video.winfo_height())
        target_w = min(display_w - 16, 960)
        target_h = min(display_h - 16, 640)

        display_img = pil_img.copy()
        display_img.thumbnail((target_w, target_h), Image.Resampling.BILINEAR)

        ctk_img = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=display_img.size)
        self.lbl_camera_video.configure(image=ctk_img, text="")
        self.lbl_camera_video.image = ctk_img

        # Cập nhật thông tin HUD
        self.lbl_cam_fps_info.configure(text=f"⚡ FPS: {fps:.1f}  |  Trễ AI: {latency_ms:.0f} ms")
        num_obj = len(detected_items)
        self.lbl_cam_detected_summary.configure(text=f"🎯 Đang phát hiện: {num_obj} vật thể")

        # Cập nhật tóm tắt các nhãn
        if detected_items:
            counts = {}
            for it in detected_items:
                em = it.get("emoji", "📦")
                vn = it["vi_name"]
                key = f"{em} {vn}"
                counts[key] = counts.get(key, 0) + 1
            summary_str = "  •  ".join([f"{k}: {v}" for k, v in counts.items()])
            self.lbl_cam_badges.configure(text=summary_str)
        else:
            self.lbl_cam_badges.configure(text="Đang quét tìm kiếm vật thể...")

        # Cập nhật bảng chi tiết vật thể đang có trong khung hình
        self.txt_cam_live_objects.delete("1.0", "end")
        if not detected_items:
            conf_pct = int(float(self.slider_cam_conf.get()) * 100)
            self.txt_cam_live_objects.insert(
                "end",
                f"Không có vật thể nào vượt ngưỡng tin cậy ({conf_pct}%).\n\n"
                "💡 Hãy thử đưa điện thoại, cốc nước, bàn tay, giày dép hoặc đồ gia dụng trước camera!"
            )
        else:
            for i, it in enumerate(detected_items, 1):
                em = it.get("emoji", "📦")
                vn = it["vi_name"]
                en = it["eng_name"]
                cf = it["conf"] * 100
                b = it["box"]
                self.txt_cam_live_objects.insert(
                    "end",
                    f"#{i} {em} {vn} ({en})\n"
                    f"   • Độ tin cậy: {cf:.1f}%\n"
                    f"   • Tọa độ: ({b[0]}, {b[1]}) → ({b[2]}, {b[3]})\n"
                    f"   • Kích thước: {b[2]-b[0]}×{b[3]-b[1]} px\n\n"
                )

        # Ghi nhận vào nhật ký nếu xuất hiện vật thể mới
        current_classes = {it["vi_name"] for it in detected_items}
        new_classes = current_classes - self.last_detected_classes
        if new_classes:
            t_str = datetime.datetime.now().strftime("%H:%M:%S")
            for vn_c in new_classes:
                matching = [it for it in detected_items if it["vi_name"] == vn_c]
                em = matching[0].get("emoji", "📦") if matching else "📦"
                cf_val = f"{matching[0]['conf']*100:.0f}%" if matching else ""
                log_line = f"[{t_str}] {em} Phát hiện: {vn_c} ({cf_val})\n"
                self.txt_cam_history_log.insert("1.0", log_line)
        self.last_detected_classes = current_classes

    def _take_camera_snapshot(self):
        """Chụp ảnh từ Camera kèm Bounding Box và lưu vào thư mục captures."""
        if self.last_camera_annotated_frame is None:
            messagebox.showwarning("Thông báo", "Vui lòng bấm '▶ Bắt đầu Camera' trước khi chụp ảnh!")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_fn = f"capture_{timestamp}.jpg"
        save_path = os.path.join(CAPTURES_DIR, save_fn)

        try:
            self.last_camera_annotated_frame.save(save_path, quality=95)
            self.last_saved_snapshot_path = save_path
            self.btn_view_last_snap.configure(state="normal")
            self.lbl_cam_snap_toast.configure(text=f"✅ Đã chụp: {save_fn}")
            self.after(4000, lambda: self.lbl_cam_snap_toast.configure(text=""))
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu ảnh chụp:\n{e}")

    def _view_last_snapshot(self):
        """Mở ảnh vừa chụp bằng trình xem ảnh mặc định của Windows."""
        if self.last_saved_snapshot_path and os.path.exists(self.last_saved_snapshot_path):
            os.startfile(self.last_saved_snapshot_path)
        else:
            messagebox.showinfo("Thông báo", "Chưa có ảnh chụp nào gần đây!")

    def _open_captures_folder(self):
        """Mở thư mục captures chứa các ảnh đã chụp từ Camera."""
        if os.path.exists(CAPTURES_DIR):
            os.startfile(CAPTURES_DIR)

    # =========================================================================
    # CHỨC NĂNG 2: NHẬN DIỆN TỪ ẢNH (IMAGE INFERENCE)
    # =========================================================================
    def _setup_view_inference(self):
        model_bar = ctk.CTkFrame(self.view_inference, corner_radius=10, fg_color="#1E293B")
        model_bar.pack(fill="x", padx=14, pady=(12, 6))

        self.lbl_loaded_model = ctk.CTkLabel(
            model_bar,
            text="🤖 Mô hình: Đang kiểm tra best.pt...",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#F59E0B"
        )
        self.lbl_loaded_model.pack(side="left", padx=14, pady=10)

        btn_load_weights = ctk.CTkButton(
            model_bar,
            text="📂 Chọn tệp mô hình (.pt)",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#3B82F6",
            hover_color="#2563EB",
            height=32,
            command=self._load_custom_model_dialog
        )
        btn_load_weights.pack(side="left", padx=6)

        btn_load_default = ctk.CTkButton(
            model_bar,
            text="⚡ Nạp lại best.pt",
            font=ctk.CTkFont(size=12),
            fg_color="#10B981",
            hover_color="#059669",
            height=32,
            command=lambda: self._load_model_weights(DEFAULT_WEIGHTS_PATH)
        )
        btn_load_default.pack(side="left", padx=6)

        # Thanh trượt độ tin cậy
        thresh_frame = ctk.CTkFrame(model_bar, fg_color="transparent")
        thresh_frame.pack(side="right", padx=14)

        self.lbl_conf = ctk.CTkLabel(thresh_frame, text="Độ tin cậy: 25%", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_conf.pack(side="left", padx=6)

        self.slider_conf = ctk.CTkSlider(
            thresh_frame,
            from_=0.10,
            to=0.90,
            number_of_steps=16,
            width=130,
            command=self._on_conf_change
        )
        self.slider_conf.set(0.25)
        self.slider_conf.pack(side="left", padx=4)

        # Thanh nút thao tác ảnh
        ctrl_bar = ctk.CTkFrame(self.view_inference, corner_radius=10, fg_color="#1E293B")
        ctrl_bar.pack(fill="x", padx=14, pady=6)

        btn_select_img = ctk.CTkButton(
            ctrl_bar,
            text="📂 Chọn ảnh từ máy tính...",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=36,
            command=self._predict_image_dialog
        )
        btn_select_img.pack(side="left", padx=(10, 6), pady=8)

        btn_sample_img = ctk.CTkButton(
            ctrl_bar,
            text="🎲 Thử nhanh ảnh ngẫu nhiên từ Dataset",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7",
            hover_color="#0369A1",
            height=36,
            command=self._predict_random_dataset_image
        )
        btn_sample_img.pack(side="left", padx=6, pady=8)

        btn_save_img = ctk.CTkButton(
            ctrl_bar,
            text="💾 Lưu ảnh kết quả",
            font=ctk.CTkFont(size=12),
            fg_color="#475569",
            hover_color="#334155",
            height=36,
            command=self._save_prediction_image
        )
        btn_save_img.pack(side="left", padx=6, pady=8)

        btn_goto_cam = ctk.CTkButton(
            ctrl_bar,
            text="📷 Chuyển sang Camera",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669",
            height=36,
            command=lambda: self.switch_view("camera")
        )
        btn_goto_cam.pack(side="right", padx=10, pady=8)

        self.lbl_infer_img_status = ctk.CTkLabel(
            ctrl_bar,
            text="Sẵn sàng nhận diện ảnh",
            font=ctk.CTkFont(size=12),
            text_color="#94A3B8"
        )
        self.lbl_infer_img_status.pack(side="left", padx=15)

        # Khung nội dung ảnh & kết quả
        content = ctk.CTkFrame(self.view_inference, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=14, pady=(6, 14))

        # Khung ảnh
        self.infer_img_display = ctk.CTkFrame(content, corner_radius=10, fg_color="#0F172A")
        self.infer_img_display.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.lbl_infer_view = ctk.CTkLabel(
            self.infer_img_display,
            text="Chọn ảnh từ máy hoặc bấm 'Thử nhanh ảnh ngẫu nhiên' để bắt đầu nhận diện",
            font=ctk.CTkFont(size=14),
            text_color="#64748B"
        )
        self.lbl_infer_view.pack(fill="both", expand=True, padx=10, pady=10)

        # Bảng tóm tắt kết quả
        summary_panel = ctk.CTkFrame(content, width=340, corner_radius=10)
        summary_panel.pack(side="right", fill="y")
        summary_panel.pack_propagate(False)

        ctk.CTkLabel(
            summary_panel,
            text="📊 Kết Quả Phát Hiện",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(padx=14, pady=12, anchor="w")

        self.lbl_infer_stats = ctk.CTkLabel(
            summary_panel,
            text="Số lượng: 0 vật thể\nThời gian xử lý: -- ms",
            justify="left",
            font=ctk.CTkFont(size=13),
            text_color="#38BDF8"
        )
        self.lbl_infer_stats.pack(padx=14, pady=4, anchor="w")

        self.txt_infer_results = ctk.CTkTextbox(
            summary_panel,
            corner_radius=8,
            font=ctk.CTkFont(family="Segoe UI", size=13)
        )
        self.txt_infer_results.pack(fill="both", expand=True, padx=14, pady=12)

    def _on_conf_change(self, val):
        self.lbl_conf.configure(text=f"Độ tin cậy: {int(val*100)}%")

    def _load_custom_model_dialog(self):
        file_p = filedialog.askopenfilename(
            title="Chọn tệp trọng số mô hình (.pt)",
            filetypes=[("Mô hình PyTorch YOLO", "*.pt"), ("Tất cả tệp", "*.*")]
        )
        if file_p:
            self._load_model_weights(file_p)

    def _load_model_weights(self, path: str, silent: bool = False):
        try:
            self.lbl_loaded_model.configure(text="⏳ Đang nạp mô hình...", text_color="#38BDF8")
            self.lbl_cam_model_badge.configure(text="⏳ Đang nạp mô hình...", text_color="#38BDF8")
            self.update_idletasks()

            from ultralytics import YOLO
            self.current_model = YOLO(path)
            self.model_name = os.path.basename(path)
            num_classes = len(self.current_model.names)
            msg = f"✅ Mô hình: {self.model_name} ({num_classes} lớp)"
            self.lbl_loaded_model.configure(text=msg, text_color="#10B981")
            self.lbl_cam_model_badge.configure(text=msg, text_color="#10B981")
            self.sidebar_model_lbl.configure(text=f"🤖 {self.model_name} ({num_classes} lớp)", text_color="#10B981")
            if not silent:
                messagebox.showinfo("Thành công", f"Đã nạp thành công mô hình:\n{path}\nSố lớp nhận diện: {num_classes}")
        except Exception as e:
            self.lbl_loaded_model.configure(text="❌ Lỗi nạp mô hình", text_color="#EF4444")
            self.lbl_cam_model_badge.configure(text="❌ Chưa nạp mô hình", text_color="#EF4444")
            self.sidebar_model_lbl.configure(text="❌ Chưa nạp mô hình", text_color="#EF4444")
            if not silent:
                messagebox.showerror("Lỗi", f"Không thể tải mô hình:\n{e}")

    def _draw_vietnamese_boxes(self, img_pil: Image.Image, boxes, names, conf_thresh: float) -> Tuple[Image.Image, List[Dict]]:
        """Vẽ khung Bounding Box và tên nhãn Tiếng Việt sắc nét lên ảnh bằng PIL."""
        draw = ImageDraw.Draw(img_pil)
        font = None
        if DEFAULT_FONT_PATH:
            try:
                font = ImageFont.truetype(DEFAULT_FONT_PATH, 16)
            except Exception:
                pass

        detected_list = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            cls_id = int(box.cls[0])
            eng_name = names.get(cls_id, f"Lớp {cls_id}")
            vi_name = get_vietnamese_name(eng_name)
            color = get_class_color(eng_name)
            emoji = CLASS_EMOJIS.get(eng_name, CLASS_EMOJIS.get(eng_name.lower(), "📦"))

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            # Vẽ khung chữ nhật
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

            # Thẻ nhãn Tiếng Việt kèm phần trăm độ tin cậy
            label_text = f" {vi_name}: {int(conf * 100)}% "
            if font:
                tbbox = draw.textbbox((0, 0), label_text, font=font)
                tw = tbbox[2] - tbbox[0]
                th = tbbox[3] - tbbox[1]
                tag_y = max(0, y1 - th - 8) if y1 >= th + 8 else y1 + 4
                tag_box = (x1, tag_y, x1 + tw + 10, tag_y + th + 6)
                draw.rectangle(tag_box, fill=color)
                draw.text((x1 + 4, tag_y + 2), label_text, fill="white", font=font)
            else:
                tag_y = max(0, y1 - 22) if y1 >= 22 else y1 + 4
                tbbox = draw.textbbox((x1, tag_y), label_text)
                draw.rectangle(tbbox, fill=color)
                draw.text((x1, tag_y), label_text, fill="white")

            detected_list.append({
                "vi_name": vi_name,
                "eng_name": eng_name,
                "emoji": emoji,
                "conf": conf,
                "box": [x1, y1, x2, y2]
            })

        return img_pil, detected_list

    def _predict_image_path(self, file_p: str):
        if not self.current_model:
            messagebox.showwarning("Chưa nạp mô hình", "Vui lòng bấm 'Chọn tệp mô hình' trước khi kiểm thử!")
            return

        try:
            conf_val = float(self.slider_conf.get())
            t0 = time.time()
            results = self.current_model.predict(source=file_p, conf=conf_val, save=False)
            t1 = time.time()
            elapsed_ms = (t1 - t0) * 1000

            res = results[0]
            boxes = res.boxes

            # Vẽ tiếng Việt lên ảnh gốc
            pil_img = Image.open(file_p).convert("RGB")
            annotated_pil, detected_items = self._draw_vietnamese_boxes(pil_img, boxes, res.names, conf_val)
            self.last_annotated_img = annotated_pil.copy()

            # Hiển thị lên giao diện
            max_w, max_h = 760, 540
            display_img = annotated_pil.copy()
            display_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=display_img.size)

            self.lbl_infer_view.configure(image=ctk_img, text="")
            self.lbl_infer_view.image = ctk_img

            self.lbl_infer_stats.configure(
                text=f"Số lượng: {len(detected_items)} vật thể phát hiện\nThời gian xử lý: {elapsed_ms:.1f} ms"
            )

            # Cập nhật khung kết quả
            self.txt_infer_results.delete("1.0", "end")
            if not detected_items:
                self.txt_infer_results.insert(
                    "end",
                    f"Không tìm thấy vật thể nào vượt ngưỡng tin cậy {int(conf_val*100)}%.\n(Hãy thử hạ thanh trượt 'Độ tin cậy' xuống 15% - 20%)"
                )
            else:
                for i, item in enumerate(detected_items, 1):
                    vn = item["vi_name"]
                    en = item["eng_name"]
                    em = item.get("emoji", "📦")
                    cf = item["conf"]
                    b = item["box"]
                    self.txt_infer_results.insert(
                        "end",
                        f"#{i} {em} {vn} ({en})\n"
                        f"  Độ tin cậy: {cf*100:.1f}%\n"
                        f"  Vị trí: ({b[0]}, {b[1]}) -> ({b[2]}, {b[3]})\n\n"
                    )

            self.lbl_infer_img_status.configure(text=f"Đã nhận diện: {os.path.basename(file_p)[:28]}...")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý ảnh:\n{e}")

    def _predict_image_dialog(self):
        file_p = filedialog.askopenfilename(
            title="Chọn ảnh kiểm thử",
            filetypes=[("Hình ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")]
        )
        if file_p:
            self._predict_image_path(file_p)

    def _predict_random_dataset_image(self):
        sample_pool = []
        if os.path.exists(os.path.join(BASE_DIR, "dollar_street_yolo", "images", "val")):
            sample_pool = glob.glob(os.path.join(BASE_DIR, "dollar_street_yolo", "images", "val", "*.jpg"))
        if not sample_pool and os.path.exists(self.dataset_dir):
            for cls in TOP_20_CLASSES[:10]:
                for f in CLASS_TO_FOLDERS.get(cls, []):
                    img_d = os.path.join(self.dataset_dir, f, "img")
                    if os.path.exists(img_d):
                        sample_pool.extend(glob.glob(os.path.join(img_d, "*.jpg"))[:5])

        if not sample_pool:
            messagebox.showwarning("Thông báo", "Không tìm thấy ảnh mẫu trong thư mục dữ liệu!")
            return

        picked = random.choice(sample_pool)
        self._predict_image_path(picked)

    def _save_prediction_image(self):
        if self.last_annotated_img is None:
            messagebox.showwarning("Thông báo", "Chưa có ảnh nhận diện nào để lưu!")
            return

        save_p = filedialog.asksaveasfilename(
            title="Lưu ảnh nhận diện",
            defaultextension=".jpg",
            filetypes=[("Ảnh JPEG", "*.jpg"), ("Ảnh PNG", "*.png")]
        )
        if save_p:
            self.last_annotated_img.save(save_p)
            messagebox.showinfo("Đã lưu", f"Ảnh đã được lưu thành công tại:\n{save_p}")

    # =========================================================================
    # CHỨC NĂNG 3: KHÁM PHÁ DATASET (DATASET EXPLORER)
    # =========================================================================
    def _setup_view_explorer(self):
        top_bar = ctk.CTkFrame(self.view_explorer, fg_color="transparent")
        top_bar.pack(fill="x", padx=16, pady=12)

        cat_lbl = ctk.CTkLabel(top_bar, text="Danh mục đối tượng:", font=ctk.CTkFont(size=14, weight="bold"))
        cat_lbl.pack(side="left", padx=(0, 10))

        self.cat_combobox = ctk.CTkComboBox(
            top_bar,
            values=["Đang quét danh mục..."],
            width=320,
            command=self._on_category_selected
        )
        self.cat_combobox.pack(side="left", padx=5)

        self.box_checkbox = ctk.CTkCheckBox(
            top_bar,
            text="Hiển thị Bounding Box",
            variable=self.show_boxes_var,
            command=self._render_current_image
        )
        self.box_checkbox.pack(side="left", padx=20)

        self.btn_prev = ctk.CTkButton(top_bar, text="◀ Trước", width=75, command=self._prev_image)
        self.btn_prev.pack(side="left", padx=5)

        self.lbl_img_counter = ctk.CTkLabel(top_bar, text="0 / 0", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_img_counter.pack(side="left", padx=8)

        self.btn_next = ctk.CTkButton(top_bar, text="Tiếp ▶", width=75, command=self._next_image)
        self.btn_next.pack(side="left", padx=5)

        self.btn_random = ctk.CTkButton(top_bar, text="🎲 Ngẫu nhiên", width=95, command=self._random_image)
        self.btn_random.pack(side="left", padx=5)

        content_frame = ctk.CTkFrame(self.view_explorer, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.image_display_frame = ctk.CTkFrame(content_frame, corner_radius=10, fg_color="#0F172A")
        self.image_display_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.lbl_image_view = ctk.CTkLabel(self.image_display_frame, text="Chọn danh mục để xem ảnh", text_color="#64748B")
        self.lbl_image_view.pack(fill="both", expand=True, padx=10, pady=10)

        info_panel = ctk.CTkFrame(content_frame, width=360, corner_radius=10)
        info_panel.pack(side="right", fill="y", padx=0)
        info_panel.pack_propagate(False)

        info_title = ctk.CTkLabel(info_panel, text="📋 Thông tin Chú thích (Annotation)", font=ctk.CTkFont(size=15, weight="bold"))
        info_title.pack(padx=14, pady=(14, 8), anchor="w")

        self.lbl_img_filename = ctk.CTkLabel(info_panel, text="File: --", font=ctk.CTkFont(size=12), text_color="#94A3B8", anchor="w")
        self.lbl_img_filename.pack(padx=14, pady=2, fill="x")

        self.lbl_img_res = ctk.CTkLabel(info_panel, text="Kích thước: --", font=ctk.CTkFont(size=12), text_color="#94A3B8", anchor="w")
        self.lbl_img_res.pack(padx=14, pady=2, fill="x")

        self.lbl_obj_count = ctk.CTkLabel(info_panel, text="Số vật thể: 0", font=ctk.CTkFont(size=13, weight="bold"), text_color="#38BDF8", anchor="w")
        self.lbl_obj_count.pack(padx=14, pady=(6, 10), fill="x")

        sep = ctk.CTkFrame(info_panel, height=2, fg_color="#334155")
        sep.pack(fill="x", padx=14, pady=4)

        box_list_title = ctk.CTkLabel(info_panel, text="Danh sách nhãn & Toạ độ:", font=ctk.CTkFont(size=13, weight="bold"))
        box_list_title.pack(padx=14, pady=6, anchor="w")

        self.txt_obj_details = ctk.CTkTextbox(info_panel, corner_radius=8, font=ctk.CTkFont(family="Segoe UI", size=13))
        self.txt_obj_details.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    def _init_dataset_categories(self):
        if not os.path.exists(self.dataset_dir):
            self.ds_status_lbl.configure(text="❌ Không tìm thấy thư mục Dollar street trial!")
            return

        subdirs = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
        obj_dirs = [d for d in subdirs if d.startswith("Objects")]
        other_dirs = [d for d in subdirs if not d.startswith("Objects")]
        self.explorer_categories = sorted(obj_dirs) + sorted(other_dirs)

        self.ds_status_lbl.configure(text=f"✅ Dollar Street: {len(subdirs)} danh mục (~27.5k ảnh)")

        display_options = []
        self.cat_map = {}
        for d in self.explorer_categories:
            clean_name = d.replace("Objects__", "").replace("Objects_", "").replace("Abstract__", "").replace("Places__", "").replace("-", " ").title()
            vn_title = get_vietnamese_name(clean_name)
            disp_str = f"{vn_title} ({d})"
            display_options.append(disp_str)
            self.cat_map[disp_str] = d

        if display_options:
            self.cat_combobox.configure(values=display_options)
            default_disp = display_options[0]
            for op in display_options:
                if "Điện thoại" in op or "phones" in op:
                    default_disp = op
                    break
            self.cat_combobox.set(default_disp)
            self._load_category_images(self.cat_map[default_disp])

    def _on_category_selected(self, choice: str):
        folder_name = self.cat_map.get(choice, choice)
        self._load_category_images(folder_name)

    def _load_category_images(self, cat_name: str):
        self.current_category = cat_name
        img_dir = os.path.join(self.dataset_dir, cat_name, "img")

        if not os.path.exists(img_dir):
            self.category_images = []
            self.lbl_img_counter.configure(text="0 / 0")
            return

        files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.category_images = files
        self.current_img_idx = 0
        self._render_current_image()

    def _prev_image(self):
        if self.category_images and self.current_img_idx > 0:
            self.current_img_idx -= 1
            self._render_current_image()

    def _next_image(self):
        if self.category_images and self.current_img_idx < len(self.category_images) - 1:
            self.current_img_idx += 1
            self._render_current_image()

    def _random_image(self):
        if self.category_images:
            self.current_img_idx = random.randint(0, len(self.category_images) - 1)
            self._render_current_image()

    def _render_current_image(self):
        if not self.category_images:
            self.lbl_image_view.configure(text="Không có ảnh trong danh mục này", image=None)
            self.lbl_img_counter.configure(text="0 / 0")
            return

        total = len(self.category_images)
        self.lbl_img_counter.configure(text=f"{self.current_img_idx + 1} / {total}")

        img_fn = self.category_images[self.current_img_idx]
        img_path = os.path.join(self.dataset_dir, self.current_category, "img", img_fn)
        ann_path = os.path.join(self.dataset_dir, self.current_category, "ann", f"{img_fn}.json")

        self.lbl_img_filename.configure(text=f"Tệp: {img_fn[:24]}...")

        objects = []
        w, h = 0, 0
        if os.path.exists(ann_path):
            try:
                w, h, objects = parse_supervisely_annotation(ann_path)
            except Exception:
                pass

        self.lbl_img_res.configure(text=f"Kích thước: {w} × {h} px" if w > 0 else "Kích thước: Tự động")
        self.lbl_obj_count.configure(text=f"Số vật thể: {len(objects)}")

        self.txt_obj_details.delete("1.0", "end")
        if objects:
            for idx, obj in enumerate(objects, 1):
                en_cls = obj["classTitle"]
                vn_cls = get_vietnamese_name(en_cls)
                b = [int(v) for v in obj["box"]]
                self.txt_obj_details.insert(
                    "end",
                    f"#{idx} {vn_cls} ({en_cls})\n"
                    f"  Hộp: ({b[0]}, {b[1]}) -> ({b[2]}, {b[3]})\n"
                    f"  Kích cỡ: {b[2]-b[0]}×{b[3]-b[1]}px\n\n"
                )
        else:
            self.txt_obj_details.insert("end", "Không có bounding box trong ảnh này.\n(Có thể là nhãn phân loại phòng/cảnh)")

        try:
            pil_img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(pil_img)
            font = None
            if DEFAULT_FONT_PATH:
                try:
                    font = ImageFont.truetype(DEFAULT_FONT_PATH, 16)
                except Exception:
                    pass

            if self.show_boxes_var.get() and objects:
                for obj in objects:
                    en_cls = obj["classTitle"]
                    vn_cls = get_vietnamese_name(en_cls)
                    color = get_class_color(en_cls)
                    xmin, ymin, xmax, ymax = obj["box"]

                    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
                    tag_text = f" {vn_cls} "
                    if font:
                        text_bbox = draw.textbbox((xmin, max(0, ymin - 22)), tag_text, font=font)
                        draw.rectangle(text_bbox, fill=color)
                        draw.text((xmin, max(0, ymin - 22)), tag_text, fill="white", font=font)
                    else:
                        text_bbox = draw.textbbox((xmin, max(0, ymin - 20)), tag_text)
                        draw.rectangle(text_bbox, fill=color)
                        draw.text((xmin, max(0, ymin - 20)), tag_text, fill="white")

            max_w, max_h = 760, 560
            pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

            self.lbl_image_view.configure(image=ctk_img, text="")
            self.lbl_image_view.image = ctk_img
        except Exception as e:
            self.lbl_image_view.configure(text=f"Lỗi tải ảnh: {e}", image=None)

    # =========================================================================
    # CHỨC NĂNG 4: XUẤT DỮ LIỆU KAGGLE
    # =========================================================================
    def _setup_view_exporter(self):
        title = ctk.CTkLabel(
            self.view_exporter,
            text="⚡ Xuất Dữ liệu Sang Định dạng YOLO & Đóng gói ZIP Cho Kaggle",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(padx=20, pady=(18, 12), anchor="w")

        desc = ctk.CTkLabel(
            self.view_exporter,
            text="Công cụ tự động chuyển đổi annotation Supervisely sang chuẩn YOLO (images/labels/data.yaml) và nén thành 1 file ZIP sẵn sàng upload lên Kaggle.",
            font=ctk.CTkFont(size=13),
            text_color="#94A3B8",
            wraplength=980,
            justify="left"
        )
        desc.pack(padx=20, pady=(0, 16), anchor="w")

        config_box = ctk.CTkFrame(self.view_exporter, corner_radius=10)
        config_box.pack(fill="x", padx=20, pady=6)

        preset_frame = ctk.CTkFrame(config_box, fg_color="transparent")
        preset_frame.pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(preset_frame, text="Bộ nhãn đối tượng:", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=(0, 15))

        self.preset_var = ctk.StringVar(value="top20")
        r1 = ctk.CTkRadioButton(preset_frame, text="Top 20 Đối tượng phổ biến (Khuyên dùng)", variable=self.preset_var, value="top20", command=self._update_classes_preview)
        r1.pack(side="left", padx=10)

        r2 = ctk.CTkRadioButton(preset_frame, text="Top 10 Đối tượng (Nhanh)", variable=self.preset_var, value="top10", command=self._update_classes_preview)
        r2.pack(side="left", padx=10)

        r3 = ctk.CTkRadioButton(preset_frame, text="Toàn bộ 94 nhãn", variable=self.preset_var, value="all", command=self._update_classes_preview)
        r3.pack(side="left", padx=10)

        vn_top20_str = ", ".join([f"{get_vietnamese_name(c)} ({c})" for c in TOP_20_CLASSES])
        self.lbl_classes_preview = ctk.CTkLabel(
            config_box,
            text="Danh sách: " + vn_top20_str,
            font=ctk.CTkFont(size=12),
            text_color="#38BDF8",
            wraplength=950,
            justify="left"
        )
        self.lbl_classes_preview.pack(fill="x", padx=16, pady=(0, 10))

        slider_frame = ctk.CTkFrame(config_box, fg_color="transparent")
        slider_frame.pack(fill="x", padx=16, pady=8)

        s1_box = ctk.CTkFrame(slider_frame, fg_color="transparent")
        s1_box.pack(side="left", fill="x", expand=True, padx=(0, 15))

        self.lbl_max_imgs = ctk.CTkLabel(s1_box, text="Giới hạn ảnh / nhãn: 100 ảnh (~1.500 ảnh tổng)", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_max_imgs.pack(anchor="w")

        self.slider_max_imgs = ctk.CTkSlider(s1_box, from_=30, to=300, number_of_steps=27, command=self._on_max_imgs_slider)
        self.slider_max_imgs.set(100)
        self.slider_max_imgs.pack(fill="x", pady=6)

        s2_box = ctk.CTkFrame(slider_frame, fg_color="transparent")
        s2_box.pack(side="right", fill="x", expand=True, padx=(15, 0))

        self.lbl_split_ratio = ctk.CTkLabel(s2_box, text="Tỉ lệ Huấn luyện (Train) / Kiểm thử (Val): 80% / 20%", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_split_ratio.pack(anchor="w")

        self.slider_split = ctk.CTkSlider(s2_box, from_=0.7, to=0.9, number_of_steps=4, command=self._on_split_slider)
        self.slider_split.set(0.8)
        self.slider_split.pack(fill="x", pady=6)

        act_frame = ctk.CTkFrame(self.view_exporter, fg_color="transparent")
        act_frame.pack(fill="x", padx=20, pady=12)

        self.btn_export = ctk.CTkButton(
            act_frame,
            text="🚀 Bắt đầu Chuyển đổi & Đóng gói ZIP",
            height=46,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669",
            command=self._start_export_thread
        )
        self.btn_export.pack(side="left", padx=(0, 15))

        self.btn_open_folder = ctk.CTkButton(
            act_frame,
            text="📂 Mở thư mục kết quả",
            height=46,
            font=ctk.CTkFont(size=14),
            fg_color="#475569",
            hover_color="#334155",
            command=self._open_output_folder
        )
        self.btn_open_folder.pack(side="left")

        self.progress_bar = ctk.CTkProgressBar(self.view_exporter)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=20, pady=(8, 4))

        self.lbl_export_status = ctk.CTkLabel(self.view_exporter, text="Sẵn sàng thực hiện.", font=ctk.CTkFont(size=12), text_color="#94A3B8")
        self.lbl_export_status.pack(padx=20, anchor="w")

        self.txt_export_log = ctk.CTkTextbox(self.view_exporter, corner_radius=8, font=ctk.CTkFont(family="Consolas", size=12))
        self.txt_export_log.pack(fill="both", expand=True, padx=20, pady=(8, 16))
        self.txt_export_log.insert("end", "=== NHẬT KÝ XỬ LÝ DỮ LIỆU ===\nChờ lệnh bắt đầu từ người dùng...\n")

    def _update_classes_preview(self):
        val = self.preset_var.get()
        if val == "top20":
            txt = "Danh sách: " + ", ".join([f"{get_vietnamese_name(c)} ({c})" for c in TOP_20_CLASSES])
        elif val == "top10":
            txt = "Danh sách: " + ", ".join([f"{get_vietnamese_name(c)} ({c})" for c in TOP_10_CLASSES])
        else:
            txt = "Danh sách: Toàn bộ 94 danh mục đối tượng trong meta.json"
        self.lbl_classes_preview.configure(text=txt)

    def _on_max_imgs_slider(self, val):
        val_int = int(val)
        est = val_int * (20 if self.preset_var.get() == "top20" else (10 if self.preset_var.get() == "top10" else 94))
        self.lbl_max_imgs.configure(text=f"Giới hạn ảnh / nhãn: {val_int} ảnh (~{est} ảnh tổng)")

    def _on_split_slider(self, val):
        train_pct = int(val * 100)
        val_pct = 100 - train_pct
        self.lbl_split_ratio.configure(text=f"Tỉ lệ Train / Val: {train_pct}% / {val_pct}%")

    def _start_export_thread(self):
        self.btn_export.configure(state="disabled")
        self.progress_bar.set(0)
        threading.Thread(target=self._run_export_process, daemon=True).start()

    def _run_export_process(self):
        try:
            choice = self.preset_var.get()
            if choice == "top20":
                target_classes = TOP_20_CLASSES
            elif choice == "top10":
                target_classes = TOP_10_CLASSES
            else:
                target_classes = get_all_dataset_classes(self.dataset_dir)

            max_per_cls = int(self.slider_max_imgs.get())
            train_ratio = float(self.slider_split.get())
            out_dir = os.path.join(BASE_DIR, "dollar_street_yolo")

            def on_progress(current, total, msg):
                pct = current / total
                self.after(0, lambda: self.progress_bar.set(pct))
                self.after(0, lambda: self.lbl_export_status.configure(text=msg))
                self.after(0, lambda: self._append_export_log(f"[{int(pct*100)}%] {msg}\n"))

            on_progress(5, 100, "Bắt đầu xuất dữ liệu YOLO...")
            zip_res = export_to_yolo_dataset(
                dataset_dir=self.dataset_dir,
                output_dir=out_dir,
                target_classes=target_classes,
                train_ratio=train_ratio,
                max_images_per_class=max_per_cls,
                progress_callback=on_progress,
                create_zip=True
            )

            zip_size_mb = os.path.getsize(zip_res) / (1024 * 1024)
            self.after(0, lambda: self._append_export_log(f"\n🎉 THÀNH CÔNG! File ZIP: {zip_res} ({zip_size_mb:.2f} MB)\n"))
            self.after(0, lambda: messagebox.showinfo("Hoàn tất", f"Đã xuất dữ liệu thành công!\n\nĐường dẫn: {zip_res}\nDung lượng: {zip_size_mb:.2f} MB"))
        except Exception as e:
            self.after(0, lambda: self._append_export_log(f"\n❌ LỖI: {e}\n"))
            self.after(0, lambda: messagebox.showerror("Lỗi", f"Quá trình xuất dữ liệu gặp lỗi:\n{e}"))
        finally:
            self.after(0, lambda: self.btn_export.configure(state="normal"))

    def _append_export_log(self, text: str):
        self.txt_export_log.insert("end", text)
        self.txt_export_log.see("end")

    def _open_output_folder(self):
        if os.path.exists(BASE_DIR):
            os.startfile(BASE_DIR)

    # =========================================================================
    # CHỨC NĂNG 5: HƯỚNG DẪN KAGGLE
    # =========================================================================
    def _setup_view_kaggle(self):
        title = ctk.CTkLabel(
            self.view_kaggle,
            text="☁️ Quy Trình Huấn Luyện YOLOv8 Trên Kaggle GPU (T4 / P100)",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(padx=20, pady=(18, 8), anchor="w")

        cards_frame = ctk.CTkFrame(self.view_kaggle, fg_color="transparent")
        cards_frame.pack(fill="x", padx=20, pady=8)

        steps = [
            ("1. Chuẩn bị File Zip", "Xuất file dollar_street_yolo.zip từ Tab 4", "📂"),
            ("2. Tạo Kaggle Dataset", "Upload zip lên Kaggle Datasets", "☁️"),
            ("3. Mở & Chạy Notebook", "Nạp file .ipynb, bật GPU T4 x2 & Run All", "⚡"),
            ("4. Tải file best.pt", "Lưu best.pt vào thư mục phần mềm", "🎯")
        ]

        for num, (st_title, st_desc, icon) in enumerate(steps, 1):
            card = ctk.CTkFrame(cards_frame, corner_radius=10, fg_color="#1E293B")
            card.pack(side="left", fill="both", expand=True, padx=4)

            icon_lbl = ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=24))
            icon_lbl.pack(pady=(12, 4))

            t_lbl = ctk.CTkLabel(card, text=st_title, font=ctk.CTkFont(size=13, weight="bold"))
            t_lbl.pack(padx=8, pady=2)

            d_lbl = ctk.CTkLabel(card, text=st_desc, font=ctk.CTkFont(size=11), text_color="#94A3B8", wraplength=190)
            d_lbl.pack(padx=8, pady=(2, 12))

        act_row = ctk.CTkFrame(self.view_kaggle, fg_color="transparent")
        act_row.pack(fill="x", padx=20, pady=10)

        btn_open_kaggle_ds = ctk.CTkButton(
            act_row,
            text="🌐 Mở Kaggle Tạo Dataset Mới",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7",
            hover_color="#0369A1",
            command=lambda: webbrowser.open("https://www.kaggle.com/datasets/new")
        )
        btn_open_kaggle_ds.pack(side="left", padx=(0, 10))

        btn_open_kaggle_code = ctk.CTkButton(
            act_row,
            text="🌐 Mở Kaggle Tạo Notebook Mới",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284C7",
            hover_color="#0369A1",
            command=lambda: webbrowser.open("https://www.kaggle.com/code/new")
        )
        btn_open_kaggle_code.pack(side="left", padx=10)

        btn_open_local_nb = ctk.CTkButton(
            act_row,
            text="📄 Mở File Notebook (.ipynb) Tại Máy",
            font=ctk.CTkFont(size=13),
            fg_color="#475569",
            hover_color="#334155",
            command=self._open_notebook_file
        )
        btn_open_local_nb.pack(side="left", padx=10)

        guide_box_title = ctk.CTkLabel(self.view_kaggle, text="📖 Hướng Dẫn Chi Tiết Từng Bước (Markdown):", font=ctk.CTkFont(size=13, weight="bold"))
        guide_box_title.pack(padx=20, pady=(6, 2), anchor="w")

        self.txt_guide = ctk.CTkTextbox(self.view_kaggle, corner_radius=8, font=ctk.CTkFont(family="Segoe UI", size=13))
        self.txt_guide.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        if os.path.exists(GUIDE_MD_PATH):
            with open(GUIDE_MD_PATH, "r", encoding="utf-8") as gf:
                self.txt_guide.insert("end", gf.read())

    def _open_notebook_file(self):
        if os.path.exists(KAGGLE_NOTEBOOK_PATH):
            os.startfile(KAGGLE_NOTEBOOK_PATH)
        else:
            messagebox.showwarning("Thông báo", "Không tìm thấy file notebook trên máy!")

    def on_closing(self):
        self._stop_camera()
        if hasattr(self, "voice_announcer"):
            self.voice_announcer.stop()
        self.destroy()


if __name__ == "__main__":
    app = DollarStreetApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

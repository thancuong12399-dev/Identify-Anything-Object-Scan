"""
Dollar Street Dataset Utilities
- Parsing Supervisely format
- Fast targeted scanning & conversion to YOLOv8 / YOLOv11 format
- Generating data.yaml and Kaggle zip package
"""

import os
import json
import shutil
import random
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

# Top 20 household and everyday object classes with high representation
TOP_20_CLASSES = [
    "Phone",
    "Computer",
    "Car",
    "Bike",
    "Bed",
    "Toilet",
    "Shoe",
    "Stove",
    "Toothbrush",
    "Cooking pot",
    "Soap",
    "Wardrobe",
    "Dish rack",
    "Kitchen sink",
    "Plate",
    "Shower",
    "Bowl",
    "Cup",
    "TV",
    "Hand"
]

TOP_10_CLASSES = [
    "Phone",
    "Computer",
    "Car",
    "Bike",
    "Bed",
    "Toilet",
    "Shoe",
    "Toothbrush",
    "Cup",
    "TV"
]

# Fast folder index mapping for common object classes
CLASS_TO_FOLDERS = {
    "Phone": ["Objects__phones"],
    "Computer": ["Objects__computers"],
    "Car": ["Objects__cars"],
    "Bike": ["Objects__bikes"],
    "Bed": ["Objects__beds", "Objects__guest-beds", "Objects__kids-bed"],
    "Toilet": ["Objects__toilets"],
    "Shoe": ["Objects_everyday-shoes", "Objects__nicest-shoes"],
    "Stove": ["Objects__stoves"],
    "Toothbrush": ["Objects_toothbrushes"],
    "Cooking pot": ["Objects__cooking-pots"],
    "Soap": ["Objects__soaps", "Objects__dish-washing-soaps"],
    "Wardrobe": ["Objects__wardrobes"],
    "Dish rack": ["Objects__dish-racks"],
    "Kitchen sink": ["Objects__kitchen-sinks"],
    "Plate": ["Objects__plates"],
    "Shower": ["Objects__showers"],
    "Bowl": ["Objects__bowls"],
    "Cup": ["Objects__cups"],
    "TV": ["Objects__tvs"],
    "Hand": ["Objects_hands"],
}


def get_all_dataset_classes(dataset_dir: str) -> List[str]:
    """Reads meta.json to get the list of all configured class titles."""
    meta_path = os.path.join(dataset_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            return [c["title"] for c in meta.get("classes", [])]
    return []


def parse_supervisely_annotation(ann_path: str) -> Tuple[int, int, List[Dict]]:
    """
    Parses a single Supervisely JSON annotation file.
    Returns: (width, height, list_of_objects)
    Each object contains: {'classTitle': ..., 'box': [xmin, ymin, xmax, ymax]}
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    w = data.get("size", {}).get("width", 0)
    h = data.get("size", {}).get("height", 0)

    objects = []
    for obj in data.get("objects", []):
        cls_title = obj.get("classTitle")
        points = obj.get("points", {}).get("exterior", [])
        if cls_title and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)

            # Clamp within image bounds
            if w > 0 and h > 0:
                xmin = max(0.0, min(float(w), float(xmin)))
                xmax = max(0.0, min(float(w), float(xmax)))
                ymin = max(0.0, min(float(h), float(ymin)))
                ymax = max(0.0, min(float(h), float(ymax)))

            if xmax > xmin and ymax > ymin:
                objects.append({
                    "classTitle": cls_title,
                    "box": [xmin, ymin, xmax, ymax],
                    "raw_obj": obj
                })

    return w, h, objects


def convert_box_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Converts pixel bounding box to YOLO normalized format: x_center, y_center, width, height."""
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    bbox_w = (xmax - xmin) / img_w
    bbox_h = (ymax - ymin) / img_h

    # Clip to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    bbox_w = max(0.0, min(1.0, bbox_w))
    bbox_h = max(0.0, min(1.0, bbox_h))

    return x_center, y_center, bbox_w, bbox_h


def get_folders_to_scan(dataset_dir: str, target_classes: Optional[List[str]] = None) -> List[str]:
    """Finds optimal subfolders to scan based on target classes."""
    all_subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not target_classes:
        return [d for d in all_subdirs if d.startswith("Objects")]

    chosen_dirs = set()
    for cls in target_classes:
        if cls in CLASS_TO_FOLDERS:
            for f in CLASS_TO_FOLDERS[cls]:
                if f in all_subdirs:
                    chosen_dirs.add(f)
        else:
            # Fallback search by normalized name
            clean_cls = cls.lower().replace("-", "").replace("_", "").replace(" ", "")
            for d in all_subdirs:
                clean_d = d.lower().replace("-", "").replace("_", "").replace(" ", "")
                if clean_cls in clean_d or clean_d in clean_cls:
                    chosen_dirs.add(d)

    # If nothing matched, scan all Objects dirs
    if not chosen_dirs:
        return [d for d in all_subdirs if d.startswith("Objects")]

    return sorted(list(chosen_dirs))


def collect_images_and_annotations(
    dataset_dir: str,
    target_classes: Optional[List[str]] = None,
    max_images_per_class: Optional[int] = None
) -> List[Dict]:
    """
    Scans the dataset directory and collects all valid (img_path, ann_path, objects) tuples.
    If target_classes is provided, only retains objects matching those classes.
    """
    subdirs = get_folders_to_scan(dataset_dir, target_classes)
    collected_entries = []
    class_img_count = {cls: 0 for cls in target_classes} if target_classes else {}
    target_set = set(target_classes) if target_classes else None

    for sd in subdirs:
        img_dir = os.path.join(dataset_dir, sd, "img")
        ann_dir = os.path.join(dataset_dir, sd, "ann")
        if not (os.path.exists(img_dir) and os.path.exists(ann_dir)):
            continue

        ann_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
        for afn in ann_files:
            ann_path = os.path.join(ann_dir, afn)
            img_filename = afn[:-5] if afn.endswith(".json") else afn
            img_path = os.path.join(img_dir, img_filename)
            if not os.path.exists(img_path):
                continue

            try:
                w, h, objects = parse_supervisely_annotation(ann_path)
                if not objects or w <= 0 or h <= 0:
                    continue

                if target_set is not None:
                    matched_objects = [o for o in objects if o["classTitle"] in target_set]
                else:
                    matched_objects = objects

                if not matched_objects:
                    continue

                if max_images_per_class is not None and target_classes:
                    under_cap = False
                    for o in matched_objects:
                        cls_name = o["classTitle"]
                        if class_img_count.get(cls_name, 0) < max_images_per_class:
                            under_cap = True
                            break
                    if not under_cap:
                        continue

                    for o in matched_objects:
                        cls_name = o["classTitle"]
                        class_img_count[cls_name] = class_img_count.get(cls_name, 0) + 1

                collected_entries.append({
                    "img_path": img_path,
                    "ann_path": ann_path,
                    "img_filename": img_filename,
                    "width": w,
                    "height": h,
                    "objects": matched_objects
                })
            except Exception:
                continue

    return collected_entries


def export_to_yolo_dataset(
    dataset_dir: str,
    output_dir: str,
    target_classes: List[str],
    train_ratio: float = 0.8,
    max_images_per_class: Optional[int] = 200,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    create_zip: bool = True
) -> str:
    """
    Exports filtered images and labels to a standard YOLO format:
    output_dir/
      images/
        train/
        val/
      labels/
        train/
        val/
      data.yaml

    Returns the path to the data.yaml file or zip file.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_train = os.path.join(output_dir, "images", "train")
    images_val = os.path.join(output_dir, "images", "val")
    labels_train = os.path.join(output_dir, "labels", "train")
    labels_val = os.path.join(output_dir, "labels", "val")

    for p in [images_train, images_val, labels_train, labels_val]:
        os.makedirs(p, exist_ok=True)

    class_to_id = {cls_name: i for i, cls_name in enumerate(target_classes)}

    if progress_callback:
        progress_callback(5, 100, "Đang quét và thu thập dữ liệu ảnh...")

    entries = collect_images_and_annotations(
        dataset_dir,
        target_classes=target_classes,
        max_images_per_class=max_images_per_class
    )

    total_entries = len(entries)
    if total_entries == 0:
        raise ValueError("Không tìm thấy ảnh nào khớp với danh sách nhãn đã chọn!")

    random.seed(42)
    random.shuffle(entries)

    split_idx = int(total_entries * train_ratio)
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]

    def process_split(split_list, img_dst_dir, lbl_dst_dir, split_name, start_prog, end_prog):
        for i, item in enumerate(split_list):
            src_img = item["img_path"]
            fname = Path(src_img).name
            stem = Path(src_img).stem

            dst_img = os.path.join(img_dst_dir, fname)
            shutil.copy2(src_img, dst_img)

            dst_lbl = os.path.join(lbl_dst_dir, f"{stem}.txt")
            w, h = item["width"], item["height"]

            lines = []
            for obj in item["objects"]:
                cls_name = obj["classTitle"]
                if cls_name not in class_to_id:
                    continue
                cls_id = class_to_id[cls_name]
                xmin, ymin, xmax, ymax = obj["box"]
                xc, yc, bw, bh = convert_box_to_yolo(xmin, ymin, xmax, ymax, w, h)
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.writelines(lines)

            if progress_callback and (i % 25 == 0 or i == len(split_list) - 1):
                pct = int(start_prog + (end_prog - start_prog) * (i + 1) / len(split_list))
                progress_callback(pct, 100, f"Đang xuất {split_name}: {i + 1}/{len(split_list)}")

    process_split(train_entries, images_train, labels_train, "Train", 10, 60)
    process_split(val_entries, images_val, labels_val, "Val", 60, 85)

    yaml_content = f"""# Dollar Street YOLOv8 / YOLOv11 Dataset Configuration
# Generated automatically by Dollar Street AI Studio

path: .
train: images/train
val: images/val

# Classes
nc: {len(target_classes)}
names: {target_classes}
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    zip_path = None
    if create_zip:
        if progress_callback:
            progress_callback(86, 100, "Đang nén file zip để sẵn sàng upload Kaggle...")
        zip_path = os.path.join(os.path.dirname(output_dir), f"{Path(output_dir).name}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, output_dir)
                    zipf.write(full_p, rel_p)

    if progress_callback:
        progress_callback(100, 100, f"Hoàn tất! Đã xuất {total_entries} ảnh và file zip.")

    return zip_path if (create_zip and zip_path) else yaml_path

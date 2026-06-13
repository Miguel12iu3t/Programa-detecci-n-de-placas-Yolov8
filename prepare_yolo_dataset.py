import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: Path) -> List[Path]:
    images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXT]
    images.sort()
    return images


def parse_bbox_from_ccpd_name(file_path: Path) -> Optional[Tuple[int, int, int, int]]:
    stem = file_path.stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None

    bbox_part = parts[2]
    if "_" not in bbox_part or "&" not in bbox_part:
        return None

    try:
        p1, p2 = bbox_part.split("_")
        x1, y1 = map(int, p1.split("&"))
        x2, y2 = map(int, p2.split("&"))
    except Exception:
        return None

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return x1, y1, x2, y2


def clip_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return x1, y1, x2, y2


def bbox_to_yolo(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def write_annotation(label_path: Path, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]):
    h, w = image_shape[:2]
    cx, cy, bw, bh = bbox_to_yolo(bbox, w, h)
    with label_path.open("w", encoding="utf-8") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def prepare_split(image_paths: List[Path], target_dir: Path, subfolder: str):
    images_out = target_dir / "images" / subfolder
    labels_out = target_dir / "labels" / subfolder
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for img_path in image_paths:
        shutil.copy2(str(img_path), str(images_out / img_path.name))


def main():
    parser = argparse.ArgumentParser(description="Prepara dataset YOLO desde CCPD2019")
    parser.add_argument("--ccpd-root", required=True, help="Ruta a CCPD2019")
    parser.add_argument("--out-dir", default="dataset", help="Carpeta de salida YOLO")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fracción de validación")
    parser.add_argument("--max-per-folder", type=int, default=500, help="Máximo de imágenes por subcarpeta")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para shuffle")
    args = parser.parse_args()

    ccpd_root = Path(args.ccpd_root)
    out_dir = Path(args.out_dir)

    if not ccpd_root.exists() or not ccpd_root.is_dir():
        raise FileNotFoundError(f"Carpeta CCPD inválida: {ccpd_root}")

    random_state = np.random.RandomState(args.seed)
    valid_images = []
    folder_counts = {}

    for subset in sorted(p for p in ccpd_root.iterdir() if p.is_dir()):
        images = collect_images(subset)
        if not images:
            print(f"[AVISO] No hay imágenes en subcarpeta: {subset.name}")
            continue

        selected = list(random_state.permutation(images))
        if len(selected) > args.max_per_folder:
            selected = selected[: args.max_per_folder]

        folder_valid = 0
        for img_path in selected:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            bbox = parse_bbox_from_ccpd_name(img_path)
            if bbox is None:
                continue
            bbox = clip_bbox(bbox, img.shape[1], img.shape[0])
            if bbox is None:
                continue
            valid_images.append((img_path, bbox, img.shape))
            folder_valid += 1

        folder_counts[subset.name] = folder_valid
        print(f"[{subset.name}] imágenes procesadas: {len(selected)}  válidas: {folder_valid}")

    if not valid_images:
        raise ValueError("No se pudo extraer ningún bbox válido desde los nombres CCPD.")

    random_state.shuffle(valid_images)
    split_idx = int(len(valid_images) * (1.0 - args.val_split))
    train_items = valid_images[:split_idx]
    val_items = valid_images[split_idx:]

    for subset_name, items in [("train", train_items), ("val", val_items)]:
        subset_images = out_dir / "images" / subset_name
        subset_labels = out_dir / "labels" / subset_name
        subset_images.mkdir(parents=True, exist_ok=True)
        subset_labels.mkdir(parents=True, exist_ok=True)

        for img_path, bbox, shape in items:
            dst_image = subset_images / img_path.name
            shutil.copy2(str(img_path), str(dst_image))
            label_path = subset_labels / f"{img_path.stem}.txt"
            write_annotation(label_path, bbox, shape)

    data_yaml = out_dir / "data_ccpd.yaml"
    data_yaml.write_text(
        f"train: {out_dir / 'images' / 'train'}\n"
        f"val: {out_dir / 'images' / 'val'}\n"
        "nc: 1\n"
        "names: ['placa']\n"
    )

    print("Dataset YOLO preparado con éxito.")
    print("Conteo válido por subcarpeta:")
    for folder_name, count in folder_counts.items():
        print(f"  {folder_name}: {count}")
    print(f"Imágenes train: {len(train_items)}")
    print(f"Imágenes val:   {len(val_items)}")
    print(f"Salida:         {out_dir.resolve()}")


if __name__ == "__main__":
    main()

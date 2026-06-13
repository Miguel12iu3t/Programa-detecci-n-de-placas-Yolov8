import argparse
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import cv2

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "numpy no está instalado. Instala con `pip install numpy`."
    ) from exc

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics no está instalado. Instala con `pip install ultralytics`."
    ) from exc


def load_reader():
    try:
        import easyocr
    except ImportError as exc:
        raise ImportError(
            "easyocr no está instalado. Instala con `pip install easyocr`."
        ) from exc
    return easyocr.Reader(['en'], gpu=False)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def filter_text(text: str) -> str:
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    return clean


def crop_plate(image, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    return image[y1:y2, x1:x2]


def make_monochrome(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return th


def draw_text(img, text, position, font_scale=0.7, thickness=2, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    x, y = position
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 8, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)


def annotate_image(image, boxes, labels):
    annotated = image.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        draw_text(annotated, label, (x1, y1), font_scale=0.6, thickness=2)
    return annotated


def open_image_file(image_path: Path):
    try:
        if platform.system() == "Windows":
            os.startfile(str(image_path))
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(image_path)], check=False)
        else:
            subprocess.run(["xdg-open", str(image_path)], check=False)
    except Exception as exc:
        print(f"No se pudo abrir la imagen {image_path}: {exc}")


def make_summary_image(annotated, plates, monos, texts):
    if len(plates) == 0:
        return annotated

    header_height = 80
    gap = 30
    side_margin = 30
    content_width = annotated.shape[1]
    plate_width = max(p.shape[1] for p in plates)
    mono_width = max(m.shape[1] for m in monos)
    right_panel_width = plate_width + mono_width + 40
    total_width = content_width + right_panel_width + side_margin * 3

    card_heights = [max(p.shape[0], m.shape[0]) + 90 for p, m in zip(plates, monos)]
    right_height = sum(card_heights) + gap * (len(plates) - 1)
    total_height = max(annotated.shape[0] + header_height + 20, right_height + header_height + 20)

    summary = np.ones((total_height, total_width, 3), dtype=np.uint8) * 245
    cv2.rectangle(summary, (0, 0), (total_width, header_height), (30, 30, 30), -1)
    draw_text(summary, "Informe de placa - Detección y OCR", (side_margin, 55), font_scale=1.1, thickness=2, bg_color=(30, 30, 30), text_color=(255, 255, 255))

    content_y = header_height + 20
    summary[content_y:content_y + annotated.shape[0], side_margin:side_margin + annotated.shape[1]] = annotated

    x_offset = side_margin + content_width + side_margin
    y_offset = header_height + 20

    for idx, (plate, mono, text) in enumerate(zip(plates, monos, texts)):
        card_h = max(plate.shape[0], mono.shape[0]) + 90
        card_w = right_panel_width
        card = np.ones((card_h, card_w, 3), dtype=np.uint8) * 255
        cv2.rectangle(card, (0, 0), (card_w - 1, card_h - 1), (200, 200, 200), 2)

        plate_x = 10
        plate_y = 20
        card[plate_y:plate_y + plate.shape[0], plate_x:plate_x + plate.shape[1]] = plate

        mono_x = plate_x + plate_width + 20
        mono_y = 20
        mono_bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        card[mono_y:mono_y + mono_bgr.shape[0], mono_x:mono_x + mono_bgr.shape[1]] = mono_bgr

        info_text = f"Placa {idx + 1}: {text or 'N/A'}"
        draw_text(card, info_text, (10, card_h - 20), font_scale=0.7, thickness=2, bg_color=(50, 50, 50), text_color=(255, 255, 255))

        summary[y_offset:y_offset + card_h, x_offset:x_offset + card_w] = card
        y_offset += card_h + gap

    cv2.rectangle(summary, (side_margin - 2, content_y - 2), (side_margin + annotated.shape[1] + 2, content_y + annotated.shape[0] + 2), (0, 120, 200), 3)
    cv2.rectangle(summary, (x_offset - 2, header_height + 18), (x_offset + right_panel_width + 2, y_offset - gap + 2), (0, 120, 200), 3)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Inferencia con modelo YOLO CCPD")
    parser.add_argument("--weights", default="runs/train/ccpd_yolo/weights/best.pt",
                        help="Ruta a los pesos YOLO entrenados")
    parser.add_argument("--source", required=True,
                        help="Ruta a la imagen o carpeta de imágenes")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral de confianza")
    parser.add_argument("--save-dir", default="results",
                        help="Carpeta para guardar resultados")
    parser.add_argument("--ocr", action="store_true",
                        help="Activar extracción OCR de texto en las placas")
    parser.add_argument("--show", action="store_true",
                        help="Abrir la imagen de informe al terminar cada procesado")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights}")

    save_dir = ensure_dir(Path(args.save_dir))
    plates_dir = ensure_dir(save_dir / "plates")
    plates_mono_dir = ensure_dir(save_dir / "plates_mono")

    reader = load_reader() if args.ocr else None

    model = YOLO(str(weights))
    results = model.predict(source=args.source, conf=args.conf, save=True, save_dir=str(save_dir))

    summary = []
    for result in results:
        image_path = Path(result.path) if hasattr(result, 'path') else None
        image = None
        if image_path is not None and image_path.exists():
            image = cv2.imread(str(image_path))
        elif hasattr(result, 'orig_img'):
            image = result.orig_img
        else:
            raise RuntimeError("No se pudo cargar la imagen de entrada para OCR.")

        image_name = image_path.name if image_path is not None else "image"
        plates = []
        monos = []
        texts = []
        boxes = []
        labels = []

        if result.boxes is not None and len(result.boxes) > 0:
            print(f"Detecciones en {image_name}: {len(result.boxes)}")
            for idx, box in enumerate(result.boxes):
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (x1, y1, x2, y2)
                print(f"  conf={conf:.3f} bbox={bbox}")

                crop = crop_plate(image, bbox)
                if crop.size == 0:
                    continue

                crop_name = plates_dir / f"{image_name}_plate_{idx + 1}.jpg"
                cv2.imwrite(str(crop_name), crop)

                mono = make_monochrome(crop)
                mono_zoom = cv2.resize(mono, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                mono_name = plates_mono_dir / f"{image_name}_plate_{idx + 1}_mono.jpg"
                cv2.imwrite(str(mono_name), mono_zoom)

                ocr_text = ""
                if reader is not None:
                    detection = reader.readtext(mono, detail=0,
                                                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    ocr_text = filter_text(" ".join(detection))

                plates.append(crop)
                monos.append(mono_zoom)
                texts.append(ocr_text or "N/A")
                boxes.append(bbox)
                labels.append(ocr_text or f"plate {idx + 1}")
        else:
            print(f"No se detectaron objetos en {image_name}.")

        annotated = annotate_image(image, boxes, labels)
        annotated_name = save_dir / f"{image_name}_annotated.jpg"
        cv2.imwrite(str(annotated_name), annotated)

        report = make_summary_image(annotated, plates, monos, texts)
        report_name = save_dir / f"{image_name}_plate_report.jpg"
        cv2.imwrite(str(report_name), report)

        if args.show:
            open_image_file(report_name)

        summary.append({
            "image": image_name,
            "plates": [
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(box.conf[0].item()) if box.conf is not None else 0.0,
                    "text": texts[idx],
                    "crop": str((plates_dir / f"{image_name}_plate_{idx + 1}.jpg").relative_to(save_dir)),
                    "mono": str((plates_mono_dir / f"{image_name}_plate_{idx + 1}_mono.jpg").relative_to(save_dir)),
                }
                for idx, (x1, y1, x2, y2) in enumerate(boxes)
            ],
            "annotated": str(annotated_name.relative_to(save_dir)),
            "report": str(report_name.relative_to(save_dir))
        })

    summary_path = save_dir / "ocr_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados guardados en: {save_dir.resolve()}")
    print(f"Resúmenes de OCR guardados en: {summary_path}")


if __name__ == "__main__":
    main()

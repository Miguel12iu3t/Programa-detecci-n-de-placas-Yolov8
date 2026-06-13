import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics no está instalado. Instala con `pip install ultralytics`."
    ) from exc


def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo YOLOv8 en CCPD.")
    parser.add_argument("--data", default="dataset/data_ccpd.yaml",
                        help="Ruta al archivo YAML de dataset YOLO")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Modelo base YOLOv8")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Número de épocas")
    parser.add_argument("--batch", type=int, default=16,
                        help="Tamaño de batch")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamaño de imagen")
    parser.add_argument("--project", default="runs/train",
                        help="Carpeta de salida de entrenamiento")
    parser.add_argument("--name", default="ccpd_yolo",
                        help="Nombre del experimento")
    parser.add_argument("--cache", action="store_true",
                        help="Cachear imágenes para acelerar el entrenamiento")
    args = parser.parse_args()

    data_file = Path(args.data)
    if not data_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_file}")

    model = YOLO(args.model)
    model.train(
        data=str(data_file),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        cache=args.cache,
    )

    print(f"Entrenamiento completado. Resultados en: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()

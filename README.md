#Descargar Dataset CCPD2019:
link: https://www.kaggle.com/datasets/binh234/ccpd2019

# YOLO CCPD Detector

Versión independiente de detección de placas con YOLO usando el dataset CCPD2019.

## Requisitos

```bash
pip install -r requirements.txt
```

## Preparar dataset YOLO

```bash
python prepare_yolo_dataset.py \
  --ccpd-root "C:/Users/gomez/OneDrive/Escritorio/placas/CCPD2019" \
  --out-dir "dataset" \
  --val-split 0.2 \
  --max-per-folder 500
```

Esto generará:

- `dataset/images/train`
- `dataset/images/val`
- `dataset/labels/train`
- `dataset/labels/val`
- `dataset/data_ccpd.yaml`

## Entrenar modelo YOLO

```bash
python train_yolo.py \
  --data "dataset/data_ccpd.yaml" \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640
```

El modelo entrenado se guardará en `runs/train/ccpd_yolo`

## Inferencia

```bash
python infer_yolo.py \
  --weights runs/train/ccpd_yolo/weights/best.pt \
  --source "path/to/image.jpg" \
  --save-dir "results"
```

La imagen resultante se guardará en `results/`.

## OCR de placas y recortes

Para extraer texto de la placa y guardar la región detectada en color y en monocromático, usa el flag `--ocr`:

```bash
python infer_yolo.py \
  --weights runs/train/ccpd_yolo/weights/best.pt \
  --source "path/to/image.jpg" \
  --save-dir "results" \
  --ocr
```

Si quieres que el informe se abra automáticamente luego de cada imagen, añade `--show`:

```bash
python infer_yolo.py \
  --weights runs/train/ccpd_yolo/weights/best.pt \
  --source "path/to/image.jpg" \
  --save-dir "results" \
  --ocr \
  --show
```

Se generarán también:

- `results/plates/`: recortes en color de cada placa detectada
- `results/plates_mono/`: recortes en monocromático y con zoom
- `results/ocr_results.json`: texto detectado y datos de cada placa

## Ejemplos de uso

### Inferir una sola imagen concreta

```bash
python infer_yolo.py \
  --weights "runs/train/ccpd_yolo_test/weights/best.pt" \
  --source "C:/Users/gomez/OneDrive/Escritorio/placas/CCPD2019/ccpd_challenge/01-0_0-277&502_421&560-420&560_277&559_278&502_421&503-0_0_28_7_24_26_24-143-28.jpg" \
  --save-dir "results" \
  --ocr
```

### Inferir todas las imágenes de una carpeta

```bash
python infer_yolo.py \
  --weights "runs/train/ccpd_yolo_test/weights/best.pt" \
  --source "C:/Users/gomez/OneDrive/Escritorio/placas/CCPD2019/ccpd_challenge" \
  --save-dir "results" \
  --ocr
```

## Salida esperada

- `results/<imagen>_annotated.jpg`: imagen original con bounding box y texto detectado
- `results/<imagen>_plate_report.jpg`: informe visual profesional con la placa recortada, el zoom en monocromático y la etiqueta OCR
- `results/plates/<imagen>_plate_1.jpg`: recorte en color de la placa
- `results/plates_mono/<imagen>_plate_1_mono.jpg`: recorte en monocromático y zoom
- `results/ocr_results.json`: resumen con texto OCR y coordenadas de cada placa

## Notas

- El script usa el mismo dataset CCPD2019 original con las cajas extraídas desde el nombre de archivo.
- El modelo de clase se entrena con un solo objeto: `placa`.

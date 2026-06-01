# Automated Engineering-Drawing Checklist

Hackathon project. Upload a drawing sheet → one YOLO model detects regions/symbols →
OCR reads the title block → a rule engine fills a checklist report (PASS / FAIL / NA).
The checklist is **dataset-driven**: every item maps to a class the model can detect,
defined in [`config.yaml`](config.yaml) — no code edits to retarget it.

See [the full plan](/Users/jitu/.claude/plans/users-jitu-library-cloudstorage-onedriv-nested-codd.md)
for the 10-day schedule and dataset decisions.

## How it fits together

```
upload sheet ─► YOLO (one model) ─► boxes+classes ─┐
                                                    ├─► rule engine ─► checklist report
            ─► crop title_block ─► OCR ─► text ─────┘                  + annotated image
```

| File | Role |
|---|---|
| `config.yaml` | classes, conf threshold, regex rules, **the checklist definition** |
| `src/detect.py` | load YOLO, run inference → list of `{cls, conf, xyxy}` |
| `src/ocr.py` | OCR the title-block crop (Tesseract) |
| `src/rules.py` | regex rules on OCR text (drawing-no, date) |
| `src/report.py` | combine detections + rules → checklist rows |
| `src/annotate.py` | draw boxes on the sheet |
| `src/pipeline.py` | the one entry point: image → (annotated, rows, debug) |
| `app.py` | Streamlit demo UI |

The app **runs today without a model** — every row shows `NA` until you add
`models/best.pt`. That lets you build the UI and rule engine before training finishes.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # torch may not support 3.14 yet
pip install -r requirements.txt
brew install tesseract                                   # the OCR binary (macOS)
streamlit run app.py
```

## Build the unified dataset (Roboflow)

Goal: ONE dataset, ONE class list, ONE model.

1. Make a free [Roboflow](https://roboflow.com) account → new **Object Detection** project.
2. From [Roboflow Universe](https://universe.roboflow.com), open each source dataset
   (listed in the plan) and **Clone** it into your project.
3. In your project, **rename/merge labels** so every source maps to the shared list:
   `title_block, revision_table, dimension, note, gdt_symbol, surface_roughness`
   (collapse all 43 GD&T types → the single `gdt_symbol`).
4. Add augmentation (flip off for drawings; use brightness/blur/rotate ±5°). Generate a
   version. **Export → YOLOv11** and copy the download snippet.

Keep `config.yaml: classes` in the **same order** as the generated `data.yaml`.

## Train the one model (Google Colab, free GPU)

New Colab notebook → Runtime → change type → **T4 GPU**:

```python
!pip install ultralytics roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")                  # from Roboflow settings
ds = rf.workspace("WS").project("PROJ").version(1).download("yolov11")

from ultralytics import YOLO
model = YOLO("yolo11n.pt")                          # nano = fast, good for a demo
model.train(data=f"{ds.location}/data.yaml", epochs=80, imgsz=960, batch=16)

# results land in runs/detect/train/ — grab the metrics screenshot for your slides
model.val()
```

Download `runs/detect/train/weights/best.pt` → put it in `models/best.pt`. Re-run the
app. Done.

**What to learn while doing this:** bounding boxes, train/val/test split,
precision/recall/mAP, reading `results.png` curves and the confusion matrix,
over- vs under-fitting. Ultralytics docs + one YOLO11 tutorial cover all of it.

## Retargeting to the real company checklist (future work)

Everything company-specific is in `config.yaml`: edit `rules.drawing_number.pattern`
to `^TR\s?\d\s?[A-Z]{3}\s?\d{4}$`, add/remove `checklist` items, point `model_path` at
a model trained on company drawings. No Python changes.

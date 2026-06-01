# Automated Engineering-Drawing Checklist

> Upload a mechanical drawing → an AI model finds the important parts, reads the title
> block, and **fills a QC checklist automatically** (PASS / FAIL / NA). A hackathon
> project that turns slow, manual drawing review into a 3-second upload.

This README is the **full story** — what we're building, why, and *how*, with real code
snippets. If you want the friendly concepts only, read [`LEARN.md`](LEARN.md). For the
deep ML internals (network architecture, label math, splits), read
[`DEEPDIVE.md`](DEEPDIVE.md).

---

## 1. The problem

Mechanical engineering drawings are checked **by hand** against a long QC checklist:
is the title block filled, is the drawing number formatted right, are the surface-
roughness symbols present, the GD&T symbols, the dimensions, the revision table…

That manual review is **slow, tedious, and inconsistent** — a tired reviewer misses a
symbol, and a wrong drawing causes scrap or rework downstream. We automate it.

**Constraint:** we can't train on the company's confidential drawings. So we train on
**free public drawing datasets** and build the checklist around what those let us detect.
The system is config-driven, so the real company checklist + company-trained model can be
dropped in later with **zero code changes**.

---

## 2. The core idea (the one thing to remember)

We split every check into two questions answered by two different tools:

| Question | Tool | Example |
|---|---|---|
| **"What is on the drawing, and where?"** | a trained **YOLO** model | there's a title block here, 3 dimensions there |
| **"Is the text / format correct?"** | plain **regex rules** | is the drawing number `TR-1-SMM-0042`? is the date `DD/MM/YY`? |

> **Model = "what + where." Rules = "is it right."** Detection needs ML; format checks
> don't. Keeping them separate makes the whole system simple and explainable.

---

## 3. How it works — the pipeline

```
upload drawing sheet
   │
   ├─► YOLO model ─────────► boxes + classes  (title_block, dimension, gdt_symbol, …)
   │
   ├─► crop the title_block box ─► OCR (Tesseract) ─► raw text
   │
   └─► Rule engine (reads config.yaml)
           • presence/count rules  ← from detections
           • regex/format rules    ← from OCR text
                     │
                     ▼
        Checklist report (Sl.No | Check point | Status | Remarks)  +  annotated image
```

One file orchestrates it — [`src/pipeline.py`](src/pipeline.py):

```python
def run(image_path, config):
    # 1) detect parts of the drawing
    detections, det_status = detect(image_path, config["model_path"], config["conf_threshold"])
    model_loaded = "No model" not in det_status and "not installed" not in det_status

    # 2) OCR only the title-block region (cleaner than the whole page)
    tb_box = first_box(detections, "title_block")
    ocr_text, ocr_status = ocr_region(image_path, tb_box) if tb_box else ("", "no title_block")

    # 3) turn detections + text into a PASS/FAIL/NA checklist
    rows = build_report(config, detections, ocr_text, model_loaded)

    # 4) draw the boxes for the demo
    annotated = annotate(image_path, detections)
    return annotated, rows, {...}
```

Let's walk each stage.

---

## 4. Detection — the model finds the parts

[`src/detect.py`](src/detect.py) wraps YOLO so the rest of the code never depends on the
ML library. It returns plain dicts:

```python
def detect(image_path, model_path, conf=0.25):
    model = _load_model(model_path)          # cached; None if file/lib missing
    if model is None:
        return [], "No model — train first."  # app still runs, rows show NA

    results = model.predict(image_path, conf=conf, verbose=False)
    names = model.names                       # {0: 'title_block', 1: 'revision_table', ...}
    dets = []
    for r in results:
        for box in r.boxes:
            dets.append({
                "cls":  names[int(box.cls[0])],          # class name
                "conf": float(box.conf[0]),              # confidence 0–1
                "xyxy": [float(v) for v in box.xyxy[0]], # pixel corners x1,y1,x2,y2
            })
    return dets, f"{len(dets)} detections."
```

So one detection looks like:
`{"cls": "title_block", "conf": 0.91, "xyxy": [610, 740, 940, 860]}`.
That's all the downstream code needs.

---

## 5. OCR — read the title block

The model gives us *where* the title block is. To read *what it says*, we crop that box
and run OCR ([`src/ocr.py`](src/ocr.py)). Cropping first = far less noise than OCRing the
whole drawing.

```python
def ocr_region(image_path, box=None):
    import pytesseract
    img = Image.open(image_path).convert("RGB")
    if box is not None:
        img = img.crop(tuple(int(v) for v in box))   # just the title block
    return pytesseract.image_to_string(img), "ok"
```

---

## 6. Rules — check the text format (no ML)

[`src/rules.py`](src/rules.py) is just regex. Patterns live in `config.yaml`, so changing
a rule never touches code:

```python
import re
def check_rule(rule_cfg, text):
    pattern = rule_cfg.get("pattern", "")
    if not pattern or not text:
        return False, None
    m = re.search(pattern, text)
    return (bool(m), m.group(0) if m else None)
```

Example: pattern `\d{2}/\d{2}/\d{2}` matches a `DD/MM/YY` date. If found → that checklist
row passes.

---

## 7. config.yaml — the brain (data, not code)

The checklist itself is **configuration**. This is the "dataset-driven checklist": every
item maps to a class the model detects or a regex rule.

```yaml
model_path: models/best.pt
conf_threshold: 0.25

classes:                      # must match the trained model's class order
  - title_block
  - revision_table
  - dimension
  - note
  - gdt_symbol
  - surface_roughness

rules:
  drawing_number:
    description: "Drawing number present & format valid"
    pattern: '[A-Z]{2,3}[-\s]?\d?[-\s]?[A-Z]{2,4}[-\s]?\d{3,4}'
  date:
    pattern: '\d{2}/\d{2}/\d{2}'

checklist:                    # each item auto-evaluated
  - id: 1
    point: "Title block present"
    type: presence            # PASS if class detected ≥1, else FAIL
    requires_class: title_block
  - id: 2
    point: "Drawing number present & format valid"
    type: rule                # PASS/FAIL from regex on title-block text
    requires_rule: drawing_number
  - id: 4
    point: "Surface-roughness symbols present"
    type: count               # PASS if ≥1, remark shows how many
    requires_class: surface_roughness
```

To retarget to the **real company checklist**: edit this file (add items, change the
drawing-number regex to `^TR\s?\d\s?[A-Z]{3}\s?\d{4}$`, point `model_path` at a company-
trained model). No Python edits.

---

## 8. Report — build PASS / FAIL / NA

[`src/report.py`](src/report.py) reads `config.checklist` and decides each row:

```python
for item in config["checklist"]:
    if item["type"] == "presence":
        n = counts.get(item["requires_class"], 0)
        status = "PASS" if n >= 1 else "FAIL"
        remarks = f"{n} found"
    elif item["type"] == "rule":
        matched, found = check_rule(rules_cfg[item["requires_rule"]], ocr_text)
        status = "PASS" if matched else "FAIL"
        remarks = f"matched: {found}" if matched else "pattern not found"
    # ... presence_optional → NA if missing, count_info → INFO
```

Status meanings: **PASS** (good), **FAIL** (problem), **NA** (not applicable / can't
check), **INFO** (just a count).

---

## 9. The demo UI — Streamlit

[`app.py`](app.py) turns the pipeline into a web page: upload → annotated image +
checklist table + a pass-score + CSV download + a confidence slider.

```python
conf = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05)
config["conf_threshold"] = conf

uploaded = st.file_uploader("Upload a drawing sheet", type=["png", "jpg", "jpeg"])
if uploaded:
    annotated, rows, debug = run(tmp_path, config)
    st.metric("Checks passed", f"{passed}/{actionable}", f"{failed} failed")
    st.image(annotated)
    st.table(rows)
    st.download_button("Download report (CSV)", rows_to_csv(rows), "report.csv")
```

Run it:

```bash
pip install -r requirements.txt
./scripts/serve.sh          # or: streamlit run app.py
```

It runs **today even without a trained model** — rows just show `NA` until you add
`models/best.pt`.

---

## 10. The data — what we train on

A detection dataset = **images + one label file per image**. Each label line is:

```
class_id  x_center  y_center  width  height      # all normalized 0–1
```

e.g. `0 0.81 0.92 0.30 0.12` = a `title_block` centered lower-right, 30% wide, 12% tall.

We **merge several free Roboflow datasets** into one, remap all their labels to our 6
classes, collapse all 43 GD&T symbol types into one `gdt_symbol` (too few examples each),
and export in YOLO format. Step-by-step in [`DATASET.md`](DATASET.md).

**Train / val / test split** (Roboflow does it): **70% / 20% / 10%**, disjoint.
- **train** — the model learns from these.
- **validation** — scored each epoch to catch overfitting + pick the best weights.
- **test** — sealed until the end for an honest final score.

Images resized to **960×960**; augment **train only** (rotate ±5°, brightness, blur — no
flips, text would mirror). Full label math + split reasoning in [`DEEPDIVE.md`](DEEPDIVE.md).

---

## 11. The model — YOLO11, and its architecture

We use **YOLO11** (Ultralytics) for object detection. Why: it gives *where* (boxes for
multiple objects), runs in one fast pass, is the easiest to train (`model.train(...)`),
and ships **pretrained weights** so our small dataset works via transfer learning.

Inside, YOLO11 has three stages:

```
image → BACKBONE (Conv + C3k2 blocks + SPPF + C2PSA attention)  → features at 3 scales
      → NECK     (PAN-FPN: fuse fine detail + big-picture context)
      → HEAD     (anchor-free, decoupled: class branch + box branch via DFL)
      → decode → confidence filter → NMS → final boxes
```

- **3 scales (P3/P4/P5)** = catches tiny symbols *and* big title blocks.
- **anchor-free** = each grid cell predicts a box directly (no templates).
- **NMS** = removes duplicate overlapping boxes.

Training minimizes a **loss** = class error (BCE) + box error (CIoU) + DFL, via
backpropagation. We fine-tune from `yolo11n.pt` (pretrained on COCO) — that's why a few
hundred drawings is enough. Full explanation in [`DEEPDIVE.md`](DEEPDIVE.md) §7–8.

### Training it (Google Colab, free GPU)

[`notebooks/train_yolo.ipynb`](notebooks/train_yolo.ipynb) does this end to end:

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")                     # pretrained nano model
model.train(data="data.yaml", epochs=80, imgsz=960, batch=16)
model.val()                                    # prints mAP, saves metric plots
# download runs/.../best.pt  →  put it in models/best.pt
```

Drop `best.pt` into `models/`, re-run the app, and the checklist comes alive.

---

## 12. Repo map

```
config.yaml              # the brain: classes, rules, the checklist
app.py                   # Streamlit demo UI
src/
  pipeline.py            # orchestrates detect → ocr → rules → report → annotate
  detect.py              # YOLO inference → [{cls, conf, xyxy}]
  ocr.py                 # Tesseract on the title-block crop
  rules.py               # regex format checks
  report.py              # build PASS/FAIL/NA rows from config.checklist
  annotate.py            # draw boxes (Pillow)
scripts/
  try_pretrained.py      # see detection work before training
  run_cli.py             # run the pipeline from terminal (offline demo)
  check_setup.py         # doctor: what's installed/missing
  serve.sh               # launch the app
notebooks/
  train_yolo.ipynb       # Colab training
docs/
  tomorrow.md            # ordered training-day steps
  pitch.md               # 5-slide demo script
  email_1367_dataset.md  # request for the bonus academic dataset
  data_yaml_reference.yaml
LEARN.md                 # friendly concepts
DEEPDIVE.md              # deep ML internals
DATASET.md               # how to build the unified dataset
```

---

## 13. Status & roadmap

**Done:** full pipeline, GUI, training notebook, dataset guide, docs. Runs end-to-end
(model-optional). **Next:** build the unified dataset, train the model on Colab, drop in
`best.pt`, collect demo drawings.

**Future work** (pitch as such): per-type GD&T classification, font/line-thickness checks,
native CAD/DWG vector parsing, multi-sheet revision linking — these need vector-CAD data
or company drawings, not image ML on public data.

---

## 14. Quickstart

```bash
git clone https://github.com/jay112298/hackathon.git
cd hackathon
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_setup.py     # see what's ready
./scripts/serve.sh                # launch the app (NA rows until best.pt exists)
```

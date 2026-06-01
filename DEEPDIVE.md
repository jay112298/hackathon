# DEEPDIVE.md — Everything under the hood

Read `LEARN.md` first for the friendly overview. This file goes deeper: data, labels,
splits, formats, the repo internals, the model, and YOLO11's actual network
architecture. Goal: you can explain any part to a judge.

---

## 1. The dataset — what it actually is

A dataset for object detection = **images + a label file per image**. Nothing more.

- **Images:** photos/scans/exports of engineering drawing sheets (`.jpg` / `.png`).
- **Labels:** for each image, a list of objects, each = `(class, box)`. The box is a
  rectangle around the thing (a title block, a dimension, a GD&T symbol, …).

We don't own a labeled drawing dataset, and we can't use company drawings. So we **merge
several free public datasets** from Roboflow Universe into one, remap all their labels to
**6 shared classes**, and train on that. (See `DATASET.md` for the click-by-click.)

Our 6 classes (the only things the model learns to find):

| id | class | what it marks on a drawing |
|----|-------|----------------------------|
| 0 | `title_block` | the name-plate box (bottom-right usually) |
| 1 | `revision_table` | the revision history table |
| 2 | `dimension` | a dimension callout (value + arrows) |
| 3 | `note` | a notes block / text note |
| 4 | `gdt_symbol` | any geometric-tolerance symbol (all 43 types collapsed to one) |
| 5 | `surface_roughness` | a surface-finish symbol |

**Why collapse 43 GD&T types → 1?** Few public images per type → the model can't learn 43
rare classes. One `gdt_symbol` class has enough examples to learn. The checklist only
needs "are GD&T symbols present," not which type. This is a deliberate trade-off.

---

## 2. Labels — the exact format

YOLO uses a dead-simple text format. For `drawing_001.jpg` there is `drawing_001.txt` in
a parallel `labels/` folder. One line per object:

```
class_id  x_center  y_center  width  height
```

- All 5 numbers except `class_id` are **normalized 0–1** (fraction of image width/height).
- `x_center, y_center` = center of the box. `width, height` = box size.
- Normalized so the label is **resolution-independent** — resize the image, label still valid.

Example `drawing_001.txt`:
```
0 0.81 0.92 0.30 0.12     # title_block, lower-right, 30% wide, 12% tall
2 0.45 0.40 0.08 0.03     # a dimension
2 0.60 0.55 0.07 0.03     # another dimension
4 0.22 0.70 0.04 0.05     # a gdt_symbol
```

Convert normalized → pixels: `x_px = x_center * image_width`, etc. To get corners
(`x1,y1,x2,y2`) you compute `x1 = x_px - w_px/2`, `y1 = y_px - h_px/2`, etc. Our
`src/detect.py` receives corners (`xyxy`) directly from Ultralytics — you rarely touch
the normalized form by hand. Roboflow writes these `.txt` files for you.

**You never label from scratch here** — you reuse + remap existing labels in Roboflow.
But know the format, because judges ask "how is a drawing labeled?"

---

## 3. Train / validation / test split

You never test a model on data it trained on — it would just recite memorized answers.
So the dataset is cut into 3 disjoint piles:

| Split | % (ours) | Used for | Model learns from it? |
|-------|---------|----------|------------------------|
| **train** | 70% | gradient updates (actual learning) | yes |
| **validation** | 20% | watched each epoch to detect overfitting + pick best epoch | no (only scored) |
| **test** | 10% | final honest score, touched once at the end | no |

- Roboflow does the split when you "Generate a version." It shuffles, then assigns each
  image to one pile.
- **Critical:** an image lives in exactly one split. If the same drawing leaked into
  train and test, your score would be fake-high ("data leakage").
- **Why a separate val AND test?** Val is peeked at repeatedly during training (to early-
  stop and choose the best weights), so it's slightly "used." Test stays sealed for a
  truly unbiased final number.

Folder layout after Roboflow export:
```
drawing-checklist-1/
  data.yaml              # class names + paths (Ultralytics reads this)
  train/ images/*.jpg  labels/*.txt
  valid/ images/*.jpg  labels/*.txt
  test/  images/*.jpg  labels/*.txt
```

`data.yaml` ties it together:
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
nc: 6
names: [title_block, revision_table, dimension, note, gdt_symbol, surface_roughness]
```

---

## 4. Size & format

- **Image format:** `.jpg`/`.png`, resized to **960×960** on export (matches training
  `imgsz=960`). Bigger = small symbols stay visible, but slower + more GPU memory.
- **Label format:** plain `.txt`, YOLO normalized (Section 2).
- **Dataset size:** depends what you cloned — expect a few hundred to ~2–3k images after
  augmentation. Small by ML standards; fine for a demo thanks to transfer learning (§7).
- **Augmentation** (Roboflow, applied to TRAIN only): rotate ±5°, brightness ±15%, slight
  blur. This synthesizes variations so the model generalizes from few originals. **No
  flips** — mirrored text/symbols would be wrong. Val/test are never augmented (they must
  reflect reality).
- **Model file:** `best.pt`, a PyTorch checkpoint (~5–20 MB for nano/small). Contains the
  learned weights + class names.

---

## 5. How the repo works (module by module)

Data flows through small single-purpose modules. The conductor is `src/pipeline.py`.

```
app.py / scripts/run_cli.py
        │  (UI / CLI — just I/O)
        ▼
src/pipeline.py  run(image, config)
        │
        ├─ src/detect.py   load YOLO best.pt, predict → [{cls, conf, xyxy}]
        ├─ src/ocr.py      crop the title_block box → Tesseract → text
        ├─ src/rules.py    regex checks on that text (drawing-no, date)
        ├─ src/report.py   detections + rules + config.checklist → PASS/FAIL/NA rows
        └─ src/annotate.py draw boxes on the image (Pillow)
        ▼
returns (annotated_image, checklist_rows, debug)
```

Module responsibilities:

| File | Input | Output | Key idea |
|------|-------|--------|----------|
| `config.yaml` | — | dict | **single source of truth**: classes, threshold, regex, the checklist itself |
| `detect.py` | image path, model path | list of detections | lazy-loads + caches model; returns `[]` gracefully if no model |
| `ocr.py` | image, box | text | only OCRs the cropped title block (cleaner than full page) |
| `rules.py` | rule cfg, text | (matched?, found) | pure regex, no ML |
| `report.py` | detections, ocr text, config | checklist rows | turns counts/rules into PASS/FAIL/NA per `config.checklist` |
| `annotate.py` | image, detections | PIL image | colored boxes + labels |
| `pipeline.py` | image | (image, rows, debug) | orchestrates the above in order |

**Why config-driven?** The checklist is *data* in `config.yaml`, not hardcoded in Python.
To retarget to the real company checklist you edit YAML — no code changes. That's the
whole "dataset-driven checklist" pitch.

**Graceful degradation:** every external dependency (model, ultralytics, tesseract) is
optional. Missing → that row shows `NA` with a reason, app still runs. Lets you build UI
before training finishes.

`scripts/`:
- `try_pretrained.py` — run a generic YOLO to *see* detection (learning).
- `run_cli.py` — run our full pipeline from terminal (offline demo / testing).
- `check_setup.py` — doctor: what's installed/missing.
- `serve.sh` — launch the Streamlit app.

---

## 6. The model — YOLO11, and why

We use **YOLO11** (the 2024/25 Ultralytics release) for **object detection**.

Why this model:
- **Detection, not classification** — we need *where* each item is (boxes), not one label.
- **Single-pass + fast** — runs in real time even on CPU for a demo.
- **Easiest to train** — Ultralytics gives `model.train(...)` / `model.predict(...)`; no
  manual training loop. Perfect for a solo beginner.
- **Pretrained weights** — `yolo11n.pt` already knows generic vision → transfer learning
  lets our tiny dataset work (§7).
- **Great tooling** — auto metrics, plots, export formats, huge community.

Why not the alternatives:
- **Faster R-CNN / RetinaNet** — accurate but slower, fiddlier, heavier to set up.
- **A plain CNN classifier** — gives one label per image, can't localize multiple objects.
- **A vision-LLM (GPT-4V)** — no training to show; judges want a trained model; also
  slower/cost per call.

**Model size choice:** `yolo11n` (nano) = smallest/fastest, good demo default. If accuracy
is low and you have GPU time, step up to `yolo11s`/`m` (more layers → more capacity →
slower).

---

## 7. Network architecture (what's inside YOLO11)

A detector is a neural network: pixels in → boxes out. YOLO11 has three stages —
**backbone → neck → head**.

```
image 960x960x3
   │
   ▼  BACKBONE  (extract features at shrinking resolution)
  Conv stem ─► C3k2 blocks ─► ... ─► SPPF ─► C2PSA(attention)
   │   produces feature maps at 3 scales: P3 (large, fine), P4 (mid), P5 (small, coarse)
   ▼  NECK  (PAN-FPN: mix features across scales)
  upsample + concat (top-down) ─► downsample + concat (bottom-up)
   │   so each scale sees both fine detail and big-picture context
   ▼  HEAD  (anchor-free, decoupled, per scale)
  for each grid cell: predict (class scores) + (box via DFL)
   │
   ▼  POST-PROCESS
  decode boxes ─► confidence filter ─► NMS ─► final [{cls, conf, xyxy}]
```

Key pieces explained:

- **Convolution (Conv):** the core operation. A small filter slides over the image
  detecting local patterns (edges, corners). Stacking many builds up from edges →
  shapes → "looks like a title block."
- **Backbone:** a stack of conv blocks that progressively shrinks spatial size and grows
  channel depth, turning raw pixels into rich **feature maps**. YOLO11 uses **C3k2**
  blocks (efficient residual conv blocks — cheap, gradient-friendly).
- **SPPF (Spatial Pyramid Pooling–Fast):** pools features at multiple receptive sizes so
  the net sees both tiny symbols and large blocks in one pass.
- **C2PSA:** a lightweight **attention** block YOLO11 added — lets the network focus on
  the informative regions, boosting accuracy at small cost.
- **Multi-scale (P3/P4/P5):** three feature maps at different resolutions. P3 (high-res)
  catches small objects (a roughness symbol); P5 (low-res) catches big ones (the title
  block). This is why YOLO handles objects of very different sizes.
- **Neck (PAN-FPN):** a Feature Pyramid Network + Path Aggregation Network. It fuses the
  three scales top-down and bottom-up so every scale carries both detail and context.
- **Head — anchor-free & decoupled:** for each cell on each scale the head outputs:
  - **class scores** (is this a title_block? a dimension? …), via a classification branch.
  - **box** via a separate regression branch using **DFL (Distribution Focal Loss)** —
    instead of directly regressing 4 numbers, it predicts a small probability
    distribution over offsets and takes the expectation, which is more accurate.
  "Anchor-free" = no predefined box templates; the cell predicts the box directly. Simpler
  and what modern YOLO uses.
- **Decode + NMS:** raw outputs are decoded to pixel boxes, low-confidence ones dropped,
  then **Non-Max Suppression** removes duplicate overlapping boxes for the same object
  (keeps the highest-confidence one). Output = the clean detection list our code uses.

You don't implement any of this — Ultralytics does. But this is the honest answer to
"what is the architecture."

---

## 8. Training internals (what `model.train` does each step)

1. **Forward pass:** push a batch of images through backbone→neck→head → predicted boxes
   + class scores.
2. **Loss** = how wrong the predictions are vs the labels. YOLO11 sums three:
   - **classification loss** (BCE) — wrong class.
   - **box/localization loss** (CIoU) — box in wrong place/size (IoU-based).
   - **DFL loss** — sharpness of the box-offset distribution.
3. **Backward pass (backprop):** compute the gradient of the loss w.r.t. every weight.
4. **Optimizer step (SGD/AdamW):** nudge weights down the gradient by the **learning
   rate**. Loss shrinks over time.
5. Repeat for all batches = 1 **epoch**; do `epochs` of them. **batch** = images per step
   (limited by GPU memory). **patience** early-stops if val stops improving.
6. Ultralytics saves `best.pt` (best val epoch) and `last.pt`.

**Transfer learning (why our small data works):** we start from `yolo11n.pt`, already
trained on the huge COCO dataset, so the backbone already knows generic edges/shapes. We
only **fine-tune** it on our drawings — far less data + time than training from scratch.

---

## 9. Inference path (the demo)

`model.predict(image)` → forward pass once → decode → NMS → boxes. No labels, no
gradients. Our `src/detect.py` wraps this and returns
`[{cls, conf, xyxy}]`. The rest of the pipeline (OCR, rules, report) is plain Python.
Fast enough on a laptop CPU for a live demo.

---

## 10. Metrics (how we judge the model)

- **IoU (Intersection over Union):** overlap between a predicted box and the true box,
  0–1. A prediction "counts" if IoU ≥ a threshold (e.g. 0.5).
- **Precision:** of boxes predicted, fraction correct. Low = false alarms.
- **Recall:** of true objects, fraction found. Low = misses.
- **mAP50:** mean Average Precision at IoU 0.5, averaged over classes — the headline
  score. `mAP50-95` averages over stricter thresholds (harder, lower number).
- **Confusion matrix:** which classes get mistaken for which (or for background).
- **PR curve:** precision vs recall as you vary the confidence threshold.

Ultralytics generates `results.png`, `confusion_matrix.png`, `PR_curve.png` after
training. Read them; screenshot for the pitch.

---

## 11. The non-ML half (don't forget it)

- **OCR (Tesseract):** image of text → string. We only OCR the detected `title_block`
  crop → cleaner result. Wrapper = `pytesseract`.
- **Regex (`src/rules.py`):** pattern-match the OCR text for formats the model can't
  judge — drawing-number pattern, `DD/MM/YY` date. Zero training, fully explainable.
- **Rule engine (`src/report.py`):** reads `config.checklist`, and per item decides
  PASS/FAIL/NA from a detection count or a regex result.

The split to remember: **model = "what + where"; regex/rules = "is the text/format
right." Two different tools for two different questions.**

---

## 12. Five things a judge might ask (and your answers)

1. *How is a drawing labeled?* → boxes + class, stored as YOLO normalized `.txt` (§2).
2. *Train/test split?* → 70/20/10, disjoint, Roboflow-generated, no leakage (§3).
3. *Why YOLO?* → fast single-pass detector, easy to train, pretrained, localizes multiple
   objects (§6).
4. *Architecture?* → backbone (C3k2/SPPF/attention) → PAN-FPN neck → anchor-free decoupled
   head + NMS (§7).
5. *Why does small data work?* → transfer learning from COCO-pretrained weights (§8).

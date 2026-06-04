# Colab Training Guide — Roboflow export → trained `best.pt`

Two parts: **(1)** finish the dataset in Roboflow, **(2)** train on Google Colab by
pasting the cells below one at a time. Each cell has a "what it does / what to change /
what you should see" note.

---

# PART 1 — Roboflow (do this first)

You've merged `symbols`, `eng drawing`, `gdt symbols` into your project. Finish it:

1. **Remap classes** (project → Classes / Modify Classes) to exactly these, delete the rest:

   | Source labels | → map to |
   |---|---|
   | Roughness | `surface_roughness` |
   | Flatness, Concentricity, Perpendicularity, Parallelity, Centrality, Angle, Position, Runout, Cylindricity, all GD&T types | `gdt_symbol` |
   | dimension, Diameter, Radius, Chamfer | `dimension` |
   | Notes / note | `note` |
   | borders, arrows, view labels, junk | **delete** |

2. **Hand-label `title_block`** (Annotate tab): draw a box around the title block on
   ~30–50 full-sheet images, class name `title_block`. (Optional: `revision_table` too.)

3. **Health Check** tab: see per-class counts. Any class with very few instances (< ~50)
   will train poorly — label more or drop it.

4. **Generate a version** (Versions → Generate New Version):
   - Preprocessing: Auto-Orient ON; Resize → **Stretch to 960×960**.
   - Train/Valid/Test split: **70/20/10**.
   - Augmentations (gentle — line art): Rotation ±5°, Brightness ±15%, Blur ≤1px.
     **No flips.** Aim ~2–3× images.
   - Click **Generate**.

5. **Export** (Versions → Export Dataset):
   - Format: **YOLOv11**.
   - Choose **"show download code"** → copy the Python snippet (it has your API key,
     workspace, project, version). You'll paste it into Cell 2.

That's the whole Roboflow side. Now Colab.

---

# PART 2 — Google Colab

## Setup before any cell

1. Go to https://colab.research.google.com → **File → New notebook**.
2. **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save.** (Free GPU.)
3. Paste each cell below into its own cell (`+ Code`), run top to bottom with **Shift+Enter**.

---

## Cell 1 — Install libraries + confirm the GPU

```python
!pip -q install ultralytics roboflow
import torch
print("Torch:", torch.__version__, "| GPU available:", torch.cuda.is_available())
!nvidia-smi -L
```

- **What it does:** installs Ultralytics (YOLO) + the Roboflow downloader. Prints whether
  a GPU is attached.
- **`!`** at the start runs a shell command (not Python). `-q` = quiet install.
- **What you should see:** `GPU available: True` and a line like `GPU 0: Tesla T4`.
- **If `False`:** you forgot Runtime → T4 GPU. Fix it and re-run.

---

## Cell 2 — Download your dataset from Roboflow

Paste **your** snippet from Roboflow export here. It looks like this (replace the
placeholders with your real values):

```python
from roboflow import Roboflow
rf = Roboflow(api_key="PASTE_YOUR_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("drawing-checklist")
dataset = project.version(1).download("yolov11")

DATA_YAML = f"{dataset.location}/data.yaml"
print("Dataset at:", dataset.location)
print("\n--- data.yaml (class order matters!) ---")
!cat {DATA_YAML}
```

- **What it does:** pulls the images + labels + `data.yaml` into the Colab machine.
- **What to change:** your API key, workspace, project name, version number — all in the
  snippet Roboflow gave you. (Keep the last 4 lines I added — they print the class order.)
- **What you should see:** a download progress bar, then the contents of `data.yaml`.
- **IMPORTANT:** look at the `names:` line. That's the exact class order the model learns.
  Send that line to me so `config.yaml` matches it. Example:
  `names: ['title_block', 'dimension', 'note', 'gdt_symbol', 'surface_roughness']`
- **Secret:** your API key is in here. Fine in Colab; never commit it to GitHub.

---

## Cell 3 — (optional) Peek at the data

```python
import glob, os
for split in ["train", "valid", "test"]:
    imgs = glob.glob(f"{dataset.location}/{split}/images/*")
    print(f"{split:6s}: {len(imgs)} images")
```

- **What it does:** counts images per split. Sanity check that the 70/20/10 split worked
  and you actually have data.
- **What you should see:** e.g. `train: 1400  valid: 400  test: 200`. If train is 0,
  something went wrong in the export.

---

## Cell 4 — Train the model

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")          # 'n' = nano: small + fast. Try 'yolo11s.pt' if time.
results = model.train(
    data=DATA_YAML,
    epochs=80,        # full passes over the data
    imgsz=960,        # input resolution (match the Roboflow 960 resize)
    batch=16,         # images per step; lower to 8 if you get an out-of-memory error
    patience=20,      # stop early if no improvement for 20 epochs
    project="runs",
    name="drawing",
    exist_ok=True,
)
print("Done. Best weights:", "runs/drawing/weights/best.pt")
```

- **What it does:** fine-tunes the pretrained `yolo11n.pt` on your drawings. This is the
  real training — takes ~20–60 min on the T4 depending on dataset size.
- **What to change:** `epochs` (raise if loss still dropping), `batch` (lower on OOM
  error), model size (`yolo11n` → `yolo11s`/`m` for more accuracy + more time).
- **What you should see:** a per-epoch table; the `box_loss` / `cls_loss` numbers should
  trend **down**. At the end it saves `best.pt`.
- **Common error:** `CUDA out of memory` → set `batch=8` (or `4`) and re-run.

---

## Cell 5 — Validate + read the metrics

```python
metrics = model.val()
print("mAP50:", round(metrics.box.map50, 3), "| mAP50-95:", round(metrics.box.map, 3))

# per-class mAP50-95
for i, name in model.names.items():
    print(f"  {name:18s} mAP50-95={metrics.box.maps[i]:.3f}")

from IPython.display import Image, display
for f in ["results.png", "confusion_matrix.png", "PR_curve.png"]:
    p = f"runs/drawing/{f}"
    if os.path.exists(p):
        print(p); display(Image(filename=p))
```

- **What it does:** scores the model on the validation set and shows the metric plots.
- **What you should see:** an overall `mAP50` (aim ~0.5+), per-class scores, and 3 charts.
- **How to read it:** a class with a low score (often crop-style `gdt_symbol`) is weak →
  either lower its confidence later, label more, or drop it from `config.yaml`.
- **Do this:** **screenshot `results.png` + `confusion_matrix.png`** — that's your "proof
  of real ML" slide.

---

## Cell 6 — Sanity-check on a few test images

```python
best = YOLO("runs/drawing/weights/best.pt")
test_imgs = glob.glob(f"{dataset.location}/test/images/*")[:3]
for img in test_imgs:
    r = best.predict(img, conf=0.25, save=True, verbose=False)
    found = [best.names[int(b.cls[0])] for b in r[0].boxes]
    print(os.path.basename(img), "->", found)
    display(Image(filename=os.path.join(r[0].save_dir, os.path.basename(img))))
```

- **What it does:** runs the trained model on 3 unseen test drawings and shows the boxes.
- **`[:3]`** takes the first 3 images. **`conf=0.25`** = show detections ≥25% confident.
- **What you should see:** drawings with colored boxes + a printed list of detected
  classes. This is the model actually working.

---

## Cell 7 — Download `best.pt`

```python
from google.colab import files
files.download("runs/drawing/weights/best.pt")
```

- **What it does:** downloads the trained model to your computer.
- **After download:** move it into the repo:
  ```bash
  mv ~/Downloads/best.pt /Users/jitu/code/hackathon/models/best.pt
  python scripts/check_setup.py     # models/best.pt now shows OK
  ./scripts/serve.sh                # app goes live
  ```

---

# Recap

1. Roboflow: remap → hand-label `title_block` → generate (960, 70/20/10, gentle augment)
   → export YOLOv11 snippet.
2. Colab: T4 GPU → Cell 1 install → Cell 2 download (send me the `names:` line) →
   Cell 4 train → Cell 5 metrics (screenshot) → Cell 7 download `best.pt`.
3. Local: drop `best.pt` in `models/`, run the app.

Total Colab time: ~1 hour, mostly waiting on Cell 4.

# BUILD_GUIDE.md — Build this project from scratch (and learn it)

This is the complete guide. If you start with an **empty folder** and follow it top to
bottom, you will end with the finished project **and** understand every piece.

It assumes you know **basic Python syntax** (variables, `if`, simple `def`, `print`). It
does **not** assume you know the "trickier" parts (`in` / `not in`, list comprehensions,
`with`, `try/except`, decorators, type hints…). Whenever one of those shows up, there is a
short **🐍 Python note** explaining it right there. There's also a full idiom cheat-sheet
in Chapter 15 you can flip back to.

> Mental anchor for the whole project: **a model answers "what is on the drawing and
> where" (boxes); plain rules answer "is the text/format right" (regex).** Two questions,
> two tools. Keep that in your head and nothing here is confusing.

**Contents**
0. How to read this
1. What we're building
2. Chapter 1 — Set up the workshop (Python, venv, pip, git)
3. Chapter 2 — Understand the data (datasets, labels, splits, formats)
4. Chapter 3 — Gather the data (Roboflow)
5. Chapter 4 — The libraries & tools we add (and why)
6. Chapter 5 — First code: the config file + loading it
7. Chapter 6 — Detection module
8. Chapter 7 — OCR module
9. Chapter 8 — Rules module
10. Chapter 9 — Report module (the checklist engine)
11. Chapter 10 — Annotate module
12. Chapter 11 — Pipeline (tie it together)
13. Chapter 12 — The app (Streamlit)
14. Chapter 13 — The model: YOLO11 + training
15. Chapter 14 — Run, test, debug
16. Chapter 15 — Python idiom cheat-sheet
17. Chapter 16 — ML glossary
18. Chapter 17 — Rebuild-from-scratch checklist

---

## 0. How to read this

Read in order. Type the code yourself (don't copy-paste) — typing is how it sticks. After
each chapter you'll have a working piece you can run. Nothing here needs a GPU except the
model training (Chapter 13), which runs free on Google Colab.

---

## 1. What we're building

A tool where you **upload an engineering drawing image** and it **auto-fills a quality
checklist**: is the title block present, is the drawing number formatted correctly, are
the surface-roughness symbols there, the GD&T symbols, the dimensions, etc. Output = a
table of PASS / FAIL / NA + the drawing with colored boxes around what it found.

Why: today this QC is done by hand — slow, boring, easy to slip. We automate it.

We can't use the company's private drawings to train, so we **train on free public
drawing datasets** and build the checklist around what we can actually detect. Everything
company-specific lives in one config file, so the real checklist can be swapped in later
without touching code.

---

## 2. Chapter 1 — Set up the workshop

You need four things before any project code: **Python, a virtual environment, pip, and
git.**

### Python
The language everything is written in. Install **Python 3.11 or 3.12** (newer 3.14 can
break some ML libraries — they lag behind). Check:

```bash
python3 --version
```

### Virtual environment (venv) — why
Each project needs specific versions of libraries. If you install everything globally,
projects clash. A **virtual environment** is a private box of libraries for *this* project
only. Create and "activate" it:

```bash
cd /path/to/your/project
python3.11 -m venv .venv          # creates a .venv/ folder = the box
source .venv/bin/activate         # step into the box (prompt shows (.venv))
```

`-m venv` means "run Python's built-in `venv` tool." After activating, `python` and `pip`
point inside the box.

### pip — installing libraries
`pip` downloads libraries from the internet. We list ours in a file called
`requirements.txt` so anyone can install all at once:

```
ultralytics>=8.3.0
streamlit>=1.30
pillow>=10.0
pyyaml>=6.0
pytesseract>=0.3.10
roboflow>=1.1.0
```

Install them all:

```bash
pip install -r requirements.txt
```

`-r` means "read this requirements file." `>=8.3.0` means "version 8.3.0 or newer."

### git — saving snapshots
`git` records snapshots ("commits") of your code so you can go back if you break
something, and push to GitHub.

```bash
git init                          # start tracking this folder
git add -A                        # stage all changes
git commit -m "first commit"      # save a snapshot with a message
```

### A note on Python files & importing
Code lives in `.py` files. A folder of `.py` files we treat as a package by putting an
empty `__init__.py` in it (we'll make a `src/` package). To use code from another file:

```python
from src.detect import detect      # import the function `detect` from src/detect.py
import yaml                         # import a whole library
```

> 🐍 **Python note — `import`:** `import yaml` loads the library so you can write
> `yaml.something()`. `from x import y` pulls one specific thing `y` out of module `x` so
> you can call `y()` directly. `src.detect` means "the file `detect.py` inside folder
> `src`."

**Project folders we'll create:**
```
config.yaml          # settings + the checklist
src/                 # the code package (with __init__.py)
scripts/             # helper command-line scripts
models/              # the trained model file goes here
data/                # datasets / demo images
notebooks/           # the Colab training notebook
```

---

## 3. Chapter 2 — Understand the data

Before gathering anything, understand what an object-detection dataset *is*.

### Images + labels
A dataset = **images** plus, for each image, a **label file** listing the objects in it.
An object = a class (what it is) + a box (where it is).

### The YOLO label format
For an image `drawing_001.jpg` there's a text file `drawing_001.txt`. One line per object:

```
class_id  x_center  y_center  width  height
```

- `class_id` is a number (0,1,2…) standing for a class name.
- the other four are the box, **normalized to 0–1** (a fraction of image width/height),
  so they stay correct even if you resize the image.

Example:
```
0 0.81 0.92 0.30 0.12     # class 0 (title_block), center at 81%,92%, size 30%×12%
2 0.45 0.40 0.08 0.03     # class 2 (dimension)
```

You won't write these by hand — the data tool (Roboflow) generates them. But judges ask
"how is a drawing labeled?" — this is the answer.

### Our classes
We use **6 classes**:
```
0 title_block
1 revision_table
2 dimension
3 note
4 gdt_symbol
5 surface_roughness
```
There are 43 different GD&T symbol types, but public data has too few of each to learn
them separately, so we **collapse all 43 into one `gdt_symbol`**. The checklist only needs
"are GD&T symbols present," not which kind.

### Train / validation / test split
You must never test a model on images it learned from — it would just recite memorized
answers. So split the data into three **disjoint** piles:

| Pile | Share | Purpose | Model learns from it? |
|---|---|---|---|
| train | 70% | the model learns (weights updated) | yes |
| validation | 20% | checked each round to catch overfitting & pick best | no, only scored |
| test | 10% | sealed; the final honest score | no |

"Disjoint" = an image is in exactly one pile. If the same drawing were in train *and*
test, your score would be a fake high (called **data leakage**).

### Size & format
- Images: `.jpg`/`.png`, resized to **960×960**.
- Labels: `.txt`, the format above.
- A `data.yaml` file lists the class names + folder paths so the trainer knows everything.
- The trained model is one file, `best.pt` (~5–20 MB).

Folder layout after export:
```
dataset/
  data.yaml
  train/ images/  labels/
  valid/ images/  labels/
  test/  images/  labels/
```

---

## 4. Chapter 3 — Gather the data (Roboflow)

We don't have labeled drawings, and labeling thousands by hand is too slow. So we reuse
**free public datasets** from **Roboflow Universe** and merge them into one.

**Roboflow** is a free website for preparing detection datasets: it hosts datasets, lets
you merge several, rename their labels to a common set, auto-create variations
(augmentation), split train/val/test, and export in YOLO format — all without code.

Steps:

1. Make a free account at roboflow.com → **Create Project** → type **Object Detection** →
   name it `drawing-checklist`. Note your **API key** (Settings → Roboflow API).
2. On Roboflow Universe, open each source dataset and **Clone** it into your project:
   - `eng-drawing` → gives `dimension`, `note`
   - `Eng_Drawing v7` → `title_block`, `revision_table`
   - `vanigaa engineering-drawing-datasets` → full-sheet objects
   - `roughness-3d` → `surface_roughness`
   - `gdt-symbols` (43 classes) → all map to `gdt_symbol`
3. **Remap labels:** in the project's Classes tool, rename every source label to one of
   our 6, and delete junk (borders, arrows). This is how 5 datasets become **one** clean
   label set → **one** model.
4. **Generate a version:** resize to 960×960; split 70/20/10; add gentle augmentation to
   the **train** pile only — rotate ±5°, brightness ±15%, slight blur. **No flips** (a
   mirrored "7" or symbol would be wrong).
5. **Export → YOLOv11** → copy the Python download snippet (you'll paste it into Colab).

That snippet looks like:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
dataset = rf.workspace("your-ws").project("drawing-checklist").version(1).download("yolov11")
```

Open the produced `data.yaml`; its `names:` order must match our class list (and our
`config.yaml`). If not, fix one to match the other.

> **Bonus dataset:** an academic set of 1,367 drawings (arXiv 2506.17374) has 9 categories
> in one clean source — perfect, but it's email-the-authors only (no public link). The
> request email is in `docs/email_1367_dataset.md`. Don't wait on it; the Roboflow merge
> is the real base.

---

## 5. Chapter 4 — The libraries & tools we add (and why)

| Tool / library | What it is | Why we use it |
|---|---|---|
| **Ultralytics (YOLO11)** | object-detection library | trains + runs our model in one line each |
| **Roboflow** | dataset prep website + small library | build the unified dataset, no code |
| **Google Colab** | free cloud notebook with a GPU | training needs a GPU we may not have |
| **Tesseract** + **pytesseract** | OCR engine + its Python wrapper | turn the title-block image into text |
| **Pillow (PIL)** | image library | open, crop, draw boxes |
| **PyYAML** | reads `.yaml` files | load `config.yaml` into Python |
| **`re`** (built-in) | regular expressions | check text formats (drawing-no, date) |
| **Streamlit** | turns a Python script into a web app | the upload-and-see demo UI |

We add each only when we need it. Next chapters build the code that uses them.

---

## 6. Chapter 5 — First code: the config file + loading it

We keep all settings **out of the code**, in `config.yaml`. That way you change behavior
(thresholds, rules, even the checklist) by editing one text file, never the Python.

### What YAML looks like
YAML is an indentation-based settings format. `key: value`; lists use `-`; nesting uses
indentation.

`config.yaml` (start small — we'll grow it):
```yaml
model_path: models/best.pt        # where the trained model will live
conf_threshold: 0.25              # ignore detections less confident than this

classes:                          # a LIST (each line starts with -)
  - title_block
  - revision_table
  - dimension
  - note
  - gdt_symbol
  - surface_roughness
```

### Load it in Python
Make `src/__init__.py` (empty — it just marks `src/` as a package), then start a loader.
We'll grow this file into `src/pipeline.py` later; for now just prove we can read config:

```python
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:        # open the file for reading
        return yaml.safe_load(f)      # parse YAML text into a Python dict

config = load_config()
print(config["conf_threshold"])       # -> 0.25
print(config["classes"])              # -> ['title_block', 'revision_table', ...]
```

> 🐍 **Python note — `with open(...) as f:`** This is a *context manager*. It opens the
> file, gives it the name `f`, and — crucially — **closes it automatically** when the
> indented block ends, even if an error happens. Without `with`, you'd have to remember
> `f.close()`. The `"r"` means "read mode."

> 🐍 **Python note — default argument `path="config.yaml"`** The `=` in a function
> definition gives a parameter a default. Call `load_config()` and it uses
> `"config.yaml"`; call `load_config("other.yaml")` to override.

> 🐍 **Python note — dict access `config["conf_threshold"]`** `yaml.safe_load` returns a
> **dictionary** (key → value). `config["key"]` fetches the value for that key. A YAML
> list becomes a Python **list** (`[...]`).

Run it: `python -c "from src.pipeline import load_config; print(load_config())"` (once the
file is `src/pipeline.py`).

---

## 7. Chapter 6 — Detection module (`src/detect.py`)

This is where the **model** runs. Goal: take an image path, return a simple list of what
was found. We hide the ML library behind a plain function so the rest of the project never
depends on it.

```python
from functools import lru_cache
import os

@lru_cache(maxsize=1)
def _load_model(model_path):
    if not os.path.exists(model_path):       # no trained model yet?
        return None
    from ultralytics import YOLO
    return YOLO(model_path)                   # load the model file

def detect(image_path, model_path, conf=0.25):
    model = _load_model(model_path)
    if model is None:
        return [], "No model — train first."  # app still works, shows NA

    results = model.predict(image_path, conf=conf, verbose=False)
    names = model.names                        # {0: 'title_block', 1: 'revision_table', ...}
    dets = []                                  # an empty list we'll fill
    for r in results:
        for box in r.boxes:                    # loop over every detected box
            dets.append({
                "cls":  names[int(box.cls[0])],
                "conf": float(box.conf[0]),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            })
    return dets, f"{len(dets)} detections."
```

What it returns — a **list of dictionaries**, e.g.:
```python
[{"cls": "title_block", "conf": 0.91, "xyxy": [610, 740, 940, 860]}]
```

Now the syntax used here:

> 🐍 **`@lru_cache(maxsize=1)`** A *decorator* — it wraps the function so its result is
> remembered ("cached"). The model file is big; we only want to load it once. With this,
> the first call loads it, later calls reuse the loaded model instantly.

> 🐍 **`if not os.path.exists(model_path):`** `os.path.exists(p)` returns `True`/`False`.
> `not` flips it. So this reads "if the file does NOT exist." `not` is how you negate a
> condition.

> 🐍 **`return None`** `None` is Python's "nothing" value. Returning it signals "no model."
> Later we check `if model is None:`. Use `is`/`is not` (not `==`) when comparing to `None`.

> 🐍 **`return [], "..."`** A function can return **two values at once**, separated by a
> comma — Python packs them into a *tuple*. The caller unpacks them:
> `dets, status = detect(...)`.

> 🐍 **`for r in results:` / `for box in r.boxes:`** A `for ... in ...:` loop runs the
> indented block once per item in a collection. `in` here means "iterate over." Nested
> loops = a loop inside a loop.

> 🐍 **`dets = []` then `dets.append(...)`** `[]` is an empty list. `.append(x)` adds `x`
> to the end. This is the standard "build a list as you go" pattern.

> 🐍 **`[float(v) for v in box.xyxy[0].tolist()]`** A *list comprehension* — a compact way
> to build a list. Read it right-to-left: "for each `v` in the box coordinates, compute
> `float(v)`, collect into a list." Equivalent long form:
> ```python
> tmp = []
> for v in box.xyxy[0].tolist():
>     tmp.append(float(v))
> ```
> `int(...)` / `float(...)` convert a value to a whole number / decimal.

> 🐍 **`f"{len(dets)} detections."`** An *f-string*. Putting `f` before the quotes lets you
> drop variables inside `{ }`. `len(dets)` is the length (item count) of the list.

We also add two tiny helpers used later:

```python
def count_by_class(detections):
    counts = {}
    for d in detections:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1   # tally per class
    return counts

def first_box(detections, cls_name):
    boxes = [d for d in detections if d["cls"] == cls_name]   # keep only this class
    if not boxes:
        return None
    best = max(boxes, key=lambda d: d["conf"])    # the most confident one
    return best["xyxy"]
```

> 🐍 **`counts.get(key, 0)`** `dict.get(key, default)` returns the value if the key exists,
> otherwise `default`. Safer than `counts[key]`, which would error on a missing key. Here
> it lets us start a tally at 0.

> 🐍 **`[d for d in detections if d["cls"] == cls_name]`** A list comprehension *with a
> filter* — only keeps items where the `if` is true. `==` tests equality.

> 🐍 **`if not boxes:`** An empty list is "falsy," so `not boxes` is `True` when the list is
> empty. Pythonic way to say "if there are none."

> 🐍 **`max(boxes, key=lambda d: d["conf"])`** `max` picks the biggest item. `key=` tells it
> *what* to compare by. `lambda d: d["conf"]` is a tiny one-line function meaning "given d,
> use its confidence." So this returns the highest-confidence box.

---

## 8. Chapter 7 — OCR module (`src/ocr.py`)

The model tells us *where* the title block is; OCR reads *what it says*. We crop to just
that box first — far cleaner than reading the whole busy drawing.

```python
from PIL import Image

def ocr_region(image_path, box=None):
    try:
        import pytesseract
    except ImportError:
        return "", "pytesseract not installed"

    img = Image.open(image_path).convert("RGB")
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]   # unpack the 4 corners
        img = img.crop((x1, y1, x2, y2))          # keep only that rectangle
    text = pytesseract.image_to_string(img)
    return text, "ok"
```

> 🐍 **`try: ... except ImportError:`** A *try/except* runs the risky code; if a specific
> error happens, it jumps to `except` instead of crashing. `ImportError` happens when a
> library isn't installed. So: "try to use OCR; if it's missing, return empty text and a
> message" — the app keeps running.

> 🐍 **`if box is not None:`** Only crop if a box was given. `is not None` = "has a real
> value." `box=None` in the signature makes the argument optional (defaults to nothing).

> 🐍 **`x1, y1, x2, y2 = [...]`** *Unpacking* — assign four list items to four variables in
> one line. The right side must have exactly four items.

---

## 9. Chapter 8 — Rules module (`src/rules.py`)

This is the **non-ML half**: checking text formats with **regular expressions (regex)**. A
regex is a pattern string. The built-in `re` library matches it against text.

```python
import re

def check_rule(rule_cfg, text):
    pattern = rule_cfg.get("pattern", "")
    if not pattern or not text:
        return False, None
    m = re.search(pattern, text)
    return (bool(m), m.group(0) if m else None)
```

Example pattern `\d{2}/\d{2}/\d{2}` means "two digits, slash, two digits, slash, two
digits" → matches a `DD/MM/YY` date. If `re.search` finds it anywhere in the text, the
check passes.

> 🐍 **`if not pattern or not text:`** `or` is true if *either* side is true. `not pattern`
> is true when the pattern string is empty; same for text. So: "if we have no pattern OR no
> text, give up." Combining `not` + `or` to guard against missing inputs is very common.

> 🐍 **`bool(m)`** `re.search` returns a *match object* if found, or `None` if not. `bool(x)`
> turns anything into `True`/`False` — a match object is truthy, `None` is falsy. So
> `bool(m)` is "did it match?"

> 🐍 **`m.group(0) if m else None`** A *ternary expression*: `VALUE_IF_TRUE if CONDITION
> else VALUE_IF_FALSE`. Reads "the matched text if there was a match, otherwise None."
> `m.group(0)` is the exact piece of text that matched.

Add the rules to `config.yaml`:
```yaml
rules:
  drawing_number:
    description: "Drawing number present & format valid"
    pattern: '[A-Z]{2,3}[-\s]?\d?[-\s]?[A-Z]{2,4}[-\s]?\d{3,4}'
  date:
    pattern: '\d{2}/\d{2}/\d{2}'
```
Change a pattern → change the rule. No code edit. (For the real company format
`TR X XXX XXXX`, the pattern becomes `^TR\s?\d\s?[A-Z]{3}\s?\d{4}$`.)

---

## 10. Chapter 9 — Report module (`src/report.py`)

The heart of the "checklist." It reads the `checklist:` section of `config.yaml` and turns
detections + OCR text into PASS / FAIL / NA rows.

First, the checklist in `config.yaml`:
```yaml
checklist:
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
    type: count               # PASS if ≥1, remark = how many
    requires_class: surface_roughness
  - id: 6
    point: "Dimensions detected"
    type: count_info          # never fails, just reports the count
    requires_class: dimension
```

The engine:
```python
from .detect import count_by_class
from .rules import check_rule

def build_report(config, detections, ocr_text, model_loaded):
    counts = count_by_class(detections)
    rules_cfg = config.get("rules", {})
    rows = []

    for item in config["checklist"]:
        itype = item["type"]
        status, remarks = "NA", ""

        if itype in ("presence", "presence_optional", "count", "count_info"):
            n = counts.get(item["requires_class"], 0)
            if not model_loaded:
                status, remarks = "NA", "model not loaded"
            elif itype == "presence":
                status = "PASS" if n >= 1 else "FAIL"
                remarks = f"{n} found"
            elif itype == "count":
                status = "PASS" if n >= 1 else "FAIL"
                remarks = f"{n} found"
            elif itype == "count_info":
                status, remarks = "INFO", f"{n} found"
            elif itype == "presence_optional":
                status = "PASS" if n >= 1 else "NA"
                remarks = f"{n} found" if n else "none (optional)"

        elif itype == "rule":
            if not ocr_text:
                status, remarks = "NA", "no title-block text"
            else:
                matched, found = check_rule(rules_cfg[item["requires_rule"]], ocr_text)
                status = "PASS" if matched else "FAIL"
                remarks = f"matched: {found}" if matched else "pattern not found"

        rows.append({"id": item["id"], "point": item["point"],
                     "status": status, "remarks": remarks})
    return rows
```

> 🐍 **`from .detect import count_by_class`** The leading dot means "from this same package
> (`src/`)." So it imports from `src/detect.py`. This is a *relative import*.

> 🐍 **`if itype in ("presence", "count", ...):`** **THE `in` keyword.** `x in collection`
> is `True` if `x` is one of the items. Here `(...)` is a *tuple* of allowed types, so this
> reads "if itype is any of these four." The opposite, `not in`, is `True` when `x` is
> absent: `if cls not in counts:`.

> 🐍 **`if / elif / else` chain** Python checks each condition top to bottom; the first true
> one runs, the rest are skipped. `elif` = "else if." Use it to pick one outcome among
> many.

> 🐍 **`status = "PASS" if n >= 1 else "FAIL"`** Ternary again — assign one of two values
> based on a condition. `>=` is "greater than or equal."

> 🐍 **`status, remarks = "NA", "model not loaded"`** Assign two variables at once. Right
> side is a tuple; it's unpacked into the two names.

Status meanings: **PASS** good, **FAIL** problem, **NA** can't check / not applicable,
**INFO** just a count.

---

## 11. Chapter 10 — Annotate module (`src/annotate.py`)

Draw the detected boxes on the image so the demo is visual.

```python
from PIL import Image, ImageDraw, ImageFont

_PALETTE = [(230,25,75), (60,180,75), (0,130,200), (245,130,48),
            (145,30,180), (70,240,240), (240,50,230), (210,245,60)]

def _color(cls_name):
    return _PALETTE[hash(cls_name) % len(_PALETTE)]   # stable color per class

def annotate(image_path, detections):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        c = _color(d["cls"])
        draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{d['cls']} {d['conf']:.2f}", fill=c)
    return img
```

> 🐍 **`hash(cls_name) % len(_PALETTE)`** `hash()` turns the class name into a big number;
> `%` (modulo) gives the remainder when divided by the palette length, so the result is
> always a valid index 0–7. Same name → same color every time. `_PALETTE[index]` picks the
> color at that position (lists are indexed from 0).

> 🐍 **`f"{d['conf']:.2f}"`** Inside an f-string, `:.2f` formats a number to 2 decimal
> places (0.9134 → `0.91`). Note single quotes inside the `{ }` because the f-string uses
> double quotes.

> 🐍 **Leading underscore `_color`, `_PALETTE`** A naming convention meaning "internal /
> private — not meant to be used outside this file." Python doesn't enforce it; it's a hint
> to humans.

---

## 12. Chapter 11 — Pipeline (`src/pipeline.py`)

This ties all modules together into one function the UI calls.

```python
import os
import yaml
from .detect import detect, first_box
from .ocr import ocr_region
from .report import build_report
from .annotate import annotate

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

def load_config(path=_CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run(image_path, config=None):
    if config is None:
        config = load_config()

    detections, det_status = detect(image_path, config["model_path"], config["conf_threshold"])
    model_loaded = "No model" not in det_status and "not installed" not in det_status

    tb_box = first_box(detections, "title_block")
    ocr_text, ocr_status = ocr_region(image_path, tb_box) if tb_box else ("", "no title_block")

    rows = build_report(config, detections, ocr_text, model_loaded)
    annotated = annotate(image_path, detections)

    debug = {"detection_status": det_status, "ocr_status": ocr_status,
             "ocr_text": ocr_text.strip(), "n_detections": len(detections)}
    return annotated, rows, debug
```

> 🐍 **`os.path.join(os.path.dirname(__file__), "..", "config.yaml")`** `__file__` is the
> path of the current file. `os.path.dirname(...)` strips to its folder. `".."` means "up
> one folder." `os.path.join` glues path pieces with the right separator for your OS. Net
> result: the path to `config.yaml` regardless of where you run the script from.

> 🐍 **`"No model" not in det_status`** `not in` on a **string** checks whether one string
> is *absent* inside another. So this is `True` when the status text doesn't contain "No
> model" — i.e. a model really loaded.

> 🐍 **`ocr_region(...) if tb_box else ("", "no title_block")`** Ternary controlling a call:
> only run OCR if we found a title block; otherwise use empty text. `if tb_box` is truthy
> when `tb_box` isn't `None`.

> 🐍 **`config=None` then `if config is None:`** A common pattern for "optional argument
> with a computed default." You can't write `config=load_config()` in the signature
> (it would run at import time), so you default to `None` and fill it inside.

---

## 13. Chapter 12 — The app (`app.py`, Streamlit)

Streamlit turns this script into a web page. Each `st.` call draws a widget. Streamlit
re-runs the whole script whenever the user interacts — so a slider change instantly
re-runs detection.

```python
import tempfile, csv, io
import streamlit as st
from src.pipeline import run, load_config

st.title("Engineering Drawing — Automated Checklist")
config = load_config()

conf = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05)
config["conf_threshold"] = conf

uploaded = st.file_uploader("Upload a drawing sheet", type=["png", "jpg", "jpeg"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    annotated, rows, debug = run(tmp_path, config)

    passed = sum(1 for r in rows if r["status"] == "PASS")
    failed = sum(1 for r in rows if r["status"] == "FAIL")
    st.metric("Checks passed", f"{passed}/{passed + failed}")
    st.image(annotated)
    st.table(rows)
```

Run with `streamlit run app.py` → opens in your browser.

> 🐍 **`sum(1 for r in rows if r["status"] == "PASS")`** A *generator expression* inside
> `sum`. It yields `1` for each row whose status is PASS; `sum` adds them → the count of
> PASS rows. Same idea as a list comprehension but without building a list. Classic
> "count items matching a condition" one-liner.

> 🐍 **`with tempfile.NamedTemporaryFile(...) as tmp:`** Streamlit gives the upload as
> in-memory bytes; YOLO wants a file path. We write the bytes to a temporary file (again
> using `with`, which cleans up handles) and pass its path on.

---

## 14. Chapter 13 — The model: YOLO11 + training

### Why YOLO
We need to find **multiple objects** and **where** they are — that's *object detection*,
not classification (one label per image). **YOLO11** (the Ultralytics library) is the
easiest detector to train, runs fast in a single pass, and ships **pretrained weights** so
a small dataset is enough.

### Architecture (what's inside)
```
image → BACKBONE (Conv + C3k2 blocks + SPPF + C2PSA attention) → features at 3 sizes
      → NECK (PAN-FPN: mixes fine detail with big-picture context)
      → HEAD (anchor-free: each grid cell predicts class + box via DFL)
      → decode → drop low confidence → NMS → final boxes
```
- **Convolutions** are filters that slide over the image detecting patterns (edges →
  shapes → "looks like a title block"). Stacked, they form the **backbone**.
- **3 feature sizes (P3/P4/P5)** let it catch tiny symbols *and* big blocks.
- **anchor-free head** = each cell predicts a box directly (no preset templates).
- **NMS** (Non-Max Suppression) removes duplicate overlapping boxes for the same object.

### How training works
1. **Forward pass:** push images through → predicted boxes.
2. **Loss:** measure error vs the true labels (class error + box error + DFL).
3. **Backpropagation:** compute how each weight affected the error.
4. **Optimizer step:** nudge weights to reduce error, by the *learning rate*.
5. One full pass over the train set = one **epoch**; do ~80.
6. **Transfer learning:** we start from `yolo11n.pt` (already trained on a huge generic
   dataset), so it knows edges/shapes already — we only *fine-tune* it on drawings. That's
   why a few hundred images suffice.

### Train it on Colab (free GPU)
`notebooks/train_yolo.ipynb` runs this:
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")              # pretrained nano model
model.train(data="data.yaml", epochs=80, imgsz=960, batch=16)
model.val()                             # prints mAP, saves metric plots
```
- `epochs` — how many passes (too few = dumb model; too many = memorizes).
- `imgsz` — input resolution (match the 960 you exported).
- `batch` — images per step (lower it if the GPU runs out of memory).

Download the resulting `best.pt` → put it in `models/best.pt` → the app comes alive.

### Reading the score
- **Precision** — of predicted boxes, how many were right (low = false alarms).
- **Recall** — of real objects, how many it found (low = misses).
- **mAP50** — the headline accuracy (higher better; ~0.5+ is a fine demo).
- **Confusion matrix** — which classes get mixed up.
Ultralytics saves these as images — screenshot them for the pitch (proof of real ML).

---

## 15. Chapter 14 — Run, test, debug

```bash
python scripts/check_setup.py          # doctor: lists what's installed / missing
python scripts/try_pretrained.py       # watch a generic YOLO detect (learning)
python scripts/run_cli.py drawing.png  # run our full pipeline in the terminal
./scripts/serve.sh                     # launch the web app
```

The CLI runner (`scripts/run_cli.py`) prints the checklist as a table and saves an
annotated image — handy for an offline backup demo (never trust venue wifi).

> 🐍 **`if __name__ == "__main__":`** At the bottom of scripts you'll see this. When you run
> a file directly, Python sets `__name__` to `"__main__"`, so the block runs. When the file
> is *imported* by another, `__name__` is the module name, so the block is skipped. It's how
> a file can be both a runnable script and an importable library.

> 🐍 **`sys.argv`** A list of the command-line words. `sys.argv[0]` is the script name;
> `sys.argv[1]` is the first argument you typed. That's how `run_cli.py drawing.png` knows
> which image to use.

---

## 16. Chapter 15 — Python idiom cheat-sheet

Everything "complex" used in this project, in one place:

| Idiom | Means | Example |
|---|---|---|
| `import x` / `from x import y` | load a library / one name from it | `from src.ocr import ocr_region` |
| `in` | "is a member of" | `if x in (1,2,3):` |
| `not in` | "is not a member of" | `if k not in d:` |
| `not` | flip a condition | `if not boxes:` (list is empty) |
| `and` / `or` | combine conditions | `if a and b:` / `if a or b:` |
| `is None` / `is not None` | compare to "nothing" | `if box is not None:` |
| `for x in items:` | loop over a collection | `for d in detections:` |
| `[f(x) for x in xs]` | list comprehension (build a list) | `[int(v) for v in box]` |
| `[x for x in xs if c]` | comprehension with filter | `[d for d in dets if d["cls"]=="note"]` |
| `sum(1 for x in xs if c)` | count matches | `sum(1 for r in rows if r["status"]=="PASS")` |
| `a if cond else b` | ternary (pick one value) | `"PASS" if n>=1 else "FAIL"` |
| `d.get(k, default)` | safe dict lookup | `counts.get(cls, 0)` |
| `a, b = x, y` | assign/unpack multiple | `x1,y1,x2,y2 = box` |
| `with open(p) as f:` | auto-closing resource | reading files |
| `try: ... except E:` | handle an error gracefully | optional libraries |
| f-string `f"{x}"` | embed values in text | `f"{n} found"` |
| `:.2f` in f-string | format a number | `f"{conf:.2f}"` |
| `@lru_cache` | remember a function's result | load model once |
| `lambda d: d["conf"]` | tiny inline function | `max(boxes, key=lambda d: d["conf"])` |
| `%` | remainder (modulo) | pick a palette color |
| `if __name__=="__main__":` | "run only when executed directly" | script entry point |

---

## 17. Chapter 16 — ML glossary

- **Object detection** — find each object's class + box (vs classification = one label).
- **Bounding box** — rectangle `(x1,y1,x2,y2)` around an object.
- **Class** — a category the model can detect (e.g. `dimension`).
- **Label / annotation** — the human-made answer key (class + box) for an image.
- **Confidence** — model's certainty for one box, 0–1.
- **Epoch** — one full pass over the training data.
- **Loss** — numeric error the training shrinks.
- **Transfer learning / fine-tuning** — start from a pretrained model, adapt to your data.
- **Inference** — using the trained model to predict on a new image.
- **mAP / precision / recall** — accuracy metrics.
- **IoU** — overlap between predicted and true box (used to decide a "correct" detection).
- **NMS** — removes duplicate overlapping boxes.
- **Overfitting** — model memorizes train data, fails on new data.
- **Weights (`.pt`)** — the learned numbers that *are* the trained model.
- **OCR** — image-of-text → text string.

---

## 18. Chapter 17 — Rebuild-from-scratch checklist

Do these in order and you've built the whole thing:

1. [ ] Install Python 3.11/3.12, make a venv, `git init`.
2. [ ] Write `requirements.txt`, `pip install -r requirements.txt`, `brew install tesseract`.
3. [ ] Create folders: `src/` (+ empty `__init__.py`), `scripts/`, `models/`, `data/`, `notebooks/`.
4. [ ] Build the unified dataset in Roboflow (Chapter 3); export YOLOv11; note the snippet.
5. [ ] Write `config.yaml` (classes, conf, rules, checklist).
6. [ ] Write `src/detect.py`, `ocr.py`, `rules.py`, `report.py`, `annotate.py`, `pipeline.py`.
7. [ ] Write `app.py`; run `streamlit run app.py` — works with NA rows (no model yet).
8. [ ] Train on Colab (`train_yolo.ipynb`); download `best.pt` → `models/`.
9. [ ] Re-run the app; verify boxes + checklist on a few drawings.
10. [ ] Prepare 3–4 demo drawings (PASS / FAIL / NA), screenshot metrics, rehearse the pitch.

If you can do these without re-reading, you understand the project. That's the goal.

---

# PART 3 — What we ACTUALLY built and trained (the real run)

Chapters 1–17 are the general method. This part is the **logbook of our actual run** —
the real data mess we hit, the real numbers, and how the trained model plugs into the
app. Read it to understand the project *as it really exists in this repo*.

---

## 19. Chapter 18 — The real dataset (635 raw classes → 4)

We merged three Roboflow datasets (`symbols`, `eng drawing`, `gdt symbols`) into one
project. Reality bit us:

- Roboflow's **Modify Classes** (to rename/merge labels) is a **paid** feature → we
  couldn't collapse classes in the UI.
- After Generate + export, the dataset had **`nc = 635` classes** — not the handful we
  expected. The merge kept every raw label:
  - **~255 pure numbers** (`100`, `-0.800`, `24.800`) = dimension *values*.
  - **~340 cryptic zone codes** (`KZ1`, `GBZ12`, `YBZ7`, `RFZ1`, `Lc-700`, single
    letters `A`–`H`, `column-*`) = annotations from a foreign dataset, useless to us.
  - **~40 real symbol names** (`Flatness`, `Perpendicularity`, `Roughness`, `Diameter`,
    `Notes`, `position`…).

**Free fix = remap the label files in code** (`scripts/remap_dataset.py`). Instead of a
635-line table, we classify each raw name by **rule**:

```python
_NUMBER = re.compile(r"-?\d+(\.\d+)?-?$")     # 100, -0.800, 1.3-
def classify(name):
    n = name.strip(); low = n.lower()
    if _NUMBER.match(n):           return "dimension"     # numbers are dimension values
    if low in _DIM_WORDS:          return "dimension"     # diameter, radius, chamfer...
    if low in _GDT_WORDS:          return "gdt_symbol"    # flatness, runout, position...
    if "roughness" in low:         return "surface_roughness"
    if low in ("note", "notes"):   return "note"
    return None                                           # drop zone codes / junk
```

The remap rewrites every YOLO label `.txt` (only the class-id on each line) and rewrites
`data.yaml`. Run once in Colab after download:

```python
remap(dataset.location)
```

> 🐍 **Python note — `re.compile(r"...")`** Pre-builds a regex pattern for reuse. The `r`
> prefix is a *raw string* so backslashes (`\d` = digit) aren't treated as escapes.
> `.match(s)` tests if the pattern matches at the **start** of `s`; the trailing `$`
> means "to the end" too. So `_NUMBER.match("100")` is truthy, `_NUMBER.match("KZ1")` is
> not.

**Real result of the remap on our data:**
```
kept 109811 boxes, dropped 48766
dimension 87744 | surface_roughness 13868 | gdt_symbol 7616 | note 583
```
No `title_block` existed (we hadn't hand-labeled it) → final model = **4 classes**.
`config.yaml` was set to match: `[dimension, note, gdt_symbol, surface_roughness]`.

**Lesson:** public/merged datasets are messy. The valuable skill is **cleaning labels in
code** — a small, rule-based script beats manual work and is reproducible.

---

## 20. Chapter 19 — The training run + reading the results

We trained YOLO11n on Colab's free T4 GPU. Split after generation was ~2.5k train / 223
val / 163 test; images at 960×960; gentle augmentation (rotate ±5°, brightness, blur —
**no flips**, text would mirror).

Command (Chapter 13 cell), `epochs=80, imgsz=960, batch=16, patience=20`. It
**early-stopped at epoch 74** (best at epoch 54) — `patience` saw no val improvement for
20 epochs and stopped. ~2 hours wall-clock.

### Final metrics (validation)
```
              mAP50   mAP50-95
all           0.725   0.460
dimension     0.927   0.686     # excellent (4298 instances)
surface_roughness 0.875 0.454   # strong   (168)
gdt_symbol    0.614   0.415     # decent   (132)
note          0.482   0.284     # weak — but only 17 val instances => noisy metric
```

### How to read our four result plots
- **Loss curves (train + val box/cls/dfl):** all slope **down** and the val curves
  *track* the train curves. That means **learning, no overfitting** (if val turned up
  while train kept dropping, that's overfitting).
- **mAP50 curve:** rises then **plateaus ~0.72** → more epochs wouldn't help much (why
  early-stop fired).
- **Confusion matrix:** strong diagonal (dimension 3873, roughness 149, gdt 78, note 10).
  The main off-diagonal is the **background** column/row — e.g. 605 real dimensions
  predicted as background (missed) and 417 background predicted as dimension (false
  alarms). Normal for dense, tiny dimension text on a big sheet. Symbols barely confuse
  with each other.
- **F1-confidence curve:** the bold "all classes" line peaks **F1 0.74 at confidence
  0.306**. That's the single most useful number for deployment →

### Why `conf_threshold: 0.30`
The F1 curve says detection quality is best around **0.30** confidence, so we set
`config.yaml: conf_threshold: 0.30`. Below it = too many false boxes; well above it = the
weak `note` class disappears. This is a **data-derived setting**, not a guess.

### Is a 5 MB model correct?
Yes. YOLO11n ("nano") has only **2.6M parameters**; the saved `best.pt` is ~5.3 MB. The
training log literally prints `Optimizer stripped from best.pt, 5.5MB`. Small = the model
size we chose, not a corrupted file. (You can verify it's a real checkpoint: it's a ZIP
archive containing `data.pkl` + weight tensors.)

---

## 21. Chapter 20 — Wiring the model + the demo app features

### Plugging in the model
`best.pt` goes in `models/` (the path in `config.yaml: model_path`). `src/detect.py`
lazy-loads it once (cached) and the whole pipeline lights up. `models/*.pt` is
**gitignored** (too big for git) — the model is rebuilt from Colab, not version-tracked.

Local setup that worked (note: **torch has no Python 3.14 wheels yet**, so we used 3.12):
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_setup.py     # all green
./scripts/serve.sh                # launch the app
```

### What `app.py` shows (demo-grade)
The pipeline returns `(annotated_image, checklist_rows, debug)` where `debug` now also
carries **per-class detection counts** (added in `src/pipeline.py` via `count_by_class`).
The app turns that into:
- a **verdict banner** — `PASS` (no fails), `NEEDS REVIEW` (some fail), or a "no
  detections, lower confidence" hint;
- **metrics** — checks passed + total detections + a progress bar;
- the **annotated drawing** (boxes) with a **Download PNG** button;
- the **checklist table** (Sl.No / Check point / Status / Remarks) with **Download CSV**;
- a **per-class bar chart** of what was detected;
- a confidence **slider** in the sidebar (re-runs detection live — great for the demo);
- a **Debug** expander (raw status + OCR text).

> 🐍 **Python note — `st.stop()`** Streamlit reruns the whole script top-to-bottom on
> every interaction. `if not uploaded: st.stop()` halts the run early when there's no
> file, so the rest doesn't execute. Cleaner than wrapping everything in one big `if`.

> 🐍 **Python note — `io.BytesIO()`** An in-memory binary "file." We save the annotated
> PIL image into it (`img.save(buf, format="PNG")`) and hand the bytes to Streamlit's
> download button — no temp file on disk needed.

### One Streamlit gotcha we fixed
`st.image(..., use_column_width=True)` is **deprecated/removed** in Streamlit 1.58. The
current arg is **`use_container_width=True`**. APIs drift between versions — when a widget
errors, check the installed version's docs.

---

## 22. Chapter 21 — Limitations & what's next

Honest state of the model (say this in the pitch — judges respect it):
- **Strong:** `dimension`, `surface_roughness`. **Decent:** `gdt_symbol`. **Weak:**
  `note` (only 583 train / 17 val boxes — too few to learn well).
- **No `title_block`** yet → the **drawing-number / date OCR check is dormant** (the two
  commented rows in `config.yaml`). The OCR + regex code is built and ready; it just needs
  a `title_block` class to crop.
- Public drawings ≠ company turbine drawings → this is a **proof of concept**;
  config-driven design lets the real checklist + a company-trained model drop in later.

**Highest-value next step:** hand-label `title_block` on ~30–50 sheets in Roboflow,
regenerate, retrain, then **uncomment the two title-block rows** in `config.yaml`. That
re-activates the OCR drawing-number check — the most "checklist-like" feature — with zero
code changes elsewhere.

Other future work: per-type GD&T classification, font/line-thickness checks (needs
vector-CAD parsing, not image ML), multi-sheet revision linking.

---

## 23. Chapter 22 — Deep dive: the YOLO11n model (architecture, parameters, math)

This chapter is the full technical answer to "what model is it, why, and how does it
work" — grounded in the **actual architecture printout from our training run**. Read it
to be able to defend every layer to a judge.

### 22.1 Why an object detector, why YOLO, why the "n" size

Our task: on one drawing, find **many** items and say **what** each is and **where**.
That is **object detection** (multiple boxes + labels per image), not:
- *classification* (one label for the whole image — can't localize), nor
- *segmentation* (pixel masks — more than we need; we want boxes).

Among detectors we chose **YOLO11** (Ultralytics, 2024/25) because:
- **single-shot** — one forward pass predicts all boxes → fast (we measured **6.6 ms /
  image** on a T4; runs fine on a laptop CPU for the demo).
- **anchor-free + modern** — simpler, accurate, current SOTA-class for its size.
- **one-line train/predict** + auto metrics/plots → ideal for a solo beginner.
- **pretrained weights** → transfer learning makes a few-thousand-image dataset enough.

Size variants trade accuracy vs speed/size: **n**(ano) < s < m < l < x. We used **`n`**
because it is smallest/fastest, trains quickly on free Colab, and still hit mAP50 0.725.
The whole model is **2.59 M parameters / ~5.3 MB**. (If accuracy mattered more than speed,
`yolo11s`/`m` add layers + parameters for a few more mAP points and slower inference.)

### 22.2 The three stages + our ACTUAL layer table

Every YOLO is **backbone → neck → head**. Here is the exact table YOLO11n printed for our
4-class model (read `from` = which previous layer feeds it, `params` = weights in that
layer, `arguments` = channel sizes):

```
       from  n    params  module        arguments                       stage
  0     -1   1       464  Conv          [3, 16, 3, 2]                  ┐
  1     -1   1      4672  Conv          [16, 32, 3, 2]                 │
  2     -1   1      6640  C3k2          [32, 64, 1, False, 0.25]       │
  3     -1   1     36992  Conv          [64, 64, 3, 2]                 │ BACKBONE
  4     -1   1     26080  C3k2          [64, 128, 1, False, 0.25]      │ (extract features,
  5     -1   1    147712  Conv          [128, 128, 3, 2]               │  shrink H×W,
  6     -1   1     87040  C3k2          [128, 128, 1, True]            │  grow channels)
  7     -1   1    295424  Conv          [128, 256, 3, 2]               │
  8     -1   1    346112  C3k2          [256, 256, 1, True]            │
  9     -1   1    164608  SPPF          [256, 256, 5]                  │
 10     -1   1    249728  C2PSA         [256, 256, 1]                  ┘
 11     -1   1         0  Upsample      [None, 2, 'nearest']           ┐
 12  [-1,6]  1         0  Concat        [1]                            │
 13     -1   1    111296  C3k2          [384, 128, 1, False]           │ NECK
 14     -1   1         0  Upsample      [None, 2, 'nearest']           │ (fuse fine detail
 15  [-1,4]  1         0  Concat        [1]                            │  with big-picture
 16     -1   1     32096  C3k2          [256, 64, 1, False]            │  context across
 17     -1   1     36992  Conv          [64, 64, 3, 2]                 │  3 scales —
 18 [-1,13]  1         0  Concat        [1]                            │  PAN-FPN)
 19     -1   1     86720  C3k2          [192, 128, 1, False]           │
 20     -1   1    147712  Conv          [128, 128, 3, 2]               │
 21 [-1,10]  1         0  Concat        [1]                            │
 22     -1   1    378880  C3k2          [384, 256, 1, True]            ┘
 23 [16,19,22] 1   431452  Detect       [4, 16, None, [64, 128, 256]]   HEAD
YOLO11n summary: 182 layers, 2,590,620 parameters, 6.4 GFLOPs
```

`from=-1` means "takes the previous layer." `from=[-1,6]` means "concatenate previous
layer with layer 6" — those skip-connections are what fuse scales. Layers with `params=0`
(Upsample, Concat) just reshape/merge — no learnable weights.

### 22.3 What each module does

- **Conv** `[in, out, k, s]` — a convolution (kernel `k`, stride `s`) + BatchNorm +
  SiLU activation. Stride 2 **halves** height/width (downsampling). This is the basic
  feature extractor: filters slide over the image learning edges → shapes → "looks like a
  roughness symbol."
- **C3k2** — YOLO11's main building block: a CSP (Cross-Stage-Partial) block that splits
  the channels, runs some through small conv bottlenecks, and concatenates. Cheap,
  gradient-friendly, the workhorse that does most of the "thinking." The `True/False` arg
  toggles a deeper variant; `0.25` is the bottleneck ratio.
- **SPPF (Spatial Pyramid Pooling – Fast)** — pools with several receptive sizes and
  merges them, so one feature map "sees" both tiny symbols and large blocks at once.
- **C2PSA** — YOLO11's lightweight **attention** block (Position-Sensitive Attention). It
  lets the network weight the informative regions more — a small accuracy boost for little
  cost.
- **Upsample + Concat** — the neck doubles a low-res map's size and concatenates it with a
  higher-res backbone map (and vice-versa). This **PAN-FPN** pattern mixes fine detail
  (good for small objects) with semantic context (good for knowing *what* it is).
- **Detect** — the output head (next section).

### 22.4 Multi-scale, anchor-free head, and DFL

The head reads **three feature maps** at different resolutions, called P3/P4/P5, with
**strides 8 / 16 / 32**. At our 960×960 input that's grids of **120×120, 60×60, 30×30**.
- P3 (120×120, fine) catches **small** objects — a tiny roughness symbol.
- P5 (30×30, coarse) catches **large** objects — a big note block.
This is why YOLO handles wildly different object sizes in one pass.

The head args `[4, 16, None, [64,128,256]]` mean:
- **4** = number of classes (our `nc`).
- **16** = `reg_max` for **DFL** (Distribution Focal Loss). Instead of directly
  regressing 4 raw box-edge distances, the head predicts, for each edge, a **probability
  distribution over 16 bins**, then takes its expected value. This sub-pixel,
  distribution-based regression is more accurate than predicting a single number.
- **[64,128,256]** = the channel widths feeding the head from P3/P4/P5.

**Anchor-free**: each grid cell predicts a box **directly** (no pre-defined anchor box
templates to tune). Per cell the head outputs `4×reg_max` box numbers (= 64) + `nc` class
scores. That 431,452-param Detect layer is the biggest single block — it carries the
classification + box-regression conv branches across all three scales.

### 22.5 Parameters & compute

- **2,590,620 parameters** total. Most live in the deeper backbone C3k2/Conv blocks
  (layers 7–10) and the Detect head (431k). Early layers are tiny (layer 0 = 464 params).
- **6.4 GFLOPs** is Ultralytics' standard measure **at 640×640**; we trained at **960**,
  so real compute is ~`(960/640)² ≈ 2.25×` higher (~14 GFLOPs/image). Bigger input = more
  compute but better small-symbol detection — a deliberate trade we made because our key
  classes (GD&T, roughness) are small.
- **File size ~5.3 MB** = these 2.59M float weights (FP32) zipped, after the optimizer
  state is stripped. Nano models are meant to be this small.

### 22.6 The loss function (what training minimizes)

YOLO11 minimizes a weighted sum of three losses (the weights are from our run's config):
- **box loss — CIoU** (weight **7.5**): how well the predicted box overlaps the true box
  (IoU-based, also accounts for center distance + aspect ratio).
- **cls loss — BCE** (weight **0.5**): wrong class (binary cross-entropy per class).
- **dfl loss** (weight **1.5**): sharpness/correctness of the DFL box-edge distributions.

`total = 7.5·box + 0.5·cls + 1.5·dfl`. These are the `box_loss / cls_loss / dfl_loss`
columns you watched drop each epoch.

### 22.7 Optimizer & key hyperparameters (from our log)

- **Optimizer: AdamW**, auto-selected (`optimizer=auto`), with **lr0 ≈ 0.00125**,
  **momentum 0.9**, **weight_decay 0.0005**. AdamW = Adam with decoupled weight decay
  (regularization that fights overfitting).
- **3 warm-up epochs** (lr ramps up slowly at first to stabilize early training).
- **imgsz 960, batch 16, epochs 80, patience 20** (early-stop). It stopped at epoch 74,
  best at 54.
- **AMP** (Automatic Mixed Precision) on — uses FP16 where safe to train faster + use less
  GPU memory.

> ML note — **learning rate** is the single most important knob: too high diverges, too
> low crawls. `optimizer=auto` picked a sensible one for us; you rarely need to hand-tune
> it for a demo.

### 22.8 Transfer learning (why ~5k images was enough)

The log shows **`Transferred 499/499 items from pretrained weights`** — we started from
`yolo11n.pt` (pretrained on the large COCO dataset). The backbone already knew generic
visual features (edges, textures, shapes), so we only had to **fine-tune** it to *our*
classes. Training from scratch would need far more data and time; transfer learning is the
reason a small drawing dataset trained to mAP50 0.725 in ~2 hours.

### 22.9 Inference path (what runs in the app)

`model.predict(image)`:
1. **Forward pass** → raw class scores + DFL box distributions at P3/P4/P5.
2. **Decode** distributions → pixel boxes `(x1,y1,x2,y2)`.
3. **Confidence filter** → drop boxes below `conf` (we use **0.30**, the F1-peak).
4. **NMS (Non-Max Suppression)** at **IoU 0.7** → remove duplicate overlapping boxes for
   the same object, keep the most confident.
5. Output = the clean detection list `src/detect.py` returns as `{cls, conf, xyxy}`.

No labels, no gradients — just one fast forward pass + cleanup.

### 22.10 Alternatives we rejected

| Model | Why not (for us) |
|---|---|
| Faster R-CNN | two-stage, accurate but slower + heavier to set up |
| RetinaNet / SSD | fine, but less tooling + community than Ultralytics YOLO |
| Plain CNN classifier | one label per image — can't localize multiple objects |
| Vision-LLM (GPT-4V) | no trained model to show judges; slower + cost per call |
| YOLO11 s/m/l/x | more accurate but slower + bigger; overkill for a demo on free GPU |

**Bottom line:** YOLO11n = the right balance of accuracy, speed, tiny size, and ease for
a solo hackathon build — and the numbers above prove it learned the task.

---

## 24. Chapter 23 — Improvements: value-checking, per-class thresholds, UI

After the base demo worked, we upgraded it from **"detects presence"** to **"verifies
values"** and made the UI demo-grade. All without retraining.

### 23.1 Value-checking: surface-roughness Ra (`src/values.py`)
The model gives us **where** each roughness symbol is. To check the *value*, we crop the
area around each box, OCR the number, and validate it against the allowed Ra set.

- `src/ocr.py → ocr_box_padded()` — expands the tiny symbol box (so the "Ra 0.8" text
  beside it is captured), grayscales + upscales the crop, and OCRs with a **digit
  whitelist** (`tessedit_char_whitelist=0123456789.`, `--psm 7` = single line).
- `src/values.py → read_ra_values()` — regex-extracts numbers, keeps a plausible one
  (`0 < v ≤ 100`), and matches it to `config.ra_allowed` within a tolerance.
- `src/report.py` — new check type **`ra_values`**: PASS if all readable Ra are in spec,
  FAIL if any aren't, INFO if symbols found but Ra unreadable, NA if no symbols.

This is the headline upgrade: row 3 now reports e.g. `PASS — Ra 0.8, 1.6 (all valid)`,
not just "present". OCR on tiny symbols is imperfect, so it degrades to "unreadable"
rather than lying.

> Why crop-then-OCR instead of OCRing the whole page? Same reason as the title block:
> a small, padded crop gives tesseract clean input. Whole-sheet OCR is noisy garbage.

### 23.2 Per-class confidence (`config.yaml: class_conf`, `src/detect.py`)
One global threshold is a compromise — `note` is weak (wants a low bar), `dimension` is
strong. Now `config.yaml` has:
```yaml
class_conf:
  note: 0.15
  gdt_symbol: 0.25
```
`detect()` predicts at the **lowest** needed threshold, then filters each box by its
class's own threshold (`class_conf.get(cls, conf_threshold)`). Weak classes surface;
strong classes stay clean.

### 23.3 Less clutter (`src/annotate.py`)
Drawings have hundreds of dimensions → boxes everywhere. Now dimensions draw as **thin,
label-less** rectangles; symbols (gdt/roughness/note) get **bold boxes + labels**. The
app sidebar also has **per-class show/hide checkboxes** (`pipeline.run(show_classes=...)`
filters *drawing only* — detection + checklist still use everything).

### 23.4 UI overhaul (`app.py`)
- **Sidebar**: confidence slider + per-class visibility toggles + model class list.
- **Verdict banner**: PASS / NEEDS REVIEW / "no detections" hint.
- **Metric row**: checks passed, total detections, confidence + progress bar.
- **Tabs**: Checklist · Detections (+ per-class bar chart) · Values (Ra table) · Debug.
- **Downloads**: CSV, **HTML report** (`src/report_html.py` — self-contained, base64
  image embedded, print-to-PDF in browser, zero dependency), annotated PNG.
- **Error handling**: the whole pipeline call is wrapped in `try/except` → a bad image
  shows a friendly message instead of crashing on stage.
- Fixed the Streamlit 1.58 deprecation (`use_container_width`).

> 🐍 **Python note — list-comprehension filter for the sidebar:**
> `show = [c for c in all_classes if st.checkbox(c, value=True)]` builds the list of
> ticked classes in one line — each `st.checkbox` returns True/False.

### 23.5 Still TODO (need labeling/retrain — your Roboflow work)
- **title_block + drawing-number/date OCR check** — hand-label `title_block`, retrain,
  uncomment the 2 rows in `config.yaml`. Code is already built and waiting.
- **Boost `note` / `gdt_symbol`** — more labels or a bigger model (`yolo11s`).
- **PDF/multi-page input, batch mode** — nice-to-have, not required for the demo.

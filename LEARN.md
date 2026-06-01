# LEARN.md — Understand the whole stack, then build it yourself

This is your study guide. Read it top to bottom once. It explains **what each
technology is, why we use it, and how it actually works** — then how they snap together
into this project. By the end you should be able to rebuild the system without copying.

The golden rule of this project: **the model only answers "what is in the drawing and
where." Everything else is plain rules.** Keep that split in your head — it makes the
whole thing simple.

---

## 0. The big picture in one paragraph

A drawing is just an image. We train a **model** to draw boxes around the important
parts (title block, dimensions, GD&T symbols, roughness symbols, notes). That gives us
*where things are*. We then **read the text** inside the title-block box with OCR, and
run **regex rules** to check formats (drawing number, date). A small **rule engine**
turns "what was found" into a checklist of PASS / FAIL / NA rows. A **Streamlit** web
page lets anyone upload a drawing and see the result. We **train on a free GPU** in
Google Colab because training needs heavy compute we don't have locally.

---

## 1. The tech stack (what each piece is and why it's here)

| Layer | Technology | Why this one |
|---|---|---|
| Language | **Python 3.11/3.12** | every ML library is Python-first |
| Isolation | **venv** (virtual environment) | keep this project's packages separate from your OS |
| The model | **YOLO11** via the **Ultralytics** library | easiest object detector to train; one line to train, one to predict |
| Data prep | **Roboflow** | merge several public datasets, relabel, augment, export — no code |
| Training compute | **Google Colab (free T4 GPU)** | training needs a GPU; Colab gives one free |
| Reading text | **Tesseract** + **pytesseract** | free OCR engine; pytesseract is its Python wrapper |
| Format checks | **regex** (Python `re`) | no ML needed to check "is this a valid date" |
| Glue rules | **plain Python** + **PyYAML** (`config.yaml`) | the checklist lives in config, not code |
| Drawing boxes | **Pillow (PIL)** | simple image drawing |
| Demo UI | **Streamlit** | turn a Python script into a web app in minutes |
| Version control | **git** | track your work |

You do **not** need to learn deep-learning math. You need to understand the *concepts*
below well enough to drive the tools and explain them to judges.

---

## 2. How each technology works

### 2.1 Object detection (the core ML idea)

**Problem it solves:** "find each object in an image and say what it is + where it is."

- **Classification** = "this whole image is a cat." One label per image.
- **Detection** = "there is a cat *here* (box) and a dog *there* (box)." Multiple boxes
  per image, each with a class label and a confidence score.

A detection output is a list of **bounding boxes**. One box =
`(class_name, confidence, x1, y1, x2, y2)` where the coordinates are pixel corners
(top-left, bottom-right). That's exactly what `src/detect.py` returns.

**Confidence** is the model's certainty, 0–1. We ignore boxes below a threshold
(`conf_threshold` in config) to cut false alarms.

### 2.2 YOLO — how the model learns and predicts

**YOLO** = "You Only Look Once." A neural network that looks at the whole image once and
predicts all boxes in a single pass — that's why it's fast.

How **training** works, conceptually:
1. You give it images **plus labels** (a human-drawn box + class for every object).
   Labels are the "answer key."
2. The network starts with random guesses, predicts boxes, and compares to the answer
   key. The gap is the **loss** (error).
3. **Backpropagation** nudges the network's millions of internal numbers (**weights**) a
   tiny bit to reduce the loss. Repeat over all images = one **epoch**.
4. Do many epochs (e.g. 80). Loss goes down, predictions get better.
5. Output = a **weights file** `best.pt`. That file *is* your trained model.

**Inference** (prediction) = feed a new image through the trained weights once → get
boxes. No labels needed. This is what runs in the app.

Key knobs you'll set (and should be able to explain):
- **epochs** — how many full passes over the data. Too few = underfit (model dumb), too
  many = overfit (memorizes training images, fails on new ones).
- **imgsz** — image resolution fed to the model (e.g. 960). Bigger = sees small symbols
  better, but slower.
- **batch** — how many images processed at once. Limited by GPU memory.
- **model size** — `yolo11n` (nano) is small/fast, good for a demo; `s/m/l` are bigger/slower/more accurate.

**Transfer learning** (why training is fast): `yolo11n.pt` is *pretrained* on millions of
generic images, so it already knows edges, shapes, text-like regions. You only
**fine-tune** it on your few hundred drawings. That's why a tiny dataset can still work.

### 2.3 How you measure if the model is good

You split data into **train / validation / test**:
- **train** — the model learns from these.
- **validation** — checked during training to watch for overfitting (model never learns
  from these directly).
- **test** — untouched until the end, your honest final score.

Metrics (memorize these — judges may ask):
- **Precision** — of the boxes the model predicted, how many were correct? (low = many
  false alarms.)
- **Recall** — of the real objects, how many did it find? (low = it misses things.)
- **mAP (mean Average Precision)** — the overall score combining both, averaged across
  classes. Higher is better; `mAP50` ~0.5+ is a fine demo result.
- **Confusion matrix** — a grid showing which classes get mixed up with which.

Ultralytics auto-generates these as `results.png` and `confusion_matrix.png` after
training — screenshot them for your slides; they prove "real ML."

### 2.4 Roboflow — data preparation without code

Training needs **labeled images**. Labeling from scratch is slow, so we reuse public
datasets. Roboflow is a web tool that lets you:
- **Clone** public datasets from Roboflow Universe into your project.
- **Merge** several datasets and **rename labels** to one shared class list (so 5
  datasets become one consistent set). This is how we get **one model from many sources**.
- **Augment** — auto-create variations (slight rotate, brightness, blur) so the model
  generalizes from few images. (For drawings, avoid flips — text would mirror.)
- **Export** in **YOLO format**: it produces image files + a `.txt` label per image plus
  a `data.yaml` listing the class names. Ultralytics reads that directly.

The YOLO label format (one line per object in a `.txt`):
`class_id  x_center  y_center  width  height` — all normalized 0–1. You won't hand-edit
these; Roboflow writes them. But know they exist.

### 2.5 Google Colab + GPU — why and how

Training does billions of multiplications. A **GPU** does many in parallel, ~10–100×
faster than a CPU. You may not have one. **Colab** is a free hosted Python notebook with
a free **T4 GPU** (Runtime → change runtime type → T4). You run the training cells there,
then download `best.pt`. Local machine only runs **inference**, which is light.

### 2.6 OCR + Tesseract — turning pixels into text

The model finds *where* the title block is, but not *what it says*. **OCR** (Optical
Character Recognition) converts an image of text into a text string. **Tesseract** is a
free OCR engine; **pytesseract** is the Python wrapper that calls it.

Trick we use: **only OCR the cropped title-block box**, not the whole noisy drawing.
Cleaner crop = better text. If Tesseract reads poorly on technical fonts, swap in
**PaddleOCR** (more accurate, heavier). See `src/ocr.py`.

### 2.7 Regex — the rule half

**Regex** (regular expressions) = a pattern language for text. Example: `\d{2}/\d{2}/\d{2}`
means "two digits, slash, two digits, slash, two digits" → matches a `DD/MM/YY` date.
We use regex to check formats the model can't judge: drawing-number pattern, date
format. No training, instant, 100% explainable. Patterns live in `config.yaml` so you
change rules without touching code. See `src/rules.py`.

### 2.8 The rule engine + config.yaml

The **rule engine** (`src/report.py`) is plain Python. It reads the `checklist:` section
of `config.yaml` and, for each item, decides PASS / FAIL / NA from either a detection
count or a regex result. Putting the checklist in **YAML** (a simple key-value text
format read by PyYAML) means the checklist is **data, not code** — you can retarget the
tool to the real company checklist by editing one file.

### 2.9 Streamlit — the demo

**Streamlit** turns a normal Python script into a web app. You write `st.title(...)`,
`st.file_uploader(...)`, `st.image(...)`, `st.table(...)` and it renders a page. No
HTML/JS. `streamlit run app.py` opens it in your browser. Perfect for a solo demo.

### 2.10 venv + pip + git (the basics)

- **pip** installs Python packages from `requirements.txt`.
- **venv** is an isolated package folder per project, so versions don't clash with your
  OS Python. `python3.11 -m venv .venv` then `source .venv/bin/activate`.
- **git** snapshots your code (`git add`, `git commit`). Already initialized here.

---

## 3. How it all connects (data flow, end to end)

```
                                  config.yaml  (classes, rules, checklist)
                                        │  read by everything
   drawing.png                          ▼
       │
       ├──────────────►  src/detect.py  ──► [boxes + classes + conf]
       │                 (loads models/best.pt via Ultralytics)
       │                          │
       │                          ├──► src/report.py ──┐
       │                          │   (counts per class │
       │                          │    → PASS/FAIL/NA)   │
       │                          ▼                      ├──► checklist rows
       └──► crop title_block ─► src/ocr.py ─► text ─► src/rules.py (regex) ┘
                                                                  │
   src/annotate.py ◄── boxes                                      │
       │  (draw boxes on image)                                   │
       ▼                                                          ▼
   annotated image  ────────────►  app.py (Streamlit)  ◄──── checklist table
```

`src/pipeline.py` is the conductor: it calls detect → ocr → rules → report → annotate in
order and hands the results to `app.py`.

---

## 4. Do-it-yourself roadmap (with what to learn at each step)

Follow the 10-day plan, but here's the learning attached to each phase:

1. **Python refresh** (½ day) — functions, lists, dicts, importing modules. You already
   have enough; skim if rusty.
2. **Object detection concept** (½ day) — watch one "what is YOLO" video; read the
   Ultralytics "Quickstart" + "Train" docs. Run `yolo predict model=yolo11n.pt
   source='https://ultralytics.com/images/bus.jpg'` to see detection work.
3. **Roboflow** (½ day) — make the unified dataset (Section 2.4 + README steps).
4. **Train** (1 day) — run the Colab cells in the README. Watch the loss fall. Read your
   `results.png`. Understand epochs/batch by changing them once and seeing the effect.
5. **Inference + report** (1 day) — drop `best.pt` in, run the app, read `src/detect.py`
   and `src/report.py` line by line until you can explain every line.
6. **OCR + regex** (1 day) — install Tesseract, test `src/ocr.py` on a cropped title
   block, tweak a regex in `config.yaml` and watch a checklist row flip.
7. **Demo polish + pitch** (rest) — see plan.

**Test of understanding:** if you can answer these, you're ready to present —
- What is the difference between classification and detection?
- What does `best.pt` contain and how was it produced?
- Why do we train on Colab and not locally?
- What is mAP, and what does low recall mean?
- Which checklist items come from the model vs from regex, and why split them?

---

## 5. Glossary (quick reference)

- **Annotation / label** — human-drawn box + class used as the training answer key.
- **Bounding box** — rectangle `(x1,y1,x2,y2)` locating an object.
- **Class** — a category the model can detect (e.g. `dimension`).
- **Confidence** — model's certainty for one box, 0–1.
- **Epoch** — one full pass over the training data.
- **Fine-tune / transfer learning** — continue training a pretrained model on your data.
- **Inference** — using a trained model to predict on new images.
- **Loss** — numeric error the training tries to minimize.
- **mAP / precision / recall** — accuracy metrics (Section 2.3).
- **Overfitting** — model memorizes training data, fails on new data.
- **OCR** — image-of-text → text string.
- **Weights (`.pt`)** — the learned numbers that define the trained model.
- **YOLO** — single-pass real-time object detector.

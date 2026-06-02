# Automated Engineering-Drawing Checklist

> Upload a mechanical drawing → an AI model finds the important parts, reads the title
> block, and **fills a QC checklist automatically** (PASS / FAIL / NA). Hackathon project
> that turns slow, manual drawing review into a 3-second upload.

**New here / want to build it yourself?** Read **[`BUILD_GUIDE.md`](BUILD_GUIDE.md)** — a
from-scratch tutorial that takes you from an empty folder to the finished project,
explaining every tool, library, and line of Python along the way.

---

## What it does

Mechanical engineering drawings are QC'd by hand against a long checklist (title block,
drawing-number format, surface-roughness symbols, GD&T symbols, dimensions, revision
table…). That's slow and error-prone. This tool automates it.

The core idea — every check is one of two questions, answered by two tools:

| Question | Tool |
|---|---|
| **"What is on the drawing, and where?"** | a trained **YOLO** object-detection model |
| **"Is the text / format correct?"** | plain **regex** rules |

> Model = "what + where." Rules = "is it right."

We can't train on confidential company drawings, so we train on **free public datasets**
and make the checklist **config-driven** — the real company checklist + a company-trained
model drop in later with zero code changes.

## How it works

```
upload drawing → YOLO model → boxes + classes ─┐
              → crop title block → OCR → text ──┤→ rule engine → checklist report (+ annotated image)
                       config.yaml ─────────────┘
```

## Quickstart

```bash
git clone https://github.com/jay112298/hackathon.git
cd hackathon
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_setup.py     # shows what's ready / missing
./scripts/serve.sh                # launch the app (rows show NA until models/best.pt exists)
```

The app runs **even without a trained model** — checklist rows show `NA` until you add
`models/best.pt` (train it via `notebooks/train_yolo.ipynb`).

## Repo map

```
config.yaml            # the brain: classes, regex rules, the checklist definition
app.py                 # Streamlit demo UI
src/
  pipeline.py          # orchestrates detect → ocr → rules → report → annotate
  detect.py            # YOLO inference → [{cls, conf, xyxy}]
  ocr.py               # Tesseract on the title-block crop
  rules.py             # regex format checks
  report.py            # build PASS/FAIL/NA rows from config
  annotate.py          # draw boxes (Pillow)
scripts/
  try_pretrained.py    # see detection work before training
  run_cli.py           # run the pipeline from the terminal
  check_setup.py       # doctor: what's installed/missing
  serve.sh             # launch the app
notebooks/
  train_yolo.ipynb     # Colab training
docs/                  # tomorrow steps, pitch script, dataset request email
BUILD_GUIDE.md         # full from-scratch tutorial + learning
```

## Status

Pipeline, GUI, training notebook, and docs are done; runs end-to-end (model-optional).
Next: build the unified dataset, train on Colab, drop in `best.pt`, collect demo drawings.

**Future work:** per-type GD&T classification, font/line-thickness checks, native CAD/DWG
parsing, multi-sheet revision linking.

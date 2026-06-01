# Pitch outline — 5 slides (Day 10)

Keep it 5 minutes. Show, don't tell. Live demo is the star.

---

## Slide 1 — Problem

- Engineering drawings are QC'd **by hand** against a long checklist (show the company
  PDF: title block, dimensions, GD&T, roughness, notes, revisions).
- Slow, tedious, error-prone, depends on the reviewer's attention.
- One missed symbol → scrap, rework, delay.

**One line:** "Manual drawing QC is slow and inconsistent. We automate it."

---

## Slide 2 — Idea / approach

- Upload a drawing → the system **detects** the parts, **reads** the title block, and
  **auto-fills the checklist** (PASS / FAIL / NA).
- Key split: a trained model answers **"what + where"**; plain rules check **formats**.
- Diagram: image → YOLO → boxes + OCR → rule engine → checklist report.

**One line:** "One model finds the parts; rules judge them."

---

## Slide 3 — Data + model (proves it's real ML)

- Trained **one YOLO11 model** on a **unified public dataset** (merged from open
  engineering-drawing datasets; couldn't use company drawings — privacy).
- Classes: title_block, revision_table, dimension, note, gdt_symbol, surface_roughness.
- **Show the metrics screenshot** (results.png / mAP / confusion matrix) — proof of
  training, not a hardcoded demo.

**One line:** "Real trained detector — here are the numbers."

---

## Slide 4 — LIVE DEMO (the moment)

- Upload a drawing → boxes appear → checklist fills → score badge → download CSV.
- Move the **confidence slider** to show live detection.
- Upload a **bad drawing** (missing title block) → watch it flag FAIL/NA. Shows it
  actually reasons, not a canned result.
- **Run offline from local files** — never trust venue wifi.

**One line:** "Watch it check a drawing in 3 seconds."

---

## Slide 5 — Impact + roadmap

- Impact: faster QC, consistent, audit trail (downloadable report).
- Built so it **retargets with zero code change**: edit `config.yaml` to load the real
  company checklist + a model trained on company drawings.
- Roadmap: per-type GD&T, font/line-thickness checks, native CAD/DWG parsing,
  multi-sheet linking.

**One line:** "Today: public drawings. Tomorrow: drop in company data, same engine."

---

## Q&A prep (likely judge questions)

- *Why public data?* Can't use confidential company drawings to train; proof-of-concept.
- *Accuracy?* Quote your mAP; note small dataset, demo not production.
- *What's ML vs rules?* Detection = ML; format checks = regex (explainable, no training).
- *Scale to real checklist?* Config-driven; rules + classes are data, not code.
- *One model or many?* One — merged datasets into one label set.

# Tomorrow — training day (do these in order, brain off)

You're rested. Just follow the list. Each step has the exact command.

## 0. Activate env + sanity check (2 min)

```bash
cd /Users/jitu/code/hackathon
python3.11 -m venv .venv          # only if .venv doesn't exist yet
source .venv/bin/activate
pip install -r requirements.txt   # first time only
python scripts/check_setup.py     # everything OK except models/best.pt
```

## 1. Build the dataset (if not done) — Roboflow (~30 min)

Follow `BUILD_GUIDE.md` Chapter 3. End result: a YOLOv11 **export snippet** + `data.yaml` whose
`names:` match `config.yaml` order. Compare against `docs/data_yaml_reference.yaml`.

## 2. Train — Google Colab (~1 hr, mostly waiting)

1. Upload `notebooks/train_yolo.ipynb` to https://colab.research.google.com
2. Runtime → Change runtime type → **T4 GPU**.
3. Cell 2: paste your Roboflow export snippet.
4. Run all cells. Wait for training.
5. Cell 4: screenshot `results.png` + `confusion_matrix.png` (for the pitch).
6. Cell 6: download `best.pt`.

## 3. Go live (5 min)

```bash
mv ~/Downloads/best.pt models/best.pt
python scripts/check_setup.py            # models/best.pt now OK
python scripts/run_cli.py path/to/a/test_drawing.png   # quick terminal check
./scripts/serve.sh                        # or: streamlit run app.py
```

Upload a drawing in the browser → boxes + checklist should appear.

## 4. If a class detects badly

- Lower the confidence slider in the app to see if it's just threshold.
- If a class (often `gdt_symbol` / `surface_roughness` from crop data) is hopeless:
  delete its line from `config.yaml: checklist` and `classes` — the app adapts, no code
  change. Dataset-driven = honest demo.

## Done = working model + live demo. Days 5–10 are polish + pitch.

Don't forget (whenever): send the email in `docs/email_1367_dataset.md`.

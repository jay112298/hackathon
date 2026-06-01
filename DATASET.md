# DATASET.md — Build the ONE unified dataset (Day 2)

Goal: turn several public Roboflow datasets into **one** dataset with **one** shared
class list, exported in YOLO format, ready to train **one** model.

Final class list (must stay in this exact order everywhere):

```
0 title_block
1 revision_table
2 dimension
3 note
4 gdt_symbol
5 surface_roughness
```

> Keep `config.yaml: classes` in this same order — the rule engine maps class index → name by it.

---

## Step 0 — account + project (5 min)

1. Sign up free at https://roboflow.com (use your Google login).
2. **Create New Project** → type **Object Detection** → name it `drawing-checklist`.
3. Note your **API key**: Settings → Roboflow API → Private API Key (needed in Colab).

---

## Step 1 — source datasets to clone

Open each on Roboflow Universe, hit **Download/Clone → Clone into your workspace**, pick
`drawing-checklist`. (If a link 404s, search Universe for the dataset name — community
datasets move.)

| # | Dataset | URL | What we keep |
|---|---|---|---|
| 1 | eng-drawing | https://universe.roboflow.com/engineering-drawing-qrlfu/eng-drawing-ukrvj | `dimension`, `note`/`Notes` |
| 2 | Eng_Drawing v7 | https://universe.roboflow.com/trial-yolo/eng_drawing-idyau | `title_block`, `revision_table` |
| 3 | vanigaa engineering-drawing-datasets | https://universe.roboflow.com/vanigaa/engineering-drawing-datasets | any full-sheet boxes |
| 4 | roughness-3d | https://universe.roboflow.com/database/roughness-3d | `surface_roughness` |
| 5 | gdt-symbols (43 classes) | https://universe.roboflow.com/gdt-a6ill/gdt-symbols | all 43 → `gdt_symbol` |

**Priority rule:** datasets 1–3 are full sheets → the backbone. Datasets 4–5 are often
*crops*; add them, but if they drag accuracy down (see Day-3 metrics) drop that class.

---

## Step 2 — remap every source label to the shared 6 classes

In Roboflow, project → **Classes** (or the **Annotate → Modify Classes / Remap** tool).
Map each source's raw label names to our 6. Anything irrelevant → **delete/ignore**.

| Source raw label (examples) | Map to |
|---|---|
| `title_block`, `titleblock`, `title block` | `title_block` |
| `revision_table`, `rev_table`, `revision` | `revision_table` |
| `dimension`, `dim`, `measure` | `dimension` |
| `note`, `Notes`, `notes_block` | `note` |
| all 43 GD&T types (`flatness`, `runout`, `position`, `datum`, `Ø`, …) | `gdt_symbol` |
| `roughness`, `Ra`, `surface_finish`, `surface roughness` | `surface_roughness` |
| anything else (borders, arrows, junk) | delete |

This **collapse** is the key beginner move: 43 sparse GD&T classes would never train on
small data; one `gdt_symbol` class will.

---

## Step 3 — generate a version (augment + split)

1. **Preprocessing:** Auto-Orient ON. Resize → **Stretch to 960×960** (matches `imgsz=960`).
2. **Train/val/test split:** 70 / 20 / 10 (Roboflow default is fine).
3. **Augmentations** (drawings are line art — be gentle, NO flips, text would mirror):
   - Rotation: ±5°
   - Brightness: ±15%
   - Blur: up to 1px
   - (skip flip, crop, mosaic, hue)
   Target ~2–3× the image count.
4. Click **Generate**.

---

## Step 4 — export

**Versions → Export Dataset → Format: YOLOv11 → "show download code"**. Copy the Python
snippet (has your workspace/project/version + API key). You'll paste it into Colab on
Day 3. Example shape:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="XXXX")
ds = rf.workspace("your-ws").project("drawing-checklist").version(1).download("yolov11")
# -> creates a folder with train/ valid/ test/ and data.yaml
```

Open the produced `data.yaml`. Confirm `names:` is exactly:
`['title_block','revision_table','dimension','note','gdt_symbol','surface_roughness']`
in that order. If not, fix the order in `config.yaml` to match `data.yaml`.

---

## Done = Day-2 dataset milestone

You now have one unified, augmented, YOLO-format dataset. Day 3 = train on it in Colab.

**Checklist before moving on:**
- [ ] 5 datasets cloned into one project
- [ ] all labels remapped to the 6 classes, junk deleted
- [ ] version generated with 960px + gentle augments
- [ ] YOLOv11 export snippet copied
- [ ] `data.yaml` class order matches `config.yaml`

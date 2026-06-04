"""Remap a Roboflow YOLO dataset's classes down to our final set — for free.

Roboflow's "Modify Classes" is a paid feature. Instead, export the dataset as-is
(with all its raw classes) and run this to rewrite the label files:
  - every box's class id is remapped to our small NEW_ORDER set, or dropped.
  - data.yaml's names/nc are rewritten to match.

Usage (locally or in Colab):
    # edit MAPPING + NEW_ORDER below, then:
    python scripts/remap_dataset.py /path/to/exported/dataset
    # or in Colab:  remap("/content/drawing-checklist-1")

The label format is YOLO: each line "class_id x y w h". We only change class_id.
"""
import glob
import os
import sys

import yaml

# old_name -> new_name. Fill from your dataset's data.yaml names. None / missing = drop.
MAPPING = {
    # "Roughness": "surface_roughness",
    # "Flatness": "gdt_symbol",
    # "Diameter": "dimension",
    # "Notes": "note",
    # "title_block": "title_block",
}

# The final class list (and order). Must match config.yaml.
NEW_ORDER = ["title_block", "dimension", "note", "gdt_symbol", "surface_roughness"]


def remap(dataset_dir):
    yml = os.path.join(dataset_dir, "data.yaml")
    cfg = yaml.safe_load(open(yml))
    old_names = cfg["names"]
    new_id = {n: i for i, n in enumerate(NEW_ORDER)}

    # old class id -> new class id (or None to drop)
    old_to_new = {}
    for oid, oname in enumerate(old_names):
        tgt = MAPPING.get(oname)
        old_to_new[oid] = new_id.get(tgt) if tgt else None

    changed, dropped = 0, 0
    for split in ["train", "valid", "test"]:
        for lf in glob.glob(os.path.join(dataset_dir, split, "labels", "*.txt")):
            keep = []
            for line in open(lf):
                p = line.split()
                if not p:
                    continue
                nid = old_to_new.get(int(p[0]))
                if nid is None:
                    dropped += 1
                    continue
                p[0] = str(nid)
                keep.append(" ".join(p))
                changed += 1
            with open(lf, "w") as f:
                f.write("\n".join(keep) + ("\n" if keep else ""))

    cfg["names"] = NEW_ORDER
    cfg["nc"] = len(NEW_ORDER)
    with open(yml, "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Remapped {changed} boxes, dropped {dropped}. Classes -> {NEW_ORDER}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/remap_dataset.py /path/to/dataset")
        sys.exit(1)
    remap(sys.argv[1])

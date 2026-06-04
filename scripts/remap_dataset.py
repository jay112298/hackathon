"""Remap a messy Roboflow YOLO dataset down to our final classes — for free.

Roboflow's "Modify Classes" is paid. Instead, export the dataset with all its raw
classes and run this to rewrite the label files + data.yaml.

Our merged export had 635 raw classes:
  - ~255 pure numbers (dimension values)        -> dimension
  - ~40 real symbol names (Flatness, Roughness) -> gdt_symbol / surface_roughness / dimension / note
  - ~340 cryptic zone codes (KZ*, GBZ*, YBZ*, RFZ*, Lc-*, single letters, column-*) -> dropped

We classify by RULE (not a 635-line table) so it's robust and reusable.

Usage:
    python scripts/remap_dataset.py /path/to/exported/dataset
    # or in Colab:  remap("/content/Drawing-Checklist-1")
"""
import glob
import os
import re
import sys

import yaml

# Final class list (and order). MUST match config.yaml.
NEW_ORDER = ["dimension", "note", "gdt_symbol", "surface_roughness"]

_DIM_WORDS = {
    "dimension", "length", "diameter", "radius", "chamfer", "angle", "degree",
    "depth", "counterbore", "countersink", "thread", "thread-unc-",
}
_GDT_WORDS = {
    "flatness", "perpendicularity", "parallelism", "parallelity", "concentricity",
    "centrality", "cylindricity", "runout", "total runout", "circ runout", "position",
    "true position", "profile", "angularity", "slope", "line", "surface", "tangent",
    "target", "through", "between", "mat_max",
}
_NUMBER = re.compile(r"-?\d+(\.\d+)?-?$")   # 100, -0.800, 1.3-, 105-, 24.800


def classify(name):
    """Map one raw class name to a final class, or None to drop it."""
    n = name.strip()
    low = n.lower()
    if _NUMBER.match(n):
        return "dimension"
    if low in _DIM_WORDS:
        return "dimension"
    if low in _GDT_WORDS:
        return "gdt_symbol"
    if "roughness" in low:
        return "surface_roughness"
    if low in ("note", "notes"):
        return "note"
    return None   # zone codes, datum letters, column-*, etc. -> dropped


def remap(dataset_dir):
    yml = os.path.join(dataset_dir, "data.yaml")
    cfg = yaml.safe_load(open(yml))
    old_names = cfg["names"]
    new_id = {c: i for i, c in enumerate(NEW_ORDER)}

    old_to_new = {}
    for oid, oname in enumerate(old_names):
        tgt = classify(oname)
        old_to_new[oid] = new_id.get(tgt) if tgt else None

    kept, dropped = 0, 0
    per_class = {c: 0 for c in NEW_ORDER}
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
                per_class[NEW_ORDER[nid]] += 1
                kept += 1
            with open(lf, "w") as f:
                f.write("\n".join(keep) + ("\n" if keep else ""))

    cfg["names"] = NEW_ORDER
    cfg["nc"] = len(NEW_ORDER)
    with open(yml, "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Kept {kept} boxes, dropped {dropped}.")
    print("Per-class box counts:", per_class)
    print("Classes ->", NEW_ORDER)
    print("\nWARNING: any class with very few boxes will train badly.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/remap_dataset.py /path/to/dataset")
        sys.exit(1)
    remap(sys.argv[1])

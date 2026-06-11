"""Batch-evaluate the pipeline over a folder of drawings.

Runs detect + Ra OCR on every image, writes a CSV of per-image results, and
ranks images by "demo quality" (roughness symbols that actually read + a good
class mix) so you can pick the best sheets for data/demo/.

Usage:
  python scripts/batch_eval.py data/roboflow/drawing-checklist-v1/test/images \
      --out batch_results.csv --copy-top 8

Skips the duplicate-dimension full-sheet OCR pass for speed (that check is
exercised in the app itself).
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import analyze, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="batch_results.csv")
    ap.add_argument("--copy-top", type=int, default=0,
                    help="copy the N best demo candidates into data/demo/")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    imgs = sorted(sum((glob.glob(os.path.join(args.folder, e))
                       for e in ("*.jpg", "*.jpeg", "*.png")), []))
    if args.limit:
        imgs = imgs[:args.limit]
    if not imgs:
        sys.exit(f"no images in {args.folder}")

    cfg = load_config()
    # skip the slow full-sheet pass: batch ranking only needs Ra + counts
    cfg["checklist"] = [i for i in cfg["checklist"] if i.get("type") != "duplicate_dims"]

    results = []
    for n, path in enumerate(imgs, 1):
        try:
            _, _, dbg = analyze(path, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[{n}/{len(imgs)}] {os.path.basename(path)}  ERROR {e}", flush=True)
            continue
        c = dbg["counts"]
        ra = dbg["ra_readings"]
        ra_read = [r for r in ra if r["value"] is not None]
        ra_match = [r for r in ra if r["valid"]]
        row = {
            "image": os.path.basename(path),
            "detections": dbg["n_detections"],
            "dimension": c.get("dimension", 0),
            "note": c.get("note", 0),
            "gdt_symbol": c.get("gdt_symbol", 0),
            "surface_roughness": c.get("surface_roughness", 0),
            "ra_read": len(ra_read),
            "ra_matched": len(ra_match),
            "ra_values": " ".join((r["grade"] or str(r["value"])) for r in ra_read),
            "path": path,
        }
        # demo quality: readable+matched Ra is gold, then class variety
        row["score"] = (10 * row["ra_matched"]
                        + 2 * min(row["surface_roughness"], 3)
                        + 3 * (row["gdt_symbol"] > 0)
                        + 2 * (row["note"] > 0)
                        + (5 <= row["dimension"] <= 120))
        results.append(row)
        print(f"[{n}/{len(imgs)}] {row['image'][:48]:50} "
              f"rough={row['surface_roughness']} read={row['ra_read']} "
              f"matched={row['ra_matched']} vals=[{row['ra_values']}] "
              f"score={row['score']}", flush=True)

    fields = ["image", "score", "detections", "dimension", "note", "gdt_symbol",
              "surface_roughness", "ra_read", "ra_matched", "ra_values", "path"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {args.out} ({len(results)} images)")

    best = sorted(results, key=lambda r: -r["score"])
    print("\n=== top demo candidates ===")
    for r in best[:15]:
        print(f"score={r['score']:3}  rough={r['surface_roughness']} "
              f"matched={r['ra_matched']}/{r['ra_read']}  [{r['ra_values']}]  {r['image']}")

    if args.copy_top:
        os.makedirs("data/demo", exist_ok=True)
        for i, r in enumerate(best[:args.copy_top], 1):
            ext = os.path.splitext(r["path"])[1]
            shutil.copy(r["path"], f"data/demo/sample_{i:02d}{ext}")
        print(f"\ncopied top {args.copy_top} -> data/demo/")


if __name__ == "__main__":
    main()

"""Run the checklist pipeline on one image from the terminal (no Streamlit).

Useful for quick testing and for an offline backup demo.

Run:
    python scripts/run_cli.py path/to/drawing.png
    python scripts/run_cli.py path/to/drawing.png --out annotated.png --conf 0.3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to a drawing image")
    ap.add_argument("--out", default="annotated.png", help="where to save annotated image")
    ap.add_argument("--conf", type=float, default=None, help="confidence threshold override")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"No such image: {args.image}")
        sys.exit(1)

    config = load_config()
    if args.conf is not None:
        config["conf_threshold"] = args.conf

    annotated, rows, debug = run(args.image, config)
    annotated.save(args.out)

    # checklist table
    print(f"\n{'Sl':<3}{'Check point':<40}{'Status':<8}Remarks")
    print("-" * 80)
    for r in rows:
        print(f"{r['id']:<3}{r['point']:<40}{r['status']:<8}{r['remarks']}")

    passed = sum(1 for r in rows if r["status"] == "PASS")
    failed = sum(1 for r in rows if r["status"] == "FAIL")
    print("-" * 80)
    print(f"Score: {passed}/{passed + failed} passed | annotated -> {args.out}")
    print(f"Debug: {debug}")


if __name__ == "__main__":
    main()

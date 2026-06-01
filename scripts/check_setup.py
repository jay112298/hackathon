"""Doctor: is everything ready to run/train? Run this first tomorrow.

    python scripts/check_setup.py

Prints a checklist of deps, the tesseract binary, the model file, and config sanity.
Green-ish = ready. Tells you exactly what's missing.
"""
import importlib.util
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has(mod):
    return importlib.util.find_spec(mod) is not None


def line(ok, label, hint=""):
    mark = "OK " if ok else "-- "
    tail = "" if ok else f"   -> {hint}"
    print(f"[{mark}] {label}{tail}")


def main():
    print("=== Setup check ===\n")

    # python packages
    pkgs = {
        "yaml": "pip install pyyaml",
        "PIL": "pip install pillow",
        "streamlit": "pip install streamlit",
        "ultralytics": "pip install ultralytics",
        "pytesseract": "pip install pytesseract",
    }
    for mod, hint in pkgs.items():
        line(_has(mod), f"python pkg: {mod}", hint)

    # tesseract binary
    line(shutil.which("tesseract") is not None, "tesseract binary",
         "brew install tesseract  (macOS)")

    # model file
    model_ok = os.path.exists(os.path.join("models", "best.pt"))
    line(model_ok, "models/best.pt (trained model)",
         "train in Colab (notebooks/train_yolo.ipynb), then drop best.pt here")

    # config sanity
    try:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        n_cls = len(cfg.get("classes", []))
        n_items = len(cfg.get("checklist", []))
        line(n_cls > 0 and n_items > 0,
             f"config.yaml ({n_cls} classes, {n_items} checklist items)")
    except Exception as e:  # noqa: BLE001
        line(False, "config.yaml", f"error: {e}")

    print("\nApp runs even with missing items (shows NA rows).")
    print("To go LIVE you need: ultralytics + models/best.pt (+ tesseract for OCR rows).")


if __name__ == "__main__":
    main()

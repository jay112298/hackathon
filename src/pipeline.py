"""End-to-end pipeline: image path -> (annotated image, checklist rows, debug).

This is the single entry point the app (or a CLI) calls.
"""
from __future__ import annotations
import os
import yaml

from .detect import detect, first_box
from .ocr import ocr_region
from .report import build_report
from .annotate import annotate

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config(path: str = _CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(image_path: str, config: dict | None = None):
    if config is None:
        config = load_config()

    model_path = config["model_path"]
    conf = config.get("conf_threshold", 0.25)

    # 1) detect
    detections, det_status = detect(image_path, model_path, conf)
    model_loaded = "No model" not in det_status and "not installed" not in det_status

    # 2) OCR the title-block region (if detected)
    tb_box = first_box(detections, "title_block")
    ocr_text, ocr_status = ocr_region(image_path, tb_box) if tb_box else ("", "no title_block detected")

    # 3) build checklist
    rows = build_report(config, detections, ocr_text, model_loaded)

    # 4) annotate
    annotated = annotate(image_path, detections)

    debug = {
        "detection_status": det_status,
        "ocr_status": ocr_status,
        "ocr_text": ocr_text.strip(),
        "n_detections": len(detections),
    }
    return annotated, rows, debug

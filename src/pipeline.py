"""End-to-end pipeline: image path -> (annotated image, checklist rows, debug).

This is the single entry point the app (or a CLI) calls.
"""
from __future__ import annotations
import os
import yaml

from .detect import detect, first_box, count_by_class
from .ocr import ocr_region
from .report import build_report
from .annotate import annotate
from .values import read_ra_values

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config(path: str = _CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(image_path: str, config: dict | None = None, show_classes=None):
    """Run the full checklist pipeline on one image.

    show_classes: optional iterable of class names to DRAW (annotation only).
                  Detection/checklist always use all classes.
    """
    if config is None:
        config = load_config()

    model_path = config["model_path"]
    conf = config.get("conf_threshold", 0.25)
    class_conf = config.get("class_conf", {})

    # 1) detect (with optional per-class confidence overrides)
    detections, det_status = detect(image_path, model_path, conf, class_conf)
    model_loaded = "No model" not in det_status and "not installed" not in det_status

    # 2) OCR the title-block region (if a title_block was detected)
    tb_box = first_box(detections, "title_block")
    ocr_text, ocr_status = ocr_region(image_path, tb_box) if tb_box else ("", "no title_block detected")

    # 3) value-level checks: read Ra values off roughness symbols
    rough_boxes = [d["xyxy"] for d in detections if d["cls"] == "surface_roughness"]
    ra_readings = read_ra_values(image_path, rough_boxes, config.get("ra_allowed", [])) \
        if rough_boxes else []

    # 4) build checklist
    rows = build_report(config, detections, ocr_text, model_loaded, ra_readings)

    # 5) annotate (optionally only some classes, to reduce clutter)
    draw = detections
    if show_classes is not None:
        draw = [d for d in detections if d["cls"] in show_classes]
    annotated = annotate(image_path, draw)

    debug = {
        "detection_status": det_status,
        "ocr_status": ocr_status,
        "ocr_text": ocr_text.strip(),
        "n_detections": len(detections),
        "counts": count_by_class(detections),
        "ra_readings": ra_readings,
    }
    return annotated, rows, debug

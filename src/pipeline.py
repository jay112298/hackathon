"""End-to-end pipeline: image path -> (annotated image, checklist rows, debug).

Split in two so the app can cache the heavy half:
  analyze()          detect + OCR + value checks + report   (slow, cacheable)
  render_annotated() draw boxes for chosen classes          (fast, re-run freely)
run() = both, for the CLI and simple callers.
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


def analyze(image_path: str, config: dict | None = None):
    """Heavy half: detection + OCR + value reads + checklist.

    Returns (detections, rows, debug).
    """
    if config is None:
        config = load_config()

    model_path = config["model_path"]
    conf = config.get("conf_threshold", 0.25)
    class_conf = config.get("class_conf", {})

    # 1) detect (with per-class confidence overrides)
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

    debug = {
        "detection_status": det_status,
        "ocr_status": ocr_status,
        "ocr_text": ocr_text.strip(),
        "n_detections": len(detections),
        "counts": count_by_class(detections),
        "ra_readings": ra_readings,
    }
    return detections, rows, debug


def render_annotated(image_path: str, detections: list, show_classes=None):
    """Cheap half: draw boxes (optionally only some classes, to cut clutter)."""
    draw = detections
    if show_classes is not None:
        draw = [d for d in detections if d["cls"] in show_classes]
    return annotate(image_path, draw)


def run(image_path: str, config: dict | None = None, show_classes=None):
    """analyze + render in one call (CLI / simple callers)."""
    detections, rows, debug = analyze(image_path, config)
    annotated = render_annotated(image_path, detections, show_classes)
    return annotated, rows, debug

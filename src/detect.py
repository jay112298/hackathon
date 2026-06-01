"""YOLO detection wrapper.

One job: take an image, return a list of detections. Each detection is a plain
dict so the rest of the pipeline never depends on ultralytics:

    {"cls": "title_block", "conf": 0.91, "xyxy": [x1, y1, x2, y2]}

If ultralytics isn't installed or the weights file is missing, we return an
empty list plus a status string. The app stays runnable either way — useful on
day 1 before you've trained anything.
"""
from __future__ import annotations
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    """Load YOLO once and cache it. Returns None if it can't load."""
    if not os.path.exists(model_path):
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    return YOLO(model_path)


def detect(image_path: str, model_path: str, conf: float = 0.25):
    """Run detection. Returns (detections, status_message)."""
    model = _load_model(model_path)
    if model is None:
        if not os.path.exists(model_path):
            return [], f"No model at '{model_path}' — train on Colab, drop best.pt here."
        return [], "ultralytics not installed — run: pip install ultralytics"

    results = model.predict(image_path, conf=conf, verbose=False)
    names = model.names  # {index: class_name}
    dets = []
    for r in results:
        for box in r.boxes:
            cls_idx = int(box.cls[0])
            dets.append(
                {
                    "cls": names.get(cls_idx, str(cls_idx)),
                    "conf": float(box.conf[0]),
                    "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                }
            )
    return dets, f"{len(dets)} detections."


def count_by_class(detections):
    """Helper: {class_name: count} from a detection list."""
    counts = {}
    for d in detections:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    return counts


def first_box(detections, cls_name):
    """Return the highest-confidence box [x1,y1,x2,y2] for a class, or None."""
    boxes = [d for d in detections if d["cls"] == cls_name]
    if not boxes:
        return None
    best = max(boxes, key=lambda d: d["conf"])
    return best["xyxy"]

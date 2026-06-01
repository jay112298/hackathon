"""Draw detection boxes on the drawing for the demo view."""
from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont

# Stable color per class so the same class is always the same color.
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
]


def _color(cls_name: str):
    return _PALETTE[hash(cls_name) % len(_PALETTE)]


def annotate(image_path: str, detections: list) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        c = _color(d["cls"])
        draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        label = f"{d['cls']} {d['conf']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=c, font=font)
    return img

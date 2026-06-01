"""OCR for the title-block crop.

We only OCR the region the detector flagged as `title_block` — far more reliable
than OCRing the whole noisy drawing. Falls back gracefully if pytesseract or the
tesseract binary is missing.
"""
from __future__ import annotations
from PIL import Image


def ocr_region(image_path: str, box=None):
    """OCR a [x1,y1,x2,y2] crop (or the whole image if box is None).

    Returns (text, status_message).
    """
    try:
        import pytesseract
    except ImportError:
        return "", "pytesseract not installed — run: pip install pytesseract"

    try:
        img = Image.open(image_path).convert("RGB")
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            img = img.crop((x1, y1, x2, y2))
        text = pytesseract.image_to_string(img)
        return text, "ok"
    except pytesseract.TesseractNotFoundError:
        return "", "tesseract binary missing — install it (brew install tesseract)"
    except Exception as e:  # noqa: BLE001 - demo robustness
        return "", f"OCR error: {e}"

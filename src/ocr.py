"""OCR helpers.

We OCR only small relevant crops (title block, or the area around a roughness
symbol) — far more reliable than OCRing the whole noisy drawing. Falls back
gracefully if pytesseract or the tesseract binary is missing.
"""
from __future__ import annotations
import re

from PIL import Image


def _ocr_pil(img, config=""):
    """OCR a PIL image. Returns (text, status)."""
    try:
        import pytesseract
    except ImportError:
        return "", "pytesseract not installed — run: pip install pytesseract"
    try:
        return pytesseract.image_to_string(img, config=config), "ok"
    except pytesseract.TesseractNotFoundError:
        return "", "tesseract binary missing — install it (brew install tesseract)"
    except Exception as e:  # noqa: BLE001 - demo robustness
        return "", f"OCR error: {e}"


def ocr_region(image_path: str, box=None):
    """OCR a [x1,y1,x2,y2] crop (or the whole image if box is None).

    Returns (text, status_message).
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return "", f"image open error: {e}"
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        img = img.crop((x1, y1, x2, y2))
    return _ocr_pil(img)


def _read_score(text: str) -> int:
    """Rank an OCR read: count digits, bonus for a decimal point (Ra has
    decimals) or an ISO N-grade like 'N8' (equally meaningful)."""
    bonus = (2 if "." in text else 0) + (3 if re.search(r"N\s?\d", text or "") else 0)
    return sum(ch.isdigit() for ch in text) + bonus


def crop_box_padded(image_path: str, box, pad_frac: float = 1.2, target_w: int = 480):
    """Return an upscaled, contrast-boosted grayscale crop around a box.

    Shared by OCR (clean input) and the UI (evidence thumbnails of what OCR saw).
    """
    from PIL import ImageOps
    img = Image.open(image_path).convert("L")
    W, H = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    crop = img.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                     min(W, int(x2 + px)), min(H, int(y2 + py))))
    crop = ImageOps.autocontrast(crop)
    if crop.width < target_w:
        s = max(2, round(target_w / max(1, crop.width)))
        crop = crop.resize((crop.width * s, crop.height * s), Image.LANCZOS)
    return crop


_EASYOCR_READER = None


def _easyocr_read(crop) -> str:
    """Digit read via EasyOCR (deep-learning OCR — much better than tesseract on
    tiny / rotated technical text). Returns '' if easyocr isn't installed."""
    global _EASYOCR_READER
    try:
        import easyocr
        import numpy as np
    except ImportError:
        return ""
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    arr = np.array(crop.convert("RGB"))
    best = ""
    # rotation_info catches Ra values written sideways along a surface;
    # 'N' in the allowlist so ISO grades (N8 = Ra 3.2) survive the read
    for text in _EASYOCR_READER.readtext(
            arr, allowlist="0123456789.N", rotation_info=[90, 180, 270], detail=0):
        if _read_score(text) > _read_score(best):
            best = text
    return best


def _tesseract_read(crop):
    """Digit read via tesseract across psm modes. Returns (best, status)."""
    binar = crop.point(lambda p: 255 if p > 128 else 0)  # high-contrast B/W
    whitelist = "-c tessedit_char_whitelist=0123456789.N"
    best = ""
    for image in (binar, crop):
        for psm in (7, 6, 8, 13):
            for wl in (whitelist, ""):
                text, status = _ocr_pil(image, config=f"--psm {psm} {wl}".strip())
                if "not installed" in status or "missing" in status:
                    return best, status
                if _read_score(text) > _read_score(best):
                    best = text
    return best, "ok"


def ocr_box_padded(image_path: str, box, pad_frac: float = 1.2):
    """OCR the area around a small symbol box.

    EasyOCR first (deep-learning, handles tiny/rotated digits), tesseract as
    backup — the richest digit read wins. Returns (text, status).
    """
    try:
        crop = crop_box_padded(image_path, box, pad_frac)
    except Exception as e:  # noqa: BLE001
        return "", f"image open error: {e}"

    easy = _easyocr_read(crop)
    tess, status = _tesseract_read(crop)
    best = easy if _read_score(easy) >= _read_score(tess) else tess
    return best, ("ok" if best else status)


def easyocr_read_full(image_path: str):
    """ONE EasyOCR pass over the whole sheet -> [{box, text, conf}].

    Used to attach text to *many* detections at once (e.g. every dimension):
    one full-image pass is far cheaper than OCRing 50 separate crops.
    box is [x1, y1, x2, y2] in image pixels.
    """
    global _EASYOCR_READER
    try:
        import easyocr
        import numpy as np
    except ImportError:
        return []
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    try:
        arr = np.array(Image.open(image_path).convert("RGB"))
    except Exception:  # noqa: BLE001
        return []
    words = []
    for pts, text, conf in _EASYOCR_READER.readtext(arr):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        words.append({"box": [min(xs), min(ys), max(xs), max(ys)],
                      "text": text, "conf": float(conf)})
    return words

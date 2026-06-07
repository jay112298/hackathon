"""OCR helpers.

We OCR only small relevant crops (title block, or the area around a roughness
symbol) — far more reliable than OCRing the whole noisy drawing. Falls back
gracefully if pytesseract or the tesseract binary is missing.
"""
from __future__ import annotations
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


def ocr_box_padded(image_path: str, box, pad_frac: float = 0.6):
    """OCR an expanded crop around a small symbol box.

    Roughness/symbol boxes are tiny and the value (e.g. 'Ra 0.8') sits next to
    the symbol, so we pad the box outward before OCR. pad_frac=0.6 grows each
    side by 60% of the box size. Upscales the crop to help OCR on small text.
    Returns (text, status).
    """
    try:
        img = Image.open(image_path).convert("L")  # grayscale helps OCR
    except Exception as e:  # noqa: BLE001
        return "", f"image open error: {e}"
    W, H = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    crop = img.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                     min(W, int(x2 + px)), min(H, int(y2 + py))))
    # upscale small crops so tesseract sees bigger glyphs
    if crop.width < 300:
        scale = max(2, int(300 / max(1, crop.width)))
        crop = crop.resize((crop.width * scale, crop.height * scale))
    # roughness Ra text is small + rotated/varied -> try several page-seg modes,
    # keep the read with the most digits. Whitelist digits + dot.
    whitelist = "-c tessedit_char_whitelist=0123456789."
    best = ""
    for psm in (7, 6, 8, 11, 13):
        text, status = _ocr_pil(crop, config=f"--psm {psm} {whitelist}")
        if "not installed" in status or "missing" in status:
            return "", status            # tooling absent -> stop early
        digits = sum(ch.isdigit() for ch in text)
        if digits > sum(ch.isdigit() for ch in best):
            best = text
    return best, "ok"

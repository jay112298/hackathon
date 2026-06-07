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


def _read_score(text: str) -> int:
    """Rank an OCR read: count digits, bonus for a decimal point (Ra has decimals)."""
    return sum(ch.isdigit() for ch in text) + (2 if "." in text else 0)


def ocr_box_padded(image_path: str, box, pad_frac: float = 1.2):
    """OCR an expanded, cleaned-up crop around a small symbol box.

    Roughness Ra text (e.g. '3.2') is tiny — the decimal point and 2nd digit get
    lost. To fix: pad the box generously, upscale a lot, binarize for crisp
    glyphs, then try several page-seg modes (with and without a digit whitelist)
    and keep the richest read. Returns (text, status).
    """
    from PIL import ImageOps
    try:
        img = Image.open(image_path).convert("L")  # grayscale
    except Exception as e:  # noqa: BLE001
        return "", f"image open error: {e}"

    W, H = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    crop = img.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                     min(W, int(x2 + px)), min(H, int(y2 + py))))

    crop = ImageOps.autocontrast(crop)
    # upscale hard so small glyphs (and the dot) survive
    if crop.width < 480:
        s = max(3, round(480 / max(1, crop.width)))
        crop = crop.resize((crop.width * s, crop.height * s), Image.LANCZOS)
    binar = crop.point(lambda p: 255 if p > 128 else 0)  # high-contrast B/W

    whitelist = "-c tessedit_char_whitelist=0123456789."
    best = ""
    for image in (binar, crop):
        for psm in (7, 6, 8, 13):
            for wl in (whitelist, ""):
                text, status = _ocr_pil(image, config=f"--psm {psm} {wl}".strip())
                if "not installed" in status or "missing" in status:
                    return "", status   # tooling absent -> stop
                if _read_score(text) > _read_score(best):
                    best = text
    return best, "ok"

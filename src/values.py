"""Value-level checks: read numbers off detected symbols and validate them.

Two jobs here:
  read_ra_values()        Ra / N-grade off each surface-roughness symbol
  read_dimension_values() text for every dimension box (one full-sheet OCR
                          pass) + find_duplicate_dims() to flag repeats

This is the step that turns "symbol present" into "symbol present AND value
valid".
"""
from __future__ import annotations
import re

from .ocr import ocr_box_padded, easyocr_read_full

_NUM = re.compile(r"\d+\.?\d*")
# ISO 1302 grade written next to the symbol, e.g. "N8" (== Ra 3.2)
_NGRADE = re.compile(r"N\s?(\d{1,2})", re.I)


def _nearest_allowed(value, allowed, tol=0.05):
    """Return an allowed Ra equal/very close to value, else None."""
    for a in allowed:
        if abs(value - a) <= tol:
            return a
    return None


def read_ra_values(image_path, roughness_boxes, ra_allowed, n_grade_map=None):
    """For each roughness box -> {raw, value, grade, valid, matched}.

    value   = parsed Ra as float (or None if nothing readable)
    grade   = the N-grade read, e.g. 'N8' (or None if a plain Ra number)
    matched = the allowed Ra it equals (or None)
    valid   = matched is not None

    Matching passes, most-specific first:
      0) N-grade — 'N8' converts to Ra 3.2 via n_grade_map, then checks
         ra_allowed. Must run first: the float pass would read the '8' out of
         'N8' as Ra 8 otherwise.
      1) float match — parse a number, match to an allowed Ra within tolerance.
      2) digit-string snap — OCR drops decimal points on tiny text, so compare
         digits only: '32' -> 3.2, '63' -> 6.3, '125' -> 12.5.
    """
    n_grade_map = n_grade_map or {}
    # map of "digits-only" -> allowed value, e.g. {'32':3.2, '63':6.3, '125':12.5}
    by_digits = {str(a).replace(".", ""): a for a in ra_allowed}

    readings = []
    for box in roughness_boxes:
        text, _ = ocr_box_padded(image_path, box)
        t = text or ""
        value, matched, grade = None, None, None

        # pass 0: N-grade (N8 == Ra 3.2)
        g = _NGRADE.search(t)
        if g:
            name = f"N{int(g.group(1))}"
            ra = n_grade_map.get(name)
            if ra is not None:
                grade, value = name, ra
                matched = _nearest_allowed(ra, ra_allowed)

        # pass 1: float match
        if grade is None:
            for tok in _NUM.findall(t):
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if v <= 0 or v > 200:
                    continue
                m = _nearest_allowed(v, ra_allowed)
                if m is not None:
                    value, matched = v, m
                    break
                if value is None:
                    value = v

        # pass 2: digit-string snap (recovers a dropped decimal point)
        if matched is None and grade is None:
            ds = "".join(c for c in t if c.isdigit())
            if ds in by_digits:
                matched = by_digits[ds]
                value = matched

        readings.append({
            "raw": t.strip(),
            "value": value,
            "grade": grade,
            "matched": matched,
            "valid": matched is not None,
            "box": [round(v, 1) for v in box],   # so the UI can show the crop
        })
    return readings


def read_dimension_values(image_path, dimension_boxes):
    """Attach OCR text to every dimension box -> [{box, text, value}].

    One EasyOCR pass over the whole sheet, then each word is assigned to the
    dimension box its center falls in (with a little padding — dimension text
    often pokes outside the detected box). value = first plausible number in
    that text, or None.
    """
    words = easyocr_read_full(image_path)
    readings = []
    for box in dimension_boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        pw, ph = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        hits = [w for w in words
                if x1 - pw <= (w["box"][0] + w["box"][2]) / 2 <= x2 + pw
                and y1 - ph <= (w["box"][1] + w["box"][3]) / 2 <= y2 + ph]
        text = " ".join(w["text"] for w in hits).strip()
        value = None
        for tok in _NUM.findall(text):
            try:
                v = float(tok)
            except ValueError:
                continue
            if 0 < v < 100000:
                value = v
                break
        readings.append({"box": [round(v_, 1) for v_ in box],
                         "text": text, "value": value})
    return readings


def find_duplicate_dims(dim_readings):
    """{value: [readings]} for every dimension value that appears 2+ times.

    A repeat *can* be legitimate (same feature dimensioned in two views), which
    is exactly what the reviewer-override flow in the app is for.
    """
    groups: dict = {}
    for r in dim_readings:
        if r["value"] is not None:
            groups.setdefault(r["value"], []).append(r)
    return {v: rs for v, rs in groups.items() if len(rs) > 1}

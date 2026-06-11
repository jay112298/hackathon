"""Value-level checks: read numbers off detected symbols and validate them.

Right now: surface-roughness Ra values. For each detected roughness box we OCR
the padded area, parse a number, and check it against the allowed Ra set in
config. This is the step that turns "symbol present" into "symbol present AND
value valid".
"""
from __future__ import annotations
import re

from .ocr import ocr_box_padded

_NUM = re.compile(r"\d+\.?\d*")


def _nearest_allowed(value, allowed, tol=0.05):
    """Return an allowed Ra equal/very close to value, else None."""
    for a in allowed:
        if abs(value - a) <= tol:
            return a
    return None


def read_ra_values(image_path, roughness_boxes, ra_allowed):
    """For each roughness box -> {raw, value, valid, matched}.

    value   = parsed float (or None if nothing readable)
    matched = the allowed Ra it equals (or None)
    valid   = matched is not None

    Two-pass matching, because OCR drops decimal points on tiny text:
      1) parse a float and match it to an allowed Ra (within tolerance), then
      2) digit-string snap — compare the read's digits (dots removed) to each
         allowed Ra's digits, so '32' -> 3.2, '63' -> 6.3, '125' -> 12.5.
    """
    # map of "digits-only" -> allowed value, e.g. {'32':3.2, '63':6.3, '125':12.5}
    by_digits = {str(a).replace(".", ""): a for a in ra_allowed}

    readings = []
    for box in roughness_boxes:
        text, _ = ocr_box_padded(image_path, box)
        t = text or ""
        value, matched = None, None

        # pass 1: float match
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
        if matched is None:
            ds = "".join(c for c in t if c.isdigit())
            if ds in by_digits:
                matched = by_digits[ds]
                value = matched

        readings.append({
            "raw": t.strip(),
            "value": value,
            "matched": matched,
            "valid": matched is not None,
            "box": [round(v, 1) for v in box],   # so the UI can show the crop
        })
    return readings

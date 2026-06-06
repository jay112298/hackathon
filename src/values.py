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
    """
    readings = []
    for box in roughness_boxes:
        text, _ = ocr_box_padded(image_path, box)
        nums = _NUM.findall(text or "")
        value, matched = None, None
        for tok in nums:
            try:
                v = float(tok)
            except ValueError:
                continue
            if v <= 0 or v > 100:      # ignore junk reads
                continue
            m = _nearest_allowed(v, ra_allowed)
            if m is not None:          # prefer a value that matches the spec
                value, matched = v, m
                break
            if value is None:
                value = v              # keep first plausible number as fallback
        readings.append({
            "raw": (text or "").strip(),
            "value": value,
            "matched": matched,
            "valid": matched is not None,
        })
    return readings

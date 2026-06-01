"""Rule half of the checklist — pure regex on OCR text. No training needed.

This is where company-specific format rules live (drawing-no pattern, date
format, etc.). All patterns come from config.yaml so you never edit code to
retarget to the real checklist.
"""
from __future__ import annotations
import re


def check_rule(rule_cfg: dict, text: str):
    """Apply one regex rule to text. Returns (matched: bool, found: str|None)."""
    pattern = rule_cfg.get("pattern", "")
    if not pattern or not text:
        return False, None
    m = re.search(pattern, text)
    return (bool(m), m.group(0) if m else None)

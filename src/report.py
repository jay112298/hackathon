"""Build the checklist report from detections + OCR text.

Turns config['checklist'] into result rows:
    {"id", "point", "status", "remarks"}
status is one of: PASS, FAIL, NA, INFO.
This mirrors the company PDF columns (Sl.No / Check point / Status / Remarks).
"""
from __future__ import annotations
from .detect import count_by_class
from .rules import check_rule


def build_report(config: dict, detections: list, ocr_text: str, model_loaded: bool,
                 ra_readings: list | None = None):
    counts = count_by_class(detections)
    rules_cfg = config.get("rules", {})
    ra_readings = ra_readings or []
    rows = []

    for item in config["checklist"]:
        itype = item["type"]
        status, remarks = "NA", ""

        if itype == "ra_values":
            if not model_loaded:
                status, remarks = "NA", "model not loaded"
            elif not ra_readings:
                status, remarks = "NA", "no roughness symbols"
            else:
                read = [r for r in ra_readings if r["value"] is not None]
                valid = [r for r in ra_readings if r["valid"]]
                if not read:
                    status = "INFO"
                    remarks = f"{len(ra_readings)} symbols, Ra unreadable"
                elif len(valid) == len(read):
                    status = "PASS"
                    remarks = "Ra " + ", ".join(str(r["matched"]) for r in valid) + " (all valid)"
                else:
                    bad = [str(r['value']) for r in read if not r["valid"]]
                    status = "FAIL"
                    remarks = f"{len(valid)}/{len(read)} valid; not in spec: {', '.join(bad)}"

        elif itype in ("presence", "presence_optional", "count", "count_info"):
            cls = item["requires_class"]
            n = counts.get(cls, 0)
            if not model_loaded:
                status, remarks = "NA", "model not loaded"
            elif itype == "presence":
                status = "PASS" if n >= 1 else "FAIL"
                remarks = f"{n} found"
            elif itype == "presence_optional":
                status = "PASS" if n >= 1 else "NA"
                remarks = f"{n} found" if n else "none (optional)"
            elif itype == "count":
                status = "PASS" if n >= 1 else "FAIL"
                remarks = f"{n} found"
            elif itype == "count_info":
                status = "INFO"
                remarks = f"{n} found"

        elif itype == "rule":
            rule = rules_cfg.get(item["requires_rule"], {})
            if not ocr_text:
                status, remarks = "NA", "no title-block text (OCR empty)"
            else:
                matched, found = check_rule(rule, ocr_text)
                status = "PASS" if matched else "FAIL"
                remarks = f"matched: {found}" if matched else "pattern not found"

        rows.append(
            {"id": item["id"], "point": item["point"], "status": status, "remarks": remarks}
        )
    return rows

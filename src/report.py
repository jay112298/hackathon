"""Build the checklist report from detections + OCR + value reads.

Each row is a clear comparison with its own columns:
    {id, point, detected, values, required, status}
- detected : what the model found (a count, or "N/M read")
- values   : the actual values read (e.g. Ra "3.0, 7.0"), or "—"
- required : what the check expects (e.g. "≥ 1", "∈ {0.4, 0.8, ...}")
- status   : PASS / FAIL / NA / INFO
"""
from __future__ import annotations
from .detect import count_by_class
from .rules import check_rule


def _row(item, detected="—", values="—", required="—", status="NA"):
    return {"id": item["id"], "point": item["point"], "detected": detected,
            "values": values, "required": required, "status": status}


def build_report(config: dict, detections: list, ocr_text: str, model_loaded: bool,
                 ra_readings: list | None = None):
    counts = count_by_class(detections)
    rules_cfg = config.get("rules", {})
    ra_readings = ra_readings or []
    rows = []

    for item in config["checklist"]:
        itype = item["type"]

        if not model_loaded and itype != "rule":
            rows.append(_row(item, detected="model not loaded", status="NA"))
            continue

        if itype == "ra_values":
            allowed = config.get("ra_allowed", [])
            req = "∈ {" + ", ".join(str(a) for a in allowed) + "}"
            strict = config.get("ra_strict", False)
            total = len(ra_readings)
            read = [r for r in ra_readings if r["value"] is not None]
            valid = [r for r in ra_readings if r["valid"]]
            if total == 0:
                rows.append(_row(item, detected="0 symbols", required=req, status="NA"))
            elif not read:
                rows.append(_row(item, detected=f"0/{total} read", values="unreadable",
                                 required=req, status="INFO"))
            else:
                vals = ", ".join(str(r["value"]) for r in read)
                if strict:
                    status = "PASS" if len(valid) == len(read) else "FAIL"
                else:
                    status = "INFO"
                rows.append(_row(item, detected=f"{len(read)}/{total} read", values=vals,
                                 required=req + f"  ({len(valid)} match)", status=status))

        elif itype in ("presence", "presence_optional", "count", "count_info"):
            n = counts.get(item["requires_class"], 0)
            if itype == "presence":
                rows.append(_row(item, detected=f"{n} found", required="≥ 1",
                                 status="PASS" if n >= 1 else "FAIL"))
            elif itype == "count":
                rows.append(_row(item, detected=f"{n} found", required="≥ 1",
                                 status="PASS" if n >= 1 else "FAIL"))
            elif itype == "count_info":
                rows.append(_row(item, detected=f"{n} found", required="informational",
                                 status="INFO"))
            elif itype == "presence_optional":
                rows.append(_row(item, detected=f"{n} found", required="optional",
                                 status="PASS" if n >= 1 else "NA"))

        elif itype == "rule":
            rule = rules_cfg.get(item["requires_rule"], {})
            req = rule.get("description", item["requires_rule"])
            if not ocr_text:
                rows.append(_row(item, detected="no text", required=req, status="NA"))
            else:
                matched, found = check_rule(rule, ocr_text)
                rows.append(_row(item, detected=found or "no match", required=req,
                                 status="PASS" if matched else "FAIL"))

    return rows

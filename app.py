"""Streamlit demo: upload a drawing -> annotated image + auto-filled checklist.

Run:  streamlit run app.py   (or ./scripts/serve.sh)
"""
import csv
import io
import tempfile

import streamlit as st

from src.pipeline import run, load_config

st.set_page_config(page_title="Drawing Checklist Automation", layout="wide")
st.title("Engineering Drawing — Automated Checklist")
st.caption("Upload a mechanical drawing → detect parts → auto-fill the QC checklist.")

config = load_config()
_STATUS_EMOJI = {"PASS": "✅", "FAIL": "❌", "NA": "➖", "INFO": "ℹ️"}


def rows_to_csv(rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Sl.No", "Check point", "Status", "Remarks"])
    for r in rows:
        w.writerow([r["id"], r["point"], r["status"], r["remarks"]])
    return buf.getvalue()


def pil_to_png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- sidebar: model info + controls ---
with st.sidebar:
    st.header("Settings")
    conf = st.slider("Detection confidence", 0.05, 0.90,
                     float(config.get("conf_threshold", 0.30)), 0.05,
                     help="Lower = more boxes (more false positives). 0.30 = model's peak F1.")
    config["conf_threshold"] = conf
    st.markdown("**Classes the model detects:**")
    st.write(", ".join(config.get("classes", [])))

uploaded = st.file_uploader("Upload a drawing sheet", type=["png", "jpg", "jpeg"])

if not uploaded:
    st.info("Upload a drawing image to run the checklist. "
            "Works without a trained model (rows show NA) until models/best.pt exists.")
    st.stop()

# write upload to a temp file (YOLO + OCR want a path)
with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

annotated, rows, debug = run(tmp_path, config)

# --- overall verdict ---
passed = sum(1 for r in rows if r["status"] == "PASS")
failed = sum(1 for r in rows if r["status"] == "FAIL")
actionable = passed + failed

if not debug.get("counts") and debug["n_detections"] == 0:
    st.warning("No detections. Lower the confidence slider, or try a clearer full-sheet drawing.")
elif failed == 0 and actionable:
    st.success(f"PASS — all {passed} checks satisfied.")
elif actionable:
    st.warning(f"NEEDS REVIEW — {failed} of {actionable} checks failed.")

if actionable:
    c1, c2 = st.columns(2)
    c1.metric("Checks passed", f"{passed}/{actionable}", f"{failed} failed")
    c2.metric("Total detections", debug["n_detections"])
    st.progress(passed / actionable)

# --- main: image + checklist ---
left, right = st.columns([3, 2])
with left:
    st.subheader("Detections")
    st.image(annotated, use_container_width=True)
    st.download_button("Download annotated image (PNG)",
                       data=pil_to_png_bytes(annotated),
                       file_name="annotated.png", mime="image/png")

with right:
    st.subheader("Checklist report")
    table = [
        {
            "Sl.No": r["id"],
            "Check point": r["point"],
            "Status": f"{_STATUS_EMOJI.get(r['status'], '')} {r['status']}",
            "Remarks": r["remarks"],
        }
        for r in rows
    ]
    st.table(table)
    st.download_button("Download report (CSV)", data=rows_to_csv(rows),
                       file_name="checklist_report.csv", mime="text/csv")

    counts = debug.get("counts", {})
    if counts:
        st.subheader("Detected per class")
        st.bar_chart(counts)

with st.expander("Debug (OCR text, raw status)"):
    st.json(debug)

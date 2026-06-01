"""Streamlit demo: upload a drawing -> annotated image + auto-filled checklist.

Run:  streamlit run app.py
"""
import csv
import io
import tempfile

import streamlit as st

from src.pipeline import run, load_config

st.set_page_config(page_title="Drawing Checklist Automation", layout="wide")
st.title("Engineering Drawing — Automated Checklist")

config = load_config()
_STATUS_EMOJI = {"PASS": "✅", "FAIL": "❌", "NA": "➖", "INFO": "ℹ️"}


def rows_to_csv(rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Sl.No", "Check point", "Status", "Remarks"])
    for r in rows:
        w.writerow([r["id"], r["point"], r["status"], r["remarks"]])
    return buf.getvalue()


# --- controls ---
conf = st.slider("Detection confidence threshold", 0.05, 0.90,
                 float(config.get("conf_threshold", 0.25)), 0.05)
config["conf_threshold"] = conf

uploaded = st.file_uploader("Upload a drawing sheet", type=["png", "jpg", "jpeg"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    annotated, rows, debug = run(tmp_path, config)

    # score = passed / actionable (ignore NA + INFO)
    passed = sum(1 for r in rows if r["status"] == "PASS")
    failed = sum(1 for r in rows if r["status"] == "FAIL")
    actionable = passed + failed
    if actionable:
        pct = passed / actionable
        st.metric("Checks passed", f"{passed}/{actionable}", f"{failed} failed")
        st.progress(pct)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Detections")
        st.image(annotated, use_column_width=True)
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
        st.download_button(
            "Download report (CSV)",
            data=rows_to_csv(rows),
            file_name="checklist_report.csv",
            mime="text/csv",
        )

    with st.expander("Debug (OCR text, raw detections)"):
        st.json(debug)
else:
    st.info("Upload a drawing image to run the checklist. "
            "Works without a trained model (rows show NA) until you add models/best.pt.")

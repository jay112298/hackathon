"""Streamlit demo: upload a drawing -> annotated image + auto-filled checklist.

Run:  streamlit run app.py   (or ./scripts/serve.sh)
"""
import csv
import io
import tempfile

import streamlit as st

from src.pipeline import run, load_config
from src.report_html import build_html

st.set_page_config(page_title="Drawing Checklist Automation", page_icon="📐", layout="wide")

_STATUS_EMOJI = {"PASS": "✅", "FAIL": "❌", "NA": "➖", "INFO": "ℹ️"}

st.markdown("""
<style>
.block-container{padding-top:2rem}
[data-testid="stMetricValue"]{font-size:1.6rem}
</style>
""", unsafe_allow_html=True)


def rows_to_csv(rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Sl.No", "Check point", "Status", "Remarks"])
    for r in rows:
        w.writerow([r["id"], r["point"], r["status"], r["remarks"]])
    return buf.getvalue()


def png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


config = load_config()
all_classes = config.get("classes", [])

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("📐 Settings")
    conf = st.slider("Detection confidence", 0.05, 0.90,
                     float(config.get("conf_threshold", 0.30)), 0.05,
                     help="Lower = more boxes. 0.30 = the model's peak-F1 point.")
    config["conf_threshold"] = conf
    st.markdown("**Show classes on image**")
    show = [c for c in all_classes if st.checkbox(c, value=True, key=f"show_{c}")]
    st.divider()
    st.caption("Model classes: " + ", ".join(all_classes))

# ---------------- header ----------------
st.title("Engineering Drawing — Automated Checklist")
st.caption("Upload a mechanical drawing → the model detects parts & reads values → "
           "the QC checklist fills itself.")

uploaded = st.file_uploader("Upload a drawing sheet", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("⬆️ Upload a drawing to run the checklist. "
            "Runs without a trained model too (rows show NA).")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

try:
    annotated, rows, debug = run(tmp_path, config, show_classes=set(show))
except Exception as e:  # noqa: BLE001 - never crash the demo
    st.error(f"Could not process this image: {e}")
    st.stop()

# ---------------- verdict + metrics ----------------
passed = sum(1 for r in rows if r["status"] == "PASS")
failed = sum(1 for r in rows if r["status"] == "FAIL")
actionable = passed + failed

if debug["n_detections"] == 0:
    st.warning("No detections. Lower the confidence slider, or try a clearer full-sheet drawing.")
elif failed == 0 and actionable:
    st.success(f"✅ PASS — all {passed} checks satisfied.")
elif actionable:
    st.warning(f"⚠️ NEEDS REVIEW — {failed} of {actionable} checks failed.")

m1, m2, m3 = st.columns(3)
m1.metric("Checks passed", f"{passed}/{actionable}" if actionable else "—",
          f"{failed} failed" if actionable else None)
m2.metric("Total detections", debug["n_detections"])
m3.metric("Confidence", f"{conf:.2f}")
if actionable:
    st.progress(passed / actionable)

# ---------------- tabs ----------------
tab_check, tab_img, tab_values, tab_dbg = st.tabs(
    ["✅ Checklist", "🖼 Detections", "🔎 Values (Ra)", "🐞 Debug"])

with tab_check:
    table = [{"Sl.No": r["id"], "Check point": r["point"],
              "Status": f"{_STATUS_EMOJI.get(r['status'],'')} {r['status']}",
              "Remarks": r["remarks"]} for r in rows]
    st.table(table)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Report (CSV)", rows_to_csv(rows),
                       "checklist_report.csv", "text/csv", use_container_width=True)
    c2.download_button("⬇️ Report (HTML / print to PDF)",
                       build_html(rows, annotated, debug["n_detections"], uploaded.name),
                       "checklist_report.html", "text/html", use_container_width=True)

with tab_img:
    st.image(annotated, use_container_width=True)
    counts = debug.get("counts", {})
    if counts:
        st.bar_chart(counts)
    st.download_button("⬇️ Annotated image (PNG)", png_bytes(annotated),
                       "annotated.png", "image/png")

with tab_values:
    ra = debug.get("ra_readings", [])
    if not ra:
        st.info("No surface-roughness symbols detected (nothing to value-check).")
    else:
        st.caption("Each roughness symbol → OCR the Ra value → check against the allowed set "
                   f"{config.get('ra_allowed', [])}.")
        st.table([{
            "#": i + 1,
            "Read (raw)": r["raw"] or "—",
            "Ra value": r["value"] if r["value"] is not None else "unreadable",
            "Matches spec": "✅ " + str(r["matched"]) if r["valid"] else "❌",
        } for i, r in enumerate(ra)])

with tab_dbg:
    st.json(debug)

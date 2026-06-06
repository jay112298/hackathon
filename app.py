"""Streamlit demo: upload a drawing -> annotated image + auto-filled checklist.

Run:  streamlit run app.py   (or ./scripts/serve.sh)
"""
import csv
import glob
import html
import io
import os
import tempfile

import streamlit as st

from src.pipeline import run, load_config
from src.report_html import build_html

st.set_page_config(page_title="Drawing Checklist Automation", page_icon="📐", layout="wide")

_EMOJI = {"PASS": "✅", "FAIL": "❌", "NA": "➖", "INFO": "ℹ️"}
_COLOR = {"PASS": "#137333", "FAIL": "#c5221f", "NA": "#777", "INFO": "#1a73e8"}
_BG = {"PASS": "#e6f4ea", "FAIL": "#fce8e6", "NA": "#fff", "INFO": "#e8f0fe"}

st.markdown("""
<style>
.block-container{padding-top:1.6rem;max-width:1300px}
[data-testid="stMetricValue"]{font-size:1.6rem}
table.cl{border-collapse:collapse;width:100%;font-size:14px}
table.cl th,table.cl td{border:1px solid #e3e3e3;padding:9px 11px;text-align:left}
table.cl th{background:#f4f6f8;font-weight:600}
</style>
""", unsafe_allow_html=True)


def rows_to_csv(rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Sl.No", "Check point", "Status", "Remarks"])
    for r in rows:
        w.writerow([r["id"], r["point"], r["status"], r["remarks"]])
    return buf.getvalue()


def checklist_html(rows) -> str:
    trs = ""
    for r in rows:
        s = r["status"]
        trs += (f"<tr style='background:{_BG.get(s,'#fff')}'>"
                f"<td>{r['id']}</td><td>{html.escape(r['point'])}</td>"
                f"<td style='color:{_COLOR[s]};font-weight:700'>{_EMOJI.get(s,'')} {s}</td>"
                f"<td>{html.escape(str(r['remarks']))}</td></tr>")
    return ("<table class='cl'><thead><tr><th>#</th><th>Check point</th>"
            "<th>Status</th><th>Remarks</th></tr></thead><tbody>"
            f"{trs}</tbody></table>")


def png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


config = load_config()
all_classes = config.get("classes", [])
demo_imgs = sorted(glob.glob("data/demo/*.png") + glob.glob("data/demo/*.jpg")
                   + glob.glob("data/demo/*.jpeg"))

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("📐 Controls")
    src = st.radio("Drawing source", ["Upload", "Demo sample"],
                   horizontal=True, disabled=not demo_imgs)
    image_path, source_name = None, "drawing"

    if src == "Upload":
        up = st.file_uploader("Upload a drawing", type=["png", "jpg", "jpeg"])
        if up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(up.read())
                image_path, source_name = tmp.name, up.name
    elif demo_imgs:
        pick = st.selectbox("Pick a sample", demo_imgs,
                            format_func=lambda p: os.path.basename(p))
        image_path, source_name = pick, os.path.basename(pick)

    st.divider()
    conf = st.slider("Detection confidence", 0.05, 0.90,
                     float(config.get("conf_threshold", 0.30)), 0.05,
                     help="Lower = more boxes. 0.30 = model's peak-F1 point.")
    config["conf_threshold"] = conf
    st.markdown("**Show on image**")
    show = [c for c in all_classes if st.checkbox(c, value=True, key=f"s_{c}")]
    st.divider()
    st.caption("Model: YOLO11n · classes: " + ", ".join(all_classes))

# ---------------- header ----------------
st.title("Engineering Drawing — Automated Checklist")
st.caption("The model detects parts & reads values → the QC checklist fills itself.")

if not image_path:
    st.info("⬅️ Upload a drawing or pick a demo sample to begin.")
    st.stop()

try:
    with st.spinner("Analyzing drawing…"):
        annotated, rows, debug = run(image_path, config, show_classes=set(show))
except Exception as e:  # noqa: BLE001
    st.error(f"Could not process this image: {e}")
    st.stop()

# ---------------- verdict + metrics ----------------
passed = sum(1 for r in rows if r["status"] == "PASS")
failed = sum(1 for r in rows if r["status"] == "FAIL")
actionable = passed + failed

if debug["n_detections"] == 0:
    st.warning("No detections. Lower the confidence slider, or try a clearer full sheet.")
elif failed == 0 and actionable:
    st.success(f"✅ PASS — all {passed} checks satisfied.")
elif actionable:
    st.warning(f"⚠️ NEEDS REVIEW — {failed} of {actionable} checks failed.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Checks passed", f"{passed}/{actionable}" if actionable else "—")
m2.metric("Failed", failed)
m3.metric("Detections", debug["n_detections"])
m4.metric("Confidence", f"{conf:.2f}")
if actionable:
    st.progress(passed / actionable)

st.divider()

# ---------------- main: image | checklist ----------------
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Detected parts")
    st.image(annotated, use_container_width=True)
    st.download_button("⬇️ Annotated image (PNG)", png_bytes(annotated),
                       "annotated.png", "image/png")
with right:
    st.subheader("Checklist report")
    st.markdown(checklist_html(rows), unsafe_allow_html=True)
    st.write("")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ CSV", rows_to_csv(rows), "checklist_report.csv",
                       "text/csv", use_container_width=True)
    d2.download_button("⬇️ HTML report", build_html(rows, annotated, debug["n_detections"],
                       source_name), "checklist_report.html", "text/html",
                       use_container_width=True)

# ---------------- details ----------------
tab_val, tab_counts, tab_dbg = st.tabs(["🔎 Values (Ra)", "📊 Per-class", "🐞 Debug"])
with tab_val:
    ra = debug.get("ra_readings", [])
    if not ra:
        st.info("No surface-roughness symbols detected (nothing to value-check).")
    else:
        st.caption(f"Each roughness symbol → OCR the Ra value → check vs allowed set "
                   f"{config.get('ra_allowed', [])}.")
        st.table([{
            "#": i + 1,
            "Read (raw)": r["raw"] or "—",
            "Ra value": r["value"] if r["value"] is not None else "unreadable",
            "Matches spec": "✅ " + str(r["matched"]) if r["valid"] else "❌",
        } for i, r in enumerate(ra)])
with tab_counts:
    counts = debug.get("counts", {})
    st.bar_chart(counts) if counts else st.info("Nothing detected.")
with tab_dbg:
    st.json(debug)

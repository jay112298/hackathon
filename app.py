"""Streamlit demo: upload a drawing -> annotated image + auto-filled checklist.

Run:  streamlit run app.py   (or ./scripts/serve.sh)

UX flow (for a first-time judge):
  pick drawing -> verdict banner -> checklist (Detected|Values|Required|Result)
  -> Evidence tab proves each reading with the actual crop OCR saw.
Heavy analysis is cached by (image, confidence): sliders/toggles redraw instantly.
"""
import csv
import hashlib
import html
import io
import glob
import os
import tempfile

import streamlit as st

from src.pipeline import analyze, render_annotated, load_config
from src.ocr import crop_box_padded
from src.report_html import build_html

st.set_page_config(page_title="Drawing Checklist Automation", page_icon="📐", layout="wide")

_EMOJI = {"PASS": "✅", "FAIL": "❌", "NA": "➖", "INFO": "ℹ️"}
_COLOR = {"PASS": "#137333", "FAIL": "#c5221f", "NA": "#777", "INFO": "#1a73e8"}
_BG = {"PASS": "#e6f4ea", "FAIL": "#fce8e6", "NA": "#fff", "INFO": "#e8f0fe"}

st.markdown("""
<style>
.block-container{padding-top:1.4rem;max-width:1340px}
[data-testid="stMetricValue"]{font-size:1.5rem}
table.cl{border-collapse:collapse;width:100%;font-size:13.5px}
table.cl th,table.cl td{border:1px solid #e3e3e3;padding:8px 10px;text-align:left}
table.cl th{background:#f4f6f8;font-weight:600}
</style>
""", unsafe_allow_html=True)


# ---------- cached heavy work ----------
@st.cache_data(show_spinner=False)
def cached_analyze(image_path: str, conf: float, _key: str):
    """Detect + OCR + checklist, cached on (image content, confidence)."""
    cfg = load_config()
    cfg["conf_threshold"] = conf
    return analyze(image_path, cfg)


def stable_tmp_path(data: bytes, suffix=".png") -> str:
    """Write upload to a content-addressed temp path (same bytes -> same path),
    so Streamlit reruns hit the analysis cache instead of re-detecting."""
    h = hashlib.sha1(data).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"drawing_{h}{suffix}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path


def rows_to_csv(rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Sl.No", "Check point", "Detected", "Detected values", "Required", "Result"])
    for r in rows:
        status = r["status"] + (" (overridden)" if r.get("overridden") else "")
        w.writerow([r["id"], r["point"], r["detected"], r["values"], r["required"], status])
    return buf.getvalue()


def checklist_html(rows) -> str:
    trs = ""
    for r in rows:
        s = r["status"]
        label = f"{_EMOJI.get(s, '')} {s}" + (" ✎" if r.get("overridden") else "")
        trs += (f"<tr style='background:{_BG.get(s, '#fff')}'>"
                f"<td>{r['id']}</td><td>{html.escape(r['point'])}</td>"
                f"<td>{html.escape(str(r['detected']))}</td>"
                f"<td>{html.escape(str(r['values']))}</td>"
                f"<td>{html.escape(str(r['required']))}</td>"
                f"<td style='color:{_COLOR[s]};font-weight:700'>{label}</td>"
                f"</tr>")
    return ("<table class='cl'><thead><tr><th>#</th><th>Check point</th>"
            "<th>Detected</th><th>Detected values</th><th>Required</th><th>Result</th>"
            f"</tr></thead><tbody>{trs}</tbody></table>")


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
            image_path = stable_tmp_path(up.getvalue())
            source_name = up.name
    elif demo_imgs:
        pick = st.selectbox("Pick a sample", demo_imgs,
                            format_func=lambda p: os.path.basename(p))
        image_path, source_name = pick, os.path.basename(pick)

    st.divider()
    conf = st.slider("Detection confidence", 0.05, 0.90,
                     float(config.get("conf_threshold", 0.30)), 0.05,
                     help="Lower = more boxes. 0.30 = the model's peak-F1 point.")
    st.markdown("**Show on image**")
    show = [c for c in all_classes if st.checkbox(c, value=True, key=f"s_{c}")]
    st.divider()
    st.caption("Model: YOLO11n (custom-trained) · OCR: EasyOCR + Tesseract")

# ---------------- header ----------------
st.title("Engineering Drawing — Automated Checklist")
st.caption("The model detects parts & reads values → the QC checklist fills itself.")

with st.expander("ℹ️ How it works"):
    st.markdown("""
1. **Detect** — a YOLO11 model (trained on 5k+ public engineering drawings) finds
   dimensions, notes, GD&T symbols and surface-roughness symbols.
2. **Read** — each roughness symbol is cropped, cleaned and OCR'd (EasyOCR + Tesseract)
   to extract its **Ra value or ISO N-grade** (N8 = Ra 3.2). One extra full-sheet OCR
   pass reads every dimension's text.
3. **Check** — a config-driven rule engine compares what was found against what the
   checklist requires (presence, counts, allowed Ra values, repeated dimensions).
4. **Review** — a human can override any failed check (with a reason) — e.g. the same
   value on two *different* features is fine. Overrides are noted in the export.
5. **Report** — the filled checklist + annotated drawing export as CSV / printable HTML.
""")

if not image_path:
    st.info("⬅️ Upload a drawing or pick a demo sample to begin.")
    st.stop()

try:
    with st.spinner("Analyzing drawing… (cached after first run)"):
        detections, rows, debug = cached_analyze(image_path, conf, source_name)
    annotated = render_annotated(image_path, detections, set(show))
except Exception as e:  # noqa: BLE001 - never crash the demo
    st.error(f"Could not process this image: {e}")
    st.stop()

# ---------------- reviewer overrides ----------------
# Widgets live under the checklist (below); their state is read here, *before*
# the verdict/metrics, so an accepted check counts as PASS everywhere
# (banner, metrics, CSV, HTML). Keys are per-image so a new drawing resets them.
img_key = hashlib.sha1(image_path.encode()).hexdigest()[:8]
for r in rows:
    if r["status"] == "FAIL" and st.session_state.get(f"ovr_{img_key}_{r['id']}"):
        reason = (st.session_state.get(f"ovr_reason_{img_key}_{r['id']}") or "").strip()
        r["status"] = "PASS"
        r["overridden"] = True
        note = "accepted by reviewer" + (f": {reason}" if reason else "")
        r["values"] = note if str(r["values"]) in ("—", "") else f"{r['values']} — {note}"
n_overridden = sum(1 for r in rows if r.get("overridden"))

# ---------------- verdict + metrics ----------------
passed = sum(1 for r in rows if r["status"] == "PASS")
failed = sum(1 for r in rows if r["status"] == "FAIL")
actionable = passed + failed

if debug["n_detections"] == 0:
    st.warning("No detections. Lower the confidence slider, or try a clearer full sheet.")
elif failed == 0 and actionable:
    st.success(f"✅ PASS — all {passed} checks satisfied."
               + (f" ({n_overridden} by reviewer override)" if n_overridden else ""))
elif actionable:
    st.warning(f"⚠️ NEEDS REVIEW — {failed} of {actionable} checks failed.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Checks passed", f"{passed}/{actionable}" if actionable else "—")
m2.metric("Failed", failed)
m3.metric("Detections", debug["n_detections"])
m4.metric("Ra values read", sum(1 for r in debug.get("ra_readings", [])
                                if r["value"] is not None))
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

    # reviewer overrides: a flag can be legitimate (e.g. the same dimension on
    # two different features) — let a human accept it, with a recorded reason
    reviewable = [r for r in rows if r["status"] == "FAIL" or r.get("overridden")]
    if reviewable:
        with st.expander(f"✏️ Reviewer overrides ({n_overridden} applied)",
                         expanded=failed > 0):
            st.caption("Accept a failed check to record a manual PASS. "
                       "The reason goes into the CSV/HTML report.")
            for r in reviewable:
                c1, c2 = st.columns([1, 2])
                c1.checkbox(f"Accept #{r['id']}", key=f"ovr_{img_key}_{r['id']}",
                            help=r["point"])
                c2.text_input("Reason", key=f"ovr_reason_{img_key}_{r['id']}",
                              placeholder="reason (recorded in report)",
                              label_visibility="collapsed")
    st.write("")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ CSV", rows_to_csv(rows), "checklist_report.csv",
                       "text/csv", use_container_width=True)
    d2.download_button("⬇️ HTML report", build_html(rows, annotated, debug["n_detections"],
                       source_name), "checklist_report.html", "text/html",
                       use_container_width=True)

# ---------------- evidence + details ----------------
tab_ev, tab_dim, tab_counts, tab_dbg = st.tabs(
    ["🔎 Evidence (Ra readings)", "📏 Dimension values", "📊 Per-class", "🐞 Debug"])

with tab_ev:
    ra = debug.get("ra_readings", [])
    if not ra:
        st.info("No surface-roughness symbols detected (nothing to value-check).")
    else:
        st.caption(f"Each roughness symbol below is the exact crop OCR analyzed. "
                   f"Allowed Ra set: {config.get('ra_allowed', [])}")
        cols_per_row = 3
        for start in range(0, len(ra), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, (i, r) in zip(cols, enumerate(ra[start:start + cols_per_row], start)):
                with col:
                    try:
                        st.image(crop_box_padded(image_path, r["box"], target_w=300),
                                 use_container_width=True)
                    except Exception:  # noqa: BLE001
                        st.caption("(crop unavailable)")
                    shown = r.get("grade") or r["value"]
                    if r["valid"]:
                        st.success(f"#{i + 1} · read **{shown}** → Ra {r['matched']} ✓")
                    elif r["value"] is not None:
                        st.warning(f"#{i + 1} · read **{shown}** — not in spec")
                    else:
                        st.error(f"#{i + 1} · unreadable (raw: '{r['raw'] or '—'}')")

with tab_dim:
    dimr = debug.get("dim_readings", [])
    dups = debug.get("dim_duplicates", {})
    if not dimr:
        st.info("No dimension text read (no dimensions detected, or check #6 disabled).")
    else:
        readable = [r for r in dimr if r["value"] is not None]
        st.caption(f"{len(readable)}/{len(dimr)} dimension boxes had readable text "
                   "(one full-sheet OCR pass).")
        if dups:
            st.warning(f"{len(dups)} value(s) repeat — possible double-dimensioning. "
                       "If they belong to different features, accept check #6 in "
                       "Reviewer overrides.")
            for v, rs in sorted(dups.items()):
                st.markdown(f"**{v}** appears **{len(rs)}×**:")
                cols = st.columns(min(4, len(rs)))
                for col, r in zip(cols, rs):
                    with col:
                        try:
                            st.image(crop_box_padded(image_path, r["box"],
                                                     pad_frac=0.4, target_w=260),
                                     use_container_width=True)
                        except Exception:  # noqa: BLE001
                            pass
                        st.caption(f"read: “{r['text']}”")
        else:
            st.success("No repeated dimension values.")
        if readable:
            st.caption("All values read: "
                       + ", ".join(str(v) for v in sorted(r["value"] for r in readable)))

with tab_counts:
    counts = debug.get("counts", {})
    st.bar_chart(counts) if counts else st.info("Nothing detected.")

with tab_dbg:
    st.json(debug)

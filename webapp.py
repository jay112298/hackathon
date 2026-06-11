"""FastAPI backend for the redesigned UI (static/index.html is the frontend).

Run:  ./scripts/serve_web.sh        (uvicorn webapp:app, port 8000)

Endpoints:
  GET  /                    the single-page app
  GET  /api/demos           demo drawings available in data/demo/
  POST /api/analyze         multipart upload -> full analysis JSON
  GET  /api/analyze_demo    ?name=sample_01.jpg -> same, for a demo image
  GET  /api/render          ?id=..&classes=a,b -> re-draw boxes for a class
                            subset (analysis stays cached server-side)

The heavy work (YOLO + OCR) runs once per unique image (content-hash cache);
re-rendering with different class filters is instant.
"""
from __future__ import annotations
import base64
import glob
import hashlib
import io
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.ocr import crop_box_padded
from src.pipeline import analyze, load_config, render_annotated

app = FastAPI(title="DrawQC")

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.join(_ROOT, "data", "demo")
_CACHE: dict[str, dict] = {}   # analysis id -> {image_path, detections, rows, debug}

os.makedirs(_DEMO_DIR, exist_ok=True)
app.mount("/demo-files", StaticFiles(directory=_DEMO_DIR), name="demo-files")


def _png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _crop_b64(image_path, box, pad=0.8, w=320) -> str | None:
    try:
        return _png_b64(crop_box_padded(image_path, box, pad_frac=pad, target_w=w))
    except Exception:  # noqa: BLE001
        return None


def _analyze_to_payload(image_path: str, source_name: str) -> dict:
    cfg = load_config()
    aid = hashlib.sha1(open(image_path, "rb").read()).hexdigest()[:12]

    if aid in _CACHE:
        c = _CACHE[aid]
        detections, rows, debug = c["detections"], c["rows"], c["debug"]
    else:
        detections, rows, debug = analyze(image_path, cfg)
        _CACHE[aid] = {"image_path": image_path, "detections": detections,
                       "rows": rows, "debug": debug}

    annotated = render_annotated(image_path, detections)

    ra_evidence = [{
        "crop": _crop_b64(image_path, r["box"], pad=1.0),
        "raw": r["raw"], "value": r["value"], "grade": r["grade"],
        "matched": r["matched"], "valid": r["valid"],
        "candidates": r.get("candidates", []),
    } for r in debug.get("ra_readings", [])]

    dup_evidence = [{
        "value": v,
        "crops": [{"img": _crop_b64(image_path, r["box"], pad=0.4, w=260),
                   "text": r["text"]} for r in rs],
    } for v, rs in sorted(debug.get("dim_duplicates", {}).items())]

    return {
        "id": aid,
        "source": source_name,
        "rows": rows,
        "counts": debug.get("counts", {}),
        "n_detections": debug.get("n_detections", 0),
        "classes": cfg.get("classes", []),
        "ra_allowed": cfg.get("ra_allowed", []),
        "annotated": _png_b64(annotated),
        "ra_evidence": ra_evidence,
        "dup_evidence": dup_evidence,
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(_ROOT, "static", "index.html"))


@app.get("/api/demos")
def demos():
    names = sorted(os.path.basename(p) for e in ("*.png", "*.jpg", "*.jpeg")
                   for p in glob.glob(os.path.join(_DEMO_DIR, e)))
    return [{"name": n, "url": f"/demo-files/{n}"} for n in names]


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    h = hashlib.sha1(data).hexdigest()[:16]
    ext = os.path.splitext(file.filename or "x.png")[1] or ".png"
    path = os.path.join(tempfile.gettempdir(), f"drawqc_{h}{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return _analyze_to_payload(path, file.filename or "upload")


@app.get("/api/analyze_demo")
def analyze_demo(name: str):
    path = os.path.join(_DEMO_DIR, os.path.basename(name))   # basename: no traversal
    if not os.path.isfile(path):
        raise HTTPException(404, f"no demo named {name}")
    return _analyze_to_payload(path, name)


@app.get("/api/render")
def render(id: str, classes: str = ""):
    c = _CACHE.get(id)
    if not c:
        raise HTTPException(404, "analysis expired — re-run analyze")
    show = set(s for s in classes.split(",") if s) or None
    img = render_annotated(c["image_path"], c["detections"], show)
    return {"annotated": _png_b64(img)}

"""Build a self-contained HTML report (printable to PDF in any browser).

Dependency-free: embeds the annotated image as base64 so the single .html file
is fully portable. Judges can open it and Cmd/Ctrl+P -> Save as PDF.
"""
from __future__ import annotations
import base64
import datetime
import html
import io

_STYLE = """
body{font-family:-apple-system,Segoe UI,Arial,sans-serif;margin:32px;color:#1a1a1a}
h1{font-size:22px;margin:0 0 4px} .sub{color:#666;margin:0 0 20px}
.score{font-size:18px;font-weight:600;margin:12px 0}
table{border-collapse:collapse;width:100%;margin:16px 0}
th,td{border:1px solid #ddd;padding:8px 10px;text-align:left;font-size:14px}
th{background:#f4f4f4}
.PASS{color:#137333;font-weight:600}.FAIL{color:#c5221f;font-weight:600}
.NA{color:#888}.INFO{color:#1a73e8}
img{max-width:100%;border:1px solid #ddd;margin-top:8px}
"""


def build_html(rows, annotated_img, n_detections, source_name="drawing"):
    passed = sum(1 for r in rows if r["status"] == "PASS")
    failed = sum(1 for r in rows if r["status"] == "FAIL")
    actionable = passed + failed
    verdict = "PASS" if (failed == 0 and actionable) else ("NEEDS REVIEW" if actionable else "—")

    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    trs = []
    for r in rows:
        st = r["status"]
        trs.append(
            f"<tr><td>{r['id']}</td><td>{html.escape(r['point'])}</td>"
            f"<td>{html.escape(str(r['detected']))}</td>"
            f"<td>{html.escape(str(r['values']))}</td>"
            f"<td>{html.escape(str(r['required']))}</td>"
            f"<td class='{st}'>{st}</td></tr>"
        )
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Drawing Checklist Report</title><style>{_STYLE}</style></head><body>
<h1>Engineering Drawing — Checklist Report</h1>
<p class="sub">Source: {html.escape(source_name)} &nbsp;|&nbsp; Generated: {ts}
 &nbsp;|&nbsp; Detections: {n_detections}</p>
<p class="score">Verdict: {verdict} &nbsp; ({passed}/{actionable} checks passed)</p>
<table><thead><tr><th>Sl.No</th><th>Check point</th><th>Detected</th>
<th>Detected values</th><th>Required</th><th>Result</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
<h3>Annotated drawing</h3>
<img src="data:image/png;base64,{b64}">
</body></html>"""

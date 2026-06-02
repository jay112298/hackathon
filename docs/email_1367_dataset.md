# Author-request email — 1,367-drawing dataset (bonus track)

**Paper:** "From Drawings to Decisions: A Hybrid Vision-Language Framework for Parsing 2D
Engineering Drawings into Structured Manufacturing Knowledge" — arXiv:2506.17374.

**Send to:** Muhammad Tayyab Khan (first/corresponding author). CC: Seung Ki Moon
(senior author, NTU Singapore).

**Find the email:** arXiv masks it. Get it from:
- Google Scholar profile "Muhammad Tayyab Khan NTU engineering drawings", or
- NTU Singapore directory / the paper's published Elsevier version, or
- the "view email" link on the arXiv abstract page (login may be needed).

---

## Subject

Request for access to the 1,367 2D engineering-drawing dataset (arXiv:2506.17374)

## Body

Dear Mr. Khan,

I am a student working on a hackathon project to automate quality-control checklists for
2D mechanical engineering drawings (detecting title blocks, dimensions, GD&T symbols,
surface-roughness symbols, and notes).

I read your paper "From Drawings to Decisions" (arXiv:2506.17374) and found the curated
dataset of 1,367 annotated drawings across nine categories an excellent match for my
work. I could not locate a public download link, so I am writing to ask whether the
dataset (and, if possible, the YOLOv11-OBB annotations) could be shared for
**non-commercial academic / educational use**.

I would gladly cite your paper and acknowledge your team in any presentation or report. I
am happy to sign any data-use agreement you require.

Thank you for your time and for releasing your research.

Best regards,
[Your name]
[Your institution / team]
[Your email]

---

**Notes**
- Send it **Day 1–2**; replies take days. Do NOT block on it — the merged Roboflow set
  (see `BUILD_GUIDE.md` Chapter 3) is the real base.
- If it arrives later: it has all 9 categories in one source → you can retrain the one
  model on it for cleaner results, and expand `config.yaml` classes.

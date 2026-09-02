# RD OCT Segmentation Tool — Design Spec

**Date:** 2026-08-31
**Status:** Approved design, pre-implementation

## Purpose

Add a new public tool to the Paulus Lab website's Tools section: a visitor uploads a single retinal
OCT B-scan image, and the site shows it segmented for four biomarkers relevant to rhegmatogenous
retinal detachment — SRF (subretinal fluid), ORC (outer retinal cysts), IRC (intraretinal cysts), and
ERM (epiretinal membrane) — using the SegFormer-B4 model developed in the `rrd-oct-pipeline` project.
The tool should explain what each biomarker means, be honest about the model's current (uneven)
per-class reliability, and let the visitor download the annotated result.

This is a showcase/research-demo tool, not a diagnostic device, and the design treats it as such
throughout (framing, disclaimers, and the honest reliability tags in section 5 are load-bearing, not
decorative).

## 1. Architecture — three independent pieces

```
┌─────────────────────────────┐      ┌──────────────────────────┐      ┌───────────────────────────┐
│ GitHub: PaulusLab/           │      │ Hugging Face Space        │      │ PaulusLab.github.io       │
│ rd-oct-segmentation-model    │      │ (Gradio app)               │      │ tools/                    │
│                              │      │                            │      │ rd-oct-segmentation-      │
│ - training/inference code    │─────▶│ - loads checkpoint from    │      │ tool.html                 │
│   (mirrored from             │ (mirror,│  HF's own storage        │      │                           │
│   rrd-oct-pipeline)          │  not a  │ - predict(image) →       │◀────▶│ - upload UI               │
│ - fold4 checkpoint via       │  runtime│  annotated mask + stats  │ (API │ - interactive result      │
│   Git LFS                    │  dep)   │ - public HTTP API        │ call)│ - legend, download button │
│ - README with real Dice      │      │                            │      │                           │
│   numbers                    │      │                            │      │                           │
└─────────────────────────────┘      └──────────────────────────┘      └───────────────────────────┘
```

**Why three pieces, not one.** GitHub Pages serves static files only — no backend, no compute. The
model (770MB checkpoint, PyTorch, SegFormer-B4) can't run there or reasonably run in-browser. Hugging
Face Spaces is purpose-built for "upload a file, run a model, get a result" and needs the weights
local to its own compute anyway, so it gets its own copy rather than fetching from GitHub at request
time. GitHub still gets a full copy via Git LFS as the versioned, citable record of what was actually
trained and shipped — kept in a **separate** repo from `PaulusLab.github.io` so multi-hundred-MB LFS
objects never touch the website's Jekyll/Actions build.

## 2. Model repo (`PaulusLab/rd-oct-segmentation-model`)

New repo. Contents:
- `src/` — mirrored from `rrd-oct-pipeline`: `models/biomarker_unet.py`, `preprocessing.py`,
  `data_loading.py` (only what's needed to load and run the checkpoint, not the full training pipeline)
- `checkpoints/biomarker_fold4_best.pth` (Git LFS) — chosen because it has the best per-class Dice of
  the 5 trained folds (see performance table below); the other 4 folds are not shipped here, they stay
  on the HPC
- `README.md` — what the model does, the 5-class scheme, the real performance table, and an explicit
  "research prototype, not for clinical use" statement
- `inference.py` — a thin, standalone script mirroring what the Gradio Space actually runs (single
  image in, annotated mask + per-class confidence out) — lets anyone reproduce what the Space does
  without needing HF at all
- `LICENSE` — whatever the lab's standard is for shared research code (flag to Yannis/lab norms, not
  assumed here)

## 3. Hugging Face Space (Gradio)

- `predict(image: PIL.Image) -> (overlay_png, per_class_stats)`
- Preprocessing mirrors the training pipeline exactly: percentile-normalize (1st–99th, matching
  `percentile_normalize()`), center-crop/pad to 512×512, duplicate the single slice into the 3-channel
  input the model expects (2.5D architecture — **this duplication is a known simplification**, called
  out explicitly in the UI, not hidden; the model was trained on genuine neighboring B-scans)
- Runs fold-4 checkpoint inference (single forward pass, no ensembling in v1)
- Returns a 5-class mask (BG/SRF/ORC/IRC/ERM), plus per-class pixel coverage % and the confidence map
- Public HTTP API (Gradio's built-in `/api/predict` or similar) — the tool page calls this directly,
  no server code needed on the website side
- Free tier to start; if the Space sleeps when idle, the website's loading state (section 6) accounts
  for the ~30–60s cold-start wake time

## 4. Model performance (shown honestly in the UI, from this session's real per-fold measurements)

| Class | Dice (fold 4) | Reliability tag |
|---|---:|---|
| SRF | 0.780 | High |
| IRC | 0.515 | Moderate |
| ORC | 0.369 | Experimental |
| ERM | 0.348 | Experimental |

These numbers, and the "research prototype" framing, live in three places that must stay in sync: the
model repo README, the Space's own info panel, and the tool page's legend — the legend is the primary
one users see, the other two are for anyone who follows the "view the model" link further.

## 5. Tool page (`tools/rd-oct-segmentation-tool.html`)

Standalone HTML file, following the existing pattern (`amd-tool.html`, `rds-calculator.html`) —
self-contained, Tailwind CDN, lab's teal design tokens, not a Jekyll-templated page.

**Layout:**
1. Header explaining what the tool does, 2-3 sentences, links to the model repo
2. Upload zone (drag-and-drop + click-to-browse), client-side validation (image format/dimensions)
   before calling the Space at all
3. On submit: loading state ("Running segmentation — first request may take up to a minute while the
   model wakes up")
4. Result view (fades in, not a hard pop):
   - Original image and annotated overlay, rendered together on a canvas
   - Per-biomarker toggle switches (show/hide each of SRF/ORC/IRC/ERM independently)
   - **Interactive hover**: moving the cursor over a highlighted region pops a tooltip at that point
     showing the biomarker name, a one-line plain-language explanation, and its reliability tag —
     exploring the image is how you learn what each finding means
   - Legend card alongside the image: color swatch → name → one-line explanation → Dice score →
     reliability tag (green/yellow/red), for all 4 classes, always visible (not just on hover)
   - **Download button**: exports the current view (whichever layers are toggled on, baked into the
     original image) as a PNG
5. Footer: "Research prototype — not for clinical use" + link to model repo for full performance
   details and methodology

**Biomarker explanations** (used in both the legend and hover tooltips):
- **SRF (Subretinal Fluid)** — fluid that has leaked beneath the retina; a hallmark of active retinal
  detachment.
- **ORC (Outer Retinal Cysts)** — fluid-filled cystic spaces in the outer retina, associated with
  chronic detachment.
- **IRC (Intraretinal Cysts)** — fluid-filled cystic spaces within the retina itself.
- **ERM (Epiretinal Membrane)** — a thin fibrous layer on the retinal surface that can distort vision
  and affect surgical outcome.

(Wording above is a starting point — worth a clinical-accuracy pass by the lab before shipping, not
treated as final copy in this spec.)

## 6. Error handling

- Invalid/non-image upload → rejected client-side with a clear message, before any API call
- Space cold-start → loading state explains the delay rather than looking broken
- Space unreachable or returns an error → graceful failure message, page doesn't crash, upload zone
  resets so the visitor can retry

## 7. Testing

Manual QA before launch: run a handful of the local annotated cases (known ground truth) through the
live tool, confirm the overlay renders correctly, hover tooltips show the right class at the right
location, toggles work independently, download produces a correct PNG, and the legend's Dice numbers
match the model repo README exactly. No automated test suite for this v1 — it's a demo tool, not a
service with an SLA.

## Out of scope (v1)

- Full B-scan volume/DICOM upload (single image only, per approved design)
- Model ensembling across multiple folds (fold 4 only)
- A generic reusable "legend" component shared across tools (this tool's legend is purpose-built; a
  shared component is a reasonable future refactor if a third tool needs the same pattern, not built
  preemptively here)
- Cross-vendor harmonization work from the earlier session (separate, unrelated effort on the HPC side)

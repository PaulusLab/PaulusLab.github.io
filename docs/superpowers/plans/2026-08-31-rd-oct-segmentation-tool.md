# RD OCT Segmentation Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a public tool at `tools/rd-oct-segmentation-tool.html` on the Paulus Lab website where a
visitor uploads one OCT B-scan image and gets back an interactive, hover-explorable segmentation
overlay for SRF/ORC/IRC/ERM, backed by a real trained model served from a Modal inference endpoint,
with the model itself versioned in a new GitHub repo.

**Architecture:** Two repos plus a serverless endpoint, one feature. `PaulusLab/rd-oct-segmentation-model`
holds inference code + the fold-4 checkpoint (Git LFS) as the citable artifact, and doubles as the
source Modal deploys from directly (no separate vendoring repo needed — Modal packages local files
into its container image at deploy time). Modal serves the public inference API — this is where
compute actually happens, billed per-invocation on its free Starter tier. `PaulusLab.github.io`'s new
static tool page calls that API directly from client-side JS via `fetch()`; no server code lives in
the website repo.

**Tech Stack:** PyTorch + HuggingFace Transformers (SegFormer-B4) for the model, Modal for serverless
serving (a FastAPI-based HTTP endpoint under Modal's `@modal.fastapi_endpoint` decorator), vanilla
HTML/CSS/JS + Tailwind (CDN) + Canvas API for the tool page — matching the existing
`amd-tool.html`/`rds-calculator.html` pattern (no build step, no framework).

## Global Constraints

- Model: `BiomarkerSegFormer` (`nvidia/mit-b4` encoder, `in_channels=3`, `out_channels=5`), exact
  architecture from `rrd-oct-pipeline/src/models/biomarker_unet.py` — do not modify.
- Checkpoint: `biomarker_fold4_best.pth`, chosen for best per-class Dice among the 5 trained folds.
  Loaded via `ckpt.get('model_state_dict', ckpt)` (matches existing inference scripts' checkpoint
  format).
- Preprocessing must match training exactly: percentile-normalize (1st–99th percentile, clip, scale to
  `[0,1]`), then center-crop/pad to 512×512. Any deviation changes what the model actually sees.
- 5-class label scheme, fixed order, never renumber: `0=Background, 1=SRF, 2=ORC, 3=IRC, 4=ERM`.
- Real performance numbers (from this session's fold measurements) — copy these exact values
  everywhere they appear (model README, Modal app's `RELIABILITY` dict, tool page legend), never
  re-derive or round differently:
  - SRF: Dice 0.780 — reliability tag "High"
  - IRC: Dice 0.515 — reliability tag "Moderate"
  - ORC: Dice 0.369 — reliability tag "Experimental"
  - ERM: Dice 0.348 — reliability tag "Experimental"
- Every surface (model README, Modal app description, tool page footer) must state: "Research
  prototype — not for clinical use." Not optional, not paraphrased away.
- Tool page is a standalone HTML file (no Jekyll front matter, no Liquid templating) — same pattern as
  `tools/amd-tool.html` and `tools/rds-calculator.html`.
- No automated JS test suite (matches site convention) — frontend tasks are verified by manual QA in
  a real browser. Python/model-repo tasks use pytest.

---

## Prerequisites (human steps — cannot be automated by the implementer)

Before Task 5, whoever executes this plan needs:
1. ~~A Hugging Face account + write-scope token~~ — no longer needed: Task 5/6 target Modal instead
   of Hugging Face Spaces (HF now requires a paid PRO subscription for Gradio/Docker Spaces even on
   free CPU hardware, discovered when actually attempting this — see Task 5's "Why Modal" note).
   Instead: a Modal account (free, no card required for the Starter tier — create at modal.com if the
   lab doesn't have one), authenticated locally via `python3 -m modal setup` (Task 5, Step 1) — this
   opens a browser to log in and stores a token locally, no value to type into any file or prompt.
2. The trained checkpoint copied from the HPC to the local machine running this plan. On the HPC:
   `scp lmunir2@<hpc-host>:/projects/retinal_ai/rd_oct_seg/pipeline/outputs/models/biomarker_fold4_best.pth ~/Downloads/`
   (the exact HPC hostname/login node depends on current cluster access — use whatever `ssh`/`scp`
   target already works for this account). Task 3 assumes the file lands at
   `~/Downloads/biomarker_fold4_best.pth`; adjust the path in that task if it lands elsewhere.

---

## Phase 1 — Model repo (`PaulusLab/rd-oct-segmentation-model`)

### Task 1: Create the repo and skeleton

**Files:**
- Create (local clone): `~/Desktop/Projects/rd-oct-segmentation-model/` (new directory, new repo)
- Create: `~/Desktop/Projects/rd-oct-segmentation-model/.gitattributes`
- Create: `~/Desktop/Projects/rd-oct-segmentation-model/requirements.txt`
- Create: `~/Desktop/Projects/rd-oct-segmentation-model/.gitignore`

**Interfaces:**
- Produces: a git repo at `PaulusLab/rd-oct-segmentation-model` (GitHub), cloned locally, with LFS
  configured for `*.pth` files — later tasks add code and the checkpoint into this structure.

- [ ] **Step 1: Create the GitHub repo**

```bash
gh repo create PaulusLab/rd-oct-segmentation-model --public \
  --description "SegFormer-B4 model for OCT retinal detachment biomarker segmentation (SRF/ORC/IRC/ERM)" \
  --clone
cd rd-oct-segmentation-model
```

- [ ] **Step 2: Configure Git LFS for the checkpoint**

```bash
git lfs install
git lfs track "*.pth"
```

- [ ] **Step 3: Write `.gitattributes` (created by `git lfs track`, verify it looks like this)**

```
*.pth filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 4: Write `.gitignore`**

```
__pycache__/
*.pyc
.DS_Store
.venv/
venv/
*.egg-info/
```

- [ ] **Step 5: Write `requirements.txt`**

```
torch>=2.0
torchvision
transformers>=4.35
numpy
scipy
pillow
pytest
```

- [ ] **Step 6: Commit the skeleton**

```bash
mkdir -p src tests checkpoints
git add .gitattributes .gitignore requirements.txt
git commit -m "chore: repo skeleton, Git LFS config for checkpoints"
git push
```

---

### Task 2: Model + preprocessing code, with a real unit test

**Files:**
- Create: `src/__init__.py` (empty)
- Create: `src/model.py`
- Create: `src/preprocessing.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/fixtures/sample_bscan.png` (copied from local OCT data — see step 1)
- Create: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `BiomarkerSegFormer(encoder, in_channels, out_channels, pretrained)` (torch `nn.Module`,
  `forward(x: Tensor[B,3,H,W]) -> Tensor[B,5,H,W]`), `percentile_normalize(data: np.ndarray, p_low=1,
  p_high=99) -> np.ndarray[float32]`, `center_crop_pad_2d(data: np.ndarray[H,W], target_shape:
  tuple[int,int]) -> np.ndarray[H,W]`.
- Consumes: nothing from other tasks (this is the foundational code layer).

- [ ] **Step 1: Copy the real test fixture image into place**

A real B-scan slice, exported from case `00030` of the local annotated dataset, already saved at
`/Users/luqmanmunir/Desktop/Projects/PaulusLab.github.io/sample_bscan_00030.png`. Copy it into the
new repo:

```bash
mkdir -p tests/fixtures
cp /Users/luqmanmunir/Desktop/Projects/PaulusLab.github.io/sample_bscan_00030.png \
   tests/fixtures/sample_bscan.png
```

- [ ] **Step 2: Write the failing test for `percentile_normalize` and `center_crop_pad_2d`**

```python
# tests/test_preprocessing.py
import numpy as np
from PIL import Image
from pathlib import Path
from src.preprocessing import percentile_normalize, center_crop_pad_2d

FIXTURE = Path(__file__).parent / "fixtures" / "sample_bscan.png"


def test_percentile_normalize_output_range():
    img = np.array(Image.open(FIXTURE).convert("L"), dtype=np.float32)
    normed = percentile_normalize(img)
    assert normed.dtype == np.float32
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0
    # a real B-scan has real contrast -- normalized output shouldn't collapse to a constant
    assert normed.std() > 0.05


def test_center_crop_pad_2d_pads_smaller_image():
    small = np.ones((100, 80), dtype=np.float32)
    result = center_crop_pad_2d(small, (512, 512))
    assert result.shape == (512, 512)
    # original content should be centered, not at the edge
    assert result[256, 256] == 1.0
    assert result[0, 0] == 0.0


def test_center_crop_pad_2d_crops_larger_image():
    large = np.ones((600, 700), dtype=np.float32)
    result = center_crop_pad_2d(large, (512, 512))
    assert result.shape == (512, 512)
    assert (result == 1.0).all()  # fully inside the source, no padding needed
```

- [ ] **Step 3: Run the test to verify it fails (module doesn't exist yet)**

Run: `pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.preprocessing'`

- [ ] **Step 4: Write `src/preprocessing.py`**

```python
"""Preprocessing for single-image OCT inference -- mirrors the training pipeline's
percentile_normalize exactly (rrd-oct-pipeline/src/preprocessing.py), and adapts
center_crop_pad from 3D volumes to a single 2D image (no Z axis here)."""

import numpy as np


def percentile_normalize(data: np.ndarray, p_low: int = 1, p_high: int = 99) -> np.ndarray:
    """Clip intensity to percentiles and scale to [0, 1]. Identical to the training
    pipeline's version -- do not change without also changing the training pipeline,
    or inference input distribution will no longer match what the model was trained on."""
    low, high = np.percentile(data, (p_low, p_high))
    data = np.clip(data, low, high)
    data = (data - low) / (high - low + 1e-8)
    return data.astype(np.float32)


def center_crop_pad_2d(data: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Crop or pad a single 2D image to target_shape, centered. 2D analog of the
    training pipeline's center_crop_pad (which operates on 3D volumes with a Z axis
    we don't have for a single uploaded image)."""
    curr_shape = data.shape
    new_data = np.zeros(target_shape, dtype=data.dtype)

    h_src_start = max(0, (curr_shape[0] - target_shape[0]) // 2)
    h_src_end = min(curr_shape[0], h_src_start + target_shape[0])
    h_dst_start = max(0, (target_shape[0] - curr_shape[0]) // 2)
    h_dst_end = min(target_shape[0], h_dst_start + (h_src_end - h_src_start))

    w_src_start = max(0, (curr_shape[1] - target_shape[1]) // 2)
    w_src_end = min(curr_shape[1], w_src_start + target_shape[1])
    w_dst_start = max(0, (target_shape[1] - curr_shape[1]) // 2)
    w_dst_end = min(target_shape[1], w_dst_start + (w_src_end - w_src_start))

    new_data[h_dst_start:h_dst_end, w_dst_start:w_dst_end] = \
        data[h_src_start:h_src_end, w_src_start:w_src_end]

    return new_data
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_preprocessing.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Write `src/model.py`** (exact mirror of `rrd-oct-pipeline/src/models/biomarker_unet.py`)

```python
"""BiomarkerSegFormer -- exact mirror of rrd-oct-pipeline/src/models/biomarker_unet.py.
Do not modify the architecture; it must match the trained checkpoint's state_dict keys."""

import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class BiomarkerSegFormer(nn.Module):
    """SegFormer-B4 encoder with a lightweight MLP decoder. Output is upsampled 4x
    (from H/4, W/4) to match input resolution."""

    def __init__(self, encoder="nvidia/mit-b4", in_channels=3, out_channels=5, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                encoder,
                num_labels=out_channels,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(encoder, num_labels=out_channels)
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
```

- [ ] **Step 7: Commit**

```bash
git add src/ tests/
git commit -m "feat: add model architecture and single-image preprocessing, with tests"
git push
```

---

### Task 3: Checkpoint + inference wrapper, with an integration test

**Files:**
- Create: `checkpoints/.gitkeep` (placeholder until the real checkpoint is added)
- Create: `checkpoints/biomarker_fold4_best.pth` (via Git LFS — see step 1, requires the Prerequisite
  checkpoint download)
- Create: `inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `BiomarkerSegFormer` and `percentile_normalize`/`center_crop_pad_2d` from Task 2.
- Produces: `predict(image_path: str, checkpoint_path: str = "checkpoints/biomarker_fold4_best.pth")
  -> dict` with keys `mask` (`np.ndarray[512,512]` uint8, values 0-4), `confidence`
  (`np.ndarray[512,512]` float32, max-softmax per pixel), `class_pixel_pct` (`dict[str, float]`, one
  entry per non-background class) — this exact return shape is what the Modal app (Task 5) and any
  future consumer build on.

- [ ] **Step 1: Copy the checkpoint into place (human-downloaded per Prerequisites)**

```bash
cp ~/Downloads/biomarker_fold4_best.pth checkpoints/biomarker_fold4_best.pth
ls -lh checkpoints/biomarker_fold4_best.pth   # sanity check: should be ~769MB
```

- [ ] **Step 2: Write the failing integration test**

```python
# tests/test_inference.py
import numpy as np
from pathlib import Path
from inference import predict

FIXTURE = Path(__file__).parent / "fixtures" / "sample_bscan.png"
CHECKPOINT = Path(__file__).parent.parent / "checkpoints" / "biomarker_fold4_best.pth"


def test_predict_returns_expected_shapes():
    result = predict(str(FIXTURE), checkpoint_path=str(CHECKPOINT))
    assert result["mask"].shape == (512, 512)
    assert result["mask"].dtype == np.uint8
    assert result["mask"].max() <= 4
    assert result["confidence"].shape == (512, 512)
    assert 0.0 <= result["confidence"].min() and result["confidence"].max() <= 1.0
    assert set(result["class_pixel_pct"].keys()) == {"SRF", "ORC", "IRC", "ERM"}
    for pct in result["class_pixel_pct"].values():
        assert 0.0 <= pct <= 100.0
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference'`

- [ ] **Step 4: Write `inference.py`**

```python
"""Single-image inference for the RD OCT segmentation model.

Loads a B-scan image, preprocesses it exactly like training (percentile-normalize,
center-crop/pad to 512x512), duplicates it into the 3-channel input the 2.5D model
expects (a known simplification -- the model was trained on 3 genuine adjacent
B-scans, not one repeated slice; documented in this repo's README, not hidden)."""

import numpy as np
import torch
from PIL import Image

from src.model import BiomarkerSegFormer
from src.preprocessing import percentile_normalize, center_crop_pad_2d

LABEL_NAMES = ["BG", "SRF", "ORC", "IRC", "ERM"]
INPUT_SIZE = (512, 512)

_model_cache = {}


def _load_model(checkpoint_path: str, device: torch.device) -> BiomarkerSegFormer:
    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]

    model = BiomarkerSegFormer(
        encoder="nvidia/mit-b4", in_channels=3, out_channels=5, pretrained=False
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    _model_cache[checkpoint_path] = model
    return model


def predict(image_path: str, checkpoint_path: str = "checkpoints/biomarker_fold4_best.pth") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)

    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    img = percentile_normalize(img)
    img = center_crop_pad_2d(img, INPUT_SIZE)

    stack = np.stack([img, img, img], axis=0)  # (3, H, W) -- single slice duplicated 3x
    input_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device)  # (1, 3, H, W)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)

    mask = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    conf_map = confidence.squeeze(0).cpu().numpy().astype(np.float32)

    total_pixels = mask.size
    class_pixel_pct = {
        LABEL_NAMES[c]: round(100.0 * (mask == c).sum() / total_pixels, 2)
        for c in range(1, 5)
    }

    return {"mask": mask, "confidence": conf_map, "class_pixel_pct": class_pixel_pct}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_inference.py -v`
Expected: PASS. (First run downloads `nvidia/mit-b4` config from HuggingFace Hub if not cached --
needs internet access once.)

- [ ] **Step 6: Commit** (checkpoint goes through Git LFS automatically per Task 1's `.gitattributes`)

```bash
git add checkpoints/biomarker_fold4_best.pth inference.py tests/test_inference.py
git commit -m "feat: add single-image inference wrapper with fold4 checkpoint, integration test"
git push
```

Verify the LFS push actually worked (not a regular git blob):
```bash
git lfs ls-files
```
Expected output includes `biomarker_fold4_best.pth`.

---

### Task 4: README with real performance numbers

**Files:**
- Create: `README.md`

**Interfaces:**
- Produces: nothing consumed by other tasks — this is documentation, but the Dice numbers here MUST
  match Task 5's Space card and the tool page's legend (Task 9) exactly.

- [ ] **Step 1: Write `README.md`**

```markdown
# RD OCT Segmentation Model

SegFormer-B4 model segmenting four retinal detachment biomarkers from OCT B-scan images:
SRF (subretinal fluid), ORC (outer retinal cysts), IRC (intraretinal cysts), and ERM (epiretinal
membrane). Developed by the Paulus Lab, Johns Hopkins University.

**⚠️ Research prototype — not for clinical use.**

## Live demo

Try it at [pauluslab.github.io/tools/rd-oct-segmentation-tool.html](https://pauluslab.github.io/tools/rd-oct-segmentation-tool.html),
served by a Modal inference endpoint (`modal_app.py`) running the exact code in this repo.

## Performance

Measured on a 5-fold cross-validation split of 35 manually annotated cases (3D Slicer, single
annotator). Fold 4 (shipped here) had the best per-class Dice of the 5 trained folds:

| Class | Dice | Reliability |
|---|---:|---|
| SRF (Subretinal Fluid) | 0.780 | High |
| IRC (Intraretinal Cysts) | 0.515 | Moderate |
| ORC (Outer Retinal Cysts) | 0.369 | Experimental |
| ERM (Epiretinal Membrane) | 0.348 | Experimental |

ORC and ERM performance is not yet reliable — predictions for these two classes should be treated as
low-confidence starting points, not findings. This model is under active development via an
active-learning correction loop (predict → manually correct → retrain), so these numbers are expected
to improve; this README will be updated as new folds are trained.

## Usage

```python
from inference import predict

result = predict("path/to/bscan.png")
result["mask"]             # (512, 512) uint8, 0=BG 1=SRF 2=ORC 3=IRC 4=ERM
result["confidence"]       # (512, 512) float32, max-softmax per pixel
result["class_pixel_pct"]  # {"SRF": 2.1, "ORC": 0.0, "IRC": 0.4, "ERM": 0.0}
```

**Known limitation:** this model is 2.5D — it was trained on 3 adjacent B-scan slices as a 3-channel
input, not a single flat image. `predict()` duplicates the single input image into all 3 channels as
a simplification for single-image use, which does not give the model the neighboring-slice context it
was trained with. Expect somewhat lower accuracy on single images than the Dice scores above, which
were measured on true 3-slice volume input.

## Model architecture

`nvidia/mit-b4` (SegFormer-B4) encoder + lightweight MLP decoder, 5-class output upsampled to input
resolution. See `src/model.py`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with real performance numbers and usage"
git push
```

---

## Phase 2 — Modal inference endpoint

**Why Modal, not Hugging Face Spaces:** the original design targeted HF Spaces, but HF now requires a
paid PRO subscription ($9/mo) to host any Gradio/Docker Space, even on free CPU hardware (confirmed
directly against the live API, not assumed). Modal's Starter tier is genuinely free ($30/month compute
credit, no card required, no idle charges — you only pay for actual inference time, and a single-image
CPU forward pass costs a small fraction of a cent), so Phase 2 targets Modal instead. This also
simplifies deployment: Modal packages local files into its container image directly at `modal deploy`
time, so there's no separate git repo to create/clone/vendor into — `modal_app.py` lives right in the
model repo alongside the `src/`, `inference.py`, and `checkpoints/` it packages.

**A note on Modal API accuracy:** Modal's web-endpoint decorator has been renamed before
(`@modal.web_endpoint` → `@modal.fastapi_endpoint` as of v0.73.82) and may change again. The code below
reflects Modal's currently-documented pattern as of this plan's writing. If `modal deploy` errors on a
decorator or class/function composition that doesn't match what's below, that's a real signal the API
moved again — check `https://modal.com/docs/examples` for the current "web endpoint with a loaded ML
model" pattern rather than fighting the exact code here; this is exactly the kind of judgment call this
task should escalate on if the fix isn't a one-line adjustment (see "When You're in Over Your Head" in
your dispatch).

### Task 5: Create and deploy the Modal inference endpoint

**Files:**
- Create: `~/Desktop/Projects/rd-oct-segmentation-model/modal_app.py`

**Interfaces:**
- Consumes: `predict()` from `inference.py` (Task 3) — imported directly, no vendoring needed since
  Modal packages this repo's own files into its deploy image.
- Produces: a public HTTPS POST endpoint (URL printed by `modal deploy`, confirmed in Task 6) that
  accepts multipart form data with an `image` field and returns JSON:
  `{"overlay_png_base64": str, "stats": {"SRF": {...}, "ORC": {...}, "IRC": {...}, "ERM": {...}}}`
  where each class's stats dict has `pixel_pct` (float), `dice` (float), `reliability` (str) — this
  exact shape is what Task 7's frontend `fetch()` call parses.

- [ ] **Step 1: Install the Modal CLI and authenticate**

```bash
pip install modal
python3 -m modal setup
```

This opens a browser to authenticate with your Modal account (free, no card required for the Starter
tier) and stores a token locally — same one-time-local-auth pattern already used for `gh` and Hugging
Face in this project, no token value needs to be typed into any file.

- [ ] **Step 2: Write `modal_app.py`**

```python
"""Modal app serving RD OCT segmentation inference. Deploy with:
    modal deploy modal_app.py
Wraps inference.predict() with a color overlay renderer and per-class stats, exposed as a public
HTTP POST endpoint (FastAPI-based under the hood, CORS enabled by default so the tool page's
cross-origin fetch() call works without extra configuration)."""

import base64
import io
import os

import modal

app = modal.App("rd-oct-segmentation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.35", "numpy", "scipy", "pillow", "fastapi[standard]")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("inference.py", remote_path="/root/inference.py")
    .add_local_file(
        "checkpoints/biomarker_fold4_best.pth",
        remote_path="/root/checkpoints/biomarker_fold4_best.pth",
    )
)

# Fixed color scheme -- the tool page's legend/hover tooltips (Task 8) must use these
# exact colors so the overlay and legend agree visually.
CLASS_COLORS = {
    1: (239, 68, 68),    # SRF -- red
    2: (245, 158, 11),   # ORC -- amber
    3: (59, 130, 246),   # IRC -- blue
    4: (168, 85, 247),   # ERM -- purple
}

RELIABILITY = {
    "SRF": {"dice": 0.780, "tag": "High"},
    "IRC": {"dice": 0.515, "tag": "Moderate"},
    "ORC": {"dice": 0.369, "tag": "Experimental"},
    "ERM": {"dice": 0.348, "tag": "Experimental"},
}

# Global cache: populated on first call in a warm container, reused on subsequent
# calls to the same warm container -- avoids reloading the model every request.
_state = {}


def _get_predict():
    if "predict" not in _state:
        import sys
        sys.path.insert(0, "/root")
        os.chdir("/root")
        from inference import predict
        _state["predict"] = predict
    return _state["predict"]


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def segment(image: "UploadFile" = None):
    from fastapi import UploadFile, File
    import numpy as np
    from PIL import Image as PILImage

    if image is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No image provided.")

    contents = await image.read()
    tmp_path = "/tmp/upload.png"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    predict = _get_predict()
    result = predict(tmp_path, checkpoint_path="/root/checkpoints/biomarker_fold4_best.pth")

    overlay = np.zeros((*result["mask"].shape, 4), dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        m = result["mask"] == class_idx
        overlay[m, 0] = color[0]
        overlay[m, 1] = color[1]
        overlay[m, 2] = color[2]
        overlay[m, 3] = 140

    overlay_img = PILImage.fromarray(overlay, mode="RGBA")
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    stats = {
        name: {
            "pixel_pct": result["class_pixel_pct"][name],
            "dice": RELIABILITY[name]["dice"],
            "reliability": RELIABILITY[name]["tag"],
        }
        for name in ["SRF", "ORC", "IRC", "ERM"]
    }

    return {"overlay_png_base64": overlay_b64, "stats": stats}
```

Note: the `image: "UploadFile" = None` type hint uses a string forward-reference deliberately, since
`UploadFile` is imported inside the function body (not at module top-level) — Modal's own examples
follow this pattern for dependencies (torch, PIL, fastapi's request-parsing types) that only need to
exist inside the deployed container, not in the local environment running `modal deploy`. If this
causes a FastAPI parameter-typing issue at deploy time, move the `from fastapi import UploadFile, File`
import to module top-level instead (fastapi is a lightweight enough dependency that this is safe
either way) and use `image: UploadFile = File(...)` as the parameter signature — try the top-level
import first if the string-annotation version doesn't validate correctly, and note in your report
which one you used.

- [ ] **Step 3: Deploy**

```bash
cd ~/Desktop/Projects/rd-oct-segmentation-model
modal deploy modal_app.py
```

This uploads the checkpoint (via Modal's image-building layer, not Git LFS) and prints the deployed
endpoint's public URL on success (format like `https://<your-modal-username>--rd-oct-segmentation-segment.modal.run`)
— copy this exact URL, Task 6 and Task 7 both need it.

---

### Task 6: Verify the live Modal endpoint works end to end

**Files:** none (verification task, no new files)

**Interfaces:**
- Consumes: the deployed Modal endpoint's public URL (from Task 5, Step 3's `modal deploy` output).
- Produces: confirmation the tool page (Task 7) has a working backend to call, and the exact URL for
  Task 7 to use.

- [ ] **Step 1: Test the live endpoint with the same fixture image**

```bash
curl -X POST "<the modal deploy URL from Task 5>" \
  -F "image=@/Users/luqmanmunir/Desktop/Projects/PaulusLab.github.io/sample_bscan_00030.png" \
  | python3 -m json.tool | head -20
```

Expected: JSON with `overlay_png_base64` (a long base64 string) and `stats` (an object with SRF/ORC/IRC/ERM
keys, each having `pixel_pct`/`dice`/`reliability`). First request after deploy may take 10-30s (cold
start — the container needs to start and load the model); that's expected, not an error. If it fails,
check `modal app logs rd-oct-segmentation` for the actual error rather than guessing.

- [ ] **Step 2: Record the exact endpoint URL for Task 7**

Note the exact URL string that worked in Step 1 — Task 7's `MODAL_ENDPOINT_URL` constant must match it
exactly (including the specific subdomain Modal assigned, which depends on your Modal username).

---

## Phase 3 — Tool page (`PaulusLab.github.io`)

All of Phase 3 happens on the existing `rd-oct-segmentation-tool` branch already pushed to
`PaulusLab/PaulusLab.github.io` (created during the design-spec step).

### Task 7: Page shell, upload, and API call

**Files:**
- Create: `tools/rd-oct-segmentation-tool.html`

**Interfaces:**
- Consumes: the Modal endpoint's public API (URL confirmed in Task 6, Step 2).
- Produces: a `runSegmentation(imageFile)` JS function that later tasks (8, 9) attach rendering logic
  to — it resolves with `{overlayImageUrl, stats}` matching the endpoint's JSON response shape.

- [ ] **Step 1: Write the page shell with upload UI**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RD OCT Segmentation Tool — Paulus Lab</title>
    <script src="vendor/tailwind.min.js"></script>
    <link href="vendor/fonts/inter/inter.css" rel="stylesheet">
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: #f0fdfa; min-height: 100vh; margin: 0; }

        .page-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #164e63 100%);
            padding: 40px 0 36px;
        }
        .page-header-inner { max-width: 900px; margin: 0 auto; padding: 0 24px; }
        .page-title { color: #fff; font-size: 2rem; font-weight: 800; letter-spacing: -0.01em; }
        .page-subtitle { color: rgba(255,255,255,0.65); font-size: 0.9rem; margin-top: 10px; max-width: 560px; line-height: 1.5; }
        .research-badge {
            display: inline-flex; align-items: center; gap: 6px; margin-top: 16px;
            background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.4);
            color: #fbbf24; padding: 6px 12px; border-radius: 999px; font-size: 0.72rem; font-weight: 700;
        }

        .content { max-width: 900px; margin: 0 auto; padding: 32px 24px 60px; }

        .upload-zone {
            border: 2px dashed #99f6e4; border-radius: 16px; background: #fff;
            padding: 48px 24px; text-align: center; cursor: pointer;
            transition: border-color 0.15s, background 0.15s;
        }
        .upload-zone.dragover { border-color: #0d9488; background: #f0fdfa; }
        .upload-zone-icon { color: #0d9488; margin-bottom: 12px; }
        .upload-zone-text { font-size: 0.95rem; color: #374151; font-weight: 600; }
        .upload-zone-sub { font-size: 0.8rem; color: #9ca3af; margin-top: 4px; }

        .loading-state { display: none; text-align: center; padding: 48px 24px; }
        .loading-state.active { display: block; }
        .loading-spinner {
            width: 36px; height: 36px; border: 3px solid #ccfbf1; border-top-color: #0d9488;
            border-radius: 50%; margin: 0 auto 16px; animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text { font-size: 0.85rem; color: #6b7280; }

        .error-state {
            display: none; background: #fef2f2; border: 1px solid #fecaca; color: #991b1b;
            border-radius: 10px; padding: 16px 20px; font-size: 0.85rem; margin-top: 16px;
        }
        .error-state.active { display: block; }
    </style>
</head>
<body>

<div class="page-header">
    <div class="page-header-inner">
        <h1 class="page-title">RD OCT Segmentation Tool</h1>
        <p class="page-subtitle">
            Upload a retinal OCT B-scan image to segment four biomarkers relevant to
            retinal detachment: subretinal fluid, outer/intraretinal cysts, and
            epiretinal membrane. Built on a SegFormer-B4 model trained by the Paulus Lab.
        </p>
        <span class="research-badge">⚠ Research Prototype — Not for Clinical Use</span>
    </div>
</div>

<div class="content">
    <div id="upload-zone" class="upload-zone">
        <div class="upload-zone-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin: 0 auto;">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
        </div>
        <div class="upload-zone-text">Drag and drop an OCT B-scan image, or click to browse</div>
        <div class="upload-zone-sub">PNG or JPG, single B-scan image</div>
        <input type="file" id="file-input" accept="image/png,image/jpeg" style="display:none;">
    </div>

    <div id="loading-state" class="loading-state">
        <div class="loading-spinner"></div>
        <div class="loading-text">Running segmentation — first request may take up to a minute while the model wakes up.</div>
    </div>

    <div id="error-state" class="error-state"></div>

    <div id="result-container"></div>

    <div style="margin-top: 48px; padding-top: 24px; border-top: 1px solid #e5e7eb; font-size: 0.75rem; color: #9ca3af; line-height: 1.5;">
        This tool is a research prototype developed by the Paulus Lab at Johns Hopkins University.
        It is not FDA-cleared and should not be used to guide clinical decisions without physician
        oversight. Model code, training details, and full performance numbers:
        <a href="https://github.com/PaulusLab/rd-oct-segmentation-model" style="color: #0d9488; font-weight: 600;">github.com/PaulusLab/rd-oct-segmentation-model</a>.
    </div>
</div>

<script>
// Modal assigns this URL dynamically at deploy time (format:
// https://<modal-username>--rd-oct-segmentation-segment.modal.run) -- by the time
// this task runs, Task 6 has already deployed and confirmed the real URL. Use
// that exact recorded value here, not a guessed or generic one.
const MODAL_ENDPOINT_URL = "<paste the exact URL confirmed working in Task 6, Step 1>";

const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const loadingState = document.getElementById("loading-state");
const errorState = document.getElementById("error-state");
const resultContainer = document.getElementById("result-container");

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function validateFile(file) {
    const validTypes = ["image/png", "image/jpeg"];
    if (!validTypes.includes(file.type)) {
        return "Please upload a PNG or JPG image.";
    }
    if (file.size > 10 * 1024 * 1024) {
        return "Image is too large (max 10MB).";
    }
    return null;
}

async function handleFile(file) {
    errorState.classList.remove("active");
    resultContainer.innerHTML = "";

    const validationError = validateFile(file);
    if (validationError) {
        errorState.textContent = validationError;
        errorState.classList.add("active");
        return;
    }

    loadingState.classList.add("active");
    try {
        const result = await runSegmentation(file);
        loadingState.classList.remove("active");
        renderResult(file, result);  // defined in Task 8
    } catch (err) {
        loadingState.classList.remove("active");
        errorState.textContent = "Segmentation failed: " + err.message + ". Please try again.";
        errorState.classList.add("active");
    }
}

async function runSegmentation(imageFile) {
    const formData = new FormData();
    formData.append("image", imageFile);

    const response = await fetch(MODAL_ENDPOINT_URL, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();
    // data is { overlay_png_base64, stats } per modal_app.py's segment() return shape
    return {
        overlayImageUrl: "data:image/png;base64," + data.overlay_png_base64,
        stats: data.stats,
    };
}

// renderResult() is added in Task 8 -- placeholder so Task 7 is independently testable
function renderResult(originalFile, result) {
    resultContainer.innerHTML = "<p>Segmentation complete. (Interactive result view added in Task 8.)</p>";
    console.log("Result stats:", result.stats);
}
</script>

</body>
</html>
```

- [ ] **Step 2: Manual test — upload flow works end to end**

Open the file directly in a browser (`open tools/rd-oct-segmentation-tool.html`) or serve the repo
locally (`python3 -m http.server` from the repo root, then visit
`http://localhost:8000/tools/rd-oct-segmentation-tool.html`). Upload
`sample_bscan_00030.png` (copy it into this repo's `tools/` directory temporarily for local testing,
or use its absolute path in the file picker). Verify:
- Drag-and-drop and click-to-browse both work
- Uploading a non-image file (e.g. a `.txt`) shows the validation error, no API call made
- Uploading a valid image shows the loading spinner, then either the placeholder result text (success)
  or a clear error message (if the Space is asleep/unreachable) — check the browser console for the
  actual `stats` object logged, confirming the API call round-trips correctly

- [ ] **Step 3: Commit**

```bash
git add tools/rd-oct-segmentation-tool.html
git commit -m "feat: add tool page shell with upload UI and Modal API call"
git push
```

---

### Task 8: Interactive canvas overlay with hover tooltips and layer toggles

**Files:**
- Modify: `tools/rd-oct-segmentation-tool.html`

**Interfaces:**
- Consumes: `result.overlayImageUrl` and `result.stats` from Task 7's `runSegmentation()`.
- Produces: replaces the placeholder `renderResult()` with the real interactive canvas; produces no
  new interface for later tasks beyond the rendered DOM (`#result-canvas`, `#layer-toggles`).

- [ ] **Step 1: Add the result view HTML structure and canvas styles**

Add inside `<style>`, after `.error-state`:

```css
.result-view { display: grid; grid-template-columns: 1fr 280px; gap: 24px; margin-top: 24px; }
@media (max-width: 720px) { .result-view { grid-template-columns: 1fr; } }

.image-panel { background: #fff; border-radius: 14px; border: 1px solid #e5e7eb; padding: 20px; }
.canvas-wrap { position: relative; width: 100%; }
#result-canvas { width: 100%; height: auto; border-radius: 8px; cursor: crosshair; display: block; }

.layer-toggles { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.layer-toggle {
    display: flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 999px;
    border: 1.5px solid #e5e7eb; background: #fff; cursor: pointer; font-size: 0.78rem;
    font-weight: 600; color: #374151; transition: all 0.15s;
}
.layer-toggle.active { border-color: currentColor; background: color-mix(in srgb, currentColor 10%, white); }
.layer-toggle-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

.hover-tooltip {
    position: absolute; pointer-events: none; background: #111827; color: #fff;
    padding: 8px 12px; border-radius: 8px; font-size: 0.78rem; max-width: 220px;
    display: none; z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.hover-tooltip-title { font-weight: 700; margin-bottom: 2px; }
.hover-tooltip-desc { color: #d1d5db; line-height: 1.4; }

.download-btn {
    display: inline-flex; align-items: center; gap: 8px; margin-top: 16px;
    background: #0d9488; color: #fff; padding: 10px 18px; border-radius: 8px;
    font-size: 0.85rem; font-weight: 600; border: none; cursor: pointer; transition: opacity 0.15s;
}
.download-btn:hover { opacity: 0.88; }
```

Add before the closing `</script>` tag's containing structure — replace the `<div id="result-container">`
contents dynamically, so no static HTML needed here (built entirely in JS in Step 3).

- [ ] **Step 2: Add the biomarker metadata constant**

Add near the top of the `<script>` block, after `MODAL_ENDPOINT_URL`:

```javascript
const BIOMARKER_INFO = {
    SRF: {
        name: "Subretinal Fluid",
        color: "#ef4444",
        description: "Fluid that has leaked beneath the retina — a hallmark of active retinal detachment.",
        dice: 0.780,
        reliability: "High",
    },
    ORC: {
        name: "Outer Retinal Cysts",
        color: "#f59e0b",
        description: "Fluid-filled cystic spaces in the outer retina, associated with chronic detachment.",
        dice: 0.369,
        reliability: "Experimental",
    },
    IRC: {
        name: "Intraretinal Cysts",
        color: "#3b82f6",
        description: "Fluid-filled cystic spaces within the retina itself.",
        dice: 0.515,
        reliability: "Moderate",
    },
    ERM: {
        name: "Epiretinal Membrane",
        color: "#a855f7",
        description: "A thin fibrous layer on the retinal surface that can distort vision and affect surgical outcome.",
        dice: 0.348,
        reliability: "Experimental",
    },
};

const RELIABILITY_COLORS = { High: "#16a34a", Moderate: "#ca8a04", Experimental: "#dc2626" };
```

- [ ] **Step 3: Replace the placeholder `renderResult()` with the real interactive canvas**

Replace the entire `renderResult()` function from Task 7 with:

```javascript
let activeLayers = new Set(Object.keys(BIOMARKER_INFO));
let currentMaskImageData = null;  // ImageData of the raw overlay, sampled on hover

async function renderResult(originalFile, result) {
    const originalUrl = URL.createObjectURL(originalFile);

    resultContainer.innerHTML = `
        <div class="result-view">
            <div class="image-panel">
                <div class="canvas-wrap">
                    <canvas id="result-canvas"></canvas>
                    <div id="hover-tooltip" class="hover-tooltip"></div>
                </div>
                <div id="layer-toggles" class="layer-toggles"></div>
                <button id="download-btn" class="download-btn">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                    Download Annotated Image
                </button>
            </div>
            <div id="legend-panel"></div>
        </div>
    `;

    const canvas = document.getElementById("result-canvas");
    const ctx = canvas.getContext("2d");

    const [originalImg, overlayImg] = await Promise.all([
        loadImage(originalUrl),
        loadImage(result.overlayImageUrl),
    ]);

    canvas.width = overlayImg.width;
    canvas.height = overlayImg.height;

    // Draw original (cropped/padded space the model saw) then composite the overlay
    ctx.drawImage(originalImg, 0, 0, canvas.width, canvas.height);
    const overlayCanvas = document.createElement("canvas");
    overlayCanvas.width = overlayImg.width;
    overlayCanvas.height = overlayImg.height;
    overlayCanvas.getContext("2d").drawImage(overlayImg, 0, 0);
    currentMaskImageData = overlayCanvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);

    redrawCanvas(ctx, originalImg, overlayImg, canvas.width, canvas.height);
    buildLayerToggles();
    buildLegendPanel(result.stats);
    attachHoverHandler(canvas);
    attachDownloadHandler(canvas);
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

function redrawCanvas(ctx, originalImg, overlayImg, width, height) {
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(originalImg, 0, 0, width, height);
    // Only draw overlay pixels whose class is in activeLayers -- done by drawing the
    // full overlay image (browser handles per-pixel alpha from the RGBA PNG), then
    // masking out inactive classes via a color-key redraw would need per-pixel work;
    // simplest correct approach: re-render overlay from currentMaskImageData, zeroing
    // alpha for inactive classes' RGB before compositing.
    const filtered = ctx.createImageData(width, height);
    const classByColor = {};
    for (const [key, info] of Object.entries(BIOMARKER_INFO)) {
        classByColor[info.color] = key;
    }
    for (let i = 0; i < currentMaskImageData.data.length; i += 4) {
        const a = currentMaskImageData.data[i + 3];
        if (a === 0) continue;
        const hex = rgbToHex(currentMaskImageData.data[i], currentMaskImageData.data[i + 1], currentMaskImageData.data[i + 2]);
        const classKey = classByColor[hex];
        if (classKey && activeLayers.has(classKey)) {
            filtered.data[i] = currentMaskImageData.data[i];
            filtered.data[i + 1] = currentMaskImageData.data[i + 1];
            filtered.data[i + 2] = currentMaskImageData.data[i + 2];
            filtered.data[i + 3] = currentMaskImageData.data[i + 3];
        }
    }
    const tmp = document.createElement("canvas");
    tmp.width = width; tmp.height = height;
    tmp.getContext("2d").putImageData(filtered, 0, 0);
    ctx.drawImage(tmp, 0, 0);
}

function rgbToHex(r, g, b) {
    return "#" + [r, g, b].map(v => v.toString(16).padStart(2, "0")).join("");
}

function buildLayerToggles() {
    const container = document.getElementById("layer-toggles");
    container.innerHTML = Object.entries(BIOMARKER_INFO).map(([key, info]) => `
        <div class="layer-toggle active" data-key="${key}" style="color: ${info.color};">
            <span class="layer-toggle-dot"></span>${key}
        </div>
    `).join("");

    container.querySelectorAll(".layer-toggle").forEach(el => {
        el.addEventListener("click", () => {
            const key = el.dataset.key;
            if (activeLayers.has(key)) {
                activeLayers.delete(key);
                el.classList.remove("active");
            } else {
                activeLayers.add(key);
                el.classList.add("active");
            }
            const canvas = document.getElementById("result-canvas");
            const ctx = canvas.getContext("2d");
            // redraw needs the original+overlay images again -- cached on the canvas element
            redrawCanvas(ctx, canvas._originalImg, canvas._overlayImg, canvas.width, canvas.height);
        });
    });
}

function buildLegendPanel(stats) {
    const panel = document.getElementById("legend-panel");
    panel.innerHTML = `
        <div style="background: #fff; border-radius: 14px; border: 1px solid #e5e7eb; padding: 20px;">
            <p style="font-size: 0.72rem; font-weight: 700; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 14px;">Biomarker Legend</p>
            ${Object.entries(BIOMARKER_INFO).map(([key, info]) => `
                <div style="margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid #f3f4f6;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="width: 10px; height: 10px; border-radius: 50%; background: ${info.color}; flex-shrink: 0;"></span>
                        <span style="font-weight: 700; font-size: 0.85rem;">${key}</span>
                        <span style="margin-left: auto; font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; color: #fff; background: ${RELIABILITY_COLORS[info.reliability]};">${info.reliability}</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #6b7280; margin-bottom: 4px;">${info.name}</div>
                    <div style="font-size: 0.76rem; color: #374151; line-height: 1.4;">${info.description}</div>
                    <div style="font-size: 0.7rem; color: #9ca3af; margin-top: 4px;">Dice: ${info.dice} · ${(stats[key]?.pixel_pct ?? 0)}% of image</div>
                </div>
            `).join("")}
        </div>
    `;
}

function attachHoverHandler(canvas) {
    const tooltip = document.getElementById("hover-tooltip");
    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = Math.floor((e.clientX - rect.left) * scaleX);
        const y = Math.floor((e.clientY - rect.top) * scaleY);

        const idx = (y * canvas.width + x) * 4;
        const a = currentMaskImageData.data[idx + 3];
        if (a === 0) {
            tooltip.style.display = "none";
            return;
        }
        const hex = rgbToHex(currentMaskImageData.data[idx], currentMaskImageData.data[idx + 1], currentMaskImageData.data[idx + 2]);
        const classKey = Object.entries(BIOMARKER_INFO).find(([, info]) => info.color === hex)?.[0];
        if (!classKey || !activeLayers.has(classKey)) {
            tooltip.style.display = "none";
            return;
        }
        const info = BIOMARKER_INFO[classKey];
        tooltip.innerHTML = `<div class="hover-tooltip-title">${classKey} — ${info.name}</div><div class="hover-tooltip-desc">${info.description}</div>`;
        tooltip.style.left = (e.clientX - rect.left + 12) + "px";
        tooltip.style.top = (e.clientY - rect.top + 12) + "px";
        tooltip.style.display = "block";
    });
    canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
}
```

- [ ] **Step 4: Cache the loaded images on the canvas element for re-draws on toggle**

Inside `renderResult()`, right after `canvas.height = overlayImg.height;`, add:

```javascript
    canvas._originalImg = originalImg;
    canvas._overlayImg = overlayImg;
```

- [ ] **Step 5: Manual test — interactivity works**

Reload the page, upload the fixture image, and verify: each of the 4 layer toggles independently
shows/hides its color region on the canvas; hovering over a colored region shows a tooltip with the
correct biomarker name and description at the cursor position; hovering over an unhighlighted
(background) area or a toggled-off region shows no tooltip.

- [ ] **Step 6: Commit**

```bash
git add tools/rd-oct-segmentation-tool.html
git commit -m "feat: add interactive canvas overlay with hover tooltips and layer toggles"
git push
```

---

### Task 9: Download button

**Files:**
- Modify: `tools/rd-oct-segmentation-tool.html`

**Interfaces:**
- Consumes: `#result-canvas` (from Task 8) — the currently-rendered canvas state, including whichever
  layers are toggled on.
- Produces: nothing consumed by later tasks (terminal UI feature).

- [ ] **Step 1: Add the download handler function**

Add after `attachHoverHandler()`:

```javascript
function attachDownloadHandler(canvas) {
    document.getElementById("download-btn").addEventListener("click", () => {
        const link = document.createElement("a");
        link.download = "oct-segmentation-result.png";
        link.href = canvas.toDataURL("image/png");
        link.click();
    });
}
```

(Already wired up via the `attachDownloadHandler(canvas)` call added in Task 8 Step 3's
`renderResult()` — this step only adds the function body.)

- [ ] **Step 2: Manual test**

Upload the fixture image, toggle off one biomarker layer, click "Download Annotated Image", and open
the downloaded file. Verify: the downloaded PNG shows the original image with only the still-toggled-on
layers overlaid — confirming the download captures current view state, not a fixed original render.

- [ ] **Step 3: Commit**

```bash
git add tools/rd-oct-segmentation-tool.html
git commit -m "feat: add download button for annotated result image"
git push
```

---

### Task 10: Add the tool to the Tools listing pages

**Files:**
- Modify: `tools/index.html`
- Create: `_projects/4_project.md`

**Interfaces:** none (pure content/linking task).

- [ ] **Step 1: Add a third card to `tools/index.html`**

Add these CSS rules inside the existing `<style>` block, alongside the `.rds`/`.amd` variants:

```css
.tool-card.oct:hover { border-color: #a855f7; }
.tool-icon.oct { background: #f3e8ff; }
.tool-badge.oct { background: #f3e8ff; color: #7e22ce; }
.dot-oct { background: #a855f7; }
.tool-cta.oct { background: #7e22ce; color: #fff; }
```

Add this card markup inside the `.grid`, after the AMD Tool `</a>` and before the closing `</div>`:

```html
        <!-- RD OCT Segmentation Tool -->
        <a href="rd-oct-segmentation-tool.html" class="tool-card oct">
            <div class="tool-icon oct">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#7e22ce" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>
            </div>
            <div class="tool-badge oct">
                <svg width="8" height="8" viewBox="0 0 10 10"><circle cx="5" cy="5" r="5" fill="#7e22ce"/></svg>
                Retinal Detachment
            </div>
            <div class="tool-title">RD OCT Segmentation<br>Tool</div>
            <div class="tool-desc">Upload an OCT B-scan and see it segmented for subretinal fluid, outer/intraretinal cysts, and epiretinal membrane — with an interactive, hover-explorable overlay.</div>
            <div class="tool-features">
                <div class="tool-feature"><span class="tool-feature-dot dot-oct"></span>Interactive hover overlay explains each finding</div>
                <div class="tool-feature"><span class="tool-feature-dot dot-oct"></span>Honest per-class reliability, not a black box</div>
                <div class="tool-feature"><span class="tool-feature-dot dot-oct"></span>Download the annotated result</div>
            </div>
            <span class="tool-cta oct">Open Tool <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg></span>
        </a>
```

- [ ] **Step 2: Add the Jekyll `_projects` entry (feeds the nav-linked `/projects/` listing page)**

```markdown
---
layout: page
title: RD OCT Segmentation Tool
description: Interactive biomarker segmentation for retinal detachment OCT images
importance: 4
related_publications: false
---

The RD OCT Segmentation Tool segments retinal detachment biomarkers from a single OCT B-scan image
using a SegFormer-B4 model trained by the Paulus Lab: subretinal fluid (SRF), outer retinal cysts
(ORC), intraretinal cysts (IRC), and epiretinal membrane (ERM).

Key features include:

- **Interactive hover overlay** — hover over any highlighted region to see what it means, in plain language.
- **Honest reliability reporting** — each biomarker's real measured accuracy is shown, not hidden behind a single blanket disclaimer.
- **Downloadable results** — export the annotated image directly from the browser.

Model code, training details, and full performance numbers: [github.com/PaulusLab/rd-oct-segmentation-model](https://github.com/PaulusLab/rd-oct-segmentation-model)

[Try the tool →](https://pauluslab.github.io/tools/rd-oct-segmentation-tool.html)
```

- [ ] **Step 3: Manual test**

Rebuild the Jekyll site locally (`bundle exec jekyll serve`, per this repo's existing `Gemfile`) and
visit `/tools/` and `/projects/` — confirm the new card appears in both listings and both links
resolve to the new tool page.

- [ ] **Step 4: Commit**

```bash
git add tools/index.html _projects/4_project.md
git commit -m "feat: list RD OCT Segmentation Tool in Tools index and projects page"
git push
```

---

### Task 11: Final manual QA pass and PR

**Files:** none (verification + PR creation).

- [ ] **Step 1: Full end-to-end walkthrough**

With the Space live (Task 6 confirmed) and all of Phase 3 committed, do a complete fresh walkthrough
as a real visitor would: navigate from the homepage nav → Tools → RD OCT Segmentation Tool card →
upload `sample_bscan_00030.png` → confirm overlay, hover tooltips, all 4 toggles, legend numbers
matching the model repo README exactly, and download all work.

- [ ] **Step 2: Cross-check performance numbers are consistent everywhere**

Confirm SRF=0.780/High, IRC=0.515/Moderate, ORC=0.369/Experimental, ERM=0.348/Experimental appear
identically in: model repo README (Task 4), `modal_app.py`'s `RELIABILITY` dict (Task 5), and the
tool page's `BIOMARKER_INFO` constant (Task 8). Any mismatch is a bug — fix before opening the PR.

- [ ] **Step 3: Open the PR**

```bash
cd ~/Desktop/Projects/PaulusLab.github.io
gh pr create --title "Add RD OCT Segmentation Tool" --body "$(cat <<'EOF'
## Summary
- New public tool: upload an OCT B-scan, get an interactive SRF/ORC/IRC/ERM segmentation overlay
- Backed by a SegFormer-B4 model (fold 4, best of 5 CV folds) served via a Modal inference endpoint
- Model code + checkpoint versioned separately at github.com/PaulusLab/rd-oct-segmentation-model
- Honest per-class reliability shown throughout (SRF high, IRC moderate, ORC/ERM experimental)

## Test plan
- [x] Uploaded a real B-scan, confirmed overlay/hover/toggles/download all work
- [x] Cross-checked Dice numbers match across model README, Space, and tool page
- [ ] Lab review of biomarker explanation wording for clinical accuracy (flagged in design spec)
- [ ] Lab decision on model repo license (flagged in design spec)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Do not merge — leave the PR open for the lab to review, per the two open items flagged in the design
spec (license, clinical-accuracy wording pass).

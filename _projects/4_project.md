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

---
name: chest-xray-triage
title: Chest radiograph findings and optional segmentation
summary: Classify a chest X-ray with TorchXRayVision, sanity-check the image, report calibrated probabilities honestly, and segment a finding with a box prompt when localisation is needed.
tags: [xray, chest, classification, 2d]
modalities: [XR]
tasks: [classification, segmentation]
models: [torchxrayvision, medsam2, sam2]
version: 1
---
## 1. Inspect
`inspect_image(cxr)` - the image must be a single frontal radiograph (PA/AP). Check bit depth
and inversion: bones should be bright, air dark. Rotated or lateral images give meaningless outputs.

## 2. Classify
`classify_image(cxr, model="torchxrayvision")` -> probabilities for 18 findings and a bar chart.
Default weights `densenet121-res224-all` (trained on eight datasets); `resnet50-res512-all` sees
more detail at 512 px.

## 3. Interpret carefully
* Probabilities are calibrated model outputs, not diagnoses; report the top findings with their
  values and say that they need radiologist confirmation.
* Findings above 0.5 are "flagged"; between 0.3 and 0.5 "possible"; do not list every low value.
* Consistency checks: cardiomegaly with effusion/edema is plausible; pneumothorax with a high
  emphysema probability may reflect hyperlucency - say so.

## 4. Localise (optional)
For a finding that needs an outline (mass, effusion), `render_view(cxr, grid=True)`, read a box
from the grid, then `segment(cxr, model="medsam2", prompts=[{"type": "box", "coords": [x0, y0, x1, y1]}])`
and `mask_stats(mask)` for area in pixels/mm^2 when spacing is known.

## 5. Report
Image path, model weights, top findings with probabilities, the chart, any localisation, and the
disclaimer.

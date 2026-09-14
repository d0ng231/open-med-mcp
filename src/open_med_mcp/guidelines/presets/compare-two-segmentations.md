---
name: compare-two-segmentations
title: Compare two segmentations of the same image
summary: Compare a prediction against a reference (or two methods against each other) with overlap and surface metrics and a visual difference map.
tags: [comparison, metrics, dice, validation]
modalities: [any]
tasks: [comparison, qc]
version: 1
---
1. Both masks must share the image geometry (`inspect_image` on each; same `size_xyz` and spacing).
   Resample or re-export otherwise - do not compare masks of different shapes.
2. `compare_masks(pred, ref, image=img)` -> Dice, IoU, precision, recall, volume difference, HD95, ASSD
   (per label and for the foreground). Treat mask **a** as the prediction and **b** as the reference.
3. `render_view(img, masks=[pred, ref], mode="contour")` - two contour colors make disagreement
   obvious; add `layout="montage"` for the extent of the disagreement along the stack.
4. Summarize: where the disagreement is (poles, boundary, extra components), its size in mL and
   whether it changes the downstream measurement.

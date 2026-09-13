---
name: qc-checklist
title: Quality-control checklist for a segmentation
summary: The minimum set of checks before a mask is accepted - visual, geometric, intensity and consistency checks with concrete tool calls.
tags: [qc, checklist, validation]
modalities: [any]
tasks: [segmentation, qc]
version: 1
---
Run through every item and record the outcome in the report.

**Visual**
- [ ] Three-plane view at the mask centroid looks anatomically plausible (`render_view(layout="three-plane")`).
- [ ] Montage of axial slices shows no leakage into neighbouring structures and no missing slices.
- [ ] First and last slices of the mask are sensible (no overshoot at the poles).

**Geometric** (`mask_stats`)
- [ ] `components` is 1 for solid organs (else run `largest_component`).
- [ ] No holes (`fill_holes`), unless the structure is genuinely hollow.
- [ ] Volume is within a plausible range for the structure and patient.
- [ ] Bounding box does not touch the image border unexpectedly.

**Intensity** (`mask_stats(mask, image=...)`)
- [ ] Mean/median intensity is consistent with the tissue (CT: HU; MRI: relative to neighbours).
- [ ] Standard deviation is not suspiciously large (mixture of tissues).

**Consistency**
- [ ] If two methods were run, `compare_masks` Dice > 0.8 for large organs (lower is acceptable
      for small lesions, but explain the difference).
- [ ] Post-processing steps are listed in the report.

**Provenance**
- [ ] Model name, variant/checkpoint, parameters, prompts and job directory are recorded.

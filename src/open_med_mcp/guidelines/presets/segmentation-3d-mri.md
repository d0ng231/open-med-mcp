---
name: segmentation-3d-mri
title: Segment a structure in a 3D MRI volume
summary: MRI-specific notes - intensity is not calibrated, choose windows from percentiles, prefer MedSAM2 with boxes or TotalSegmentator total_mr, QC per sequence.
tags: [segmentation, mri, mr, 3d]
modalities: [MR]
tasks: [segmentation]
models: [medsam2, totalsegmentator, classical]
version: 1
---
## Key differences from CT
* Intensities are arbitrary: always use `window: "auto"` (robust percentiles) or explicit percentiles.
  Thresholds in `classical` must be derived from `inspect_image` statistics of *this* image.
* Bias fields and coil profiles make a single global threshold unreliable - use `region_grow` with a
  seed and `tolerance`, or a promptable model.
* Check the acquisition plane: many MRI series are thick-slice 2D acquisitions. Propagate along the
  acquisition plane (usually the plane with the smallest number of slices) - pass it as `plane`.

## Protocol
1. `inspect_image(mri)` -> spacing, planes, percentiles.
2. Anatomy: `run_model("totalsegmentator", {"image": mri}, {"task": "total_mr"})` (or
   `roi_subset`). Lesions: `segment(mri, model="medsam2", prompts=[box], plane=<acquisition plane>)`
   (`medsam2_mri_liver_lesion` for liver lesions).
3. QC in three planes; MRI masks often need `fill_holes` and `remove_small`.
4. Report volumes in mL and the sequence/series description.

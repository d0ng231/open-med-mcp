---
name: brain-mri-preprocessing
title: Brain MRI preprocessing - reorient, bias correction, skull stripping
summary: Standard brain MRI preparation - check orientation, N4 bias correction, skull stripping with HD-BET or SynthStrip, QC of the brain mask, and hand-off to downstream models.
tags: [mri, brain, preprocessing, skull-stripping, n4]
modalities: [MR]
tasks: [preprocessing]
models: [hdbet, synthstrip, medsam2]
version: 1
---
## 1. Inspect
`inspect_image(t1)` - note orientation (e.g. RAS/LPS), spacing (anisotropic clinical scans are
common) and whether intensities look bias-affected (bright center, dark periphery in `render_view`).

## 2. Reorient (optional)
Most tools handle any orientation, but a canonical frame simplifies reporting:
`reorient_image(t1, orientation="RAS")`.

## 3. Bias field correction
`n4_bias_correction(t1)` when a smooth intensity gradient is visible. Re-render and compare with
the original (`render_view(..., window="auto")`). Skip for already corrected research data.

## 4. Skull stripping
| situation | tool |
|---|---|
| standard adult T1/T2/FLAIR | `run_model("hdbet", {"image": t1})` (GPU) |
| unusual contrast, CT, infant, PET, or no GPU | `run_model("synthstrip", {"image": t1})` (official FreeSurfer container, CPU) |
Both return `mask` (binary brain) and `brain` (stripped image).

## 5. QC the mask
`render_view(t1, masks=[mask], layout="three-plane", mode="contour")` and a `montage`. Check the
inferior frontal/temporal lobes and cerebellum (typical under-segmentation), eyes and dura
(typical leakage). `mask_stats(mask)` - adult intracranial volume is roughly 1200-1700 mL;
`components` should be 1. If two tools disagree, `compare_masks(hdbet_mask, synthstrip_mask)`.

## 6. Hand-off
Use the stripped image (`brain`) for registration (`register_images`), lesion segmentation
(`segment(..., model="medsam2")`) or nnU-Net/MONAI models. Record every step and run directory in
the report.

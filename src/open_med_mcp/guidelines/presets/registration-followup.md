---
name: registration-followup
title: Longitudinal follow-up - register, propagate a mask, compare volumes
summary: Align a follow-up scan to a baseline (rigid/affine/deformable), propagate the baseline mask, re-segment on the follow-up, and report volume change with a visual check.
tags: [registration, longitudinal, follow-up, comparison]
modalities: [CT, MR]
tasks: [registration, comparison]
models: [totalsegmentator, medsam2, hdbet]
version: 1
---
1. `inspect_image` both scans - same modality? similar spacing? For brain MRI skull-strip both
   first (`hdbet`/`synthstrip`) so registration is driven by brain, not scalp.
2. `register_images(fixed=baseline, moving=followup, transform="rigid")` for same-patient scans;
   `affine` if the field of view or scanner differs; `bspline` only for soft-tissue deformation and
   after a rigid check. Use `metric="mattes"` across modalities. Look at the returned overlay.
3. Propagate: `apply_transform(transform_file, moving=<followup mask>, reference=baseline, is_mask=True)`
   or pass `moving_masks=[...]` to `register_images` directly.
4. Independent re-segmentation on the follow-up (same model and parameters as baseline) is the
   more honest measurement; use the propagated mask as a prompt (`mask_to_prompts`) or as a QC reference.
5. `compare_masks(baseline_mask, propagated_or_new_mask, image=baseline)` -> volume difference in mL
   and %, Dice, HD95. `mask_stats` on both for absolute volumes.
6. Report: transform type, final metric value, both volumes, change, and the overlay figures.
   State that volume changes below the inter-method variability (compare_masks between two
   methods on the same scan) are not meaningful.

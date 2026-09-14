---
name: segmentation-3d-ct
title: Segment an organ or lesion in a 3D CT volume
summary: End-to-end protocol for 3D CT segmentation - inspect, choose automatic (TotalSegmentator) vs promptable (MedSAM2) models, QC in three planes, measure volume, report.
tags: [segmentation, ct, 3d, organ, lesion, volume]
modalities: [CT]
tasks: [segmentation]
models: [totalsegmentator, voxtell, medsam2, classical]
version: 1
---
## 1. Inspect
* `inspect_image(ct)` - confirm `modality_guess == CT` (HU range roughly -1000 .. 3000). Note the
  spacing (anisotropic stacks with slice thickness > 3 mm propagate less reliably) and the number of
  axial slices.
* If the image is a DICOM folder, the tool converts it; keep the reported `series_description`.

## 2. Choose the strategy
| target | first choice | fallback |
|---|---|---|
| named anatomy (liver, spleen, kidneys, vertebrae, aorta, lungs ...) | `totalsegmentator` with `roi_subset` | `voxtell` with a text prompt ("left kidney"), or `medsam2` with a box |
| focal lesion / tumour / nodule / cyst | `medsam2` (variant `medsam2_ct_lesion` for lesions) with a box on the slice where it is largest | `voxtell` with a descriptive prompt ("liver tumor"); `classical` region growing with a seed |
| structure without a TotalSegmentator label (e.g. "gallbladder stone", "pancreatic duct") | `voxtell` text prompt, then QC | `medsam2` box |
| air / bone / high-contrast region | `classical` threshold (lung: -1000..-500 HU, bone: > 250 HU) | - |

Use `fast: true` for TotalSegmentator on CPU or for a quick first pass; rerun at full resolution for
final numbers.

## 3. Segment
* Automatic: `run_model("totalsegmentator", {"image": ct}, {"roi_subset": ["liver"], "fast": false})`.
  The response lists label ids -> names; use `postprocess_mask(mask, [{"op": "keep_labels", "labels": [id]}])`
  to isolate one structure.
* Promptable: locate the target first (render the middle slices with a `soft-tissue` window, or use
  `mask_to_prompts` on a coarse mask), then
  `segment(ct, model="medsam2", prompts=[{"type": "box", "coords": [x0, y0, x1, y1], "slice": z}], plane="axial")`.
  A box should enclose the structure with a 2-5 voxel margin. Prefer a box over single points.
  For long structures set `max_slices` to a plausible extent to avoid drift.
* Text: `segment(ct, model="voxtell", prompts=[{"type": "text", "text": "liver"}, {"type": "text", "text": "spleen"}])`.
  One prompt per structure, specific wording ("right kidney"), and always inspect the preview - an
  unknown concept can return an empty or wrong mask (the response warns about empty prompts).

## 4. Quality control (mandatory)
1. `render_view(ct, masks=[mask], layout="three-plane", window="soft-tissue")` - is the shape anatomically
   plausible in all three planes? Does the mask leak into adjacent organs, vessels or the body wall?
2. `render_view(..., layout="montage", plane="axial")` - check the first and last slices of the mask:
   propagation often overshoots at the poles. Restrict with `keep_slices` or `max_slices` if needed.
3. `mask_stats(mask, image=ct)` - compare the mean HU with expectations (liver ~50-70 HU on portal
   venous phase, lung -700 HU, bone > 250 HU); a volume far from population norms is a red flag.
4. Apply `postprocess_mask` with `largest_component` + `fill_holes` for solid organs; do NOT use
   `largest_component` for multi-focal targets.

## 5. Report
Include: image path, geometry, model + parameters + variant, job directory, post-processing steps,
volume in mL, mean/SD HU, the QC figures, and an explicit statement of any residual concerns.
Use `write_report(...)`.

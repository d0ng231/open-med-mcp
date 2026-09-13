---
name: lung-ct-analysis
title: Lung CT - lungs, lobes, volumes and emphysema index
summary: Quantitative lung CT - lungmask lungs/lobes, per-lobe volumes, LAA% emphysema index, optional vessel/airway masks with TotalSegmentator, and a montage QC.
tags: [ct, lung, lobes, emphysema, quantification]
modalities: [CT]
tasks: [segmentation, quantification]
models: [lungmask, totalsegmentator, classical]
version: 1
---
1. `inspect_image(ct)` - CT in HU (min near -1000). Note slice thickness; thick slices (> 3 mm)
   bias LAA%.
2. Lungs: `run_model("lungmask", {"image": ct})` (labels 1 = right, 2 = left).
   Lobes: `run_model("lungmask", {"image": ct}, task="lobes")` (five lobes).
   The response carries `low_attenuation_area.per_label.laa_percent` (voxels < -950 HU) and mean HU.
3. QC: `render_view(ct, masks=[mask], layout="montage", window="lung", n_slices=12)`. Check that the
   trachea/main bronchi are excluded, that dense consolidations are included (R231 is trained for
   that), and that the fissures look plausible in the lobe map.
4. Optional structures: `run_model("totalsegmentator", {"image": ct}, task="lung_vessels")`
   (vessels + airways) or `roi_subset` with `lung_trachea_bronchia`.
5. Numbers: `mask_stats(mask, image=ct)` -> volume (mL) and mean HU per label; combine lobes to
   lungs with `combine_masks(..., mode="union")` if needed. Report LAA% per lung/lobe; typical
   emphysema threshold: LAA-950 > 6-10 % is abnormal (state the threshold you used).
6. `write_report` with the table, montage and run directories.

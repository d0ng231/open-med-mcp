---
name: multi-organ-ct-report
title: Multi-organ CT volumetry report
summary: Produce a volumetry table for several organs from one CT with TotalSegmentator, with per-organ QC figures and a Markdown/HTML report.
tags: [segmentation, ct, volumetry, report, anatomy]
modalities: [CT]
tasks: [segmentation, report]
models: [totalsegmentator]
version: 1
---
1. `inspect_image(ct)`; abort with an explanation if the image is not a CT or covers < 5 cm.
2. `run_model("totalsegmentator", {"image": ct}, {"roi_subset": [<organs>], "fast": false})`.
   Common sets: abdomen = `[liver, spleen, kidney_left, kidney_right, pancreas, gallbladder, stomach]`;
   thorax = `[lung_upper_lobe_left, lung_lower_lobe_left, lung_upper_lobe_right, lung_middle_lobe_right, lung_lower_lobe_right, heart, aorta]`.
3. `mask_stats(mask, image=ct)` -> table of label, name, volume_ml, mean HU, components.
4. For each organ render one three-plane figure (`render_view(..., labels=[id])`).
5. Flag: organs with `components > 1`, volumes outside typical ranges, mean HU outside the
   expected tissue range, and structures truncated by the field of view (bbox touching the border).
6. `write_report(...)` with the table, figures, flags and provenance.

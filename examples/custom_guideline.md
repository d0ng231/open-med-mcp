---
name: segmentation-3d-ct
title: Segment an organ in CT (lab protocol v2)
summary: Our lab's variant of the CT protocol - always run TotalSegmentator at full resolution and report HU statistics.
tags: [segmentation, ct, 3d, lab]
modalities: [CT]
tasks: [segmentation]
models: [totalsegmentator, medsam2]
version: 2
---
Copy this file to `<workspace>/omm_guidelines/segmentation-3d-ct.md` to override the preset with
the same name. Everything below is what the agent will read.

1. `inspect_image` and abort if spacing along z is > 5 mm (report why).
2. `run_model("totalsegmentator", ..., {"fast": false, "roi_subset": [<organ>]})`.
3. `render_view(layout="three-plane")` and `render_view(layout="montage", n_slices=12)`; describe
   any leakage or truncation you see.
4. `mask_stats(mask, image)`: report volume (mL), mean and SD HU, number of components.
5. If components > 1: `postprocess_mask(mask, ["largest_component", "fill_holes"])` and re-check.
6. `write_report` with the figures, numbers and the run directory.

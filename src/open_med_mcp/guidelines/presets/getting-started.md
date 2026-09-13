---
name: getting-started
title: Getting started - how to work with open-med-mcp
summary: Orientation for agents - what the tools are, the coordinate conventions, and the standard loop (inspect -> segment -> look -> refine -> measure -> report).
tags: [overview, workflow]
modalities: [any]
tasks: [any]
version: 1
---
You are working with **open-med-mcp**, a tool server for medical image analysis. Everything runs
locally on the user's machine; images never leave it. Follow this loop for any task:

1. **Understand the data** - `inspect_image(path)` returns geometry (size, spacing, orientation),
   intensity statistics, a modality guess and a preview. Do this before choosing a model or a window.
2. **Pick a guideline** - `list_guidelines()` then `get_guideline(name)` for the closest protocol
   (e.g. `segmentation-3d-ct`, `segmentation-2d-prompted`). Follow it step by step; it encodes what
   experts check.
3. **Pick a model** - `list_models()` / `describe_model(name)`. Automatic models (`totalsegmentator`)
   need no prompts; promptable models (`medsam2`, `sam2`) need a box or points; `classical` is a fast
   baseline for high-contrast structures (air, bone, lungs) and for creating seed boxes.
4. **Segment** - `run_model(...)` or the convenience wrapper `segment(...)`. Every run writes a job
   directory under `omm_outputs/` with `request.json`, `response.json`, `log.txt` and the outputs.
5. **Look at the result** - `render_view(...)` returns a PNG you can see. Always inspect at least the
   slice with the largest mask area and the first/last slices where the mask exists. Check for
   leakage into neighbouring structures, missing slices and holes.
6. **Refine** - add negative points, tighten the box, restrict `max_slices`, or apply
   `postprocess_mask` (largest component, fill holes, remove small islands). Re-render after each change.
7. **Measure** - `mask_stats(mask, image)` gives volumes (mL), bounding boxes and intensity stats;
   `compare_masks(a, b)` gives Dice/IoU/HD95 between two masks.
8. **Report** - `write_report(...)` assembles the figures, numbers and provenance into Markdown + HTML.

## Coordinate conventions (important)

* All voxel coordinates are **native index coordinates** `(x, y, z)` in ITK order, exactly as
  reported by `inspect_image` (`size_xyz`). No re-orientation is applied.
* Rendered views draw tick labels in these coordinates, so you can read prompt positions off a
  picture. `mask_to_prompts(mask)` turns any mask into a box/point prompt for a promptable model.
* Points: `{"type": "point", "coords": [x, y, z], "label": 1}` (label 0 = background/exclude).
* Boxes: `{"type": "box", "coords": [x0, y0, x1, y1], "slice": z}` or `[x0, y0, z0, x1, y1, z1]`.
* Planes: `axial` (through z for standard volumes), `coronal` (through y), `sagittal` (through x);
  the exact array axis is derived from the image orientation and shown by `inspect_image`.

## Safety and honesty

This software is for research. Never present a model output as a clinical finding; report what was
measured, on which image, with which model and parameters, and what the visual checks showed.
If a result looks wrong, say so and describe how you tried to fix it.

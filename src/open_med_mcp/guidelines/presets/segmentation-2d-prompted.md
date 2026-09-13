---
name: segmentation-2d-prompted
title: Segment a structure in a 2D image with point or box prompts
summary: Protocol for 2D images (X-ray, ultrasound frame, dermoscopy, endoscopy, histology tile, single slice) with SAM 2 / MedSAM2 - place prompts, verify, refine with negative points.
tags: [segmentation, 2d, prompt, sam, xray, ultrasound, dermoscopy, endoscopy]
modalities: [XR, US, RGB, any]
tasks: [segmentation]
models: [sam2, medsam2, classical]
version: 1
---
## 1. Inspect
`inspect_image(path)` - note `size_xyz` (width, height), whether the image is RGB, and the intensity
range. For 16-bit radiographs choose a window (`auto` is usually fine).

## 2. Locate the target
Render with a coordinate grid: `render_view(path, grid=True)`. Read the approximate bounding box of
the target from the tick labels (x = columns, y = rows, origin top-left).

## 3. Segment
`segment(path, model="medsam2", prompts=[{"type": "box", "coords": [x0, y0, x1, y1]}])`
(`sam2` for natural/RGB images). A box that tightly encloses the target works best; add
`{"type": "point", "coords": [x, y], "label": 0}` to push the mask away from a region it wrongly
included, and positive points to pull in missed parts. Set `multimask: true` in `params` when a
single ambiguous point is the only prompt.

## 4. Verify
`render_view(path, masks=[mask], mode="contour")` - does the contour follow the boundary? Iterate at
most 3-4 times; if it does not converge, report the best result and the failure mode.

## 5. Measure and report
`mask_stats(mask, image=path)` gives area in pixels (and mm^2 when spacing is known), bounding box
and centroid. Report the prompts you used so the result is reproducible.

# Conventions

## Coordinates
* **Native voxel index coordinates** `(x, y, z)` in ITK order are used everywhere: prompts, boxes,
  centroids, bounding boxes, slice indices. They are exactly what `inspect_image` reports as
  `size_xyz` and what SimpleITK/ITK use. The NumPy array behind a 3D image is `array[z, y, x]`
  (2D: `array[y, x]`), but you never need to think about that unless you write an adapter.
* No image is ever re-oriented or resampled. Anatomical planes are mapped to native axes from the
  direction cosines: `inspect_image(...).planes` tells you which index axis each plane is stacked
  along (`axial` -> `z` for standard volumes) and how many slices it has.
* Rendered views (`render_view`) apply display flips only, so anatomy appears in radiological
  convention (patient left on screen right, anterior at the top). Tick labels are native voxel
  indices - read prompt coordinates directly off the picture. The `panels` field of the result says
  which native axis is on screen x / y and whether it is flipped.
* Bounding boxes are inclusive: `[x0, y0, z0, x1, y1, z1]` (3D) or `[x0, y0, x1, y1]` (2D).

## Prompts
```json
{"type": "point", "coords": [x, y, z], "label": 1, "object_id": 1}
{"type": "point", "coords": [x, y], "slice": z, "label": 0}
{"type": "box",   "coords": [x0, y0, x1, y1], "slice": z}
{"type": "box",   "coords": [x0, y0, z0, x1, y1, z1]}
```
`label` 1 = include, 0 = exclude (points only). `object_id` groups prompts of one object; the output
label map uses the object id as label value. For 3D images a 2D prompt needs `slice`, interpreted
along the `plane` given to `segment` (default `axial`). Boxes work best; add negative points to
remove leakage.

## Masks
* Label maps are integer images with `0` = background. 3D masks are saved as `.nii.gz` with the
  geometry of the image; 2D masks as `.png` label maps. A sidecar `<mask>.labels.json` maps label
  ids to names when known (TotalSegmentator, multi-object SAM runs).
* Volumes are reported in mL (`mm^3 / 1000`) using the image spacing.

## Units and windows
* CT intensities are Hounsfield units; presets: `soft-tissue` (C40/W400), `lung` (C-600/W1500),
  `bone` (C400/W1800), `brain` (C40/W80), `liver` (C60/W160), `abdomen`, `mediastinum`, `angio`.
* MRI/other: `auto` = robust 0.5-99.5 percentile window; or `{"lower": a, "upper": b}`.

## Runs and provenance
* Every model run lives in `omm_outputs/<timestamp>-<model>-<id>/` with `request.json`,
  `inputs/`, `outputs/`, `response.json` and `log.txt`. `omm_outputs/provenance.jsonl` records every
  tool call that produced files. Quote the run directory in reports.

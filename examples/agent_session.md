# An annotated end-to-end session

Task given to the agent (Claude Code with open-med-mcp registered, workspace = a folder holding
`example_ct_sm.nii.gz`, the small public abdominal CT from `examples/get_sample_data.sh`):

> Segment the liver in `example_ct_sm.nii.gz`, verify the result visually, compare an automatic
> and a promptable model, and report the volume.

Below, each step lists the tool call the agent made and what came back (abridged). The numbers are
from a real run on an L40S GPU with the local backend.

---

**1. Conventions and data**

```text
get_conventions()
inspect_image("example_ct_sm.nii.gz")
```
```json
{"size_xyz": [122, 101, 30], "spacing_xyz_mm": [3.0, 3.0, 3.0], "orientation": "RAS",
 "modality_guess": "CT", "intensity": {"min": -1100, "max": 1207, "median": -52.0},
 "planes": {"axial": {"index_axis": 2, "n_slices": 30}, "coronal": {"index_axis": 1, "n_slices": 101}, "sagittal": {"index_axis": 0, "n_slices": 122}},
 "hints": ["CT in Hounsfield units: use window presets soft-tissue/lung/bone; totalsegmentator gives anatomy without prompts."]}
```
plus a three-plane preview PNG (`docs/assets/screenshots/inspect_preview.png`).

**2. Protocol**

```text
get_guideline("segmentation-3d-ct")
```
The guideline says: named anatomy -> TotalSegmentator first, MedSAM2 with a box to refine or
cross-check; QC in three planes and on the first/last slices; report volume + HU statistics.

**3. Automatic model**

```text
run_model("totalsegmentator", {"image": "example_ct_sm.nii.gz"}, params={"roi_subset": ["liver", "spleen", "kidney_left", "kidney_right"], "fast": true})
```
```json
{"runner": "local", "device": "cuda", "run_dir": "omm_outputs/20260913-053118-totalsegmentator-1f2a",
 "outputs": {"mask": "omm_outputs/20260913-053118-totalsegmentator-1f2a/outputs/mask.nii.gz"},
 "labels": {"1": "spleen", "2": "kidney_right", "3": "kidney_left", "5": "liver"},
 "mask_stats": {"labels": [{"label": 5, "name": "liver", "volume_ml": 1062.1, "components": 1, "intensity": {"mean": 71.4}}, "..."]}}
```
The agent isolates the liver: `postprocess_mask(mask, [{"op": "keep_labels", "labels": [5]}, "binarize"], output="liver_ts.nii.gz")`.

**4. Promptable model seeded from the automatic result**

```text
mask_to_prompts("liver_ts.nii.gz", plane="axial")
```
```json
{"slice": 24, "boxes": [[40, 25, 24, 106, 83, 24]], "points": [[77.7, 57.6, 24.0]], "slice_range": [0, 29]}
```
```text
segment("example_ct_sm.nii.gz", model="medsam2", prompts=[{"type": "box", "coords": [40, 25, 24, 106, 83, 24]}], params={"window": "soft-tissue", "largest_component": true})
```
```json
{"model": "medsam2", "timing": {"model_loaded": 2.2, "propagated": 3.1, "total_s": 3.4},
 "mask_stats": {"labels": [{"label": 1, "volume_ml": 1069.7, "components": 1}]}}
```
plus the preview (`docs/assets/screenshots/segment_medsam2_liver.png`). The agent looks at it:
smooth liver contour in all three planes, no leakage into the heart or stomach.

**5. Cross-check**

```text
compare_masks("liver_ts.nii.gz", "<medsam2 mask>", image="example_ct_sm.nii.gz")
```
```json
{"foreground": {"dice": 0.963, "iou": 0.928, "precision": 0.959, "recall": 0.966,
                "volume_diff_ml": 7.5, "hd95_mm": 4.2, "assd_mm": 1.01}}
```
(`docs/assets/screenshots/compare_masks.png`). A Dice of 0.96 and an HD95 of 4 mm (about one voxel) between
two independent methods is strong agreement; the residual differences sit at the inferior tip.

**6. Report**

```text
write_report("Liver volumetry - example_ct_sm", sections=[
  {"heading": "Method", "text": "TotalSegmentator (fast) vs MedSAM2 seeded with the TotalSegmentator bounding box on axial slice 24 ..."},
  {"heading": "Result", "table": [{"method": "TotalSegmentator", "volume_ml": 1062.1, "mean_HU": 71.4}, {"method": "MedSAM2", "volume_ml": 1069.7}], "figures": ["omm_outputs/views/....png"]},
  {"heading": "QC", "text": "Three-plane and montage views inspected; single component; no leakage. Dice 0.963 between methods."}],
  metadata={"image": "example_ct_sm.nii.gz", "spacing_mm": "3x3x3", "run_dirs": "..."})
```
-> `omm_outputs/reports/liver_volumetry_-_example_ct_sm.md` and `.html`.

The agent's final answer to the user quotes the volume (1062 mL and 1070 mL), the
agreement metrics, the QC observations and the run directories - and states that this is a
research measurement, not a clinical finding.

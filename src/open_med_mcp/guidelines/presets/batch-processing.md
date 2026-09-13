---
name: batch-processing
title: Process a cohort - batch runs, QC sampling, summary table
summary: Run a model over many images with run_batch, sample cases for visual QC, handle failures, and deliver a CSV plus a short QC report.
tags: [batch, cohort, workflow]
modalities: [any]
tasks: [batch]
version: 1
---
1. `list_workspace(pattern="*.nii.gz", recursive=true)` or `list_dicom_series(folder)` to build the
   case list; convert DICOM series with `convert_image` first.
2. Pilot on 2-3 cases with `run_model` and full QC (three-plane + montage) to fix parameters.
3. `run_batch(model, pattern="cohort/*.nii.gz", params={...}, output_dir="masks")` -> CSV/JSON with
   per-case status, run directory and volumes. Failures are recorded, not fatal.
4. QC sampling: render at least 10 % of cases (min 5), prioritising outliers: the smallest and
   largest volumes, `components > 1`, and any case whose volume is > 2 SD from the cohort mean.
5. Re-run failed or rejected cases with adjusted parameters; keep the CSV rows consistent.
6. `write_report` with the summary table (n, failures, volume mean/SD/range), the QC figures and the
   parameter set. Never present an unchecked batch as final.

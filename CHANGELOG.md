# Changelog

## 0.1.0 - 2026-09-13

Initial public release.

* MCP server (Python SDK v2) with 34 tools, 12 guideline prompts and `guideline://`, `model://`,
  `omm://conventions` resources.
* Model zoo with a uniform job-directory contract and three execution backends (local Python,
  Docker, Apptainer), including a *wrapped image* mode for official third-party containers:
  `medsam2`, `sam2` (SAM 2.1), `voxtell` (free-text prompts), `totalsegmentator`, `lungmask`, `hdbet` (HD-BET), `synthstrip`
  (FreeSurfer container), `nnunet` (any nnU-Net v2 model), `monai` (MONAI Model Zoo bundles),
  `torchxrayvision` (chest X-ray classification), `classical` (thresholds / region growing),
  and an adapter template.
* Tools: inspect / DICOM series listing / conversion, segment / run_model / classify_image /
  run_batch, mask statistics / post-processing / comparison / features / meshes / combination,
  resampling, re-orientation, cropping, N4 bias correction, registration (rigid, affine,
  B-spline) and transform application, PNG views, HTML viewer, reports, guidelines.
* Preset guidelines: getting-started, segmentation-3d-ct, segmentation-3d-mri,
  segmentation-2d-prompted, brain-mri-preprocessing, chest-xray-triage, lung-ct-analysis,
  registration-followup, batch-processing, multi-organ-ct-report, compare-two-segmentations,
  qc-checklist.
* Viewer: PNG renderer (single / three-plane / montage, prompts, native-coordinate grid),
  self-contained HTML slice viewer with prompt-coordinate picking, Markdown/HTML reports, NiiVue
  3D page, renderer plugin registry.
* Plug-ins: drop-in tool modules (`omm_plugins/`, `OMM_PLUGIN_DIRS`, entry points) with a stable
  `open_med_mcp.plugin_api`; scaffolding (`new plugin|model|guideline`); `install` for Claude Code / Codex.
* Robustness: progress notifications and background jobs (`wait=false`, `get_job`), thread-safe
  caches/rendering, capped inline images, text-only mode, stderr/file logging, HTTP path policy,
  scenario tests over real stdio and streamable-HTTP transports.
* CLI: serve, doctor, models (list/info/check/download/build/pull/def), run, view, serve-viewer,
  guidelines, plugins, new, install, client-config. CI (lint, tests, wheel) and GHCR container builds.

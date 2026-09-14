# Guidelines

A guideline is a protocol an agent can follow: which tools to call in which order, what a good
result looks like, what to check, what to report. They turn "segment the liver" into a
reproducible procedure.

## Format

```markdown
---
name: segmentation-3d-ct           # identifier used by get_guideline / prompts / resources
title: Segment an organ or lesion in a 3D CT volume
summary: One line shown in listings.
tags: [segmentation, ct, 3d]
modalities: [CT]                   # used by list_guidelines(modality=...)
tasks: [segmentation]
models: [totalsegmentator, medsam2, classical]
version: 1
---
## 1. Inspect
...
```

The body is plain Markdown. Write for an agent: exact tool names and parameters, decision tables,
numeric sanity ranges, explicit QC criteria, and what the report must contain.

## Presets

| name | use it for |
|---|---|
| `getting-started` | orientation: the standard loop, conventions, safety |
| `segmentation-3d-ct` | organs/lesions in CT: TotalSegmentator vs MedSAM2 vs classical, QC, volumes |
| `segmentation-3d-mri` | MRI specifics: uncalibrated intensities, acquisition plane, region growing |
| `segmentation-2d-prompted` | X-ray, ultrasound frames, dermoscopy, endoscopy, histology tiles |
| `brain-mri-preprocessing` | reorient, N4, skull stripping (HD-BET / SynthStrip), QC, hand-off |
| `chest-xray-triage` | TorchXRayVision findings, honest reporting, optional 2D segmentation |
| `lung-ct-analysis` | lungmask lungs/lobes, volumes, LAA% emphysema index, vessels |
| `registration-followup` | register follow-up to baseline, propagate masks, volume change |
| `batch-processing` | run_batch over a cohort, QC sampling, summary CSV |
| `multi-organ-ct-report` | volumetry table for several organs with per-organ QC figures |
| `compare-two-segmentations` | prediction vs reference: metrics + visual difference |
| `qc-checklist` | the minimum checks before accepting a mask |
| `chest-xray-anatomy-and-ctr` | anatomy segmentation + cardiothoracic ratio on a frontal radiograph |
| `recist-1-1` | measurable lesions, SLD, CR/PR/SD/PD (Eisenhauer 2009) |
| `fleischner-2017` | incidental pulmonary nodule follow-up (MacMahon 2017) |
| `lung-rads-2022` | screening CT categories (ACR) |
| `li-rads-2018` | HCC categories on multiphase CT/MRI (ACR) |
| `acr-ti-rads-2017` | thyroid nodule points, levels and FNA thresholds |
| `coronary-calcium-agatston` | calcium scoring requirements and categories |
| `organ-volume-reference-ranges` | adult organ volume ranges with sources, for QC |

## Customizing

* Workspace-level: put `*.md` files in `<workspace>/omm_guidelines/`.
* Machine-level: `OMM_GUIDELINE_DIRS=/path/a:/path/b`.
* A user file with the same `name` as a preset replaces it; other files are added.
* `list_guidelines` re-reads the directories on every call, so edits are live.

## How agents get them

* Tools: `list_guidelines(query, tags, modality)`, `get_guideline(name)`.
* MCP prompts: every guideline is a prompt with optional `image` and `goal` arguments
  (Claude Code: `/mcp__open-med-mcp__segmentation-3d-ct`).
* MCP resources: `guideline://<name>` (Markdown).
* The server `instructions` tell the agent to read `getting-started`/`get_conventions` first.

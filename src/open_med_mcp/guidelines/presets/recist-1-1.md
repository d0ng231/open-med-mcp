---
name: recist-1-1
title: RECIST 1.1 - measuring solid tumours and classifying response
summary: How to select, measure and follow target lesions on CT/MRI and classify response (CR/PR/SD/PD) per RECIST 1.1, with the tools that compute the numbers.
tags: [clinical, oncology, recist, response, measurement]
modalities: [CT, MR]
tasks: [measurement, follow-up]
version: 1
---
**Source**: Eisenhauer EA et al. New response evaluation criteria in solid tumours: revised RECIST
guideline (version 1.1). *Eur J Cancer* 2009;45:228-247. This preset is a faithful summary for
agent use; consult the paper for edge cases.

## Measurability
* Target lesion: longest diameter >= 10 mm on CT (slice thickness <= 5 mm; if thicker, >= 2 x
  thickness); >= 20 mm on chest X-ray. Lymph nodes: **short axis** >= 15 mm target, 10-14 mm
  non-target, < 10 mm normal.
* Up to 5 target lesions in total, max 2 per organ; all others are non-target (recorded as
  present / absent / unequivocal progression).
* Not measurable: bone lesions without soft-tissue component, cystic/necrotic lesions unless a
  solid component qualifies, lesions in previously irradiated fields (unless progressing).

## Measuring with the tools
1. Segment each target lesion (`segment(..., model="medsam2")` with a box on the slice where it is
   largest, or `voxtell` with a descriptive prompt); QC the mask (`render_view`, contour mode).
2. `measure_lesion(mask, image=...)` returns the longest axial diameter, the perpendicular short
   axis, the slice used and the endpoints; it flags whether the lesion is measurable. Use the
   short axis for lymph nodes.
3. Sum the longest diameters of the target lesions (short axes for nodes) -> SLD. The tool
   returns `sum_of_longest_diameters_mm` for the measured labels.

## Response (target lesions)
| category | rule |
|---|---|
| CR | disappearance of all target lesions; any pathological node < 10 mm short axis |
| PR | >= 30 % decrease in SLD vs baseline |
| PD | >= 20 % increase in SLD vs the **nadir** (smallest SLD on study, including baseline) **and** absolute increase >= 5 mm; or new lesions |
| SD | neither PR nor PD |

`recist_response(baseline_sld_mm, current_sld_mm, nadir_sld_mm, new_lesions=...)` applies these
rules. Overall response also depends on non-target lesions and new lesions (unequivocal
progression of non-target disease is PD). Report the exact diameters, slice numbers and the
scanner/slice thickness; small lesions (< 10 mm) are recorded as 5 mm default when present but
too small to measure ("too small to measure").

---
name: coronary-calcium-agatston
title: Coronary artery calcium scoring (Agatston)
summary: Requirements and steps to compute an Agatston score from non-contrast ECG-gated cardiac CT with TotalSegmentator's coronary segmentation and the agatston_score tool, with severity categories.
tags: [clinical, cardiac, calcium, agatston, ct]
modalities: [CT]
tasks: [quantification]
version: 1
---
**Sources**: Agatston AS et al. *J Am Coll Cardiol* 1990;15:827-832; Hecht HS et al. SCCT/STR
guidelines for coronary artery calcium scoring of noncontrast noncardiac chest CT scans. *J
Cardiovasc Comput Tomogr* 2017;11:74-84.

## Requirements
* Non-contrast CT (contrast invalidates the 130 HU threshold), ideally ECG-gated with 2.5-3 mm
  slices; the tool scales area x density to the 3 mm convention using the actual slice thickness.
* A mask restricting the scoring to the coronary arteries (or at least the heart, then review
  every lesion to exclude aortic, valvular and pericardial calcium).

## Steps
1. `inspect_image(ct)`: confirm HU calibration, slice thickness, absence of contrast (aorta ~ 40 HU).
2. `run_model("totalsegmentator", {"image": ct}, task="coronary_arteries")` (or `total` with
   `roi_subset: [heart]` as a coarse region).
3. `agatston_score(ct, mask)` -> per-label and total scores, calcium volume, category
   (0 / 1-10 minimal / 11-100 mild / 101-400 moderate / > 400 severe).
4. `render_view(ct, masks=[mask], window="bone", layout="montage")` and look at every scored
   lesion: exclude non-coronary calcium and motion / noise artefacts.
5. Report the total score, per-vessel scores, category, and that percentile interpretation
   requires age, sex and ethnicity (MESA calculator).

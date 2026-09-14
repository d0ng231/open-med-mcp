---
name: fleischner-2017
title: Fleischner Society 2017 - incidental pulmonary nodules
summary: Management recommendations for incidental solid and subsolid pulmonary nodules on CT in adults (not screening), with the tool that returns the recommendation.
tags: [clinical, chest, lung, nodule, fleischner]
modalities: [CT]
tasks: [follow-up]
version: 1
---
**Source**: MacMahon H et al. Guidelines for management of incidental pulmonary nodules detected
on CT images: from the Fleischner Society 2017. *Radiology* 2017;284:228-243.

**Scope**: incidentally detected nodules in adults >= 35 years. Not for lung-cancer screening
(use Lung-RADS), immunocompromised patients or patients with known primary cancer.

## Measurement
* Size = average of the long and short axis diameters on the same slice (thin-section CT,
  lung window), rounded to the nearest millimetre; report the solid component of part-solid nodules.
* `measure_lesion(nodule_mask, image=ct)` gives long/short axes; average them.
* Risk: high risk = smoking or other risk factors (family history, spiculation, upper-lobe location,
  older age, emphysema/fibrosis); low risk = minimal or absent risk factors.

## Recommendations (tool: `fleischner_recommendation`)
Solid, single: < 6 mm: no routine follow-up (high risk: optional CT at 12 months). 6-8 mm: CT at
6-12 months, then consider (low risk) / perform (high risk) CT at 18-24 months. > 8 mm: consider
CT at 3 months, PET/CT, or tissue sampling.

Solid, multiple: < 6 mm: no routine follow-up (high risk: optional CT at 12 months). >= 6 mm: CT at
3-6 months, then consider (low) / perform (high) CT at 18-24 months.

Subsolid, single: ground-glass < 6 mm: no routine follow-up; >= 6 mm: CT at 6-12 months to confirm
persistence, then every 2 years until 5 years. Part-solid < 6 mm: no routine follow-up; >= 6 mm:
CT at 3-6 months to confirm persistence; if persistent and the solid component is < 6 mm, annual CT
for 5 years (a solid component >= 6 mm is highly suspicious).

Subsolid, multiple: < 6 mm: CT at 3-6 months; if stable, consider CT at 2 and 4 years. >= 6 mm:
CT at 3-6 months; subsequent management based on the most suspicious nodule.

## Reporting
State the nodule type, size (average diameter), number, location, risk category and the exact
recommendation text; nodules with clearly benign features (fat, popcorn/central calcification)
need no follow-up. Perifissural / subpleural lymph nodes < 10 mm (triangular, lentiform) are benign.

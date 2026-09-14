---
name: acr-ti-rads-2017
title: ACR TI-RADS 2017 - thyroid nodules on ultrasound
summary: Points-based risk stratification of thyroid nodules (composition, echogenicity, shape, margin, echogenic foci) with FNA and follow-up size thresholds; computed by the tirads_score tool.
tags: [clinical, thyroid, ultrasound, ti-rads]
modalities: [US]
tasks: [diagnosis]
version: 1
---
**Source**: Tessler FN et al. ACR Thyroid Imaging, Reporting and Data System (TI-RADS): White
Paper of the ACR TI-RADS Committee. *J Am Coll Radiol* 2017;14:587-595.

Assign points for each of the five categories and sum them (`tirads_score(...)` does this):

| category | 0 points | 1 point | 2 points | 3 points |
|---|---|---|---|---|
| composition | cystic / spongiform | mixed cystic-solid | solid / almost completely solid | - |
| echogenicity | anechoic | hyperechoic / isoechoic | hypoechoic | very hypoechoic |
| shape | wider-than-tall | - | - | taller-than-wide |
| margin | smooth / ill-defined | - | lobulated / irregular | extra-thyroidal extension |
| echogenic foci (sum all) | none / large comet-tail | macrocalcifications | peripheral (rim) | punctate echogenic foci |

Levels: TR1 (0 points, benign) and TR2 (2 points, not suspicious): no FNA. TR3 (3 points, mildly
suspicious): FNA >= 2.5 cm, follow >= 1.5 cm. TR4 (4-6 points, moderately suspicious): FNA >= 1.5
cm, follow >= 1.0 cm. TR5 (>= 7 points, highly suspicious): FNA >= 1.0 cm, follow >= 0.5 cm.
Follow-up ultrasound intervals: TR3 at 1, 3, 5 years; TR4 at 1, 2, 3, 5 years; TR5 annually up to
5 years. Do not biopsy more than two nodules per gland (select the highest-scoring ones).

Report the feature values, points, level, maximum diameter and the resulting recommendation.

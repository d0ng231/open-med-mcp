---
name: li-rads-2018
title: LI-RADS v2018 - hepatocellular carcinoma on CT/MRI in patients at risk
summary: How to categorise liver observations (LR-1 to LR-5, LR-M, LR-TIV) in cirrhosis / chronic HBV using major features on multiphase CT or MRI.
tags: [clinical, liver, hcc, li-rads]
modalities: [CT, MR]
tasks: [diagnosis]
version: 1
---
**Source**: American College of Radiology, CT/MRI LI-RADS v2018 core (https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/LI-RADS);
Chernyak V et al. *Radiology* 2018;289:816-830.

**Population**: adults with cirrhosis, chronic hepatitis B, or current/prior HCC. Not for patients
< 18 years, cirrhosis from congenital hepatic fibrosis or vascular disorders.

## Major features (arterial + portal venous + delayed phases needed)
* **APHE** - non-rim arterial-phase hyperenhancement.
* **Washout** - non-peripheral washout on portal venous / delayed phase.
* **Enhancing capsule**.
* **Size** - largest outer-edge dimension on the phase where the margins are clearest (not arterial
  if edges are blurred); measured with `measure_lesion` after segmenting the observation.
* **Threshold growth** - >= 50 % size increase in <= 6 months.

## Categories
| category | meaning |
|---|---|
| LR-NC | not categorizable (omitted / degraded images) |
| LR-TIV | definite tumor in vein |
| LR-1 / LR-2 | definitely / probably benign (cyst, hemangioma, perfusion alteration ...) |
| LR-3 | intermediate probability of malignancy |
| LR-4 | probably HCC |
| LR-5 | definitely HCC |
| LR-M | probably or definitely malignant, not HCC specific (targetoid: rim APHE, peripheral washout, delayed central enhancement) |

Diagnostic table for APHE-positive observations (no APHE: < 20 mm -> LR-3 (LR-4 with >= 2 additional
features), >= 20 mm -> LR-3/LR-4): with non-rim APHE, **< 10 mm**: LR-3 (LR-4 with any additional
major feature); **10-19 mm**: LR-3 with none, LR-4 with one, LR-5 with washout or threshold growth,
LR-4 with capsule only; **>= 20 mm**: LR-4 with none, LR-5 with any additional major feature.

Ancillary features may adjust by one category (never to LR-5). Report every major feature
explicitly, the phases available, size with slice/phase, and the category with its rationale.

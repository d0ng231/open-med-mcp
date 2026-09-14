---
name: organ-volume-reference-ranges
title: Reference ranges for common organ volumes (adults)
summary: Approximate adult reference ranges for organ volumes measured on CT/MRI, with sources, for sanity-checking segmentation results and flagging outliers.
tags: [reference, volumetry, qc]
modalities: [CT, MR]
tasks: [qc]
version: 1
---
Use these ranges to flag implausible masks and for context in reports; they are approximate and
depend on sex, body size, age, contrast phase and segmentation protocol. Always cite the source
you used and prefer population-specific references when available.

| organ | typical adult range | source |
|---|---|---|
| liver | 1200-1800 mL (mean ~1500 mL; correlates with body surface area) | Vauthey JN et al. *Liver Transpl* 2002;8:233-240 |
| spleen | 100-300 mL; upper limit of normal ~314 mL | Prassopoulos P et al. *Eur Radiol* 1997;7:246-248 |
| kidney (each) | 120-200 mL (mean ~150 mL; men > women) | Cheong B et al. *Clin J Am Soc Nephrol* 2007;2:38-45 |
| pancreas | 60-110 mL (declines with age) | Saisho Y et al. *Clin Anat* 2007;20:933-942 |
| total lung capacity (CT lung volume at full inspiration) | 4-7 L | Stocks J, Quanjer PH. *Eur Respir J* 1995;8:492-506 |
| heart (cardiac volume on CT) | 600-900 mL | Kim TS et al. *Radiology* 2004 (population data vary) |
| intracranial volume | 1300-1700 mL | Whitwell JL et al. *AJNR* 2001;22:1483-1489 |
| total brain volume (adult) | 1100-1400 mL (declines ~0.2-0.5 %/year after 40) | Fjell AM, Walhovd KB. *Rev Neurosci* 2010 |
| thyroid | 10-20 mL (iodine sufficient regions) | WHO/ICCIDD reference values |
| prostate | < 30 mL normal; > 40 mL enlarged | Roehrborn CG. *Rev Urol* 2005 |

Rules of thumb for QC: a segmentation more than about 30 % outside these ranges deserves a visual
check (`render_view`) before it is reported; a volume that touches the field-of-view border is
truncated and must be flagged.

---
name: chest-xray-anatomy-and-ctr
title: Chest radiograph anatomy segmentation and cardiothoracic ratio
summary: Segment the lungs, heart and other structures on a frontal chest X-ray with TorchXRayVision's PSPNet and compute the cardiothoracic ratio, with the known caveats.
tags: [clinical, xray, chest, cardiomegaly, ctr]
modalities: [XR]
tasks: [measurement]
version: 1
---
1. `inspect_image(cxr)` - frontal projection only; note whether PA or AP (from the DICOM header
   or the request). CTR is only interpretable on PA erect films (AP / supine magnify the heart).
2. `run_model("torchxrayvision", {"image": cxr}, task="segment")` -> 14-label anatomy map (label
   names in the response: Left/Right Lung, Heart, Clavicles, Scapulae, Mediastinum, Spine, ...).
3. `render_view(cxr, masks=[mask], mode="contour")` - check that the heart contour follows the
   cardiac silhouette and that both lungs reach the costophrenic angles.
4. `cardiothoracic_ratio(mask)` -> widest heart width / widest thoracic width (lung outer margins,
   an approximation of the inner rib margins). CTR > 0.5 on a PA film suggests cardiomegaly.
5. Optional: `classify_image(cxr)` for the 18-finding probabilities and
   `run_model("torchxrayvision", {"image": cxr}, task="age")` for the biological-age estimate.
6. Report projection, CTR with the two widths in pixels (and mm when pixel spacing is known), the
   segmentation QC, and the disclaimer that AP films overestimate the ratio.

Sources: Danzer CS. *Am J Med Sci* 1919; Cohen JP et al. TorchXRayVision, MIDL 2022 (chestx-det
anatomy model: Lian J et al., 2021).

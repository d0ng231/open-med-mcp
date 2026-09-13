# monai

Run MONAI Model Zoo segmentation bundles. The bundle is downloaded on first use.

```text
run_model("monai", {"image": "ct.nii.gz"}, params={"bundle": "spleen_ct_segmentation"})
run_model("monai", {}, task="download", params={"bundle": "wholeBody_ct_segmentation"})
```

The adapter overrides `datalist` (one image) and `output_dir`; pass other config overrides in
`overrides` (e.g. `{"inferer#roi_size": [96, 96, 96]}`). Label names come from the bundle's
`metadata.json` (`channel_def`).

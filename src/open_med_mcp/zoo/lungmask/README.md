# lungmask

Chest CT lung (`lungs`, R231) and lobe (`lobes`, LTRCLobes + R231 fill) segmentation from
[JoHof/lungmask](https://github.com/JoHof/lungmask). The response also contains, per label, the
low-attenuation-area percentage below `laa_threshold_hu` (default -950 HU), a common emphysema index.

```text
run_model("lungmask", {"image": "chest_ct.nii.gz"}, task="lobes")
```

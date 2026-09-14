# radiomics

[pyradiomics](https://github.com/AIM-Harvard/pyradiomics) feature extraction (shape, first order,
GLCM, GLRLM, GLSZM, GLDM, NGTDM; optional LoG/wavelet filters) per label of a mask.

```text
run_model("radiomics", {"image": "ct.nii.gz", "mask": "liver.nii.gz"}, params={"bin_width": 25, "resample_spacing": [1, 1, 1]})
```

The result inlines `features.json` (`results.features.labels[label][feature] = value`) and writes a
CSV twin. Because pyradiomics 3.1 only ships wheels for Python <= 3.9, run it in its own
interpreter (`OMM_ZOO_RADIOMICS_PYTHON`) or in the container.

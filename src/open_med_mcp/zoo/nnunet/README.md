# nnunet

Run **any** nnU-Net v2 model. Three ways to point at a model:

```text
run_model("nnunet", {"image": "ct.nii.gz"}, params={"model_dir": "/models/Dataset001_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres"})
run_model("nnunet", {"image": "ct.nii.gz"}, params={"dataset": "Dataset001_Liver", "configuration": "3d_fullres"})   # under $OMM_HOME/weights/nnunet/results
run_model("nnunet", {}, task="install", params={"zip_url": "https://.../model.zip"})                                  # then use dataset=...
```

Multi-channel models: pass `image`, `image_1`, `image_2`, ... in the training channel order.
Label names are read from the model's `dataset.json`.

# torchxrayvision

Chest X-ray models from [TorchXRayVision](https://github.com/mlmed/torchxrayvision):

| task | model | output |
|---|---|---|
| `classify` | DenseNet-121 / ResNet-50 (`weights` param) | calibrated probabilities for 18 findings (`classify_image` renders a chart) |
| `segment` | chestx-det PSPNet | 14-structure anatomy label map (lungs, heart, clavicles, scapulae, mediastinum, spine, airways ...) - feeds `cardiothoracic_ratio` |
| `age` | RIKEN xray-age | biological age estimate |

```text
classify_image("cxr.png")
run_model("torchxrayvision", {"image": "cxr.png"}, task="segment")   # then cardiothoracic_ratio(mask)
run_model("torchxrayvision", {"image": "cxr.png"}, task="age")
```

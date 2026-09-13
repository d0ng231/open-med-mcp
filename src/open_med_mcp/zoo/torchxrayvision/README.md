# torchxrayvision

Chest X-ray classification with [TorchXRayVision](https://github.com/mlmed/torchxrayvision).
Returns calibrated probabilities for 18 findings; the `classify_image` tool renders them as a bar
chart for the agent.

```text
classify_image("cxr.png", model="torchxrayvision")
```

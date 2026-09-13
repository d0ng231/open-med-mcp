# classical

Threshold / Otsu / multi-range / seeded region growing. No weights, no GPU, runs in any of the
three backends. It is the model used by the test-suite and a practical way to obtain coarse masks
(e.g. lungs in CT with `lower=-1000, upper=-500, keep=largest`) that seed promptable models.

```text
tasks: threshold | otsu | multi_threshold | region_grow
```

See `classical.yaml` for parameters.

# sam2 / medsam2

One adapter, two manifests:

| model     | default checkpoint | best for |
|-----------|--------------------|----------|
| `sam2`    | `sam2.1_hiera_small` | natural / RGB images, generic objects |
| `medsam2` | `medsam2_latest`     | CT / MRI / PET / ultrasound / endoscopy, lesions and organs |

* 2D images: `SAM2ImagePredictor` with point/box prompts.
* 3D volumes: `SAM2VideoPredictor`; slices along the requested plane are written as frames, prompts
  are placed on their slice and propagated forward and backward. Propagation stops after
  `stop_after_empty` consecutive empty slices.
* Multiple objects: give each prompt an `object_id`; the output label equals the object id.

Weights are downloaded with `open-med-mcp models download medsam2` (or `sam2 --weights <id>`) into
`$OMM_HOME/weights/sam2/` and mounted at `/weights` inside containers.

Install for the local runner: `pip install "open-med-mcp[sam2]"` (needs a recent PyTorch). Set
`OMM_ZOO_SAM2_PYTHON=/path/to/venv/bin/python` to run it from a dedicated environment.

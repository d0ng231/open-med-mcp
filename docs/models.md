# Models, adapters and the job contract

## Model zoo

| name | adapter | category | tasks | prompts | notes |
|---|---|---|---|---|---|
| `medsam2` | `sam2` | segmentation | `segment` | box, point | default checkpoint `medsam2_latest`; also `medsam2_ct_lesion`, `medsam2_mri_liver_lesion`, `medsam2_us_heart`, `medsam2_2411` |
| `sam2` | `sam2` | segmentation | `segment` | box, point | `sam2.1_hiera_tiny/small/base_plus/large` |
| `voxtell` | `voxtell` | segmentation | `segment`, `download` | text | free-text prompts, one per structure; label i = prompt i; needs a GPU |
| `totalsegmentator` | `totalsegmentator` | segmentation | `total`, `total_mr`, `total_fast`, `lung_vessels`, `body`, ... `download` | - | multi-label output + label names; `roi_subset` limits structures |
| `lungmask` | `lungmask` | segmentation | `lungs`, `lobes` | - | LAA% emphysema index in `stats.low_attenuation_area` |
| `hdbet` | `hdbet` | preprocessing | `extract` | - | outputs `mask` and `brain` |
| `synthstrip` | `synthstrip` | preprocessing | `strip` | - | **wrapped** official `freesurfer/synthstrip` image |
| `nnunet` | `nnunet` | segmentation | `predict`, `install` | - | any nnU-Net v2 model folder / zip; multi-channel via `image_1`, `image_2` |
| `monai` | `monai` | segmentation | `segment`, `download` | - | MONAI Model Zoo bundles; overrides via `overrides` |
| `torchxrayvision` | `torchxrayvision` | classification | `classify` | - | JSON `predictions` output; `classify_image` renders a chart |
| `classical` | `classical` | segmentation | `threshold`, `otsu`, `multi_threshold`, `region_grow` | seeds | CPU, no weights |

`open-med-mcp models info <name>` prints the manifest, the JSON schema of its parameters, weight
status and which backends can run it right now. `run_model` inlines JSON outputs (`results`) and
previews mask outputs; `outputs` of kind `image` (e.g. HD-BET's stripped brain) are returned as paths.

## Execution backends

| backend | when | how |
|---|---|---|
| `local` | the adapter's imports work in the server's interpreter or in `OMM_ZOO_<ADAPTER>_PYTHON` | `python run.py --job <dir>` as a subprocess |
| `docker` | `docker info` works and the image exists locally (`models pull` / `models build`) | `docker run --rm -v <dir>:/job -v <weights>:/weights [--gpus all] <image> python /app/run.py --job /job` |
| `apptainer` | `apptainer`/`singularity` on PATH | `apptainer exec [--nv] -B <dir>:/job -B <weights>:/weights <sif or docker://image> python /app/run.py --job /job` |

`OMM_RUNNER=auto` (default) tries local -> docker -> apptainer and reports every reason when none
works. Weights live in `$OMM_HOME/weights/<adapter>/` on the host and are mounted read-write at
`/weights`; containers never need network access when weights were downloaded beforehand
(TotalSegmentator downloads its own weights on first use - do it once with
`open-med-mcp models download totalsegmentator`).

## The job contract (v1)

```text
<run_dir>/                 # omm_outputs/<timestamp>-<model>-<id>/  (mounted at /job in containers)
  request.json             # written by the server
  inputs/image.nii.gz      # staged inputs (.nii.gz for volumes, .png for 2D integer images)
  outputs/                 # written by the adapter
  response.json            # written by the adapter
  log.txt                  # adapter log + runner stdout/stderr
```

`request.json`
```json
{
  "contract_version": 1,
  "model": "medsam2", "adapter": "sam2", "task": "segment",
  "inputs": {"image": "inputs/image.nii.gz"},
  "params": {"variant": "medsam2_latest", "axis": 0,
             "prompts": [{"type": "box", "label": 1, "object_id": 1, "slice": 97, "coords": [120, 140, 330, 360]}]},
  "output_dir": "outputs",
  "resources": {"device": "cuda", "weights_dir": "/home/me/.cache/open-med-mcp/weights/sam2"}
}
```

`response.json`
```json
{
  "contract_version": 1, "status": "ok",
  "outputs": {"mask": "outputs/mask.nii.gz"},
  "labels": {"1": "object_1"},
  "stats": {"labels_present": [1], "foreground_voxels": 183211, "per_label": {"1": {"voxels": 183211, "volume_ml": 1432.1, "bbox_xyz": [118, 138, 61, 333, 362, 133]}}},
  "model_info": {"adapter": "sam2", "variant": "medsam2_latest", "device": "cuda"},
  "warnings": [], "timing": {"model_loaded": 2.1, "propagated": 9.8, "total_s": 10.4}
}
```
On failure: `{"status": "error", "error": {"type": "...", "message": "...", "traceback": "..."}}`.

Environment inside the adapter process: `OMM_WEIGHTS_DIR` (weights directory), `OMM_DEVICE`
(`cuda`/`cpu`). Inputs are staged so the adapter only ever needs to read `.nii.gz`/`.nrrd`/... via
SimpleITK or `.png` via Pillow, both wrapped by `omm_job.load_image`.

## Prompt wire format (promptable adapters)

The server converts agent prompts (native `x, y, z`) into per-slice 2D coordinates so the adapter
does not need orientation logic:

```json
{"type": "point", "label": 1, "object_id": 1, "slice": 97, "coords": [col, row]}
{"type": "box",   "label": 1, "object_id": 2, "slice": 97, "coords": [col0, row0, col1, row1]}
```
`params.axis` is the NumPy axis of the volume to slice along (0 for axial in a `[z, y, x]` array);
`slice[row, col]` addresses the 2D slice `np.take(volume, slice, axis=axis)`.

## Writing an adapter

```text
src/open_med_mcp/zoo/<adapter>/      (or $OMM_MODEL_DIRS/<adapter>/ for user models)
  <name>.yaml        manifest (one file per model name; several manifests may share an adapter)
  run.py             entry point: python run.py --job <dir>
  Dockerfile         build context = repository root; COPY zoo/_sdk/omm_job.py and run.py to /app
  README.md
```

`run.py` skeleton (see `zoo/_template/run.py`):

```python
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "_sdk")]
import omm_job

def run(job: omm_job.Job) -> None:
    image, geom = omm_job.load_image(job.input_path("image"))
    device = omm_job.resolve_device(job.device_pref)
    ckpt = job.weights_dir / "mymodel.pt"          # declared in the manifest -> downloaded by the server
    mask = my_segmentation(image, ckpt, device, threshold=float(job.param("threshold", 0.5)))
    out = omm_job.save_mask(mask, geom, job.output_path("mask.nii.gz"))
    job.finish({"mask": out}, stats=omm_job.mask_summary(mask, geom), labels={1: "object"})

if __name__ == "__main__":
    omm_job.main(run)
```

Manifest essentials:

```yaml
name: mymodel
tasks: [segment]
modalities: [CT]
dims: [3]
prompt_types: []                 # [] = automatic; [point, box] = promptable
params:
  threshold: {type: number, default: 0.5, minimum: 0, maximum: 1, description: "..."}
weights:
  - {id: default, file: mymodel.pt, url: "https://...", sha256: "..."}
container: {gpu: preferred}      # image defaults to $OMM_IMAGE_PREFIX-<adapter>:<version>
local: {entrypoint: run.py, requires: [torch, mypackage], extra: mymodel}
runners: [local, docker, apptainer]
```

Then: `open-med-mcp models check mymodel`, `open-med-mcp run mymodel --image x.nii.gz`,
`open-med-mcp models build mymodel` (Docker) or `--engine apptainer`.

## Wrapped images (no adapter code)

Many tools already ship an official container. A manifest can drive such an image directly with a
command template; the server stages the input, mounts the job directory at `/job`, runs the
command and converts the produced files into the standard response:

```yaml
name: synthstrip
category: preprocessing
outputs:
  mask:  {kind: mask,  file: outputs/mask.nii.gz}
  brain: {kind: image, file: outputs/brain.nii.gz}
labels: {1: brain}
params:
  no_csf: {type: boolean, default: false}
  border: {type: integer}
container:
  mode: wrapped
  image: freesurfer/synthstrip:1.8
  gpu_image: freesurfer/synthstrip:1.8-gpu       # used when a GPU is visible
  command: ["mri_synthstrip", "-i", "{inputs.image}", "-o", "{outputs.brain}", "-m", "{outputs.mask}",
            "{flag:no_csf:--no-csf}", "{opt:border:-b}"]
local:
  command: ["mri_synthstrip", "-i", "{inputs.image}", "-o", "{outputs.brain}", "-m", "{outputs.mask}",
            "{flag:no_csf:--no-csf}", "{opt:border:-b}"]   # host CLI, used when on PATH
runners: [docker, apptainer, local]
```

Placeholders: `{inputs.<key>}`, `{outputs.<key>}`, `{params.<name>}` (lists expand to several
arguments), `{flag:<param>:<literal>}` (emitted when the parameter is truthy), `{opt:<param>:<opt>}`
(emits `<opt> <value>` when set), `{job}`, `{weights}`. Docker runs the image with
`--entrypoint <command[0]>`; Apptainer uses `exec`. Outputs marked `required: false` may be absent.

## Containers

Images are published by `.github/workflows/containers.yml` to
`ghcr.io/d0ng231/open-med-mcp-<adapter>:<version>`. Base images: `python:3.12-slim` (classical,
torchxrayvision with CPU PyTorch), `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (sam2,
totalsegmentator, lungmask, hdbet, nnunet, monai; `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime` for voxtell). Weights are never baked in; wrapped models use
the upstream image as is.

Apptainer definitions are generated from the Dockerfiles by
`open_med_mcp.models.containers.dockerfile_to_def` (supports FROM, ARG, ENV, WORKDIR, COPY, RUN,
ENTRYPOINT, CMD, LABEL) - `open-med-mcp models def <name>` prints it.

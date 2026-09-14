# HPC and SLURM

Typical cluster constraints: no Docker daemon, Apptainer/Singularity available, GPUs only through the
scheduler, shared network file systems. open-med-mcp handles all three.

## Containers with Apptainer

```bash
export OMM_HOME=/scratch/$USER/open-med-mcp        # weights + .sif images on a large file system
open-med-mcp models pull medsam2 --engine apptainer  # docker://ghcr.io/... -> $OMM_HOME/images/sam2-0.1.sif
# or build from the Dockerfile (converted to an Apptainer definition; --fakeroot if your site needs it)
open-med-mcp models build medsam2 --engine apptainer
open-med-mcp models download medsam2                # weights are mounted at /weights, not baked in
export OMM_RUNNER=apptainer
```

`--nv` (GPU passthrough) is added automatically when `nvidia-smi` sees a GPU.

## Local backend in a GPU venv

If Apptainer is not available on the compute nodes either, use a Python environment with PyTorch:

```bash
uv venv /scratch/$USER/venvs/omm-sam2 --python 3.12
uv pip install --python /scratch/$USER/venvs/omm-sam2/bin/python torch sam2 hydra-core SimpleITK pillow scipy TotalSegmentator
export OMM_ZOO_SAM2_PYTHON=/scratch/$USER/venvs/omm-sam2/bin/python
export OMM_ZOO_TOTALSEGMENTATOR_PYTHON=$OMM_ZOO_SAM2_PYTHON
```

The server itself is light (no PyTorch) and can run on a login node; heavy adapters run wherever the
server process runs, so start the server (or the agent) inside a GPU allocation:

```bash
srun -p gpu --gres=gpu:1 -c 8 --mem=32G -t 04:00:00 --pty bash
claude   # with open-med-mcp registered; OMM_WORKSPACE points at your study directory
```

## Batch use without an agent

```bash
sbatch --gres=gpu:1 --wrap "open-med-mcp run totalsegmentator --image ct.nii.gz --params '{\"roi_subset\": [\"liver\"]}' --output liver.nii.gz"
```

## Verification performed for this release

On a SLURM cluster (login node without GPU; L40S/H100 nodes without Docker or Apptainer):

* `classical` through the local runner and inside an Apptainer image built from the converted
  Dockerfile (CPU); `synthstrip` through the official image pulled with Apptainer (CPU).
* `medsam2`, `sam2` (2D + 3D, single and multi-object, axial and coronal propagation),
  `totalsegmentator` (fast and full resolution), `lungmask` (lungs, lobes), `hdbet`, `nnunet`
  (TotalSegmentator's 3 mm model as a generic nnU-Net folder), `monai` (spleen bundle),
  `voxtell` (six text prompts; Dice 0.81-0.94 against TotalSegmentator), `torchxrayvision`
  (classification, anatomy segmentation, age), `vlm` (Qwen2.5-VL and MedGemma), `radiomics`
  (pyradiomics, 200 features) and `monai` detection (RetinaNet lung nodules) through the local
  runner in GPU/CPU venvs; the clinical criteria tools (RECIST, Fleischner, TI-RADS, Agatston,
  CTR, FLR, Mayo ADPKD) are unit-tested and exercised through the MCP server.

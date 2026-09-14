#!/usr/bin/env bash
# Public sample data used in the documentation and for trying the tools:
#   - a small abdominal CT (122x101x30 voxels, 3 mm) from the TotalSegmentator repository (Apache-2.0)
#   - the MNI152 2009a symmetric T1 template (1 mm) as shipped with nilearn (ICBM, free for research use)
#   - a chest radiograph from the TorchXRayVision test set (NIH ChestX-ray14, public)
set -euo pipefail
mkdir -p sample && cd sample
curl -L -o example_ct_sm.nii.gz "https://github.com/wasserth/TotalSegmentator/raw/master/tests/reference_files/example_ct_sm.nii.gz"
curl -L -o mni152_t1.nii.gz "https://raw.githubusercontent.com/nilearn/nilearn/main/nilearn/datasets/data/mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz"
curl -L -o chest_xray.png "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/00000001_000.png"
echo "downloaded sample/example_ct_sm.nii.gz, sample/mni152_t1.nii.gz, sample/chest_xray.png"
echo "try:  open-med-mcp view sample/example_ct_sm.nii.gz --layout three-plane --window soft-tissue"
echo "      open-med-mcp run lungmask --image sample/example_ct_sm.nii.gz"
echo "      open-med-mcp run synthstrip --image sample/mni152_t1.nii.gz        # needs docker or apptainer"
echo "      open-med-mcp run torchxrayvision --image sample/chest_xray.png --task segment   # anatomy -> cardiothoracic_ratio"
echo "      open-med-mcp run radiomics --image sample/example_ct_sm.nii.gz --output ...   # needs a Python 3.9 venv"

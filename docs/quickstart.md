# Quick start and client setup

## Install

```bash
pip install "open-med-mcp @ git+https://github.com/d0ng231/open-med-mcp"
```

Extras for running models *locally* (without containers):

| extra | installs | note |
|---|---|---|
| `[sam2]` | `torch`, `torchvision`, `sam2`, `hydra-core` | MedSAM2 / SAM 2.1; GPU strongly recommended |
| `[totalsegmentator]` | `TotalSegmentator` (+ nnU-Net, torch) | weights download on first run |
| `[lungmask]` | `lungmask` | CPU works (about a minute per scan) |
| `[hdbet]` | `hd-bet` | weights (~300 MB) download on first run; GPU recommended |
| `[nnunet]` | `nnunetv2` | bring your own trained model |
| `[monai]` | `monai[nibabel,ignite]` | MONAI Model Zoo bundles download on first run |
| `[torchxrayvision]` | `torchxrayvision` (+ torch) | CPU is fine |
| `[models]` | all of the above | |

`synthstrip` has no Python dependency: it runs the official FreeSurfer image (Docker/Apptainer) or
`mri_synthstrip` if FreeSurfer is installed.

Prefer a dedicated environment for heavy models and point the server at it:

```bash
uv venv ~/.venvs/omm-sam2 && uv pip install --python ~/.venvs/omm-sam2/bin/python torch sam2 hydra-core SimpleITK pillow scipy
export OMM_ZOO_SAM2_PYTHON=~/.venvs/omm-sam2/bin/python
```

Or use containers (`open-med-mcp models pull medsam2` for Docker; `--engine apptainer` on HPC).

## Check and prepare

```bash
open-med-mcp doctor                      # python, GPU, docker/apptainer, models x backends, weights
open-med-mcp models download medsam2     # ~156 MB into $OMM_HOME/weights/sam2/
open-med-mcp models download sam2 -w sam2.1_hiera_small
```

## Register with an agent

The server speaks MCP over stdio; the workspace is the directory with your images (relative paths
in tool calls resolve against it, outputs go to `<workspace>/omm_outputs/`).

**Claude Code**
```bash
open-med-mcp install claude-code --workspace /data/study01     # runs the command below for you
claude mcp add open-med-mcp -e OMM_WORKSPACE=/data/study01 -- open-med-mcp serve
```
or in `.mcp.json` at the project root:
```json
{"mcpServers": {"open-med-mcp": {"command": "open-med-mcp", "args": ["serve"], "env": {"OMM_WORKSPACE": "/data/study01"}}}}
```

**Codex CLI**
```bash
open-med-mcp install codex --workspace /data/study01
codex mcp add open-med-mcp --env OMM_WORKSPACE=/data/study01 -- open-med-mcp serve
```
or `~/.codex/config.toml`:
```toml
[mcp_servers.open_med_mcp]
command = "open-med-mcp"
args = ["serve"]
[mcp_servers.open_med_mcp.env]
OMM_WORKSPACE = "/data/study01"
```

Codex runs tools annotated as read-only (inspect, list, stats, render ...) without asking; tools that
write files (`segment`, `run_model`, ...) need an approval, so non-interactive runs should use
`codex exec --full-auto` (workspace-write sandbox) or approve the calls in the TUI.

**Claude Desktop** (`claude_desktop_config.json`), **Cursor** (`.cursor/mcp.json`), **Gemini CLI**:
`open-med-mcp client-config claude-desktop` prints the JSON with absolute paths.

**HTTP transport** (remote agents, several clients):
```bash
open-med-mcp serve --transport streamable-http --host 127.0.0.1 --port 8765 --workspace /data/study01
```

Long model runs: tools stream progress; if your client enforces a short tool timeout, call
`run_model`/`segment` with `wait=false` and poll `get_job`. Text-only clients: set
`OMM_RETURN_IMAGES=0` (results still name the saved PNGs). Logs: `$OMM_HOME/logs/server.log`.

## First conversation

Ask the agent to read the conventions and a guideline first; the server's `instructions` say so,
but being explicit helps smaller models:

> Use open-med-mcp. Read `get_conventions` and the `segmentation-3d-ct` guideline, then segment the
> spleen in `ct.nii.gz` with TotalSegmentator, verify it visually, and report the volume.

## Without an agent (CLI)

```bash
open-med-mcp run classical --image ct.nii.gz --task threshold --params '{"lower": -1000, "upper": -500, "keep": "largest"}' --output lungs.nii.gz
open-med-mcp view ct.nii.gz -m lungs.nii.gz --layout three-plane --window lung
open-med-mcp view ct.nii.gz -m lungs.nii.gz --html            # interactive slice viewer
open-med-mcp serve-viewer ct.nii.gz -m lungs.nii.gz           # NiiVue 3D on http://127.0.0.1:8766
```

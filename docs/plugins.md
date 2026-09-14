# Plug-ins

open-med-mcp is extended in four places; none of them requires forking the package.

| what you add | how | where it shows up |
|---|---|---|
| **tools** (new MCP tools) | a Python file with `register(server)` in `<workspace>/omm_plugins/`, a directory in `OMM_PLUGIN_DIRS`, or a package exposing the `open_med_mcp.plugins` entry point | `list_tools`, `list_plugins` |
| **models** (adapters or wrapped images) | a folder with a manifest (+ `run.py`, Dockerfile) in `<workspace>/omm_models/` or `OMM_MODEL_DIRS` | `list_models`, `segment`, `run_model` |
| **guidelines** | Markdown with YAML front matter in `<workspace>/omm_guidelines/` or `OMM_GUIDELINE_DIRS` | `list_guidelines`, MCP prompts and resources |
| **viewer renderers** | `@register_renderer("name")` in a plug-in, or the `open_med_mcp.renderers` entry point | `render_view(renderer="name")`, `list_renderers` |

Scaffold any of them:

```bash
open-med-mcp new plugin lesion-count          # -> omm_plugins/lesion_count.py
open-med-mcp new model my-unet                # -> omm_models/my-unet/{my-unet.yaml, run.py, Dockerfile}
open-med-mcp new model synthseg --wrapped-image freesurfer/synthseg:latest   # manifest only
open-med-mcp new guideline my-protocol
open-med-mcp plugins list                     # what would load, and any import/registration errors
```

## Tool plug-ins

```python
# <workspace>/omm_plugins/lesion_count.py
from typing import Annotated
from pydantic import Field
from open_med_mcp.plugin_api import load_mask_cached, resolve, result, tool_errors, render_preview, load_image_cached

def register(server):                       # `server` is the MCPServer; use every SDK decorator
    @server.tool()
    @tool_errors                            # exceptions -> is_error results with a readable message
    def count_lesions(mask: Annotated[str, Field(description="Label map path")], min_voxels: int = 5, image: str | None = None):
        """Number and size of connected components in a mask."""
        m = load_mask_cached(resolve(mask))
        ...
        images = [render_preview(load_image_cached(resolve(image)), mask)] if image else []
        return result({"n_components": n, "components": comps}, images)
```

Rules that keep plug-ins stable in every client:

* Type every parameter (pydantic generates the JSON schema the client validates against) and
  describe it with `Field(description=...)`; write a docstring - it is what the agent reads.
* Resolve paths with `resolve()` (workspace-relative, honours `OMM_ALLOW_OUTSIDE_WORKSPACE`).
* Return `result(payload, images)` for JSON + previews, or plain JSON-serializable values.
* Never print to stdout (stdio transport); use `logging.getLogger("open_med_mcp.plugins.<name>")`.
* Long work: run it in your own thread and expose a status tool, or accept a `wait` flag like
  `run_model` does; keep single calls well under the client's tool timeout.

Plug-ins load once at server start (`open-med-mcp plugins list` shows what loads and why something
failed). A failing plug-in is reported, never fatal. Files starting with `_` are ignored.

`open_med_mcp.plugin_api` re-exports the helpers you need: `resolve`, `resolve_output`,
`load_image_cached`, `load_mask_cached`, `labels_for`, `result`, `image_block`, `tool_errors`,
`render_preview`, `get_renderer`, `register_renderer`, `ViewSpec`, `MaskLayer`, `MedicalImage`,
`save_mask`, `mask_stats`, `postprocess`, `compare_masks`, `record_provenance`, `new_run_dir`,
`get_settings`.

## Packaged plug-ins

```toml
# pyproject.toml of your package
[project.entry-points."open_med_mcp.plugins"]
mylab = "mylab.omm_plugin"          # module with register(server)

[project.entry-points."open_med_mcp.renderers"]
gif = "mylab.renderers:GifRenderer"
```

Install the package into the server's environment; nothing else to configure.

## Model plug-ins

See [models.md](models.md): a manifest plus `run.py` against the job contract (any framework,
any language for the container), or a manifest alone for a wrapped image. Models found in
`omm_models/` override bundled ones with the same name, so you can also tune a bundled manifest.

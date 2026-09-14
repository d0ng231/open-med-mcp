# Viewer

The viewer has two audiences: the **agent**, which needs pictures it can reason about, and the
**human**, who wants to scroll, toggle and pick coordinates.

## For agents: `render_view`

```text
render_view(image, masks=[...], plane="axial", layout="single|three-plane|montage",
            slices=[...], n_slices=9, window="soft-tissue", mode="fill|contour|both", alpha=0.35,
            labels=[...], colors=["#ff3b30"], prompts=[...], grid=True, crop_to_mask=False,
            center_xyz=[x, y, z], title=..., max_px=900, renderer="png", output=...)
```

* `single` picks the slice with the largest mask area (or the middle slice) unless `slices` is given.
* `three-plane` crosses at `center_xyz`, the mask centroid or the volume center.
* `montage` spreads `n_slices` over the mask extent (or the whole stack).
* Tick labels are native voxel indices; `panels[]` in the result reports the screen-axis mapping.
* Prompts are drawn on their slice only (green `+` positive point, red `x` negative, cyan dashed box).
* `crop_to_mask=True` zooms to the mask bounding box with a margin.
* The PNG is returned inline (image content) **and** saved under `omm_outputs/views/`.

![three-plane](assets/screenshots/segment_medsam2_liver.png)

## For humans

* `export_viewer(image, masks, output)` writes **one HTML file** with pre-rendered slices
  (JPEG) and overlays (PNG) embedded as base64 - no server, no CDN, opens anywhere. Features:
  slider / wheel / arrow keys, overlay toggle and opacity, legend, and **click to read native
  voxel coordinates**; drag to get a ready-to-paste box prompt.
* `write_report(title, sections, output, metadata)` writes Markdown plus an HTML twin with
  embedded figures. Sections carry `heading`, `text` (Markdown), `figures`, `table`/`columns`.
* `open-med-mcp serve-viewer image.nii.gz -m mask.nii.gz` serves a NiiVue page (WebGL, from CDN)
  with the volumes over a localhost HTTP server for real 3D inspection.

## Customizing renderers

A renderer is any object with `render(image: MedicalImage, masks, spec: ViewSpec) -> RenderResult`,
where `masks` is a list of `(MedicalImage, {"layer": MaskLayer, "labels": {id: name}})`.

```python
from open_med_mcp.viewer import register_renderer
from open_med_mcp.viewer.registry import RenderResult
from open_med_mcp.viewer.slicing import display_slice

@register_renderer("gif")
class GifRenderer:
    def render(self, image, masks, spec):
        frames = [display_slice(image, spec.plane, i).array for i in range(image.n_slices(spec.plane))]
        data = make_gif(frames)                 # your code
        return RenderResult(data, "image/gif", ".gif", {"frames": len(frames)})
```

Install it in the server's environment and expose it through the entry-point group
`open_med_mcp.renderers` in your `pyproject.toml`:

```toml
[project.entry-points."open_med_mcp.renderers"]
gif = "mypkg.renderers:GifRenderer"
```

Agents then call `render_view(..., renderer="gif")`; `list_renderers` shows what is installed.
Templates for the HTML viewer and reports live in `open_med_mcp/viewer/templates/` (Jinja2) and can
be copied and adapted; the slice/prompt geometry helpers in `open_med_mcp.viewer.slicing` are the
building blocks (`display_slice`, `DisplaySlice.to_display/to_native`, `auto_slices`).

"""Example open-med-mcp plug-in: count connected components ("lesions") in a mask.

Copy to <workspace>/omm_plugins/lesion_count.py and restart the server; the tool appears as
`count_lesions` next to the built-in tools.
"""

from typing import Annotated

import numpy as np
from pydantic import Field
from scipy import ndimage

from open_med_mcp.plugin_api import load_image_cached, load_mask_cached, render_preview, resolve, result, tool_errors


def register(server):
    @server.tool()
    @tool_errors
    def count_lesions(
        mask: Annotated[str, Field(description="Label map path")],
        min_voxels: Annotated[int, Field(ge=1, description="Ignore components smaller than this")] = 5,
        image: Annotated[str | None, Field(description="Image for a preview")] = None,
    ):
        """Number and size of connected components in the foreground of a mask (sorted by volume)."""
        mp = resolve(mask)
        m = load_mask_cached(mp)
        labeled, n = ndimage.label(m.array != 0)
        vox = m.voxel_volume_mm3()
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n + 1)) if n else []
        comps = sorted(
            ({"component": i + 1, "voxels": int(s), "volume_ml": round(float(s) * vox / 1000, 3)} for i, s in enumerate(sizes) if s >= min_voxels),
            key=lambda c: -c["voxels"],
        )
        payload = {"mask": mask, "n_components": len(comps), "components": comps[:50]}
        images = [render_preview(load_image_cached(resolve(image)), mask, title=f"{len(comps)} component(s)")] if image else []
        return result(payload, images)

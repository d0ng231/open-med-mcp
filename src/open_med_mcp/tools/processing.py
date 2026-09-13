"""Image processing tools: resample, reorient, crop, N4, registration, mask algebra, features, meshes, DICOM."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import Field

from open_med_mcp.config import get_settings
from open_med_mcp.core import processing as proc
from open_med_mcp.core.image import save_mask, write_labels_sidecar
from open_med_mcp.tools._common import (
    labels_for,
    load_image_cached,
    load_mask_cached,
    resolve,
    resolve_output,
    result,
    tool_errors,
)
from open_med_mcp.viewer.registry import get_renderer
from open_med_mcp.viewer.spec import MaskLayer, ViewSpec
from open_med_mcp.workspace import display_path, record_provenance

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _derived(path: Path, suffix_tag: str, ext: str | None = None) -> Path:
    name = path.name
    for sfx in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        if name.endswith(sfx):
            return path.with_name(name[: -len(sfx)] + f"_{suffix_tag}" + (ext or sfx))
    return path.with_name(name + f"_{suffix_tag}" + (ext or ".nii.gz"))


def register(server: MCPServer) -> None:
    @server.tool()
    @tool_errors
    def resample_image(
        path: str,
        spacing: Annotated[
            list[float] | None,
            Field(description="Target voxel spacing in mm, e.g. [1, 1, 1] (or one value for isotropic)"),
        ] = None,
        reference: Annotated[
            str | None,
            Field(description="Resample onto the grid of this image (size, spacing, origin, direction)"),
        ] = None,
        factor: Annotated[
            float | None, Field(description="Upsample (>1) or downsample (<1) isotropically by this factor")
        ] = None,
        interpolation: Literal["linear", "nearest", "bspline"] = "linear",
        is_mask: Annotated[
            bool, Field(description="Use nearest-neighbour interpolation and preserve labels")
        ] = False,
        output: str | None = None,
    ) -> dict[str, Any]:
        """Resample an image or mask to a new spacing, by a factor, or onto a reference grid
        (e.g. bring a mask into the geometry of another image before compare_masks)."""
        p = resolve(path)
        img = load_mask_cached(p) if is_mask else load_image_cached(p)
        ref = load_image_cached(resolve(reference)) if reference else None
        out_img = proc.resample(img, spacing, ref, factor, interpolation, is_mask)
        out = resolve_output(output, _derived(p, "resampled"))
        if is_mask:
            save_mask(out_img.array, out_img, out)
            if labels_for(p):
                write_labels_sidecar(out, labels_for(p))
        else:
            out_img.save(out)
        record_provenance(
            "resample_image",
            {"path": display_path(p), "spacing": spacing, "reference": reference, "factor": factor},
            {"output": display_path(out)},
        )
        return {
            "output": display_path(out),
            "size_xyz": list(out_img.size_xyz),
            "spacing_xyz_mm": [round(float(s), 4) for s in out_img.spacing],
            "input_size_xyz": list(img.size_xyz),
        }

    @server.tool()
    @tool_errors
    def reorient_image(
        path: str,
        orientation: Annotated[str, Field(description="Target axis code such as RAS, LPS, LPI, RAI")] = "RAS",
        is_mask: bool = False,
        output: str | None = None,
    ) -> dict[str, Any]:
        """Re-order/flip the axes of a 3D image (or mask) to a standard orientation code."""
        p = resolve(path)
        img = load_mask_cached(p) if is_mask else load_image_cached(p)
        out_img = proc.reorient(img, orientation)
        out = resolve_output(output, _derived(p, orientation.lower()))
        if is_mask:
            save_mask(out_img.array, out_img, out)
        else:
            out_img.save(out)
        record_provenance(
            "reorient_image",
            {"path": display_path(p), "orientation": orientation},
            {"output": display_path(out)},
        )
        return {
            "output": display_path(out),
            "orientation": out_img.orientation_code(),
            "input_orientation": img.orientation_code(),
            "size_xyz": list(out_img.size_xyz),
        }

    @server.tool()
    @tool_errors
    def crop_image(
        path: str,
        box_xyz: Annotated[
            list[int] | None,
            Field(description="Inclusive native box [x0, y0, z0, x1, y1, z1] (2D: [x0, y0, x1, y1])"),
        ] = None,
        mask: Annotated[
            str | None, Field(description="Crop to the bounding box of this mask instead of box_xyz")
        ] = None,
        margin: Annotated[int, Field(description="Extra voxels around the box")] = 5,
        also: Annotated[
            list[str] | None, Field(description="Masks to crop with the same box (saved next to the output)")
        ] = None,
        output: str | None = None,
    ) -> dict[str, Any]:
        """Crop an image (and optionally masks) to a box or to a mask's bounding box. Geometry is
        preserved so cropped results stay aligned with the original in physical space."""
        from open_med_mcp.core.masks import bbox_xyz

        p = resolve(path)
        img = load_image_cached(p)
        if mask:
            m = load_mask_cached(resolve(mask), img)
            box = bbox_xyz(m.array != 0)
            if box is None:
                raise ValueError("mask is empty")
        elif box_xyz:
            box = box_xyz
        else:
            raise ValueError("give box_xyz or mask")
        cropped, used = proc.crop(img, box, margin)
        out = resolve_output(output, _derived(p, "crop"))
        cropped.save(out)
        extra = {}
        for mp in also or []:
            mpath = resolve(mp)
            mm = load_mask_cached(mpath, img)
            cm, _ = proc.crop(mm, box, margin)
            mo = out.with_name(_derived(mpath, "crop").name)
            save_mask(cm.array, cm, mo)
            if labels_for(mpath):
                write_labels_sidecar(mo, labels_for(mpath))
            extra[display_path(mpath)] = display_path(mo)
        record_provenance(
            "crop_image", {"path": display_path(p), "box": used}, {"output": display_path(out), **extra}
        )
        return {
            "output": display_path(out),
            "box_xyz": used,
            "size_xyz": list(cropped.size_xyz),
            "cropped_masks": extra,
        }

    @server.tool()
    @tool_errors
    def n4_bias_correction(
        path: str,
        mask: Annotated[str | None, Field(description="Foreground mask (default: Otsu)")] = None,
        shrink: Annotated[
            int, Field(ge=1, le=8, description="Shrink factor for speed (4 = fast, 1 = full resolution)")
        ] = 4,
        output: str | None = None,
        save_bias_field: bool = False,
    ) -> dict[str, Any]:
        """N4 bias field correction for MRI (removes low-frequency intensity inhomogeneity)."""
        p = resolve(path)
        img = load_image_cached(p)
        m = load_mask_cached(resolve(mask), img) if mask else None
        corrected, field = proc.n4_bias_correction(img, m, shrink)
        out = resolve_output(output, _derived(p, "n4"))
        corrected.save(out)
        res = {"output": display_path(out), "shrink": shrink}
        if save_bias_field:
            fo = out.with_name(_derived(p, "n4_bias").name)
            field.save(fo)
            res["bias_field"] = display_path(fo)
        record_provenance("n4_bias_correction", {"path": display_path(p)}, {"output": display_path(out)})
        return res

    @server.tool()
    @tool_errors
    def register_images(
        fixed: Annotated[str, Field(description="Reference image (stays fixed)")],
        moving: Annotated[str, Field(description="Image to align to the fixed image")],
        transform: Literal["rigid", "affine", "bspline"] = "rigid",
        metric: Annotated[
            Literal["mattes", "correlation", "meansquares"],
            Field(
                description="mattes = mutual information (multi-modal), correlation/meansquares for same modality"
            ),
        ] = "mattes",
        iterations: Annotated[int, Field(ge=10, le=2000)] = 200,
        sampling: Annotated[
            float, Field(gt=0, le=1, description="Fraction of voxels sampled for the metric")
        ] = 0.2,
        moving_masks: Annotated[
            list[str] | None, Field(description="Masks in the moving space to warp with the same transform")
        ] = None,
        output: Annotated[str | None, Field(description="Warped moving image path")] = None,
        preview: bool = True,
    ) -> CallToolResult:
        """Register the moving image to the fixed image (rigid/affine/deformable), save the
        transform (.tfm), the warped image and optionally warped masks; returns a checkerboard-style
        overlay preview (fixed = gray, warped moving = red contour of its Otsu foreground)."""
        pf, pm = resolve(fixed), resolve(moving)
        fixed_img, moving_img = load_image_cached(pf), load_image_cached(pm)
        tx, info = proc.register(fixed_img, moving_img, transform, metric, sampling, iterations)
        warped = proc.apply_transform(moving_img, fixed_img, tx)
        out = resolve_output(output, _derived(pm, f"to_{pf.name.split('.')[0]}"))
        warped.save(out)
        tfm = proc.write_transform(tx, out.with_name(out.name.split(".")[0] + ".tfm"))
        payload: dict[str, Any] = {
            "output": display_path(out),
            "transform_file": display_path(tfm),
            **info,
            "warped_masks": {},
        }
        for mp in moving_masks or []:
            mpath = resolve(mp)
            mm = load_mask_cached(mpath, moving_img)
            wm = proc.apply_transform(mm, fixed_img, tx, is_mask=True)
            mo = out.with_name(_derived(mpath, f"to_{pf.name.split('.')[0]}").name)
            save_mask(wm.array, wm, mo)
            if labels_for(mpath):
                write_labels_sidecar(mo, labels_for(mpath))
            payload["warped_masks"][display_path(mpath)] = display_path(mo)
        images = []
        if preview:
            import numpy as np
            import SimpleITK as sitk

            fg = sitk.OtsuThreshold(sitk.Cast(warped.to_sitk(), sitk.sitkFloat32), 0, 1, 128)
            fg_img = fixed_img.with_array(sitk.GetArrayFromImage(fg).astype(np.uint8))
            tmp = out.with_name(out.name.split(".")[0] + "_fgpreview.nii.gz")
            save_mask(fg_img.array, fixed_img, tmp)
            layer = MaskLayer(
                path=str(tmp),
                name="warped moving (foreground contour)",
                color="#ff3b30",
                mode="contour",
                alpha=0.0,
            )
            spec = ViewSpec(
                image=str(pf),
                masks=[layer],
                layout="three-plane" if not fixed_img.is_2d else "single",
                title=f"{transform} registration: {pm.name} -> {pf.name}",
                max_px=get_settings().preview_max_px,
            )
            images.append(
                get_renderer("png").render(
                    fixed_img, [(load_mask_cached(tmp, fixed_img), {"layer": layer})], spec
                )
            )
            tmp.unlink(missing_ok=True)
        record_provenance(
            "register_images",
            {"fixed": display_path(pf), "moving": display_path(pm), "transform": transform},
            {"output": display_path(out), "transform_file": display_path(tfm)},
        )
        return result(payload, images)

    @server.tool()
    @tool_errors
    def apply_transform(
        transform_file: Annotated[str, Field(description=".tfm file written by register_images")],
        moving: str,
        reference: Annotated[str, Field(description="Image defining the output grid (the fixed image)")],
        is_mask: bool = False,
        interpolation: Literal["linear", "nearest", "bspline"] = "linear",
        output: str | None = None,
    ) -> dict[str, Any]:
        """Apply a saved transform to another image or mask (e.g. propagate a baseline mask to a follow-up)."""
        pt, pm, pr = resolve(transform_file), resolve(moving), resolve(reference)
        tx = proc.read_transform(pt)
        ref = load_image_cached(pr)
        mov = load_mask_cached(pm) if is_mask else load_image_cached(pm)
        warped = proc.apply_transform(mov, ref, tx, is_mask, interpolation)
        out = resolve_output(output, _derived(pm, "warped"))
        if is_mask:
            save_mask(warped.array, warped, out)
            if labels_for(pm):
                write_labels_sidecar(out, labels_for(pm))
        else:
            warped.save(out)
        record_provenance(
            "apply_transform",
            {"transform": display_path(pt), "moving": display_path(pm)},
            {"output": display_path(out)},
        )
        return {"output": display_path(out), "size_xyz": list(warped.size_xyz)}

    @server.tool()
    @tool_errors
    def combine_masks(
        masks: Annotated[list[str], Field(min_length=1, description="Label maps with identical geometry")],
        mode: Annotated[
            Literal["union", "intersection", "subtract", "label"],
            Field(
                description="union/intersection/subtract on foreground (binary result); label = mask i becomes label i+1"
            ),
        ] = "union",
        names: Annotated[list[str] | None, Field(description="label mode: names for the labels")] = None,
        output: str | None = None,
    ) -> dict[str, Any]:
        """Merge several masks into one (e.g. lobes -> lung, organ minus lesion, stack organs as labels)."""
        paths = [resolve(m) for m in masks]
        loaded = [load_mask_cached(p) for p in paths]
        arr = proc.combine([m.array for m in loaded], mode)
        out = resolve_output(output, _derived(paths[0], mode))
        save_mask(arr, loaded[0], out)
        labels = {}
        if mode == "label":
            labels = {
                i + 1: (names[i] if names and i < len(names) else paths[i].name.split(".")[0])
                for i in range(len(paths))
            }
        elif mode != "subtract":
            labels = {1: mode}
        if labels:
            write_labels_sidecar(out, labels)
        record_provenance(
            "combine_masks",
            {"masks": [display_path(p) for p in paths], "mode": mode},
            {"output": display_path(out)},
        )
        return {"output": display_path(out), "labels": labels, "foreground_voxels": int((arr != 0).sum())}

    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def mask_features(
        mask: str,
        image: Annotated[str | None, Field(description="Image for first-order intensity features")] = None,
        labels: Annotated[
            list[int] | None, Field(description="Label ids (default: every label present)")
        ] = None,
    ) -> dict[str, Any]:
        """Radiomics-style shape features (volume, surface area, sphericity, elongation, flatness,
        PCA axes, max extent) and first-order intensity statistics (percentiles, skewness, kurtosis,
        entropy) per label. Dependency-free; for full texture radiomics use pyradiomics on the same files."""
        import numpy as np

        mp = resolve(mask)
        img = load_image_cached(resolve(image)) if image else None
        m = load_mask_cached(mp, img)
        names = labels_for(mp)
        present = [int(v) for v in np.unique(m.array) if v != 0]
        wanted = [lab for lab in (labels or present) if lab in present]
        feats = {
            str(lab): {
                "name": names.get(lab),
                **proc.mask_features(m.array, m.spacing, img.scalar_array if img else None, lab),
            }
            for lab in wanted
        }
        return {
            "mask": display_path(mp),
            "spacing_xyz_mm": [round(float(s), 4) for s in m.spacing],
            "features": feats,
        }

    @server.tool()
    @tool_errors
    def mask_to_mesh(
        mask: str,
        label: Annotated[int | None, Field(description="Label to extract (default: all foreground)")] = None,
        output: Annotated[str | None, Field(description=".stl (binary) or .obj file")] = None,
        step: Annotated[
            int, Field(ge=1, le=8, description="Marching-cubes step size (larger = coarser mesh)")
        ] = 1,
        smooth: Annotated[
            int, Field(ge=0, le=5, description="Gaussian smoothing iterations before meshing")
        ] = 1,
    ) -> dict[str, Any]:
        """Export a surface mesh (STL/OBJ, physical mm coordinates) for 3D printing or 3D viewers."""
        mp = resolve(mask)
        m = load_mask_cached(mp)
        verts, faces = proc.mask_to_mesh(m, label, step, smooth)
        out = resolve_output(output, _derived(mp, f"label{label}" if label else "mesh", ".stl"))
        proc.write_mesh(verts, faces, out)
        record_provenance(
            "mask_to_mesh", {"mask": display_path(mp), "label": label}, {"output": display_path(out)}
        )
        return {
            "output": display_path(out),
            "vertices": int(len(verts)),
            "faces": int(len(faces)),
            "coordinate_space": "LPS mm",
        }

    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def list_dicom_series(
        folder: Annotated[str, Field(description="Directory containing DICOM files (searched recursively)")],
        recursive: bool = True,
    ) -> dict[str, Any]:
        """List every DICOM series in a folder with modality, description, slice count and geometry,
        so you can pick the right series before convert_image / inspect_image."""
        p = resolve(folder)
        series = proc.list_dicom_series(p, recursive)
        return {"folder": display_path(p), "n_series": len(series), "series": series}

"""Clinical measurement tools implementing widely used published criteria."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import Field

from open_med_mcp.config import get_settings
from open_med_mcp.core import clinical
from open_med_mcp.core.image import Plane
from open_med_mcp.tools._common import (
    labels_for,
    load_image_cached,
    load_mask_cached,
    resolve,
    result,
    tool_errors,
)
from open_med_mcp.viewer.registry import get_renderer
from open_med_mcp.viewer.spec import MaskLayer, ViewSpec

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def register(server: MCPServer) -> None:
    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def measure_lesion(
        mask: Annotated[str, Field(description="Label map of the lesion(s)")],
        image: Annotated[
            str | None, Field(description="Image for intensity statistics and the preview")
        ] = None,
        labels: Annotated[list[int] | None, Field(description="Label ids to measure (default: all)")] = None,
        plane: Annotated[
            Plane, Field(description="Plane in which diameters are measured (RECIST: axial)")
        ] = "axial",
        preview: bool = True,
    ) -> CallToolResult:
        """RECIST 1.1-style measurements per lesion: longest in-plane diameter, perpendicular short
        axis, the slice where it was measured, endpoints (native voxel coordinates), volume, and
        whether the lesion qualifies as a measurable target lesion."""
        mp = resolve(mask)
        img = load_image_cached(resolve(image)) if image else None
        m = load_mask_cached(mp, img)
        names = labels_for(mp)
        import numpy as np

        present = [int(v) for v in np.unique(m.array) if v]
        wanted = [lab for lab in (labels or present) if lab in present]
        rows = []
        for lab in wanted:
            r = clinical.measure_lesion(m, lab, plane, img)
            r["name"] = names.get(lab)
            rows.append(r)
        payload: dict[str, Any] = {
            "mask": mask,
            "plane": plane,
            "lesions": rows,
            "sum_of_longest_diameters_mm": round(
                sum(r.get("long_axis_mm", 0) for r in rows if r.get("measurable_target_lesion")), 1
            ),
            "reference": clinical.RECIST_REF,
        }
        images = []
        if preview and img is not None and rows and rows[0].get("long_axis_endpoints_xyz"):
            r0 = max(rows, key=lambda r: r.get("long_axis_mm", 0))
            a, b = r0["long_axis_endpoints_xyz"]
            prompts = [{"type": "point", "coords": a}, {"type": "point", "coords": b}]
            layer = MaskLayer(path=str(mp), name="lesion", mode="contour", alpha=0.0)
            spec = ViewSpec(
                image=str(img.path),
                masks=[layer],
                plane=plane,
                slices=[r0["slice"]] if not img.is_2d else None,
                prompts=prompts,
                title=f"longest diameter {r0['long_axis_mm']} mm (label {r0['label']})",
                crop_to_mask=True,
                crop_margin=20,
                max_px=get_settings().preview_max_px,
            )  # type: ignore[arg-type]
            images.append(get_renderer("png").render(img, [(m, {"layer": layer, "labels": names})], spec))
        return result(payload, images)

    @server.tool(annotations=READ_ONLY)
    def recist_response(
        baseline_sld_mm: Annotated[
            float, Field(description="Sum of longest diameters of target lesions at baseline (mm)")
        ],
        current_sld_mm: Annotated[float, Field(description="Current sum of longest diameters (mm)")],
        nadir_sld_mm: Annotated[
            float | None, Field(description="Smallest SLD recorded so far (default: baseline)")
        ] = None,
        new_lesions: bool = False,
        non_target_progression: Annotated[
            bool, Field(description="Unequivocal progression of non-target lesions")
        ] = False,
        all_nodes_short_axis_below_10mm: Annotated[
            bool, Field(description="Required for CR when lymph nodes were targets")
        ] = True,
    ) -> dict[str, Any]:
        """RECIST 1.1 response category (CR / PR / SD / PD) from target-lesion sums with the percent
        changes versus baseline and nadir."""
        return clinical.recist_response(
            baseline_sld_mm,
            current_sld_mm,
            nadir_sld_mm,
            new_lesions,
            non_target_progression,
            all_nodes_short_axis_below_10mm,
        )

    @server.tool(annotations=READ_ONLY)
    def fleischner_recommendation(
        nodule_type: Literal["solid", "part-solid", "ground-glass"],
        size_mm: Annotated[float, Field(description="Average of long and short axis diameters, nearest mm")],
        multiple: Annotated[bool, Field(description="More than one nodule")] = False,
        high_risk: Annotated[
            bool,
            Field(
                description="Risk factors per the guideline (smoking, family history, upper-lobe location, spiculation ...)"
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Fleischner Society 2017 follow-up recommendation for an incidental pulmonary nodule."""
        return clinical.fleischner_recommendation(nodule_type, size_mm, multiple, high_risk)

    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def tirads_score(
        composition: Annotated[
            Literal["cystic", "spongiform", "mixed", "solid"], Field(description="Nodule composition")
        ],
        echogenicity: Annotated[
            Literal["anechoic", "hyperechoic", "isoechoic", "hypoechoic", "very hypoechoic"],
            Field(description="Echogenicity relative to thyroid parenchyma"),
        ],
        shape: Literal["wider-than-tall", "taller-than-wide"],
        margin: Literal["smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension"],
        echogenic_foci: Annotated[
            list[Literal["none", "comet-tail", "macrocalcifications", "peripheral", "rim", "punctate"]]
            | None,
            Field(description="All that apply"),
        ] = None,
        max_diameter_mm: Annotated[
            float | None, Field(description="Largest diameter for the FNA / follow-up decision")
        ] = None,
    ) -> dict[str, Any]:
        """ACR TI-RADS (2017) points, level and size-based recommendation for a thyroid nodule."""
        return clinical.tirads_score(
            composition, echogenicity, shape, margin, list(echogenic_foci or []), max_diameter_mm
        )

    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def agatston_score(
        image: Annotated[str, Field(description="Non-contrast ECG-gated cardiac CT (HU)")],
        mask: Annotated[
            str,
            Field(
                description="Mask of the coronary arteries or the heart region (e.g. totalsegmentator coronary_arteries / heart)"
            ),
        ],
        threshold_hu: float = 130.0,
        min_area_mm2: float = 1.0,
    ) -> dict[str, Any]:
        """Agatston coronary artery calcium score inside a mask (per label and total) with the usual
        severity categories; requires non-contrast ECG-gated CT."""
        ip, mp = resolve(image), resolve(mask)
        img = load_image_cached(ip)
        m = load_mask_cached(mp, img)
        return clinical.agatston_score(img, m, threshold_hu, min_area_mm2, labels_for(mp))

    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def cardiothoracic_ratio(
        mask: Annotated[
            str,
            Field(
                description="Chest X-ray anatomy mask (e.g. from run_model('torchxrayvision', task='segment'))"
            ),
        ],
        heart_labels: Annotated[
            list[int] | None, Field(description="Label ids of the heart (default: labels named 'Heart')")
        ] = None,
        thorax_labels: Annotated[
            list[int] | None, Field(description="Label ids spanning the thorax (default: left + right lung)")
        ] = None,
    ) -> dict[str, Any]:
        """Cardiothoracic ratio from a frontal chest radiograph anatomy mask (heart width / thoracic width)."""
        mp = resolve(mask)
        m = load_mask_cached(mp)
        names = {k: v.lower() for k, v in labels_for(mp).items()}
        heart = heart_labels or [k for k, v in names.items() if "heart" in v]
        thorax = thorax_labels or [k for k, v in names.items() if "lung" in v]
        if not heart or not thorax:
            raise ValueError("could not infer heart/thorax labels; pass heart_labels and thorax_labels")
        out = clinical.cardiothoracic_ratio(m, heart, thorax)
        out.update({"heart_labels": heart, "thorax_labels": thorax})
        return out

    @server.tool(annotations=READ_ONLY)
    def future_liver_remnant(
        total_liver_ml: float,
        remnant_ml: Annotated[
            float, Field(description="Volume of the liver that will remain after resection (mL)")
        ],
        tumor_ml: Annotated[
            float,
            Field(description="Tumor volume inside the liver (mL), subtracted from the functional liver"),
        ] = 0.0,
        liver_condition: Literal["healthy", "chemotherapy", "cirrhosis"] = "healthy",
        body_weight_kg: float | None = None,
    ) -> dict[str, Any]:
        """Future liver remnant (FLR %) with the commonly used adequacy thresholds."""
        return clinical.future_liver_remnant(
            total_liver_ml, remnant_ml, tumor_ml, liver_condition, body_weight_kg
        )

    @server.tool(annotations=READ_ONLY)
    def mayo_adpkd_class(
        total_kidney_volume_ml: float,
        height_m: float,
        age_years: float,
    ) -> dict[str, Any]:
        """Mayo Imaging Classification (1A-1E) of typical ADPKD from height-adjusted total kidney volume and age."""
        return clinical.mayo_adpkd_class(total_kidney_volume_ml, height_m, age_years)

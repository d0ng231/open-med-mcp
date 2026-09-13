"""Model zoo tools: discovery, weights, running models (segmentation / classification / batch)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import Field

from open_med_mcp.config import get_settings
from open_med_mcp.core.image import MedicalImage, Plane, write_labels_sidecar
from open_med_mcp.core.masks import mask_stats
from open_med_mcp.core.prompts import Prompt, normalize_prompts
from open_med_mcp.models.job import prepare_job, read_response
from open_med_mcp.models.manifest import ModelManifest
from open_med_mcp.models.registry import get_registry
from open_med_mcp.models.runners import resolve_device, runner_report, select_runner
from open_med_mcp.models.weights import ensure_weights, missing_weights, weights_dir_for, weights_status
from open_med_mcp.tools._common import (
    load_image_cached,
    load_mask_cached,
    resolve,
    resolve_output,
    result,
    tool_errors,
)
from open_med_mcp.viewer.registry import RenderResult, get_renderer
from open_med_mcp.viewer.spec import MaskLayer, ViewSpec
from open_med_mcp.workspace import Timer, display_path, new_run_dir, record_provenance, stage_file

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _preview(image: MedicalImage, mask_path: Path, labels: dict[int, str], title: str) -> RenderResult:
    settings = get_settings()
    mask = load_mask_cached(mask_path, image)
    spec = ViewSpec(
        image=str(image.path),
        masks=[MaskLayer(path=str(mask_path), name="result")],
        layout="three-plane" if not image.is_2d else "single",
        title=title,
        label_names=labels or None,
        max_px=settings.preview_max_px,
    )
    return get_renderer("png").render(image, [(mask, {"layer": spec.masks[0], "labels": labels})], spec)


def run_job(
    manifest: ModelManifest, task: str, inputs: dict[str, Path], params: dict[str, Any]
) -> tuple[Path, dict[str, Any], str, str, float]:
    """Prepare and execute one job. Returns ``(run_dir, response, runner_name, device, wall_seconds)``."""
    settings = get_settings()
    settings.ensure_dirs()
    device = resolve_device(settings)
    wdir = weights_dir_for(manifest, settings)
    if manifest.weights and settings.auto_download_weights:
        ids = (
            [params["variant"]]
            if params.get("variant")
            else ([manifest.weights[0].id] if not manifest.is_promptable else None)
        )
        ensure_weights(manifest, ids, settings)
        still = [w.id for w in missing_weights(manifest, ids, settings)] if ids else []
        if still:
            raise RuntimeError(
                f"weights {still} are missing in {wdir}; run download_weights('{manifest.name}')"
            )
    runner = select_runner(manifest, settings)
    run_dir = new_run_dir(manifest.name, settings)
    prepare_job(run_dir, manifest, task, inputs, params, device, wdir)
    with Timer() as timer:
        outcome = runner.run(manifest, run_dir, wdir, device)
    if outcome.returncode != 0 and not (run_dir / "response.json").exists():
        tail = (
            (run_dir / "log.txt").read_text(errors="replace")[-3000:]
            if (run_dir / "log.txt").exists()
            else outcome.stderr_tail
        )
        raise RuntimeError(f"{runner.name} runner exited with code {outcome.returncode}\n{tail}")
    return run_dir, read_response(run_dir), runner.name, device, timer.seconds


def execute_model(
    manifest: ModelManifest,
    task: str,
    inputs: dict[str, Path],
    params: dict[str, Any],
    output: str | None,
    preview: bool,
    tool_name: str,
    extra: dict[str, Any] | None = None,
) -> CallToolResult:
    run_dir, resp, runner_name, device, wall = run_job(manifest, task, inputs, params)
    outputs = {k: Path(v) for k, v in resp["outputs"].items()}
    labels = {int(k): v for k, v in (resp.get("labels") or {}).items()}
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "model": manifest.name,
        "task": task,
        "category": manifest.category,
        "runner": runner_name,
        "device": device,
        "run_dir": display_path(run_dir),
        "outputs": {k: display_path(v) for k, v in outputs.items()},
        "labels": labels,
        "params": {k: v for k, v in request["params"].items() if k != "prompts"},
        "stats": resp.get("stats") or {},
        "model_info": resp.get("model_info") or {},
        "warnings": list(resp.get("warnings") or []),
        "timing": {**(resp.get("timing") or {}), "wall_s": wall},
    }
    if extra:
        payload.update(extra)
    # JSON outputs (classification, detection ...) are inlined
    for key, spec in manifest.outputs.items():
        if spec.kind == "json" and key in outputs and outputs[key].exists():
            try:
                payload.setdefault("results", {})[key] = json.loads(outputs[key].read_text(encoding="utf-8"))
            except Exception as exc:
                payload["warnings"].append(f"could not parse {key}: {exc}")
    images: list[RenderResult] = []
    if "mask" in outputs:
        mask_path = outputs["mask"]
        if labels:
            write_labels_sidecar(mask_path, labels)
        if output:
            dst = resolve_output(output, mask_path)
            stage_file(mask_path, dst)
            if labels:
                write_labels_sidecar(dst, labels)
            payload["outputs"]["mask"] = display_path(dst)
            mask_path = dst
        image_input = inputs.get("image")
        if image_input is not None:
            image = load_image_cached(Path(image_input))
            try:
                mask = load_mask_cached(mask_path, image)
                payload["mask_stats"] = mask_stats(mask, image, label_names=labels)
                if preview:
                    images.append(_preview(image, mask_path, labels, f"{manifest.name} - {task}"))
            except Exception as exc:  # preview must never fail the run
                payload["warnings"].append(f"preview/stats skipped: {exc}")
    record_provenance(
        tool_name,
        {
            "model": manifest.name,
            "task": task,
            "inputs": {k: display_path(v) for k, v in inputs.items()},
            "params": request["params"],
        },
        payload["outputs"],
        run_dir=display_path(run_dir),
        runner=runner_name,
    )
    return result(payload, images)


def _bar_chart(probabilities: dict[str, float], title: str, top: int = 18) -> RenderResult:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(probabilities.items(), key=lambda kv: kv[1])[-top:]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(items) + 1.2), dpi=120)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    colors = ["#ff3b30" if v >= 0.5 else "#0a84ff" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="#ffffff", alpha=0.3, linewidth=0.8, linestyle="--")
    ax.tick_params(colors="#dddddd", labelsize=8)
    ax.set_xlabel("probability", color="#dddddd", fontsize=8)
    for sp in ax.spines.values():
        sp.set_color("#555555")
    ax.set_title(title, color="white", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(
            min(v + 0.01, 0.97),
            i,
            f"{v:.2f}",
            va="center",
            ha="left" if v < 0.85 else "right",
            color="white",
            fontsize=7,
        )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return RenderResult(buf.getvalue(), "image/png", ".png", {"kind": "bar-chart"})


def register(server: MCPServer) -> None:
    @server.tool(annotations=READ_ONLY)
    def list_models(
        task: Annotated[
            str | None, Field(description="Filter by task name (e.g. segment, total, lobes)")
        ] = None,
        modality: Annotated[
            str | None, Field(description="Filter by modality (CT, MR, XR, US, RGB ...)")
        ] = None,
        category: Annotated[
            str | None, Field(description="segmentation | classification | preprocessing | detection")
        ] = None,
        check_backends: Annotated[
            bool, Field(description="Also report which execution backends can run each model (slower)")
        ] = False,
    ) -> dict[str, Any]:
        """List available models (segmentation, classification, preprocessing) with tasks, modalities
        and prompt types. Use describe_model for parameters."""
        settings = get_settings()
        reg = get_registry(settings)
        items = []
        for m in reg.all():
            if task and task not in m.tasks:
                continue
            if modality and not ({modality.lower(), "any"} & {x.lower() for x in m.modalities}):
                continue
            if category and m.category != category:
                continue
            s = m.summary()
            if check_backends:
                s["backends"] = runner_report(m, settings)
            items.append(s)
        return {
            "models": items,
            "runner_setting": settings.runner,
            "device": resolve_device(settings),
            "manifest_errors": reg.errors,
        }

    @server.tool(annotations=READ_ONLY)
    @tool_errors
    def describe_model(
        name: Annotated[str, Field(description="Model name from list_models")],
    ) -> dict[str, Any]:
        """Full description of a model: tasks, parameters (JSON schema), outputs, weights status,
        backends, license and citation."""
        settings = get_settings()
        m = get_registry(settings).get(name)
        return {
            **m.model_dump(mode="json", exclude={"adapter_dir"}),
            "params_schema": m.params_json_schema(),
            "weights_status": weights_status(m, settings),
            "backends": runner_report(m, settings),
            "container_image": m.container.image or f"{settings.image_prefix}-{m.adapter}:{m.version}",
        }

    @server.tool()
    @tool_errors
    def download_weights(
        model: str,
        weights: Annotated[
            list[str] | None, Field(description="Weight ids to fetch (default: all declared)")
        ] = None,
    ) -> dict[str, Any]:
        """Download a model's weights into the local weights directory (needed once per machine).
        Models that fetch their own weights on first run report that instead."""
        settings = get_settings()
        settings.ensure_dirs()
        m = get_registry(settings).get(model)
        if not m.weights:
            return {
                "model": model,
                "note": "this model downloads its own weights on first run (or use its 'download' task via run_model)",
                "weights_dir": str(weights_dir_for(m, settings)),
                "tasks": m.tasks,
            }
        fetched = ensure_weights(m, weights, settings)
        return {
            "model": model,
            "downloaded": [str(p) for p in fetched],
            "status": weights_status(m, settings),
        }

    @server.tool()
    @tool_errors
    def run_model(
        model: Annotated[
            str,
            Field(
                description="Model name, e.g. totalsegmentator, lungmask, hdbet, synthstrip, nnunet, monai, classical"
            ),
        ],
        inputs: Annotated[
            dict[str, str],
            Field(description="Input files by role, usually {'image': 'path'} (some tasks need none)"),
        ],
        task: Annotated[
            str | None,
            Field(description="Task name (see describe_model); default = the model's default task"),
        ] = None,
        params: Annotated[
            dict[str, Any] | None, Field(description="Model parameters (see describe_model params_schema)")
        ] = None,
        output: Annotated[str | None, Field(description="Copy the main mask output to this path")] = None,
        preview: Annotated[
            bool, Field(description="Return a three-plane preview PNG when the result is a mask")
        ] = True,
    ) -> CallToolResult:
        """Run any model of the zoo through the job contract (local / Docker / Apptainer backend chosen
        automatically). Returns output paths, label names, statistics, inlined JSON results and a preview."""
        settings = get_settings()
        m = get_registry(settings).get(model)
        task = task or m.default_task
        paths = {k: resolve(v) for k, v in inputs.items()}
        return execute_model(m, task, paths, dict(params or {}), output, preview, "run_model")

    @server.tool()
    @tool_errors
    def segment(
        image: Annotated[str, Field(description="Image path (2D or 3D)")],
        model: Annotated[
            str,
            Field(
                description="Promptable model (medsam2, sam2) or automatic model (totalsegmentator, lungmask, classical ...)"
            ),
        ] = "medsam2",
        prompts: Annotated[
            list[Prompt] | None,
            Field(
                description="Point/box prompts in native voxel coordinates (required for promptable models)"
            ),
        ] = None,
        plane: Annotated[
            Plane, Field(description="3D: plane along which 2D prompts are placed and the mask is propagated")
        ] = "axial",
        params: Annotated[
            dict[str, Any] | None,
            Field(
                description="Extra model parameters, e.g. {'variant': 'medsam2_ct_lesion', 'window': 'lung', 'max_slices': 20}"
            ),
        ] = None,
        task: Annotated[
            str | None,
            Field(
                description="Task for automatic models (e.g. 'threshold' for classical, 'lobes' for lungmask)"
            ),
        ] = None,
        output: Annotated[str | None, Field(description="Copy the mask to this path")] = None,
        preview: bool = True,
    ) -> CallToolResult:
        """Segment a structure. For promptable models give at least one box or point; the result is a
        label map (label = object_id), statistics and a preview. Look at the preview, then refine."""
        settings = get_settings()
        m = get_registry(settings).get(model)
        if m.category not in ("segmentation", "preprocessing"):
            raise ValueError(f"model {model!r} is a {m.category} model; use run_model or classify_image")
        p = resolve(image)
        img = load_image_cached(p)
        params = dict(params or {})
        extra: dict[str, Any] = {}
        if m.is_promptable:
            if not prompts:
                raise ValueError(f"model {model!r} needs prompts (points/boxes); see get_conventions()")
            wire = normalize_prompts(prompts, img, plane)
            params["prompts"] = wire
            params["axis"] = img.numpy_axis_for_plane(plane)
            extra = {"prompts": [pp.model_dump() for pp in prompts], "plane": plane, "wire_prompts": wire}
        task = task or m.default_task
        return execute_model(m, task, {"image": p}, params, output, preview, "segment", extra)

    @server.tool()
    @tool_errors
    def classify_image(
        image: Annotated[str, Field(description="Image path (e.g. a chest radiograph)")],
        model: Annotated[
            str, Field(description="Classification model, e.g. torchxrayvision")
        ] = "torchxrayvision",
        params: dict[str, Any] | None = None,
        preview: Annotated[bool, Field(description="Return a bar chart of the probabilities")] = True,
    ) -> CallToolResult:
        """Run a classification model and return per-class probabilities (plus a bar chart)."""
        settings = get_settings()
        m = get_registry(settings).get(model)
        if m.category != "classification":
            raise ValueError(f"model {model!r} is a {m.category} model; use segment or run_model")
        p = resolve(image)
        res = execute_model(
            m, m.default_task, {"image": p}, dict(params or {}), None, False, "classify_image"
        )
        if res.is_error or not preview:
            return res
        payload = res.structured_content or {}
        results = payload.get("results") or {}
        probs = None
        for v in results.values():
            if isinstance(v, dict) and isinstance(v.get("probabilities"), dict):
                probs = v["probabilities"]
                break
        if probs:
            chart = _bar_chart(probs, f"{model}: {p.name}")
            return result(payload, [chart])
        return res

    @server.tool()
    @tool_errors
    def run_batch(
        model: str,
        images: Annotated[list[str] | None, Field(description="Explicit image paths")] = None,
        pattern: Annotated[
            str | None, Field(description="Glob pattern relative to the workspace, e.g. 'cohort/*.nii.gz'")
        ] = None,
        task: str | None = None,
        params: dict[str, Any] | None = None,
        output_dir: Annotated[
            str | None, Field(description="Copy each mask here as <image name>_<model>.nii.gz")
        ] = None,
        max_cases: Annotated[int, Field(ge=1, le=10000)] = 500,
        continue_on_error: bool = True,
    ) -> dict[str, Any]:
        """Run one model over many images (cohort processing). Returns a per-case table with status,
        run directory, foreground volume and per-label volumes, and writes it as CSV + JSON."""
        settings = get_settings()
        m = get_registry(settings).get(model)
        task = task or m.default_task
        paths: list[Path] = [resolve(i) for i in (images or [])]
        if pattern:
            paths += sorted(p for p in settings.workspace.glob(pattern) if p.is_file() or p.is_dir())
        paths = list(dict.fromkeys(paths))[:max_cases]
        if not paths:
            raise ValueError("no images matched")
        out_dir = resolve_output(output_dir, settings.outputs_dir / "batch") if output_dir else None
        rows: list[dict[str, Any]] = []
        for p in paths:
            row: dict[str, Any] = {"image": display_path(p), "status": "ok"}
            try:
                run_dir, resp, runner_name, device, wall = run_job(m, task, {"image": p}, dict(params or {}))
                row.update({"run_dir": display_path(run_dir), "runner": runner_name, "seconds": wall})
                labels = {int(k): v for k, v in (resp.get("labels") or {}).items()}
                stats = resp.get("stats") or {}
                if "mask" in resp["outputs"]:
                    mp = Path(resp["outputs"]["mask"])
                    if labels:
                        write_labels_sidecar(mp, labels)
                    if out_dir is not None:
                        dst = out_dir / f"{p.name.split('.')[0]}_{model}.nii.gz"
                        stage_file(mp, dst)
                        if labels:
                            write_labels_sidecar(dst, labels)
                        mp = dst
                    row["mask"] = display_path(mp)
                    row["foreground_ml"] = round(
                        float(stats.get("foreground_voxels", 0))
                        * float(stats.get("voxel_volume_mm3", 0))
                        / 1000,
                        3,
                    )
                    for lab, info in (stats.get("per_label") or {}).items():
                        row[f"{labels.get(int(lab), 'label_' + str(lab))}_ml"] = info.get("volume_ml")
                for key, spec in m.outputs.items():
                    if spec.kind == "json" and key in resp["outputs"]:
                        try:
                            data = json.loads(Path(resp["outputs"][key]).read_text(encoding="utf-8"))
                            for t in (data.get("top") or [])[:3]:
                                row[f"top_{t['finding']}"] = t["probability"]
                        except Exception:
                            pass
            except Exception as exc:  # noqa: BLE001
                row.update({"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                if not continue_on_error:
                    rows.append(row)
                    break
            rows.append(row)
        summary_dir = out_dir or (settings.outputs_dir / "batch")
        summary_dir.mkdir(parents=True, exist_ok=True)
        stamp = new_run_dir("batch", settings).name  # unique id (dir kept as a marker)
        csv_path = summary_dir / f"{model}_{stamp}.csv"
        cols: list[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        (csv_path.with_suffix(".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        n_ok = sum(1 for r in rows if r["status"] == "ok")
        record_provenance(
            "run_batch", {"model": model, "task": task, "n": len(rows)}, {"csv": display_path(csv_path)}
        )
        return {
            "model": model,
            "task": task,
            "n_cases": len(rows),
            "n_ok": n_ok,
            "n_failed": len(rows) - n_ok,
            "csv": display_path(csv_path),
            "json": display_path(csv_path.with_suffix(".json")),
            "rows": rows[:50],
        }

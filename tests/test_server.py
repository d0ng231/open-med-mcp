import pytest
from mcp import Client
from mcp.types import ImageContent, TextContent

from open_med_mcp.server import create_server


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(isolated_settings):
    server = create_server(isolated_settings)
    async with Client(server, raise_exceptions=True) as c:
        yield c


@pytest.mark.anyio
async def test_catalogue(client: Client):
    tools = {t.name for t in (await client.list_tools()).tools}
    assert {
        "inspect_image",
        "segment",
        "run_model",
        "render_view",
        "mask_stats",
        "postprocess_mask",
        "compare_masks",
        "mask_to_prompts",
        "list_guidelines",
        "get_guideline",
        "get_conventions",
        "write_report",
        "export_viewer",
        "list_models",
        "describe_model",
        "download_weights",
        "list_workspace",
        "convert_image",
    } <= tools
    prompts = {p.name for p in (await client.list_prompts()).prompts}
    assert "segmentation-3d-ct" in prompts
    res = await client.read_resource("guideline://qc-checklist")
    assert "Quality-control" in res.contents[0].text
    res = await client.read_resource("omm://conventions")
    assert "Native voxel index" in res.contents[0].text
    res = await client.read_resource("model://classical")
    assert '"name": "classical"' in res.contents[0].text
    assert "open-med-mcp" in (client.instructions or "")


@pytest.mark.anyio
async def test_end_to_end_classical(client: Client, ct_volume):
    r = await client.call_tool("inspect_image", {"path": "ct.nii.gz"})
    assert not r.is_error and isinstance(r.content[0], TextContent) and isinstance(r.content[1], ImageContent)
    assert r.structured_content["size_xyz"] == [48, 40, 24] and r.structured_content["modality_guess"] == "CT"

    r = await client.call_tool(
        "segment",
        {
            "image": "ct.nii.gz",
            "model": "classical",
            "task": "threshold",
            "params": {"lower": 200, "upper": 1000, "keep": "largest"},
        },
    )
    assert not r.is_error, r.content[0].text
    sc = r.structured_content
    mask = sc["outputs"]["mask"]
    assert mask.startswith("omm_outputs/") and sc["runner"] == "local" and sc["mask_stats"]["n_labels"] == 1
    assert any(isinstance(c, ImageContent) for c in r.content)

    r = await client.call_tool("mask_stats", {"mask": mask, "image": "ct.nii.gz"})
    assert r.structured_content["labels"][0]["intensity"]["mean"] > 200

    r = await client.call_tool("mask_to_prompts", {"mask": mask, "plane": "axial"})
    prompts = r.structured_content["prompts"]
    assert prompts[0]["type"] == "box" and len(prompts[0]["coords"]) == 6

    r = await client.call_tool(
        "render_view",
        {
            "image": "ct.nii.gz",
            "masks": [mask],
            "layout": "three-plane",
            "prompts": prompts,
            "window": "soft-tissue",
        },
    )
    assert (
        not r.is_error
        and r.structured_content["output"].endswith(".png")
        and isinstance(r.content[1], ImageContent)
    )
    r = await client.call_tool(
        "render_view",
        {"image": "ct.nii.gz", "masks": [mask], "layout": "montage", "n_slices": 4, "renderer": "html"},
    )
    assert r.structured_content["output"].endswith(".html")

    r = await client.call_tool(
        "postprocess_mask",
        {
            "mask": mask,
            "ops": ["fill_holes", {"op": "dilate", "radius": 1}],
            "image": "ct.nii.gz",
            "preview": True,
        },
    )
    pp = r.structured_content["output"]
    assert r.structured_content["after_voxels"] > r.structured_content["before_voxels"]

    r = await client.call_tool("compare_masks", {"mask_a": mask, "mask_b": pp, "image": "ct.nii.gz"})
    assert (
        0 < r.structured_content["foreground"]["dice"] < 1 and "hd95_mm" in r.structured_content["foreground"]
    )
    r = await client.call_tool(
        "compare_masks",
        {"mask_a": mask, "mask_b": "ball_mask.nii.gz", "surface_metrics": False, "preview": False},
    )
    assert r.structured_content["foreground"]["dice"] > 0.9

    r = await client.call_tool("export_viewer", {"image": "ct.nii.gz", "masks": [mask]})
    assert r.structured_content["output"].endswith("_viewer.html")

    r = await client.call_tool(
        "write_report",
        {
            "title": "Ball",
            "sections": [{"heading": "Result", "text": "ok", "table": [{"volume_ml": 1.5}]}],
            "metadata": {"image": "ct.nii.gz"},
        },
    )
    assert r.structured_content["markdown"].endswith(".md") and (client and True)

    r = await client.call_tool("list_workspace", {"pattern": "*.nii.gz"})
    assert any(i["path"] == "ct.nii.gz" and i["type"] == "image" for i in r.structured_content["items"])
    r = await client.call_tool("convert_image", {"path": "ct.nii.gz", "output": "copy.nrrd"})
    assert r.structured_content["output"] == "copy.nrrd"


@pytest.mark.anyio
async def test_2d_and_errors(client: Client, blob_png):
    r = await client.call_tool("segment", {"image": "blob.png", "model": "classical", "task": "otsu"})
    assert not r.is_error
    assert r.structured_content["outputs"]["mask"].endswith(".png")
    r = await client.call_tool(
        "compare_masks",
        {
            "mask_a": r.structured_content["outputs"]["mask"],
            "mask_b": "blob_mask.png",
            "surface_metrics": False,
            "preview": False,
        },
    )
    assert r.structured_content["foreground"]["dice"] > 0.95
    r = await client.call_tool("segment", {"image": "blob.png", "model": "medsam2"})
    assert r.is_error and "needs prompts" in r.content[0].text
    r = await client.call_tool(
        "segment", {"image": "blob.png", "model": "medsam2", "prompts": [{"type": "text", "text": "liver"}]}
    )
    assert r.is_error and "accepts" in r.content[0].text
    r = await client.call_tool("segment", {"image": "blob.png", "model": "voxtell"})
    assert r.is_error and "text" in r.content[0].text
    r = await client.call_tool("segment", {"image": "missing.png", "model": "classical"})
    assert r.is_error and "does not exist" in r.content[0].text
    r = await client.call_tool("describe_model", {"name": "nope"})
    assert r.is_error
    r = await client.call_tool("get_guideline", {"name": "segmentation-2d-prompted"})
    assert "box" in r.content[0].text
    r = await client.get_prompt("qc-checklist", {"image": "blob.png"})
    assert "blob.png" in r.messages[0].content.text


@pytest.mark.anyio
async def test_list_and_describe_models(client: Client):
    r = await client.call_tool("list_models", {"modality": "CT"})
    names = {m["name"] for m in r.structured_content["models"]}
    assert {"classical", "medsam2", "totalsegmentator"} <= names
    r = await client.call_tool("describe_model", {"name": "sam2"})
    assert "params_schema" in r.structured_content and "backends" in r.structured_content
    r = await client.call_tool("list_guidelines", {"tags": ["qc"]})
    assert "qc-checklist" in [g["name"] for g in r.structured_content["guidelines"]]

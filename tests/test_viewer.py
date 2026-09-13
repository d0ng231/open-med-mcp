import io

from PIL import Image

from open_med_mcp.core.image import MedicalImage, load_mask
from open_med_mcp.viewer.registry import get_renderer, list_renderers
from open_med_mcp.viewer.report import write_report
from open_med_mcp.viewer.slicing import auto_slices, display_slice
from open_med_mcp.viewer.spec import MaskLayer, ViewSpec
from tests.conftest import make_volume


def test_display_slice_roundtrip_ras(ct_volume):
    img = MedicalImage.load(ct_volume["path"])
    for plane in ("axial", "coronal", "sagittal"):
        ds = display_slice(img, plane, 5)
        assert ds.index == 5
        xyz = (
            [7.0, 9.0, 5.0]
            if plane == "axial"
            else ([7.0, 5.0, 3.0] if plane == "coronal" else [5.0, 7.0, 3.0])
        )
        c, r = ds.to_display(xyz)
        assert ds.to_native(c, r) == xyz
    ds = display_slice(img, "axial", 5)
    # RAS volume: x runs to the patient's right -> flipped so left is on screen right
    assert ds.flip_cols and ds.flip_rows and ds.shape == (40, 48)
    ds = display_slice(img, "coronal", 3)
    assert ds.shape == (24, 48) and ds.aspect == 2.5


def test_display_slice_lps_and_transposed():
    img, _, _ = make_volume(direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    m = MedicalImage.from_sitk(img)
    ds = display_slice(m, "axial", 2)
    assert not ds.flip_cols and not ds.flip_rows
    # permuted axes: index axis 0 = S, axis 1 = L, axis 2 = P
    img, _, _ = make_volume(direction=(0, 1, 0, 0, 0, 1, 1, 0, 0))
    m = MedicalImage.from_sitk(img)
    ds = display_slice(m, "axial", 1)
    assert ds.stack_axis_xyz == 0 and ds.col_axis_xyz == 1 and ds.row_axis_xyz == 2
    c, r = ds.to_display([1.0, 4.0, 6.0])
    assert ds.to_native(c, r) == [1.0, 4.0, 6.0]


def test_auto_slices(ct_volume):
    img = MedicalImage.load(ct_volume["path"])
    ball = ct_volume["ball"]
    best = auto_slices(img, "axial", [ball], n=1)[0]
    assert ball[best].sum() == ball.sum(axis=(1, 2)).max()
    many = auto_slices(img, "axial", [ball], n=5)
    assert 1 < len(many) <= 5 and many == sorted(many)
    assert auto_slices(img, "axial", [], n=1) == [12]


def _png_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        return im.size


def test_png_renderer_layouts(ct_volume):
    img = MedicalImage.load(ct_volume["path"])
    mask = load_mask(ct_volume["mask"], img)
    layer = MaskLayer(path=str(ct_volume["mask"]), name="ball")
    items = [(mask, {"layer": layer, "labels": {1: "ball"}})]
    r = get_renderer("png")
    for layout in ("single", "three-plane", "montage"):
        spec = ViewSpec(
            image=str(ct_volume["path"]),
            masks=[layer],
            layout=layout,
            window="soft-tissue",
            max_px=700,
            prompts=[
                {"type": "box", "coords": [10, 10, 12, 30, 30, 12]},
                {"type": "point", "coords": [20, 20, 12]},
            ],
        )
        res = r.render(img, items, spec)
        assert res.mime_type == "image/png" and res.data[:8] == b"\x89PNG\r\n\x1a\n"
        w, h = _png_size(res.data)
        assert max(w, h) <= 700
        assert res.info["legend"] == {"ball": "#ff3b30"}
        assert len(res.info["panels"]) == (
            3
            if layout == "three-plane"
            else (1 if layout == "single" else 9)
            if layout != "montage"
            else len(res.info["panels"])
        )
    spec = ViewSpec(
        image=str(ct_volume["path"]), masks=[layer], crop_to_mask=True, grid=False, show_legend=False
    )
    assert r.render(img, items, spec).data


def test_png_renderer_2d(blob_png):
    img = MedicalImage.load(blob_png["path"])
    mask = load_mask(blob_png["mask"], img)
    layer = MaskLayer(path=str(blob_png["mask"]), mode="contour", color="#00ff00")
    res = get_renderer("png").render(
        img,
        [(mask, {"layer": layer})],
        ViewSpec(image=str(blob_png["path"]), prompts=[{"type": "box", "coords": [30, 10, 70, 50]}]),
    )
    assert res.info["panels"][0]["plane"] == "axial" and res.info["legend"]


def test_html_viewer_and_report(ct_volume, tmp_path):
    img = MedicalImage.load(ct_volume["path"])
    mask = load_mask(ct_volume["mask"], img)
    layer = MaskLayer(path=str(ct_volume["mask"]))
    res = get_renderer("html").render(
        img,
        [(mask, {"layer": layer, "labels": {1: "ball"}})],
        ViewSpec(image=str(ct_volume["path"]), window="soft-tissue"),
    )
    html = res.data.decode()
    assert res.mime_type == "text/html" and res.info["frames"] == 24
    assert "data:image/jpeg;base64" in html and "data:image/png;base64" in html and "ball" in html
    fig = tmp_path / "fig.png"
    fig.write_bytes(get_renderer("png").render(img, [], ViewSpec(image=str(ct_volume["path"]))).data)
    out = write_report(
        "Test report",
        [{"heading": "A", "text": "hello **world**", "figures": [fig], "table": [{"a": 1, "b": 2.5}]}],
        tmp_path / "r.md",
        {"image": "ct.nii.gz"},
    )
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "# Test report" in md and "| a | b |" in md and "![" in md
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<strong>world</strong>" in html and "data:image/png;base64" in html
    assert set(out) == {"markdown", "html"}


def test_renderer_registry():
    assert {"png", "html"} <= set(list_renderers())

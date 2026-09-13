from open_med_mcp.guidelines.loader import GuidelineLibrary, get_library, parse_guideline


def test_presets(isolated_settings):
    lib = get_library(isolated_settings, reload=True)
    names = lib.names()
    assert {"getting-started", "segmentation-3d-ct", "segmentation-2d-prompted", "qc-checklist"} <= set(names)
    g = lib.get("segmentation_3d_ct")  # tolerant lookup
    assert g.title.startswith("Segment an organ") and "CT" in g.modalities and g.source == "preset"
    assert g.render().startswith("# Segment an organ")
    assert [x.name for x in lib.search(modality="MR")] and all(
        "MR" in x.modalities or "any" in x.modalities for x in lib.search(modality="MR")
    )
    assert lib.search(query="dice")[0].name == "compare-two-segmentations"


def test_user_override_and_front_matter(isolated_settings, tmp_path):
    d = isolated_settings.workspace / "omm_guidelines"
    d.mkdir()
    (d / "qc-checklist.md").write_text(
        "---\nname: qc-checklist\ntitle: My QC\ntags: [custom]\n---\n# My QC\n\nDo the thing.\n"
    )
    (d / "plain.md").write_text("# Plain guideline\n\nFirst paragraph is the summary.\n\nMore.")
    lib = GuidelineLibrary(isolated_settings)
    g = lib.get("qc-checklist")
    assert (
        g.source == "user"
        and g.title == "My QC"
        and g.tags == ["custom"]
        and g.body.strip() == "Do the thing."
    )
    p = lib.get("plain")
    assert p.title == "Plain guideline" and p.summary == "First paragraph is the summary."
    g2 = parse_guideline(d / "plain.md")
    assert g2.name == "plain"

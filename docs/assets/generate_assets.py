"""Generate the logo, banner and architecture diagram (SVG + PNG) with matplotlib.

Run: python docs/assets/generate_assets.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

HERE = Path(__file__).resolve().parent
INK = "#0f172a"
TEAL = "#0ea5a4"
CORAL = "#ff6b6b"
SKY = "#38bdf8"
PAPER = "#f8fafc"


NAVY = "#0f172a"
NAVY_LIGHT = "#1e2a47"
RING = "#2c3a57"
TEAL = "#2dd4bf"
CORAL = "#ff6b6b"
MINT = "#a3ff6b"


def _catmull_rom_path(points, tension=0.0):
    """Closed, smooth Bezier path through ``points`` (Catmull-Rom converted to cubic Beziers)."""
    n = len(points)
    verts = [points[0]]
    codes = [MplPath.MOVETO]
    k = (1 - tension) / 6.0
    for i in range(n):
        p0, p1, p2, p3 = (points[(i - 1) % n], points[i], points[(i + 1) % n], points[(i + 2) % n])
        c1 = (p1[0] + k * (p2[0] - p0[0]), p1[1] + k * (p2[1] - p0[1]))
        c2 = (p2[0] - k * (p3[0] - p1[0]), p2[1] - k * (p3[1] - p1[1]))
        verts += [c1, c2, p2]
        codes += [MplPath.CURVE4] * 3
    verts.append(points[0])
    codes.append(MplPath.CLOSEPOLY)
    return MplPath(verts, codes)


ORGAN = [(58, 71), (69, 66), (75, 55), (72, 43), (63, 34), (52, 33), (43, 40), (44, 50), (41, 60), (47, 69)]


def logo_mark(ax, tile=True):
    """The square mark in a 100 x 100 coordinate system."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    if tile:
        bg = FancyBboxPatch(
            (0, 0), 100, 100, boxstyle="round,pad=0,rounding_size=22", fc=NAVY, ec="none", zorder=0
        )
        ax.add_patch(bg)
        yy, xx = np.mgrid[0:100:200j, 0:100:200j]
        glow = np.exp(-(((xx - 55) ** 2 + (yy - 55) ** 2) / (2 * 34**2)))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("g", [NAVY, NAVY_LIGHT])
        im = ax.imshow(
            glow,
            extent=(0, 100, 0, 100),
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=1,
            zorder=1,
            interpolation="bilinear",
        )
        im.set_clip_path(bg)
    ax.add_patch(patches.Circle((50, 51), 31, fill=False, ec=RING, lw=5.5, zorder=2))
    ax.add_patch(
        patches.Arc((50, 51), 62, 62, theta1=215, theta2=350, ec=TEAL, lw=5.5, capstyle="round", zorder=3)
    )
    path = _catmull_rom_path(ORGAN)
    ax.add_patch(patches.PathPatch(path, fc=CORAL, ec="none", zorder=4))
    ax.add_patch(patches.PathPatch(path, fc="none", ec="white", lw=2.6, joinstyle="round", zorder=5))
    for x, y in ORGAN[::2]:
        ax.add_patch(patches.Circle((x, y), 2.1, fc="white", ec=NAVY, lw=0.9, zorder=6))
    hub = (23.5, 51)
    for node in ((10, 36), (10, 66)):
        ax.plot([node[0], hub[0]], [node[1], hub[1]], color=TEAL, lw=3.2, solid_capstyle="round", zorder=3)
        ax.add_patch(patches.Circle(node, 3.3, fc=TEAL, ec=NAVY, lw=1.2, zorder=7))
    ax.add_patch(patches.Circle(hub, 4.0, fc="white", ec=TEAL, lw=2.4, zorder=8))
    ax.plot([57], [52], marker="+", ms=11, mew=2.8, color=MINT, zorder=9)


def save_logo():
    fig, ax = plt.subplots(figsize=(4, 4), dpi=256)
    fig.patch.set_alpha(0)
    fig.subplots_adjust(0, 0, 1, 1)
    logo_mark(ax)
    fig.savefig(HERE / "logo.png", transparent=True)
    fig.savefig(HERE / "logo.svg", transparent=True)
    plt.close(fig)
    fig = plt.figure(figsize=(12.0, 3.6), dpi=200)
    fig.patch.set_alpha(0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 36)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.5, 0.5),
            119,
            35,
            boxstyle="round,pad=0,rounding_size=6",
            fc=NAVY,
            ec="#26324d",
            lw=1.2,
            zorder=0,
        )
    )
    mark_ax = fig.add_axes((0.035, 0.12, 0.23, 0.76))
    logo_mark(mark_ax, tile=False)
    ax.text(
        34,
        22.6,
        "open-med-mcp",
        fontsize=44,
        fontweight="bold",
        color="white",
        va="center",
        ha="left",
        family="DejaVu Sans",
    )
    ax.text(
        34.3,
        12.8,
        "medical image analysis tools for AI agents",
        fontsize=17,
        color="#a9b4c8",
        va="center",
        ha="left",
        family="DejaVu Sans",
    )
    ax.text(
        34.3,
        6.8,
        "MedSAM2 · VoxTell · TotalSegmentator · nnU-Net · MONAI · HD-BET · SynthStrip · TorchXRayVision",
        fontsize=10.5,
        color="#6d7a94",
        va="center",
        ha="left",
        family="DejaVu Sans",
    )
    fig.savefig(HERE / "banner.png", transparent=True)
    plt.close(fig)


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    lines=(),
    fc="#ffffff",
    ec="#cbd5e1",
    title_color=INK,
    fs=10.5,
    line_fs=8.6,
    line_gap=0.36,
):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc=fc, ec=ec, lw=1.4, zorder=2
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.3,
        title,
        ha="center",
        va="top",
        fontsize=fs,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    for i, line in enumerate(lines):
        ax.text(
            x + w / 2,
            y + h - 0.7 - i * line_gap,
            line,
            ha="center",
            va="top",
            fontsize=line_fs,
            color="#334155",
            zorder=3,
            family="DejaVu Sans",
        )


def small_box(ax, x, y, w, h, name, sub, fc="#f8fafc", ec="#cbd5e1", name_fs=9.0, sub_fs=7.4):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=1.2, zorder=3
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.66,
        name,
        ha="center",
        va="center",
        fontsize=name_fs,
        fontweight="bold",
        color=INK,
        zorder=5,
    )
    ax.text(
        x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=sub_fs, color="#64748b", zorder=5
    )


def arrow(ax, x0, y0, x1, y1, text="", color="#64748b", ls="-", text_dy=0.14, text_fs=8):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6, "ls": ls, "shrinkA": 2, "shrinkB": 2},
        zorder=4,
    )
    if text:
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2 + text_dy,
            text,
            ha="center",
            va="bottom",
            fontsize=text_fs,
            color=color,
            zorder=5,
        )


def save_architecture() -> None:
    W, H = 14.2, 8.3
    fig, ax = plt.subplots(figsize=(W, H), dpi=120)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    # --- left column: clients + notes
    box(
        ax,
        0.3,
        6.2,
        2.7,
        1.75,
        "AI agent clients",
        ("Claude Code · Codex CLI", "Claude Desktop · Cursor", "any MCP client"),
        fc="#e0f2fe",
        ec="#7dd3fc",
    )
    ax.text(
        0.35,
        5.55,
        "Everything runs locally:\nimages never leave the machine.",
        fontsize=8.6,
        color="#475569",
        va="top",
        linespacing=1.5,
    )
    ax.text(0.35, 4.55, "Three pillars", fontsize=9.2, color=INK, va="top", fontweight="bold")
    ax.text(
        0.35,
        4.25,
        "1. containerized specialised models\n2. preset / custom guidelines\n3. code-customizable viewer",
        fontsize=8.6,
        color="#475569",
        va="top",
        linespacing=1.55,
    )
    # --- server
    box(
        ax,
        3.6,
        5.05,
        6.4,
        2.9,
        "open-med-mcp server   (MCP over stdio or HTTP)",
        (),
        fc="#ffffff",
        ec="#94a3b8",
        fs=11.5,
    )
    box(
        ax,
        3.8,
        5.25,
        1.95,
        2.15,
        "tools (31)",
        (
            "inspect · DICOM · convert",
            "segment · run_model · batch",
            "masks · metrics · features",
            "resample · register · N4",
            "render_view · report",
        ),
        fc="#f1f5f9",
        ec="#cbd5e1",
        fs=9.5,
        line_fs=8.0,
        line_gap=0.31,
    )
    box(
        ax,
        5.9,
        5.25,
        1.95,
        2.15,
        "guidelines (12)",
        ("preset protocols", "workspace overrides", "MCP prompts", "MCP resources"),
        fc="#fef3c7",
        ec="#fcd34d",
        fs=9.5,
        line_fs=8.0,
        line_gap=0.31,
    )
    box(
        ax,
        8.0,
        5.25,
        1.8,
        2.15,
        "viewer",
        ("PNG for agents", "HTML for humans", "reports", "renderer plugins", "NiiVue 3D"),
        fc="#ede9fe",
        ec="#c4b5fd",
        fs=9.5,
        line_fs=8.0,
        line_gap=0.31,
    )
    # --- core
    box(
        ax,
        3.6,
        2.95,
        6.4,
        1.75,
        "core",
        (
            "MedicalImage: NIfTI · DICOM · NRRD · MetaImage · PNG / JPEG",
            "native voxel coordinates · windowing · masks · metrics · prompts",
            "resampling · registration · features · meshes",
        ),
        fc="#ffffff",
        ec="#94a3b8",
        fs=10.5,
        line_fs=8.4,
        line_gap=0.34,
    )
    # --- zoo + runners
    box(ax, 3.6, 0.35, 6.4, 2.3, "model zoo + job contract", (), fc="#ffffff", ec="#94a3b8", fs=10.5)
    ax.text(
        6.8,
        1.98,
        "request.json  →  inputs/  →  outputs/  →  response.json   (identical for every backend)",
        ha="center",
        va="top",
        fontsize=8.0,
        color="#64748b",
        zorder=3,
    )
    for x, w, name, sub, fc, ec in (
        (3.8, 1.42, "local", "pip extras / venv", "#dcfce7", "#86efac"),
        (5.32, 1.42, "docker", "GHCR · --gpus all", "#dcfce7", "#86efac"),
        (6.84, 1.42, "apptainer", "HPC · .def / .sif", "#dcfce7", "#86efac"),
        (8.36, 1.45, "wrapped image", "official images", "#fee2e2", "#fca5a5"),
    ):
        small_box(ax, x, 0.5, w, 1.1, name, sub, fc=fc, ec=ec, name_fs=9.6, sub_fs=7.4)
    # --- right column: workspace + adapters
    box(
        ax,
        10.6,
        5.05,
        3.3,
        2.9,
        "workspace (local disk)",
        (
            "images · masks",
            "omm_outputs/<run>/",
            "request.json · outputs/",
            "response.json · log.txt",
            "provenance.jsonl",
            "reports · viewers",
        ),
        fc="#fff7ed",
        ec="#fdba74",
        fs=10.5,
        line_fs=8.2,
        line_gap=0.33,
    )
    box(
        ax,
        10.6,
        0.35,
        3.3,
        4.25,
        "adapters",
        (),
        fc="#ffffff",
        ec="#94a3b8",
        fs=10.5,
    )
    adapters = (
        ("medsam2 / sam2", "point + box prompts"),
        ("voxtell", "free-text prompts"),
        ("totalsegmentator", "117 CT structures"),
        ("lungmask", "lungs · lobes · LAA%"),
        ("hdbet", "MRI brain extraction"),
        ("synthstrip", "FreeSurfer image"),
        ("nnunet", "any nnU-Net model"),
        ("monai", "model-zoo bundles"),
        ("torchxrayvision", "chest X-ray findings"),
        ("classical", "threshold · region grow"),
    )
    col_w, row_h, gap = 1.52, 0.6, 0.07
    for i, (name, sub) in enumerate(adapters):
        col, row = i % 2, i // 2
        x = 10.73 + col * (col_w + 0.1)
        y = 4.6 - 0.5 - row_h - row * (row_h + gap)
        small_box(
            ax,
            x,
            y,
            col_w,
            row_h,
            name,
            sub,
            fc="#f8fafc",
            ec="#cbd5e1",
            name_fs=8.0,
            sub_fs=6.8,
        )
    ax.text(
        12.25,
        0.52,
        "+ your own model: copy zoo/_template",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#b91c1c",
        zorder=5,
    )
    # --- arrows
    arrow(ax, 3.0, 7.0, 3.6, 7.0, "MCP", text_dy=0.1)
    arrow(ax, 6.8, 5.05, 6.8, 4.7)
    arrow(ax, 6.8, 2.95, 6.8, 2.65)
    arrow(ax, 10.0, 6.5, 10.6, 6.5, "files", text_dy=0.1)
    arrow(ax, 10.0, 1.75, 10.6, 1.75, "runs", text_dy=0.1)
    arrow(ax, 10.6, 0.95, 10.0, 0.95, "masks", ls="--", text_dy=0.1)
    fig.savefig(HERE / "architecture.png", bbox_inches="tight", pad_inches=0.15, facecolor=PAPER)
    fig.savefig(HERE / "architecture.svg", bbox_inches="tight", pad_inches=0.15, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    save_logo()
    save_architecture()
    print("assets written to", HERE)

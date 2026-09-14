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

HERE = Path(__file__).resolve().parent
INK = "#0f172a"
TEAL = "#0ea5a4"
CORAL = "#ff6b6b"
SKY = "#38bdf8"
PAPER = "#f8fafc"


def logo(ax: plt.Axes, with_text: bool = False) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.25, 1.25 if not with_text else 4.9)
    ax.set_ylim(-1.25, 1.25)
    # rounded tile
    tile = FancyBboxPatch(
        (-1.1, -1.1), 2.2, 2.2, boxstyle="round,pad=0,rounding_size=0.42", fc=INK, ec="none"
    )
    ax.add_patch(tile)
    # "slice": a soft ellipse like an axial section
    t = np.linspace(0, 2 * np.pi, 200)
    ax.fill(0.78 * np.cos(t), 0.62 * np.sin(t) - 0.05, color="#1e293b", zorder=2)
    ax.plot(0.78 * np.cos(t), 0.62 * np.sin(t) - 0.05, color="#334155", lw=2, zorder=3)
    # segmentation contour (organ-like blob) with a filled semi-transparent region
    tt = np.linspace(0, 2 * np.pi, 300)
    r = 0.36 + 0.06 * np.sin(3 * tt) + 0.03 * np.cos(5 * tt)
    bx, by = 0.16 + r * np.cos(tt) * 1.1, -0.02 + r * np.sin(tt)
    ax.fill(bx, by, color=CORAL, alpha=0.45, zorder=4)
    ax.plot(bx, by, color=CORAL, lw=3.2, zorder=5, solid_capstyle="round")
    # prompt box (dashed) and a point prompt
    ax.add_patch(
        patches.Rectangle(
            (-0.35, -0.5), 1.0, 0.95, fill=False, ec=SKY, lw=2.4, ls=(0, (4, 3)), zorder=6, joinstyle="round"
        )
    )
    ax.plot([0.16], [-0.02], marker="+", ms=13, mew=3.2, color="#a3ff6b", zorder=7)
    # MCP-style connection nodes on the left edge
    for (x, y), c in (((-0.86, 0.55), TEAL), ((-0.86, 0.0), TEAL), ((-0.86, -0.55), TEAL)):
        ax.plot([x, -0.5], [y, 0.0], color=TEAL, lw=2.2, alpha=0.9, zorder=3)
        ax.plot([x], [y], marker="o", ms=9, color=c, mec=INK, mew=1.5, zorder=8)
    ax.plot([-0.5], [0.0], marker="o", ms=10, color=PAPER, mec=INK, mew=1.5, zorder=8)
    if with_text:
        ax.text(
            1.45,
            0.33,
            "open-med-mcp",
            fontsize=44,
            fontweight="bold",
            color=INK,
            va="center",
            ha="left",
            family="DejaVu Sans",
        )
        ax.text(
            1.47,
            -0.33,
            "medical image analysis tools for AI agents",
            fontsize=17,
            color="#475569",
            va="center",
            ha="left",
            family="DejaVu Sans",
        )


def save_logo() -> None:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=128)
    fig.patch.set_alpha(0)
    logo(ax)
    fig.savefig(HERE / "logo.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / "logo.svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12.3, 5), dpi=128)
    fig.patch.set_alpha(0)
    logo(ax, with_text=True)
    fig.savefig(HERE / "banner.png", transparent=True, bbox_inches="tight", pad_inches=0.05)
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
        "1. containerized specialised models\n2. preset / custom guidelines\n3. code-customisable viewer",
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

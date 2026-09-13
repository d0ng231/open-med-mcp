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


def box(ax, x, y, w, h, title, lines=(), fc="#ffffff", ec="#cbd5e1", title_color=INK, fs=10.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc=fc, ec=ec, lw=1.4, zorder=2
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.32,
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
            y + h - 0.72 - i * 0.36,
            line,
            ha="center",
            va="top",
            fontsize=8.6,
            color="#334155",
            zorder=3,
            family="DejaVu Sans",
        )


def arrow(ax, x0, y0, x1, y1, text="", color="#64748b", ls="-"):
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
            (y0 + y1) / 2 + 0.16,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            zorder=5,
            bbox={"fc": PAPER, "ec": "none", "pad": 1.5},
        )


def save_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.4), dpi=120)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    # agents
    box(
        ax,
        0.3,
        5.3,
        2.6,
        1.7,
        "AI agent clients",
        ("Claude Code · Codex CLI", "Claude Desktop · Cursor", "any MCP client"),
        fc="#e0f2fe",
        ec="#7dd3fc",
    )
    # server
    box(
        ax,
        3.6,
        4.15,
        6.0,
        2.85,
        "open-med-mcp server  (MCP, stdio / HTTP)",
        (),
        fc="#ffffff",
        ec="#94a3b8",
        fs=11.5,
    )
    box(
        ax,
        3.8,
        4.35,
        1.8,
        2.1,
        "tools",
        ("inspect_image", "segment / run_model", "mask_stats · compare", "render_view · report"),
        fc="#f1f5f9",
        ec="#cbd5e1",
        fs=9.5,
    )
    box(
        ax,
        5.75,
        4.35,
        1.8,
        2.1,
        "guidelines",
        ("preset protocols", "user overrides", "MCP prompts +", "resources"),
        fc="#fef3c7",
        ec="#fcd34d",
        fs=9.5,
    )
    box(
        ax,
        7.7,
        4.35,
        1.75,
        2.1,
        "viewer",
        ("PNG for agents", "HTML for humans", "renderer plugins", "NiiVue 3D"),
        fc="#ede9fe",
        ec="#c4b5fd",
        fs=9.5,
    )
    # core
    box(
        ax,
        3.6,
        2.55,
        6.0,
        1.25,
        "core",
        (
            "MedicalImage: NIfTI · DICOM · NRRD · MetaImage · PNG/JPEG",
            "native voxel coordinates · windowing · masks · metrics · prompts",
        ),
        fc="#ffffff",
        ec="#94a3b8",
        fs=10.5,
    )
    # zoo / runners
    box(ax, 3.6, 0.35, 6.0, 1.85, "model zoo + job contract", (), fc="#ffffff", ec="#94a3b8", fs=10.5)
    ax.text(
        6.6,
        1.68,
        "request.json  →  outputs/  →  response.json",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#64748b",
        zorder=3,
    )
    box(ax, 3.8, 0.5, 1.35, 1.0, "local", ("pip extras · venv",), fc="#dcfce7", ec="#86efac", fs=9.5)
    box(ax, 5.3, 0.5, 1.35, 1.0, "docker", ("GHCR · --gpus",), fc="#dcfce7", ec="#86efac", fs=9.5)
    box(ax, 6.8, 0.5, 1.35, 1.0, "apptainer", ("HPC · .def/.sif",), fc="#dcfce7", ec="#86efac", fs=9.5)
    box(ax, 8.3, 0.5, 1.1, 1.0, "yours", ("manifest+run.py",), fc="#fee2e2", ec="#fca5a5", fs=9.0)
    # adapters column
    box(ax, 10.3, 0.35, 2.6, 3.45, "adapters (containerized)", (), fc="#ffffff", ec="#94a3b8", fs=10.5)
    for i, (name, sub) in enumerate(
        (
            ("medsam2 / sam2", "promptable, 2D + 3D"),
            ("totalsegmentator", "117 CT structures"),
            ("classical", "threshold · region grow"),
            ("_template", "add your own"),
        )
    ):
        y0 = 2.6 - i * 0.72
        ax.add_patch(
            FancyBboxPatch(
                (10.5, y0),
                2.2,
                0.66,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc="#f8fafc",
                ec="#cbd5e1",
                lw=1.2,
                zorder=3,
            )
        )
        ax.text(
            11.6,
            y0 + 0.45,
            name,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=INK,
            zorder=5,
        )
        ax.text(11.6, y0 + 0.19, sub, ha="center", va="center", fontsize=7.6, color="#64748b", zorder=5)
    # data
    box(
        ax,
        10.3,
        4.15,
        2.6,
        2.85,
        "workspace (local disk)",
        (
            "images · masks",
            "omm_outputs/<run>/",
            "  request.json · outputs/",
            "  response.json · log.txt",
            "provenance.jsonl",
            "reports · viewers",
        ),
        fc="#fff7ed",
        ec="#fdba74",
    )
    # arrows
    arrow(ax, 2.9, 6.1, 3.6, 6.1, "MCP")
    arrow(ax, 6.6, 4.15, 6.6, 3.8)
    arrow(ax, 6.6, 2.55, 6.6, 2.2)
    arrow(ax, 9.6, 1.3, 10.3, 1.3, "runs")
    arrow(ax, 9.6, 5.6, 10.3, 5.6, "reads / writes")
    arrow(ax, 10.3, 3.0, 9.65, 3.0, "masks", ls="--")
    ax.text(
        0.3,
        4.75,
        "Everything runs locally:\nimages never leave\nthe machine.",
        fontsize=8.6,
        color="#475569",
        va="top",
    )
    ax.text(
        0.3,
        3.35,
        "Three pillars\n1. containerized models\n2. preset / custom guidelines\n3. code-customisable viewer",
        fontsize=8.6,
        color="#475569",
        va="top",
    )
    fig.savefig(HERE / "architecture.png", bbox_inches="tight", pad_inches=0.15, facecolor=PAPER)
    fig.savefig(HERE / "architecture.svg", bbox_inches="tight", pad_inches=0.15, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    save_logo()
    save_architecture()
    print("assets written to", HERE)

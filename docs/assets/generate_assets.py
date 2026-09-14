"""Generate the logo and banner (SVG + PNG) with matplotlib.

Run: python docs/assets/generate_assets.py

The architecture diagram (docs/assets/architecture.png) is NOT generated here - it is designed in
Figma and exported at 2x. Source file:
https://www.figma.com/design/xuM8tncKOhA5XeRBGS7nlY/open-med-mcp---Architecture
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


if __name__ == "__main__":
    save_logo()
    print("logo + banner written to", HERE)

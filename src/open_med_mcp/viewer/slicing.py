"""Turn a native 3D array into correctly oriented 2D display slices (and back for prompt drawing)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from open_med_mcp.core.image import MedicalImage, Plane

# screen x should increase towards +L (axial, coronal) or +P (sagittal); screen y (downwards)
# towards +P (axial) or -S i.e. inferior (coronal, sagittal). Physical axes in LPS: 0=L 1=P 2=S.
_SCREEN_AXES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "axial": ((0, +1), (1, +1)),
    "coronal": ((0, +1), (2, -1)),
    "sagittal": ((1, +1), (2, -1)),
}


@dataclass
class DisplaySlice:
    """A 2D array ready for ``imshow`` plus the mapping back to native voxel indices."""

    array: np.ndarray
    plane: str
    index: int
    n_slices: int
    stack_axis_xyz: int  # native index axis (0=x,1=y,2=z) that we sliced along
    col_axis_xyz: int  # native axis shown along screen x
    row_axis_xyz: int  # native axis shown along screen y
    flip_cols: bool
    flip_rows: bool
    aspect: float  # row spacing / col spacing
    spacing_col: float
    spacing_row: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape[0], self.array.shape[1]

    def to_display(self, xyz: tuple[float, float, float] | list[float]) -> tuple[float, float]:
        """Native voxel coords -> (display col, display row)."""
        c = float(xyz[self.col_axis_xyz])
        r = float(xyz[self.row_axis_xyz])
        if self.flip_cols:
            c = self.shape[1] - 1 - c
        if self.flip_rows:
            r = self.shape[0] - 1 - r
        return c, r

    def to_native(self, col: float, row: float) -> list[float]:
        """(display col, display row) -> native voxel coords (3 values; stack axis = slice index)."""
        c, r = float(col), float(row)
        if self.flip_cols:
            c = self.shape[1] - 1 - c
        if self.flip_rows:
            r = self.shape[0] - 1 - r
        out = [0.0, 0.0, 0.0]
        out[self.col_axis_xyz] = c
        out[self.row_axis_xyz] = r
        out[self.stack_axis_xyz] = float(self.index)
        return out

    def native_ticks(self, axis: str, n: int = 6) -> tuple[list[float], list[str]]:
        """Tick positions (display) and labels (native index) for ``axis`` in {"x","y"} (screen)."""
        length = self.shape[1] if axis == "x" else self.shape[0]
        flip = self.flip_cols if axis == "x" else self.flip_rows
        step = _nice_step(length, n)
        natives = list(range(0, length, step))
        pos = [(length - 1 - v) if flip else v for v in natives]
        return [float(p) for p in pos], [str(v) for v in natives]

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        """Orient a full-size mask array the same way as this display slice."""
        return _orient(
            np.take(mask, self.index, axis=2 - self.stack_axis_xyz) if mask.ndim == 3 else mask, self
        )


def _nice_step(length: int, n: int) -> int:
    raw = max(length / max(n, 1), 1)
    for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if s >= raw:
            return s
    return int(raw)


def _orient(arr2d: np.ndarray, ds: DisplaySlice) -> np.ndarray:
    return _orient_raw(arr2d, ds.flip_cols, ds.flip_rows, ds._transpose)  # type: ignore[attr-defined]


def _orient_raw(arr2d: np.ndarray, flip_cols: bool, flip_rows: bool, transpose: bool) -> np.ndarray:
    a = arr2d
    if transpose:
        a = np.swapaxes(a, 0, 1)
    if flip_rows:
        a = a[::-1]
    if flip_cols:
        a = a[:, ::-1]
    return np.ascontiguousarray(a)


def display_slice(
    img: MedicalImage, plane: Plane, index: int, array: np.ndarray | None = None
) -> DisplaySlice:
    """Extract slice ``index`` of ``plane`` from ``img`` (or from ``array`` with the same geometry)."""
    data = img.array if array is None else array
    if img.is_2d:
        ds = DisplaySlice(
            np.ascontiguousarray(data),
            "axial",
            0,
            1,
            2,
            0,
            1,
            False,
            False,
            float(img.spacing[1] / img.spacing[0]) if len(img.spacing) > 1 else 1.0,
            float(img.spacing[0]),
            float(img.spacing[1]) if len(img.spacing) > 1 else 1.0,
        )
        ds._transpose = False  # type: ignore[attr-defined]
        return ds
    stack_xyz = img.index_axis_for_plane(plane)
    np_axis = 2 - stack_xyz
    n = data.shape[np_axis]
    index = int(np.clip(index, 0, n - 1))
    sl = np.take(data, index, axis=np_axis)  # remaining numpy axes in order -> native axes (descending)
    remaining_np = [ax for ax in (0, 1, 2) if ax != np_axis]
    remaining_xyz = [2 - ax for ax in remaining_np]  # e.g. axial: [1, 0] -> rows=y, cols=x
    d = img.direction_matrix
    (col_phys, col_sign), (row_phys, row_sign) = _SCREEN_AXES[plane]
    # which native axis is dominated by the screen-x physical axis?
    dom = {ax: int(np.argmax(np.abs(d[:, ax]))) for ax in remaining_xyz}
    col_xyz = next((ax for ax in remaining_xyz if dom[ax] == col_phys), remaining_xyz[1])
    row_xyz = next((ax for ax in remaining_xyz if ax != col_xyz), remaining_xyz[0])
    transpose = remaining_xyz[1] != col_xyz  # rows/cols swapped relative to the raw slice
    flip_cols = np.sign(d[col_phys, col_xyz]) != col_sign
    flip_rows = np.sign(d[row_phys, row_xyz]) != row_sign
    arr = _orient_raw(sl, bool(flip_cols), bool(flip_rows), transpose)
    sp_col, sp_row = float(img.spacing[col_xyz]), float(img.spacing[row_xyz])
    ds = DisplaySlice(
        arr,
        plane,
        index,
        n,
        stack_xyz,
        col_xyz,
        row_xyz,
        bool(flip_cols),
        bool(flip_rows),
        sp_row / sp_col,
        sp_col,
        sp_row,
    )
    ds._transpose = transpose  # type: ignore[attr-defined]
    return ds


def auto_slices(
    img: MedicalImage, plane: Plane, masks: list[np.ndarray], n: int = 1, margin: int = 1
) -> list[int]:
    """Choose slice indices: the ones with the largest mask area (``n==1``) or evenly spaced across
    the mask extent; without masks, the centre (``n==1``) or evenly spaced through the volume."""
    if img.is_2d:
        return [0]
    np_axis = img.numpy_axis_for_plane(plane)
    total = img.shape_zyx[np_axis]
    area = np.zeros(total)
    for m in masks:
        other = tuple(i for i in range(3) if i != np_axis)
        area += (m != 0).sum(axis=other)
    if area.sum() > 0:
        nz = np.nonzero(area)[0]
        if n == 1:
            return [int(np.argmax(area))]
        lo, hi = max(int(nz.min()) - margin, 0), min(int(nz.max()) + margin, total - 1)
    else:
        if n == 1:
            return [total // 2]
        lo, hi = 0, total - 1
    return sorted({int(round(v)) for v in np.linspace(lo, hi, num=min(n, hi - lo + 1))})


def mask_centroid_xyz(masks: list[np.ndarray]) -> list[float] | None:
    from scipy import ndimage

    for m in masks:
        b = m != 0
        if b.any():
            c = ndimage.center_of_mass(b)
            return [float(v) for v in reversed(c)]
    return None

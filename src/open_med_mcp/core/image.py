"""Loading, saving and describing medical images.

Everything downstream works on :class:`MedicalImage`, a thin wrapper around a NumPy array plus
ITK-style geometry (spacing / origin / direction in **LPS** physical space).

Coordinate conventions (see ``docs/coordinates.md``)
----------------------------------------------------
* Voxel coordinates exposed to agents are ``(x, y, z)`` **native index coordinates** in ITK order,
  i.e. exactly what SimpleITK uses. The NumPy array is stored as ``array[z, y, x]`` for 3D images
  and ``array[y, x]`` (or ``array[y, x, channels]``) for 2D images.
* We never re-orient the data. Anatomical planes (axial/coronal/sagittal) are mapped onto native
  array axes using the direction cosines, and only *display* flips are applied by the viewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import SimpleITK as sitk
from PIL import Image as PILImage

Plane = Literal["axial", "coronal", "sagittal"]
PLANES: tuple[Plane, ...] = ("axial", "coronal", "sagittal")

VOLUME_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd", ".img", ".hdr", ".gipl", ".vtk")
IMAGE2D_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DICOM_SUFFIXES = (".dcm", ".dicom", ".ima")

#: physical axis index (LPS) that each anatomical plane slices *through*
_PLANE_TO_PHYSICAL_AXIS: dict[str, int] = {"sagittal": 0, "coronal": 1, "axial": 2}


def file_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def detect_format(path: Path) -> str:
    """Return a short format label: nifti, nrrd, metaimage, dicom-series, dicom, png, jpeg ..."""
    if path.is_dir():
        return "dicom-series"
    sfx = file_suffix(path)
    if sfx in (".nii", ".nii.gz"):
        return "nifti"
    if sfx in (".nrrd", ".nhdr"):
        return "nrrd"
    if sfx in (".mha", ".mhd"):
        return "metaimage"
    if sfx in DICOM_SUFFIXES:
        return "dicom"
    if sfx in IMAGE2D_SUFFIXES:
        return sfx.lstrip(".").replace("jpg", "jpeg").replace("tif", "tiff")
    if sfx == ".npy":
        return "numpy"
    # DICOM files frequently have no suffix
    if path.is_file() and _looks_like_dicom(path):
        return "dicom"
    return sfx.lstrip(".") or "unknown"


def _looks_like_dicom(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


@dataclass
class MedicalImage:
    """A 2D or 3D image with ITK geometry. ``array`` is ``[z, y, x]`` (3D) or ``[y, x(, c)]`` (2D)."""

    array: np.ndarray
    spacing: tuple[float, ...]  # ITK order (x, y[, z])
    origin: tuple[float, ...]
    direction: tuple[float, ...]  # flattened row-major 2x2 or 3x3
    path: Path | None = None
    format: str = "array"
    metadata: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------- basics
    @property
    def is_2d(self) -> bool:
        return self.array.ndim == 2 or (self.array.ndim == 3 and self.is_rgb)

    @property
    def is_rgb(self) -> bool:
        return (
            self.array.ndim == 3
            and self.array.shape[-1] in (3, 4)
            and self.metadata.get("channels_last", False)
        )

    @property
    def ndim(self) -> int:
        return 2 if self.is_2d else 3

    @property
    def shape_zyx(self) -> tuple[int, ...]:
        """Array shape without channels."""
        return tuple(self.array.shape[:-1]) if self.is_rgb else tuple(self.array.shape)

    @property
    def size_xyz(self) -> tuple[int, ...]:
        """Size in ITK order ``(x, y[, z])``."""
        return tuple(reversed(self.shape_zyx))

    @property
    def scalar_array(self) -> np.ndarray:
        """Grayscale array (RGB is converted with luma weights)."""
        if self.is_rgb:
            rgb = self.array[..., :3].astype(np.float32)
            return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return self.array

    @property
    def direction_matrix(self) -> np.ndarray:
        n = 2 if len(self.direction) == 4 else 3
        return np.asarray(self.direction, dtype=float).reshape(n, n)

    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    # ---------------------------------------------------------- orientation
    def orientation_code(self) -> str | None:
        """Three-letter code (e.g. ``LPS``, ``RAS``) describing the direction of the index axes."""
        if self.is_2d or len(self.direction) != 9:
            return None
        try:
            return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(list(self.direction))
        except Exception:  # pragma: no cover - very old SimpleITK
            return None

    def index_axis_for_plane(self, plane: Plane) -> int:
        """Native **ITK index axis** (0=x, 1=y, 2=z) that an anatomical plane is stacked along."""
        if self.is_2d:
            return 2  # a 2D image is a single axial slice by convention
        phys = _PLANE_TO_PHYSICAL_AXIS[plane]
        d = np.abs(self.direction_matrix)
        # column j = physical direction of index axis j; pick the index axis dominated by `phys`
        dominant = [int(np.argmax(d[:, j])) for j in range(3)]
        if phys in dominant:
            return dominant.index(phys)
        # oblique volumes: fall back to conventional axis order
        return phys

    def numpy_axis_for_plane(self, plane: Plane) -> int:
        """Array axis (in ``[z, y, x]`` order) that ``plane`` slices along."""
        if self.is_2d:
            return 0
        return 2 - self.index_axis_for_plane(plane)

    def n_slices(self, plane: Plane) -> int:
        if self.is_2d:
            return 1
        return int(self.shape_zyx[self.numpy_axis_for_plane(plane)])

    def index_to_physical(self, xyz: tuple[float, ...]) -> tuple[float, ...]:
        m = self.direction_matrix
        idx = np.asarray(xyz, dtype=float)[: m.shape[0]]
        phys = np.asarray(self.origin)[: m.shape[0]] + m @ (idx * np.asarray(self.spacing)[: m.shape[0]])
        return tuple(float(v) for v in phys)

    # ------------------------------------------------------------ statistics
    def intensity_stats(self, sample: int = 2_000_000) -> dict[str, float]:
        arr = self.scalar_array
        flat = arr.reshape(-1)
        if flat.size > sample:
            rng = np.random.default_rng(0)
            flat = flat[rng.choice(flat.size, sample, replace=False)]
        flat = flat.astype(np.float64)
        pct = np.percentile(flat, [0.5, 1, 5, 50, 95, 99, 99.5])
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "p0_5": float(pct[0]),
            "p1": float(pct[1]),
            "p5": float(pct[2]),
            "median": float(pct[3]),
            "p95": float(pct[4]),
            "p99": float(pct[5]),
            "p99_5": float(pct[6]),
        }

    def modality_guess(self) -> str:
        """Heuristic modality label used to pick default windows and models."""
        hint = str(self.metadata.get("modality") or "").upper()
        if hint in {"CT", "MR", "PT", "US", "CR", "DX", "MG", "OT", "XA", "RF", "NM"}:
            return {"PT": "PET", "CR": "XR", "DX": "XR", "MG": "MG"}.get(hint, hint)
        if self.is_rgb:
            return "RGB"
        arr = self.scalar_array
        lo, hi = float(arr.min()), float(arr.max())
        if lo <= -500 and hi >= 300 and self.ndim == 3:
            return "CT"
        if self.ndim == 2 and arr.dtype in (np.uint8, np.uint16):
            return "XR/2D"
        return "MR/unknown"

    # ----------------------------------------------------------------- I/O
    def to_sitk(self) -> sitk.Image:
        arr = self.array
        if self.is_rgb:
            img = sitk.GetImageFromArray(np.ascontiguousarray(arr[..., :3]), isVector=True)
        else:
            img = sitk.GetImageFromArray(np.ascontiguousarray(arr))
        img.SetSpacing(tuple(float(s) for s in self.spacing[: img.GetDimension()]))
        img.SetOrigin(tuple(float(o) for o in self.origin[: img.GetDimension()]))
        if len(self.direction) == img.GetDimension() ** 2:
            img.SetDirection(tuple(float(d) for d in self.direction))
        return img

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sfx = file_suffix(path)
        if sfx in IMAGE2D_SUFFIXES:
            arr = self.array
            if arr.dtype not in (np.uint8, np.uint16) and not self.is_rgb:
                arr = _to_uint8(arr)
            PILImage.fromarray(arr).save(path)
        else:
            sitk.WriteImage(self.to_sitk(), str(path), useCompression=True)
        return path

    def with_array(self, array: np.ndarray, **metadata: Any) -> MedicalImage:
        """Same geometry, different voxels (e.g. a mask that matches this image)."""
        meta = {**self.metadata, **metadata}
        meta.pop("channels_last", None)
        return MedicalImage(array, self.spacing, self.origin, self.direction, None, self.format, meta)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path) -> MedicalImage:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        fmt = detect_format(path)
        if fmt == "dicom-series":
            return cls._load_dicom_series(path)
        if fmt == "numpy":
            arr = np.load(path)
            n = arr.ndim
            return cls(arr, (1.0,) * n, (0.0,) * n, tuple(np.eye(n).ravel()), path, fmt)
        if file_suffix(path) in IMAGE2D_SUFFIXES:
            return cls._load_pil(path)
        img = sitk.ReadImage(str(path))
        return cls.from_sitk(img, path=path, fmt=fmt)

    @classmethod
    def from_sitk(cls, img: sitk.Image, path: Path | None = None, fmt: str = "sitk") -> MedicalImage:
        meta: dict[str, Any] = {}
        for key in img.GetMetaDataKeys():
            if key in ("0008|0060", "0008|103e", "0010|0020", "0020|000e", "0008|0016"):
                meta[
                    {
                        "0008|0060": "modality",
                        "0008|103e": "series_description",
                        "0010|0020": "patient_id",
                        "0020|000e": "series_uid",
                        "0008|0016": "sop_class",
                    }[key]
                ] = img.GetMetaData(key).strip()
        # collapse singleton z (single-slice DICOM / 2D NIfTI saved as 3D)
        if img.GetDimension() == 3 and img.GetSize()[2] == 1:
            meta["collapsed_singleton_z"] = True
            meta["spacing_3d"] = tuple(img.GetSpacing())
            img = img[:, :, 0]
        if img.GetDimension() == 4:
            meta["dropped_4th_dim"] = img.GetSize()[3]
            img = img[:, :, :, 0]
        if img.GetNumberOfComponentsPerPixel() > 1:
            arr = sitk.GetArrayFromImage(img)
            meta["channels_last"] = True
        else:
            arr = sitk.GetArrayFromImage(img)
        return cls(
            arr, tuple(img.GetSpacing()), tuple(img.GetOrigin()), tuple(img.GetDirection()), path, fmt, meta
        )

    @classmethod
    def _load_pil(cls, path: Path) -> MedicalImage:
        with PILImage.open(path) as im:
            mode = im.mode
            if mode in ("I;16", "I;16B", "I;16L", "I"):
                arr = np.asarray(im).astype(np.uint16 if "16" in mode else np.int32)
                meta: dict[str, Any] = {}
            elif mode in ("L", "1"):
                arr = np.asarray(im.convert("L"))
                meta = {}
            elif mode == "F":
                arr = np.asarray(im).astype(np.float32)
                meta = {}
            else:
                arr = np.asarray(im.convert("RGB"))
                meta = {"channels_last": True}
        return cls(arr, (1.0, 1.0), (0.0, 0.0), (1.0, 0.0, 0.0, 1.0), path, detect_format(path), meta)

    @classmethod
    def _load_dicom_series(cls, directory: Path) -> MedicalImage:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(directory))
        if not series_ids:
            raise ValueError(f"no DICOM series found in {directory}")
        best: list[str] = []
        best_id = ""
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(str(directory), sid)
            if len(files) > len(best):
                best, best_id = list(files), sid
        reader.SetFileNames(best)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOff()
        img = reader.Execute()
        out = cls.from_sitk(img, path=directory, fmt="dicom-series")
        out.metadata.update({"series_uid": best_id, "n_files": len(best), "n_series_in_dir": len(series_ids)})
        try:
            for tag, name in (("0008|0060", "modality"), ("0008|103e", "series_description")):
                if reader.HasMetaDataKey(0, tag):
                    out.metadata[name] = reader.GetMetaData(0, tag).strip()
        except Exception:
            pass
        return out

    # -------------------------------------------------------------- describe
    def describe(self) -> dict[str, Any]:
        stats = self.intensity_stats()
        info: dict[str, Any] = {
            "path": str(self.path) if self.path else None,
            "format": self.format,
            "ndim": self.ndim,
            "size_xyz": list(self.size_xyz),
            "array_shape": list(self.array.shape),
            "dtype": str(self.array.dtype),
            "spacing_xyz_mm": [round(float(s), 4) for s in self.spacing],
            "origin": [round(float(o), 3) for o in self.origin],
            "orientation": self.orientation_code(),
            "is_rgb": self.is_rgb,
            "modality_guess": self.modality_guess(),
            "intensity": {k: round(v, 3) for k, v in stats.items()},
            "metadata": {k: v for k, v in self.metadata.items() if k not in ("channels_last",)},
        }
        if self.ndim == 3:
            info["planes"] = {
                p: {"index_axis": self.index_axis_for_plane(p), "n_slices": self.n_slices(p)} for p in PLANES
            }
            info["voxel_volume_mm3"] = round(self.voxel_volume_mm3(), 5)
            info["fov_mm"] = [round(s * n, 1) for s, n in zip(self.spacing, self.size_xyz, strict=False)]
        return info


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def load_mask(path: str | Path, like: MedicalImage | None = None) -> MedicalImage:
    """Load a label map and (optionally) check that its geometry matches ``like``."""
    mask = MedicalImage.load(path)
    if mask.is_rgb:  # color PNG masks -> label ids by unique color is ambiguous; use luma > 0
        mask = MedicalImage(
            (mask.scalar_array > 0).astype(np.uint8),
            mask.spacing,
            mask.origin,
            mask.direction,
            mask.path,
            mask.format,
        )
    if not np.issubdtype(mask.array.dtype, np.integer):
        mask.array = np.rint(mask.array).astype(np.int32)
    if like is not None and mask.shape_zyx != like.shape_zyx:
        raise ValueError(f"mask shape {mask.shape_zyx} does not match image shape {like.shape_zyx}")
    return mask


def save_mask(mask: np.ndarray, like: MedicalImage, path: str | Path) -> Path:
    """Save ``mask`` (integer labels) with the geometry of ``like``.

    3D masks are written with the image geometry (NIfTI/NRRD/...), 2D masks as PNG label maps.
    """
    path = Path(path)
    mask = np.asarray(mask)
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)
    elif mask.max(initial=0) <= 255 and mask.min(initial=0) >= 0:
        mask = mask.astype(np.uint8)
    else:
        mask = mask.astype(np.int32)
    out = like.with_array(mask)
    if like.is_2d and file_suffix(path) in IMAGE2D_SUFFIXES:
        PILImage.fromarray(mask.astype(np.uint8 if mask.dtype == np.uint8 else np.int32)).save(path)
        return path
    return out.save(path)


def labels_sidecar_path(mask_path: Path) -> Path:
    name = mask_path.name
    for sfx in (".nii.gz", ".nii", ".nrrd", ".mha", ".png"):
        if name.endswith(sfx):
            return mask_path.with_name(name[: -len(sfx)] + ".labels.json")
    return mask_path.with_suffix(".labels.json")


def read_labels_sidecar(mask_path: Path) -> dict[int, str] | None:
    p = labels_sidecar_path(mask_path)
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in raw.items()}
    return None


def write_labels_sidecar(mask_path: Path, labels: dict[int, str]) -> Path:
    p = labels_sidecar_path(mask_path)
    p.write_text(json.dumps({str(k): v for k, v in sorted(labels.items())}, indent=2), encoding="utf-8")
    return p

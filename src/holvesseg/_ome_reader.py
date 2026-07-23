"""OME-TIFF reader with voxel spacing from embedded OME-XML metadata."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import tifffile as tf


LayerDataTuple = Tuple[Any, Dict[str, Any], str]


_UNIT_TO_MICROMETER = {
    "µm": 1.0,
    "\u03bcm": 1.0,
    "um": 1.0,
    "micrometer": 1.0,
    "micrometers": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "nm": 1e-3,
    "mm": 1000.0,
    "cm": 10000.0,
    "m": 1e6,
}


def _norm_unit(u: Optional[str]) -> str:
    if u is None:
        return "micrometer"
    u = u.strip()
    return _UNIT_TO_MICROMETER.get(u, u.lower())


def _to_micrometers(value: float, unit: Optional[str]) -> float:
    if unit is None:
        return float(value)
    key = _norm_unit(unit)
    factor = _UNIT_TO_MICROMETER.get(key)
    if factor is None:
        return float(value)
    return float(value) * factor


def _parse_pixels_element(xml_text: str) -> Optional[ET.Element]:
    if not xml_text:
        return None
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "Pixels":
            return el
    return None


def _ome_axis_scales_microns(omexml: str) -> Tuple[Dict[str, float], bool]:
    """Return ``(scales, explicit)`` with sizes in micrometres and whether OME set any.

    ``explicit`` is True if at least one ``PhysicalSize{X,Y,Z}`` attribute was
    present so we can attach ``units=µm`` without lying for purely default scales.
    """
    out = {"X": 1.0, "Y": 1.0, "Z": 1.0}
    pixels = _parse_pixels_element(omexml)
    if pixels is None:
        return out, False
    explicit = False
    for axis in ("X", "Y", "Z"):
        size_key = f"PhysicalSize{axis}"
        unit_key = f"PhysicalSize{axis}Unit"
        raw = pixels.attrib.get(size_key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        explicit = True
        unit = pixels.attrib.get(unit_key)
        out[axis] = _to_micrometers(val, unit)
    return out, explicit


def _index_first_along_axis(data: np.ndarray, axis: int) -> np.ndarray:
    """Select index ``0`` along *axis* using slicing (often a view on ``memmap``).

    ``numpy.take(..., 0, axis=...)`` can materialize a full copy for large arrays,
    which is unsafe for multi-gigabyte memory-mapped volumes.
    """
    if axis < 0:
        axis += data.ndim
    slc: List[slice | int] = [slice(None)] * data.ndim
    slc[axis] = 0
    return data[tuple(slc)]


def _squeeze_trailing_extras(data: np.ndarray, axes: str) -> Tuple[np.ndarray, str]:
    """Drop length-1 *T* / *S* / *I* / *H* / *A* dimensions (not *C*)."""
    d = data
    ax = axes
    changed = True
    while changed:
        changed = False
        for i, letter in enumerate(ax):
            if i >= d.ndim:
                break
            if d.shape[i] == 1 and letter.upper() in "TSIHA":
                d = _index_first_along_axis(d, i)
                ax = ax[:i] + ax[i + 1 :]
                changed = True
                break
    return d, ax


def _take_axis0(data: np.ndarray, axes: str, letter: str) -> Tuple[np.ndarray, str]:
    """If *letter* is present, index ``0`` along that axis (drop the axis)."""
    ax_u = axes.upper()
    if letter not in ax_u:
        return data, axes
    idx = ax_u.index(letter)
    d = _index_first_along_axis(data, idx)
    ax = axes[:idx] + axes[idx + 1 :]
    return d, ax


def _collapse_time_and_channel(data: np.ndarray, axes: str) -> Tuple[np.ndarray, str]:
    """Remove singleton *T*/*C* or keep only the first index when *T*/*C* > 1."""
    d, ax = data, axes
    # Time: singletons are already squeezed; if multiple frames remain, use t=0.
    while True:
        ax_u = ax.upper()
        if "T" not in ax_u:
            break
        ti = ax_u.index("T")
        if d.shape[ti] == 1:
            d = np.squeeze(d, axis=ti)
            ax = ax[:ti] + ax[ti + 1 :]
            continue
        d, ax = _take_axis0(d, ax, "T")
        break
    # Channel: this plugin expects a single 3-D scalar volume.
    while True:
        ax_u = ax.upper()
        if "C" not in ax_u:
            break
        ci = ax_u.index("C")
        if d.shape[ci] == 1:
            d = np.squeeze(d, axis=ci)
            ax = ax[:ci] + ax[ci + 1 :]
            continue
        d, ax = _take_axis0(d, ax, "C")
        break
    return d, ax


def transpose_perm_to_zyx(axes: str, ndim: int) -> Tuple[int, int, int]:
    """Permutation for ``numpy.transpose`` / ``dask.transpose`` to obtain ``ZYX`` order."""
    ax_u = axes.upper()
    if ndim != len(ax_u):
        raise ValueError(f"axes {axes!r} does not match array ndim {ndim}")
    if ax_u == "ZYX":
        return (0, 1, 2)
    if set(ax_u) >= {"Z", "Y", "X"}:
        return (ax_u.index("Z"), ax_u.index("Y"), ax_u.index("X"))
    raise ValueError(
        f"Cannot map axes {axes!r} to ZYX for 3-D vessel segmentation "
        "(need Z, Y, and X dimensions)."
    )


def _transpose_to_zyx(data: np.ndarray, axes: str) -> Tuple[np.ndarray, str]:
    """Return *data* shaped ``(Z, Y, X)`` with *axes* ``'ZYX'`` when possible."""
    perm = transpose_perm_to_zyx(axes, data.ndim)
    if perm == (0, 1, 2):
        return data, "ZYX"
    return np.transpose(data, perm), "ZYX"


def _scale_for_axes(axes_zyx: str, scales_xyz: Dict[str, float]) -> Tuple[float, float, float]:
    """``napari`` image scale order matches array axes (here Z, Y, X)."""
    mapping = {"Z": scales_xyz["Z"], "Y": scales_xyz["Y"], "X": scales_xyz["X"]}
    return tuple(float(mapping[a]) for a in axes_zyx.upper())


def volume_spacing_meta_pre_transpose(
    data: np.ndarray,
    axes: str,
    omexml: str,
    path_for_errors: str | Path,
) -> Tuple[np.ndarray, str, Tuple[float, float, float], Dict[str, Any], bool]:
    """Squeeze/collapse to 3-D and spacing/meta, **before** mapping axes to ``ZYX``.

    Keeping the array in on-disk axis order avoids building a non-contiguous
    transposed view of a huge memory map (which can trigger SIGBUS when read in
    arbitrary chunk shapes).
    """
    path_for_errors = Path(path_for_errors)
    scales_xyz, physical_sizes_in_ome = _ome_axis_scales_microns(omexml)
    meta: Dict[str, Any] = {
        "ome_voxel_size_microns": {
            "x": scales_xyz["X"],
            "y": scales_xyz["Y"],
            "z": scales_xyz["Z"],
        },
        "ome_axes_original": axes,
        "source_path": str(path_for_errors.resolve()),
    }
    # ``np.asarray`` can drop the ``memmap`` subclass and break downstream mmap checks;
    # ``asanyarray`` keeps memory-backed subclasses (e.g. slab temp file).
    d, ax = _squeeze_trailing_extras(np.asanyarray(data), axes)
    # Warn before silently keeping only index 0 of a multi-time / multi-channel stack.
    ax_u = ax.upper()
    for letter, label in (("T", "time points"), ("C", "channels")):
        if letter in ax_u:
            n = int(d.shape[ax_u.index(letter)])
            if n > 1:
                warnings.warn(
                    f"{path_for_errors.name}: OME-TIFF has {n} {label}; this reader "
                    "uses index 0 only (single 3-D scalar volume).",
                    stacklevel=2,
                )
    d, ax = _collapse_time_and_channel(d, ax)

    if d.ndim != 3:
        raise ValueError(
            f"{path_for_errors}: after removing singletons / picking a channel, expected "
            f"a 3-D volume, got shape {d.shape} with axes {ax!r}."
        )
    try:
        transpose_perm_to_zyx(ax, d.ndim)
    except ValueError as exc:
        raise ValueError(f"{path_for_errors}: {exc}") from exc

    scale = _scale_for_axes("ZYX", scales_xyz)
    return d, ax, scale, meta, physical_sizes_in_ome


def volume_zyx_spacing_meta_from_stack(
    data: np.ndarray,
    axes: str,
    omexml: str,
    path_for_errors: str | Path,
) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, Any], bool]:
    """Normalize a TIFF series stack to ``ZYX`` plus spacing and OME-style metadata.

    *data* may be a memory-mapped array; ``numpy.asarray`` is used only where a
    view is insufficient for axis cleanup.
    """
    d, ax, scale, meta, physical_sizes_in_ome = volume_spacing_meta_pre_transpose(
        data, axes, omexml, path_for_errors
    )
    d, ax = _transpose_to_zyx(d, ax)
    return d, scale, meta, physical_sizes_in_ome


def read_ome_tiff(path: str | Path) -> List[LayerDataTuple]:
    """Load the first series of an OME-TIFF as a single-channel ``ZYX`` volume."""
    path = Path(path)
    with tf.TiffFile(path) as tif:
        if not tif.series:
            return []
        series = tif.series[0]
        # Prefer a memory map so multi-GB volumes are not fully loaded into RAM;
        # tifffile falls back with ValueError for compressed/non-contiguous data.
        try:
            data = series.asarray(out="memmap")
        except (ValueError, MemoryError, NotImplementedError):
            data = series.asarray()
        axes = series.axes
        omexml = tif.ome_metadata or ""

    d, scale, meta, physical_sizes_in_ome = volume_zyx_spacing_meta_from_stack(
        data, axes, omexml, path
    )
    kwargs: Dict[str, Any] = {
        "name": path.stem,
        "rgb": False,
        "scale": np.asarray(scale, dtype=float),
        "metadata": meta,
    }
    if physical_sizes_in_ome:
        # ``PhysicalSize*`` values are converted to micrometres above; match units
        # so napari does not warn when labels/points copy this layer's metadata.
        kwargs["units"] = ("μm", "μm", "μm")

    return [(d, kwargs, "image")]


def napari_get_reader(path: Any) -> Optional[Callable[[str], List[LayerDataTuple]]]:
    """``npe2`` / napari reader hook for ``*.ome.tif`` / ``*.ome.tiff``."""
    if isinstance(path, (list, tuple)):
        if not path:
            return None
        path = path[0]
    path = str(path)
    lower = path.lower()
    if not (lower.endswith(".ome.tif") or lower.endswith(".ome.tiff")):
        return None

    def _reader(_path: str) -> List[LayerDataTuple]:
        return read_ome_tiff(_path)

    return _reader

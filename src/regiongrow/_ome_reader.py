"""OME-TIFF reader with voxel spacing from embedded OME-XML metadata."""

from __future__ import annotations

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
                d = np.take(d, 0, axis=i)
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
    d = np.take(data, 0, axis=idx)
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


def _transpose_to_zyx(data: np.ndarray, axes: str) -> Tuple[np.ndarray, str]:
    """Return *data* shaped ``(Z, Y, X)`` with *axes* ``'ZYX'`` when possible."""
    ax_u = axes.upper()
    if data.ndim != len(ax_u):
        raise ValueError(f"axes {axes!r} does not match array ndim {data.ndim}")

    if ax_u == "ZYX":
        return data, "ZYX"

    if set(ax_u) >= {"Z", "Y", "X"}:
        perm = [ax_u.index("Z"), ax_u.index("Y"), ax_u.index("X")]
        d = np.transpose(data, perm)
        return d, "ZYX"

    raise ValueError(
        f"Cannot map axes {axes!r} to ZYX for 3-D vessel segmentation "
        "(need Z, Y, and X dimensions)."
    )


def _scale_for_axes(axes_zyx: str, scales_xyz: Dict[str, float]) -> Tuple[float, float, float]:
    """``napari`` image scale order matches array axes (here Z, Y, X)."""
    mapping = {"Z": scales_xyz["Z"], "Y": scales_xyz["Y"], "X": scales_xyz["X"]}
    return tuple(float(mapping[a]) for a in axes_zyx.upper())


def read_ome_tiff(path: str | Path) -> List[LayerDataTuple]:
    """Load the first series of an OME-TIFF as a single-channel ``ZYX`` volume."""
    path = Path(path)
    with tf.TiffFile(path) as tif:
        if not tif.series:
            return []
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes
        omexml = tif.ome_metadata or ""

    scales_xyz, physical_sizes_in_ome = _ome_axis_scales_microns(omexml)
    meta: Dict[str, Any] = {
        "ome_voxel_size_microns": {
            "x": scales_xyz["X"],
            "y": scales_xyz["Y"],
            "z": scales_xyz["Z"],
        },
        "ome_axes_original": axes,
        "source_path": str(path.resolve()),
    }
    d, ax = _squeeze_trailing_extras(np.asarray(data), axes)
    d, ax = _collapse_time_and_channel(d, ax)
    try:
        d, ax = _transpose_to_zyx(d, ax)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    if d.ndim != 3:
        raise ValueError(
            f"{path}: after removing singletons / picking a channel, expected a 3-D "
            f"volume, got shape {d.shape} with axes {ax!r}."
        )

    scale = _scale_for_axes("ZYX", scales_xyz)
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

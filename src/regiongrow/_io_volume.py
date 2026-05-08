"""Load 3-D ZYX volumes + voxel spacing from TIFF (OME or generic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tifffile as tf

from ._ome_reader import read_ome_tiff, volume_zyx_spacing_meta_from_stack


def load_volume_zyx_from_path(path: str | Path) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, Any]]:
    """Load a single-channel 3-D volume as ZYX and spacing (Z, Y, X) in micrometres.

    Uses the same axis / OME spacing rules as :func:`regiongrow._ome_reader.read_ome_tiff`.
    For ``*.ome.tif`` / ``*.ome.tiff`` delegates to that reader. Other TIFFs use tifffile
    series 0 with spacing defaulting to 1 µm if no OME XML is present.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    lower = path.name.lower()
    if lower.endswith(".ome.tif") or lower.endswith(".ome.tiff"):
        layers = read_ome_tiff(path)
        if not layers:
            raise ValueError(f"No image data in {path}")
        data, kwargs, _ = layers[0]
        arr = np.asarray(data)
        meta = dict(kwargs.get("metadata", {}))
        sc = np.asarray(kwargs.get("scale", [1.0, 1.0, 1.0]), dtype=float).ravel()
        if sc.size < 3:
            spacing = (1.0, 1.0, 1.0)
        else:
            spacing = (float(sc[-3]), float(sc[-2]), float(sc[-1]))
        return arr, spacing, meta

    with tf.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"No TIFF series in {path}")
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes
        omexml = tif.ome_metadata or ""

    d, spacing, meta_full, _ = volume_zyx_spacing_meta_from_stack(
        np.asarray(data), axes, omexml, path
    )
    meta = {
        "source_path": meta_full["source_path"],
        "ome_axes_original": meta_full.get("ome_axes_original", axes),
    }
    return d, spacing, meta

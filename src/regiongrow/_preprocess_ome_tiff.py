"""In-memory preprocessing for OME-TIFF volumes (same steps as the Zarr pipeline).

Mean downsample (optional) then contrast stretch (optional), using
:func:`regiongrow._preprocessing.contrast_range_from_crop` and
:func:`regiongrow._preprocessing.stretch_contrast`.

The full volume is loaded into RAM. Writes use ``tifffile`` ``ome=True`` with **lossless
zlib** by default (predictor on) so outputs are much smaller than uncompressed TIFF;
override with ``compression='none'`` if needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tifffile as tf

from regiongrow._ome_reader import volume_zyx_spacing_meta_from_stack
from regiongrow._preprocessing import contrast_range_from_crop, stretch_contrast


def _mean_pool_zyx(vol: np.ndarray, fz: int, fxy: int) -> np.ndarray:
    """Integer mean pool (same rule as :func:`regiongrow._preprocessing_zarr._mean_pool_write`)."""
    fz = max(1, int(fz))
    fxy = max(1, int(fxy))
    z, y, x = vol.shape
    nz, ny, nx = z // fz, y // fxy, x // fxy
    if nz < 1 or ny < 1 or nx < 1:
        raise ValueError("Downsample factors too large for this volume.")
    v = vol[: nz * fz, : ny * fxy, : nx * fxy]
    tb = v.reshape(nz, fz, ny, fxy, nx, fxy)
    return tb.mean(axis=(1, 3, 5)).astype(vol.dtype, copy=False)


def load_ome_tiff_zyx(path: str | Path) -> Tuple[np.ndarray, Tuple[float, float, float], bool]:
    """Return contiguous ``(Z,Y,X)``, spacing ``(µm, µm, µm)``, and whether OME had explicit sizes."""
    path = Path(path)
    with tf.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"No TIFF series in {path}")
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes
        omexml = tif.ome_metadata or ""
    d, scale, _meta, physical_sizes_in_ome = volume_zyx_spacing_meta_from_stack(
        data, axes, omexml, path
    )
    return np.ascontiguousarray(d), scale, physical_sizes_in_ome


def write_ome_tiff_zyx(
    path: str | Path,
    arr: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    *,
    physical_sizes_in_ome: bool,
    compression: str = "zlib",
    compression_level: Optional[int] = 6,
    predictor: bool = True,
    tile_zyx: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Write a single-channel OME-TIFF with ``ZYX`` axes and optional physical pixel sizes.

    By default uses **lossless zlib** (DEFLATE) with a horizontal **predictor**, which
    typically shrinks microscopy volumes versus uncompressed TIFF (often the default when
    no ``compression`` is passed to ``tifffile``).
    """
    path = Path(path)
    sz, sy, sx = (float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2]))
    metadata: Dict[str, Any] = {
        "axes": "ZYX",
        "PhysicalSizeZ": sz,
        "PhysicalSizeY": sy,
        "PhysicalSizeX": sx,
    }
    if physical_sizes_in_ome:
        u = "µm"
        metadata["PhysicalSizeZUnit"] = u
        metadata["PhysicalSizeYUnit"] = u
        metadata["PhysicalSizeXUnit"] = u

    write_kw: Dict[str, Any] = {
        "ome": True,
        "metadata": metadata,
        "photometric": "minisblack",
    }
    c = (compression or "none").strip().lower()
    if c not in ("none", "off", ""):
        write_kw["compression"] = c
        lev = compression_level
        if lev is not None:
            lev = int(lev)
            if c in ("zlib", "deflate"):
                write_kw["compressionargs"] = {"level": max(0, min(9, lev))}
            elif c == "zstd":
                write_kw["compressionargs"] = {"level": max(0, min(22, lev))}
        if predictor:
            write_kw["predictor"] = True
    if tile_zyx is not None:
        tz, ty, tx = (max(1, int(x)) for x in tile_zyx)
        write_kw["tile"] = (tz, ty, tx)
        write_kw["volumetric"] = True
    # Classic TIFF is limited to ~4 GiB per IFD; enable BigTIFF early for large volumes.
    if int(arr.nbytes) >= 3_500_000_000:
        write_kw["bigtiff"] = True

    tf.imwrite(str(path), arr, **write_kw)


def run_preprocess_ome_tiff_pipeline(
    inp: str | Path,
    outp: str | Path,
    *,
    apply_downsample: bool,
    downsample_z: int,
    downsample_xy: int,
    apply_stretch: bool,
    stretch_mode: str,
    percentile_low: float,
    percentile_high: float,
    fixed_background: float,
    fixed_vessel_max: float,
    out_dtype: str = "uint8",
    finest_only: bool = False,
    compression: str = "zlib",
    compression_level: Optional[int] = 6,
    predictor: bool = True,
    tile_zyx: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    """Preprocess one OME-TIFF volume (optional mean downsample + optional contrast stretch).

    ``finest_only`` matches the Zarr pipeline signature and is ignored (single-resolution TIFF).
    """
    _ = finest_only
    inp, outp = Path(inp), Path(outp)
    post: Dict[str, Any] = {}
    arr, spacing, physical_sizes_in_ome = load_ome_tiff_zyx(inp)
    fz, fxy = max(1, int(downsample_z)), max(1, int(downsample_xy))

    if apply_downsample and (fz > 1 or fxy > 1):
        arr = _mean_pool_zyx(arr, fz, fxy)
        spacing = (spacing[0] * fz, spacing[1] * fxy, spacing[2] * fxy)
        post["downsample"] = {
            "factor_z": fz,
            "factor_xy": fxy,
            "spacing_after": list(spacing),
        }

    if apply_stretch:
        if stretch_mode == "percentile":
            lo, hi = contrast_range_from_crop(
                arr, float(percentile_low), float(percentile_high)
            )
            post["contrast_stretch"] = {
                "mode": "percentile",
                "input_min": lo,
                "input_max": hi,
                "percentile_low": float(percentile_low),
                "percentile_high": float(percentile_high),
            }
        else:
            lo, hi = float(fixed_background), float(fixed_vessel_max)
            post["contrast_stretch"] = {
                "mode": "fixed",
                "input_min": lo,
                "input_max": hi,
            }
        arr, sinfo = stretch_contrast(arr, lo, hi, out_dtype=out_dtype)
        post["contrast_stretch"].update(sinfo)

    write_ome_tiff_zyx(
        outp,
        arr,
        spacing,
        physical_sizes_in_ome=physical_sizes_in_ome,
        compression=compression,
        compression_level=compression_level,
        predictor=predictor,
        tile_zyx=tile_zyx,
    )
    post["output_path"] = str(outp.resolve())
    post["shape_zyx"] = tuple(int(x) for x in arr.shape)
    post["tiff_write"] = {
        "compression": compression,
        "compression_level": compression_level,
        "predictor": predictor,
        "tile_zyx": tile_zyx,
    }
    return post


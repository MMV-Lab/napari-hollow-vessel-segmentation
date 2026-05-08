"""Helpers for napari Image layers: multiscale, lazy arrays, RAM estimates."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    import dask.array as da
except ImportError:  # pragma: no cover
    da = None  # type: ignore


def is_multiscale_image_layer(layer: Any) -> bool:
    return bool(getattr(layer, "multiscale", False))


def _multiscale_levels_data(layer: Any) -> Any:
    """Return indexable per-level storage, or ``None`` if not multiscale / single array.

    Napari wraps multiscale image ``data`` in :class:`napari.layers._multiscale_data.MultiScaleData`
    (a ``Sequence`` but not a ``list``/``tuple``). OME-Zarr and other readers use that path.
    """
    if not is_multiscale_image_layer(layer):
        return None
    d = getattr(layer, "data", None)
    if d is None:
        return None
    if isinstance(d, (list, tuple)):
        return d
    if isinstance(d, np.ndarray):
        return None
    if hasattr(d, "__len__") and hasattr(d, "__getitem__"):
        try:
            n = len(d)
        except Exception:  # pragma: no cover
            return None
        if n == 0:
            return None
        try:
            el0 = d[0]
            if not hasattr(el0, "shape"):
                return None
        except Exception:  # pragma: no cover
            return None
        return d
    return None


def multiscale_level_count(layer: Any) -> int:
    levels = _multiscale_levels_data(layer)
    if levels is not None:
        return len(levels)
    return 1


def _unwrap_data_element(x: Any) -> Any:
    """Return underlying array-like (dask or numpy) for one multiscale level."""
    return x


def _arraylike_shape(x: Any) -> Tuple[int, ...]:
    """Return ``shape`` without ``np.asarray`` (Zarr/Dask would otherwise load the whole volume)."""
    el = _unwrap_data_element(x)
    shp = getattr(el, "shape", None)
    if shp is not None:
        return tuple(int(s) for s in shp)
    return tuple(np.asarray(el).shape)


def layer_data_shape(layer: Any) -> Tuple[int, ...]:
    """Shape of ``layer.data`` without materializing lazy arrays (Labels / Image)."""
    d = getattr(layer, "data", None)
    if d is None:
        return tuple()
    return _arraylike_shape(d)


def image_level_shape(layer: Any, level: int = 0) -> Tuple[int, ...]:
    """Shape of *layer*'s data at *level* (0 = finest for napari multiscale)."""
    levels = _multiscale_levels_data(layer)
    if levels is not None:
        idx = int(np.clip(level, 0, len(levels) - 1))
        return _arraylike_shape(levels[idx])
    d = getattr(layer, "data", None)
    if d is None:
        return tuple()
    shp = getattr(d, "shape", None)
    if shp is not None:
        return tuple(int(s) for s in shp)
    return tuple(np.asarray(d).shape)


def image_finest_shape(layer: Any) -> Tuple[int, ...]:
    return image_level_shape(layer, 0)


def is_lazy_array(x: Any) -> bool:
    if da is not None and isinstance(x, da.Array):
        return True
    mod = type(x).__module__ or ""
    if "dask" in mod and hasattr(x, "compute"):
        return True
    return False


def image_level_is_lazy(layer: Any, level: int = 0) -> bool:
    levels = _multiscale_levels_data(layer)
    if levels is not None:
        idx = int(np.clip(level, 0, len(levels) - 1))
        return is_lazy_array(levels[idx])
    return is_lazy_array(layer.data)


def estimate_dense_bytes(shape: Tuple[int, ...], dtype: Any, copies: float = 1.0) -> float:
    """Approximate RAM in bytes for *copies* contiguous arrays of *shape*."""
    item = np.dtype(dtype).itemsize if dtype is not None else 8
    n = int(np.prod(shape, dtype=np.int64)) if shape else 0
    return float(n * item * copies)


def voxel_spacing_zyx_finest(layer: Any) -> Tuple[float, float, float]:
    """Voxel spacing (Z, Y, X) treating ``layer.scale`` as finest-level µm/voxel."""
    sc = np.asarray(layer.scale, dtype=np.float64).ravel()
    if sc.size < 3:
        return (1.0, 1.0, 1.0)
    s = sc[-3:].copy()
    s[s <= 0] = 1.0
    return (float(s[0]), float(s[1]), float(s[2]))


def voxel_spacing_zyx_for_level(
    layer: Any,
    level: int,
    level_shape: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Physical spacing (Z, Y, X) matching *level_shape* (finest scale × size ratio)."""
    sz0, sy0, sx0 = voxel_spacing_zyx_finest(layer)
    if not is_multiscale_image_layer(layer):
        return (sz0, sy0, sx0)
    finest = image_finest_shape(layer)
    if len(finest) != 3 or len(level_shape) != 3:
        return (sz0, sy0, sx0)
    rz = finest[0] / max(level_shape[0], 1)
    ry = finest[1] / max(level_shape[1], 1)
    rx = finest[2] / max(level_shape[2], 1)
    return (sz0 * rz, sy0 * ry, sx0 * rx)


def tube_radius_voxels_for_work_level(
    radius_finest_isotropic_voxels: float,
    layer: Any,
    level: int,
    level_shape: Tuple[int, int, int],
) -> float:
    """Map UI tube radius (finest isotropic voxel radii) to voxels at *level*.

    :func:`regiongrow._algorithm.polyline_tube_mask` and MGAC use
    ``radius_phys = radius_vox * min(spacing)``.  Coarser pyramid levels have
    larger *spacing*, so the same spin would otherwise grow in world units.
    This keeps ``radius_phys = radius_ui * min(finest spacing)`` at every level.
    """
    r_ui = float(radius_finest_isotropic_voxels)
    if r_ui <= 0:
        return r_ui
    sf = np.asarray(voxel_spacing_zyx_finest(layer), dtype=np.float64).ravel()[:3]
    sw = np.asarray(
        voxel_spacing_zyx_for_level(layer, level, level_shape), dtype=np.float64
    ).ravel()[:3]
    min_f = float(np.min(sf))
    min_w = float(np.min(sw))
    if min_w <= 0:
        return r_ui
    return r_ui * (min_f / min_w)


def axis_margin_voxels_for_work_level(
    margin_finest_mean_voxels: float,
    layer: Any,
    level: int,
    level_shape: Tuple[int, int, int],
) -> float:
    """Scale length-margin spin so ``margin × mean(spacing_work)`` matches finest.

    :func:`regiongrow._algorithm.region_grow` uses
    ``margin_phys = margin * mean(spacing)``.  Interpreting the spin at the
    finest level gives reproducible physical slack when switching pyramid levels.
    """
    m = float(margin_finest_mean_voxels)
    sf = np.asarray(voxel_spacing_zyx_finest(layer), dtype=np.float64).ravel()[:3]
    sw = np.asarray(
        voxel_spacing_zyx_for_level(layer, level, level_shape), dtype=np.float64
    ).ravel()[:3]
    mean_f = float(np.mean(sf))
    mean_w = float(np.mean(sw))
    if mean_w <= 0:
        return m
    return m * (mean_f / mean_w)


def materialize_image_level(
    layer: Any,
    level: int,
    dtype: Optional[Any] = np.float64,
    *,
    slices: Optional[Tuple[slice, ...]] = None,
) -> np.ndarray:
    """``np.asarray`` of image data at *level* (finest = 0 when multiscale).

    If *slices* is set (e.g. ``(slice(z0, z1), …)``), only that subvolume is
    materialized (tests / advanced callers); omit for the full level.
    """
    levels = _multiscale_levels_data(layer)
    if levels is not None:
        idx = int(np.clip(level, 0, len(levels) - 1))
        block = _unwrap_data_element(levels[idx])
    else:
        if is_multiscale_image_layer(layer):
            raise TypeError(
                "multiscale image has no indexable level data "
                "(expected list/tuple or napari MultiScaleData)."
            )
        block = layer.data
    if slices is not None:
        block = block[slices]
    if dtype is None:
        return np.asarray(block)
    return np.asarray(block, dtype=dtype)


def check_materialization_budget(
    shape: Tuple[int, ...],
    dtype: Any,
    *,
    max_bytes: float,
    copies: float,
    context: str,
) -> Optional[str]:
    """Return error string if estimated RAM exceeds *max_bytes*, else None."""
    need = estimate_dense_bytes(shape, dtype, copies=copies)
    if need > max_bytes:
        gb = need / 1e9
        cap = max_bytes / 1e9
        return (
            f"{context}: need ~{gb:.1f} GB for {shape} {dtype} (~{copies} full copies). "
            f"Limit is {cap:.1f} GB. Use a coarser pyramid level "
            f"or run on a machine with more RAM."
        )
    return None


def multiscale_level_label(layer: Any, level: int) -> str:
    shp = image_level_shape(layer, level)
    return f"Level {level} — shape {shp}"

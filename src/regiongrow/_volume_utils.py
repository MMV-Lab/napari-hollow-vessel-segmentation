"""Helpers for napari Image layers: multiscale, lazy arrays, RAM estimates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Session cache: (layer name, pyramid level, dtype name, data id) → full-level array.
# The data-id component invalidates the entry automatically when a layer's data
# is replaced (the backing array object changes) while keeping it valid across
# renames (same array object), which matches the user's intent.
CacheKey = Tuple[str, int, str, int]
_IMAGE_LEVEL_CACHE: Dict[CacheKey, np.ndarray] = {}
_IMAGE_LEVEL_CACHE_ORDER: List[CacheKey] = []
# Default ~6 GB; evict oldest levels when exceeded (grow + 3D display share this cache).
_IMAGE_LEVEL_CACHE_MAX_BYTES = int(6e9)

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


def default_pyramid_level_index(num_levels: int, preferred: int = 3) -> int:
    """Default working pyramid index: *preferred* when present, else coarsest."""
    n = int(num_levels)
    if n <= 1:
        return 0
    return min(int(preferred), n - 1)


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


def image_level_data(layer: Any, level: int = 0) -> Any:
    """Array-like data at *level* without materializing lazy stores."""
    levels = _multiscale_levels_data(layer)
    if levels is not None:
        idx = int(np.clip(level, 0, len(levels) - 1))
        return _unwrap_data_element(levels[idx])
    return getattr(layer, "data", None)


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


def labels_pyramid_level_for_image_level(
    labels_layer: Any, image_layer: Any, image_level: int
) -> int:
    """Pick the labels pyramid index whose grid matches *image_level*."""
    try:
        target = tuple(int(x) for x in image_level_shape(image_layer, int(image_level)))
    except (TypeError, ValueError, IndexError):
        return 0
    if len(target) != 3:
        return 0
    levels = _multiscale_levels_data(labels_layer)
    if levels is None:
        return 0
    for idx in range(len(levels)):
        try:
            shp = tuple(int(x) for x in image_level_shape(labels_layer, idx))
        except (TypeError, ValueError, IndexError):
            continue
        if shp == target:
            return int(idx)
    return int(np.clip(int(image_level), 0, len(levels) - 1))


def materialize_labels_level(layer: Any, level: int = 0) -> np.ndarray:
    """Dense snapshot of a Labels layer at multiscale *level* (0 = finest).

  ``np.asarray(labels_layer.data)`` on multiscale labels returns the **coarsest**
    level in napari — never use that for saving the edited mask.
    """
    block = image_level_data(layer, int(level))
    if block is None:
        return np.asarray(getattr(layer, "data", np.zeros(1, dtype=np.uint8)))
    if is_lazy_array(block):
        if da is not None and isinstance(block, da.Array):
            return np.asarray(block.compute())
        return np.asarray(block)
    return np.asarray(block)


def image_finest_shape(layer: Any) -> Tuple[int, ...]:
    return image_level_shape(layer, 0)


def finest_labels_data_shape(labels_data: Any) -> Tuple[int, ...]:
    """Shape (…, Z, Y, X) of the finest grid in saved / napari multiscale labels data."""
    if isinstance(labels_data, (list, tuple)):
        if not labels_data:
            return tuple()
        return _arraylike_shape(labels_data[0])
    return _arraylike_shape(labels_data)


def image_level_index_for_shape(
    image_layer: Any, shape: Tuple[int, ...]
) -> Optional[int]:
    """Pyramid level whose grid matches *shape* (exact, else closest ZYX size)."""
    tgt = tuple(int(x) for x in shape)
    if len(tgt) < 3:
        return None
    tgt3 = tuple(int(x) for x in tgt[-3:])
    best_lvl: Optional[int] = None
    best_score = float("inf")
    for lvl in range(multiscale_level_count(image_layer)):
        try:
            shp = tuple(int(x) for x in image_level_shape(image_layer, lvl))
        except (TypeError, ValueError, IndexError):
            continue
        if len(shp) < 3:
            continue
        shp3 = tuple(int(x) for x in shp[-3:])
        if shp3 == tgt3:
            return int(lvl)
        score = sum(
            abs(
                float(np.log(max(int(a), 1)))
                - float(np.log(max(int(b), 1)))
            )
            for a, b in zip(shp3, tgt3)
        )
        if score < best_score:
            best_score = score
            best_lvl = int(lvl)
    return best_lvl


def _level_downsample_factors(
    layer: Any,
    level: int,
    level_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[float, float, float]:
    """Per-axis (Z, Y, X) downsample factors of *level* relative to finest.

    Single source of truth for pyramid geometry: prefer napari
    ``layer.downsample_factors`` (what the canvas and coordinate transforms use)
    and fall back to the finest/level shape ratio only when factors are missing.
    Keeping spacing, slider steps, and polyline mapping on the same factors
    avoids divergence on padded / non-uniform pyramids.
    """
    level = int(level)
    if level <= 0:
        return (1.0, 1.0, 1.0)
    if is_multiscale_image_layer(layer):
        try:
            df = np.asarray(
                layer.downsample_factors[level], dtype=np.float64
            ).ravel()
            if df.size >= 3:
                df3 = df[-3:].copy()
            else:
                df3 = np.ones(3, dtype=np.float64)
                df3[-df.size :] = df
            df3[df3 <= 0] = 1.0
            return (float(df3[0]), float(df3[1]), float(df3[2]))
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    finest = image_finest_shape(layer)
    work = level_shape if level_shape is not None else image_level_shape(layer, level)
    if len(finest) >= 3 and len(work) >= 3:
        return (
            float(finest[-3]) / max(int(work[-3]), 1),
            float(finest[-2]) / max(int(work[-2]), 1),
            float(finest[-1]) / max(int(work[-1]), 1),
        )
    return (1.0, 1.0, 1.0)


def pyramid_axis_steps(layer: Any, level: int) -> Tuple[int, int, int]:
    """Finest-voxel steps per Z,Y,X so one slider tick crosses one working-level voxel.

    At coarse pyramid levels each displayed slice spans several finest indices
    (block subsampling). Use these steps with napari ``dims.set_range(..., step)``.
    """
    if int(level) <= 0:
        return (1, 1, 1)
    fz, fy, fx = _level_downsample_factors(layer, int(level))
    return (max(1, int(round(fz))), max(1, int(round(fy))), max(1, int(round(fx))))


def pyramid_navigation_axis_ranges_zyx(
    layer: Any, level: int
) -> Tuple[
    Optional[Tuple[float, float, float]],
    Optional[Tuple[float, float, float]],
    Optional[Tuple[float, float, float]],
]:
    """World ``(lo, hi, step)`` per Z, Y, X for coarse pyramid dims navigation.

    ``lo``/``hi`` follow the **working** pyramid level (matches displayed voxels).
    ``step`` is ``downsample_factor × finest_spacing`` so each tick is one
    working-level plane without overshooting past the last valid slice.
    """
    from regiongrow._spatial import world_bounds_zyx_for_pyramid_level

    steps = pyramid_axis_steps(layer, int(level))
    if steps == (1, 1, 1):
        return (None, None, None)
    finest = voxel_spacing_zyx_finest(layer)
    work_bounds = world_bounds_zyx_for_pyramid_level(layer, int(level))
    out: List[Optional[Tuple[float, float, float]]] = []
    for i in range(3):
        step_vox = int(steps[i])
        if step_vox <= 1:
            out.append(None)
            continue
        lo_b, hi_b = work_bounds[i]
        fs = float(finest[i])
        if fs <= 0:
            fs = 1.0
        world_step = max(float(step_vox) * fs, fs)
        out.append((float(lo_b), float(hi_b), world_step))
    return (out[0], out[1], out[2])


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
    """Voxel spacing (Z, Y, X) at the finest pyramid level in physical units.

    Prefers NGFF ``coordinateTransformations`` on ``layer.metadata`` when present
    (OME-Zarr often stores anisotropic Z,Y,X scale only there while napari leaves
    ``layer.scale`` at ``(1, 1, 1)``).
    """
    meta = getattr(layer, "metadata", None) or {}
    ngff = ngff_finest_voxel_spacing_zyx(meta)
    if ngff is not None:
        return ngff
    sc = np.asarray(layer.scale, dtype=np.float64).ravel()
    if sc.size < 3:
        return (1.0, 1.0, 1.0)
    s = sc[-3:].copy()
    s[s <= 0] = 1.0
    return (float(s[0]), float(s[1]), float(s[2]))


def ngff_scale_zyx_from_transforms(
    transforms: Any,
) -> Optional[Tuple[float, float, float]]:
    """Extract ``(s_z, s_y, s_x)`` from one NGFF ``coordinateTransformations`` list."""
    if not isinstance(transforms, list):
        return None
    for entry in transforms:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type", "")).lower() != "scale":
            continue
        raw = entry.get("scale")
        if raw is None:
            continue
        sc = np.asarray(raw, dtype=np.float64).ravel()
        if sc.size < 3:
            continue
        s = sc[-3:].copy()
        if np.any(s <= 0) or not np.all(np.isfinite(s)):
            continue
        return (float(s[0]), float(s[1]), float(s[2]))
    return None


def ngff_finest_voxel_spacing_zyx(metadata: Any) -> Optional[Tuple[float, float, float]]:
    """Finest-level physical spacing from napari/OME-Zarr ``metadata`` dict."""
    if not isinstance(metadata, dict):
        return None
    cts = metadata.get("coordinateTransformations")
    if not isinstance(cts, list) or len(cts) == 0:
        return None
    level0 = cts[0]
    if isinstance(level0, list):
        return ngff_scale_zyx_from_transforms(level0)
    if isinstance(level0, dict):
        return ngff_scale_zyx_from_transforms([level0])
    return None


def voxel_spacing_zyx_for_level(
    layer: Any,
    level: int,
    level_shape: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Physical spacing (Z, Y, X) at *level*, using napari downsample factors.

    Derived from the finest ``layer.scale`` times the per-axis pyramid
    downsample factors (with a shape-ratio fallback), so physical tube radius
    and axis margins at coarse levels match the displayed pyramid geometry.
    """
    sz0, sy0, sx0 = voxel_spacing_zyx_finest(layer)
    if not is_multiscale_image_layer(layer) or int(level) <= 0:
        return (sz0, sy0, sx0)
    fz, fy, fx = _level_downsample_factors(layer, int(level), level_shape)
    return (sz0 * fz, sy0 * fy, sx0 * fx)


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


def _image_level_cache_touch(key: CacheKey) -> None:
    try:
        _IMAGE_LEVEL_CACHE_ORDER.remove(key)
    except ValueError:
        pass
    _IMAGE_LEVEL_CACHE_ORDER.append(key)


def _image_level_cache_evict_if_needed() -> None:
    total = sum(int(a.nbytes) for a in _IMAGE_LEVEL_CACHE.values())
    while total > _IMAGE_LEVEL_CACHE_MAX_BYTES and _IMAGE_LEVEL_CACHE_ORDER:
        old = _IMAGE_LEVEL_CACHE_ORDER.pop(0)
        arr = _IMAGE_LEVEL_CACHE.pop(old, None)
        if arr is not None:
            total -= int(arr.nbytes)


def clear_image_level_cache() -> None:
    """Drop all cached pyramid levels (e.g. when closing the widget)."""
    _IMAGE_LEVEL_CACHE.clear()
    _IMAGE_LEVEL_CACHE_ORDER.clear()


def invalidate_image_level_cache(layer_name: str) -> None:
    """Remove cached levels for one image layer name."""
    drop = [k for k in _IMAGE_LEVEL_CACHE if k[0] == layer_name]
    for k in drop:
        del _IMAGE_LEVEL_CACHE[k]
        try:
            _IMAGE_LEVEL_CACHE_ORDER.remove(k)
        except ValueError:
            pass


def _image_level_cache_key(layer: Any, level: int, dtype: Optional[Any]) -> CacheKey:
    name = str(getattr(layer, "name", id(layer)))
    dname = "raw" if dtype is None else str(np.dtype(dtype))
    try:
        data_id = id(image_level_data(layer, int(level)))
    except Exception:
        data_id = 0
    return (name, int(level), dname, data_id)


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


def materialize_image_level_cached(
    layer: Any,
    level: int,
    dtype: Optional[Any] = np.float32,
    *,
    slices: Optional[Tuple[slice, ...]] = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Materialize with optional session cache of the full pyramid level.

    When *slices* is set, returns only that subvolume. If the full level is
    already cached, slices are taken from RAM; otherwise only the ROI is read
    from the backing store (no full-volume decode).
    """
    if not use_cache:
        return materialize_image_level(layer, level, dtype=dtype, slices=slices)

    key = _image_level_cache_key(layer, level, dtype)
    if key in _IMAGE_LEVEL_CACHE:
        _image_level_cache_touch(key)
        full = _IMAGE_LEVEL_CACHE[key]
        # Copy on return so a consumer (grow worker / 3D display) that mutates or
        # hands the array to napari cannot corrupt the shared cached master.
        if slices is None:
            return np.array(full, copy=True)
        return np.asarray(full[slices], dtype=full.dtype if dtype is None else dtype)

    if slices is not None:
        return materialize_image_level(layer, level, dtype=dtype, slices=slices)

    arr = materialize_image_level(layer, level, dtype=dtype)
    _IMAGE_LEVEL_CACHE[key] = arr
    _image_level_cache_touch(key)
    _image_level_cache_evict_if_needed()
    return arr


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


def multiscale_level_label(layer: Any, level: int, *, include_shape: bool = False) -> str:
    """Short label for dock combos; full shape via :func:`multiscale_level_tooltip`."""
    if not include_shape:
        return f"Level {level}"
    shp = image_level_shape(layer, level)
    return f"Level {level} — shape {shp}"


def multiscale_level_tooltip(layer: Any, level: int) -> str:
    shp = image_level_shape(layer, level)
    return f"Pyramid level {level}, shape Z×Y×X = {tuple(int(x) for x in shp)}"

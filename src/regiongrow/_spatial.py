"""Helpers to keep napari layer spatial metadata consistent."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def spatial_alignment_kwargs(ref_layer: Any) -> Dict[str, Any]:
    """Return kwargs so a new layer shares *ref_layer*'s data→world transform.

    Copies ``scale`` / ``translate`` / ``rotate`` / ``shear`` and, when present,
    ``units``.  Napari compares ``units`` across layers for physical rendering;
    mixing defaults (pixel) with micrometre-based images triggers warnings unless
    child layers reuse the image's ``units``.
    """
    out: Dict[str, Any] = {}
    for key in ("scale", "translate", "rotate", "shear"):
        val = getattr(ref_layer, key, None)
        if val is None:
            continue
        arr = np.asarray(val, dtype=float)
        out[key] = arr.copy()
    units = getattr(ref_layer, "units", None)
    if units is not None:
        out["units"] = units
    return out


def spatial_alignment_for_pyramid_level(ref_layer: Any, level: int) -> Dict[str, Any]:
    """Spatial kwargs for a layer whose ``data.shape`` matches pyramid *level*.

    Napari multiscale Images keep ``scale`` in **finest-voxel** physical units.
    A labels array at a coarser level has one voxel per ``downsample_factors[level]``
    finest voxels along each axis, so each coarse voxel must be drawn ``factor``
    times larger in world space — multiply ``scale`` accordingly. ``translate`` /
    ``rotate`` / ``shear`` stay unchanged (shared origin).
    """
    kwargs = spatial_alignment_kwargs(ref_layer)
    if not getattr(ref_layer, "multiscale", False):
        return kwargs
    if "scale" not in kwargs:
        return kwargs
    try:
        factors = np.asarray(
            ref_layer.downsample_factors[int(level)], dtype=np.float64
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return kwargs
    orig = np.asarray(kwargs["scale"], dtype=np.float64)
    flat = orig.ravel()
    fac = factors.ravel()
    if fac.size < flat.size:
        fac = np.pad(fac, (flat.size - fac.size, 0), constant_values=1.0)
    elif fac.size > flat.size:
        fac = fac[-flat.size :]
    fac[fac <= 0] = 1.0
    kwargs["scale"] = (flat * fac).reshape(orig.shape)
    return kwargs


def spatial_alignment_for_saved_labels(
    image_layer: Any, labels_data: Any
) -> Dict[str, Any]:
    """Spatial kwargs for NGFF labels loaded next to a (multiscale) image.

    Saved labels are often stored at a **working** pyramid resolution, not the
    image finest grid.  Align to whichever image pyramid level matches the
    finest labels array shape so the mask is not shifted off-screen (looks empty).
    """
    from regiongrow._volume_utils import (
        finest_labels_data_shape,
        image_level_index_for_shape,
    )

    shp = finest_labels_data_shape(labels_data)
    if len(shp) < 3:
        return spatial_alignment_kwargs(image_layer)
    shp3 = tuple(int(x) for x in shp[-3:])
    matched = image_level_index_for_shape(image_layer, shp3)
    if matched is not None:
        return spatial_alignment_for_pyramid_level(image_layer, int(matched))
    return spatial_alignment_kwargs(image_layer)


def world_bounds_zyx_for_pyramid_level(
    ref_layer: Any, level: int
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Inclusive world (min, max) along Z, Y, X for voxels at pyramid *level*."""
    from regiongrow._volume_utils import image_level_shape

    skw = spatial_alignment_for_pyramid_level(ref_layer, int(level))
    try:
        shp = tuple(int(x) for x in image_level_shape(ref_layer, int(level)))
    except (TypeError, ValueError, IndexError):
        shp = ()
    trans = np.asarray(
        skw.get("translate", getattr(ref_layer, "translate", 0)), dtype=np.float64
    ).ravel()
    scale = np.asarray(
        skw.get("scale", getattr(ref_layer, "scale", 1)), dtype=np.float64
    ).ravel()
    if trans.size < 3:
        trans = np.pad(trans, (3 - trans.size, 0), constant_values=0.0)
    if scale.size < 3:
        scale = np.pad(scale, (3 - scale.size, 0), constant_values=1.0)
    t3 = trans[-3:]
    s3 = scale[-3:].copy()
    s3[s3 <= 0] = 1.0
    if len(shp) >= 3:
        n3 = np.asarray(shp[-3:], dtype=np.float64)
    else:
        n3 = np.ones(3, dtype=np.float64)
    lo = t3
    hi = t3 + np.maximum(n3 - 1.0, 0.0) * s3
    return (
        (float(lo[0]), float(hi[0])),
        (float(lo[1]), float(hi[1])),
        (float(lo[2]), float(hi[2])),
    )


def scaled_spatial_kwargs(
    ref_layer: Any, *, scale_multiplier: np.ndarray | float | int
) -> Dict[str, Any]:
    """Like :func:`spatial_alignment_kwargs` but multiply *ref_layer*'s scale.

    Used after isotropic downsampling: each output voxel spans *factor* more
    physical extent along every axis, so ``new_scale = old_scale * factor``.
    """
    kwargs = spatial_alignment_kwargs(ref_layer)
    if "scale" not in kwargs:
        return kwargs
    mult = np.asarray(scale_multiplier, dtype=float)
    if mult.ndim == 0:
        mult = np.broadcast_to(mult, kwargs["scale"].shape)
    kwargs["scale"] = kwargs["scale"] * mult
    return kwargs

"""Helpers to keep napari layer spatial metadata consistent."""

from __future__ import annotations

from typing import Any, Dict

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

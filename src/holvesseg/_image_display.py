"""Copy napari Image layer display settings between layers (gamma, contrast, …)."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

# Visual / rendering knobs on ``napari.layers.Image`` (not data or pyramid grid).
_IMAGE_DISPLAY_ATTRS: Tuple[str, ...] = (
    "opacity",
    "blending",
    "gamma",
    "interpolation2d",
    "interpolation3d",
    "rendering",
    "depiction",
    "rgb",
    "attenuation",
    "iso_threshold",
    "projection_mode",
    "colormap",
    "contrast_limits",
)


def _is_contrast_limits_pair(x: Any) -> bool:
    if not isinstance(x, (list, tuple, np.ndarray)) or len(x) != 2:
        return False
    try:
        float(x[0])
        float(x[1])
        return True
    except (TypeError, ValueError):
        return False


def _is_per_level_display_list(val: Any) -> bool:
    """True if *val* looks like one entry per pyramid level (not a single contrast pair)."""
    if not isinstance(val, (list, tuple)) or len(val) < 2:
        return False
    if _is_contrast_limits_pair(val):
        return False
    return True


def _read_display_value(layer: Any, attr: str, *, pyramid_level: int) -> Any:
    val = getattr(layer, attr, None)
    if val is None:
        return None
    if _is_per_level_display_list(val):
        idx = int(np.clip(pyramid_level, 0, len(val) - 1))
        return val[idx]
    return val


def _write_display_value(
    layer: Any, attr: str, value: Any, *, pyramid_level: int
) -> None:
    if value is None:
        return
    current = getattr(layer, attr, None)
    if bool(getattr(layer, "multiscale", False)) and _is_per_level_display_list(
        current
    ):
        new_list: List[Any] = list(current)
        idx = int(pyramid_level)
        while len(new_list) <= idx:
            new_list.append(value)
        new_list[idx] = value
        setattr(layer, attr, new_list)
        return
    setattr(layer, attr, value)


def copy_image_display_settings(
    source: Any,
    target: Any,
    *,
    source_level: int = 0,
    target_level: int = 0,
) -> None:
    """Copy gamma / contrast / colormap / blending etc. from *source* to *target*."""
    if source is None or target is None:
        return
    for attr in _IMAGE_DISPLAY_ATTRS:
        try:
            val = _read_display_value(source, attr, pyramid_level=int(source_level))
            if val is None:
                continue
            _write_display_value(target, attr, val, pyramid_level=int(target_level))
        except (AttributeError, ValueError, TypeError, KeyError, IndexError) as exc:
            # A single unsupported display attribute should not abort the copy,
            # but log it so silent contrast/colormap drift can be diagnosed.
            _logger.debug("could not copy display attr %r: %s", attr, exc)
            continue


def copy_proxy_display_to_multiscale_source(
    proxy: Any,
    source: Any,
    *,
    pyramid_level: int,
) -> None:
    """Restore settings edited on a single-scale proxy onto the hidden multiscale layer."""
    copy_image_display_settings(
        proxy, source, source_level=0, target_level=int(pyramid_level)
    )


def copy_multiscale_source_display_to_proxy(
    source: Any,
    proxy: Any,
    *,
    pyramid_level: int,
) -> None:
    """Apply multiscale layer display state to a single-scale display proxy."""
    copy_image_display_settings(
        source, proxy, source_level=int(pyramid_level), target_level=0
    )

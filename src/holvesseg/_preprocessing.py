"""In-memory contrast-stretch helpers (shared by the OME-Zarr / OME-TIFF CLIs).

Suitable for full volumes or crops. Large 3-D volumes use a slab-wise float32
path so a full-volume float copy is never allocated.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# Above this flat-sample count, ``contrast_range_from_crop`` subsamples (stable RNG) so
# ``np.percentile`` does not need O(n) working memory on terapixel-class volumes.
_CONTRAST_PERCENTILE_MAX_SAMPLES = 12_000_000
# Volumes larger than this use slab-wise stretch (float32 temp) instead of a full copy.
_STRETCH_CHUNK_ENTRY_BYTES = 32 * 1024**2
# Max float32 working set for one (Y,X) slab or tile block (~256 MiB default).
_STRETCH_FLOAT_BUFFER_BUDGET_BYTES = 256 * 1024**2


def contrast_range_from_crop(
    crop: np.ndarray, p_low: float, p_high: float
) -> Tuple[float, float]:
    flat = np.asarray(crop).ravel()
    n = int(flat.size)
    if n > _CONTRAST_PERCENTILE_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        flat = flat[rng.choice(n, size=_CONTRAST_PERCENTILE_MAX_SAMPLES, replace=False)]
    in_min = float(np.percentile(flat, p_low))
    in_max = float(np.percentile(flat, p_high))
    if in_max <= in_min:
        in_max = in_min + 1.0
    return in_min, in_max


def _stretch_contrast_dense_float32(
    crop: np.ndarray,
    in_min: float,
    in_max: float,
    out_dtype: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Full-array path using float32 (sufficient for 8/16-bit mapping)."""
    crop_f = np.asarray(crop, dtype=np.float32)
    span = float(in_max - in_min)
    if span <= 0:
        span = 1.0
    out_f = (crop_f - np.float32(in_min)) / np.float32(span)
    out_f = np.clip(out_f, 0.0, 1.0)
    if out_dtype == "uint16":
        out = (out_f * np.float32(65535.0)).round().astype(np.uint16)
        post: Dict[str, Any] = {
            "input_min": in_min,
            "input_max": in_max,
            "output_range": [0, 65535],
        }
    else:
        out = (out_f * np.float32(255.0)).round().astype(np.uint8)
        post = {"input_min": in_min, "input_max": in_max, "output_range": [0, 255]}
    post["output_dtype"] = out_dtype
    return out, post


def _stretch_contrast_zyx_slabwise(
    crop: np.ndarray,
    in_min: float,
    in_max: float,
    out_dtype: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """3-D ZYX stretch without a full-volume float buffer (one slab / XY tile at a time)."""
    crop = np.asarray(crop)
    if crop.ndim != 3:
        raise ValueError("slabwise stretch expects a 3-D (Z, Y, X) array")
    z, y, x = (int(crop.shape[0]), int(crop.shape[1]), int(crop.shape[2]))
    out_dtype_np = np.uint16 if out_dtype == "uint16" else np.uint8
    out = np.empty((z, y, x), dtype=out_dtype_np)
    span = float(in_max - in_min)
    if span <= 0:
        span = 1.0
    scale = 65535.0 if out_dtype == "uint16" else 255.0
    f_in_min = np.float32(in_min)
    f_span = np.float32(span)
    f_scale = np.float32(scale)

    plane_elems = y * x
    plane_f32_bytes = plane_elems * 4
    budget = _STRETCH_FLOAT_BUFFER_BUDGET_BYTES

    if plane_f32_bytes <= budget:
        for zi in range(z):
            # Always copy: ``astype(..., copy=False)`` can return a view on float32 source.
            t = crop[zi].astype(np.float32, copy=True)
            t -= f_in_min
            t /= f_span
            np.clip(t, 0.0, 1.0, out=t)
            np.round(t * f_scale, out=t)
            out[zi] = t.astype(out_dtype_np, copy=False)
    else:
        # One XY plane does not fit the float32 budget — tile within each Z slice.
        max_elems = max(4096, budget // 4)
        ty = max(64, int(np.sqrt(max_elems)))
        ty = min(y, ty)
        tx = max(64, min(x, max_elems // max(ty, 1)))
        buf = np.empty((ty, tx), dtype=np.float32)
        for zi in range(z):
            for y0 in range(0, y, ty):
                y1 = min(y, y0 + ty)
                for x0 in range(0, x, tx):
                    x1 = min(x, x0 + tx)
                    hh, ww = y1 - y0, x1 - x0
                    t = buf[:hh, :ww]
                    np.copyto(
                        t, crop[zi, y0:y1, x0:x1].astype(np.float32, copy=True)
                    )
                    t -= f_in_min
                    t /= f_span
                    np.clip(t, 0.0, 1.0, out=t)
                    np.round(t * f_scale, out=t)
                    out[zi, y0:y1, x0:x1] = t.astype(out_dtype_np, copy=False)

    post: Dict[str, Any] = {
        "input_min": in_min,
        "input_max": in_max,
        "output_dtype": out_dtype,
        "stretch_path": "zyx_slabwise_float32",
    }
    if out_dtype == "uint16":
        post["output_range"] = [0, 65535]
    else:
        post["output_range"] = [0, 255]
    return out, post


def stretch_contrast(
    crop: np.ndarray,
    in_min: float,
    in_max: float,
    out_dtype: str = "uint8",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Map intensities to ``uint8`` / ``uint16`` using ``[in_min, in_max]`` → full range.

    Large **(Z, Y, X)** volumes use a **slab-wise float32** path so a full float64/float32
    copy of the volume is never allocated (that caused multi-hundred GiB peaks when stretching).
    """
    crop = np.asarray(crop)
    if (
        crop.ndim == 3
        and crop.size > 262144
        and int(crop.nbytes) >= _STRETCH_CHUNK_ENTRY_BYTES
    ):
        return _stretch_contrast_zyx_slabwise(crop, in_min, in_max, out_dtype)
    return _stretch_contrast_dense_float32(crop, in_min, in_max, out_dtype)

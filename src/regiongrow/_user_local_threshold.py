"""Local (neighborhood-based) threshold maps for the user fill algorithm."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, uniform_filter
from skimage.morphology import disk
from skimage.segmentation import (
    inverse_gaussian_gradient,
    morphological_geodesic_active_contour,
)


def _local_mean_std_uniform(
    image: np.ndarray, size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std with an isotropic uniform window (voxels, odd side length)."""
    im = np.asarray(image, dtype=np.float64)
    mean = uniform_filter(im, size=size, mode="nearest")
    mean_sq = uniform_filter(im * im, size=size, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean, std


def _local_mean_std_gaussian(
    image: np.ndarray, sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted local mean and std using Gaussian smoothing (σ in voxels, isotropic)."""
    im = np.asarray(image, dtype=np.float64)
    mean = gaussian_filter(im, sigma=sigma, mode="nearest")
    mean_sq = gaussian_filter(im * im, sigma=sigma, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean, std


def _odd_window(w: int) -> int:
    w = max(3, int(w))
    return w if w % 2 == 1 else w + 1


def local_threshold_mask(
    image: np.ndarray,
    method: str,
    *,
    window: int = 15,
    k: float = 0.35,
    sauvola_r: float = 0.0,
    gaussian_sigma: float = 4.0,
    domain_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a binary mask from local intensity statistics.

    Parameters
    ----------
    image
        3-D scalar array (Z, Y, X), any dtype coerced to float64.
    method
        One of:
        - ``"Uniform mean + k·σ (bright)"`` — foreground where ``I ≥ μ + k σ`` (inclusive so
          saturated plateaus, e.g. ``I = 255`` with ``μ = 255``, ``σ = 0``, still pass).
        - ``"Uniform mean − k·σ (dark)"`` — foreground where ``I ≤ μ − k σ`` (inclusive at 0).
        - ``"Sauvola (local contrast-scaled)"`` — ``I ≥ μ (1 + k (σ/R − 1))`` with ``R`` from
          ``sauvola_r`` if positive, else global peak-to-peak of finite values.
        - ``"Gaussian mean + k·σ (bright)"`` — like the first, but μ, σ from Gaussian-weighted locals.
    window
        Isotropic uniform window side length (voxels); forced odd, minimum 3.
        Used by all *Uniform* / *Sauvola* methods.
    k
        Sensitivity; typical bright defaults are ``0.2–0.6`` for σ-based rules.
    sauvola_r
        Dynamic-range scale ``R`` for Sauvola. Non-positive selects ``max(finite) − min(finite)``.
    gaussian_sigma
        Standard deviation (voxels) for the Gaussian *Gaussian mean + k·σ* variant.
    domain_mask
        If given, local statistics use an image where voxels outside the mask are
        replaced by the median intensity **inside** the mask (so filters are not
        driven by distant tissue). The returned mask is always restricted to this
        domain (and finite voxels).

    Returns
    -------
    mask
        Boolean array, same shape as *image*.
    """
    # float64 preserves uint8/uint16 dynamic range; no clipping here.
    im_orig = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(im_orig)
    if not np.any(finite):
        return np.zeros(im_orig.shape, dtype=bool)

    # Intensity actually thresholded (never the temporary fill values).
    im_obs = np.where(finite, im_orig, 0.0)

    roi = None
    if domain_mask is not None:
        roi = np.asarray(domain_mask, dtype=bool)
        if roi.shape != im_orig.shape:
            raise ValueError(
                f"domain_mask shape {roi.shape} != image shape {im_orig.shape}"
            )
        inside = roi & finite
        if not np.any(inside):
            return np.zeros(im_orig.shape, dtype=bool)
        fill = float(np.median(im_obs[inside]))
        im_for_filter = np.where(roi & finite, im_obs, fill)
    else:
        im_for_filter = im_obs

    def _sauvola_r_scale() -> float:
        if sauvola_r > 0:
            return float(sauvola_r)
        if roi is not None:
            vals = im_obs[roi & finite]
        else:
            vals = im_obs[finite]
        return float(np.ptp(vals)) if vals.size else 1.0

    if method == "Uniform mean + k·σ (bright)":
        size = _odd_window(window)
        mean, std = _local_mean_std_uniform(im_for_filter, size)
        thr = mean + float(k) * std
        # >= : strict > misses saturated maxima (e.g. 255) when μ+kσ rounds to 255.
        out = (im_orig >= thr) & finite
        if roi is not None:
            out &= roi
        return out

    if method == "Uniform mean − k·σ (dark)":
        size = _odd_window(window)
        mean, std = _local_mean_std_uniform(im_for_filter, size)
        thr = mean - float(k) * std
        # <= : symmetric to bright rule at 0 saturation (μ−kσ == 0).
        out = (im_orig <= thr) & finite
        if roi is not None:
            out &= roi
        return out

    if method == "Sauvola (local contrast-scaled)":
        size = _odd_window(window)
        mean, std = _local_mean_std_uniform(im_for_filter, size)
        r = _sauvola_r_scale()
        r = max(r, 1e-6)
        thr = mean * (1.0 + float(k) * (std / r - 1.0))
        out = (im_orig >= thr) & finite
        if roi is not None:
            out &= roi
        return out

    if method == "Gaussian mean + k·σ (bright)":
        sig = max(float(gaussian_sigma), 0.5)
        mean, std = _local_mean_std_gaussian(im_for_filter, sig)
        thr = mean + float(k) * std
        out = (im_orig >= thr) & finite
        if roi is not None:
            out &= roi
        return out

    raise ValueError(f"Unknown local threshold method: {method!r}")


def close_ring_slicewise(
    wall_mask: np.ndarray,
    seed_tube: np.ndarray,
    corridor: np.ndarray,
    axis: int,
    closing_radius_vox: float,
    *,
    image: Optional[np.ndarray] = None,
    morph_gac_iters: int = 80,
    morph_gac_balloon: float = 0.5,
    morph_gac_smoothing: int = 1,
    igg_alpha: float = 100.0,
    igg_sigma: float = 1.5,
    morph_gac_edge_threshold: float = 0.3,
) -> np.ndarray:
    """Fill incomplete vessel-wall rings on each 2-D slice along *axis*.

    For each slice perpendicular to *axis* (ZYX volume: axis 0 → YX planes, etc.):

    1. Morphologically close the wall mask, union seed-tube cross-section → ``U0``.
    2. If *image* is given (recommended): run **morphological geodesic active contour**
       (``skimage.segmentation.morphological_geodesic_active_contour``) on an
       **inverse Gaussian gradient** edge image of the slice, initialized from ``U0``.
       A small **balloon** expands the contour across low-gradient gaps while strong
       edges (vessel wall) slow evolution — better than ``binary_fill_holes`` alone
       when the lumen is still connected to the slice boundary through a gap.
    3. If *image* is omitted or MGAC fails on a slice: fall back to ``binary_fill_holes(U0)``.
    4. Clamp to the corridor mask on that slice.

    Parameters
    ----------
    wall_mask
        Binary threshold mask (fragmented ring), shape *(Z, Y, X)*.
    seed_tube
        Binary seed tube mask, same shape (branch polyline tube).
    corridor
        Binary threshold corridor (tube around centerline), same shape.
    axis
        Slice normal: ``0`` = Z (YX planes), ``1`` = Y (ZX planes), ``2`` = X (ZY planes).
    closing_radius_vox
        Radius (voxels) of the isotropic disk structuring element for 2-D closing before MGAC.
    image
        Grayscale volume (same shape), e.g. fluorescence at the working pyramid level.
    morph_gac_iters, morph_gac_balloon, morph_gac_smoothing
        Passed to :func:`morphological_geodesic_active_contour`.
    igg_alpha, igg_sigma
        Passed to :func:`inverse_gaussian_gradient` on the normalized slice.
    morph_gac_edge_threshold
        ``g`` values below this count as strong edges for MGAC. Values ``≤ 0`` use
        ``threshold=\"auto\"`` in skimage (often too conservative on normalized slices).

    Returns
    -------
    ndarray bool
        Same shape as inputs.
    """
    W = np.asarray(wall_mask, dtype=bool)
    C = np.asarray(seed_tube, dtype=bool)
    Co = np.asarray(corridor, dtype=bool)
    if W.shape != C.shape or W.shape != Co.shape:
        raise ValueError(
            f"Shape mismatch: wall {W.shape}, seed_tube {C.shape}, corridor {Co.shape}"
        )
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    img_vol = None
    if image is not None:
        img_vol = np.asarray(image, dtype=np.float64)
        if img_vol.shape != W.shape:
            raise ValueError(f"image shape {img_vol.shape} != mask shape {W.shape}")

    r = max(float(closing_radius_vox), 1e-6)
    footprint = disk(r)
    out = np.zeros(W.shape, dtype=bool)
    n = W.shape[axis]
    iters = max(1, int(morph_gac_iters))
    smooth = max(1, int(morph_gac_smoothing))
    bal = float(morph_gac_balloon)
    alpha = float(igg_alpha)
    sig = max(float(igg_sigma), 1e-3)
    thr_edge = float(morph_gac_edge_threshold)
    thr_arg: float | str = "auto" if thr_edge <= 0.0 else thr_edge

    for i in range(n):
        idx = [slice(None), slice(None), slice(None)]
        idx[axis] = i
        t_idx = tuple(idx)
        Ws = W[t_idx]
        Cs = C[t_idx]
        Cos = Co[t_idx]
        W_closed = binary_closing(Ws, structure=footprint)
        U0 = W_closed | Cs
        filled_fallback = binary_fill_holes(U0) & Cos

        if img_vol is None:
            out[t_idx] = filled_fallback
            continue

        img_s = img_vol[t_idx]
        finite = np.isfinite(img_s)
        if not np.any(finite):
            out[t_idx] = filled_fallback
            continue
        vals = img_s[finite]
        mn, mx = float(np.min(vals)), float(np.max(vals))
        if mx <= mn + 1e-9:
            out[t_idx] = filled_fallback
            continue
        img_n = np.zeros_like(img_s, dtype=np.float64)
        img_n[finite] = (img_s[finite] - mn) / (mx - mn)

        try:
            g = inverse_gaussian_gradient(img_n, alpha=alpha, sigma=sig)
            ls = morphological_geodesic_active_contour(
                g,
                num_iter=iters,
                init_level_set=U0.astype(np.float64),
                smoothing=smooth,
                balloon=bal,
                threshold=thr_arg,
            )
            seg = ls.astype(bool)
            out[t_idx] = binary_fill_holes(seg) & Cos
        except Exception:
            out[t_idx] = filled_fallback
    return out

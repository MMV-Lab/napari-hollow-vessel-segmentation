"""ROI cropping for branch Grow: local subvolume around the polyline seed."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

SliceTriple = Tuple[slice, slice, slice]


def _normalize_spacing(spacing, shape) -> np.ndarray:
    if spacing is None:
        return np.ones(3, dtype=np.float64)
    s = np.asarray(spacing, dtype=np.float64).ravel()
    if s.size == 1:
        s = np.broadcast_to(s, (3,))
    if s.size < 3:
        s = np.pad(s, (0, 3 - s.size), constant_values=1.0)
    s = s[-3:].copy()
    s[s <= 0] = 1.0
    return s


def grow_roi_slices_zyx(
    shape: Sequence[int],
    poly_zyx: np.ndarray,
    spacing,
    tube_radius_vox: float,
    margin_voxels: float = 0.0,
    *,
    pad_z_tube_factor: float = 2.0,
    pad_xy_tube_factor: float = 1.5,
    edge_sigma_phys: float = 0.0,
) -> SliceTriple:
    """Bounding box around the branch polyline with generous physical padding.

    Padding beyond the polyline extent uses ``pad_z_tube_factor`` and
    ``pad_xy_tube_factor`` times the seed tube radius (physical units), plus
    the MGAC corridor margin convention (``margin_voxels * mean(spacing)``).
    An optional ``edge_sigma_phys`` halo (≈3× σ) keeps Gaussian edge filtering
    inside the ROI.
    """
    sz, sy, sx = (int(shape[0]), int(shape[1]), int(shape[2]))
    poly = np.asarray(poly_zyx, dtype=np.int64).reshape(-1, 3)
    if poly.shape[0] == 0:
        raise ValueError("empty polyline: at least one branch point is required")

    # Reject polylines that fall entirely outside the working grid; this would
    # otherwise produce an inverted/empty slice (e.g. slice(90, 50)) downstream.
    z0r, y0r, x0r = (int(poly[:, 0].min()), int(poly[:, 1].min()), int(poly[:, 2].min()))
    z1r, y1r, x1r = (int(poly[:, 0].max()), int(poly[:, 1].max()), int(poly[:, 2].max()))
    if (
        z1r < 0 or z0r >= sz
        or y1r < 0 or y0r >= sy
        or x1r < 0 or x0r >= sx
    ):
        raise ValueError(
            "branch points fall outside the working volume "
            f"(shape {(sz, sy, sx)}); re-place the points on the image grid."
        )

    spacing = _normalize_spacing(spacing, shape)
    min_s = float(np.min(spacing))
    mean_s = float(np.mean(spacing))
    tube_phys = float(tube_radius_vox) * min_s
    margin_phys = float(margin_voxels) * mean_s

    pad_z_phys = float(pad_z_tube_factor) * tube_phys + margin_phys
    pad_y_phys = float(pad_xy_tube_factor) * tube_phys + margin_phys
    pad_x_phys = float(pad_xy_tube_factor) * tube_phys + margin_phys

    if edge_sigma_phys > 0:
        halo = 3.0 * float(edge_sigma_phys)
        pad_z_phys += halo
        pad_y_phys += halo
        pad_x_phys += halo

    vz = max(1, int(np.ceil(pad_z_phys / spacing[0])))
    vy = max(1, int(np.ceil(pad_y_phys / spacing[1])))
    vx = max(1, int(np.ceil(pad_x_phys / spacing[2])))

    # Clamp the bounding box into the volume so partially out-of-range points
    # still yield a valid, non-empty ROI.
    z0, z1 = int(np.clip(z0r, 0, sz - 1)), int(np.clip(z1r, 0, sz - 1))
    y0, y1 = int(np.clip(y0r, 0, sy - 1)), int(np.clip(y1r, 0, sy - 1))
    x0, x1 = int(np.clip(x0r, 0, sx - 1)), int(np.clip(x1r, 0, sx - 1))
    zs = max(0, z0 - vz)
    ze = min(sz, z1 + vz + 1)
    ys = max(0, y0 - vy)
    ye = min(sy, y1 + vy + 1)
    xs = max(0, x0 - vx)
    xe = min(sx, x1 + vx + 1)
    return (slice(zs, ze), slice(ys, ye), slice(xs, xe))


def roi_shape_from_slices(slices: SliceTriple) -> Tuple[int, int, int]:
    return (
        slices[0].stop - slices[0].start,
        slices[1].stop - slices[1].start,
        slices[2].stop - slices[2].start,
    )


def polyline_to_roi_local(poly_zyx: np.ndarray, roi_slices: SliceTriple) -> np.ndarray:
    """Shift full-grid ZYX indices into ROI-local coordinates."""
    poly = np.asarray(poly_zyx, dtype=np.int64).reshape(-1, 3)
    origin = np.array(
        [roi_slices[0].start, roi_slices[1].start, roi_slices[2].start],
        dtype=np.int64,
    )
    return poly - origin.reshape(1, 3)


def crop_bool_mask_to_roi(mask: np.ndarray, roi_slices: SliceTriple) -> np.ndarray:
    return np.asarray(mask, dtype=bool)[roi_slices]


def paste_roi_mask_into_full(
    full_shape: Sequence[int],
    roi_slices: SliceTriple,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """Embed an ROI boolean mask into a full-volume array."""
    out = np.zeros(tuple(int(x) for x in full_shape), dtype=bool)
    out[roi_slices] = np.asarray(roi_mask, dtype=bool)
    return out

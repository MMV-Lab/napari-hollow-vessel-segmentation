"""Priority-queue region growing with gradient-flux-weighted cost.

Combines three literature-based stopping criteria for robust vessel
boundary detection:

1. **Edge-weighted accumulated cost** (Dijkstra-style min-heap front)
   Each voxel has a base traversal cost = 1 / (eps + g(x)) where
   g(x) = exp(-beta |grad I|^2 / kappa^2) is the edge indicator.
   Growth proceeds cheapest-first via a min-heap; accumulated cost
   rises steeply at edges, giving a natural stop.

2. **Gradient flux as soft cost modifier** (inspired by Vasilevskiy &
   Siddiqi 2002)
   flux = grad I . n_outward.   Negative flux means intensity
   decreases outward (wall->background).  Instead of a hard reject, the
   base cost is *multiplied* by (1 + w * max(0, -cos theta)^2).
   This allows growth through wrinkled surfaces where flux is
   transiently negative, while strongly penalising sustained boundary
   crossings where cos theta ~ -1.

3. **Adaptive region statistics** (Confidence Connected / ITK;
   Pohle & Toennies 2001)
   Running mean mu and std sigma maintained with Welford's algorithm.
   Candidates far below the region mean are rejected.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    generate_binary_structure,
    binary_dilation,
)
from skimage.filters import threshold_otsu, threshold_triangle, threshold_li


# ──────────────────────────── helpers ────────────────────────────────────── #

_OFFSETS_6 = np.array(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
    dtype=np.int32,
)


def _normalize_spacing(spacing, shape):
    """Return ``(sz, sy, sx)`` as float64, defaulting to isotropic 1 if missing."""
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


def _step_distance(offset, spacing):
    """Physical edge length for one 6-neighbour step *offset* (±1 on one axis)."""
    o = np.asarray(offset, dtype=np.int32)
    if int(np.sum(np.abs(o))) != 1:
        raise ValueError("only 6-connected offsets are supported")
    if o[0] != 0:
        return float(spacing[0])
    if o[1] != 0:
        return float(spacing[1])
    return float(spacing[2])


def _precompute_edge_map(image, sigma, spacing):
    """Return edge indicator *g*, physical gradient, magnitude, and *kappa*.

    Gaussian smoothing uses *sigma* interpreted as an **isotropic physical**
    standard deviation in the same units as *spacing* (converted to voxel
    sigmas per axis).  Gradients use ``numpy.gradient`` with *spacing* so
    shallow intensity changes across thick Z-slices are not mistaken for
    strong edges.
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    sigma = float(sigma)
    sigma_vox = tuple(float(sigma) / max(spacing[i], 1e-12) for i in range(3))
    smoothed = gaussian_filter(image, sigma=sigma_vox)
    grad = np.array(
        np.gradient(smoothed, spacing[0], spacing[1], spacing[2])
    )  # (3, Z, Y, X), physical derivatives
    grad_mag = np.sqrt(np.sum(grad ** 2, axis=0))

    nonzero = grad_mag[grad_mag > 0]
    kappa = float(np.percentile(nonzero, 90)) if nonzero.size > 0 else 1.0
    kappa = max(kappa, 1e-10)

    g = 1.0 / (1.0 + (grad_mag / kappa) ** 2)
    return g, grad, grad_mag, kappa


def _seed_boundary(seed_mask):
    """6-connected dilation of *seed_mask* minus the seed itself."""
    struct = generate_binary_structure(3, 1)
    dilated = binary_dilation(seed_mask, structure=struct)
    return np.where(dilated & ~seed_mask)


def _init_heap_from_seed_boundary(
    local_cost,
    seed_mask,
    spacing,
    shape,
    acc_cost,
    heap,
    forbidden=None,
):
    """Push seed-front voxels with physically weighted costs."""
    spacing = np.asarray(spacing, dtype=np.float64)
    bz, by, bx = _seed_boundary(seed_mask)
    for i in range(len(bz)):
        z, y, x = int(bz[i]), int(by[i]), int(bx[i])
        if forbidden is not None and forbidden[z, y, x]:
            continue
        best = np.inf
        for off in _OFFSETS_6:
            nz = z - int(off[0])
            ny = y - int(off[1])
            nx = x - int(off[2])
            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                if seed_mask[nz, ny, nx]:
                    step = _step_distance(off, spacing)
                    c = float(local_cost[z, y, x]) * step
                    if c < best:
                        best = c
        acc_cost[z, y, x] = best
        heapq.heappush(heap, (best, z, y, x))


def compute_upper_threshold(image, method):
    """Compute an upper intensity hard-stop threshold.

    Parameters
    ----------
    image : ndarray
        3-D image (any dtype; converted to float64 internally).
    method : str
        One of ``'otsu'``, ``'triangle'``, ``'li'``, ``'p90'``, ``'p95'``.

    Returns
    -------
    float
        Threshold value above which voxels are rejected during growing.
    """
    arr = np.asarray(image, dtype=np.float64)
    if method not in {"otsu", "triangle", "li", "p90", "p95"}:
        raise ValueError(f"Unknown threshold method: {method!r}")
    try:
        if method == "otsu":
            return float(threshold_otsu(arr))
        if method == "triangle":
            return float(threshold_triangle(arr))
        if method == "li":
            return float(threshold_li(arr))
        if method == "p90":
            return float(np.percentile(arr, 90))
        return float(np.percentile(arr, 95))
    except (ValueError, RuntimeError, IndexError):
        # Histogram methods fail on constant images; p95 is always computable.
        return float(np.percentile(arr, 95))


# ──────────────────────── main entry point ──────────────────────────────── #

def region_grow(
    image,
    seed_mask,
    start_point,
    end_point,
    # ── edge / speed ──
    sigma=2.0,
    spacing=None,
    # ── stopping gates ──
    cost_budget=None,           # max accumulated cost (auto if None)
    flux_weight=15.0,           # soft flux penalty weight
    intensity_tolerance=3.0,    # reject if intensity < mu - N*sigma
    upper_threshold=None,        # hard stop: reject if intensity > this
    # ── geometry ──
    margin=5.0,
    # ── visualisation ──
    yield_every=500,
    # ── optional: Welford stats from this mask instead of full seed_mask ──
    stats_seed_mask=None,
    forbidden_mask=None,
):
    """Priority-queue region growing for 3-D vessel segmentation.

    Parameters
    ----------
    image : ndarray (Z, Y, X)
        3-D fluorescence image.
    seed_mask : bool ndarray
        Initial region (user brush stroke along vessel centre).
    start_point, end_point : array-like (z, y, x)
        Vessel start and end coordinates for length constraint.
    sigma : float
        Isotropic Gaussian smoothing **physical** standard deviation (same
        units as *spacing*).  Converted to per-axis voxel sigmas internally.
    spacing : sequence of 3 floats, optional
        Voxel spacing ``(s_z, s_y, s_x)`` matching *image* axes.  When omitted,
        behaviour matches the former isotropic ``(1, 1, 1)`` implementation.
    cost_budget : float or None
        Maximum accumulated growth cost a voxel may have to be accepted.
        If *None*, auto-calibrated from the image edge indicator.
        Values ``> 0`` are scaled by the mean spacing so manual budgets remain
        comparable when switching between isotropic and anisotropic grids.
    flux_weight : float
        Soft flux penalty weight.  When the normalised gradient flux
        (cos theta) is negative, the local traversal cost is multiplied
        by  ``1 + flux_weight * max(0, -cos_theta)^2``.
        Higher -> stronger penalty at boundary crossings but still allows
        growth through transiently negative-flux wrinkle regions.
    intensity_tolerance : float
        Reject a candidate whose intensity is more than this many standard
        deviations below the adaptive region mean.
    margin : float
        Extra leeway along the vessel axis in **physical** length, computed as
        ``margin * mean(spacing)`` (the spin box is still labelled as a voxel
        margin; this scales it to world units consistently with anisotropic
        grids).
    yield_every : int
        Yield (step, mask) every *N* accepted voxels for animation.
    stats_seed_mask : bool ndarray, optional
        If given, initial mean/variance for the intensity gate are taken from
        ``image[stats_seed_mask]`` instead of ``image[seed_mask]``.  Use a thin
        A* skeleton (excluding a thick overlap with an existing segmentation)
        so the adaptive gate matches the branch lumen.
    forbidden_mask : bool ndarray, optional
        Same shape as *image*. True voxels are never grown into (walls); seed
        voxels overlapping forbidden are cleared before propagation.

    Yields
    ------
    (int, ndarray) -- step counter and current boolean mask.
    """

    image = np.asarray(image, dtype=np.float64)
    seed_mask = np.asarray(seed_mask, dtype=bool)
    shape = image.shape
    if seed_mask.shape != shape:
        raise ValueError(
            f"seed_mask shape {seed_mask.shape} != image shape {shape}"
        )
    if not np.all(np.isfinite(image)):
        raise ValueError(
            "image contains NaN/Inf; region growing requires finite values "
            "(check the loaded pyramid level / contrast)."
        )
    spacing = _normalize_spacing(spacing, shape)
    mean_s = float(np.mean(spacing))

    forbidden = None
    if forbidden_mask is not None:
        forbidden = np.asarray(forbidden_mask, dtype=bool)
        if forbidden.shape != shape:
            raise ValueError(
                f"forbidden_mask shape {forbidden.shape} != image shape {shape}"
            )
        seed_mask = seed_mask & ~forbidden

    if not np.any(seed_mask):
        raise ValueError(
            "seed_mask is empty (no seed voxels to grow from). Increase the "
            "tube radius, check point placement, or verify the blocker mask "
            "is not covering the seed."
        )

    # ── 1. Pre-compute edge indicator & gradient vector field ───────────
    g, grad, grad_mag, kappa = _precompute_edge_map(image, sigma, spacing)

    epsilon = 0.01
    local_cost = 1.0 / (epsilon + g)        # high at edges, ~1 in smooth
    del g  # edge indicator no longer needed; free a full-volume float array

    # ── 2. Auto-calibrate cost budget (physical path length) ───────────
    if cost_budget is None:
        p95 = float(np.percentile(local_cost, 95))
        p50 = float(np.median(local_cost))
        cost_budget = (p95 * 30 + p50 * 50) * mean_s
    else:
        cost_budget = float(cost_budget) * mean_s

    # ── 3. Axis constraint (physical coordinates) ───────────────────────
    start = np.asarray(start_point, dtype=np.float64)
    end = np.asarray(end_point, dtype=np.float64)
    axis_idx = end - start
    axis_phys = axis_idx * spacing
    axis_len_phys = float(np.linalg.norm(axis_phys))
    if axis_len_phys <= float(np.min(spacing)):
        raise ValueError(
            "start_point and end_point coincide (or are within one voxel); the "
            "vessel-axis length constraint cannot be defined. Place at least two "
            "distinct branch points."
        )
    axis_dir_phys = axis_phys / axis_len_phys
    margin_phys = float(margin) * mean_s

    # ── 4. Region mask & Welford running statistics ─────────────────────
    region = seed_mask.copy()
    if stats_seed_mask is not None:
        stat_mask = np.asarray(stats_seed_mask, dtype=bool)
        if stat_mask.shape != shape:
            raise ValueError(
                f"stats_seed_mask shape {stat_mask.shape} != image shape {shape}"
            )
        if not np.any(stat_mask):
            # Empty stats skeleton: fall back to the full seed for statistics.
            stat_mask = seed_mask
    else:
        stat_mask = seed_mask
    seed_vals = image[stat_mask]
    n_acc = int(seed_vals.size)
    mu = float(np.mean(seed_vals))
    m2 = float(np.sum((seed_vals - mu) ** 2))

    # ── 5. Accumulated-cost map & min-heap ──────────────────────────────
    acc_cost = np.full(shape, np.inf)
    acc_cost[seed_mask] = 0.0

    heap: list = []
    _init_heap_from_seed_boundary(
        local_cost, seed_mask, spacing, shape, acc_cost, heap, forbidden
    )

    # ── 6. Yield initial state ──────────────────────────────────────────
    yield 0, region.copy()
    step = 0
    last_yielded = 0

    # ── 7. Main loop ─────────────────────────────────────────────
    while heap:
        cost_val, z, y, x = heapq.heappop(heap)

        if region[z, y, x]:
            continue
        if cost_val > acc_cost[z, y, x]:
            continue

        if forbidden is not None and forbidden[z, y, x]:
            continue

        # gate A: accumulated cost budget
        if cost_val > cost_budget:
            break

        step += 1

        # gate B: length constraint (physical projection)
        if axis_dir_phys is not None:
            r_phys = (np.array([z, y, x], dtype=np.float64) - start) * spacing
            proj = float(np.dot(r_phys, axis_dir_phys))
            if proj < -margin_phys or proj > axis_len_phys + margin_phys:
                continue

        # gate C: adaptive intensity
        std = np.sqrt(m2 / max(n_acc, 1)) if n_acc > 1 else 0.0
        val = image[z, y, x]
        if std > 0 and val < mu - intensity_tolerance * std:
            continue

        # gate D: upper threshold hard stop
        if upper_threshold is not None and val > upper_threshold:
            continue

        # ── ACCEPT ──
        region[z, y, x] = True

        # Welford update
        n_acc += 1
        delta = val - mu
        mu += delta / n_acc
        delta2 = val - mu
        m2 += delta * delta2

        # ── Compute flux-weighted cost for neighbours ──
        # Outward normal from segmented-neighbour centroid (physical space)
        cz_s, cy_s, cx_s, n_seg = 0.0, 0.0, 0.0, 0
        for off in _OFFSETS_6:
            nz, ny, nx = z + off[0], y + off[1], x + off[2]
            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                if region[nz, ny, nx]:
                    cz_s += nz
                    cy_s += ny
                    cx_s += nx
                    n_seg += 1

        for off in _OFFSETS_6:
            nz = z + int(off[0])
            ny = y + int(off[1])
            nx = x + int(off[2])
            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                if not region[nz, ny, nx]:
                    if forbidden is not None and forbidden[nz, ny, nx]:
                        continue
                    step_len = _step_distance(off, spacing)
                    base_inc = float(local_cost[nz, ny, nx]) * step_len

                    # Flux-based soft cost multiplier (physical outward direction)
                    flux_mult = 1.0
                    if n_seg > 0:
                        dz_i = nz - cz_s / n_seg
                        dy_i = ny - cy_s / n_seg
                        dx_i = nx - cx_s / n_seg
                        dz_p = dz_i * spacing[0]
                        dy_p = dy_i * spacing[1]
                        dx_p = dx_i * spacing[2]
                        norm = float(np.sqrt(dz_p * dz_p + dy_p * dy_p + dx_p * dx_p))
                        if norm > 1e-10:
                            uz = dz_p / norm
                            uy = dy_p / norm
                            ux = dx_p / norm
                            gz = float(grad[0, nz, ny, nx])
                            gy = float(grad[1, nz, ny, nx])
                            gx = float(grad[2, nz, ny, nx])
                            gm = float(grad_mag[nz, ny, nx])
                            if gm > 1e-10:
                                cos_theta = (gz * uz + gy * uy + gx * ux) / gm
                                if cos_theta < 0:
                                    flux_mult = (
                                        1.0 + flux_weight * cos_theta * cos_theta
                                    )

                    nc = cost_val + base_inc * flux_mult
                    if nc < acc_cost[nz, ny, nx]:
                        acc_cost[nz, ny, nx] = nc
                        heapq.heappush(heap, (nc, nz, ny, nx))

        # Yield for animation
        if step % yield_every == 0:
            yield step, region.copy()
            last_yielded = step

    # Always yield final state
    if step != last_yielded:
        yield step, region.copy()


# ──────────────────────── branch polyline tube (no A*) ─────────────────── #


def polyline_to_line_mask(shape: tuple, poly_zyx: np.ndarray) -> np.ndarray:
    """One-voxel-thick polyline through ordered ``(z,y,x)`` integer knots."""
    try:
        from skimage.draw import line_nd as _line_nd
    except ImportError:
        _line_nd = None

    poly = np.asarray(poly_zyx, dtype=np.int64).reshape(-1, 3)
    mask = np.zeros(shape, dtype=bool)
    if poly.shape[0] == 0:
        return mask
    for i in range(poly.shape[0] - 1):
        a = np.asarray(poly[i], dtype=np.int64).ravel()[:3]
        b = np.asarray(poly[i + 1], dtype=np.int64).ravel()[:3]
        if _line_nd is not None:
            try:
                idx = _line_nd(a, b, endpoint=True)
            except TypeError:
                idx = _line_nd(a, b)
            # ``line_nd`` can return coordinates outside the volume on some paths;
            # bulk ``mask[idx] = True`` then raises and skips the segment.  Paint
            # in-bounds voxels only so face/corner polylines still connect.
            for j in range(len(idx[0])):
                vz, vy, vx = int(idx[0][j]), int(idx[1][j]), int(idx[2][j])
                if (
                    0 <= vz < shape[0]
                    and 0 <= vy < shape[1]
                    and 0 <= vx < shape[2]
                ):
                    mask[vz, vy, vx] = True
        else:
            n = int(np.max(np.abs(b - a))) + 1
            for t in np.linspace(0.0, 1.0, max(n, 2)):
                vz, vy, vx = np.round(a + t * (b - a)).astype(np.int64)
                if (
                    0 <= vz < shape[0]
                    and 0 <= vy < shape[1]
                    and 0 <= vx < shape[2]
                ):
                    mask[vz, vy, vx] = True
    # Guarantee every user knot lies on the centerline (faces / last slice).
    for k in range(poly.shape[0]):
        z, y, x = int(poly[k, 0]), int(poly[k, 1]), int(poly[k, 2])
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            mask[z, y, x] = True
    return mask


def polyline_tube_mask(
    shape: tuple,
    poly_zyx: np.ndarray,
    radius_vox: float,
    spacing,
) -> np.ndarray:
    """Physical tube around the polyline (same radius convention as MGAC EDT tube).

    Distance transform is computed on the full ``shape`` volume; use a coarser
    OME-Zarr / napari pyramid level when memory or runtime is limiting.
    """
    from scipy.ndimage import distance_transform_edt

    spacing = _normalize_spacing(spacing, shape)
    poly = np.asarray(poly_zyx, dtype=np.int64).reshape(-1, 3)
    if poly.shape[0] == 0:
        return np.zeros(shape, dtype=bool)
    radius_phys = float(radius_vox) * float(np.min(spacing))
    line = polyline_to_line_mask(shape, poly)
    if not np.any(line):
        return np.zeros(shape, dtype=bool)
    dist = distance_transform_edt(
        ~line, sampling=tuple(float(x) for x in spacing)
    )
    return (dist <= radius_phys).astype(bool)


def polyline_corridor_mask(
    shape: tuple,
    poly_zyx: np.ndarray,
    spacing,
    margin_voxels: float,
    tube_radius_voxels: float,
) -> np.ndarray:
    """Voxels within physical distance *R* of the rasterized polyline centerline.

    Branch MGAC used to clip with :func:`holvesseg._active_contour._length_mask`
    (cylinder between first and last knot only).  Curved branches then had most
    of the seed tube **outside** that cylinder, so ``init_level_set & lmask``
    was tiny or empty and the contour eroded away.  This mask follows the
    **entire** polyline; *R* combines the same ``margin`` convention as the
    chord mask (``margin_voxels * mean(spacing)``) with several times the seed
    tube radius so balloon expansion can fill the branch lumen.

    EDT uses the full ``shape`` volume; prefer a coarser pyramid level for large
    data instead of spatial cropping.
    """
    from scipy.ndimage import distance_transform_edt

    spacing = _normalize_spacing(spacing, shape)
    poly = np.asarray(poly_zyx, dtype=np.int64).reshape(-1, 3)
    if poly.shape[0] == 0:
        return np.zeros(shape, dtype=bool)
    mean_s = float(np.mean(spacing))
    min_s = float(np.min(spacing))
    margin_phys = float(margin_voxels) * mean_s
    tube_phys = float(tube_radius_voxels) * min_s
    r_phys = margin_phys + max(5.0 * tube_phys, tube_phys + 2.5 * margin_phys)
    line = polyline_to_line_mask(shape, poly)
    if not np.any(line):
        return np.zeros(shape, dtype=bool)
    dist = distance_transform_edt(
        ~line.astype(bool), sampling=tuple(float(x) for x in spacing)
    )
    return (dist <= r_phys).astype(bool)

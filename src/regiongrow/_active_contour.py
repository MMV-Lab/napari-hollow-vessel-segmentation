"""3-D morphological geodesic active contour for vessel segmentation.

Uses the Morphological Geodesic Active Contour (MGAC) algorithm
(Márquez-Neila et al. 2014):

  - A tube is initialised around the user-drawn seed centerline using
    the Euclidean distance transform with voxel **spacing** so the tube
    is round in physical space.
  - The tube is evolved toward the vessel boundary via morphological
    dilation/erosion guided by an inverse-Gaussian-gradient edge image
    built with **physical** gradients (same convention as plain growing).
  - Balloon dilation uses a **structuring element that is a discrete
    physical ball** (not a voxel cube) so inflation is approximately
    isotropic in world units despite anisotropic voxels.
  - A length constraint clips the evolving mask to the start→end axis in
    **physical** coordinates.

The MGAC inner loop follows ``skimage.segmentation.morphsnakes`` but
replaces isotropic ``np.ones((3,)*ndim)`` morphology with spacing-aware
operators and physical ``np.gradient`` of the edge image.

Reference:
  P. Márquez-Neila, L. Baumela, and L. Álvarez,
  "A morphological approach to curvature-based evolution of curves and
  surfaces", IEEE TPAMI 2014.
"""

from itertools import cycle
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, generate_binary_structure, binary_dilation
from skimage.segmentation import morphsnakes as _msm


# 3×3×3 plane structuring elements (same as skimage.segmentation.morphsnakes._P3).
_MGAC_P3 = [np.zeros((3, 3, 3), dtype=np.int8) for _ in range(9)]
_MGAC_P3[0][:, :, 1] = 1
_MGAC_P3[1][:, 1, :] = 1
_MGAC_P3[2][1, :, :] = 1
_MGAC_P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
_MGAC_P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
_MGAC_P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
_MGAC_P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
_MGAC_P3[7][[0, 1, 2], [0, 1, 2], :] = 1
_MGAC_P3[8][[0, 1, 2], [2, 1, 0], :] = 1


def _sup_inf_edgesafe(u: np.ndarray) -> np.ndarray:
    """SI operator with boundary-safe erosions (see :func:`_curvop_edgesafe`)."""
    erosions = []
    for P_i in _MGAC_P3:
        try:
            erosions.append(
                ndi.binary_erosion(u, P_i, border_value=1).astype(np.int8)
            )
        except TypeError:  # older scipy without border_value
            erosions.append(ndi.binary_erosion(u, P_i).astype(np.int8))
    return np.stack(erosions, axis=0).max(0)


def _inf_sup_edgesafe(u: np.ndarray) -> np.ndarray:
    """IS operator (dilations unchanged at borders)."""
    dilations = []
    for P_i in _MGAC_P3:
        dilations.append(ndi.binary_dilation(u, P_i).astype(np.int8))
    return np.stack(dilations, axis=0).min(0)


_curvop_edgesafe_fns = cycle(
    [
        lambda u: _sup_inf_edgesafe(_inf_sup_edgesafe(u)),
        lambda u: _inf_sup_edgesafe(_sup_inf_edgesafe(u)),
    ]
)


def _curvop_edgesafe(u: np.ndarray) -> np.ndarray:
    """Morphological curvature like ``morphsnakes._curvop`` without edge shrink.

    Default ``binary_erosion(..., border_value=0)`` treats voxels outside the
    volume as background, so repeated SI/IS curvature steps erode the level set
    several voxels in from every face. Using ``border_value=1`` for erosions
    only matches an absorbing “outside is foreground” boundary so the contour
    can reach the image extent.
    """
    return next(_curvop_edgesafe_fns)(u)


# ─────────────────────────── helpers ─────────────────────────────────────── #


def _normalize_spacing(spacing, shape):
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


def _inverse_gaussian_gradient_physical(image, alpha, sigma, spacing):
    """Like ``skimage.segmentation.inverse_gaussian_gradient`` with *spacing*."""
    spacing = np.asarray(spacing, dtype=np.float64)
    sigma = float(sigma)
    sigma_vox = tuple(sigma / max(spacing[i], 1e-12) for i in range(3))
    smoothed = ndi.gaussian_filter(image, sigma=sigma_vox, mode="nearest")
    grad = np.array(
        np.gradient(smoothed, spacing[0], spacing[1], spacing[2])
    )
    grad_mag = np.sqrt(np.sum(grad ** 2, axis=0))
    return 1.0 / np.sqrt(1.0 + alpha * grad_mag)


def _binary_closing_edgesafe(u: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Binary closing without the default border erosion shrink.

    ``ndi.binary_closing`` performs a dilation followed by an erosion.
    SciPy's erosion treats out-of-bounds as background by default, which
    systematically removes a few voxels at the volume border (exactly the
    artifact you observed: a 2–3 voxel margin plus small holes near the edge).
    We use ``border_value=1`` for the erosion step when supported, so the border
    behaves like absorbing foreground and the contour can expand flush to the
    volume boundary.
    """
    d = ndi.binary_dilation(u, structure=structure)
    try:
        return ndi.binary_erosion(d, structure=structure, border_value=1)
    except TypeError:  # older scipy without border_value
        # Best-effort fallback: accept default behavior.
        return ndi.binary_erosion(d, structure=structure)


def heal_mgac_binary_gaps(mask: np.ndarray, spacing) -> np.ndarray:
    """Close 1-voxel pinholes / checkerboard gaps from discrete MGAC updates.

    Morphological MGAC uses ``np.gradient`` on a binary mask; the edge update
    can leave a **striped** pattern of missing voxels.  One ``binary_closing``
    with the same physical ball as balloon dilation reconnects neighbours
    without the global shrink of ``_curvop`` each iteration.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    shape = m.shape
    spacing = _normalize_spacing(spacing, shape)
    struct = _physical_ball_structure(spacing)
    # Two passes are more robust against the "checkerboard / pinhole" pattern
    # produced by discrete MGAC edge updates (especially when smoothing=0).
    m = _binary_closing_edgesafe(m, structure=struct)
    m = _binary_closing_edgesafe(m, structure=struct)
    return np.asarray(m, dtype=bool)


def stabilize_mgac_mask(mask: np.ndarray, spacing) -> np.ndarray:
    """Stabilize MGAC masks by closing micro-gaps and filling enclosed cavities.

    MGAC updates can create transient pinholes / cavities that then persist or
    oscillate. A light physical-ball closing fixes 1-voxel stripes, and a 3-D
    hole fill enforces a solid lumen interior.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    m = heal_mgac_binary_gaps(m, spacing)
    # Fill enclosed voids (3-D); safe after corridor/blocker clipping.
    m = ndi.binary_fill_holes(m)
    # Also fill per-slice cavities (common in anisotropic volumes where a
    # cavity may be connected through a one-voxel tunnel in Z).
    if m.ndim == 3 and m.shape[0] > 1:
        m = np.stack([ndi.binary_fill_holes(m[z]) for z in range(m.shape[0])], axis=0)
    # Final micro-gap heal after hole fill (prevents thin "notches" remaining).
    m = heal_mgac_binary_gaps(m, spacing)
    return np.asarray(m, dtype=bool)


def _physical_ball_structure(spacing):
    """3-D structuring element: voxels whose centres lie inside a physical ball.

    Radius ``max(spacing)`` ensures all 6 face-adjacent voxel centres are
    included, while diagonals that are farther in **physical** space than
    that radius are excluded — reducing spurious Z leakage on anisotropic
    grids compared to a full ``3×3×3`` cube of ones.
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    sz, sy, sx = float(spacing[0]), float(spacing[1]), float(spacing[2])
    r = max(sz, sy, sx) * 1.0000001
    nz = int(np.ceil(r / sz)) + 1
    ny = int(np.ceil(r / sy)) + 1
    nx = int(np.ceil(r / sx)) + 1
    shape = (2 * nz + 1, 2 * ny + 1, 2 * nx + 1)
    cz, cy, cx = nz, ny, nx
    struct = np.zeros(shape, dtype=np.int8)
    for iz in range(shape[0]):
        for iy in range(shape[1]):
            for ix in range(shape[2]):
                dz = iz - cz
                dy = iy - cy
                dx = ix - cx
                dphys = np.sqrt((dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2)
                if dphys <= r:
                    struct[iz, iy, ix] = 1
    struct[cz, cy, cx] = 1
    return struct


def _init_tube(seed_mask, radius, spacing):
    """Tube where EDT(*spacing*) ≤ physical radius ``radius * min(spacing)``."""
    spacing = np.asarray(spacing, dtype=np.float64)
    dist = distance_transform_edt(~seed_mask.astype(bool), sampling=spacing)
    radius_phys = float(radius) * float(np.min(spacing))
    return dist <= radius_phys


def _length_mask(shape, start, end, margin, spacing):
    """Mask voxels whose centre lies within *margin* (physical) of the axis segment."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    axis_idx = end - start
    axis_phys = axis_idx * spacing
    axis_len = float(np.linalg.norm(axis_phys))
    if axis_len < 1e-10:
        return np.ones(shape, dtype=bool)
    axis_dir = axis_phys / axis_len
    mean_s = float(np.mean(spacing))
    margin_phys = float(margin) * mean_s

    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    coords = np.stack(
        [
            (zz - start[0]) * spacing[0],
            (yy - start[1]) * spacing[1],
            (xx - start[2]) * spacing[2],
        ],
        axis=-1,
    )
    proj = coords @ axis_dir
    mask = (proj >= -margin_phys) & (proj <= axis_len + margin_phys)
    return mask


def _smoothing_iterations(smoothing: int, spacing: np.ndarray) -> int:
    """Reduce morphological smoothing when Z and XY voxel pitches differ strongly."""
    smoothing = int(smoothing)
    if smoothing <= 0:
        return 0
    smoothing = max(1, smoothing)
    ratio = float(np.min(spacing)) / float(np.max(spacing))
    ratio = max(ratio, 0.2)
    return max(1, min(int(round(smoothing * ratio)), smoothing * 3))


def _morphological_geodesic_active_contour_aniso(
    gimage,
    num_iter,
    init_level_set,
    spacing,
    smoothing=1,
    threshold="auto",
    balloon=0,
    blocker_mask=None,
    *,
    apply_edge_term: bool = True,
    balloon_percentile: float = 40.0,
):
    """MGAC from ``morphsnakes`` with physical-ball morphology and physical ∇g.

    **Smoothing (``_curvop``):** skimage's morphological curvature operator runs
    ``smoothing``-scaled times **inside every outer iteration**, immediately after
    the balloon and edge updates — it is not a separate post-pass at the end.
    That repeated opening/closing behaviour **thins** narrow tubes and rounds
    corners, which is why ``smoothing=0`` often tracks vessel walls better for
    thin seeds while higher values look smoother but can eat the initialization.
    """
    image = gimage
    u = np.int8(init_level_set > 0)
    se_ball = _physical_ball_structure(spacing)
    blocker = (
        np.asarray(blocker_mask, dtype=bool) if blocker_mask is not None else None
    )

    if threshold == "auto":
        pct = float(np.clip(balloon_percentile, 1.0, 99.0))
        threshold = float(np.percentile(image, pct))

    dimage = None
    if apply_edge_term:
        dimage = np.array(np.gradient(image, spacing[0], spacing[1], spacing[2]))
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)

    sm_iters = _smoothing_iterations(smoothing, spacing)

    for _ in range(int(num_iter)):
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure=se_ball)
        elif balloon < 0:
            aux = ndi.binary_erosion(u, structure=se_ball)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]

        if apply_edge_term and dimage is not None:
            aux = np.zeros_like(image)
            du = np.gradient(u)
            for el1, el2 in zip(dimage, du):
                aux += el1 * el2
            u[aux > 0] = 1
            u[aux < 0] = 0

        # Surface tension: prefer volume-preserving smoothing over _curvop.
        # _curvop behaves like alternating opening/closing and can rapidly thin
        # or erase interior lumen masks; spacing-aware closing stabilizes the
        # surface without systematic shrink.
        for _s in range(sm_iters):
            u = _binary_closing_edgesafe(u, structure=se_ball).astype(np.int8)

        if blocker is not None:
            u[blocker] = 0

    return u


# ─────────────────────────── main entry point ────────────────────────────── #


def active_contour_grow(
    image,
    seed_mask,
    start_point,
    end_point,
    # ── initialisation ──
    radius=10.0,
    spacing=None,
    # ── edge image ──
    sigma=10.0,
    edge_alpha: float = 100.0,
    # ── intensity flattening (optional) ──
    low_intensity_equalize_below: float = 0.0,
    # ── MGAC parameters ──
    balloon=0.5,        # >0 = inflate; helps cross weak-edge interior regions
    smoothing=1,        # morphological smoothing steps per iteration
    total_iter=200,     # total evolution iterations
    # ── animation ──
    yield_every=5,      # iterations between display updates
    # ── geometry ──
    margin=5.0,
    blocker_mask=None,
    init_level_set=None,
    corridor_mask: Optional[np.ndarray] = None,
):
    """3-D morphological geodesic active contour iterator.

    Parameters
    ----------
    image : ndarray (Z, Y, X)
        3-D fluorescence image.
    seed_mask : bool ndarray
        User's centerline brush stroke.
    start_point, end_point : array-like (z, y, x)
        Vessel endpoints for the length constraint.
    radius : float
        Nominal tube radius in **isotropic-voxel units**: the physical tube
        radius is ``radius * min(spacing)`` (same convention as before when
        spacing was ``(1,1,1)``).
    spacing : sequence of 3 floats, optional
        ``(s_z, s_y, s_x)`` in world units.  Omitted or ``None`` ⇒ ``(1,1,1)``.
    sigma : float
        Physical isotropic Gaussian scale for the edge image (see
        :func:`_inverse_gaussian_gradient_physical`).
    balloon : float
        Balloon (inflation) coefficient (see scikit-image MGAC).
    smoothing : int
        Morphological smoothing iterations per evolution step (scaled down
        when the grid is strongly anisotropic).
    total_iter : int
        Total number of MGAC iterations.
    yield_every : int
        Iterations between animation-frame yields.
    margin : float
        Physical slack ``margin * mean(spacing)`` along the vessel axis.
    blocker_mask : bool ndarray, optional
        Voxels where the speed image is forced to 0 and the level set is
        cleared every iteration (hard barrier).
    init_level_set : bool ndarray, optional
        If given, used (after clipping with the length mask) instead of the
        EDT tube from *seed_mask*.
    corridor_mask : bool ndarray, optional
        If given, replaces the default start--end **length** cylinder: only
        voxels where this mask is True may be occupied by the level set each
        iteration.  Use a mask that surrounds the full vessel axis (e.g. an
        envelope around a polyline) so curved branches are not clipped away.
        Also enables a **balloon-first warmup**: the edge/shrink step of MGAC
        removes thin seed tubes in very few iterations, so the first fraction
        of iterations only applies balloon inflation (with a relaxed speed
        threshold) before the full morphological AC runs.

    Yields
    ------
    (int, ndarray bool)
        Iteration counter and current boolean segmentation mask.
    """
    image = np.asarray(image, dtype=np.float64)
    # Optionally flatten lumen/background: treat all values <= threshold as 0
    # so tiny intensity variations do not create micro-gradients and pinholes.
    thr0 = float(low_intensity_equalize_below)
    if thr0 > 0:
        image = image.copy()
        image[image <= thr0] = 0.0
    seed_mask = np.asarray(seed_mask, dtype=bool)
    shape = image.shape
    spacing = _normalize_spacing(spacing, shape)

    # Normalise to [0, 1] — required by inverse_gaussian_gradient convention
    imin, imax = image.min(), image.max()
    img_norm = (
        (image - imin) / (imax - imin) if imax > imin else np.zeros_like(image)
    )

    gimage = _inverse_gaussian_gradient_physical(
        img_norm, alpha=float(edge_alpha), sigma=sigma, spacing=spacing
    )
    if blocker_mask is not None:
        bm = np.asarray(blocker_mask, dtype=bool)
        gimage = np.asarray(gimage, dtype=np.float64).copy()
        gimage[bm] = 0.0

    if corridor_mask is not None:
        lmask = np.asarray(corridor_mask, dtype=bool)
        if lmask.shape != shape:
            raise ValueError(
                f"corridor_mask shape {lmask.shape} != image shape {shape}"
            )
    else:
        lmask = _length_mask(
            shape,
            np.asarray(start_point, dtype=np.float64),
            np.asarray(end_point, dtype=np.float64),
            margin,
            spacing,
        )

    if init_level_set is not None:
        ls = np.asarray(init_level_set, dtype=bool) & lmask
    else:
        ls = _init_tube(seed_mask, radius, spacing) & lmask
    ls = stabilize_mgac_mask(ls, spacing)
    if blocker_mask is not None:
        ls &= ~np.asarray(blocker_mask, dtype=bool)

    yield 0, ls.copy()

    steps_done = 0
    outer_steps = max(1, (total_iter + yield_every - 1) // yield_every)
    # NOTE: This plugin now focuses on segmenting one branch at a time.
    # The earlier "balloon-first warmup" made the contour aggressively search for
    # new branches and could leak through small wall holes. We therefore always
    # run the full MGAC update (edge term on) from the start and never override
    # the user-provided balloon sign/magnitude.
    eff_balloon = float(balloon)

    for _ in range(outer_steps):
        iters_this = min(yield_every, total_iter - steps_done)
        if iters_this <= 0:
            break

        edge_on = True
        balloon_pct = 40.0
        smooth_use = smoothing

        ls = _morphological_geodesic_active_contour_aniso(
            gimage,
            iters_this,
            ls,
            spacing,
            smoothing=smooth_use,
            threshold="auto",
            balloon=eff_balloon,
            blocker_mask=blocker_mask,
            apply_edge_term=edge_on,
            balloon_percentile=balloon_pct,
        )

        ls = np.asarray(ls, dtype=bool) & lmask
        if blocker_mask is not None:
            ls &= ~np.asarray(blocker_mask, dtype=bool)
        ls = stabilize_mgac_mask(ls, spacing)

        steps_done += iters_this
        yield steps_done, ls.copy()

        if steps_done >= total_iter:
            break

"""Branch-point → working-grid mapping (face slices, anisotropic pyramids)."""

from __future__ import annotations

import numpy as np


def test_world_to_voxel_indices_at_z_faces() -> None:
    from regiongrow._spatial import world_to_voxel_indices_zyx

    shape = (25, 32, 32)
    scale = np.array([2.0, 1.0, 1.0])
    translate = np.array([0.0, 0.0, 0.0])
    world = np.array(
        [
            [0.0, 16.0, 16.0],
            [48.0, 16.0, 16.0],
        ]
    )
    out = world_to_voxel_indices_zyx(
        world, translate=translate, scale=scale, shape=shape
    )
    assert tuple(int(x) for x in out[0]) == (0, 16, 16)
    assert tuple(int(x) for x in out[1]) == (24, 16, 16)


def test_working_grid_matches_ngff_coarse_face() -> None:
    """Last coarse Z voxel maps via level spacing, not finest index // df."""
    from regiongrow._spatial import (
        scale_translate_zyx_from_spatial_kwargs,
        spatial_alignment_for_pyramid_level,
        world_to_voxel_indices_zyx,
    )

    class _Img:
        multiscale = True
        ndim = 3
        scale = (2.05, 0.86, 0.86)
        translate = (0, 0, 0)
        metadata = {
            "coordinateTransformations": [
                [{"type": "scale", "scale": [2.05, 0.86, 0.86]}],
                [{"type": "scale", "scale": [20.5, 8.6, 8.6]}],
            ]
        }
        downsample_factors = [
            np.array([1.0, 1.0, 1.0]),
            np.array([10.0, 10.0, 10.0]),
        ]
        data = [np.zeros((20, 10, 10)), np.zeros((2, 1, 1))]

    img = _Img()
    shape_work = (2, 1, 1)
    skw = spatial_alignment_for_pyramid_level(img, 1)
    translate, scale = scale_translate_zyx_from_spatial_kwargs(skw)
    # World position of coarse voxel z=1 (last face).
    world_last = np.array([[20.5, 0.0, 0.0]])
    work = world_to_voxel_indices_zyx(
        world_last, translate=translate, scale=scale, shape=shape_work
    )
    assert int(work[0, 0]) == 1

    world_first = np.array([[0.0, 0.0, 0.0]])
    work0 = world_to_voxel_indices_zyx(
        world_first, translate=translate, scale=scale, shape=shape_work
    )
    assert int(work0[0, 0]) == 0


def test_polyline_tube_covers_working_grid_face_knots() -> None:
    from regiongrow._algorithm import polyline_tube_mask
    from regiongrow._volume_utils import voxel_spacing_zyx_for_level

    class _Img:
        multiscale = True
        scale = (2.05, 0.86, 0.86)
        metadata = {
            "coordinateTransformations": [
                [{"type": "scale", "scale": [2.05, 0.86, 0.86]}],
            ]
        }

    shape = (25, 32, 32)
    spacing = voxel_spacing_zyx_for_level(_Img(), 0, shape)
    poly = np.array([[0, 16, 16], [24, 16, 16]], dtype=np.int64)
    tube = polyline_tube_mask(shape, poly, radius_vox=15.0, spacing=spacing)
    for z, y, x in poly:
        assert tube[int(z), int(y), int(x)]

from __future__ import annotations

import numpy as np


class _FakeMultiscaleImage:
    multiscale = True
    ndim = 3
    scale = (2.0, 1.0, 1.0)

    def __init__(self) -> None:
        self.downsample_factors = [
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([4.0, 4.0, 4.0]),
        ]
        self._shapes = [(8, 10, 12), (4, 5, 6), (2, 3, 3)]
        self.data = [np.zeros(s, dtype=np.uint8) for s in self._shapes]


def test_pyramid_axis_steps_from_downsample_factors() -> None:
    from regiongrow._volume_utils import pyramid_axis_steps

    lyr = _FakeMultiscaleImage()
    assert pyramid_axis_steps(lyr, 0) == (1, 1, 1)
    assert pyramid_axis_steps(lyr, 1) == (2, 2, 2)
    assert pyramid_axis_steps(lyr, 2) == (4, 4, 4)


def test_pyramid_axis_steps_shape_ratio_fallback() -> None:
    from regiongrow._volume_utils import pyramid_axis_steps

    class _Plain:
        multiscale = False
        ndim = 3
        data = np.zeros((8, 10, 12), dtype=np.uint8)

    lyr = _Plain()
    assert pyramid_axis_steps(lyr, 0) == (1, 1, 1)


def test_world_bounds_zyx_for_pyramid_level() -> None:
    from regiongrow._spatial import world_bounds_zyx_for_pyramid_level

    lyr = _FakeMultiscaleImage()
    # Level 2: shape (2,3,3), factors (4,4,4) → scale (8,4,4) with base scale (2,1,1)
    z_lo, z_hi = world_bounds_zyx_for_pyramid_level(lyr, 2)[0]
    assert z_lo == 0.0
    assert z_hi == 8.0  # (2-1) * 8

    y_lo, y_hi = world_bounds_zyx_for_pyramid_level(lyr, 2)[1]
    assert y_lo == 0.0
    assert y_hi == 8.0  # (3-1) * 4

    # Level 1: Z=4 voxels, step factor 2 → world scale 4 on Z
    z_lo, z_hi = world_bounds_zyx_for_pyramid_level(lyr, 1)[0]
    assert z_lo == 0.0
    assert z_hi == 12.0  # (4-1) * 4

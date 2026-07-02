from __future__ import annotations

import numpy as np


class _FakeMultiscaleImage:
    multiscale = True
    ndim = 3
    scale = (1.0, 1.0, 1.0)
    downsample_factors = [
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 2.0, 2.0]),
        np.array([4.0, 4.0, 4.0]),
    ]
    data = [
        np.zeros((400, 200, 200), dtype=np.uint8),
        np.zeros((200, 100, 100), dtype=np.uint8),
        np.zeros((100, 50, 50), dtype=np.uint8),
    ]


def test_branch_point_base_size_scales_with_pyramid_level() -> None:
    from regiongrow._widget import _suggested_branch_point_base_size

    lyr = _FakeMultiscaleImage()
    finest = _suggested_branch_point_base_size(lyr, 0)
    coarse = _suggested_branch_point_base_size(lyr, 2)
    assert coarse <= finest
    assert coarse == max(8.0, finest / 4.0)

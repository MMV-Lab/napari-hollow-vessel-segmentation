"""Tests for polyline rasterisation used by branch preview and grow."""

from __future__ import annotations

import numpy as np

from regiongrow._algorithm import polyline_to_line_mask, polyline_tube_mask


def test_polyline_line_mask_includes_face_and_corner_knots() -> None:
    shape = (10, 32, 32)
    poly = np.array(
        [
            [0, 16, 16],
            [9, 16, 16],
            [9, 31, 31],
        ],
        dtype=np.int64,
    )
    line = polyline_to_line_mask(shape, poly)
    for k in range(len(poly)):
        z, y, x = int(poly[k, 0]), int(poly[k, 1]), int(poly[k, 2])
        assert line[z, y, x], f"knot {k} missing from centerline"
    assert int(line.sum()) >= len(poly)


def test_polyline_tube_covers_last_slice_knot() -> None:
    shape = (8, 24, 24)
    poly = np.array([[2, 12, 12], [7, 12, 12]], dtype=np.int64)
    tube = polyline_tube_mask(shape, poly, radius_vox=4.0, spacing=(1.0, 1.0, 1.0))
    z, y, x = int(poly[-1, 0]), int(poly[-1, 1]), int(poly[-1, 2])
    assert tube[z, y, x]

from __future__ import annotations

import numpy as np


def test_materialize_labels_level_uses_finest_not_coarsest() -> None:
    from regiongrow._volume_utils import materialize_labels_level

    class _FakeMultiscaleLabels:
        multiscale = True

        def __init__(self) -> None:
            self.data = [
                np.ones((8, 10, 12), dtype=np.uint8),
                np.ones((4, 5, 6), dtype=np.uint8) * 2,
            ]

    lyr = _FakeMultiscaleLabels()
    fine = materialize_labels_level(lyr, 0)
    assert fine.shape == (8, 10, 12)
    coarse = materialize_labels_level(lyr, 1)
    assert coarse.shape == (4, 5, 6)


def test_labels_pyramid_level_for_image_level_matches_shape() -> None:
    from regiongrow._volume_utils import labels_pyramid_level_for_image_level

    class _Layer:
        def __init__(self, shapes) -> None:
            self.multiscale = True
            self.data = [np.zeros(s, dtype=np.uint8) for s in shapes]

    img = _Layer([(8, 10, 12), (4, 5, 6)])
    lbl = _Layer([(8, 10, 12), (4, 5, 6)])
    assert labels_pyramid_level_for_image_level(lbl, img, 1) == 1
    assert labels_pyramid_level_for_image_level(lbl, img, 0) == 0

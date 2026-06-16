from __future__ import annotations

import numpy as np


def test_zarr_create_array_on_group() -> None:
    import zarr

    from regiongrow._zarr_compat import zarr_create_array

    root = zarr.open_group("/tmp/regiongrow_zarr_compat_test", mode="w")
    arr = zarr_create_array(
        root,
        "0",
        shape=(4, 8, 16),
        chunks=(2, 4, 8),
        dtype=np.uint8,
    )
    assert arr.shape == (4, 8, 16)
    arr[0, 0, 0] = 7
    assert int(root["0"][0, 0, 0]) == 7

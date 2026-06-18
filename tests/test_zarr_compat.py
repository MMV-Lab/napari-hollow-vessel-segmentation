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


def test_zarr_create_array_without_create_dataset() -> None:
    """Zarr-Python 3 removed ``Group.create_dataset``; only ``create_array`` remains."""
    import zarr
    from numcodecs import Blosc

    from regiongrow._zarr_compat import zarr_create_array

    root = zarr.open_group("/tmp/regiongrow_zarr_compat_v3api", mode="w", zarr_format=2)
    staging = root.create_group("staging")

    class _V3ApiOnly:
        def __init__(self, inner: zarr.Group) -> None:
            self._inner = inner

        def __contains__(self, key: object) -> bool:
            return key in self._inner

        def __getitem__(self, key: str) -> object:
            return self._inner[key]

        def __delitem__(self, key: str) -> None:
            del self._inner[key]

        @property
        def metadata(self) -> object:
            return self._inner.metadata

        def create_array(self, *args: object, **kwargs: object) -> zarr.Array:
            return self._inner.create_array(*args, **kwargs)

    wrapper = _V3ApiOnly(staging)
    arr = zarr_create_array(
        wrapper,  # type: ignore[arg-type]
        "0",
        shape=(4, 8, 16),
        chunks=(2, 4, 8),
        dtype=np.uint8,
        compressor=Blosc(),
    )
    assert arr.shape == (4, 8, 16)
    arr[0, 0, 0] = 3
    assert int(staging["0"][0, 0, 0]) == 3

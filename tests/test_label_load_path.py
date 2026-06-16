from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np


def _make_tiny_omezarr(path: Path) -> None:
    import zarr

    root = zarr.open_group(str(path), mode="w")
    root.create_dataset(
        "s0",
        shape=(8, 10, 12),
        dtype=np.uint8,
        chunks=(4, 5, 6),
        compressor=None,
        data=np.zeros((8, 10, 12), dtype=np.uint8),
    )
    root.create_dataset(
        "s1",
        shape=(4, 5, 6),
        dtype=np.uint8,
        chunks=(2, 5, 6),
        compressor=None,
        data=np.zeros((4, 5, 6), dtype=np.uint8),
    )
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "image",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                    ],
                },
                {
                    "path": "s1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [2.0, 2.0, 2.0]}
                    ],
                },
            ],
        }
    ]


def test_resolve_label_load_target() -> None:
    from regiongrow._omezarr_reader import resolve_label_load_target

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        (store / "labels" / "segmentation_v7").mkdir(parents=True)

        root_store, name = resolve_label_load_target(store)
        assert root_store == store.resolve()
        assert name is None

        labels_store, name = resolve_label_load_target(store / "labels")
        assert labels_store == store.resolve()
        assert name is None

        direct_store, name = resolve_label_load_target(
            store / "labels" / "segmentation_v7"
        )
        assert direct_store == store.resolve()
        assert name == "segmentation_v7"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_materialize_finest_labels_to_coarse_working_shape() -> None:
    from regiongrow._omezarr_reader import materialize_saved_labels_at_shape
    from regiongrow._save_segmentation_zarr import (
        read_zarr_labels_at_shape,
        write_segmentation_labels_to_ome_zarr,
    )

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg_fine = np.zeros((8, 10, 12), dtype=np.uint8)
        seg_fine[2:6, 3:7, 4:10] = 1
        write_segmentation_labels_to_ome_zarr(
            store,
            seg_fine,
            build_pyramid=False,
            save_resolution="finest",
        )

        import zarr

        arr = zarr.open_group(str(store), mode="r")["labels"]["segmentation"]["0"]
        assert tuple(arr.shape) == (8, 10, 12)

        loaded = materialize_saved_labels_at_shape(store, (4, 5, 6))
        assert loaded is not None
        data, name = loaded
        assert name == "segmentation"
        assert data.shape == (4, 5, 6)
        assert int(data.sum()) > 0

        # Chunked zarr read must match itself (no full-volume materialize path).
        chunked = read_zarr_labels_at_shape(arr, (4, 5, 6))
        np.testing.assert_array_equal((data > 0).astype(np.uint8), chunked)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_saved_labels_layerdata_lazy_pyramid() -> None:
    from regiongrow._omezarr_reader import load_saved_labels_layerdata
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg_fine = np.zeros((8, 10, 12), dtype=np.uint8)
        seg_fine[2:6, 3:7, 4:10] = 1
        write_segmentation_labels_to_ome_zarr(
            store,
            seg_fine,
            build_pyramid=True,
            save_resolution="finest",
        )

        layerdata = load_saved_labels_layerdata(store, label_name="segmentation")
        assert layerdata is not None
        ldata, lkwargs, kind = layerdata
        assert kind == "labels"
        assert lkwargs["name"] == "segmentation"
        assert isinstance(ldata, (list, tuple))
        assert len(ldata) == 2
        for level in ldata:
            assert not isinstance(level, np.ndarray)
            assert hasattr(level, "shape")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

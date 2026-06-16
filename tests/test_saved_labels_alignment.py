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
    root.create_dataset(
        "s2",
        shape=(2, 3, 3),
        dtype=np.uint8,
        chunks=(2, 3, 3),
        compressor=None,
        data=np.zeros((2, 3, 3), dtype=np.uint8),
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
                {
                    "path": "s2",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [4.0, 4.0, 4.0]}
                    ],
                },
            ],
        }
    ]


class _FakeMultiscaleImage:
    multiscale = True
    ndim = 3
    scale = (1.0, 1.0, 1.0)
    translate = (0.0, 0.0, 0.0)
    rotate = (0.0, 0.0, 0.0)
    shear = (0.0, 0.0, 0.0)

    def __init__(self) -> None:
        self.downsample_factors = [
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([4.0, 4.0, 4.0]),
        ]
        self.data = [
            np.zeros((8, 10, 12), dtype=np.uint8),
            np.zeros((4, 5, 6), dtype=np.uint8),
            np.zeros((2, 3, 3), dtype=np.uint8),
        ]


def test_spatial_alignment_for_saved_labels_coarse_grid() -> None:
    from regiongrow._spatial import (
        spatial_alignment_for_pyramid_level,
        spatial_alignment_for_saved_labels,
        spatial_alignment_kwargs,
    )

    img = _FakeMultiscaleImage()
    coarse = np.zeros((4, 5, 6), dtype=np.uint8)
    finest_kw = spatial_alignment_kwargs(img)
    coarse_kw = spatial_alignment_for_saved_labels(img, coarse)
    expected = spatial_alignment_for_pyramid_level(img, 1)
    np.testing.assert_allclose(coarse_kw["scale"], expected["scale"])
    assert not np.allclose(coarse_kw["scale"], finest_kw["scale"])


def test_save_load_roundtrip_at_working_pyramid_level() -> None:
    from regiongrow._omezarr_reader import materialize_saved_labels_at_shape
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr
    from regiongrow._spatial import spatial_alignment_for_pyramid_level
    from regiongrow._widget import _binary_segmentation_colormap
    import napari

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg = np.zeros((4, 5, 6), dtype=np.uint8)
        seg[1:3, 2:4, 2:5] = 1
        write_segmentation_labels_to_ome_zarr(
            store, seg, build_pyramid=True, save_resolution="working"
        )
        img = _FakeMultiscaleImage()
        lvl = 1
        tgt = tuple(int(x) for x in img.data[lvl].shape)
        loaded = materialize_saved_labels_at_shape(store, tgt)
        assert loaded is not None
        data, _name = loaded
        assert int(data.sum()) > 0
        skw = spatial_alignment_for_pyramid_level(img, lvl)
        v = napari.Viewer(show=False)
        lyr = v.add_labels(
            np.asarray(data, dtype=np.int32),
            colormap=_binary_segmentation_colormap("cyan"),
            **skw,
        )
        assert int(np.asarray(lyr.data).sum()) > 0
        v.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_working_save_at_coarse_pyramid_writes_expected_levels() -> None:
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg = np.zeros((4, 5, 6), dtype=np.uint8)
        seg[1:3, 2:4, 2:5] = 1
        meta = write_segmentation_labels_to_ome_zarr(
            store,
            seg,
            build_pyramid=True,
            save_resolution="working",
        )
        assert meta["levels_written"] == 2
        assert meta["image_pyramid_index"] == 1

        import zarr

        g = zarr.open_group(str(store), mode="r")["labels"]["segmentation"]
        assert tuple(g["0"].shape) == (4, 5, 6)
        assert tuple(g["1"].shape) == (2, 3, 3)
        assert int(np.asarray(g["0"]).sum()) > 0
        assert "2" not in g
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

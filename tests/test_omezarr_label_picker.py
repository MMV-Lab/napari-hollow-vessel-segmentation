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
            ],
        }
    ]


def test_list_segmentation_label_groups() -> None:
    from regiongrow._omezarr_reader import (
        format_label_group_choice,
        list_segmentation_label_groups,
    )
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg = np.zeros((8, 10, 12), dtype=np.uint8)
        seg[1:3, 2:4, 3:6] = 1
        write_segmentation_labels_to_ome_zarr(store, seg, build_pyramid=False)
        write_segmentation_labels_to_ome_zarr(store, seg, build_pyramid=False)
        groups = list_segmentation_label_groups(store, check_foreground=True)
        names = {g["name"] for g in groups}
        assert names == {"segmentation", "segmentation_v2"}
        assert all(g["has_foreground"] for g in groups)
        fast = list_segmentation_label_groups(store, check_foreground=False)
        assert {g["name"] for g in fast} == names
        assert all(g["has_foreground"] is None for g in fast)
        assert "segmentation" in format_label_group_choice(groups[0])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_list_segmentation_label_groups_ignores_stale_attrs() -> None:
    from regiongrow._omezarr_reader import (
        _ngff_label_names_from_store,
        list_segmentation_label_groups,
    )
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg = np.zeros((8, 10, 12), dtype=np.uint8)
        seg[1:3, 2:4, 3:6] = 1
        write_segmentation_labels_to_ome_zarr(store, seg, build_pyramid=False)

        import zarr

        root = zarr.open_group(str(store), mode="r+")
        root.attrs["labels"] = [
            "segmentation",
            "segmentation_v2",
            "segmentation_v99",
            "segmentation__tmp_deadbeef",
        ]

        names = _ngff_label_names_from_store(store)
        assert names == ["segmentation"]

        groups = list_segmentation_label_groups(store, check_foreground=False)
        assert [g["name"] for g in groups] == ["segmentation"]
        assert groups[0]["shape"] == (8, 10, 12)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

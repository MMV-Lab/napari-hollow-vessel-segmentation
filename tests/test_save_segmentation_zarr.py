from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np


def _make_tiny_omezarr(path: Path) -> None:
    import zarr

    root = zarr.open_group(str(path), mode="w")
    s0 = root.create_dataset(
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
    # minimal NGFF multiscales
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
                    "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
                },
                {
                    "path": "s1",
                    "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0, 2.0]}],
                },
            ],
        }
    ]
    # keep mypy/linters happy (unused local)
    _ = s0


def test_write_segmentation_labels_versioned() -> None:
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)

        seg = np.zeros((8, 10, 12), dtype=np.uint8)
        seg[2:6, 3:7, 4:10] = 1

        meta1 = write_segmentation_labels_to_ome_zarr(store, seg, build_pyramid=True)
        assert meta1["labels_group"] == "labels/segmentation"

        meta2 = write_segmentation_labels_to_ome_zarr(store, seg, build_pyramid=True)
        assert meta2["labels_group"] == "labels/segmentation_v2"

        import zarr

        root = zarr.open_group(str(store), mode="r")
        assert "labels" in root.attrs
        assert "segmentation" in root.attrs["labels"]
        assert "segmentation_v2" in root.attrs["labels"]
        assert "labels" in root
        assert "segmentation" in root["labels"]
        g = root["labels/segmentation"]
        assert "multiscales" in g.attrs
        assert "0" in g and "1" in g
        np.testing.assert_array_equal(np.asarray(g["0"]), (seg > 0).astype(np.uint8))
        assert tuple(g["1"].shape) == (4, 5, 6)

        # If ome-zarr-py is installed, ensure it can discover the labels node.
        try:
            from ome_zarr.io import ZarrLocation
            from ome_zarr.reader import Reader
        except Exception:
            ZarrLocation = None  # type: ignore[assignment]
        if ZarrLocation is not None:
            loc = ZarrLocation(str(store), mode="r")
            nodes = list(Reader(loc)())
            # At least one node should exist; labels are discovered by the Labels spec.
            assert len(nodes) >= 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_upsample_labels_nearest() -> None:
    from regiongrow._save_segmentation_zarr import upsample_labels_nearest

    src = np.zeros((2, 3, 4), dtype=np.uint8)
    src[1, 1, 1] = 1
    out = upsample_labels_nearest(src, (4, 6, 8))
    assert out.shape == (4, 6, 8)
    assert out.dtype == np.uint8
    assert int(out.sum()) >= 1


def test_omezarr_reader_opens_image_only() -> None:
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr
    from regiongrow._omezarr_reader import read_omezarr_image, read_omezarr_with_segmentation

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)

        seg = np.zeros((8, 10, 12), dtype=np.uint8)
        seg[1:4, 2:5, 3:8] = 1
        write_segmentation_labels_to_ome_zarr(store, seg, build_pyramid=True)

        for reader in (read_omezarr_image, read_omezarr_with_segmentation):
            layers = reader(store)
            kinds = [k for _d, _m, k in layers]
            assert kinds == ["image"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_napari_sort_multiscale_list_decreasing() -> None:
    from regiongrow._omezarr_reader import _napari_sort_multiscale_list

    a_fine = np.zeros((20, 20, 20), dtype=np.uint8)
    a_mid = np.zeros((10, 10, 10), dtype=np.uint8)
    a_coarse = np.zeros((5, 5, 5), dtype=np.uint8)
    wrong = [a_coarse, a_fine, a_mid]
    meta = {
        "coordinateTransformations": [
            [{"type": "scale", "scale": [4.0, 1.0, 1.0]}],
            [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
            [{"type": "scale", "scale": [2.0, 1.0, 1.0]}],
        ]
    }
    data, m2 = _napari_sort_multiscale_list(wrong, meta)
    sizes = [int(x.size) for x in data]
    assert sizes == sorted(sizes, reverse=True)
    assert all(sizes[i] > sizes[i + 1] for i in range(len(sizes) - 1))
    assert len(m2["coordinateTransformations"]) == 3
    assert m2["coordinateTransformations"][0][0]["scale"][0] == 1.0

    dup_a = np.zeros((8, 8, 8), dtype=np.uint8)
    dup_b = np.zeros((8, 8, 8), dtype=np.uint8)
    d2, m3 = _napari_sort_multiscale_list(
        [dup_a, dup_b, a_coarse],
        {"coordinateTransformations": [[{"i": 0}], [{"i": 1}], [{"i": 2}]]},
    )
    assert len(d2) == 2


def test_ngff_label_names_nested_ome_attrs() -> None:
    from regiongrow._omezarr_reader import _ngff_label_names_from_store
    import zarr

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "nested.ome.zarr"
        root = zarr.open_group(str(store), mode="w")
        root.attrs["ome"] = {"labels": ["segmentation", "ghost_v99"]}
        lg = root.create_group("labels").create_group("segmentation")
        lg.create_dataset(
            "0",
            shape=(4, 5, 6),
            dtype=np.uint8,
            chunks=(2, 5, 6),
            compressor=None,
            data=np.zeros((4, 5, 6), dtype=np.uint8),
        )
        lg.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": "0"}]}]
        assert _ngff_label_names_from_store(store) == ["segmentation"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_save_coarse_segmentation_upsamples_to_finest() -> None:
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        # Coarse mask (level-1 shape) saved while curating at a downsampled grid.
        seg_coarse = np.zeros((4, 5, 6), dtype=np.uint8)
        seg_coarse[1:3, 2:4, 2:5] = 1
        write_segmentation_labels_to_ome_zarr(
            store, seg_coarse, build_pyramid=True, save_resolution="finest"
        )
        import zarr

        g = zarr.open_group(str(store), mode="r")["labels"]["segmentation"]
        fine = np.asarray(g["0"])
        assert fine.shape == (8, 10, 12)
        assert int(fine.sum()) > int(seg_coarse.sum())
        coarse = np.asarray(g["1"])
        assert coarse.shape == (4, 5, 6)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_save_working_resolution_keeps_coarse_shape() -> None:
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr
    from regiongrow._omezarr_reader import materialize_saved_labels_at_shape

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg_coarse = np.zeros((4, 5, 6), dtype=np.uint8)
        seg_coarse[1:3, 2:4, 2:5] = 1
        meta = write_segmentation_labels_to_ome_zarr(
            store,
            seg_coarse,
            build_pyramid=False,
            save_resolution="working",
        )
        assert meta["save_resolution"] == "working"
        assert meta["levels_written"] == 1
        import zarr

        g = zarr.open_group(str(store), mode="r")["labels"]["segmentation"]
        assert tuple(g["0"].shape) == (4, 5, 6)
        np.testing.assert_array_equal(np.asarray(g["0"]), seg_coarse)
        loaded = materialize_saved_labels_at_shape(store, (4, 5, 6))
        assert loaded is not None
        arr, name = loaded
        assert name == "segmentation"
        assert arr.shape == (4, 5, 6)
        assert int(arr.sum()) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chunked_finest_matches_in_memory_upsample() -> None:
    from regiongrow._save_segmentation_zarr import (
        upsample_labels_nearest,
        write_segmentation_labels_to_ome_zarr,
    )

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg_coarse = np.zeros((4, 5, 6), dtype=np.uint8)
        seg_coarse[1:3, 2:4, 2:5] = 1
        expected = upsample_labels_nearest(seg_coarse, (8, 10, 12))
        write_segmentation_labels_to_ome_zarr(
            store, seg_coarse, build_pyramid=False, save_resolution="finest"
        )
        import zarr

        g = zarr.open_group(str(store), mode="r")["labels"]["segmentation"]
        np.testing.assert_array_equal(np.asarray(g["0"]), expected)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_overwrite_fixed_labels_name() -> None:
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg1 = np.zeros((8, 10, 12), dtype=np.uint8)
        seg1[2:6, 3:7, 4:10] = 1
        write_segmentation_labels_to_ome_zarr(store, seg1, build_pyramid=False)
        seg2 = np.zeros((8, 10, 12), dtype=np.uint8)
        seg2[1:3, 1:3, 1:3] = 1
        meta = write_segmentation_labels_to_ome_zarr(
            store,
            seg2,
            build_pyramid=False,
            labels_name="segmentation",
        )
        assert meta["labels_name"] == "segmentation"
        import zarr

        g = zarr.open_group(str(store), mode="r")["labels"]["segmentation"]
        np.testing.assert_array_equal(np.asarray(g["0"]), (seg2 > 0).astype(np.uint8))
        root = zarr.open_group(str(store), mode="r")
        assert root.attrs["labels"] == ["segmentation"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_autosave_checkpoint_replaces_different_shape() -> None:
    from regiongrow._save_segmentation_zarr import write_segmentation_checkpoint

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg_fine = np.zeros((8, 10, 12), dtype=np.uint8)
        seg_fine[2:6, 3:7, 4:10] = 1
        write_segmentation_checkpoint(
            store,
            seg_fine,
            save_resolution="finest",
            build_pyramid=False,
        )
        seg_coarse = np.zeros((4, 5, 6), dtype=np.uint8)
        seg_coarse[1:3, 2:4, 2:5] = 1
        write_segmentation_checkpoint(store, seg_coarse)
        import zarr

        g = zarr.open_group(str(store), mode="r")["labels"]["segmentation_autosave"]
        assert tuple(g["0"].shape) == (4, 5, 6)
        assert "1" not in g
        np.testing.assert_array_equal(np.asarray(g["0"]), seg_coarse)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_autosave_checkpoint_working_single_level() -> None:
    from regiongrow._save_segmentation_zarr import write_segmentation_checkpoint

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg = np.zeros((4, 5, 6), dtype=np.uint8)
        seg[1:3, 2:4, 2:5] = 1
        meta = write_segmentation_checkpoint(store, seg)
        assert meta["levels_written"] == 1
        assert meta["save_resolution"] == "working"
        import zarr

        g = zarr.open_group(str(store), mode="r")["labels"]["segmentation_autosave"]
        assert tuple(g["0"].shape) == (4, 5, 6)
        assert "1" not in g
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_materialize_saved_labels_at_working_shape() -> None:
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr
    from regiongrow._omezarr_reader import materialize_saved_labels_at_shape

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg = np.zeros((8, 10, 12), dtype=np.uint8)
        seg[2:6, 3:7, 4:10] = 1
        write_segmentation_labels_to_ome_zarr(store, seg, build_pyramid=True)
        out, name = materialize_saved_labels_at_shape(store, (4, 5, 6))
        assert name == "segmentation"
        assert out.shape == (4, 5, 6)
        assert int(out.sum()) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pick_skips_empty_autosave() -> None:
    from regiongrow._save_segmentation_zarr import (
        write_segmentation_checkpoint,
        write_segmentation_labels_to_ome_zarr,
    )
    from regiongrow._omezarr_reader import _pick_saved_segmentation_label

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg = np.zeros((8, 10, 12), dtype=np.uint8)
        seg[1:3, 2:4, 3:6] = 1
        write_segmentation_labels_to_ome_zarr(store, seg, build_pyramid=True)
        write_segmentation_checkpoint(store, np.zeros_like(seg), build_pyramid=True)
        import zarr

        root = zarr.open_group(str(store), mode="r")
        names = list(root.attrs["labels"])
        chosen = _pick_saved_segmentation_label(names, store)
        assert chosen == "segmentation"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_latest_saved_labels_layerdata() -> None:
    from regiongrow._save_segmentation_zarr import write_segmentation_labels_to_ome_zarr
    from regiongrow._omezarr_reader import load_latest_saved_labels_layerdata

    tmp = Path(tempfile.mkdtemp())
    try:
        store = tmp / "img.ome.zarr"
        _make_tiny_omezarr(store)
        seg = np.zeros((8, 10, 12), dtype=np.uint8)
        seg[1:4, 2:5, 3:8] = 1
        write_segmentation_labels_to_ome_zarr(store, seg, build_pyramid=True)
        tup = load_latest_saved_labels_layerdata(store)
        assert tup is not None
        data, meta, ltype = tup
        assert ltype == "labels"
        assert meta.get("name") == "segmentation"
        assert isinstance(data, list) and len(data) >= 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


from __future__ import annotations

import numpy as np


def test_ngff_axis_name_maps_bioio_dims() -> None:
    from regiongrow._bioio_to_omezarr import _ngff_axis_name

    assert _ngff_axis_name("Z") == "z"
    assert _ngff_axis_name("C") == "c"
    assert _ngff_axis_name("T") == "t"


def test_ngff_writer_kwargs_defaults_for_numpy_bioimage() -> None:
    from bioio import BioImage

    from regiongrow._bioio_to_omezarr import ngff_writer_kwargs_from_bioimage

    arr = np.zeros((5, 8, 16), dtype=np.uint16)
    img = BioImage(arr)
    xr = img.xarray_dask_data.squeeze(drop=True)
    shape = tuple(int(x) for x in xr.shape)
    kw = ngff_writer_kwargs_from_bioimage(
        img,
        tuple(str(d) for d in xr.dims),
        shape,
        __import__("pathlib").Path("test_volume.tif"),
    )
    assert kw["axes_names"] == ["z", "y", "x"]
    assert kw["axes_types"] == ["space", "space", "space"]
    assert kw["physical_pixel_size"] == [1.0, 1.0, 1.0]
    assert len(kw["channels"]) == 1
    assert kw["creator_info"]["name"] == "regiongrow"


def test_ngff_writer_kwargs_voxel_size_override() -> None:
    from bioio import BioImage

    from regiongrow._bioio_to_omezarr import ngff_writer_kwargs_from_bioimage

    arr = np.zeros((4, 10, 12), dtype=np.uint8)
    img = BioImage(arr)
    xr = img.xarray_dask_data.squeeze(drop=True)
    shape = tuple(int(x) for x in xr.shape)
    kw = ngff_writer_kwargs_from_bioimage(
        img,
        tuple(str(d) for d in xr.dims),
        shape,
        __import__("pathlib").Path("plain.tif"),
        voxel_size_zyx=(2.5, 0.5, 0.5),
        voxel_unit="micrometer",
    )
    assert kw["physical_pixel_size"] == [2.5, 0.5, 0.5]
    assert kw["axes_units"] == ["micrometer", "micrometer", "micrometer"]


def test_ngff_writer_kwargs_multichannel() -> None:
    from bioio import BioImage

    from regiongrow._bioio_to_omezarr import ngff_writer_kwargs_from_bioimage

    arr = np.zeros((3, 1, 6, 8), dtype=np.uint16)
    img = BioImage(arr)
    xr = img.xarray_dask_data.squeeze(drop=True)
    shape = tuple(int(x) for x in xr.shape)
    kw = ngff_writer_kwargs_from_bioimage(
        img,
        tuple(str(d) for d in xr.dims),
        shape,
        __import__("pathlib").Path("channels.ome.tif"),
    )
    assert kw["axes_names"][0] == "c"
    assert len(kw["channels"]) == 3


def test_dask_chunk_limit_covers_portal_like_level1_chunk() -> None:
    from regiongrow._bioio_to_omezarr import (
        _dask_array_chunk_size_limit,
        _max_on_disk_chunk_bytes,
    )

    # Portal-scale level 1 from bioio 16 MiB target (uint16).
    chunks = [(1, 1153, 7270), (1, 2307, 3635), (2, 1818, 1818)]
    dtype = np.dtype(np.uint16)
    nbytes = _max_on_disk_chunk_bytes(chunks, dtype)
    assert nbytes == 16_771_890
    assert _dask_array_chunk_size_limit(chunks, dtype) >= nbytes


def test_chunk_shapes_recomputed_per_pyramid_level() -> None:
    from regiongrow._bioio_to_omezarr import _chunk_shapes_for_levels

    level_shapes = [(530, 7270, 7270), (265, 3635, 3635), (133, 1818, 1818)]
    dtype = np.dtype(np.uint16)
    chunks = _chunk_shapes_for_levels(
        level_shapes,
        dtype,
        16 << 20,
        chunks=None,
        level0_shape=level_shapes[0],
        viz_level_shapes=level_shapes,
    )
    assert len(chunks) == 3
    assert chunks[1][1] == 2307

from __future__ import annotations

import numpy as np


def test_ngff_finest_spacing_from_metadata() -> None:
    from regiongrow._volume_utils import ngff_finest_voxel_spacing_zyx

    meta = {
        "coordinateTransformations": [
            [{"type": "scale", "scale": [2.05, 0.86, 0.86]}],
            [{"type": "scale", "scale": [20.5, 8.6, 8.6]}],
        ]
    }
    assert ngff_finest_voxel_spacing_zyx(meta) == (2.05, 0.86, 0.86)


def test_voxel_spacing_finest_prefers_ngff_over_unit_scale() -> None:
    from regiongrow._volume_utils import voxel_spacing_zyx_finest

    class _Layer:
        scale = (1.0, 1.0, 1.0)
        metadata = {
            "coordinateTransformations": [
                [{"type": "scale", "scale": [2.05, 0.86, 0.86]}],
            ]
        }

    assert voxel_spacing_zyx_finest(_Layer()) == (2.05, 0.86, 0.86)


def test_polyline_tube_isotropic_in_physical_space() -> None:
    from regiongrow._algorithm import polyline_tube_mask

    spacing = (2.057, 0.863, 0.863)
    shape = (40, 80, 80)
    poly = np.array([[20, 40, 40]], dtype=np.int64)
    tube = polyline_tube_mask(shape, poly, radius_vox=40.0, spacing=spacing)
    zz, yy, xx = np.nonzero(tube)
    z_phys = (int(zz.max() - zz.min()) + 1) * spacing[0]
    y_phys = (int(yy.max() - yy.min()) + 1) * spacing[1]
    x_phys = (int(xx.max() - xx.min()) + 1) * spacing[2]
    assert abs(z_phys - y_phys) / y_phys < 0.08
    assert abs(z_phys - x_phys) / x_phys < 0.08


def test_read_omezarr_image_sets_scale_from_ngff(tmp_path) -> None:
    import zarr

    from regiongrow._omezarr_reader import read_omezarr_image
    from regiongrow._zarr_compat import zarr_create_array

    store = tmp_path / "img.ome.zarr"
    root = zarr.open_group(str(store), mode="w", zarr_format=2)
    zarr_create_array(
        root, "0", shape=(20, 10, 10), chunks=(10, 5, 5), dtype=np.uint8
    )
    zarr_create_array(
        root, "1", shape=(2, 5, 5), chunks=(2, 5, 5), dtype=np.uint8
    )
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [2.05, 0.86, 0.86]}
                    ],
                },
                {
                    "path": "1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [20.5, 8.6, 8.6]}
                    ],
                },
            ],
        }
    ]
    _data, meta, kind = read_omezarr_image(store)[0]
    assert kind == "image"
    assert tuple(meta["scale"]) == (2.05, 0.86, 0.86)

"""Convert BioIO-readable images to multiscale OME-Zarr with complete NGFF metadata."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_BIOIO_TO_NGFF_AXIS = {
    "T": "t",
    "C": "c",
    "Z": "z",
    "Y": "y",
    "X": "x",
}
_NGFF_AXIS_TYPE = {
    "t": "time",
    "c": "channel",
    "z": "space",
    "y": "space",
    "x": "space",
}
_SCALE_KEYS = ("T", "C", "Z", "Y", "X")
_CHANNEL_COLORS_HEX = (
    "ffffff",
    "ff0000",
    "00ff00",
    "0000ff",
    "ffff00",
    "ff00ff",
    "00ffff",
    "ff8800",
    "8800ff",
    "00ff88",
)


def _pyramid_level_shapes(
    level0_shape: Tuple[int, ...],
    *,
    n_spatial: int,
    num_levels: int,
) -> List[Tuple[int, ...]]:
    """Halve the last ``n_spatial`` axes at each level (bioio pyramid rule)."""
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")
    ndim = len(level0_shape)
    spatial_start = ndim - min(n_spatial, ndim)
    spatial_indices = list(range(spatial_start, ndim))
    levels: List[Tuple[int, ...]] = []
    for lev in range(num_levels):
        factor = 2**lev
        shape = [int(x) for x in level0_shape]
        for i in spatial_indices:
            shape[i] = max(1, int(level0_shape[i]) // factor)
        t = tuple(shape)
        if levels and t == levels[-1]:
            break
        levels.append(t)
    return levels


def _chunks_tuple_from_user(spec: str, ndim: int) -> Tuple[int, ...]:
    parts = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not parts:
        raise ValueError("empty --chunks")
    if ndim >= 3 and len(parts) == 3:
        pad = ndim - 3
        return (1,) * pad + tuple(parts)
    if len(parts) == ndim:
        return tuple(parts)
    raise ValueError(
        f"--chunks: need {ndim} integers (full shape) or three values Z,Y,X; got {len(parts)}"
    )


def _per_level_chunks_explicit(
    base_chunk: Sequence[int],
    level_shapes: List[Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
    return [
        tuple(min(int(c), int(s)) for c, s in zip(base_chunk, shp))
        for shp in level_shapes
    ]


def _max_on_disk_chunk_bytes(
    chunk_shapes_per_level: Sequence[Sequence[int]],
    dtype: np.dtype,
) -> int:
    """Largest decoded Zarr chunk in bytes across pyramid levels."""
    itemsize = int(np.dtype(dtype).itemsize)
    if not chunk_shapes_per_level:
        return 16 << 20
    return max(int(np.prod(c)) * itemsize for c in chunk_shapes_per_level)


def _dask_array_chunk_size_limit(
    chunk_shapes_per_level: Sequence[Sequence[int]],
    dtype: np.dtype,
) -> int:
    """``dask.config`` ``array.chunk-size`` for safe aligned Zarr writes."""
    nbytes = _max_on_disk_chunk_bytes(chunk_shapes_per_level, dtype)
    # BioIO / Dask warn when below the largest on-disk chunk (~16 MiB viz default).
    return max(16 << 20, nbytes + 4096)


def _chunk_shapes_for_levels(
    level_shapes: List[Tuple[int, ...]],
    dtype: np.dtype,
    memory_target: int,
    *,
    chunks: Optional[str],
    level0_shape: Tuple[int, ...],
    viz_level_shapes: Optional[List[Tuple[int, ...]]] = None,
) -> List[Tuple[int, ...]]:
    """Per-pyramid-level Zarr chunk shapes (always level-aware)."""
    from bioio_ome_zarr.writers.utils import multiscale_chunk_size_from_memory_target

    if chunks is not None:
        base_chunk = _chunks_tuple_from_user(chunks, len(level0_shape))
        return _per_level_chunks_explicit(base_chunk, level_shapes)

    if viz_level_shapes is not None and list(level_shapes) == [
        tuple(int(x) for x in s) for s in viz_level_shapes
    ]:
        # BioIO viz preset: ~16 MiB target, recomputed per pyramid level.
        return [
            tuple(
                int(x)
                for x in multiscale_chunk_size_from_memory_target(
                    [shp], dtype, memory_target
                )[0]
            )
            for shp in level_shapes
        ]

    chunks_per_level = multiscale_chunk_size_from_memory_target(
        level_shapes,
        dtype,
        memory_target,
    )
    return [tuple(int(x) for x in c) for c in chunks_per_level]


def _ngff_axis_name(bioio_dim: str) -> str:
    key = str(bioio_dim).strip()
    if key in _BIOIO_TO_NGFF_AXIS:
        return _BIOIO_TO_NGFF_AXIS[key]
    low = key.lower()
    if low in _NGFF_AXIS_TYPE:
        return low
    if len(low) == 1:
        return low
    raise ValueError(f"Unsupported BioIO dimension name: {bioio_dim!r}")


def _scale_key_for_dim(bioio_dim: str) -> Optional[str]:
    key = str(bioio_dim).strip().upper()
    if key in _SCALE_KEYS:
        return key
    return None


def _safe_bioimage_scale(img: Any) -> Any:
    from bioio_base.types import PhysicalPixelSizes, Scale

    try:
        return img.scale
    except (AttributeError, TypeError):
        pass
    try:
        pps = img.physical_pixel_sizes
        if isinstance(pps, PhysicalPixelSizes):
            return Scale(T=None, C=None, Z=pps.Z, Y=pps.Y, X=pps.X)
    except (AttributeError, TypeError):
        pass
    return Scale(T=None, C=None, Z=None, Y=None, X=None)


def _ome_spatial_scales_microns(path: Path) -> Tuple[Dict[str, float], bool]:
    """Parse OME-XML from a TIFF when BioIO did not attach physical sizes."""
    try:
        import tifffile as tf

        from holvesseg._ome_reader import _ome_axis_scales_microns

        with tf.TiffFile(path) as tif:
            xml = tif.ome_metadata
        if not xml:
            return {}, False
        return _ome_axis_scales_microns(xml)
    except Exception:
        return {}, False


def _pint_unit_to_ngff(unit: Any) -> Optional[str]:
    if unit is None:
        return None
    text = str(unit).strip().lower().replace("µ", "u")
    aliases = {
        "micrometer": "micrometer",
        "micrometre": "micrometer",
        "um": "micrometer",
        "millimeter": "millimeter",
        "mm": "millimeter",
        "nanometer": "nanometer",
        "nm": "nanometer",
        "second": "second",
        "s": "second",
        "millisecond": "millisecond",
        "ms": "millisecond",
    }
    return aliases.get(text, text or None)


def _dimension_property_unit(img: Any, scale_key: str) -> Optional[str]:
    try:
        props = img.dimension_properties
        prop = getattr(props, scale_key, None)
        if prop is not None and getattr(prop, "unit", None) is not None:
            return _pint_unit_to_ngff(prop.unit)
    except (AttributeError, TypeError):
        pass
    return None


def _image_name(img: Any, input_path: Path, override: Optional[str]) -> str:
    if override:
        return override.strip()
    try:
        ome = img.ome_metadata
        if ome is not None and getattr(ome, "images", None):
            name = getattr(ome.images[0], "name", None)
            if name:
                return str(name)
    except (AttributeError, TypeError, IndexError, NotImplementedError):
        pass
    return input_path.stem


def _channel_count(dim_names: Sequence[str], shape: Sequence[int]) -> int:
    for i, dim in enumerate(dim_names):
        if _ngff_axis_name(dim) == "c":
            return max(int(shape[i]), 1)
    return 1


def _channels_for_writer(
    img: Any,
    dim_names: Sequence[str],
    shape: Sequence[int],
) -> List[Any]:
    from bioio_ome_zarr.writers import Channel

    n_ch = _channel_count(dim_names, shape)
    labels: List[str] = []
    try:
        labels = [str(x) for x in img.channel_names]
    except (AttributeError, TypeError, ValueError, KeyError):
        labels = []
    if len(labels) < n_ch:
        labels.extend(f"C:{i}" for i in range(len(labels), n_ch))
    labels = labels[:n_ch]
    return [
        Channel(
            label=label,
            color=_CHANNEL_COLORS_HEX[i % len(_CHANNEL_COLORS_HEX)],
        )
        for i, label in enumerate(labels)
    ]


def ngff_writer_kwargs_from_bioimage(
    img: Any,
    dim_names: Sequence[str],
    data_shape: Sequence[int],
    input_path: Path,
    *,
    image_name: Optional[str] = None,
    voxel_size_zyx: Optional[Tuple[float, float, float]] = None,
    voxel_unit: str = "micrometer",
) -> Dict[str, Any]:
    """Build OMEZarrWriter metadata kwargs from a :class:`bioio.BioImage`."""
    scale = _safe_bioimage_scale(img)
    ome_scales, ome_explicit = _ome_spatial_scales_microns(input_path)

    axes_names: List[str] = []
    axes_types: List[str] = []
    axes_units: List[Optional[str]] = []
    physical_pixel_size: List[float] = []

    missing_spatial = False
    for dim in dim_names:
        ngff = _ngff_axis_name(dim)
        axes_names.append(ngff)
        axes_types.append(_NGFF_AXIS_TYPE[ngff])

        scale_key = _scale_key_for_dim(dim)
        value = 1.0
        unit: Optional[str] = None
        if scale_key is not None:
            raw = getattr(scale, scale_key, None)
            if raw is not None and float(raw) > 0:
                value = float(raw)
            elif scale_key in ome_scales and ome_scales[scale_key] > 0:
                value = float(ome_scales[scale_key])
                unit = "micrometer"
            elif (
                voxel_size_zyx is not None
                and scale_key in ("Z", "Y", "X")
            ):
                idx = {"Z": 0, "Y": 1, "X": 2}[scale_key]
                value = float(voxel_size_zyx[idx])
                unit = voxel_unit
            elif ngff in ("z", "y", "x"):
                missing_spatial = True

            prop_unit = _dimension_property_unit(img, scale_key)
            if prop_unit is not None:
                unit = prop_unit

        if unit is None and ngff in ("z", "y", "x") and ome_explicit:
            unit = "micrometer"
        if unit is None and ngff in ("z", "y", "x") and voxel_size_zyx is not None:
            unit = voxel_unit

        axes_units.append(unit)
        physical_pixel_size.append(value)

    if missing_spatial and voxel_size_zyx is None:
        print(
            "Note: spatial voxel size not found in source metadata; "
            "using 1.0 per axis. Pass --voxel-size Z,Y,X to set physical scale.",
            file=sys.stderr,
        )

    return {
        "image_name": _image_name(img, input_path, image_name),
        "axes_names": axes_names,
        "axes_types": axes_types,
        "axes_units": axes_units,
        "physical_pixel_size": physical_pixel_size,
        "channels": _channels_for_writer(img, dim_names, data_shape),
        "creator_info": {
            "name": "holvesseg",
            "version": __import__("holvesseg", fromlist=["__version__"]).__version__,
            "source": str(input_path),
        },
    }


def convert_image_to_omezarr(
    input_path: Path,
    output_path: Path,
    *,
    downsample_z: bool = True,
    zarr_format: int = 2,
    num_levels: int = 3,
    chunk_target_mib: float = 16.0,
    chunks: Optional[str] = None,
    image_name: Optional[str] = None,
    voxel_size_zyx: Optional[Tuple[float, float, float]] = None,
    voxel_unit: str = "micrometer",
) -> None:
    """Read any BioIO-supported image and write a multiscale OME-Zarr store."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    import dask
    from bioio import BioImage
    from bioio_ome_zarr.writers import OMEZarrWriter, get_default_config_for_viz

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        img = BioImage(input_path)

    img_xr = img.xarray_dask_data.squeeze(drop=True)
    dim_names = tuple(str(d) for d in img_xr.dims)
    level0_shape: Tuple[int, ...] = tuple(int(x) for x in img_xr.data.shape)

    meta_kw = ngff_writer_kwargs_from_bioimage(
        img,
        dim_names,
        level0_shape,
        input_path,
        image_name=image_name,
        voxel_size_zyx=voxel_size_zyx,
        voxel_unit=voxel_unit,
    )

    viz_config = get_default_config_for_viz(img_xr.data, downsample_z=downsample_z)
    dtype = np.dtype(viz_config["dtype"])
    n_spatial = 3 if downsample_z else 2

    level_shapes = _pyramid_level_shapes(
        level0_shape, n_spatial=n_spatial, num_levels=num_levels
    )
    if len(level_shapes) < num_levels:
        print(
            f"Note: pyramid stopped at {len(level_shapes)} level(s) "
            f"(shape plateau); requested {num_levels}.",
            file=sys.stderr,
        )

    memory_target = max(
        int(dtype.itemsize),
        int(np.ceil(float(chunk_target_mib) * 1024 * 1024)),
    )

    use_viz_shapes = (
        chunks is None
        and num_levels == 3
        and abs(float(chunk_target_mib) - 16.0) < 1e-6
    )
    viz_level_shapes = (
        [tuple(int(x) for x in s) for s in viz_config["level_shapes"]]
        if use_viz_shapes
        else None
    )
    chunk_shapes_per_level = _chunk_shapes_for_levels(
        level_shapes,
        dtype,
        memory_target,
        chunks=chunks,
        level0_shape=level0_shape,
        viz_level_shapes=viz_level_shapes,
    )
    chunk_kw: dict = {"chunk_shape": chunk_shapes_per_level}
    dask_chunk_limit = _dask_array_chunk_size_limit(chunk_shapes_per_level, dtype)

    writer = OMEZarrWriter(
        store=str(output_path),
        zarr_format=zarr_format,
        level_shapes=level_shapes,
        dtype=dtype,
        **meta_kw,
        **chunk_kw,
    )
    writer._initialize()
    with dask.config.set({"array.chunk-size": dask_chunk_limit}):
        writer.write_full_volume(img_xr.data)


# Backwards-compatible alias.
convert_ome_tiff_to_zarr = convert_image_to_omezarr

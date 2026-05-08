"""Convert OME-TIFF to OME-Zarr using BioIO and bioio-ome-zarr (OMEZarrWriter)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _scale_for_dim_names(scale, dim_names: tuple[str, ...]) -> list[float]:
    """Map xarray dimension names to BioImage ``Scale`` (T, C, Z, Y, X); default 1.0."""
    out: list[float] = []
    for dim in dim_names:
        key = dim.upper() if len(dim) == 1 else dim
        val = getattr(scale, key, None)
        if val is None or val == 0:
            val = 1.0
        out.append(float(val))
    return out


def _pyramid_level_shapes(
    level0_shape: Tuple[int, ...],
    *,
    n_spatial: int,
    num_levels: int,
) -> List[Tuple[int, ...]]:
    """Halve the last ``n_spatial`` axes at each level (same rule as bioio ``config._pyramid_level_shapes``).

    Stops early if a further level would repeat the same shape (all spatial axes at 1).
    """
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
    """Parse ``--chunks``: either ``ndim`` comma-separated ints or three spatial ``Z,Y,X`` (leading axes = 1)."""
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


def convert_ome_tiff_to_zarr(
    input_path: Path,
    output_path: Path,
    *,
    downsample_z: bool = True,
    zarr_format: int = 2,
    num_levels: int = 3,
    chunk_target_mib: float = 16.0,
    chunks: Optional[str] = None,
) -> None:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    import dask
    from bioio import BioImage
    from bioio_ome_zarr.writers import OMEZarrWriter, get_default_config_for_viz
    from bioio_ome_zarr.writers.utils import multiscale_chunk_size_from_memory_target

    img = BioImage(input_path)
    img_xr = img.xarray_dask_data.squeeze(drop=True)
    physical_pixel_size = _scale_for_dim_names(img.scale, tuple(img_xr.dims))

    viz_config = get_default_config_for_viz(img_xr.data, downsample_z=downsample_z)
    dtype = np.dtype(viz_config["dtype"])
    n_spatial = 3 if downsample_z else 2
    level0_shape: Tuple[int, ...] = tuple(int(x) for x in img_xr.data.shape)

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

    if chunks is not None:
        base_chunk = _chunks_tuple_from_user(chunks, len(level0_shape))
        chunk_shapes_per_level = _per_level_chunks_explicit(base_chunk, level_shapes)
        chunk_bytes = max(
            int(np.prod(c)) * dtype.itemsize for c in chunk_shapes_per_level
        )
        chunk_kw: dict = {"chunk_shape": chunk_shapes_per_level}
    elif num_levels == 3 and abs(float(chunk_target_mib) - 16.0) < 1e-6:
        # Match bioio ``get_default_config_for_viz`` exactly (single chunk from level 0, ~16 MiB).
        level_shapes = viz_config["level_shapes"]
        chunk_kw = {"chunk_shape": viz_config["chunk_shape"]}
        chunk_bytes = int(np.prod(viz_config["chunk_shape"])) * dtype.itemsize
    else:
        chunks_per_level = multiscale_chunk_size_from_memory_target(
            level_shapes,
            dtype,
            memory_target,
        )
        chunk_shapes_per_level = [tuple(int(x) for x in c) for c in chunks_per_level]
        chunk_bytes = max(
            int(np.prod(c)) * dtype.itemsize for c in chunk_shapes_per_level
        )
        chunk_kw = {"chunk_shape": chunk_shapes_per_level}

    min_chunk = max(16 * 1024 * 1024, int(chunk_bytes))

    writer = OMEZarrWriter(
        store=str(output_path),
        zarr_format=zarr_format,
        level_shapes=level_shapes,
        dtype=dtype,
        physical_pixel_size=physical_pixel_size,
        **chunk_kw,
    )
    with dask.config.set({"array.chunk-size": min_chunk}):
        writer.write_full_volume(img_xr.data)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Requires Python >= 3.11 and: pip install bioio bioio-ome-tiff 'bioio-ome-zarr>=3'. "
            "From a napari env on Python 3.10, use the repo helper: "
            "./run_ome_tiff_to_zarr_bioio.sh … (conda env ome-zarr-bioio)."
        ),
    )
    p.add_argument(
        "input",
        type=Path,
        help="Path to the OME-TIFF file (.ome.tif / .ome.tiff / .tif with OME metadata).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Zarr directory (default: <input_stem>.ome.zarr next to the input file).",
    )
    p.add_argument(
        "--no-downsample-z",
        action="store_true",
        help="Do not add Z downsampling levels in the visualization pyramid.",
    )
    p.add_argument(
        "--levels",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Number of pyramid resolutions (level 0 = full resolution, each next level "
            "halves Y/X or Z/Y/X with --no-downsample-z off). Default: 3 (same as bioio "
            "``get_default_config_for_viz``). Stops early if dimensions reach 1."
        ),
    )
    p.add_argument(
        "--chunk-target-mib",
        type=float,
        default=16.0,
        metavar="MIB",
        help=(
            "Target decoded size per on-disk chunk in MiB (BioIO grows X→Y→Z within this "
            "budget). Default: 16 (same as bioio). Ignored if --chunks is set. With "
            "--levels 3 and default 16, chunk layout matches ``get_default_config_for_viz``; "
            "otherwise chunks are derived per resolution level."
        ),
    )
    p.add_argument(
        "--chunks",
        type=str,
        default=None,
        metavar="Z,Y,X",
        help=(
            "Fixed chunk shape: three integers Z,Y,X for the trailing spatial axes "
            "(leading axes default to 1), or one comma-separated value per array dimension. "
            "Overrides --chunk-target-mib. Each pyramid level uses these sizes capped by "
            "that level's shape."
        ),
    )
    p.add_argument(
        "--zarr-format",
        type=int,
        choices=(2, 3),
        default=2,
        help=(
            "Zarr format: 2 (NGFF 0.4, .zgroup — works with napari / napari-ome-zarr) "
            "or 3 (NGFF 0.5, zarr.json — needs a Zarr v3-aware viewer). Default: 2."
        ),
    )
    args = p.parse_args(argv)

    inp = args.input
    out = args.output
    if out is None:
        out = inp.parent / f"{inp.stem}.ome.zarr"

    try:
        if args.levels < 1 or args.levels > 32:
            print(
                "Error: --levels must be between 1 and 32.",
                file=sys.stderr,
            )
            return 2
        if args.chunk_target_mib <= 0:
            print("Error: --chunk-target-mib must be positive.", file=sys.stderr)
            return 2
        convert_ome_tiff_to_zarr(
            inp,
            out,
            downsample_z=not args.no_downsample_z,
            zarr_format=args.zarr_format,
            num_levels=args.levels,
            chunk_target_mib=args.chunk_target_mib,
            chunks=args.chunks,
        )
    except ImportError as e:
        print(e, file=sys.stderr)
        print(
            "Install: pip install bioio bioio-ome-tiff 'bioio-ome-zarr>=3' "
            "(Python >= 3.11 required for the writer; conda env napari-dev is often 3.10).",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

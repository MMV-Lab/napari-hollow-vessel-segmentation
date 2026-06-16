"""CLI: convert BioIO-readable images to multiscale OME-Zarr."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

from regiongrow._bioio_to_omezarr import convert_image_to_omezarr


def _parse_voxel_size(spec: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("expected three comma-separated values: Z,Y,X")
    if any(v <= 0 for v in parts):
        raise ValueError("voxel sizes must be positive")
    return (parts[0], parts[1], parts[2])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Convert microscopy images to multiscale OME-Zarr using BioIO. "
            "Supports any format with an installed BioIO reader plugin "
            "(OME-TIFF, TIFF, PNG, JPEG, OME-Zarr, and more)."
        ),
    )
    p.add_argument(
        "input",
        type=Path,
        help="Input image path (any BioIO-supported format).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Zarr directory (default: <input_stem>.ome.zarr next to the input).",
    )
    p.add_argument(
        "--image-name",
        type=str,
        default=None,
        help="NGFF image name (default: OME name or input file stem).",
    )
    p.add_argument(
        "--voxel-size",
        type=str,
        default=None,
        metavar="Z,Y,X",
        help=(
            "Physical voxel size for Z, Y, X when missing from source metadata "
            "(same units as --voxel-unit). Example: 2.0,0.65,0.65"
        ),
    )
    p.add_argument(
        "--voxel-unit",
        type=str,
        default="micrometer",
        help="NGFF unit for --voxel-size spatial axes (default: micrometer).",
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
            "Number of pyramid resolutions (level 0 = full resolution). "
            "Default: 3. Stops early if dimensions reach 1."
        ),
    )
    p.add_argument(
        "--chunk-target-mib",
        type=float,
        default=16.0,
        metavar="MIB",
        help="Target decoded chunk size in MiB (default: 16). Ignored if --chunks is set.",
    )
    p.add_argument(
        "--chunks",
        type=str,
        default=None,
        metavar="Z,Y,X",
        help=(
            "Fixed chunk shape: Z,Y,X for trailing spatial axes, or one value per "
            "array dimension. Overrides --chunk-target-mib."
        ),
    )
    p.add_argument(
        "--zarr-format",
        type=int,
        choices=(2, 3),
        default=2,
        help=(
            "Zarr format: 2 (NGFF 0.4, napari-compatible) or 3 (NGFF 0.5). Default: 2."
        ),
    )
    args = p.parse_args(argv)

    inp = args.input
    out = args.output or inp.parent / f"{inp.stem}.ome.zarr"

    voxel_size: Optional[Tuple[float, float, float]] = None
    if args.voxel_size is not None:
        try:
            voxel_size = _parse_voxel_size(args.voxel_size)
        except ValueError as e:
            print(f"Error: --voxel-size {e}", file=sys.stderr)
            return 2

    try:
        if args.levels < 1 or args.levels > 32:
            print("Error: --levels must be between 1 and 32.", file=sys.stderr)
            return 2
        if args.chunk_target_mib <= 0:
            print("Error: --chunk-target-mib must be positive.", file=sys.stderr)
            return 2
        convert_image_to_omezarr(
            inp,
            out,
            downsample_z=not args.no_downsample_z,
            zarr_format=args.zarr_format,
            num_levels=args.levels,
            chunk_target_mib=args.chunk_target_mib,
            chunks=args.chunks,
            image_name=args.image_name,
            voxel_size_zyx=voxel_size,
            voxel_unit=str(args.voxel_unit).strip() or "micrometer",
        )
    except ImportError as e:
        print(e, file=sys.stderr)
        print("Missing dependency — reinstall: pip install -e .", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Preprocess an OME-TIFF on disk (mean downsample + contrast stretch), then convert to Zarr separately.

Uses the same parameters as ``holvesseg-preprocess-zarr`` (except ``--finest-only``, which
is ignored for a single-resolution TIFF). Requires ``tifffile`` (already a core dependency).

Example::

    holvesseg-preprocess-ome-tiff in.ome.tif out.ome.tif --stretch
    holvesseg-convert-to-ome-zarr out.ome.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from holvesseg._preprocess_ome_tiff import run_preprocess_ome_tiff_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Load an OME-TIFF as a single-channel (Z, Y, X) volume, optionally mean-downsample "
            "in Z and XY, optionally contrast-stretch, and write a new OME-TIFF. "
            "Convert to OME-Zarr afterwards (e.g. holvesseg-convert-to-ome-zarr)."
        )
    )
    p.add_argument("input", type=Path, help="Input OME-TIFF (.ome.tif / .ome.tiff)")
    p.add_argument(
        "output",
        type=Path,
        help="Output OME-TIFF path (must not exist unless --overwrite)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    p.add_argument("--downsample-z", type=int, default=1)
    p.add_argument("--downsample-xy", type=int, default=1)
    p.add_argument("--no-downsample", action="store_true")
    p.add_argument("--stretch", action="store_true", help="Apply contrast stretch (default off)")
    p.add_argument("--stretch-mode", choices=("percentile", "fixed"), default="percentile")
    p.add_argument("--percentile-low", type=float, default=1.0)
    p.add_argument("--percentile-high", type=float, default=99.0)
    p.add_argument("--fixed-min", type=float, default=0.0)
    p.add_argument("--fixed-max", type=float, default=255.0)
    p.add_argument("--out-dtype", choices=("uint8", "uint16"), default="uint8")
    p.add_argument(
        "--finest-only",
        action="store_true",
        help="Ignored for TIFF (single resolution); kept for CLI parity with preprocess-zarr.",
    )
    p.add_argument(
        "--compression",
        choices=("none", "zlib", "lzw", "zstd"),
        default="zlib",
        help="Lossless TIFF compression (default: zlib). Use 'none' only if you need raw strips.",
    )
    p.add_argument(
        "--compression-level",
        type=int,
        default=None,
        metavar="N",
        help="zlib/deflate: 0–9 (default 6). zstd: 1–22 (default 3). Ignored for none/lzw.",
    )
    p.add_argument(
        "--no-predictor",
        action="store_true",
        help="Disable horizontal differencing (slightly larger files; rarely needed).",
    )
    p.add_argument(
        "--tile",
        type=str,
        default=None,
        metavar="Z,Y,X",
        help="Optional 3D tile shape for writes (e.g. 128,128,128); can improve compression and seek.",
    )
    args = p.parse_args(argv)

    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    in_low = str(inp).lower()
    if in_low.endswith(".ome.zarr") or in_low.endswith(".zarr"):
        print(
            "This command preprocesses OME-TIFF files only. For OME-Zarr input/output use:\n"
            "  holvesseg-preprocess-zarr …\n"
            f"(got: {inp})",
            file=sys.stderr,
        )
        return 1
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1
    if out.exists() and not args.overwrite:
        print(
            f"Output already exists: {out}\n"
            "Pass --overwrite to replace it.",
            file=sys.stderr,
        )
        return 1
    if out.exists() and args.overwrite:
        out.unlink()

    apply_ds = not args.no_downsample and (args.downsample_z > 1 or args.downsample_xy > 1)
    tile_zyx = None
    if args.tile:
        parts = [int(x.strip()) for x in args.tile.split(",") if x.strip()]
        if len(parts) != 3:
            print("--tile must be three integers: Z,Y,X", file=sys.stderr)
            return 1
        tile_zyx = (parts[0], parts[1], parts[2])

    clevel = args.compression_level
    if args.compression == "none":
        clevel = None
    elif clevel is None:
        if args.compression == "zstd":
            clevel = 3
        elif args.compression == "zlib":
            clevel = 6

    try:
        meta = run_preprocess_ome_tiff_pipeline(
            inp,
            out,
            apply_downsample=apply_ds,
            downsample_z=args.downsample_z,
            downsample_xy=args.downsample_xy,
            apply_stretch=args.stretch,
            stretch_mode=args.stretch_mode,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            fixed_background=args.fixed_min,
            fixed_vessel_max=args.fixed_max,
            out_dtype=args.out_dtype,
            finest_only=args.finest_only,
            compression=args.compression,
            compression_level=clevel,
            predictor=not args.no_predictor,
            tile_zyx=tile_zyx,
        )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    print(meta)
    print("Done:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

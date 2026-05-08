"""Chunk-wise preprocessing from one OME-Zarr store to another (disk streaming).

Requires optional deps in the **same** env: ``pip install -e ".[zarr-cli]"``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from regiongrow._preprocessing_zarr import run_preprocess_zarr_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    try:
        import zarr  # noqa: F401
    except ImportError as exc:
        print(
            "Missing zarr (and possibly other zarr-cli extras).\n"
            'Install into this same Python environment:\n  pip install -e ".[zarr-cli]"',
            file=sys.stderr,
        )
        print(f"  {sys.executable}\n  {exc!r}", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(
        description=(
            "Preprocess the finest NGFF resolution of an OME-Zarr image (optional mean "
            "downsample + optional contrast stretch), then rebuild all coarser pyramid "
            "levels from that result so the output keeps the same multiscale paths as the "
            "input. Use --finest-only to write a single-resolution store (legacy)."
        )
    )
    p.add_argument("input", type=Path, help="Input .ome.zarr directory")
    p.add_argument(
        "output",
        type=Path,
        help="Output .ome.zarr directory (must not exist unless --overwrite)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and replace the output directory if it already exists.",
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
        help="Write only the finest level (omit coarser NGFF datasets).",
    )
    args = p.parse_args(argv)

    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not inp.is_dir():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1
    if out.exists() and not args.overwrite:
        print(
            f"Output already exists: {out}\n"
            "Pass --overwrite to remove it and write a new store.",
            file=sys.stderr,
        )
        return 1

    apply_ds = not args.no_downsample and (args.downsample_z > 1 or args.downsample_xy > 1)
    try:
        meta = run_preprocess_zarr_pipeline(
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
        )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    print(meta)
    print("Done:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

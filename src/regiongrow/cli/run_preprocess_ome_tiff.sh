#!/usr/bin/env bash
set -euo pipefail
# Run OME-TIFF preprocessing without needing ``pip install -e .`` (uses PYTHONPATH).
# Optional: conda env ``ome-zarr-bioio`` if present; otherwise uses current ``python``.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="${SRC}${PYTHONPATH:+:${PYTHONPATH}}"
if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -q '^ome-zarr-bioio[[:space:]]'; then
  exec conda run --no-capture-output -n ome-zarr-bioio python -m regiongrow.cli.preprocess_ome_tiff "$@"
else
  exec python -m regiongrow.cli.preprocess_ome_tiff "$@"
fi

#!/usr/bin/env bash
# Dev helper: run conversion from a clone without pip install (sets PYTHONPATH).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="${SRC}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m regiongrow.cli.to_ome_zarr "$@"

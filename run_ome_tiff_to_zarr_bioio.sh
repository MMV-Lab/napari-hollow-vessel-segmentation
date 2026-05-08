#!/usr/bin/env bash
# Launcher from repo root; implementation lives under src/regiongrow/cli/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/src/regiongrow/cli/run_ome_tiff_to_zarr_bioio.sh" "$@"

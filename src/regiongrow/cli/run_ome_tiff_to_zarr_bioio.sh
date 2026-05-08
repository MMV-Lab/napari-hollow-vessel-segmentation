#!/usr/bin/env bash
set -euo pipefail
# Runs TIFF→OME-Zarr conversion (BioIO + OMEZarrWriter) in conda env ``ome-zarr-bioio``.
# Create/update env: conda create -n ome-zarr-bioio python=3.12 -y
#                   conda run -n ome-zarr-bioio pip install 'bioio>=3' 'bioio-ome-tiff>=1' 'bioio-ome-zarr>=3'
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="${SRC}${PYTHONPATH:+:${PYTHONPATH}}"
exec conda run --no-capture-output -n ome-zarr-bioio python -m regiongrow.cli.ome_tiff_to_zarr_bioio "$@"

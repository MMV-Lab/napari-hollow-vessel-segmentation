"""Deprecated module path; use :mod:`regiongrow.cli.to_ome_zarr`."""

from regiongrow._bioio_to_omezarr import convert_image_to_omezarr, convert_ome_tiff_to_zarr
from regiongrow.cli.to_ome_zarr import main

__all__ = ["convert_image_to_omezarr", "convert_ome_tiff_to_zarr", "main"]

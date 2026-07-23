"""Deprecated module path; use :mod:`holvesseg.cli.to_ome_zarr`."""

from holvesseg._bioio_to_omezarr import convert_image_to_omezarr, convert_ome_tiff_to_zarr
from holvesseg.cli.to_ome_zarr import main

__all__ = ["convert_image_to_omezarr", "convert_ome_tiff_to_zarr", "main"]

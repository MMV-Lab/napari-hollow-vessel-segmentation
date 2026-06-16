"""Zarr v2/v3 API shims used across the package."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import zarr

try:
    from numcodecs import Blosc
except ImportError:  # pragma: no cover
    Blosc = None  # type: ignore[misc, assignment]


def zarr_format_of(node: Any) -> int:
    """Return ``2`` or ``3`` from Zarr metadata (default ``2``)."""
    zfmt = int(getattr(getattr(node, "metadata", None), "zarr_format", None) or 2)
    return 2 if zfmt not in (2, 3) else zfmt


def zarr_array_keys(group: zarr.Group) -> List[str]:
    """Child array names on *group* (``array_keys`` with ``items()`` fallback)."""
    if hasattr(group, "array_keys"):
        return [str(k) for k in group.array_keys()]
    return [str(k) for k, v in group.items() if isinstance(v, zarr.Array)]


def zarr_subgroup_keys(group: zarr.Group) -> List[str]:
    """Child group names on *group* (``group_keys`` with ``items()`` fallback)."""
    if hasattr(group, "group_keys"):
        return [str(k) for k in group.group_keys()]
    return [str(k) for k, v in group.items() if isinstance(v, zarr.Group)]


def zarr_v3_blosc_kwargs() -> Dict[str, Any]:
    """Default Blosc codec kwargs for Zarr v3 arrays."""
    try:
        from zarr.codecs import BloscCodec, BloscShuffle

        return {
            "compressors": [
                BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)
            ]
        }
    except ImportError:  # pragma: no cover
        return {}


def zarr_numcodecs_blosc_like_input(arr_in: zarr.Array) -> Optional[Any]:
    """Best-effort numcodecs Blosc matching *arr_in* (Zarr v2 arrays only)."""
    if Blosc is None or zarr_format_of(arr_in) != 2:
        return None
    c: Any = None
    try:
        c = arr_in.compressor
    except (AttributeError, TypeError):
        return None
    if c is not None and hasattr(c, "cname"):
        try:
            sh = getattr(c, "shuffle", Blosc.BITSHUFFLE)
            return Blosc(
                cname=str(c.cname),
                clevel=int(getattr(c, "clevel", 3)),
                shuffle=sh,
            )
        except (TypeError, ValueError):
            pass
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def zarr_output_codec_kwargs(
    ref: zarr.Array,
    *,
    zfmt: Optional[int] = None,
    compressor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return ``compressor`` / ``compressors`` kwargs for ``zarr_create_array``."""
    fmt = zarr_format_of(ref) if zfmt is None else int(zfmt)
    if compressor is not None:
        if fmt == 3:
            if isinstance(compressor, (list, tuple)):
                return {"compressors": list(compressor)}
            return {"compressors": compressor}
        return {"compressor": compressor}
    if fmt == 3:
        return zarr_v3_blosc_kwargs()
    comp = zarr_numcodecs_blosc_like_input(ref)
    if comp is not None:
        return {"compressor": comp}
    return {}


def zarr_create_array(
    group: zarr.Group,
    name: str,
    *,
    shape: Tuple[int, ...],
    chunks: Tuple[int, ...],
    dtype: Any,
    overwrite: bool = False,
    **codec_kw: Any,
) -> zarr.Array:
    """Create a child array on *group* (``create_array`` on Zarr v3, else ``create_dataset``)."""
    if overwrite and name in group:
        del group[name]
    shape_n = tuple(int(x) for x in shape)
    chunks_n = tuple(max(1, int(c)) for c in chunks)
    if hasattr(group, "create_array"):
        return group.create_array(
            name,
            shape=shape_n,
            chunks=chunks_n,
            dtype=dtype,
            overwrite=overwrite,
            **codec_kw,
        )
    if hasattr(group, "create_dataset"):
        return group.create_dataset(
            name,
            shape=shape_n,
            chunks=chunks_n,
            dtype=dtype,
            overwrite=overwrite,
            **codec_kw,
        )
    raise RuntimeError("Zarr Group has neither create_array nor create_dataset")

"""Chunk-wise preprocessing for Zarr-backed volumes (no full-RAM materialize).

Supports percentile contrast limits (same contract as in-memory
``contrast_range_from_crop`` / ``stretch_contrast``) and integer mean downsample.
3D non-local means is **not** implemented on Zarr (use in-memory preprocessing for denoise).

By default, writes a **full NGFF pyramid** (same ``datasets[].path`` and shapes as the
input): preprocess the **finest** level, then rebuild coarser levels by block-mean when
dimensions divide evenly, otherwise ``skimage.transform.resize`` (order=1) per chunk.
Pass ``finest_only=True`` for a single-resolution output (legacy).

Output arrays use Blosc (zstd) when ``numcodecs`` is available, matching the finest-level
input compressor when possible.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import zarr

from regiongrow._preprocessing import contrast_range_from_crop, stretch_contrast
from regiongrow._zarr_compat import (
    zarr_create_array,
    zarr_format_of,
    zarr_numcodecs_blosc_like_input,
    zarr_output_codec_kwargs,
)

try:
    from numcodecs import Blosc
except ImportError:  # pragma: no cover
    Blosc = None  # type: ignore[misc, assignment]


def _read_multiscales_datasets(group: zarr.Group) -> List[Dict[str, Any]]:
    attrs = dict(group.attrs)
    if "multiscales" in attrs:
        return list(attrs["multiscales"][0]["datasets"])
    ome = attrs.get("ome")
    if isinstance(ome, dict) and "multiscales" in ome:
        return list(ome["multiscales"][0]["datasets"])
    raise ValueError(
        "Not an OME-Zarr multiscale group: missing multiscales metadata "
        "(expected top-level 'multiscales' or attrs['ome']['multiscales'])."
    )


def open_finest_zarr_array(store: str | Path) -> Tuple[zarr.Array, zarr.Group, List[Dict[str, Any]]]:
    """Return the finest-resolution ``zarr.Array``, its group, and multiscale *datasets* metadata."""
    g = zarr.open_group(str(store), mode="r")
    ds = _read_multiscales_datasets(g)
    path = ds[0]["path"]
    node = g[path]
    if not isinstance(node, zarr.Array):
        raise TypeError(f"Expected zarr.Array at {path!r}, got {type(node)}")
    return node, g, ds


def _chunk_slices_3d(arr: zarr.Array):
    z0, y0, x0 = arr.chunks
    Z, Y, X = arr.shape
    for zs in range(0, Z, z0):
        ze = min(zs + z0, Z)
        for ys in range(0, Y, y0):
            ye = min(ys + y0, Y)
            for xs in range(0, X, x0):
                xe = min(xs + x0, X)
                yield (slice(zs, ze), slice(ys, ye), slice(xs, xe))


def _histogram_uint_n(arr: zarr.Array) -> np.ndarray:
    """Merge per-chunk histograms for uint8 / uint16."""
    dt = arr.dtype
    if not np.issubdtype(dt, np.integer):
        raise TypeError(f"Histogram path requires integer dtype, got {dt}")
    nbin = 256 if np.iinfo(dt).bits == 8 else int(np.iinfo(dt).max - np.iinfo(dt).min + 1)
    counts = np.zeros(nbin, dtype=np.int64)
    for sl in _chunk_slices_3d(arr):
        block = np.asarray(arr[sl], dtype=np.int64).ravel()
        bc = np.bincount(block, minlength=nbin)
        counts[: len(bc)] += bc.astype(np.int64, copy=False)
    return counts


def _percentile_from_hist(counts: np.ndarray, p_low: float, p_high: float) -> Tuple[float, float]:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0, 1.0
    cdf = np.cumsum(counts, dtype=np.float64) / total
    idx = np.arange(counts.size, dtype=np.float64)
    lo = float(np.interp(p_low / 100.0, cdf, idx))
    hi = float(np.interp(p_high / 100.0, cdf, idx))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _percentile_limits_like_in_memory(
    arr: zarr.Array, p_low: float, p_high: float, *, max_samples: int = 12_000_000
) -> Tuple[float, float]:
    """Same contract as :func:`regiongrow._preprocessing.contrast_range_from_crop` (``np.percentile``)."""
    parts: List[np.ndarray] = []
    n = 0
    for sl in _chunk_slices_3d(arr):
        v = np.asarray(arr[sl]).ravel()
        parts.append(v)
        n += v.size
        if n >= max_samples:
            break
    if not parts:
        return 0.0, 1.0
    flat = np.concatenate(parts)
    if flat.size > max_samples:
        rng = np.random.default_rng(42)
        flat = rng.choice(flat, size=max_samples, replace=False)
    return contrast_range_from_crop(flat, float(p_low), float(p_high))


def _stretch_block(block: np.ndarray, lo: float, hi: float, out_dtype_str: str) -> np.ndarray:
    """Match :func:`regiongrow._preprocessing.stretch_contrast` (in-memory preprocessing / widget)."""
    out, _ = stretch_contrast(np.asarray(block), lo, hi, out_dtype=out_dtype_str)
    return out


def _output_blosc_compressor(arr_in: zarr.Array) -> Optional[Any]:
    """Match input Blosc settings when possible; else zstd/clevel=3/bitshuffle (v2 only)."""
    return zarr_numcodecs_blosc_like_input(arr_in)


def _finest_scale_from_group(g: zarr.Group) -> Tuple[float, float, float]:
    ds = _read_multiscales_datasets(g)[0]
    ct = ds.get("coordinateTransformations") or [
        {"type": "scale", "scale": [1.0, 1.0, 1.0]},
    ]
    for t in ct:
        if isinstance(t, dict) and t.get("type") == "scale":
            sc = list(t["scale"])
            if len(sc) >= 3:
                return (float(sc[0]), float(sc[1]), float(sc[2]))
    return (1.0, 1.0, 1.0)


def _mean_pool_write(
    arr_in: zarr.Array,
    out_arr: zarr.Array,
    fz: int,
    fxy: int,
) -> None:
    nz, ny, nx = out_arr.shape
    oz0, oy0, ox0 = out_arr.chunks
    for z0 in range(0, nz, oz0):
        z1 = min(z0 + oz0, nz)
        for y0 in range(0, ny, oy0):
            y1 = min(y0 + oy0, ny)
            for x0 in range(0, nx, ox0):
                x1 = min(x0 + ox0, nx)
                zs, ze = z0 * fz, z1 * fz
                ys, ye = y0 * fxy, y1 * fxy
                xs, xe = x0 * fxy, x1 * fxy
                block = np.asarray(arr_in[zs:ze, ys:ye, xs:xe], dtype=np.float64)
                tb = block.reshape(z1 - z0, fz, y1 - y0, fxy, x1 - x0, fxy)
                out_arr[z0:z1, y0:y1, x0:x1] = tb.mean(axis=(1, 3, 5)).astype(
                    out_arr.dtype, copy=False
                )


def _mean_pool_write_xyz(
    arr_in: zarr.Array,
    out_arr: zarr.Array,
    fz: int,
    fy: int,
    fx: int,
) -> None:
    """Mean-pool ``arr_in`` to ``out_arr`` with integer factors ``fz,fy,fx`` per axis."""
    nz, ny, nx = out_arr.shape
    oz0, oy0, ox0 = out_arr.chunks
    for z0 in range(0, nz, oz0):
        z1 = min(z0 + oz0, nz)
        for y0 in range(0, ny, oy0):
            y1 = min(y0 + oy0, ny)
            for x0 in range(0, nx, ox0):
                x1 = min(x0 + ox0, nx)
                zs, ze = z0 * fz, z1 * fz
                ys, ye = y0 * fy, y1 * fy
                xs, xe = x0 * fx, x1 * fx
                block = np.asarray(arr_in[zs:ze, ys:ye, xs:xe], dtype=np.float64)
                tb = block.reshape(z1 - z0, fz, y1 - y0, fy, x1 - x0, fx)
                out_arr[z0:z1, y0:y1, x0:x1] = tb.mean(axis=(1, 3, 5)).astype(
                    out_arr.dtype, copy=False
                )


def _multiscales_layout(group: zarr.Group) -> Literal["top", "ome"]:
    attrs = dict(group.attrs)
    if "multiscales" in attrs:
        return "top"
    ome = attrs.get("ome")
    if isinstance(ome, dict) and "multiscales" in ome:
        return "ome"
    raise ValueError("Not an OME-Zarr multiscale group (missing multiscales).")


def _get_multiscales_image_dict(group: zarr.Group) -> Tuple[Dict[str, Any], Literal["top", "ome"]]:
    """Return a deep-copied multiscales[0] image dict and where it lives (top vs ome)."""
    layout = _multiscales_layout(group)
    attrs = dict(group.attrs)
    if layout == "top":
        src = attrs["multiscales"][0]
    else:
        src = attrs["ome"]["multiscales"][0]
    return copy.deepcopy(src), layout


def _scale_from_dataset_entry(ds_entry: Dict[str, Any]) -> Tuple[float, float, float]:
    ct = ds_entry.get("coordinateTransformations") or [
        {"type": "scale", "scale": [1.0, 1.0, 1.0]},
    ]
    for t in ct:
        if isinstance(t, dict) and t.get("type") == "scale":
            sc = list(t["scale"])
            if len(sc) >= 3:
                return (float(sc[0]), float(sc[1]), float(sc[2]))
    return (1.0, 1.0, 1.0)


def _dst_range_to_src_range(lo: int, hi: int, s_src: int, s_dst: int) -> Tuple[int, int]:
    s = lo * s_src // s_dst
    e = hi * s_src // s_dst
    if hi >= s_dst:
        e = s_src
    return s, e


def _fill_coarse_from_fine(fine: zarr.Array, coarse: zarr.Array) -> None:
    """Fill ``coarse`` by downsampling ``fine`` to match ``coarse.shape`` (chunked)."""
    Zs, Ys, Xs = (int(fine.shape[0]), int(fine.shape[1]), int(fine.shape[2]))
    Zo, Yo, Xo = (int(coarse.shape[0]), int(coarse.shape[1]), int(coarse.shape[2]))
    if Zs % Zo == 0 and Ys % Yo == 0 and Xs % Xo == 0:
        fz, fy, fx = Zs // Zo, Ys // Yo, Xs // Xo
        _mean_pool_write_xyz(fine, coarse, fz, fy, fx)
        return
    from skimage.transform import resize

    imax = float(np.iinfo(coarse.dtype).max)
    for zz, yy, xx in _chunk_slices_3d(coarse):
        zs, ze = _dst_range_to_src_range(zz.start, zz.stop, Zs, Zo)
        ys, ye = _dst_range_to_src_range(yy.start, yy.stop, Ys, Yo)
        xs, xe = _dst_range_to_src_range(xx.start, xx.stop, Xs, Xo)
        block = np.asarray(fine[zs:ze, ys:ye, xs:xe], dtype=np.float64)
        out_sz = (zz.stop - zz.start, yy.stop - yy.start, xx.stop - xx.start)
        out = resize(
            block,
            out_sz,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
        out = np.clip(np.round(out), 0.0, imax).astype(coarse.dtype, copy=False)
        coarse[zz, yy, xx] = out


def _set_scale_on_dataset_entry(
    entry: Dict[str, Any],
    shape_level: Tuple[int, int, int],
    finest_shape: Tuple[int, int, int],
    finest_scale: Tuple[float, float, float],
) -> None:
    Zi, Yi, Xi = shape_level
    Z0, Y0, X0 = finest_shape
    sz = finest_scale[0] * (Z0 / max(Zi, 1))
    sy = finest_scale[1] * (Y0 / max(Yi, 1))
    sx = finest_scale[2] * (X0 / max(Xi, 1))
    cts = entry.setdefault("coordinateTransformations", [])
    replaced = False
    for t in cts:
        if isinstance(t, dict) and t.get("type") == "scale":
            t["scale"] = [sz, sy, sx]
            replaced = True
            break
    if not replaced:
        cts.insert(0, {"type": "scale", "scale": [sz, sy, sx]})


def _default_axes_zyx() -> List[Dict[str, str]]:
    """NGFF 0.4-style axis dicts for a (z, y, x) volume (required by ome-zarr-py Multiscales)."""
    return [
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]


def _ensure_multiscales_image_for_reader(block: Dict[str, Any]) -> None:
    """Fill in ``version`` / ``axes`` so napari's ome-zarr plugin can parse the image.

    ``ome_zarr`` uses ``Axes(..., fmt)`` which leaves ``axes`` unset for NGFF ≥0.3 when
    ``axes`` is missing (then ``validate()`` fails). Nested ``attrs['ome']`` stores are
    loaded with ``root_attrs = attrs['ome']``, so ``multiscales`` there must be complete.
    """
    if block.get("version") in (None, ""):
        block["version"] = "0.4"
    block.setdefault("name", "image")
    ax = block.get("axes")
    ok = isinstance(ax, list) and len(ax) == 3
    if ok:
        names: List[str] = []
        for a in ax:
            if isinstance(a, str):
                names.append(a.lower())
            elif isinstance(a, dict) and "name" in a:
                names.append(str(a["name"]).lower())
            else:
                ok = False
                break
        if ok and names != ["z", "y", "x"]:
            # Mismatch with our (z,y,x) arrays — safer to normalize for the Zarr pipeline.
            ok = False
    if not ok:
        block["axes"] = _default_axes_zyx()


def _write_pyramid_metadata(
    root: zarr.Group,
    *,
    image_meta: Dict[str, Any],
    layout: Literal["top", "ome"],
    datasets_out: List[Dict[str, Any]],
    dtype_str: str,
) -> None:
    """Write multiscales block mirroring input layout (top-level vs ``attrs['ome']``)."""
    block = copy.deepcopy(image_meta)
    block["datasets"] = datasets_out
    _ensure_multiscales_image_for_reader(block)
    if layout == "top":
        root.attrs["multiscales"] = [block]
    else:
        ome = dict(root.attrs.get("ome", {}))
        ome["multiscales"] = [block]
        root.attrs["ome"] = ome
    root.attrs["regiongrow_dtype"] = dtype_str


def _write_omero_for_layout(
    root: zarr.Group,
    omero: Dict[str, Any],
    layout: Literal["top", "ome"],
) -> None:
    """Persist ``omero`` where ome-zarr-py reads it (see ``ome_zarr.io.ZarrLocation``)."""
    if layout == "ome":
        ome = dict(root.attrs.get("ome", {}))
        ome["omero"] = omero
        root.attrs["ome"] = ome
    else:
        root.attrs["omero"] = omero


def _merge_top_level_rdefs_into_omero(
    omero: Dict[str, Any], attrs_src: Dict[str, Any]
) -> None:
    """OME-Zarr readers take ``rdefs`` from ``omero``; merge top-level ``rdefs`` if needed."""
    if "rdefs" in omero:
        return
    rd = attrs_src.get("rdefs")
    if not isinstance(rd, dict):
        return
    try:
        json.dumps(rd)
        omero["rdefs"] = copy.deepcopy(rd)
    except (TypeError, ValueError):
        pass


def _input_omero_from_attrs(attrs_in: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``omero`` may live at top level or under ``attrs['ome']`` (NGFF / Zarr v3 style)."""
    o = attrs_in.get("omero")
    if isinstance(o, dict):
        return o
    ome = attrs_in.get("ome")
    if isinstance(ome, dict):
        o2 = ome.get("omero")
        if isinstance(o2, dict):
            return o2
    return None


def _finalize_root_attrs(
    gin: zarr.Group,
    root: zarr.Group,
    *,
    layout: Literal["top", "ome"],
    input_dtype: np.dtype,
    output_dtype_str: str,
    stretch_applied: bool,
) -> None:
    """Fix or rebuild ``omero`` (and ``rdefs``) for napari and ome-zarr-py.

    ``ome_zarr.io.ZarrLocation`` replaces ``root_attrs`` with ``attrs['ome']`` when that
    key exists, so for layout ``\"ome\"`` we store ``omero`` inside ``attrs['ome']`` next
    to ``multiscales``. Otherwise the plugin never sees ``omero`` or top-level ``rdefs``.
    """
    attrs_in = dict(gin.attrs)
    w_max = 65535 if np.dtype(output_dtype_str) == np.uint16 else 255
    dtype_changed = np.dtype(input_dtype) != np.dtype(output_dtype_str)
    raw_omero = _input_omero_from_attrs(attrs_in)
    preserve_omero = (
        isinstance(raw_omero, dict)
        and not stretch_applied
        and not dtype_changed
    )
    if preserve_omero:
        try:
            json.dumps(raw_omero)
            omero_preserved = copy.deepcopy(raw_omero)
            _merge_top_level_rdefs_into_omero(omero_preserved, attrs_in)
            _write_omero_for_layout(root, omero_preserved, layout)
        except (TypeError, ValueError):
            preserve_omero = False
    if not preserve_omero:
        omero: Dict[str, Any] = (
            copy.deepcopy(raw_omero) if isinstance(raw_omero, dict) else {}
        )
        chans = omero.get("channels")
        if not isinstance(chans, list) or not chans:
            chans = [{"label": "preprocessed", "active": True}]
        new_chans: List[Dict[str, Any]] = []
        for ch in chans:
            if not isinstance(ch, dict):
                continue
            nc = copy.deepcopy(ch)
            nc.setdefault("label", "preprocessed")
            col = str(nc.get("color", "")).replace("#", "").strip()
            if not col or col.upper() == "000000":
                nc["color"] = "FFFFFF"
            else:
                nc["color"] = col.upper()
            nc.setdefault("family", "linear")
            nc["window"] = {"min": 0, "max": w_max, "start": 0, "end": w_max}
            new_chans.append(nc)
        if not new_chans:
            new_chans = [{"label": "preprocessed", "color": "FFFFFF", "active": True, "family": "linear"}]
        omero["channels"] = new_chans
        omero.setdefault("version", "0.4")
        _merge_top_level_rdefs_into_omero(omero, attrs_in)
        _write_omero_for_layout(root, omero, layout)


def run_preprocess_zarr_pipeline(
    inp: str | Path,
    outp: str | Path,
    *,
    apply_downsample: bool,
    downsample_z: int,
    downsample_xy: int,
    apply_stretch: bool,
    stretch_mode: str,
    percentile_low: float,
    percentile_high: float,
    fixed_background: float,
    fixed_vessel_max: float,
    out_dtype: str = "uint8",
    finest_only: bool = False,
) -> Dict[str, Any]:
    """Preprocess finest NGFF level (optional mean downsample + stretch), then full pyramid.

    Unless ``finest_only`` is True, rebuilds every coarser ``datasets[].path`` from the
    preprocessed finest array (block-mean when divisible, else trilinear resize per chunk).
    """
    inp, outp = Path(inp), Path(outp)
    if outp.exists():
        shutil.rmtree(outp)

    arr_in, gin, ds_list = open_finest_zarr_array(inp)
    image_meta, layout = _get_multiscales_image_dict(gin)
    spacing = _finest_scale_from_group(gin)
    fz, fxy = max(1, int(downsample_z)), max(1, int(downsample_xy))
    Z, Y, X = (int(arr_in.shape[0]), int(arr_in.shape[1]), int(arr_in.shape[2]))
    post: Dict[str, Any] = {}
    finest_path = str(ds_list[0]["path"])

    zfmt = zarr_format_of(gin)
    _ds_kw = zarr_output_codec_kwargs(arr_in, zfmt=zfmt)

    root = zarr.open_group(str(outp), mode="w", zarr_format=zfmt)
    work: zarr.Array
    work_name = "_work" if apply_stretch else finest_path

    if apply_downsample and (fz > 1 or fxy > 1):
        nz, ny, nx = Z // fz, Y // fxy, X // fxy
        if nz < 1 or ny < 1 or nx < 1:
            raise ValueError("Downsample factors too large for this volume.")
        new_spacing = (spacing[0] * fz, spacing[1] * fxy, spacing[2] * fxy)
        cz, cy, cx = arr_in.chunks
        out_chunks = (max(1, cz // fz), max(1, cy // fxy), max(1, cx // fxy))
        work = zarr_create_array(
            root,
            work_name,
            shape=(nz, ny, nx),
            chunks=out_chunks,
            dtype=arr_in.dtype,
            **_ds_kw,
        )
        _mean_pool_write(arr_in, work, fz, fxy)
        spacing = new_spacing
        post["downsample"] = {
            "factor_z": fz,
            "factor_xy": fxy,
            "spacing_after": list(spacing),
        }
    elif apply_stretch:
        if stretch_mode == "percentile":
            lo, hi = _percentile_limits_like_in_memory(
                arr_in, float(percentile_low), float(percentile_high)
            )
            post["contrast_stretch"] = {
                "mode": "percentile",
                "input_min": lo,
                "input_max": hi,
                "percentile_low": float(percentile_low),
                "percentile_high": float(percentile_high),
            }
        else:
            lo, hi = float(fixed_background), float(fixed_vessel_max)
            post["contrast_stretch"] = {"mode": "fixed", "input_min": lo, "input_max": hi}
        od: Any = np.uint16 if out_dtype == "uint16" else np.uint8
        work = zarr_create_array(
            root,
            finest_path,
            shape=(Z, Y, X),
            chunks=tuple(int(c) for c in arr_in.chunks),
            dtype=od,
            **_ds_kw,
        )
        for sl in _chunk_slices_3d(arr_in):
            work[sl] = _stretch_block(np.asarray(arr_in[sl]), lo, hi, out_dtype)
        dtype_str = str(np.dtype(od))
        finest_shape_out = tuple(int(x) for x in work.shape)
        if finest_only or len(ds_list) <= 1:
            ds_out = [copy.deepcopy(ds_list[0])]
            _set_scale_on_dataset_entry(
                ds_out[0], finest_shape_out, finest_shape_out, spacing
            )
            _write_pyramid_metadata(
                root,
                image_meta=image_meta,
                layout=layout,
                datasets_out=ds_out,
                dtype_str=dtype_str,
            )
        else:
            _fill_pyramid_coarse_levels(
                root,
                gin,
                ds_list,
                finest_path,
                np.dtype(od),
                _ds_kw,
            )
            ds_out = _build_datasets_meta(ds_list, gin, finest_shape_out, spacing)
            _write_pyramid_metadata(
                root,
                image_meta=image_meta,
                layout=layout,
                datasets_out=ds_out,
                dtype_str=dtype_str,
            )
        _finalize_root_attrs(
            gin,
            root,
            layout=layout,
            input_dtype=np.dtype(arr_in.dtype),
            output_dtype_str=out_dtype,
            stretch_applied=True,
        )
        return post
    else:
        work = zarr_create_array(
            root,
            work_name,
            shape=(Z, Y, X),
            chunks=tuple(int(c) for c in arr_in.chunks),
            dtype=arr_in.dtype,
            **_ds_kw,
        )
        for sl in _chunk_slices_3d(arr_in):
            work[sl] = np.asarray(arr_in[sl])

    if apply_stretch:
        if stretch_mode == "percentile":
            lo, hi = _percentile_limits_like_in_memory(
                work, float(percentile_low), float(percentile_high)
            )
            post["contrast_stretch"] = {
                "mode": "percentile",
                "input_min": lo,
                "input_max": hi,
                "percentile_low": float(percentile_low),
                "percentile_high": float(percentile_high),
            }
        else:
            lo, hi = float(fixed_background), float(fixed_vessel_max)
            post["contrast_stretch"] = {"mode": "fixed", "input_min": lo, "input_max": hi}
        od = np.uint16 if out_dtype == "uint16" else np.uint8
        s0 = zarr_create_array(
            root,
            finest_path,
            shape=tuple(int(x) for x in work.shape),
            chunks=tuple(int(c) for c in work.chunks),
            dtype=od,
            **_ds_kw,
        )
        for sl in _chunk_slices_3d(work):
            s0[sl] = _stretch_block(np.asarray(work[sl]), lo, hi, out_dtype)
        try:
            del root["_work"]
        except KeyError:
            pass
        dtype_str = str(np.dtype(od))
        finest_src = s0
    else:
        dtype_str = str(work.dtype)
        finest_src = work

    finest_shape_out = tuple(int(x) for x in finest_src.shape)

    if finest_only or len(ds_list) <= 1:
        ds_out = [copy.deepcopy(ds_list[0])]
        _set_scale_on_dataset_entry(ds_out[0], finest_shape_out, finest_shape_out, spacing)
        _write_pyramid_metadata(
            root,
            image_meta=image_meta,
            layout=layout,
            datasets_out=ds_out,
            dtype_str=dtype_str,
        )
    else:
        _fill_pyramid_coarse_levels(
            root,
            gin,
            ds_list,
            finest_path,
            finest_src.dtype,
            _ds_kw,
        )
        ds_out = _build_datasets_meta(ds_list, gin, finest_shape_out, spacing)
        _write_pyramid_metadata(
            root,
            image_meta=image_meta,
            layout=layout,
            datasets_out=ds_out,
            dtype_str=dtype_str,
        )
    _finalize_root_attrs(
        gin,
        root,
        layout=layout,
        input_dtype=np.dtype(arr_in.dtype),
        output_dtype_str=dtype_str,
        stretch_applied=apply_stretch,
    )
    return post


def _build_datasets_meta(
    ds_list: List[Dict[str, Any]],
    gin: zarr.Group,
    finest_shape: Tuple[int, int, int],
    finest_scale: Tuple[float, float, float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in ds_list:
        e = copy.deepcopy(entry)
        path = e.get("path")
        if path is None:
            continue
        node = gin[path]
        if not isinstance(node, zarr.Array):
            continue
        shp = tuple(int(x) for x in node.shape)
        _set_scale_on_dataset_entry(e, shp, finest_shape, finest_scale)
        out.append(e)
    return out


def _fill_pyramid_coarse_levels(
    root: zarr.Group,
    gin: zarr.Group,
    ds_list: List[Dict[str, Any]],
    finest_path: str,
    out_dtype: np.dtype,
    _ds_kw: Dict[str, Any],
) -> None:
    fine = root[finest_path]
    if not isinstance(fine, zarr.Array):
        raise TypeError(finest_path)
    for entry in ds_list[1:]:
        path = entry.get("path")
        if path is None:
            continue
        tpl = gin[path]
        if not isinstance(tpl, zarr.Array):
            continue
        ch = tuple(int(c) for c in tpl.chunks)
        out = zarr_create_array(
            root,
            str(path),
            shape=tuple(int(x) for x in tpl.shape),
            chunks=ch,
            dtype=out_dtype,
            **_ds_kw,
        )
        _fill_coarse_from_fine(fine, out)


def stretch_zarr_to_new_store(
    inp: str | Path,
    outp: str | Path,
    *,
    lo: float,
    hi: float,
    out_dtype: str = "uint8",
) -> None:
    """Contrast-stretch finest level to a new store (used by tests / simple CLI)."""
    od = np.uint16 if out_dtype == "uint16" else np.uint8
    run_preprocess_zarr_pipeline(
        inp,
        outp,
        apply_downsample=False,
        downsample_z=1,
        downsample_xy=1,
        apply_stretch=True,
        stretch_mode="fixed",
        percentile_low=0.0,
        percentile_high=100.0,
        fixed_background=lo,
        fixed_vessel_max=hi,
        out_dtype=out_dtype,
    )

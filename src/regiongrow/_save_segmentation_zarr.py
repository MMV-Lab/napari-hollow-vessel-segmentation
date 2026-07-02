"""Write segmentation labels into an existing OME-Zarr (NGFF) store.

We store the segmentation as an NGFF *labels* image under ``labels/<name>``,
versioned as ``segmentation_vN``. This avoids rewriting the raw image pyramid
and stays editable in napari.
"""

from __future__ import annotations

import copy
import errno
import json
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import zarr

# Reuse robust NGFF metadata helpers already used by the zarr preprocessing pipeline.
from regiongrow._preprocessing_zarr import (  # noqa: WPS433 (internal import ok within package)
    _chunk_slices_3d,
    _dst_range_to_src_range,
    _ensure_multiscales_image_for_reader,
    _get_multiscales_image_dict,
    _multiscales_layout,
    _read_multiscales_datasets,
)
from regiongrow._zarr_compat import (
    zarr_array_keys,
    zarr_create_array,
    zarr_open_group_append,
    zarr_output_codec_kwargs,
)


Layout = Literal["top", "ome"]
SaveResolution = Literal["working", "finest"]


def _labels_root_attrs(root: zarr.Group, layout: Layout) -> Dict[str, Any]:
    if layout == "ome":
        ome = dict(root.attrs.get("ome", {}))
        return ome
    return dict(root.attrs)


def _set_labels_list(root: zarr.Group, layout: Layout, labels: List[str]) -> None:
    """Persist root ``labels`` where ome-zarr-py reads it (layout-aware)."""
    if layout == "ome":
        ome = dict(root.attrs.get("ome", {}))
        ome["labels"] = list(labels)
        root.attrs["ome"] = ome
    else:
        root.attrs["labels"] = list(labels)


def _existing_label_names(root: zarr.Group, layout: Layout) -> List[str]:
    attrs = _labels_root_attrs(root, layout)
    ls = attrs.get("labels", [])
    if isinstance(ls, list):
        return [str(x) for x in ls]
    return []


def _next_versioned_name(existing: List[str], base: str = "segmentation") -> str:
    if base not in existing:
        return base
    n = 2           
    while f"{base}_v{n}" in existing:
        n += 1
    return f"{base}_v{n}"


def _default_axes_zyx() -> List[Dict[str, str]]:
    return [
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]


def _dataset_shape(root: zarr.Group, entry: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    p = entry.get("path")
    if p is None:
        return None
    node = root[p]
    if not isinstance(node, zarr.Array):
        return None
    return tuple(int(x) for x in node.shape)


def _find_pyramid_index_for_shape(
    root: zarr.Group, ds_list: List[Dict[str, Any]], shape: Tuple[int, int, int]
) -> int:
    """Return image pyramid index whose array shape matches *shape* (0 = finest)."""
    tgt = tuple(int(x) for x in shape)
    best_i: Optional[int] = None
    best_score = float("inf")
    for i, entry in enumerate(ds_list):
        shp = _dataset_shape(root, entry)
        if shp is None:
            continue
        shp3 = tuple(int(x) for x in shp)
        if shp3 == tgt:
            return i
        score = sum(
            abs(
                float(np.log(max(a, 1))) - float(np.log(max(b, 1)))
            )
            for a, b in zip(shp3, tgt)
        )
        if score < best_score:
            best_score = score
            best_i = i
    if best_i is not None:
        return int(best_i)
    raise ValueError(
        f"Segmentation shape {tgt} does not match any image pyramid level. "
        "Pick a pyramid level under Layers that matches the mask grid."
    )


# Label group names become on-disk directories under ``labels/``; restrict them so a
# user/UI-supplied name cannot escape the store (path traversal) or create odd paths.
_VALID_LABEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Process-wide locks keyed by resolved store path. Manual save and the merge
# autosave both run as background workers in the same napari process; serialising
# writes to one store prevents interleaved metadata/array updates from corrupting it.
_STORE_LOCKS: Dict[str, threading.Lock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


def _store_lock(store_path: str) -> threading.Lock:
    with _STORE_LOCKS_GUARD:
        lk = _STORE_LOCKS.get(store_path)
        if lk is None:
            lk = threading.Lock()
            _STORE_LOCKS[store_path] = lk
        return lk


def _validate_label_name(name: str) -> str:
    name = str(name)
    if not _VALID_LABEL_NAME.match(name) or ".." in name:
        raise ValueError(
            f"invalid label group name {name!r}: use letters, digits, '.', '_', '-' "
            "(no path separators)."
        )
    return name


def _open_existing_omezarr(store: str | Path) -> Tuple[Path, zarr.Group]:
    """Validate that *store* is an existing ``.ome.zarr`` NGFF image, then open it.

    Guards against the segmentation writer silently creating a brand-new (broken)
    zarr store when the user picks the wrong directory in a dialog.
    """
    p = Path(store).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(
            f"OME-Zarr store not found (expected an existing directory): {p}"
        )
    if not p.name.lower().endswith(".ome.zarr"):
        raise ValueError(
            f"{p} is not an .ome.zarr store. Select the store root "
            "(mydata.ome.zarr), not a folder inside labels/."
        )
    root = zarr_open_group_append(p)
    # Raises if the store has no readable NGFF multiscales image.
    _read_multiscales_datasets(root)
    return p, root


def _nn_resize_block(
    block: np.ndarray, out_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Nearest-neighbour resize a 3D block to *out_shape*."""
    if tuple(int(x) for x in block.shape) == tuple(int(x) for x in out_shape):
        return (block > 0).astype(np.uint8, copy=False)
    from skimage.transform import resize

    out = resize(
        np.asarray(block, dtype=np.float32),
        out_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return (out > 0.5).astype(np.uint8)


def _nn_map_target_to_source(
    target_idx: np.ndarray, src_size: int, target_size: int
) -> np.ndarray:
    """Map target voxel indices to source indices (matches ``scipy.ndimage.zoom`` order=0)."""
    if target_size <= 0 or src_size <= 0:
        raise ValueError("Invalid shape for NN map.")
    return (np.asarray(target_idx, dtype=np.intp) * int(src_size) // int(target_size))


def _write_labels_chunked(
    out: zarr.Array,
    src: np.ndarray,
    *,
    src_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int],
) -> None:
    """Write *src* into zarr *out* via NN resample, one output chunk at a time."""
    src_u8 = (np.asarray(src) > 0).astype(np.uint8, copy=False)
    Zs, Ys, Xs = (int(src_shape[0]), int(src_shape[1]), int(src_shape[2]))
    Zt, Yt, Xt = (int(target_shape[0]), int(target_shape[1]), int(target_shape[2]))
    if (Zs, Ys, Xs) == (Zt, Yt, Xt):
        for sl in _chunk_slices_3d(out):
            out[sl] = src_u8[sl]
        return
    for zz, yy, xx in _chunk_slices_3d(out):
        z0, z1 = zz.start, zz.stop
        y0, y1 = yy.start, yy.stop
        x0, x1 = xx.start, xx.stop
        z_idx = _nn_map_target_to_source(np.arange(z0, z1), Zs, Zt)[:, None, None]
        y_idx = _nn_map_target_to_source(np.arange(y0, y1), Ys, Yt)[None, :, None]
        x_idx = _nn_map_target_to_source(np.arange(x0, x1), Xs, Xt)[None, None, :]
        out[zz, yy, xx] = src_u8[z_idx, y_idx, x_idx]


def _max_pool_write_xyz(
    arr_in: zarr.Array,
    out_arr: zarr.Array,
    fz: int,
    fy: int,
    fx: int,
) -> None:
    """Max-pool ``arr_in`` into ``out_arr`` with integer factors per axis (chunked)."""
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
                block = np.asarray(arr_in[zs:ze, ys:ye, xs:xe], dtype=np.uint8)
                tb = block.reshape(z1 - z0, fz, y1 - y0, fy, x1 - x0, fx)
                out_arr[z0:z1, y0:y1, x0:x1] = tb.max(axis=(1, 3, 5)).astype(
                    np.uint8, copy=False
                )


def _fill_coarse_labels_from_fine_zarr(fine: zarr.Array, coarse: zarr.Array) -> None:
    """Fill ``coarse`` labels by max-pooling / NN downsampling ``fine`` (chunked)."""
    Zs, Ys, Xs = (int(fine.shape[0]), int(fine.shape[1]), int(fine.shape[2]))
    Zo, Yo, Xo = (int(coarse.shape[0]), int(coarse.shape[1]), int(coarse.shape[2]))
    if Zs % Zo == 0 and Ys % Yo == 0 and Xs % Xo == 0:
        fz, fy, fx = Zs // Zo, Ys // Yo, Xs // Xo
        _max_pool_write_xyz(fine, coarse, fz, fy, fx)
        return
    for zz, yy, xx in _chunk_slices_3d(coarse):
        zs, ze = _dst_range_to_src_range(zz.start, zz.stop, Zs, Zo)
        ys, ye = _dst_range_to_src_range(yy.start, yy.stop, Ys, Yo)
        xs, xe = _dst_range_to_src_range(xx.start, xx.stop, Xs, Xo)
        block = np.asarray(fine[zs:ze, ys:ye, xs:xe], dtype=np.uint8)
        out_sz = (zz.stop - zz.start, yy.stop - yy.start, xx.stop - xx.start)
        coarse[zz, yy, xx] = _nn_resize_block(block, out_sz)


def _codec_kwargs_for_labels(
    ref_node: zarr.Array, compressor: Optional[Any]
) -> Dict[str, Any]:
    """Return ``compressor`` or ``compressors`` kwargs for label array creation."""
    return zarr_output_codec_kwargs(ref_node, compressor=compressor)


def _ct_from_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    ct = entry.get("coordinateTransformations")
    if ct:
        return copy.deepcopy(ct)
    return [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]


def _label_array_keys(lg: zarr.Group) -> List[str]:
    return zarr_array_keys(lg)


def _remove_stale_label_arrays(
    lg: zarr.Group, planned: Dict[str, Tuple[int, int, int]]
) -> None:
    """Drop label arrays that are absent from *planned* or have the wrong shape."""
    for key in _label_array_keys(lg):
        if key not in planned:
            del lg[key]
    for key, shp in planned.items():
        if key not in lg:
            continue
        node = lg[key]
        if not isinstance(node, zarr.Array):
            del lg[key]
            continue
        if tuple(int(x) for x in node.shape) != tuple(int(x) for x in shp):
            del lg[key]


def _create_labels_dataset(
    lg: zarr.Group,
    name: str,
    *,
    shape: Tuple[int, int, int],
    chunks: Tuple[int, int, int],
    codec_kw: Dict[str, Any],
) -> zarr.Array:
    """Create or replace a uint8 labels array (``require_dataset`` keeps wrong shapes)."""
    if name in lg:
        del lg[name]
    return zarr_create_array(
        lg,
        name,
        shape=shape,
        dtype=np.uint8,
        chunks=chunks,
        overwrite=True,
        **codec_kw,
    )


def _write_labels_pyramid(
    lg: zarr.Group,
    root: zarr.Group,
    ds_list: List[Dict[str, Any]],
    seg: np.ndarray,
    *,
    save_resolution: SaveResolution,
    build_pyramid: bool,
    fine_node: zarr.Array,
    compressor: Optional[Any],
) -> List[Dict[str, Any]]:
    """Chunked write of labels level 0 and optional coarser pyramid levels."""
    seg_u8 = (np.asarray(seg) > 0).astype(np.uint8, copy=False)
    src_shape = tuple(int(x) for x in seg_u8.shape)
    finest_shp = tuple(int(x) for x in fine_node.shape)
    codec_kw = _codec_kwargs_for_labels(fine_node, compressor)
    datasets_out: List[Dict[str, Any]] = []

    if save_resolution == "finest":
        target_shape = finest_shp
        level0_entry = ds_list[0]
        level0_ct = _ct_from_entry(level0_entry)
        pyramid_entries = list(enumerate(ds_list[1:], start=1)) if build_pyramid else []
    else:
        working_idx = _find_pyramid_index_for_shape(root, ds_list, src_shape)
        target_shape = src_shape
        level0_entry = ds_list[working_idx]
        level0_ct = _ct_from_entry(level0_entry)
        if build_pyramid:
            pyramid_entries = [
                (out_i, ds_list[img_i])
                for out_i, img_i in enumerate(range(working_idx + 1, len(ds_list)), start=1)
            ]
        else:
            pyramid_entries = []

    level0_node = root[str(level0_entry["path"])]
    level0_chunks = getattr(level0_node, "chunks", None) or (64, 64, 64)

    planned_shapes: Dict[str, Tuple[int, int, int]] = {"0": target_shape}
    for out_i, entry in pyramid_entries:
        shp = _dataset_shape(root, entry)
        if shp is not None:
            planned_shapes[str(out_i)] = shp
    _remove_stale_label_arrays(lg, planned_shapes)

    a0 = _create_labels_dataset(
        lg,
        "0",
        shape=target_shape,
        chunks=tuple(int(c) for c in level0_chunks),
        codec_kw=codec_kw,
    )
    _write_labels_chunked(
        a0, seg_u8, src_shape=src_shape, target_shape=target_shape
    )
    datasets_out.append({"path": "0", "coordinateTransformations": level0_ct})

    for out_i, entry in pyramid_entries:
        shp = _dataset_shape(root, entry)
        if shp is None:
            continue
        ref_node = root[str(entry["path"])]
        ch = getattr(ref_node, "chunks", None) or (
            max(1, shp[0] // 8),
            256,
            256,
        )
        ai = _create_labels_dataset(
            lg,
            str(out_i),
            shape=shp,
            chunks=tuple(int(c) for c in ch),
            codec_kw=codec_kw,
        )
        _fill_coarse_labels_from_fine_zarr(a0, ai)
        datasets_out.append(
            {
                "path": str(out_i),
                "coordinateTransformations": _ct_from_entry(entry),
            }
        )

    return datasets_out


def upsample_labels_nearest(
    labels_zyx: np.ndarray, target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Nearest-neighbour upsample/downsample a binary labels volume to *target_shape*."""
    src = (np.asarray(labels_zyx) > 0).astype(np.uint8)
    tgt = tuple(int(x) for x in target_shape)
    if tuple(int(x) for x in src.shape) == tgt:
        return src
    from scipy.ndimage import zoom as ndimage_zoom

    zoom_f = [float(o) / float(s) for o, s in zip(tgt, src.shape)]
    out = ndimage_zoom(src.astype(np.float32), zoom_f, order=0)
    return (out > 0.5).astype(np.uint8)


def _output_chunk_slices_3d(
    shape: Tuple[int, int, int],
    chunks: Tuple[int, int, int],
) -> Any:
    """Yield ZYX slice triples covering *shape* with at most *chunks* voxels per block."""
    zc, yc, xc = (max(1, int(chunks[0])), max(1, int(chunks[1])), max(1, int(chunks[2])))
    z, y, x = (int(shape[0]), int(shape[1]), int(shape[2]))
    for z0 in range(0, z, zc):
        z1 = min(z0 + zc, z)
        for y0 in range(0, y, yc):
            y1 = min(y0 + yc, y)
            for x0 in range(0, x, xc):
                x1 = min(x0 + xc, x)
                yield (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def read_zarr_labels_at_shape(
    arr: Any,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Read labels from a zarr array (or ndarray) at *target_shape* without a full-RAM copy.

    Nearest-neighbour resampling matches :func:`upsample_labels_nearest`.  When the
    on-disk array is much larger than the working pyramid level (e.g. finest save
    loaded at a coarse napari level), only the source slabs needed for each output
    chunk are decoded.
    """
    tgt = tuple(int(x) for x in target_shape)
    if len(tgt) != 3 or any(t <= 0 for t in tgt):
        raise ValueError(f"target_shape must be positive 3-D (Z,Y,X); got {target_shape!r}")

    zarr_chunks = getattr(arr, "chunks", None)
    if zarr_chunks is not None and len(zarr_chunks) >= 3:
        out_chunks = tuple(max(1, int(c)) for c in zarr_chunks[-3:])
    else:
        out_chunks = (32, 128, 128)

    src_shape = tuple(int(s) for s in getattr(arr, "shape", np.asarray(arr).shape))
    if src_shape == tgt:
        out = np.zeros(tgt, dtype=np.uint8)
        for zz, yy, xx in _output_chunk_slices_3d(tgt, out_chunks):
            block = np.asarray(arr[zz, yy, xx])
            out[zz, yy, xx] = (block > 0).astype(np.uint8, copy=False)
        return out

    Zs, Ys, Xs = src_shape
    Zt, Yt, Xt = tgt
    out = np.zeros(tgt, dtype=np.uint8)
    for zz, yy, xx in _output_chunk_slices_3d(tgt, out_chunks):
        z0, z1 = zz.start, zz.stop
        y0, y1 = yy.start, yy.stop
        x0, x1 = xx.start, xx.stop
        z_idx = _nn_map_target_to_source(np.arange(z0, z1), Zs, Zt)
        y_idx = _nn_map_target_to_source(np.arange(y0, y1), Ys, Yt)
        x_idx = _nn_map_target_to_source(np.arange(x0, x1), Xs, Xt)
        zs, ze = int(z_idx.min()), int(z_idx.max()) + 1
        ys, ye = int(y_idx.min()), int(y_idx.max()) + 1
        xs, xe = int(x_idx.min()), int(x_idx.max()) + 1
        block = (np.asarray(arr[zs:ze, ys:ye, xs:xe]) > 0).astype(np.uint8, copy=False)
        rel_z = z_idx - zs
        rel_y = y_idx - ys
        rel_x = x_idx - xs
        out[zz, yy, xx] = block[
            rel_z[:, np.newaxis, np.newaxis],
            rel_y[np.newaxis, :, np.newaxis],
            rel_x[np.newaxis, np.newaxis, :],
        ]
    return out


def _write_omero_attrs(lg: zarr.Group, name: str, label_color: str) -> None:
    omero = {
        "version": "0.4",
        "channels": [
            {
                "label": name,
                "color": str(label_color).replace("#", "").upper() or "FF0000",
                "active": True,
                "window": {"min": 0, "max": 1, "start": 0, "end": 1},
                "family": "linear",
            }
        ],
    }
    try:
        json.dumps(omero)
    except (TypeError, ValueError):
        # Sanitize an unserializable colour rather than dropping all display metadata.
        omero["channels"][0]["color"] = "FF0000"
    lg.attrs["omero"] = omero


def _is_staging_label_name(name: str) -> bool:
    return "__tmp_" in str(name)


def _atomic_replace_path(src: Path, dst: Path) -> None:
    """Rename *src* over *dst*, with a cross-device ``shutil.move`` fallback."""
    try:
        os.replace(src, dst)
    except OSError as exc:
        if getattr(exc, "errno", None) != errno.EXDEV:
            raise
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.move(str(src), str(dst))


def _swap_labels_group_on_disk(
    store_path: Path, final_name: str, staging_name: str
) -> None:
    """Atomically replace ``labels/<final_name>`` with ``labels/<staging_name>``.

    The slow write goes to the staging group; the visible swap is two fast
    directory renames, so a crash leaves either the old group or the new group
    intact — never a half-written one.
    """
    labels_dir = store_path / "labels"
    final_dir = labels_dir / final_name
    staging_dir = labels_dir / staging_name
    if not staging_dir.exists():
        raise FileNotFoundError(
            f"Staging labels group missing: {staging_dir}"
        )
    # Zarr v3 stores may use ``labels/zarr.json``; child groups are still directories.
    if not staging_dir.is_dir():
        raise TypeError(
            f"Expected a directory for labels staging group: {staging_dir}"
        )
    old_dir: Optional[Path] = None
    if final_dir.exists():
        old_dir = labels_dir / f"{final_name}.old_{uuid.uuid4().hex[:8]}"
        _atomic_replace_path(final_dir, old_dir)
    try:
        _atomic_replace_path(staging_dir, final_dir)
    except OSError:
        # Roll back so the previous group is not lost.
        if old_dir is not None and not final_dir.exists():
            _atomic_replace_path(old_dir, final_dir)
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    if old_dir is not None:
        shutil.rmtree(old_dir, ignore_errors=True)
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)


def _write_segmentation_labels(
    store: str | Path,
    segmentation_zyx: np.ndarray,
    *,
    name_base: str,
    labels_name: Optional[str],
    versioned: bool,
    label_color: str,
    build_pyramid: bool,
    save_resolution: SaveResolution,
    compressor: Optional[Any],
) -> Dict[str, Any]:
    """Shared writer: validate store, lock, stage labels, then atomically swap in."""
    seg = np.asarray(segmentation_zyx)
    if seg.ndim != 3:
        raise ValueError(
            f"segmentation must be 3-D (Z, Y, X); got shape {seg.shape}"
        )

    store_path, root = _open_existing_omezarr(store)

    with _store_lock(str(store_path)):
        layout: Layout = _multiscales_layout(root)
        ds_list = _read_multiscales_datasets(root)
        image_meta, _layout2 = _get_multiscales_image_dict(root)
        if layout != _layout2:
            layout = _layout2

        # Resolve the final name *under the lock* so concurrent saves cannot pick
        # the same version number.
        existing = _existing_label_names(root, layout)
        if labels_name is not None:
            name = _validate_label_name(labels_name)
        elif versioned:
            name = _next_versioned_name(existing, base=_validate_label_name(name_base))
        else:
            name = _validate_label_name(name_base)

        finest_path = str(ds_list[0]["path"])
        fine_node = root[finest_path]
        if not isinstance(fine_node, zarr.Array):
            raise TypeError(f"Expected zarr.Array at {finest_path!r}")

        labels_root = root.require_group("labels")
        staging_name = f"{name}__tmp_{uuid.uuid4().hex[:8]}"
        staging = labels_root.require_group(staging_name)

        try:
            datasets_out = _write_labels_pyramid(
                staging,
                root,
                ds_list,
                seg,
                save_resolution=save_resolution,
                build_pyramid=build_pyramid,
                fine_node=fine_node,
                compressor=compressor,
            )

            block = copy.deepcopy(image_meta)
            block.setdefault("version", "0.4")
            block.setdefault("name", name)
            block["axes"] = _default_axes_zyx()
            block.pop("type", None)
            _ensure_multiscales_image_for_reader(block)
            block["datasets"] = datasets_out
            staging.attrs["multiscales"] = [block]
            staging.attrs["image-label"] = {"version": "0.4", "source": {"image": "0"}}
            _write_omero_attrs(staging, name, label_color)

            # Commit: swap the finished staging group over the live group, then
            # advertise the name in the root list only after data is in place.
            _swap_labels_group_on_disk(store_path, name, staging_name)
            from regiongrow._omezarr_reader import _ngff_label_names_from_store

            _set_labels_list(root, layout, _ngff_label_names_from_store(store_path))
        except BaseException:
            # Best-effort cleanup of the partial staging group.
            try:
                del labels_root[staging_name]
            except (KeyError, OSError):
                shutil.rmtree(
                    store_path / "labels" / staging_name, ignore_errors=True
                )
            raise

    return {
        "labels_name": name,
        "labels_group": f"labels/{name}",
        "levels_written": len(datasets_out),
        "layout": layout,
        "save_resolution": save_resolution,
        "image_pyramid_index": (
            _find_pyramid_index_for_shape(root, ds_list, tuple(int(x) for x in seg.shape))
            if save_resolution == "working"
            else 0
        ),
    }


def write_segmentation_labels_to_ome_zarr(
    store: str | Path,
    segmentation_zyx: np.ndarray,
    *,
    name_base: str = "segmentation",
    labels_name: Optional[str] = None,
    label_color: str = "FF0000",
    build_pyramid: bool = True,
    save_resolution: SaveResolution = "finest",
    compressor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Write segmentation into ``labels/<versioned-name>`` inside *store*.

    Parameters
    ----------
    store
        Path to the existing ``.ome.zarr`` directory (validated; never created).
    segmentation_zyx
        3D array (Z,Y,X). Non-zero values are treated as foreground; stored as uint8.
    name_base
        Base label name used for versioning when *labels_name* is omitted.
    labels_name
        If set, write (overwrite) this fixed group under ``labels/<labels_name>``
        instead of creating a new ``segmentation_vN`` version.
    label_color
        Hex RGB string for display (OMERO metadata on the labels image).
    build_pyramid
        If True, write coarser multiscale label levels (max-pool / NN fallback).
    save_resolution
        ``"finest"`` upsamples to the image finest grid (chunked); ``"working"``
        keeps the mask at its current pyramid resolution.
    compressor
        Optional zarr compressor override for v2 stores.
    """
    return _write_segmentation_labels(
        store,
        segmentation_zyx,
        name_base=name_base,
        labels_name=labels_name,
        versioned=labels_name is None,
        label_color=label_color,
        build_pyramid=build_pyramid,
        save_resolution=save_resolution,
        compressor=compressor,
    )


CHECKPOINT_LABELS_NAME = "segmentation_autosave"


def write_segmentation_checkpoint(
    store: str | Path,
    segmentation_zyx: np.ndarray,
    *,
    name: str = CHECKPOINT_LABELS_NAME,
    label_color: str = "FF0000",
    build_pyramid: bool = False,
    save_resolution: SaveResolution = "working",
    compressor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Overwrite ``labels/<name>`` in *store* (crash-recovery autosave).

    Unlike :func:`write_segmentation_labels_to_ome_zarr`, reuses a fixed
    label group name instead of creating ``segmentation_vN`` versions.
    Defaults to working-resolution, single-level writes for low RAM use.
    """
    return _write_segmentation_labels(
        store,
        segmentation_zyx,
        name_base=name,
        labels_name=name,
        versioned=False,
        label_color=label_color,
        build_pyramid=build_pyramid,
        save_resolution=save_resolution,
        compressor=compressor,
    )

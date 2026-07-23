"""Helpers to capture napari viewer frames and write growth-animation GIFs."""

from __future__ import annotations

from typing import Any, List

import numpy as np


def _rgba_to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    """Drop alpha; ensure uint8 (H, W, 3)."""
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("Expected an image array with shape (H, W, C), C>=3")
    rgb = a[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _resize_rgb(rgb: np.ndarray, scale_percent: float) -> np.ndarray:
    """Downscale *rgb* when scale_percent < 100 (keeps uint8 output)."""
    sp = float(scale_percent)
    if sp >= 99.5:
        return rgb
    from skimage.transform import resize

    h, w = rgb.shape[:2]
    nh = max(1, int(round(h * sp / 100.0)))
    nw = max(1, int(round(w * sp / 100.0)))
    if nh == h and nw == w:
        return rgb
    f = resize(
        rgb.astype(np.float64) / 255.0,
        (nh, nw, 3),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    return (np.clip(f, 0, 1) * 255.0).astype(np.uint8)


def letterbox_rgb_to_canvas(
    rgb: np.ndarray, out_h: int, out_w: int
) -> np.ndarray:
    """Scale *rgb* (H, W, 3) uniformly to fit inside (out_h, out_w), pad with black."""
    from skimage.transform import resize

    rgb = np.asarray(rgb[..., :3], dtype=np.uint8)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    out_h = max(1, int(out_h))
    out_w = max(1, int(out_w))
    scale = min(out_w / max(w, 1), out_h / max(h, 1))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if nh == h and nw == w:
        sm = rgb
    else:
        sm = resize(
            rgb.astype(np.float64) / 255.0,
            (nh, nw, 3),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
        sm = (np.clip(sm, 0, 1) * 255.0).astype(np.uint8)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0 = (out_h - nh) // 2
    x0 = (out_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = sm
    return canvas


def normalize_gif_frames(
    frames: List[np.ndarray],
    *,
    canvas_height: int = 0,
    canvas_width: int = 0,
) -> List[np.ndarray]:
    """Return RGB uint8 frames of identical (H, W) for GIF encoding.

    If *canvas_height* or *canvas_width* is <= 0, use the max height and max width
    seen across *frames* (letterbox each frame to that canvas).
    If both are > 0, letterbox every frame to that fixed size.
    """
    if not frames:
        return []
    arrs = [np.asarray(f, dtype=np.uint8) for f in frames]
    for a in arrs:
        if a.ndim != 3 or a.shape[2] < 3:
            raise ValueError("Each frame must be (H, W, 3) RGB")
    if canvas_height <= 0 or canvas_width <= 0:
        oh = max(int(a.shape[0]) for a in arrs)
        ow = max(int(a.shape[1]) for a in arrs)
    else:
        oh, ow = int(canvas_height), int(canvas_width)
    oh = max(1, oh)
    ow = max(1, ow)
    shapes = {(a.shape[0], a.shape[1]) for a in arrs}
    if len(shapes) == 1 and next(iter(shapes)) == (oh, ow):
        return [np.ascontiguousarray(a[..., :3]) for a in arrs]
    return [letterbox_rgb_to_canvas(a, oh, ow) for a in arrs]


def capture_viewer_frame(
    viewer: Any,
    *,
    canvas_only: bool,
    scale_percent: float = 100.0,
) -> np.ndarray | None:
    """Return RGB uint8 (H, W, 3) or None if screenshot fails."""
    try:
        try:
            arr = viewer.screenshot(canvas_only=canvas_only, flash=False)
        except TypeError:
            arr = viewer.screenshot()
    except Exception:
        return None
    if arr is None or np.asarray(arr).size == 0:
        return None
    rgb = _rgba_to_rgb_u8(np.asarray(arr))
    return _resize_rgb(rgb, scale_percent)


def save_gif(
    frames: List[np.ndarray],
    path: str,
    fps: float,
    *,
    canvas_height: int = 0,
    canvas_width: int = 0,
) -> None:
    """Write *frames* (RGB uint8) to *path* as an animated GIF.

    All frames are letterboxed to a common size: use *canvas_width* and
    *canvas_height* when both > 0; otherwise the canvas is the max width/height
    across the batch (needed for combined clips from different viewer sizes).
    """
    if not frames:
        raise ValueError("No frames to save")
    frames = normalize_gif_frames(
        frames, canvas_height=canvas_height, canvas_width=canvas_width
    )
    fps = float(fps)
    if fps <= 0:
        fps = 1.0
    duration = 1.0 / fps
    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None  # type: ignore[assignment]
    if imageio is not None:
        imageio.mimsave(path, frames, format="GIF", duration=duration, loop=0)
        return
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    first, rest = imgs[0], imgs[1:]
    duration_ms = max(20, int(round(1000.0 / fps)))
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        format="GIF",
    )

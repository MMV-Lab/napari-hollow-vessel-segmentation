"""Napari widget for interactive 3D vessel segmentation via region growing."""

from datetime import datetime
from typing import Any, List, Optional

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QFrame,
    QMessageBox,
    QRadioButton,
    QButtonGroup,
    QFileDialog,
    QApplication,
)
from napari.qt.threading import thread_worker
import napari
from napari.utils.colormaps import DirectLabelColormap
from scipy.ndimage import (
    zoom as ndimage_zoom,
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
)

from ._spatial import spatial_alignment_for_pyramid_level, spatial_alignment_kwargs
from ._preprocessing import apply_preprocess_chain
from ._volume_utils import (
    axis_margin_voxels_for_work_level,
    check_materialization_budget,
    image_finest_shape,
    image_level_is_lazy,
    image_level_shape,
    is_multiscale_image_layer,
    layer_data_shape,
    materialize_image_level,
    multiscale_level_count,
    multiscale_level_label,
    tube_radius_voxels_for_work_level,
    voxel_spacing_zyx_for_level,
)

# Rough RAM cap before Grow / in-memory preprocessing (float64 image + masks).
_MAX_MATERIALIZE_GB = 12.0
_MAX_MATERIALIZE_BYTES = _MAX_MATERIALIZE_GB * 1e9

# Three-layer workflow for vessel branches.
MERGED_SEG_LAYER_NAME = "Merged_Segmentation"
DRAFT_BRANCH_LAYER_NAME = "Draft_Branch"
BLOCKER_MASK_LAYER_NAME = "Blocker_Mask"
THRESHOLD_MASK_LAYER_NAME = "Threshold_Mask"


def _polyline_indices_for_level(
    poly_fine: np.ndarray,
    image_layer: Any,
    level: int,
    shape_work: tuple,
) -> np.ndarray:
    """Map integer ZYX indices on the finest grid to the chosen pyramid level.

    Uses napari's ``downsample_factors`` for multiscale images so indices match
    the same pyramid convention as the viewer (not only ``shape_work/shape_fine``
    ratios, which can disagree for non-uniform level sizes).
    """
    if poly_fine.size == 0:
        return poly_fine.astype(np.int64)
    sz = tuple(int(x) for x in shape_work)
    if is_multiscale_image_layer(image_layer):
        df = np.asarray(
            image_layer.downsample_factors[int(level)], dtype=np.float64
        ).ravel()
    else:
        df = np.ones(max(poly_fine.shape[1], 3), dtype=np.float64)
    if df.size >= 3:
        df3 = df[-3:].copy()
    else:
        df3 = np.ones(3, dtype=np.float64)
        df3[-df.size :] = df
    df3[df3 <= 0] = 1.0
    q = np.rint(poly_fine.astype(np.float64) / df3.reshape(1, 3)).astype(
        np.int64
    )
    q[:, 0] = np.clip(q[:, 0], 0, max(sz[0] - 1, 0))
    q[:, 1] = np.clip(q[:, 1], 0, max(sz[1] - 1, 0))
    q[:, 2] = np.clip(q[:, 2], 0, max(sz[2] - 1, 0))
    return q


def _branch_effective_margin(skel: np.ndarray, margin: float, spacing) -> float:
    """Extra length margin for branch grow so the chord axis mask fits curved paths."""
    sk = np.asarray(skel, dtype=bool)
    zz, yy, xx = np.nonzero(sk)
    if zz.size == 0:
        return float(margin)
    s = np.asarray(spacing, dtype=np.float64).ravel()[:3]
    ex = max(
        float(zz.max() - zz.min()) * s[0],
        float(yy.max() - yy.min()) * s[1],
        float(xx.max() - xx.min()) * s[2],
    )
    mean_s = float(np.mean(s))
    return float(margin) + 0.45 * ex + 3.5 * mean_s


def _voxel_spacing_zyx(image_layer: Any) -> tuple:
    """``(s_z, s_y, s_x)`` from the napari image layer (last three scale entries)."""
    s = np.asarray(image_layer.scale, dtype=np.float64).ravel()
    if s.size < 3:
        return (1.0, 1.0, 1.0)
    s = s[-3:].copy()
    s[s <= 0] = 1.0
    return tuple(float(x) for x in s)


def _points_world_to_image_zyx(
    image_layer: Any, points_layer: Any, shape: tuple
) -> np.ndarray:
    """Map user points to integer ``(z, y, x)`` indices on *image_layer*.

    ``Points.data`` lives in the points layer's data coordinate system, not
    necessarily world space.  We convert ``data → world`` with the points
    layer, then ``world → data`` with the image layer so downsampling,
    anisotropic scale, and translate match the selected image grid.
    """
    pts = np.asarray(points_layer.data, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    out = np.zeros((pts.shape[0], 3), dtype=np.int64)
    zmax, ymax, xmax = int(shape[0]) - 1, int(shape[1]) - 1, int(shape[2]) - 1
    for i in range(pts.shape[0]):
        row = np.asarray(pts[i], dtype=np.float64).ravel()
        if row.size < 3:
            raise ValueError("Each point must have at least 3 coordinates.")
        world = np.asarray(
            points_layer.data_to_world(row[:3]), dtype=np.float64
        ).ravel()
        data_xyz = np.asarray(
            image_layer.world_to_data(world), dtype=np.float64
        ).ravel()
        out[i, 0] = int(np.round(data_xyz[0]))
        out[i, 1] = int(np.round(data_xyz[1]))
        out[i, 2] = int(np.round(data_xyz[2]))
        out[i, 0] = int(np.clip(out[i, 0], 0, max(zmax, 0)))
        out[i, 1] = int(np.clip(out[i, 1], 0, max(ymax, 0)))
        out[i, 2] = int(np.clip(out[i, 2], 0, max(xmax, 0)))
    return out


def _is_auto_sized_branch_points_name(name: str) -> bool:
    """Layers we resize for visibility (default naming from the plugin / README)."""
    return name == "BranchPoints" or name.startswith("BranchPoints_")


def _suggested_branch_point_base_size(image_layer: Any) -> float:
    """Marker diameter in Points *data* coordinates for a readable default on huge volumes."""
    shape = image_finest_shape(image_layer)
    if len(shape) >= 3:
        m = float(min(int(shape[-3]), int(shape[-2]), int(shape[-1])))
    elif len(shape) >= 1:
        m = float(min(int(s) for s in shape))
    else:
        m = 128.0
    # Slightly smaller than before for cleaner overlays; cap avoids extreme GPU paths.
    return float(max(22.0, min(m * 0.020, 420.0)))


def _apply_spatial_kwargs_to_layer(lyr: Any, skw: dict) -> None:
    """Update *lyr* transform metadata from ``spatial_alignment_*`` dict."""
    for key in ("scale", "translate", "rotate", "shear", "units"):
        if key in skw:
            setattr(lyr, key, skw[key])


def _tip(text):
    """Return a styled circular '?' label that shows *text* on hover."""
    lbl = QLabel("?")
    lbl.setFixedSize(16, 16)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setCursor(Qt.WhatsThisCursor)
    lbl.setToolTip(text)
    lbl.setStyleSheet(
        "QLabel { border: 1px solid #888; border-radius: 8px;"
        " color: #555; font-size: 9px; font-weight: bold;"
        " background: #f5f5f5; }"
        "QLabel:hover { background: #ddd; }"
    )
    return lbl


def _row(widget, tip_text=None):
    """Wrap *widget* with a fixed-size '?' badge at its right edge.

    *tip_text* defaults to the widget's own toolTip() when omitted,
    so callers only need to call setToolTip() once.
    The widget stretches; the badge stays compact.
    """
    box = QHBoxLayout()
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(widget, 1)
    box.addWidget(
        _tip(tip_text if tip_text is not None else widget.toolTip()), 0
    )
    return box


def _collapsible_section(
    title: str,
    inner: QWidget,
    *,
    start_open: bool = True,
    header_tooltip: Optional[str] = None,
) -> QWidget:
    """Return a widget with a clickable header that shows or hides *inner*."""
    outer = QWidget()
    v = QVBoxLayout(outer)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(2)
    toggle = QToolButton()
    toggle.setText(title)
    if header_tooltip:
        toggle.setToolTip(header_tooltip)
    toggle.setCheckable(True)
    toggle.setChecked(start_open)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.DownArrow if start_open else Qt.RightArrow)
    toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    toggle.setStyleSheet(
        "QToolButton { text-align: left; font-weight: bold; padding: 2px; }"
    )
    frame = QFrame()
    fl = QVBoxLayout(frame)
    fl.setContentsMargins(4, 0, 4, 6)
    fl.addWidget(inner)

    def _on_toggled(checked: bool) -> None:
        frame.setVisible(checked)
        toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    toggle.toggled.connect(_on_toggled)
    frame.setVisible(start_open)
    v.addWidget(toggle)
    v.addWidget(frame)
    return outer


class RegionGrowWidget(QWidget):
    """Widget for 3D vessel segmentation via polyline-based plain / MGAC grow and merge."""

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        self._result_layer = None
        self._preprocessed_images = {}
        self._preview_layer = None
        self._preprocess_worker = None
        self._image_working_metadata: dict = {}
        self._pending_branch_grow_cleanup = False
        self._active_branch_job = None
        self._branch_pts_sync = None
        self._branch_step_target_layer = None
        # Draft branch masks are archived when a new Grow starts before Merge.
        # Cycle label-1 color so multiple unmerged drafts stay visually distinct.
        self._draft_branch_color_cycle = [
            "magenta",
            "yellow",
            "cyan",
            "lime",
            "orange",
            "deepskyblue",
            "violet",
        ]
        self._draft_branch_color_index = 0

        self._growth_capture_frames: List[np.ndarray] = []
        self._growth_capture_step_counter = 0
        self._growth_capture_finalized = False
        self._growth_capture_hit_max = False
        self._gif_encode_worker = None
        self._gif_capture_combined_frames: List[np.ndarray] = []
        self._gif_capture_pending_segment: List[np.ndarray] = []
        self._gif_capture_combine_this_run = False
        self._skeletal_preview_vispy_registered = False
        self._branch_point_size_bases: dict = {}
        self._branch_point_zoom_ref: Optional[float] = None

        self._build_ui()
        self._sync_save_combined_gif_button()
        self._connect_signals()
        self._refresh_layers()
        self._on_ndisplay_changed()

        # Keep combos in sync with layer changes
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        # 2D canvas + empty Points can trigger VisPy glTexSubImage2D(height=0); hide those layers.
        try:
            self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)
        except AttributeError:
            pass

    def _draft_branch_label_color(self) -> str:
        i = int(getattr(self, "_draft_branch_color_index", 0))
        cyc = list(getattr(self, "_draft_branch_color_cycle", ["magenta"]))
        if not cyc:
            return "magenta"
        return str(cyc[i % len(cyc)])

    @staticmethod
    def _worker_exception_message(exc) -> str:
        if isinstance(exc, BaseException):
            return str(exc)
        if isinstance(exc, tuple) and len(exc) >= 2 and exc[1] is not None:
            return str(exc[1])
        return str(exc)

    def _cleanup_preprocess_worker_ui(self) -> None:
        self.progress_bar.hide()
        self.btn_apply_preprocess.setEnabled(True)
        if self._worker is None:
            self.btn_grow_branches.setEnabled(True)
        self._preprocess_worker = None

    def _sync_save_combined_gif_button(self) -> None:
        n = len(getattr(self, "_gif_capture_combined_frames", []))
        self.btn_save_combined_gif.setEnabled(n > 0)

    def _on_capture_growth_toggled(self, checked: bool) -> None:
        self.capture_options.setVisible(checked)
        self.capture_combine_grows_check.setEnabled(checked)
        if not checked:
            self.capture_combine_grows_check.setChecked(False)
            self._gif_capture_pending_segment.clear()
        self._sync_save_combined_gif_button()

    def _on_capture_combine_toggled(self, checked: bool) -> None:
        if not checked:
            self._gif_capture_pending_segment.clear()

    def _save_combined_growth_gif(self) -> None:
        frames = list(self._gif_capture_combined_frames)
        if not frames:
            self.status_label.setText("No combined GIF frames — Merge after grows with combine mode on.")
            return
        default_name = (
            f"vessel_growth_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        )
        path, _sel = QFileDialog.getSaveFileName(
            self,
            "Save combined growth GIF",
            default_name,
            "GIF (*.gif);;All files (*)",
        )
        if not path:
            self.status_label.setText("Combined GIF save cancelled.")
            return
        if not path.lower().endswith(".gif"):
            path = path + ".gif"
        fps = float(self.capture_output_fps_spin.value())
        self._start_gif_encode_worker(
            frames,
            path,
            fps,
            hit_max=False,
            had_error=False,
            clear_combined_after=True,
        )

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        root_layout = QVBoxLayout()
        self.setLayout(root_layout)

        # Use a scroll area so the dock widget stays vertically resizable
        # even when the parameter panel grows taller than the window.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)

        # --- Shared: Layer selection ---
        layer_inner = QWidget()
        layer_form = QFormLayout(layer_inner)

        self.image_combo = QComboBox()
        layer_form.addRow("Image:", self.image_combo)

        self.ms_level_combo = QComboBox()
        self.ms_level_combo.setToolTip(
            "For multiscale / pyramid images (e.g. OME-Zarr), pick which resolution "
            "Grow and masks use. Level 0 is finest; higher levels are smaller and faster."
        )
        self._ms_level_row_label = QLabel("Pyramid level:")
        self._ms_level_row_label.setVisible(False)
        self.ms_level_combo.setVisible(False)
        layer_form.addRow(self._ms_level_row_label, self.ms_level_combo)

        layout.addWidget(_collapsible_section("Layers", layer_inner))

        # --- Preprocessing (denoise → contrast; use Pyramid level to change resolution) ---
        prep_inner = QWidget()
        prep_form = QFormLayout(prep_inner)
        prep_note = QLabel(
            "Pipeline order: denoise → contrast stretch. "
            "Use **Pyramid level** (multiscale images) to work on a coarser grid instead of "
            "resampling here. Non-local means is heavy on large volumes (background worker). "
            "For disk-streamed OME-Zarr see: regiongrow-preprocess-zarr (README)."
        )
        prep_note.setWordWrap(True)
        prep_form.addRow(prep_note)

        self.prep_denoise_check = QCheckBox("Non-local means denoise")
        self.prep_denoise_check.setChecked(False)
        self.prep_denoise_check.setToolTip(
            "3D skimage denoise_nl_means. Expensive on big images — prefer crops or "
            "turn off for full volumes."
        )
        prep_form.addRow(self.prep_denoise_check)

        self.prep_denoise_patch_spin = QSpinBox()
        self.prep_denoise_patch_spin.setRange(3, 15)
        self.prep_denoise_patch_spin.setSingleStep(2)
        self.prep_denoise_patch_spin.setValue(5)
        self.prep_denoise_patch_spin.setToolTip("Patch size (odd integer; 5 is typical).")
        prep_form.addRow("Denoise patch size:", _row(self.prep_denoise_patch_spin))

        self.prep_denoise_dist_spin = QSpinBox()
        self.prep_denoise_dist_spin.setRange(1, 20)
        self.prep_denoise_dist_spin.setValue(6)
        self.prep_denoise_dist_spin.setToolTip("Search distance (larger = slower, often better).")
        prep_form.addRow("Denoise patch distance:", _row(self.prep_denoise_dist_spin))

        self.prep_denoise_h_spin = QDoubleSpinBox()
        self.prep_denoise_h_spin.setRange(0.001, 0.5)
        self.prep_denoise_h_spin.setDecimals(3)
        self.prep_denoise_h_spin.setSingleStep(0.01)
        self.prep_denoise_h_spin.setValue(0.05)
        self.prep_denoise_h_spin.setToolTip("Filter strength h (e.g. 0.03–0.1).")
        prep_form.addRow("Denoise strength (h):", _row(self.prep_denoise_h_spin))

        self.prep_stretch_check = QCheckBox("Contrast stretch")
        self.prep_stretch_check.setChecked(True)
        self.prep_stretch_check.setToolTip(
            "Map intensities to uint8/uint16 display range (after denoise if enabled)."
        )
        prep_form.addRow(self.prep_stretch_check)

        self.prep_stretch_mode_combo = QComboBox()
        self.prep_stretch_mode_combo.addItems(["percentile", "fixed"])
        self.prep_stretch_mode_combo.setToolTip('percentile: per-image percentiles; fixed: min/max bounds.')
        prep_form.addRow("Stretch mode:", _row(self.prep_stretch_mode_combo))

        self.prep_pct_low_spin = QDoubleSpinBox()
        self.prep_pct_low_spin.setRange(0.0, 50.0)
        self.prep_pct_low_spin.setDecimals(1)
        self.prep_pct_low_spin.setValue(2.0)
        prep_form.addRow("Percentile low (%):", _row(self.prep_pct_low_spin))

        self.prep_pct_high_spin = QDoubleSpinBox()
        self.prep_pct_high_spin.setRange(50.0, 100.0)
        self.prep_pct_high_spin.setDecimals(1)
        self.prep_pct_high_spin.setValue(98.0)
        prep_form.addRow("Percentile high (%):", _row(self.prep_pct_high_spin))

        self.prep_fixed_bg_spin = QDoubleSpinBox()
        self.prep_fixed_bg_spin.setRange(-1e9, 1e9)
        self.prep_fixed_bg_spin.setDecimals(2)
        self.prep_fixed_bg_spin.setValue(150.0)
        prep_form.addRow("Fixed background max:", _row(self.prep_fixed_bg_spin))

        self.prep_fixed_max_spin = QDoubleSpinBox()
        self.prep_fixed_max_spin.setRange(-1e9, 1e9)
        self.prep_fixed_max_spin.setDecimals(2)
        self.prep_fixed_max_spin.setValue(500.0)
        prep_form.addRow("Fixed vessel max:", _row(self.prep_fixed_max_spin))

        self.prep_out_dtype_combo = QComboBox()
        self.prep_out_dtype_combo.addItems(["uint8", "uint16"])
        self.prep_out_dtype_combo.setCurrentIndex(0)
        prep_form.addRow("Stretch output dtype:", _row(self.prep_out_dtype_combo))

        out_mode = QHBoxLayout()
        self.prep_out_mode_group = QButtonGroup(self)
        self.prep_radio_new = QRadioButton("Add new layer")
        self.prep_radio_replace = QRadioButton("Replace current layer")
        self.prep_radio_new.setChecked(True)
        self.prep_out_mode_group.addButton(self.prep_radio_new, 0)
        self.prep_out_mode_group.addButton(self.prep_radio_replace, 1)
        out_mode.addWidget(self.prep_radio_new)
        out_mode.addWidget(self.prep_radio_replace)
        prep_form.addRow("Output:", out_mode)

        self.btn_apply_preprocess = QPushButton("Run preprocessing")
        self.btn_apply_preprocess.setToolTip(
            "Runs enabled steps in order on a copy of the selected Image, then adds or replaces a layer."
        )
        prep_form.addRow(_row(self.btn_apply_preprocess))

        # --- Thresholding (2D: current view plane only; 3D: full volume) ---
        thr_note = QLabel(
            "Threshold mask helper. In 3D view it thresholds the full volume. "
            "In 2D view it thresholds only the currently visible slice plane (whatever axes you are viewing)."
        )
        thr_note.setWordWrap(True)
        prep_form.addRow(thr_note)

        self.prep_thr_method_combo = QComboBox()
        self.prep_thr_method_combo.addItems(
            ["Otsu", "Li", "Triangle", "Yen", "Mean", "90th percentile", "95th percentile"]
        )
        self.prep_thr_method_combo.setToolTip(
            "Thresholding method. Produces/updates a labels layer named Threshold_Mask."
        )
        prep_form.addRow("Threshold method:", _row(self.prep_thr_method_combo))

        self.btn_apply_threshold = QPushButton("Apply threshold mask")
        self.btn_apply_threshold.setToolTip(
            "Creates/updates Threshold_Mask. In 2D view, only the visible plane is updated."
        )
        prep_form.addRow(_row(self.btn_apply_threshold))

        layout.addWidget(
            _collapsible_section(
                "Preprocessing",
                prep_inner,
                start_open=False,
                header_tooltip=(
                    "Optional non-local means denoise and contrast stretch on the current "
                    "pyramid level. Resolution is chosen under Layers, not here."
                ),
            )
        )

        # --- Shared: Visualization (valid for both modes) ---
        vis_inner = QWidget()
        vis_form = QFormLayout(vis_inner)

        self.animate_check = QCheckBox("Animate growth")
        self.animate_check.setChecked(True)
        self.animate_check.setToolTip(
            "Show the growing contour at each display step.\n"
            "Disable for maximum speed.\n"
            "Update frequency is set under Segmentation parameters (Plain / MGAC)."
        )
        self.capture_growth_check = QCheckBox("Capture animation (GIF)")
        self.capture_growth_check.setChecked(False)
        self.capture_growth_check.setToolTip(
            "Record the viewer during each Grow.\n"
            "Either save one GIF per Grow, or enable “Combine…” to build one GIF from "
            "several grows (frames are added only when you Merge; Reset branch preview "
            "discards the last grow’s frames).\n"
            "Uses the same steps as Animate growth; turn Animate on for a smooth clip."
        )
        anim_cap_row = QWidget()
        anim_cap_layout = QHBoxLayout(anim_cap_row)
        anim_cap_layout.setContentsMargins(0, 0, 0, 0)
        anim_cap_layout.addWidget(self.animate_check)
        anim_cap_layout.addWidget(self.capture_growth_check)
        anim_cap_layout.addStretch(1)
        vis_form.addRow(anim_cap_row)

        self.capture_options = QWidget()
        cap_form = QFormLayout(self.capture_options)
        cap_form.setContentsMargins(12, 0, 0, 0)

        self.capture_output_fps_spin = QDoubleSpinBox()
        self.capture_output_fps_spin.setRange(0.5, 60.0)
        self.capture_output_fps_spin.setValue(10.0)
        self.capture_output_fps_spin.setDecimals(1)
        self.capture_output_fps_spin.setSingleStep(0.5)
        self.capture_output_fps_spin.setToolTip(
            "GIF playback speed (frames per second in the exported file)."
        )
        cap_form.addRow("GIF playback FPS:", _row(self.capture_output_fps_spin))

        self.capture_subsample_spin = QSpinBox()
        self.capture_subsample_spin.setRange(1, 100)
        self.capture_subsample_spin.setValue(1)
        self.capture_subsample_spin.setToolTip(
            "Keep every N-th displayed grow step in the GIF (1 = all steps shown in the viewer)."
        )
        cap_form.addRow("Frame subsample (N):", _row(self.capture_subsample_spin))

        self.capture_max_frames_spin = QSpinBox()
        self.capture_max_frames_spin.setRange(0, 50_000)
        self.capture_max_frames_spin.setValue(800)
        self.capture_max_frames_spin.setSpecialValueText("No limit")
        self.capture_max_frames_spin.setToolTip(
            "Stop adding frames after this many (0 = no limit). The grow still runs to the end."
        )
        cap_form.addRow("Max frames:", _row(self.capture_max_frames_spin))

        self.capture_region_combo = QComboBox()
        self.capture_region_combo.addItems(["Viewer canvas", "Full napari window"])
        self.capture_region_combo.setToolTip(
            "Canvas: 3D viewer only. Full window: includes docks (e.g. this plugin panel)."
        )
        cap_form.addRow("Capture region:", _row(self.capture_region_combo))

        self.capture_scale_spin = QSpinBox()
        self.capture_scale_spin.setRange(10, 100)
        self.capture_scale_spin.setValue(100)
        self.capture_scale_spin.setSuffix(" %")
        self.capture_scale_spin.setToolTip(
            "Scale each frame to this percentage of width/height before saving (smaller GIF files)."
        )
        cap_form.addRow("Frame scale:", _row(self.capture_scale_spin))

        self.capture_gif_canvas_w_spin = QSpinBox()
        self.capture_gif_canvas_w_spin.setRange(0, 8192)
        self.capture_gif_canvas_w_spin.setValue(0)
        self.capture_gif_canvas_w_spin.setSpecialValueText("Auto")
        self.capture_gif_canvas_w_spin.setToolTip(
            "Output width in pixels for every GIF frame (letterboxed). "
            "Auto = use the widest frame in this save; set both width and height "
            "for a fixed canvas (recommended for combined GIFs)."
        )
        cap_form.addRow("GIF canvas width:", _row(self.capture_gif_canvas_w_spin))

        self.capture_gif_canvas_h_spin = QSpinBox()
        self.capture_gif_canvas_h_spin.setRange(0, 8192)
        self.capture_gif_canvas_h_spin.setValue(0)
        self.capture_gif_canvas_h_spin.setSpecialValueText("Auto")
        self.capture_gif_canvas_h_spin.setToolTip(
            "Output height in pixels for every GIF frame (letterboxed). "
            "Auto = use the tallest frame in this save; set both width and height "
            "for a fixed canvas (recommended for combined GIFs)."
        )
        cap_form.addRow("GIF canvas height:", _row(self.capture_gif_canvas_h_spin))

        self.capture_combine_grows_check = QCheckBox(
            "Combine branch grows in one GIF (commit on Merge)"
        )
        self.capture_combine_grows_check.setChecked(False)
        self.capture_combine_grows_check.setEnabled(False)
        self.capture_combine_grows_check.setToolTip(
            "Each Grow is recorded, but frames are appended to the combined clip only "
            "when you click Merge. Reset branch preview discards the last grow’s "
            "recording without merging. Starting a new Grow drops any unmerged "
            "recording from the previous attempt. Use Save combined GIF when done."
        )
        cap_form.addRow(_row(self.capture_combine_grows_check))

        self.btn_save_combined_gif = QPushButton("Save combined GIF…")
        self.btn_save_combined_gif.setEnabled(False)
        self.btn_save_combined_gif.setToolTip(
            "Save all segments committed with Merge while combine mode was on."
        )
        cap_form.addRow(_row(self.btn_save_combined_gif))

        self.capture_options.setVisible(False)
        vis_form.addRow(self.capture_options)

        layout.addWidget(
            _collapsible_section("Visualization", vis_inner, start_open=False)
        )

        # --- Segmentation (polyline + grow + merge; same flow for trunk and side branches) ---
        seg_inner = QWidget()
        seg_form = QFormLayout(seg_inner)

        self.branch_trunk_combo = QComboBox()
        self.branch_trunk_combo.setToolTip(
            "Labels layer that receives Merge and supplies the existing mask during Grow "
            "(union with the growing region). Same shape as Image. "
            "If the list is empty when you Grow or Merge, an empty Segmentation Result "
            "is created on this grid."
        )
        mask_row_widget = QWidget()
        mask_row = QHBoxLayout(mask_row_widget)
        mask_row.setContentsMargins(0, 0, 0, 0)
        mask_row.addWidget(self.branch_trunk_combo, 1)
        self.btn_new_segmentation_mask = QPushButton("New Mask")
        self.btn_new_segmentation_mask.setToolTip(
            "Add a new empty labels layer on the current image grid and select it here."
        )
        mask_row.addWidget(self.btn_new_segmentation_mask)
        seg_form.addRow("Segmentation mask:", mask_row_widget)

        pts_btn_row = QHBoxLayout()
        self.btn_reset_branch_points_layer = QPushButton("Reset branch points")
        self.btn_reset_branch_points_layer.setToolTip(
            "Clears all points in the layer currently selected under Branch points layer."
        )
        self.btn_new_branch_points_layer = QPushButton("New BranchPoints Layer")
        self.btn_new_branch_points_layer.setToolTip(
            "Adds a new points layer (e.g. BranchPoints_2) and selects it; "
            "existing branch point layers are kept."
        )
        pts_btn_row.addWidget(self.btn_new_branch_points_layer)
        pts_btn_row.addWidget(self.btn_reset_branch_points_layer)
        seg_form.addRow(pts_btn_row)

        self.branch_combo = QComboBox()
        self.branch_combo.setToolTip(
            "Points layer: all points in data order form one polyline (at least two points)."
        )
        seg_form.addRow("Branch points layer:", _row(self.branch_combo))

        blocker_row = QWidget()
        blocker_h = QHBoxLayout(blocker_row)
        blocker_h.setContentsMargins(0, 0, 0, 0)
        self.blocker_combo = QComboBox()
        self.blocker_combo.setToolTip(
            "Optional labels layer on the same grid as Image / pyramid level: painted "
            "foreground blocks Plain grow and MGAC (speed 0, mask cleared each step).\n"
            "Pick (none) to disable."
        )
        blocker_h.addWidget(self.blocker_combo, 1)
        self.btn_new_blocker_layer = QPushButton("New Blocker")
        self.btn_new_blocker_layer.setToolTip(
            f'Add an empty labels layer named "{BLOCKER_MASK_LAYER_NAME}" (or with suffix) aligned to the '
            "current Image and pyramid level — paint walls where segmentation must not leak."
        )
        blocker_h.addWidget(self.btn_new_blocker_layer)
        seg_form.addRow("Blocker_Mask (optional):", blocker_row)

        self.branch_ac_radius_spin = QDoubleSpinBox()
        self.branch_ac_radius_spin.setRange(0.5, 500.0)
        self.branch_ac_radius_spin.setValue(60.0)
        self.branch_ac_radius_spin.setSingleStep(0.5)
        self.branch_ac_radius_spin.setToolTip(
            "Finest-level isotropic voxel radii: physical radius is value × min(finest spacing),\n"
            "independent of the pyramid level used for Grow. Builds the polyline seed tube "
            "for Plain and MGAC and is the MGAC nominal radius for 3D Active Contour."
        )

        self.branch_method_combo = QComboBox()
        self.branch_method_combo.addItems(
            [
                "Plain Region Growing",
                "3D Active Contour",
            ]
        )
        self.branch_method_combo.setToolTip(
            "Fill method for Compute Branch.\n"
            "Plain = priority-queue region growing. 3D Active Contour = morphological MGAC on the edge image."
        )
        seg_form.addRow("Fill with:", _row(self.branch_method_combo))

        branch_params_inner = QWidget()
        branch_params_layout = QVBoxLayout(branch_params_inner)
        branch_params_layout.setContentsMargins(0, 0, 0, 0)
        branch_tube_form = QFormLayout()
        branch_tube_form.addRow("Seed tube radius (vox):", _row(self.branch_ac_radius_spin))
        branch_params_layout.addLayout(branch_tube_form)

        branch_plain_inner = QWidget()
        branch_plain_form = QFormLayout(branch_plain_inner)
        self.branch_plain_section = _collapsible_section(
            "Plain parameters (branch)", branch_plain_inner, start_open=True
        )
        self.branch_plain_section.setToolTip(
            "Used when Fill with is Plain Region Growing."
        )

        self.branch_plain_sigma_spin = QDoubleSpinBox()
        self.branch_plain_sigma_spin.setRange(0.1, 20.0)
        self.branch_plain_sigma_spin.setValue(2.0)
        self.branch_plain_sigma_spin.setSingleStep(0.5)
        self.branch_plain_sigma_spin.setToolTip(
            "Gaussian σ for edge costs (branch plain only)."
        )
        branch_plain_form.addRow("Smoothing σ:", _row(self.branch_plain_sigma_spin))

        self.branch_plain_flux_spin = QDoubleSpinBox()
        self.branch_plain_flux_spin.setRange(0.0, 50.0)
        self.branch_plain_flux_spin.setValue(15.0)
        self.branch_plain_flux_spin.setSingleStep(1.0)
        self.branch_plain_flux_spin.setDecimals(1)
        self.branch_plain_flux_spin.setToolTip("Flux penalty weight (branch plain only).")
        branch_plain_form.addRow("Flux penalty:", _row(self.branch_plain_flux_spin))

        self.branch_plain_intensity_tol_spin = QDoubleSpinBox()
        self.branch_plain_intensity_tol_spin.setRange(0.5, 10.0)
        self.branch_plain_intensity_tol_spin.setValue(3.0)
        self.branch_plain_intensity_tol_spin.setSingleStep(0.5)
        self.branch_plain_intensity_tol_spin.setToolTip(
            "Intensity gate: σ below region mean (branch plain only)."
        )
        branch_plain_form.addRow(
            "Intensity tolerance:", _row(self.branch_plain_intensity_tol_spin)
        )

        self.branch_plain_cost_budget_spin = QDoubleSpinBox()
        self.branch_plain_cost_budget_spin.setRange(0.0, 100000.0)
        self.branch_plain_cost_budget_spin.setValue(0.0)
        self.branch_plain_cost_budget_spin.setDecimals(1)
        self.branch_plain_cost_budget_spin.setToolTip(
            "Accumulated growth-cost budget; 0 = auto (branch plain only)."
        )
        branch_plain_form.addRow(
            "Cost budget (0=auto):", _row(self.branch_plain_cost_budget_spin)
        )

        self.branch_plain_margin_spin = QDoubleSpinBox()
        self.branch_plain_margin_spin.setRange(0, 200)
        self.branch_plain_margin_spin.setValue(0.0)
        self.branch_plain_margin_spin.setToolTip(
            "Length slack along the branch axis: margin × mean(finest spacing), scaled to\n"
            "the working level (stable across pyramid levels)."
        )
        branch_plain_form.addRow("Length margin:", _row(self.branch_plain_margin_spin))

        self.branch_plain_upper_thr_check = QCheckBox("Enable upper threshold")
        self.branch_plain_upper_thr_check.setChecked(False)
        self.branch_plain_upper_thr_check.setToolTip(
            "Hard intensity cap for branch plain (same idea as trunk Plain tab)."
        )
        branch_plain_form.addRow(_row(self.branch_plain_upper_thr_check))

        self.branch_plain_upper_thr_combo = QComboBox()
        self.branch_plain_upper_thr_combo.addItems([
            "Otsu",
            "Triangle",
            "Li",
            "90th percentile",
            "95th percentile",
        ])
        self.branch_plain_upper_thr_combo.setEnabled(False)
        self.branch_plain_upper_thr_combo.setToolTip(
            "Method for upper threshold when enabled (branch plain only)."
        )
        branch_plain_form.addRow("Threshold method:", _row(self.branch_plain_upper_thr_combo))

        self.branch_plain_step_spin = QSpinBox()
        self.branch_plain_step_spin.setRange(1, 10000)
        self.branch_plain_step_spin.setValue(500)
        self.branch_plain_step_spin.setToolTip(
            "Animate branch plain growth every N accepted voxels when Animate growth is on."
        )
        branch_plain_form.addRow("Every N voxels:", _row(self.branch_plain_step_spin))

        branch_ac_inner = QWidget()
        branch_ac_form = QFormLayout(branch_ac_inner)
        self.branch_ac_section = _collapsible_section(
            "MGAC parameters (branch)", branch_ac_inner, start_open=True
        )
        self.branch_ac_section.setToolTip(
            "Shown when Fill with is 3D Active Contour (MGAC on the polyline tube)."
        )

        self.branch_ac_margin_spin = QDoubleSpinBox()
        self.branch_ac_margin_spin.setRange(0, 200)
        self.branch_ac_margin_spin.setValue(0.15)
        self.branch_ac_margin_spin.setToolTip(
            "Extra slack for the MGAC polyline corridor (same convention as Plain length margin:\n"
            "margin × mean(finest spacing), scaled to the working pyramid level)."
        )
        branch_ac_form.addRow("Corridor length margin:", _row(self.branch_ac_margin_spin))

        self.branch_ac_sigma_spin = QDoubleSpinBox()
        self.branch_ac_sigma_spin.setRange(0.1, 20.0)
        self.branch_ac_sigma_spin.setValue(10.0)
        self.branch_ac_sigma_spin.setSingleStep(0.5)
        self.branch_ac_sigma_spin.setToolTip(
            "Gaussian σ for the inverse-gradient edge image (branch MGAC only)."
        )
        branch_ac_form.addRow("Smoothing σ:", _row(self.branch_ac_sigma_spin))

        self.branch_ac_low_clip_spin = QDoubleSpinBox()
        self.branch_ac_low_clip_spin.setRange(0.0, 100000.0)
        self.branch_ac_low_clip_spin.setValue(0.0)
        self.branch_ac_low_clip_spin.setSingleStep(1.0)
        self.branch_ac_low_clip_spin.setDecimals(1)
        self.branch_ac_low_clip_spin.setToolTip(
            "Optional lumen/background flattening: set all intensities ≤ this value to 0 before MGAC.\n"
            "Useful when lumen/background is ~1–20 but noisy; reduces micro-gradients that create holes.\n"
            "Set to 0 to disable."
        )
        branch_ac_form.addRow("Low-intensity clip (≤ → 0):", _row(self.branch_ac_low_clip_spin))

        self.branch_ac_balloon_spin = QDoubleSpinBox()
        self.branch_ac_balloon_spin.setRange(-5.0, 5.0)
        self.branch_ac_balloon_spin.setValue(0.5)
        self.branch_ac_balloon_spin.setSingleStep(0.1)
        self.branch_ac_balloon_spin.setDecimals(2)
        self.branch_ac_balloon_spin.setToolTip("Balloon coefficient (branch MGAC only).")
        branch_ac_form.addRow("Balloon:", _row(self.branch_ac_balloon_spin))

        self.branch_ac_smoothing_spin = QSpinBox()
        self.branch_ac_smoothing_spin.setRange(0, 10)
        self.branch_ac_smoothing_spin.setValue(0)
        self.branch_ac_smoothing_spin.setToolTip(
            "Morphological curvature (_curvop) per MGAC iteration after balloon+edge.\n"
            "Higher values act like a stronger smoothing / shrink prior (γ-like).\n"
            "0 avoids thinning narrow tubes. A light binary closing after each chunk\n"
            "still fills 1-voxel stripe gaps."
        )
        branch_ac_form.addRow("Smoothing γ (steps):", _row(self.branch_ac_smoothing_spin))

        self.branch_ac_total_iter_spin = QSpinBox()
        self.branch_ac_total_iter_spin.setRange(1, 5000)
        self.branch_ac_total_iter_spin.setValue(85)
        self.branch_ac_total_iter_spin.setToolTip("Total MGAC iterations (branch only).")
        branch_ac_form.addRow("Total iterations:", _row(self.branch_ac_total_iter_spin))

        self.branch_ac_yield_spin = QSpinBox()
        self.branch_ac_yield_spin.setRange(1, 500)
        self.branch_ac_yield_spin.setValue(5)
        self.branch_ac_yield_spin.setToolTip(
            "Refresh the viewer every N MGAC iterations when Animate growth is on."
        )
        branch_ac_form.addRow("Every N iterations:", _row(self.branch_ac_yield_spin))

        branch_params_layout.addWidget(self.branch_plain_section)
        branch_params_layout.addWidget(self.branch_ac_section)
        seg_form.addRow(
            _collapsible_section(
                "Segmentation parameters (tube + Plain / MGAC)",
                branch_params_inner,
            )
        )

        self.btn_grow_branches = QPushButton("Compute Branch")
        self.btn_grow_branches.setToolTip(
            "Uses all points in the branch layer (in order) as one polyline (at least two points).\n"
            f'Writes the current computation result to "{DRAFT_BRANCH_LAYER_NAME}".'
        )
        grow_row = QHBoxLayout()
        grow_row.addWidget(self.btn_grow_branches)
        self.btn_reset_branch_seg = QPushButton("Reset Branch")
        self.btn_reset_branch_seg.setToolTip(
            f'Clears "{DRAFT_BRANCH_LAYER_NAME}" and shows the branch points layer again '
            "(does not remove points)."
        )
        grow_row.addWidget(self.btn_reset_branch_seg)
        self.btn_merge_branch_seg = QPushButton("Merge Branch")
        self.btn_merge_branch_seg.setToolTip(
            f'Logical OR from "{DRAFT_BRANCH_LAYER_NAME}" into the merged labels layer.'
        )
        grow_row.addWidget(self.btn_merge_branch_seg)
        seg_form.addRow(grow_row)

        seg_ctrl = QHBoxLayout()
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        seg_ctrl.addWidget(self.btn_stop)
        seg_form.addRow(seg_ctrl)

        layout.addWidget(
            _collapsible_section(
                "Segmentation",
                seg_inner,
                header_tooltip=(
                    "Add ordered points along one vessel segment, Compute Branch, then Merge Branch. "
                    "Use Blocker_Mask to paint walls where segmentation must not leak."
                ),
            )
        )

        # --- Post-processing (mask on working grid must exist; use Merge first) ---
        post_inner = QWidget()
        post_form = QFormLayout(post_inner)

        self.btn_postprocess = QPushButton("Upsample Result to Original Size")
        self.btn_postprocess.setToolTip(
            "Enabled when the segmentation mask shape differs from the original "
            "image grid (e.g. after **Grow** on a coarser pyramid level).\n"
            "Creates a full-resolution mask by zooming to the original shape."
        )
        self.btn_postprocess.setEnabled(False)
        post_form.addRow(_row(self.btn_postprocess))

        self.morph_op_combo = QComboBox()
        self.morph_op_combo.addItems(["None", "Dilation", "Erosion"])
        self.morph_op_combo.setCurrentIndex(0)
        self.morph_op_combo.setToolTip(
            "Apply morphological refinement to the segmentation:\n"
            "  - Dilation: expands the mask (fills tiny gaps/connects close parts)\n"
            "  - Erosion: shrinks the mask (removes thin over-segmentation)\n\n"
            "For anisotropic volumes, a common cleanup is one erosion\n"
            "with radius = 1 to reduce slight extra thickness along Z."
        )
        post_form.addRow("Operation:", _row(self.morph_op_combo))

        self.morph_radius_spin = QSpinBox()
        self.morph_radius_spin.setRange(1, 50)
        self.morph_radius_spin.setValue(1)
        self.morph_radius_spin.setToolTip(
            "Ball radius (voxels) for morphological refinement.\n"
            "Start with radius = 1 for minimal correction.\n"
            "In anisotropic images, radius-1 erosion is often enough\n"
            "to clean slight Z-direction over-segmentation."
        )
        post_form.addRow("Ball radius (vox):", _row(self.morph_radius_spin))

        self.btn_apply_morph = QPushButton("Apply Morphological Operation")
        self.btn_apply_morph.setToolTip(
            "Apply the selected operation to the tracked segmentation layer\n"
            "after visual inspection. Creates a new refined result layer."
        )
        post_form.addRow(_row(self.btn_apply_morph))

        layout.addWidget(
            _collapsible_section(
                "Post-processing",
                post_inner,
                start_open=False,
                header_tooltip=(
                    "Uses the last merged labels layer tracked by the plugin (often Segmentation Result). "
                    "Run Merge first if upsample or morphology is disabled."
                ),
            )
        )

        # --- Shared: Status ---
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self._update_branch_method_dependent_widgets()
        self._update_prep_stretch_mode_widgets()
        self._apply_combo_width_policies()
        layout.addStretch()

    def _apply_combo_width_policies(self) -> None:
        """Avoid dock layout jump when layer names are long."""
        for cb in (
            self.image_combo,
            self.branch_combo,
            self.blocker_combo,
            self.morph_op_combo,
            self.branch_method_combo,
            self.branch_plain_upper_thr_combo,
            self.branch_trunk_combo,
            self.capture_region_combo,
        ):
            cb.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon
            )
            cb.setMinimumContentsLength(16)
            cb.setMaximumWidth(260)
            cb.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    # ------------------------------------------------------------- signals --
    def _connect_signals(self):
        self.btn_reset_branch_points_layer.clicked.connect(
            self._reset_branch_points_layer
        )
        self.btn_new_branch_points_layer.clicked.connect(
            self._create_new_branch_points_layer
        )
        self.btn_new_segmentation_mask.clicked.connect(
            self._create_new_segmentation_mask_layer
        )
        self.btn_new_blocker_layer.clicked.connect(
            self._create_new_blocker_labels_layer
        )
        self.btn_grow_branches.clicked.connect(self._grow_branches_into_result)
        self.btn_reset_branch_seg.clicked.connect(self._reset_branch_segmentation)
        self.btn_merge_branch_seg.clicked.connect(self._merge_branch_segmentation)
        self.branch_method_combo.currentTextChanged.connect(
            self._update_branch_method_dependent_widgets
        )
        self.branch_plain_upper_thr_check.toggled.connect(
            self.branch_plain_upper_thr_combo.setEnabled
        )
        self.btn_apply_preprocess.clicked.connect(self._apply_preprocessing)
        self.btn_apply_threshold.clicked.connect(self._apply_threshold_mask)
        self.prep_stretch_mode_combo.currentTextChanged.connect(
            self._update_prep_stretch_mode_widgets
        )
        self.btn_postprocess.clicked.connect(self._upsample_result_to_original)
        self.btn_apply_morph.clicked.connect(self._apply_morphological_operation)
        self.btn_stop.clicked.connect(self._stop)
        self.image_combo.currentTextChanged.connect(self._on_image_selection_changed)
        self.ms_level_combo.currentIndexChanged.connect(self._refresh_branch_trunk_combo)
        self.capture_growth_check.toggled.connect(self._on_capture_growth_toggled)
        self.capture_combine_grows_check.toggled.connect(
            self._on_capture_combine_toggled
        )
        self.btn_save_combined_gif.clicked.connect(self._save_combined_growth_gif)
        self.viewer.camera.events.zoom.connect(self._on_camera_for_branch_points)
        self.viewer.camera.events.center.connect(self._on_camera_for_branch_points)

    # -------------------------------------------------------- layer helpers --
    def _selected_pyramid_level(self) -> int:
        if not self.ms_level_combo.isVisible() or self.ms_level_combo.count() == 0:
            return 0
        return int(self.ms_level_combo.currentIndex())

    def _refresh_multiscale_level_combo(self, *_args: Any) -> None:
        iname = self.image_combo.currentText()
        lyr = None
        if iname and iname in self.viewer.layers:
            cand = self.viewer.layers[iname]
            if isinstance(cand, napari.layers.Image):
                lyr = cand
        if lyr is None:
            self.ms_level_combo.blockSignals(True)
            self.ms_level_combo.clear()
            self.ms_level_combo.blockSignals(False)
            self.ms_level_combo.setVisible(False)
            self._ms_level_row_label.setVisible(False)
            self._refresh_branch_trunk_combo()
            return
        n = multiscale_level_count(lyr)
        if n <= 1:
            self.ms_level_combo.blockSignals(True)
            self.ms_level_combo.clear()
            self.ms_level_combo.blockSignals(False)
            self.ms_level_combo.setVisible(False)
            self._ms_level_row_label.setVisible(False)
            self._refresh_branch_trunk_combo()
            return
        prev = self.ms_level_combo.currentIndex()
        self.ms_level_combo.blockSignals(True)
        self.ms_level_combo.clear()
        for i in range(n):
            self.ms_level_combo.addItem(multiscale_level_label(lyr, i))
        idx = prev if 0 <= prev < n else 0
        self.ms_level_combo.setCurrentIndex(idx)
        self.ms_level_combo.blockSignals(False)
        self.ms_level_combo.setVisible(True)
        self._ms_level_row_label.setVisible(True)
        self._refresh_branch_trunk_combo()

    def _on_image_selection_changed(self, *_args: Any) -> None:
        self._refresh_multiscale_level_combo()
        self._update_postprocess_button()
        self._sync_branch_point_bases_from_image()

    def _refresh_layers(self, event=None):
        for combo, layer_type in [
            (self.image_combo, napari.layers.Image),
            (self.branch_combo, napari.layers.Points),
        ]:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for layer in self.viewer.layers:
                if isinstance(layer, layer_type):
                    combo.addItem(layer.name)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self._refresh_multiscale_level_combo()
        self._update_postprocess_button()
        alive = {lyr.name for lyr in self.viewer.layers}
        for k in list(self._branch_point_size_bases.keys()):
            if k not in alive:
                del self._branch_point_size_bases[k]
        self._refresh_branch_trunk_combo()
        self._sync_branch_point_bases_from_image()

    def _sync_branch_point_bases_from_image(self) -> None:
        """Assign default marker sizes for BranchPoints* from the current image extent."""
        iname = self.image_combo.currentText()
        if not iname or iname not in self.viewer.layers:
            self._on_camera_for_branch_points()
            return
        img = self.viewer.layers[iname]
        if not isinstance(img, napari.layers.Image):
            self._on_camera_for_branch_points()
            return
        base = _suggested_branch_point_base_size(img)
        for lyr in self.viewer.layers:
            if not isinstance(lyr, napari.layers.Points):
                continue
            if not _is_auto_sized_branch_points_name(lyr.name):
                continue
            self._branch_point_size_bases.setdefault(lyr.name, base)
        self._on_camera_for_branch_points()

    def _on_camera_for_branch_points(self, event=None) -> None:
        """Keep BranchPoints* marker diameter ~stable on screen as the camera zoom changes."""
        try:
            z = float(self.viewer.camera.zoom)
        except (TypeError, ValueError):
            return
        if z <= 1e-9 or np.isnan(z):
            return
        if self._branch_point_zoom_ref is None:
            self._branch_point_zoom_ref = z
        z0 = float(self._branch_point_zoom_ref)
        scale = z / max(z0, 1e-9)
        for lyr in self.viewer.layers:
            if not isinstance(lyr, napari.layers.Points):
                continue
            if not _is_auto_sized_branch_points_name(lyr.name):
                continue
            if len(lyr.data) == 0:
                continue
            base = self._branch_point_size_bases.get(lyr.name)
            if base is None:
                continue
            new_size = float(np.clip(base * scale, 8.0, 900.0))
            if len(lyr.size):
                cur = float(np.max(lyr.size))
                if abs(cur - new_size) <= max(0.5, 0.02 * new_size):
                    continue
            lyr.size = new_size

    def _on_ndisplay_changed(self, event=None) -> None:
        """Avoid VisPy GLError (invalid glTexSubImage2D) when ndisplay=2 and Points.data is empty."""
        try:
            nd = int(self.viewer.dims.ndisplay)
        except (TypeError, ValueError, AttributeError):
            return
        for lyr in list(self.viewer.layers):
            if not isinstance(lyr, napari.layers.Points):
                continue
            if not _is_auto_sized_branch_points_name(lyr.name):
                continue
            md = dict(getattr(lyr, "metadata", {}) or {})
            was_auto_hidden = bool(md.get("_rg_hide_empty_2d"))

            # Only auto-hide on an actual 3D→2D switch event. Keep newly created empty
            # BranchPoints layers visible in 2D so the user can start drawing immediately.
            if event is not None and nd <= 2 and len(lyr.data) == 0:
                if lyr.visible:
                    lyr.metadata = {**md, "_rg_hide_empty_2d": True}
                    lyr.visible = False
            elif was_auto_hidden and (nd >= 3 or len(lyr.data) > 0):
                lyr.metadata = {
                    k: v for k, v in md.items() if k != "_rg_hide_empty_2d"
                }
                lyr.visible = True

    def _update_postprocess_button(self):
        name = self.image_combo.currentText()
        info = self._preprocessed_images.get(name)
        meta = self._image_working_metadata.get(name)
        enabled = False
        if info is not None:
            orig = tuple(info.get("original_shape", ()))
            work = tuple(info.get("working_shape", orig))
            if orig and work and orig != work:
                enabled = True
            elif int(info.get("factor", 1)) > 1:
                enabled = True
        elif meta is not None:
            o = tuple(meta.get("finest_shape", ()))
            w = tuple(meta.get("working_shape", ()))
            if o and w and o != w:
                enabled = True
        self.btn_postprocess.setEnabled(enabled)

    def _update_prep_stretch_mode_widgets(self) -> None:
        fixed = self.prep_stretch_mode_combo.currentText() == "fixed"
        self.prep_pct_low_spin.setVisible(not fixed)
        self.prep_pct_high_spin.setVisible(not fixed)
        self.prep_fixed_bg_spin.setVisible(fixed)
        self.prep_fixed_max_spin.setVisible(fixed)

    def _collect_preprocess_params(self) -> dict:
        return dict(
            apply_denoise=self.prep_denoise_check.isChecked(),
            denoise_patch_size=int(self.prep_denoise_patch_spin.value()),
            denoise_patch_distance=int(self.prep_denoise_dist_spin.value()),
            denoise_h=float(self.prep_denoise_h_spin.value()),
            apply_stretch=self.prep_stretch_check.isChecked(),
            stretch_mode=self.prep_stretch_mode_combo.currentText(),
            percentile_low=float(self.prep_pct_low_spin.value()),
            percentile_high=float(self.prep_pct_high_spin.value()),
            fixed_background=float(self.prep_fixed_bg_spin.value()),
            fixed_vessel_max=float(self.prep_fixed_max_spin.value()),
            out_dtype=self.prep_out_dtype_combo.currentText(),
        )

    def _apply_preprocessing(self) -> None:
        if self._worker is not None:
            self.status_label.setText(
                "Wait for the current Grow job to finish (or Stop)."
            )
            return
        if self._preprocess_worker is not None:
            self.status_label.setText("Preprocessing is already running.")
            return
        name = self.image_combo.currentText()
        if not name or name not in self.viewer.layers:
            self.status_label.setText("Select an image layer first.")
            return
        layer = self.viewer.layers[name]
        if not isinstance(layer, napari.layers.Image):
            self.status_label.setText("Selected layer is not an Image.")
            return
        level = self._selected_pyramid_level()
        if image_level_is_lazy(layer, level):
            self.status_label.setText(
                "Preprocessing needs the image in RAM. For Zarr/Dask sources, run "
                "``regiongrow-preprocess-zarr`` to write a smaller array you can open in napari, "
                "or duplicate the layer after exporting a TIFF from napari."
            )
            return
        shape_work = image_level_shape(layer, level)
        if len(shape_work) != 3:
            self.status_label.setText("Image must be 3-D.")
            return
        msg = check_materialization_budget(
            shape_work,
            np.dtype(np.float64),
            max_bytes=_MAX_MATERIALIZE_BYTES,
            copies=2.0,
            context="Preprocessing",
        )
        if msg:
            self.status_label.setText(msg)
            return
        finest_shape = (
            image_finest_shape(layer)
            if is_multiscale_image_layer(layer)
            else shape_work
        )
        arr0 = materialize_image_level(layer, level, dtype=None)
        if arr0.ndim != 3:
            self.status_label.setText("Image must be 3-D.")
            return

        p = self._collect_preprocess_params()
        if not (p["apply_denoise"] or p["apply_stretch"]):
            self.status_label.setText("Enable at least one preprocessing step.")
            return

        replace = self.prep_radio_replace.isChecked()
        if replace:
            r = QMessageBox.question(
                self,
                "Replace layer",
                f'Overwrite data in layer "{name}"? This cannot be undone.',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                return

        upsample_target_shape = tuple(finest_shape)
        spacing0 = voxel_spacing_zyx_for_level(layer, level, shape_work)
        arr_copy = arr0.copy()

        if p["apply_denoise"]:
            self.btn_apply_preprocess.setEnabled(False)
            self.btn_grow_branches.setEnabled(False)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.show()
            self.status_label.setText(
                "Preprocessing (3D non-local means is slow on large volumes)…"
            )

            @thread_worker
            def _work():
                out, sp, meta = apply_preprocess_chain(arr_copy, spacing0, **p)
                yield ("result", out, sp, meta)

            worker = _work()

            def _on_yield(msg):
                if not isinstance(msg, tuple) or not msg:
                    return
                if msg[0] == "result":
                    self._finish_preprocess_output(
                        layer,
                        name,
                        upsample_target_shape,
                        msg[1],
                        msg[2],
                        msg[3],
                        replace,
                    )

            def _on_err(exc):
                self.status_label.setText(
                    "Preprocessing error: "
                    f"{self._worker_exception_message(exc)}"
                )

            worker.yielded.connect(_on_yield)
            worker.errored.connect(_on_err)
            worker.finished.connect(
                lambda *args: self._cleanup_preprocess_worker_ui()
            )
            worker.start()
            self._preprocess_worker = worker
            return

        out, sp, meta = apply_preprocess_chain(arr_copy, spacing0, **p)
        self._finish_preprocess_output(
            layer, name, upsample_target_shape, out, sp, meta, replace
        )

    def _finish_preprocess_output(
        self,
        layer: Any,
        layer_name: str,
        orig_shape: tuple,
        out_arr: np.ndarray,
        spacing_zyx: tuple,
        meta: dict,
        replace: bool,
    ) -> None:
        skw = spatial_alignment_kwargs(layer)
        sc = np.asarray(skw.get("scale", np.ones(3)), dtype=np.float64).ravel()
        if sc.size < 3:
            sc = np.ones(3)
        sc = sc[-3:].copy()
        sc[0] = float(spacing_zyx[0])
        sc[1] = float(spacing_zyx[1])
        sc[2] = float(spacing_zyx[2])
        if skw.get("scale") is not None:
            full = np.asarray(skw["scale"], dtype=np.float64).copy().ravel()
            if full.size >= 3:
                full[-3:] = sc
                skw["scale"] = full
            else:
                skw["scale"] = sc
        else:
            skw["scale"] = sc

        out_name = f"{layer_name} preprocessed"
        n = 1
        while out_name in self.viewer.layers:
            n += 1
            out_name = f"{layer_name} preprocessed ({n})"

        if replace:
            try:
                layer.data = out_arr
                if "scale" in skw:
                    layer.scale = tuple(np.asarray(skw["scale"], dtype=float).tolist())
                reg_name = layer_name
                self._preprocessed_images[reg_name] = {
                    "original_name": layer_name,
                    "original_shape": orig_shape,
                    "working_shape": tuple(out_arr.shape),
                    "meta": meta,
                }
                self._refresh_layers()
                self.image_combo.setCurrentText(reg_name)
                self.status_label.setText(
                    f'Updated "{reg_name}" ({out_arr.dtype}, shape {out_arr.shape}).'
                )
            except Exception as exc:
                self.status_label.setText(
                    f"Replace failed ({exc}); try Add new layer instead."
                )
            self._update_postprocess_button()
            return

        if out_name in self.viewer.layers:
            self.viewer.layers.remove(out_name)
        self.viewer.add_image(out_arr, name=out_name, **skw)
        self._preprocessed_images[out_name] = {
            "original_name": layer_name,
            "original_shape": orig_shape,
            "working_shape": tuple(out_arr.shape),
            "meta": meta,
        }
        self._refresh_layers()
        self.image_combo.setCurrentText(out_name)
        self.status_label.setText(
            f'Added "{out_name}" ({out_arr.dtype}, shape {out_arr.shape}).'
        )
        self._update_postprocess_button()

    def _ensure_threshold_mask_layer(
        self, image_layer: Any, shape: tuple, pyramid_level: int
    ) -> Any:
        """Labels layer (0/1) for thresholding."""
        skw = spatial_alignment_for_pyramid_level(image_layer, int(pyramid_level))
        if THRESHOLD_MASK_LAYER_NAME in self.viewer.layers:
            lyr = self.viewer.layers[THRESHOLD_MASK_LAYER_NAME]
            if tuple(lyr.data.shape) != tuple(shape):
                lyr.data = np.zeros(shape, dtype=np.uint32)
            _apply_spatial_kwargs_to_layer(lyr, skw)
            try:
                lyr.opacity = 0.45
                lyr.color = {1: "yellow"}
            except Exception:
                pass
            return lyr
        lyr = self.viewer.add_labels(
            np.zeros(shape, dtype=np.uint32),
            name=THRESHOLD_MASK_LAYER_NAME,
            opacity=0.45,
            **skw,
        )
        try:
            lyr.color = {1: "yellow"}
        except Exception:
            pass
        return lyr

    def _threshold_value(self, arr: np.ndarray, method_label: str) -> float:
        from skimage.filters import (
            threshold_otsu,
            threshold_triangle,
            threshold_li,
            threshold_yen,
        )

        a = np.asarray(arr, dtype=np.float64)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return float("nan")
        if finite.size > 12_000_000:
            rng = np.random.default_rng(42)
            finite = finite[rng.choice(finite.size, size=12_000_000, replace=False)]

        m = method_label.strip().lower()
        if m == "otsu":
            return float(threshold_otsu(finite))
        if m == "li":
            return float(threshold_li(finite))
        if m == "triangle":
            return float(threshold_triangle(finite))
        if m == "yen":
            return float(threshold_yen(finite))
        if m == "mean":
            return float(np.mean(finite))
        if "90" in m:
            return float(np.percentile(finite, 90))
        if "95" in m:
            return float(np.percentile(finite, 95))
        raise ValueError(f"Unknown threshold method: {method_label!r}")

    def _apply_threshold_mask(self) -> None:
        """3D view: threshold full volume. 2D view: only current visible plane."""
        if self._worker is not None:
            self.status_label.setText(
                "Wait for the current Grow job to finish (or Stop)."
            )
            return
        if self._preprocess_worker is not None:
            self.status_label.setText("Wait for preprocessing to finish.")
            return
        name = self.image_combo.currentText()
        if not name or name not in self.viewer.layers:
            self.status_label.setText("Select an image layer first.")
            return
        image_layer = self.viewer.layers[name]
        if not isinstance(image_layer, napari.layers.Image):
            self.status_label.setText("Selected layer is not an Image.")
            return

        level = self._selected_pyramid_level()
        shape_work = image_level_shape(image_layer, level)
        if len(shape_work) != 3:
            self.status_label.setText("Image must be 3-D.")
            return

        method = self.prep_thr_method_combo.currentText()
        nd = int(getattr(self.viewer.dims, "ndisplay", 2))
        thr_layer = self._ensure_threshold_mask_layer(image_layer, shape_work, level)

        if nd >= 3:
            if image_level_is_lazy(image_layer, level):
                self.status_label.setText(
                    "3D threshold mask needs the full image in RAM. "
                    "Pick a coarser pyramid level or preprocess/export first."
                )
                return
            msg = check_materialization_budget(
                shape_work,
                np.dtype(np.float32),
                max_bytes=_MAX_MATERIALIZE_BYTES,
                copies=2.0,
                context="Threshold mask (3D)",
            )
            if msg:
                self.status_label.setText(msg)
                return
            arr = materialize_image_level(image_layer, level, dtype=np.float32)
            t = self._threshold_value(arr, method)
            if not np.isfinite(t):
                self.status_label.setText("Threshold failed (no finite values).")
                return
            thr_layer.data = (np.asarray(arr) >= float(t)).astype(np.uint32)
            self.status_label.setText(f'Threshold_Mask (3D): {method} thr={t:.3g}')
            return

        # 2D: apply to current visible plane only (respect current displayed axes)
        try:
            displayed = tuple(int(a) for a in self.viewer.dims.displayed)
        except Exception:
            displayed = (1, 2)
        steps = tuple(
            int(s)
            for s in getattr(self.viewer.dims, "current_step", (0, 0, 0))
        )
        slc = []
        for ax in range(3):
            if ax in displayed:
                slc.append(slice(None))
            else:
                slc.append(int(steps[ax]) if ax < len(steps) else 0)
        slc_t = tuple(slc)
        plane = materialize_image_level(
            image_layer, level, dtype=np.float32, slices=slc_t
        )
        t = self._threshold_value(plane, method)
        if not np.isfinite(t):
            self.status_label.setText("Threshold failed (no finite values).")
            return
        out_plane = (np.asarray(plane) >= float(t))
        data = np.asarray(thr_layer.data)
        data[slc_t] = out_plane.astype(np.uint32)
        thr_layer.data = data.astype(np.uint32, copy=False)
        self.status_label.setText(
            f'Threshold_Mask (2D plane): {method} thr={t:.3g} on axes {displayed}'
        )

    def _upsample_result_to_original(self):
        res_layer = self._result_layer
        if res_layer is None or res_layer not in self.viewer.layers:
            res_layer = self._current_segmentation_target_layer()
        if res_layer is None or res_layer not in self.viewer.layers:
            self.status_label.setText(
                "No labels mask on this grid — run Merge first "
                '(e.g. into "Segmentation Result").'
            )
            return

        image_name = self.image_combo.currentText()
        info = self._preprocessed_images.get(image_name)
        meta = self._image_working_metadata.get(image_name)
        if info is not None:
            target_shape = tuple(info["original_shape"])
            orig_name = str(info["original_name"])
        elif meta is not None:
            target_shape = tuple(meta["finest_shape"])
            orig_name = str(meta.get("base_image_name", image_name))
        else:
            self.status_label.setText(
                "No upsample metadata for this image — use a preprocessed layer or Grow on a pyramid level."
            )
            self._update_postprocess_button()
            return

        if orig_name not in self.viewer.layers:
            self.status_label.setText(f'Original layer "{orig_name}" is not in the viewer.')
            self._update_postprocess_button()
            return

        mask = np.asarray(res_layer.data) > 0
        if tuple(mask.shape) == target_shape:
            self.status_label.setText(
                "Mask shape already matches the original image — upsampling not needed."
            )
            self._update_postprocess_button()
            return

        zoom = [o / s for o, s in zip(target_shape, mask.shape)]
        upsampled = ndimage_zoom(mask.astype(np.float64), zoom, order=0) > 0.5

        result_name = "Segmentation Result (Original Size)"
        if result_name in self.viewer.layers:
            self.viewer.layers[result_name].data = upsampled.astype(np.int32)
        else:
            orig_layer = self.viewer.layers[orig_name]
            self.viewer.add_labels(
                upsampled.astype(np.int32),
                name=result_name,
                opacity=0.5,
                **spatial_alignment_kwargs(orig_layer),
            )
        self._result_layer = res_layer
        self.status_label.setText("Postprocessing complete: upsampled result created.")

    def _apply_morphological_operation(self):
        """Apply morphological dilation or erosion to the result layer."""
        res_layer = self._result_layer
        if res_layer is None or res_layer not in self.viewer.layers:
            res_layer = self._current_segmentation_target_layer()
        if res_layer is None or res_layer not in self.viewer.layers:
            self.status_label.setText(
                "No labels mask on this grid — run Merge first."
            )
            return

        operation = self.morph_op_combo.currentText()
        if operation == "None":
            self.status_label.setText("Select an operation (Dilation or Erosion).")
            return

        radius = self.morph_radius_spin.value()
        mask = np.asarray(res_layer.data) > 0

        # Create ball structuring element
        struct = generate_binary_structure(3, 3)  # 3D, full connectivity
        struct = struct.astype(bool)
        # Scale the structuring element by radius (approx. ball shape)
        if radius > 1:
            # Create larger ball by repeated dilation
            for _ in range(radius - 1):
                struct = binary_dilation(struct, structure=struct)

        # Apply operation
        if operation == "Dilation":
            result = binary_dilation(mask, structure=struct)
            op_name = "Dilation"
        elif operation == "Erosion":
            result = binary_erosion(mask, structure=struct)
            op_name = "Erosion"
        else:
            self.status_label.setText("Unknown operation.")
            return

        result_name = f"Segmentation Result ({op_name} r={radius})"
        if result_name in self.viewer.layers:
            self.viewer.layers[result_name].data = result.astype(np.int32)
        else:
            self.viewer.add_labels(
                result.astype(np.int32),
                name=result_name,
                opacity=0.5,
                **spatial_alignment_kwargs(res_layer),
            )

        self._result_layer = res_layer
        self.status_label.setText(
            f"Postprocessing complete: {op_name} (radius={radius}) applied."
        )

    def _sync_branch_point_features_layer(self, layer: Any) -> None:
        """Keep ``features`` row count aligned with ``data`` (no branch-label column)."""
        import pandas as pd

        n = len(layer.data)
        layer.features = (
            pd.DataFrame(index=range(n)) if n else pd.DataFrame()
        )

    def _wire_branch_points_sync(self, pts: Any) -> None:
        if hasattr(self, "_branch_pts_sync") and self._branch_pts_sync:
            old_pts, old_cb = self._branch_pts_sync
            if old_pts in self.viewer.layers:
                try:
                    old_pts.events.data.disconnect(old_cb)
                except (AttributeError, RuntimeError, TypeError):
                    pass

        def _sync(ev=None):
            self._sync_branch_point_features_layer(pts)
            self._on_ndisplay_changed()

        pts.events.data.connect(_sync)
        self._branch_pts_sync = (pts, _sync)

    def _reset_branch_points_layer(self):
        import pandas as pd

        sel = self.branch_combo.currentText()
        if not sel or sel not in self.viewer.layers:
            self.status_label.setText(
                "Pick a layer in Branch points layer first, then reset."
            )
            return
        lyr = self.viewer.layers[sel]
        if not isinstance(lyr, napari.layers.Points):
            self.status_label.setText("Branch points layer must be a Points layer.")
            return
        lyr.data = np.empty((0, 3))
        lyr.features = pd.DataFrame()
        self._wire_branch_points_sync(lyr)
        self._refresh_layers()
        self.branch_combo.setCurrentText(sel)
        self._on_ndisplay_changed()
        self.status_label.setText(
            f'"{sel}" cleared — add at least two points in order, then Grow.'
        )

    def _allocate_extra_branch_points_name(self) -> str:
        n = getattr(self, "_branch_points_extra_id", 2)
        while f"BranchPoints_{n}" in self.viewer.layers:
            n += 1
        self._branch_points_extra_id = n + 1
        return f"BranchPoints_{n}"

    def _create_new_branch_points_layer(self):
        import pandas as pd

        name = self.image_combo.currentText()
        if not name or name not in self.viewer.layers:
            self.status_label.setText("Select an image layer first.")
            return
        spatial_kw = spatial_alignment_kwargs(self.viewer.layers[name])
        bname = self._allocate_extra_branch_points_name()
        img = self.viewer.layers[name]
        base = _suggested_branch_point_base_size(img)
        self._branch_point_size_bases[bname] = base
        empty_feat = pd.DataFrame()
        pts = self.viewer.add_points(
            np.empty((0, 3)),
            ndim=3,
            name=bname,
            size=base,
            features=empty_feat,
            face_color="cyan",
            border_color="white",
            **spatial_kw,
        )
        pts.mode = "add"
        self._wire_branch_points_sync(pts)
        self._refresh_layers()
        self.branch_combo.setCurrentText(bname)
        self._on_ndisplay_changed()
        self.status_label.setText(
            f'Created "{bname}" — add at least two points in order, then Grow.'
        )

    def _update_branch_method_dependent_widgets(self) -> None:
        m = self.branch_method_combo.currentText()
        is_plain = m.startswith("Plain")
        is_ac = m.startswith("3D Active Contour")
        self.branch_plain_section.setVisible(is_plain)
        self.branch_ac_section.setVisible(is_ac)

    def _ensure_draft_branch_layer(
        self, image_layer: Any, shape: tuple, pyramid_level: int
    ) -> Any:
        """Volatile layer for the current branch computation output."""
        skw = spatial_alignment_for_pyramid_level(image_layer, int(pyramid_level))
        col = self._draft_branch_label_color()
        if DRAFT_BRANCH_LAYER_NAME in self.viewer.layers:
            lyr = self.viewer.layers[DRAFT_BRANCH_LAYER_NAME]
            if tuple(lyr.data.shape) != tuple(shape):
                lyr.data = np.zeros(shape, dtype=np.int32)
            _apply_spatial_kwargs_to_layer(lyr, skw)
            # Keep it visually distinct from the merged layer.
            try:
                lyr.opacity = 0.7
                lyr.color = {1: col}
            except Exception:
                pass
            return lyr
        lyr = self.viewer.add_labels(
            np.zeros(shape, dtype=np.int32),
            name=DRAFT_BRANCH_LAYER_NAME,
            opacity=0.7,
            **skw,
        )
        try:
            lyr.color = {1: col}
        except Exception:
            pass
        return lyr

    def _reset_branch_segmentation(self) -> None:
        """Reset Branch: clears Draft_Branch; keeps branch points and unhides the points layer."""
        if DRAFT_BRANCH_LAYER_NAME in self.viewer.layers:
            lyr = self.viewer.layers[DRAFT_BRANCH_LAYER_NAME]
            if isinstance(lyr, napari.layers.Labels):
                lyr.data = np.zeros_like(np.asarray(lyr.data), dtype=np.int32)
        bname = self.branch_combo.currentText()
        if bname and bname in self.viewer.layers:
            pl = self.viewer.layers[bname]
            if isinstance(pl, napari.layers.Points):
                md = dict(getattr(pl, "metadata", {}) or {})
                if md.get("_rg_hide_empty_2d"):
                    pl.metadata = {
                        k: v for k, v in md.items() if k != "_rg_hide_empty_2d"
                    }
                pl.visible = True
        if self.capture_growth_check.isChecked() and (
            self.capture_combine_grows_check.isChecked()
        ):
            self._gif_capture_pending_segment.clear()
        self.status_label.setText(
            f'Reset: cleared "{DRAFT_BRANCH_LAYER_NAME}"; branch points unchanged and layer shown.'
        )

    def _merge_branch_segmentation(self) -> None:
        tgt = self._branch_trunk_labels_layer()
        if tgt is None:
            tgt = self._ensure_trunk_when_missing()
        if tgt is None:
            self.status_label.setText(
                "Select a segmentation mask (labels, same shape as Image)."
            )
            return
        if DRAFT_BRANCH_LAYER_NAME not in self.viewer.layers:
            self.status_label.setText(
                f'No "{DRAFT_BRANCH_LAYER_NAME}" — run Compute Branch first.'
            )
            return
        br = self.viewer.layers[DRAFT_BRANCH_LAYER_NAME]
        if tuple(br.data.shape) != tuple(tgt.data.shape):
            self.status_label.setText(
                "Branch layer shape does not match the selected segmentation mask."
            )
            return
        bsel = np.asarray(br.data) > 0
        if not np.any(bsel):
            self.status_label.setText(
                f"{DRAFT_BRANCH_LAYER_NAME} is empty — nothing to merge."
            )
            return
        res = np.asarray(tgt.data, dtype=np.int32).copy()
        res[bsel] = np.maximum(res[bsel], np.asarray(br.data, dtype=np.int32)[bsel])
        tgt.data = res
        try:
            # Merged segmentation is always shown as label 1 = red.
            tgt.color = {1: "red"}
        except Exception:
            pass
        br.data = np.zeros_like(res, dtype=np.int32)
        # Start the next draft run from the first distinct preview color again.
        self._draft_branch_color_index = 0
        self._result_layer = tgt
        msg = f'Merged into "{tgt.name}"; {DRAFT_BRANCH_LAYER_NAME} cleared.'
        if self.capture_growth_check.isChecked() and (
            self.capture_combine_grows_check.isChecked()
        ):
            pend = list(self._gif_capture_pending_segment)
            if pend:
                self._gif_capture_combined_frames.extend(pend)
                self._gif_capture_pending_segment.clear()
                self._sync_save_combined_gif_button()
                msg += (
                    f" Combined GIF: +{len(pend)} frames "
                    f"(total {len(self._gif_capture_combined_frames)})."
                )
        self.status_label.setText(msg)
        self._update_postprocess_button()

    def _get_image_layer(self) -> Optional[Any]:
        """Return the selected Image layer or None."""
        name = self.image_combo.currentText()
        if not name or name not in self.viewer.layers:
            self.status_label.setText("Select an Image layer.")
            return None
        lyr = self.viewer.layers[name]
        if not isinstance(lyr, napari.layers.Image):
            self.status_label.setText("Selected layer is not an Image.")
            return None
        return lyr

    def _allocate_new_segmentation_mask_name(self) -> str:
        names = {lyr.name for lyr in self.viewer.layers}
        for i in range(1, 1000):
            cand = "Mask" if i == 1 else f"Mask_{i}"
            if cand not in names:
                return cand
        return "Mask_extra"

    def _create_new_segmentation_mask_layer(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None:
            return
        try:
            shp = image_level_shape(
                image_layer, self._selected_pyramid_level()
            )
        except (TypeError, ValueError, IndexError):
            shp = ()
        if len(shp) != 3:
            self.status_label.setText("Image must be 3-D.")
            return
        nm = self._allocate_new_segmentation_mask_name()
        lvl = self._selected_pyramid_level()
        self.viewer.add_labels(
            np.zeros(shp, dtype=np.int32),
            name=nm,
            opacity=0.5,
            **spatial_alignment_for_pyramid_level(image_layer, lvl),
        )
        self._refresh_layers()
        self.branch_trunk_combo.setCurrentText(nm)
        self.status_label.setText(f'New empty mask layer "{nm}".')

    def _allocate_new_blocker_name(self) -> str:
        names = {lyr.name for lyr in self.viewer.layers}
        if BLOCKER_MASK_LAYER_NAME not in names:
            return BLOCKER_MASK_LAYER_NAME
        for i in range(2, 1000):
            cand = f"{BLOCKER_MASK_LAYER_NAME}_{i}"
            if cand not in names:
                return cand
        return f"{BLOCKER_MASK_LAYER_NAME}_extra"

    def _create_new_blocker_labels_layer(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None:
            return
        try:
            shp = image_level_shape(
                image_layer, self._selected_pyramid_level()
            )
        except (TypeError, ValueError, IndexError):
            shp = ()
        if len(shp) != 3:
            self.status_label.setText("Image must be 3-D.")
            return
        nm = self._allocate_new_blocker_name()
        lvl = self._selected_pyramid_level()
        self.viewer.add_labels(
            np.zeros(shp, dtype=np.int32),
            name=nm,
            opacity=0.45,
            **spatial_alignment_for_pyramid_level(image_layer, lvl),
        )
        self._refresh_layers()
        self.blocker_combo.setCurrentText(nm)
        self.status_label.setText(
            f'New empty blocker "{nm}" — paint foreground where growth must stop.'
        )

    def _ensure_segmentation_result_for_image(self, image_layer: Any) -> None:
        """Create an empty merged segmentation layer on the current grid if missing."""
        try:
            shp = image_level_shape(
                image_layer, self._selected_pyramid_level()
            )
        except (TypeError, ValueError, IndexError):
            return
        if len(shp) != 3:
            return
        if MERGED_SEG_LAYER_NAME in self.viewer.layers:
            ex = self.viewer.layers[MERGED_SEG_LAYER_NAME]
            if layer_data_shape(ex) == shp:
                lvl = self._selected_pyramid_level()
                _apply_spatial_kwargs_to_layer(
                    ex,
                    spatial_alignment_for_pyramid_level(image_layer, lvl),
                )
                return
            # Same name, wrong shape: do not add a second layer automatically.
            return
        lvl = self._selected_pyramid_level()
        self.viewer.add_labels(
            np.zeros(shp, dtype=np.int32),
            name=MERGED_SEG_LAYER_NAME,
            opacity=0.5,
            **spatial_alignment_for_pyramid_level(image_layer, lvl),
        )

    def _current_segmentation_target_layer(self) -> Optional[Any]:
        """Live result layer used for post-processing and branch attachment."""
        if self._result_layer is not None and self._result_layer in self.viewer.layers:
            return self._result_layer
        if MERGED_SEG_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[MERGED_SEG_LAYER_NAME]
        return None

    def _branch_trunk_labels_layer(self) -> Optional[Any]:
        """Labels layer selected as segmentation mask (merge target / grow context)."""
        name = self.branch_trunk_combo.currentText()
        if not name or name not in self.viewer.layers:
            return None
        lyr = self.viewer.layers[name]
        if not isinstance(lyr, napari.layers.Labels):
            return None
        return lyr

    def _refresh_branch_trunk_combo(self) -> None:
        """Repopulate trunk-mask combo with Labels layers on the current image grid."""
        cb = self.branch_trunk_combo
        cur = cb.currentText()
        cb.blockSignals(True)
        cb.clear()
        iname = self.image_combo.currentText()
        if iname in self.viewer.layers:
            img = self.viewer.layers[iname]
            if isinstance(img, napari.layers.Image):
                lvl = self._selected_pyramid_level()
                try:
                    shp = image_level_shape(img, lvl)
                except (TypeError, ValueError, IndexError):
                    shp = ()
                if len(shp) == 3:
                    for lyr in self.viewer.layers:
                        if isinstance(lyr, napari.layers.Labels):
                            if layer_data_shape(lyr) == shp:
                                cb.addItem(lyr.name)
        idx = cb.findText(cur)
        if idx >= 0:
            cb.setCurrentIndex(idx)
        else:
            for pref in (MERGED_SEG_LAYER_NAME, "Segmentation Result"):
                j = cb.findText(pref)
                if j >= 0:
                    cb.setCurrentIndex(j)
                    break
        cb.blockSignals(False)
        self._refresh_blocker_combo()

    def _refresh_blocker_combo(self) -> None:
        """Labels on the current image pyramid grid; first entry is no blocker."""
        cb = self.blocker_combo
        cur = cb.currentText()
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("(none)")
        iname = self.image_combo.currentText()
        if iname in self.viewer.layers:
            img = self.viewer.layers[iname]
            if isinstance(img, napari.layers.Image):
                lvl = self._selected_pyramid_level()
                try:
                    shp = image_level_shape(img, lvl)
                except (TypeError, ValueError, IndexError):
                    shp = ()
                if len(shp) == 3:
                    for lyr in self.viewer.layers:
                        if isinstance(lyr, napari.layers.Labels):
                            if layer_data_shape(lyr) == shp:
                                cb.addItem(lyr.name)
        idx = cb.findText(cur)
        cb.setCurrentIndex(idx if idx >= 0 else 0)
        cb.blockSignals(False)

    def _ensure_trunk_when_missing(self) -> Optional[Any]:
        """Pick a segmentation mask, or create Segmentation Result if none exist on grid."""
        self._refresh_branch_trunk_combo()
        cb = self.branch_trunk_combo
        iname = self.image_combo.currentText()
        if not iname or iname not in self.viewer.layers:
            return self._branch_trunk_labels_layer()
        img = self.viewer.layers[iname]
        if not isinstance(img, napari.layers.Image):
            return self._branch_trunk_labels_layer()
        try:
            shp = image_level_shape(img, self._selected_pyramid_level())
        except (TypeError, ValueError, IndexError):
            return self._branch_trunk_labels_layer()
        if len(shp) != 3:
            return self._branch_trunk_labels_layer()
        if cb.count() == 0:
            self._ensure_segmentation_result_for_image(img)
            self._refresh_branch_trunk_combo()
        if cb.count() == 0:
            return None
        name = cb.currentText()
        if not name or name not in self.viewer.layers:
            cb.blockSignals(True)
            cb.setCurrentIndex(0)
            cb.blockSignals(False)
        return self._branch_trunk_labels_layer()

    def _maybe_archive_nonempty_branch_preview(
        self, image_layer: Any, shape: tuple
    ) -> None:
        """If the live draft layer already has foreground, rename it so Compute can start fresh."""
        name = DRAFT_BRANCH_LAYER_NAME
        if name not in self.viewer.layers:
            return
        lyr = self.viewer.layers[name]
        if layer_data_shape(lyr) != tuple(shape):
            return
        if not np.any(np.asarray(lyr.data) > 0):
            return
        # Preserve this draft's label color so archived unmerged branches remain distinct.
        try:
            col = self._draft_branch_label_color()
            lyr.color = {1: col}
        except Exception:
            pass
        k = 1
        base = DRAFT_BRANCH_LAYER_NAME
        while f"{base} ({k})" in self.viewer.layers:
            k += 1
        lyr.name = f"{base} ({k})"
        # New draft run should use a new label color so multiple unmerged drafts are distinct.
        self._draft_branch_color_index = int(getattr(self, "_draft_branch_color_index", 0)) + 1

    def _finish_branch_grow_preview_if_needed(self) -> None:
        if not getattr(self, "_pending_branch_grow_cleanup", False):
            return
        self._pending_branch_grow_cleanup = False
        if "Skeletal Preview" in self.viewer.layers:
            self.viewer.layers["Skeletal Preview"].visible = False

    def _register_skeletal_preview_vispy_if_needed(self, lyr: Any) -> None:
        """VisPy only maps layers that were shown at least once; hidden-at-create breaks layer list."""
        if self._skeletal_preview_vispy_registered:
            return
        lyr.visible = True
        QApplication.processEvents()
        lyr.visible = False
        self._skeletal_preview_vispy_registered = True

    def _ensure_skeletal_preview_layer(
        self, image_layer: Any, shape: tuple, pyramid_level: int
    ) -> None:
        skw = spatial_alignment_for_pyramid_level(image_layer, int(pyramid_level))
        if "Skeletal Preview" in self.viewer.layers:
            pl = self.viewer.layers["Skeletal Preview"]
            pl.data = np.zeros(shape, dtype=np.uint32)
            _apply_spatial_kwargs_to_layer(pl, skw)
            self._register_skeletal_preview_vispy_if_needed(pl)
            pl.visible = False
            self._preview_layer = pl
            return
        lbl = self.viewer.add_labels(
            np.zeros(shape, dtype=np.uint32),
            name="Skeletal Preview",
            opacity=0.55,
            colormap=DirectLabelColormap(
                color_dict={
                    0: "transparent",
                    1: "#00cc33",
                    None: "transparent",
                }
            ),
            **skw,
        )
        self._preview_layer = lbl
        self._skeletal_preview_vispy_registered = False
        self._register_skeletal_preview_vispy_if_needed(lbl)

    def _clear_skeletal_preview_data(self, shape: tuple) -> None:
        if "Skeletal Preview" not in self.viewer.layers:
            return
        self.viewer.layers["Skeletal Preview"].data = np.zeros(shape, dtype=np.uint32)

    # ----------------------------------------------------------- execution --
    def _optional_branch_plain_upper_threshold(
        self, image_data: np.ndarray
    ):
        """Return upper intensity threshold for branch plain mode, or None."""
        if not self.branch_plain_upper_thr_check.isChecked():
            return None
        from ._algorithm import compute_upper_threshold

        _method_map = {
            "Otsu": "otsu",
            "Triangle": "triangle",
            "Li": "li",
            "90th percentile": "p90",
            "95th percentile": "p95",
        }
        method = _method_map[self.branch_plain_upper_thr_combo.currentText()]
        return compute_upper_threshold(image_data, method)

    def _maybe_append_growth_capture_frame(self) -> None:
        if not getattr(self, "_growth_capture_run", False):
            return
        if getattr(self, "_active_branch_job", None) != "grow":
            return
        self._growth_capture_step_counter += 1
        n_sub = max(1, int(self.capture_subsample_spin.value()))
        if (self._growth_capture_step_counter % n_sub) != 0:
            return
        max_f = int(self.capture_max_frames_spin.value())
        if max_f > 0 and len(self._growth_capture_frames) >= max_f:
            if not self._growth_capture_hit_max:
                self._growth_capture_hit_max = True
            return
        QApplication.processEvents()
        canvas_only = (
            self.capture_region_combo.currentText() == "Viewer canvas"
        )
        scale_pct = float(self.capture_scale_spin.value())
        from ._growth_capture import capture_viewer_frame

        img = capture_viewer_frame(
            self.viewer,
            canvas_only=canvas_only,
            scale_percent=scale_pct,
        )
        if img is not None:
            self._growth_capture_frames.append(img)

    def _finalize_growth_capture(self, had_error: bool) -> bool:
        """Return True if a background GIF encode was started."""
        if getattr(self, "_growth_capture_finalized", False):
            return False
        if not getattr(self, "_growth_capture_run", False):
            return False
        self._growth_capture_finalized = True
        self._growth_capture_run = False
        frames = list(self._growth_capture_frames)
        self._growth_capture_frames.clear()
        hit_max = self._growth_capture_hit_max
        self._growth_capture_hit_max = False

        combine = getattr(self, "_gif_capture_combine_this_run", False)

        if not frames:
            if had_error:
                return False
            msg = (
                "GIF capture: no frames captured — enable Animate growth or set "
                "subsample to 1."
            )
            self.status_label.setText(msg)
            return False

        if combine:
            if had_error:
                return False
            self._gif_capture_pending_segment = list(frames)
            extra_hit = (
                " (max frame cap reached; grow continued)" if hit_max else ""
            )
            self.status_label.setText(
                f'Preview written to "{DRAFT_BRANCH_LAYER_NAME}". '
                "Combined GIF: last grow buffered — Merge to append, or Reset branch "
                f"preview to discard.{extra_hit}"
            )
            return False

        default_name = (
            f"vessel_growth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        )
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save growth GIF",
            default_name,
            "GIF (*.gif);;All files (*)",
        )
        if not path:
            self.status_label.setText("GIF capture cancelled (no file chosen).")
            return False
        if not path.lower().endswith(".gif"):
            path = path + ".gif"

        fps = float(self.capture_output_fps_spin.value())
        self._start_gif_encode_worker(
            frames, path, fps, hit_max, had_error, clear_combined_after=False
        )
        return True

    def _start_gif_encode_worker(
        self,
        frames: List[np.ndarray],
        path: str,
        fps: float,
        hit_max: bool,
        had_error: bool,
        *,
        clear_combined_after: bool = False,
    ) -> None:
        err_prefix = "Partial GIF (grow had errors): " if had_error else ""
        ch = int(self.capture_gif_canvas_h_spin.value())
        cw = int(self.capture_gif_canvas_w_spin.value())

        @thread_worker
        def _encode():
            from ._growth_capture import save_gif

            save_gif(frames, path, fps, canvas_height=ch, canvas_width=cw)
            yield path

        worker = _encode()

        def _on_enc_done(result):
            p = result if isinstance(result, str) else str(result)
            extra = " — max frame cap reached; grow may have continued" if hit_max else ""
            if clear_combined_after:
                self.status_label.setText(
                    f"Saved combined GIF ({len(frames)} frames): {p}"
                )
                self._gif_capture_combined_frames.clear()
                self._sync_save_combined_gif_button()
            else:
                self.status_label.setText(
                    f"{err_prefix}Saved GIF ({len(frames)} frames){extra}: {p}"
                )

        def _on_enc_err(exc):
            self.status_label.setText(
                f"GIF save failed: {self._worker_exception_message(exc)}"
            )

        worker.yielded.connect(_on_enc_done)
        worker.errored.connect(_on_enc_err)
        worker.finished.connect(lambda: setattr(self, "_gif_encode_worker", None))
        worker.start()
        self._gif_encode_worker = worker

    def _grow_branches_into_result(self):
        image_layer = self._get_image_layer()
        if image_layer is None:
            return

        branch_name = self.branch_combo.currentText()
        if not branch_name or branch_name not in self.viewer.layers:
            self.status_label.setText(
                'Pick a branch points layer (e.g. "BranchPoints").'
            )
            return
        branch_layer = self.viewer.layers[branch_name]
        if len(branch_layer.data) < 1:
            self.status_label.setText("Place at least one branch tip.")
            return

        tgt = self._branch_trunk_labels_layer()
        if tgt is None:
            tgt = self._ensure_trunk_when_missing()
        if tgt is None:
            self.status_label.setText(
                "Select a segmentation mask (labels, same shape as Image)."
            )
            return

        level = self._selected_pyramid_level()

        shape_fine = image_finest_shape(image_layer)
        shape_work = image_level_shape(image_layer, level)
        if len(shape_work) != 3:
            self.status_label.setText("Image must be 3-D.")
            return

        if layer_data_shape(tgt) != shape_work:
            self.status_label.setText(
                "Segmentation mask shape must match the selected image pyramid level."
            )
            return

        poly_fine = _points_world_to_image_zyx(
            image_layer, branch_layer, shape_fine
        )
        branch_idx = _polyline_indices_for_level(
            poly_fine, image_layer, level, shape_work
        )
        if branch_idx.shape[0] < 2:
            self.status_label.setText(
                "Need at least two branch points in click order along one branch."
            )
            return

        method = self.branch_method_combo.currentText()

        bname = self.blocker_combo.currentText()
        if bname and bname != "(none)":
            if bname not in self.viewer.layers:
                self.status_label.setText("Selected blocker layer is missing — pick another or (none).")
                return
            bl = self.viewer.layers[bname]
            if not isinstance(bl, napari.layers.Labels):
                self.status_label.setText("Blocker mask must be a labels layer.")
                return
            if layer_data_shape(bl) != shape_work:
                self.status_label.setText(
                    "Blocker labels must match the selected image pyramid level shape."
                )
                return

        if self._preprocess_worker is not None:
            self.status_label.setText("Wait for preprocessing to finish.")
            return

        lazy_img = image_level_is_lazy(image_layer, level)
        if lazy_img:
            QApplication.processEvents()
            self.status_label.setText(
                "Compute Branch: loading the full pyramid level from Zarr/Dask. "
                "If it is still too large, pick a coarser pyramid level."
            )

        # Full working level is materialized in the worker (no polyline crop).
        seg_mask = np.asarray(tgt.data) > 0

        self._image_working_metadata[image_layer.name] = {
            "finest_shape": tuple(shape_fine),
            "working_shape": tuple(shape_work),
            "level": int(level),
            "base_image_name": image_layer.name,
        }

        self._ensure_skeletal_preview_layer(image_layer, shape_work, level)

        # If a previous preview exists and wasn't reset/merged, archive it so
        # a new Compute Branch run starts with a fresh Draft_Branch layer.
        self._maybe_archive_nonempty_branch_preview(image_layer, shape_work)

        draft_layer = self._ensure_draft_branch_layer(image_layer, shape_work, level)
        draft_layer.data = np.zeros(shape_work, dtype=np.int32)
        self._branch_step_target_layer = draft_layer
        self._pending_branch_grow_cleanup = True
        self._last_connect_shape = shape_work
        self._active_branch_job = "grow"
        self._branch_grow_image_layer = image_layer
        self._branch_grow_pyramid_level = int(level)

        self._growth_capture_finalized = False
        self._growth_capture_run = self.capture_growth_check.isChecked()
        self._gif_capture_combine_this_run = (
            self._growth_capture_run
            and self.capture_combine_grows_check.isChecked()
        )
        if self._growth_capture_run:
            self._growth_capture_frames = []
            self._growth_capture_step_counter = 0
            self._growth_capture_hit_max = False
        if self._gif_capture_combine_this_run:
            self._gif_capture_pending_segment.clear()

        self.btn_stop.setEnabled(True)
        self.btn_grow_branches.setEnabled(False)
        self.btn_apply_preprocess.setEnabled(False)
        self.progress_bar.show()

        spacing = voxel_spacing_zyx_for_level(image_layer, level, shape_work)
        radius_ui = float(self.branch_ac_radius_spin.value())
        radius_work = tube_radius_voxels_for_work_level(
            radius_ui, image_layer, level, shape_work
        )
        b_margin_ui = float(self.branch_plain_margin_spin.value())
        b_margin = axis_margin_voxels_for_work_level(
            b_margin_ui, image_layer, level, shape_work
        )
        ac_margin_ui = float(self.branch_ac_margin_spin.value())
        ac_margin = axis_margin_voxels_for_work_level(
            ac_margin_ui, image_layer, level, shape_work
        )
        want_animate = (
            self.animate_check.isChecked() or self.capture_growth_check.isChecked()
        )
        plain_yield = (
            self.branch_plain_step_spin.value()
            if want_animate
            else 10**9
        )
        b_cost = self.branch_plain_cost_budget_spin.value()
        rg_params = dict(
            sigma=self.branch_plain_sigma_spin.value(),
            spacing=spacing,
            cost_budget=b_cost if b_cost > 0 else None,
            flux_weight=self.branch_plain_flux_spin.value(),
            intensity_tolerance=self.branch_plain_intensity_tol_spin.value(),
            margin=b_margin,
            upper_threshold=None,
        )
        _bac_total = self.branch_ac_total_iter_spin.value()
        _bac_yield = (
            self.branch_ac_yield_spin.value()
            if want_animate
            else _bac_total
        )
        # MGAC branch corridor / effective margin use branch_ac_margin_spin.
        ac_params = dict(
            radius=radius_work,
            spacing=spacing,
            sigma=self.branch_ac_sigma_spin.value(),
            low_intensity_equalize_below=self.branch_ac_low_clip_spin.value(),
            balloon=self.branch_ac_balloon_spin.value(),
            smoothing=self.branch_ac_smoothing_spin.value(),
            total_iter=_bac_total,
            yield_every=_bac_yield,
            margin=ac_margin,
        )
        bn = self.blocker_combo.currentText()
        blocker_full = None
        if bn and bn != "(none)" and bn in self.viewer.layers:
            bl = self.viewer.layers[bn]
            if isinstance(bl, napari.layers.Labels):
                blocker_full = np.asarray(bl.data) > 0

        _image_layer = image_layer
        _level = int(level)
        _shape_work = tuple(int(x) for x in shape_work)
        _spacing = spacing
        _branch_idx_arr = branch_idx
        _method = method
        _plain_yield = plain_yield
        _rg_params = rg_params
        _ac_params = ac_params
        _blocker_full = blocker_full
        _seg_mask_full = seg_mask
        _branch_radius = float(radius_work)

        @thread_worker
        def _work():
            from ._algorithm import (
                polyline_corridor_mask,
                polyline_to_line_mask,
                polyline_tube_mask,
                region_grow,
            )
            from ._active_contour import active_contour_grow

            poly = _branch_idx_arr.astype(np.int64)

            # Full pyramid level only (no polyline ROI). Scale via OME-Zarr / napari downsampling.
            msg = check_materialization_budget(
                _shape_work,
                np.dtype(np.float64),
                max_bytes=_MAX_MATERIALIZE_BYTES,
                copies=4.0,
                context="Compute Branch (full volume at selected pyramid level)",
            )
            if msg:
                yield ("error", msg + " (Pick a coarser pyramid level.)")
                return

            img_sub = materialize_image_level(_image_layer, _level, dtype=np.float64)
            if not np.any(np.isfinite(img_sub)):
                yield (
                    "error",
                    "Error: Loaded image has no finite values at this pyramid level "
                    "(NaN/inf everywhere). Check the Zarr level / contrast or materialization.",
                )
                return
            poly_loc = poly.astype(np.int64)
            blocker_sub = (
                np.asarray(_blocker_full, dtype=bool)
                if _blocker_full is not None
                else None
            )

            # Skeletal/tube preview (shown in Skeletal Preview layer).
            tube_preview_sub = polyline_tube_mask(
                img_sub.shape, poly_loc, _branch_radius, _spacing
            )
            yield -1, tube_preview_sub

            # Common: thin centerline mask for margin scaling / stats.
            line_m = polyline_to_line_mask(img_sub.shape, poly_loc)

            if not np.any(tube_preview_sub):
                yield ("error", "Seed tube is empty — increase tube radius or check points.")
                return

            # Plain region growing or MGAC.
            start_f = poly_loc[0].astype(np.float64)
            end_f = poly_loc[-1].astype(np.float64)

            if _method == "Plain Region Growing":
                bm_rg = _branch_effective_margin(line_m, _rg_params["margin"], _spacing)
                upper_thr = self._optional_branch_plain_upper_threshold(img_sub)
                rg_local = {**_rg_params, "margin": bm_rg, "upper_threshold": upper_thr}
                fb = blocker_sub if (blocker_sub is not None and np.any(blocker_sub)) else None
                for step, m in region_grow(
                    img_sub,
                    tube_preview_sub,
                    start_f,
                    end_f,
                    yield_every=_plain_yield,
                    stats_seed_mask=line_m,
                    forbidden_mask=fb,
                    **rg_local,
                ):
                    yield step, np.asarray(m, dtype=bool)
                return

            # MGAC (legacy)
            bm_ac = _branch_effective_margin(line_m, _ac_params["margin"], _spacing)
            ac_local = {**_ac_params, "margin": bm_ac}
            corridor = polyline_corridor_mask(
                img_sub.shape,
                poly_loc,
                _spacing,
                bm_ac,
                _branch_radius,
            )
            dummy_seed = np.zeros(img_sub.shape, dtype=bool)
            init_ls = tube_preview_sub & corridor
            gen = active_contour_grow(
                img_sub,
                dummy_seed,
                start_f,
                end_f,
                blocker_mask=blocker_sub,
                init_level_set=init_ls,
                corridor_mask=corridor,
                **ac_local,
            )
            for it, m in gen:
                yield it, np.asarray(m, dtype=bool)

        worker = _work()
        worker.yielded.connect(self._on_step)
        worker.errored.connect(self._on_worker_error)
        worker.finished.connect(self._on_finished)
        worker.start()
        self._worker = worker
        self.status_label.setText("Growing…")

    def _on_worker_error(self, exc):
        self._finish_branch_grow_preview_if_needed()
        self.btn_stop.setEnabled(False)
        self.btn_grow_branches.setEnabled(True)
        self.btn_apply_preprocess.setEnabled(True)
        self.progress_bar.hide()
        self._worker = None
        self.status_label.setText(f"Error: {self._worker_exception_message(exc)}")
        self._branch_step_target_layer = None

    def _on_step(self, result):
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "error":
            self.status_label.setText(str(result[1]))
            return
        iteration, mask = result
        if iteration >= 0:
            dl = (
                self._branch_step_target_layer
                if getattr(self, "_active_branch_job", None) == "grow"
                else self._result_layer
            )
            if dl is None:
                return
            if isinstance(mask, tuple) and len(mask) >= 1:
                draft_b = mask[0]
            else:
                draft_b = mask
            dl.data = np.asarray(draft_b, dtype=bool).astype(np.int32)
            n_voxels = int(np.asarray(draft_b, dtype=bool).sum())
            if getattr(self, "_active_branch_job", None) == "grow":
                self.status_label.setText(
                    f"Step {iteration} — {n_voxels:,} voxels (preview)"
                )
                self._maybe_append_growth_capture_frame()
            else:
                self.status_label.setText(
                    f"Step {iteration} — {n_voxels:,} voxels segmented"
                )
            return
        if iteration < 0:
            if "Skeletal Preview" in self.viewer.layers:
                sp = self.viewer.layers["Skeletal Preview"]
                sp.data = mask.astype(np.uint32)
            self.status_label.setText(
                f"Branch tube preview — {int(mask.sum()):,} voxels"
            )
            if getattr(self, "_active_branch_job", None) == "grow":
                self._maybe_append_growth_capture_frame()
            return

    def _on_finished(self):
        self.btn_stop.setEnabled(False)
        self.btn_grow_branches.setEnabled(True)
        self.btn_apply_preprocess.setEnabled(True)
        self.progress_bar.hide()
        self._finish_branch_grow_preview_if_needed()
        st = self.status_label.text()
        job = getattr(self, "_active_branch_job", None)
        self._active_branch_job = None
        self._branch_step_target_layer = None
        err = st.startswith("Error:") or st.startswith("Branch step")
        combine_run = getattr(self, "_gif_capture_combine_this_run", False)
        enc_started = False
        if job == "grow":
            enc_started = self._finalize_growth_capture(had_error=err)
        if job == "grow" and not err:
            if enc_started:
                self.status_label.setText(
                    f'Preview written to "{DRAFT_BRANCH_LAYER_NAME}". Encoding GIF in background…'
                )
            elif not combine_run:
                self.status_label.setText(
                    f'Preview written to "{DRAFT_BRANCH_LAYER_NAME}". Use Merge Branch when satisfied.'
                )
        elif st.startswith("Step"):
            self.status_label.setText(st + "  ✓ Done")
        elif not err:
            self.status_label.setText("Done")
        if job == "grow":
            self._gif_capture_combine_this_run = False
        self._update_postprocess_button()
        self._worker = None

    def _stop(self):
        if self._worker is not None:
            self._worker.quit()
        self._finish_branch_grow_preview_if_needed()
        self.btn_stop.setEnabled(False)
        self.btn_grow_branches.setEnabled(True)
        self.btn_apply_preprocess.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText("Stopped by user")
        self._worker = None
        self._branch_step_target_layer = None

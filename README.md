# Region Grow — 3D vessel segmentation for napari

Interactive **semi-automatic** segmentation of **hollow vessels** in 3D fluorescence microscopy. You place ordered points along a branch; the plugin grows a 3D mask along that path and you **merge** it into a master labels layer, branch by branch.

**Default fill method:** **3D Active Contour (MGAC)**. Alternative: **Plain region growing**.

---

## At a glance

| Concept | In the UI / layers |
|--------|---------------------|
| Master mask (merge target) | **Segmentation mask** → layer **`Segmentation`**, **`Segmentation_1`**, … |
| Branch markers | **Branch points layer** → **`BranchPoints`**, **`BranchPoints_1`**, … |
| Per-branch preview | **`Draft_Branch`** (live); archives **`Draft_Branch (1)`**, **`(2)`**, … |
| Optional walls | **Blocker_Mask (optional)** → **`Blocker_Mask`**, … |
| Run growth | **Compute Branch** |
| Commit preview | **Merge Branch** |
| Clear preview only | **Reset Branch** |

**Strategy:** work on a **coarse pyramid level** first (fast, low RAM), then move to **finer levels** for detail. See [Workflow checklist](#workflow-checklist).

---

## What this is for

- Bright vessel walls, darker lumen, darker background  
- Folds and wrinkles where global thresholding leaks  
- Large volumes (OME-Zarr pyramids) where you need **human guidance**, not fully automatic whole-organ segmentation  

---

## Install

**Python ≥ 3.11**, napari with a Qt backend (e.g. `pip install "napari[all]"`).

```bash
pip install -e .
```

Development / tests:

```bash
pip install -e ".[test]"
```

**Console scripts**

| Command | Purpose |
|---------|---------|
| `regiongrow-convert-to-ome-zarr` | BioIO-readable image → multiscale `.ome.zarr` |
| `regiongrow-preprocess-zarr` | Contrast stretch / downsample on OME-Zarr |
| `regiongrow-preprocess-ome-tiff` | Contrast stretch / downsample on OME-TIFF |

Open the dock: **Plugins → Region Grow Vessel Segmentation**.

---

## Data formats

### Zarr and OME-Zarr (recommended for large volumes)

**Zarr** stores arrays as **chunked**, **compressed** files on disk. Viewers read **regions**, not necessarily the full volume — important for whole-organ datasets.

**OME-Zarr** is a microscopy convention (NGFF): a **multiscale pyramid** of the same field of view (`0/` = finest, `1/`, `2/`, … = coarser), plus **voxel spacing** in metadata. Project folders usually end in **`.ome.zarr`**.

This plugin:

- Opens the **image pyramid** at the store root (reader: **Read OME-Zarr image**).  
- Writes segmentations under **`labels/<name>/`** (separate label pyramids; raw image unchanged).  
- **Autosave after Merge** (when the image path is known): overwrites **`labels/segmentation_autosave`**.  
- Manual **Saving**: **`segmentation`**, then **`segmentation_v2`**, … for **New version**.

Pick the **`.ome.zarr` root** in dialogs — not a subfolder inside `labels/` unless you intentionally load one group.

**Preprocess on disk** (contrast stretch, optional downsample) with the CLI tools above; the dock widget does not preprocess in memory.

**Other readers shipped with the plugin**

- **OME-TIFF** (`*.ome.tif` / `*.ome.tiff`) — first series; multi-channel/time uses index 0 with a warning.  
- **Load saved segmentation from OME-Zarr…** (under **Layers**) — imports an existing labels group; resamples to your current **Pyramid level** without loading full finest into RAM when possible.

**napari-ome-zarr** can also open the same stores; choose the reader that fits your workflow.

### Example CLI

```bash
regiongrow-convert-to-ome-zarr volume.ome.tif -o volume.ome.zarr --levels 6
regiongrow-convert-to-ome-zarr stack.tif -o stack.ome.zarr --voxel-size 2.0,0.65,0.65
regiongrow-preprocess-zarr volume.ome.zarr volume_stretched.ome.zarr --stretch --no-downsample
```

---

## Workflow checklist

**Coarse first, detail later:** sketch the vessel tree on a **coarse Pyramid level**, then refine on finer levels or re-grow segments.

- [ ] **1. Setup** — Load a 3D volume in napari. Open **Region Grow Vessel Segmentation**.
- [ ] **2. Coarse level** — **Layers → Pyramid level** → choose a **coarse** level (not finest).
- [ ] **3. Image & mask** — Select **Image**. Choose or create **Segmentation mask** (empty is OK for the first branch).
- [ ] **4. (Optional) Blockers** — **New Blocker** if growth leaks at organ boundaries.
- [ ] **5. One branch** — Add branch points (click order) → **Compute Branch** → edit **`Draft_Branch`** if needed → **Merge Branch**.
- [ ] **6. More branches?** — **Reset branch points** (or new points layer) → **repeat step 5**.
- [ ] **7. Finer detail?** — Finer **Pyramid level** (mask resamples when you change level) → **repeat from step 2**.
- [ ] **8. Export** — **Saving → Save segmentation** (plus optional post-processing below).

---

## Step-by-step (detailed)

1. **Load data** — Single-channel 3D stack (Z×Y×X). For OME-Zarr, open the `.ome.zarr` directory (plugin or napari-ome-zarr reader).

2. **Open the plugin** — **Plugins → Region Grow Vessel Segmentation**.

3. **Layers section**  
   - **Image** — volume to segment.  
   - **Pyramid level** — resolution used for grow, masks, and points (level **0** = finest).  
   - **Enable multiscale rendering in 2D** — **off by default**; when off, the canvas stays on the selected level while you zoom (good for layout on coarse levels). When on, napari swaps pyramid levels while zooming.  
   - **Adapt Z step to pyramid level** — **on by default** on coarse levels (sensible Z slider steps).  
   - **Crop Compute Branch to polyline ROI** — **on by default**; loads only a box around the branch (faster, less RAM).  
   - **Cache pyramid level in session** — **on by default**; caches the working level in RAM after first read (float32).  
   - **Load saved segmentation from OME-Zarr…** — import a labels group from disk.

4. **Segmentation section**  
   - **Segmentation mask** — labels layer that receives **Merge Branch** and supplies context during grow. Selecting an image creates **`Segmentation`** when no suitable mask exists on the current grid. **New Mask** adds **`Segmentation_1`**, etc.  
   - **Branch points layer** — all points in **data order** form one polyline (≥2 points). **New BranchPoints Layer** / **Reset branch points**.  
   - **Branch preview (merge from)** — usually live **`Draft_Branch`**; can merge an archived **`Draft_Branch (N)`**.  
   - **Blocker_Mask (optional)** — painted foreground blocks Plain and MGAC.  
   - **Seed tube radius** — see [Default parameters](#default-parameters).  
   - **Fill with** — **3D Active Contour** (default) or **Plain Region Growing**.  
   - **Compute Branch** — writes to **`Draft_Branch`** (archives non-empty live preview to **`Draft_Branch (1)`**, … if you start again without merge/reset).  
   - **Reset Branch** — clears **`Draft_Branch`** data; keeps branch points.  
   - **Merge Branch** — unions selected preview into **Segmentation mask**. Optional **CleanUp** (**on by default**): removes archived drafts and extra branch-point layers; keeps one empty **`Draft_Branch`** and **`BranchPoints`**.  
   - **Stop** — aborts a running grow.

5. **Branch point placement**  
   - **Tortuosity:** more bends → **more points** along the centerline.  
   - **Long branches / changing diameter:** split into several grows (merge between segments); one grow uses one tube radius and one corridor margin.

6. **After merge** — Autosave to **`segmentation_autosave`** ~2.5 s later if the `.ome.zarr` path is known. Reset points and repeat for the next branch.

7. **Finer resolution** — Change **Pyramid level** and continue growing, or use **Post-processing** (below).

---

## Dock sections (quick lookup)

| Section | Main actions |
|---------|----------------|
| **Layers** | Image, pyramid level, ROI crop, cache, load saved labels |
| **Visualization** | Animate growth, GIF capture, layer color, Plain/MGAC step display rates |
| **Segmentation** | Mask, points, blockers, parameters, Compute / Reset / Merge |
| **Post-processing** | Upsample, morphology (after merge) |
| **Saving** | Export labels to OME-Zarr |

Hover **Segmentation**, **Post-processing**, and **Saving** section headers for short in-app notes.

---

## Post-processing

Requires a mask on the **working pyramid grid** — **Merge Branch** at least once so the plugin tracks a segmentation layer.

| Control | Effect |
|---------|--------|
| **Upsample Result to Original Size** | Nearest-neighbour zoom to finest grid when metadata says working shape ≠ finest (legacy **Grow**-on-coarse workflow). |
| **Upsample segmentation to finer level…** | New editable labels layer on the **next finer** pyramid level. |
| **Operation / Ball radius / Apply Morphological Operation** | 3D dilation or erosion; creates a new result layer (default operation **None**, radius **1**). |

**Saving** is separate: writes the selected **Segmentation mask** to **`labels/…`** in the OME-Zarr store.

**Save resolution (default: Working pyramid level)** — keeps mask on current grid + matching coarser label levels; lowest RAM. **Full finest resolution** — chunked upsample on write (disk can be very large).

**Save target (default: New version)** — `segmentation_vN`, overwrite **`segmentation_autosave`**, overwrite loaded group, or choose another store.

---

## Default parameters

Values below are **widget defaults** at startup (finest-isotropic units for tube radius where noted).

### Layers / grow

| Setting | Default |
|---------|---------|
| Crop Compute Branch to polyline ROI | On |
| Cache pyramid level in session | On |
| Enable multiscale rendering in 2D | Off |
| Adapt Z step to pyramid level | On |
| Animate growth | Off |
| Merge → CleanUp | On |

### Segmentation (shared)

| Setting | Default |
|---------|---------|
| Fill with | 3D Active Contour |
| Seed tube radius | **40** (× min finest spacing → physical radius; scales to working level) |

### Plain Region Growing (branch)

| Setting | Default |
|---------|---------|
| Smoothing σ | 2.0 |
| Flux penalty | 15.0 |
| Intensity tolerance | 3.0 |
| Cost budget | 0 (= auto) |
| Length margin | 0.0 |
| Upper threshold | Off |

### 3D Active Contour / MGAC (branch)

| Setting | Default |
|---------|---------|
| Corridor length margin | 0.15 |
| Edge σ (physical) | 3.0 |
| Low clip | 0.0 |
| Balloon | 0.1 |
| Smoothing γ (steps) | 0 |
| Total iterations | **40** |
| Early stop (unchanged steps) | **2** (0 = run all iterations) |

### Visualization (when enabled)

| Setting | Default |
|---------|---------|
| Plain: refresh every N steps | 500 |
| MGAC: refresh every N iterations | 5 |
| GIF capture | Off |
| Combine branch grows in one GIF | Off |

---

## Algorithms

### Plain region growing

Priority-queue (Dijkstra-style) expansion on an **edge-weighted** cost map, with:

1. **Gradient-based edge cost** — crossing strong edges is expensive.  
2. **Flux penalty** — discourages leaking outward through walls.  
3. **Adaptive intensity gate** — running statistics reject outliers.  
4. **Length constraint** — growth limited along the polyline axis (chord + margin).

### 3D active contour (MGAC)

1. **Seed tube** — polyline rasterized between consecutive points; distance transform builds a tube (same idea as seed for Plain).  
2. **Polyline corridor** — MGAC is clipped to an envelope around the **full** polyline (not just the straight line from first to last point), so curved branches are not eroded away.  
3. **Balloon-first warmup** — short inflate phase so thin tubes survive early morphological shrink.  
4. **MGAC loop** — morphological geodesic active contours on an inverse-gradient edge image; balloon force; optional **smoothing γ** each iteration (default **0**; higher values thin narrow tubes).  
5. **Post-chunk closing** — binary closing fills typical 1-voxel stripe artifacts from discrete updates.

### Blockers

Foreground in **Blocker_Mask** is non-traversable for Plain and zeros MGAC edge speed; mask voxels are cleared from the contour each iteration.

### Coordinates

Branch points use the points layer transform; growth uses the image grid at the selected **Pyramid level** (`world_to_data` / spacing-aware tube radius).

---

## Branch point tips

- Add a point wherever the vessel **deviates** from a straight line between existing points.  
- **Split** long or tapering branches into multiple **Compute Branch → Merge** cycles.  
- Use **Blocker_Mask** on whole-organ data where vessels meet the organ surface.

---

## Parameter tuning (beyond defaults)

### Plain

- Increase **σ** or **flux** if leakage; decrease flux if growth stops early.  
- Increase **intensity tolerance** if true lumen voxels are rejected.  
- Increase **cost budget** only when growth stops prematurely (0 = auto).

### MGAC

- Decrease **Seed tube radius** for very thin vessels.  
- **σ** is physical (µm-scale); lower on coarse pyramid levels if edges blur.  
- **Balloon** ~0.1–0.3 for strong edges; higher for weak edges.  
- **Smoothing γ** > 0 thins masks each iteration — use sparingly.  
- Raise **early stop** if the preview stabilizes too soon; set to **0** to force all iterations.

### Morphology

- **Erosion** radius 1 often removes one-voxel **Z** over-thickness in anisotropic data.  
- Large 3D radii change topology quickly.

---

## GIF capture

Under **Visualization**, enable **Capture animation (GIF)**.

- **One GIF per grow** — leave **Combine branch grows in one GIF** unchecked; save path after each successful **Compute Branch**.  
- **Combined GIF** — enable combine mode; frames append on **Merge Branch** only; **Reset Branch** drops unmerged recording; **Save combined GIF…** when done.

**Skeletal Preview** (polyline tube before propagation) is hidden by default; toggle visibility in napari if needed.

Subsampling, max frames, scale, and canvas size control file size (capture is RAM-capped ~1.5 GB).

---

## Troubleshooting

| Message / issue | What to try |
|-----------------|-------------|
| **seed_mask is empty** | Increase **Seed tube radius**; check points sit on the vessel; check blockers do not cover the seed. |
| **start_point and end_point coincide** | ≥2 distinct points; avoid double-click duplicates. |
| **NaN/Inf in image** | Reload; another pyramid level; preprocess with `regiongrow-preprocess-zarr`. |
| **does not match any image pyramid level** (save) | Match **Pyramid level** to mask grid, or save at **Full finest resolution**. |
| **not an .ome.zarr store** | Select store **root** (`name.ome.zarr`), not an arbitrary subfolder. |
| **Out-of-memory on grow** | Keep **ROI crop** on; coarser **Pyramid level**; preprocess/downsample on disk. |
| **Eraser / paint on mask** | Use napari labels **paint** or **erase** on **Segmentation** / **Draft_Branch**; plugin keeps masks editable (3D paint if `n_edit_dimensions` = 3). |
| **GIF stops mid-grow** | Lower frame scale or increase subsample. |

---

## References

1. Dijkstra EW. A note on two problems in connexion with graphs. *Numerische Mathematik* 1959;1:269–271.  
2. Vasilevskiy A, Siddiqi K. Flux maximizing geometric flows. *IEEE TPAMI* 2002;24(12):1565–1578.  
3. Marquez-Neila P, Baumela L, Alvarez L. A morphological approach to curvature-based evolution of curves and surfaces. *IEEE TPAMI* 2014;36(1):2–17.  
4. Welford BP. Note on a method for calculating corrected sums of squares and products. *Technometrics* 1962;4(3):419–420.

---

## License

BSD-3-Clause

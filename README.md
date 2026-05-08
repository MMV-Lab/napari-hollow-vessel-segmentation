# Region Grow - 3D Vessel Segmentation for napari

A [napari](https://napari.org) plugin for interactive segmentation of hollow
vessels in 3D fluorescence microscopy.

The plugin uses one workflow for every vessel segment (main trunk or side branch):

- **3D active contour (MGAC)** (default) or **plain region growing** along a user-drawn **polyline** (two or more ordered points in a points layer).
- Each **Grow** writes a preview to **Branch Segmentation**; **Merge** unions that preview into the selected **Segmentation mask** labels layer (often **Segmentation Result**). That mask may be **empty** for the first segment—no separate seed labels layer or two-point-only “start/end” layer.
- Optional **Blocker mask** (labels on the image grid): painted foreground blocks **both** MGAC and Plain growth — useful on whole-organ scans to stop leakage where vessels meet the organ boundary. Use **New Blocker** or choose any matching labels layer in **Blocker mask (optional)**. Optional **Upsample** / **morphology** run after you have a mask on the working grid (merge at least once so the plugin tracks a labels layer for post-processing).

## What this is for

- Hollow vessels with bright walls, darker lumen, and darker background
- Surfaces with folds/wrinkles where simple thresholding leaks
- Users who want interactive seed-and-run segmentation in napari

## Install

Editable install **with** Zarr preprocessing extras:

```bash
pip install -e ".[zarr-cli]"
```

**OME-TIFF → OME-Zarr conversion** uses BioIO’s writer and needs **Python ≥ 3.11** plus `zarr-cli-bioio` (often a separate conda env, e.g. `ome-zarr-bioio`):

```bash
pip install -e ".[zarr-cli-bioio]"
```

If you already ran `pip install -e .` only, add extras **in the same conda/venv** you use for each task.

Console scripts: `regiongrow-convert-to-ome-zarr` (BioIO path), `regiongrow-preprocess-zarr`.  
From the repo with `zarr-cli-bioio` installed: `python -m regiongrow.cli.ome_tiff_to_zarr_bioio …`.  
From a napari/Python 3.10 env, use **`./run_ome_tiff_to_zarr_bioio.sh`** (wraps `conda run -n ome-zarr-bioio …`).

## OME-Zarr and large volumes

- Open **`.ome.zarr`** directories in napari with **[napari-ome-zarr](https://github.com/ome/napari-ome-zarr)** (NGFF multiscale, chunked storage on disk).
- This plugin still ships an **OME-TIFF** reader for `*.ome.tif` / `*.ome.tiff`; you may see both readers in napari’s open dialog—choose the one you want.
- **RAM vs chunks:** Zarr chunking helps on disk and in napari’s viewer, but **Grow** and **in-memory Preprocessing** still load a **dense numpy** slab for the selected **Pyramid level**. If the level is still too large, use a coarser level or run **`regiongrow-preprocess-zarr`** for optional mean-downsample / histogram stretch on disk without holding the full volume in the widget.
- Under **Layers**, use **Pyramid level** when the image is multiscale. **Post-processing → Upsample** zooms a mask to the **finest** resolution when the working grid is coarser than finest (e.g. after **Grow** on a pyramid level; metadata may also come from an older preprocessed layer).

### Example CLI

```bash
regiongrow-convert-to-ome-zarr volume.ome.tif -o volume.ome.zarr --levels 6
# or from repo root when napari env is not 3.11+:
./run_ome_tiff_to_zarr_bioio.sh volume.ome.tif -o volume.ome.zarr --levels 6

regiongrow-preprocess-zarr volume.ome.zarr volume_stretched.ome.zarr --stretch --no-downsample
```

## Step-by-step guide

1. Open napari and load a 3D image (single channel, shape Z×Y×X). For **OME-Zarr**, install **napari-ome-zarr** and open the `.ome.zarr` directory.
2. Open the widget: **Plugins → Region Grow Vessel Segmentation**.
3. Select the image in **Layers → Image**. The plugin does **not** add an empty labels layer until your first **Grow** or **Merge** when no labels layer exists on that grid (then it creates **Segmentation Result**).
4. Under **Segmentation**, use **Segmentation mask** to choose the labels layer that receives **Merge** and provides context for **Grow** (e.g. **Segmentation Result**). Use **New Mask** to add another empty labels layer on the grid. An **empty** mask is valid for the **first** segment: **Grow** ORs it with the growing preview.
5. **New BranchPoints Layer** adds a new points layer (**BranchPoints_2**, …). **Reset branch points** clears the layer currently selected in **Branch points layer** (not only the default **BranchPoints** name). Optionally use **Blocker mask (optional)** / **New Blocker** to paint walls on the same pyramid grid as the image (whole-organ leakage).
6. Add **at least two** points in **click order** along one vessel segment. Open **Segmentation parameters** for **Seed tube radius** and **Fill with** (**3D Active Contour** is the default; switch to Plain if needed). Defaults: MGAC **Corridor length margin** 0, tube radius 20, smoothing γ 0, 100 iterations; Plain **Length margin** 0. **Grow** writes to **Branch Segmentation**; if that layer already held a preview and you did not reset or merge, the old mask is renamed to **Branch Segmentation (1)**, **(2)**, … and a fresh preview layer is used. Hover the **Segmentation** / **Post-processing** section titles for workflow notes.
7. When satisfied, **Merge**. Clear or reset points, then repeat for the next segment.
8. **Reset branch preview** clears **Branch Segmentation** only. To clear a labels mask, edit or delete the layer in napari.
9. For large volumes: pick a coarser **Pyramid level** under **Layers**. Optionally open **Preprocessing** for **3D non-local means** denoise and/or **contrast stretch** on that level (no integer downsample here—resolution is the pyramid). Use **Run preprocessing** to add a new image layer (or replace after confirmation). **Post-processing → Upsample** and morphology need a merged mask on the working grid; the plugin tracks the last layer you merged into (or **Segmentation Result** if present). Denoise is the heaviest step and runs in a background worker; full finest-level volumes can take a long time.

**Archiving:** the old **Run** path that renamed **Segmentation Result** to **Result_v*** before each run has been removed. Rename or duplicate layers in napari if you want snapshots.

### GIF capture (growth animation)

Under **Visualization**, enable **Capture animation (GIF)** to record the viewer during each **Grow**. By default, **Skeletal Preview** (the polyline tube before propagation) stays **hidden** in the layer list; show it in napari if you want that overlay.

**One GIF per Grow:** leave **Combine branch grows in one GIF** unchecked. After a successful grow you are prompted for a save path; encoding runs in the background.

**One GIF for several branches:** check **Combine branch grows in one GIF (commit on Merge)**. Each grow is still recorded, but frames are appended to the combined clip only when you click **Merge**. **Reset branch preview** discards the last grow’s recording without merging. Starting a new **Grow** clears any unmerged recording from the previous attempt. When finished, use **Save combined GIF…** (encoding clears the in-memory combined buffer after a successful save).

Playback length follows the number of captured frames and **GIF playback FPS**. **Frame subsample (N)** keeps every *N*-th displayed step (use with **Animate growth** and the Plain / MGAC step controls). **Max frames** caps frames per grow segment. **Capture region** is viewer canvas or full napari window; **Frame scale** shrinks frames before encoding. **GIF canvas width / height** control the pixel size of every frame in the saved GIF (letterboxing): leave both at **Auto** to use the largest width and height seen in that save (enough for a single grow); for **combined GIFs**, set both to the same fixed size (e.g. 960×720) so segments from different window layouts still stack. If **Animate growth** is off but capture is on, the plugin still uses the same step intervals as when animation is on so the recording is usable.

## Technical notes (short)

1. **Coordinates:** Points use `data_to_world` on the points layer and `world_to_data` on the image layer so grids match after **Preprocessing** (dtype / contrast changes) or translation.
2. **Branch seed:** `skimage.draw.line_nd` (or a fallback) rasterizes segment between consecutive knots; `scipy.ndimage.distance_transform_edt` builds the same style of tube as main MGAC. **Branch MGAC** clips evolution with a **polyline corridor** (EDT envelope around the full centerline), not only the straight segment between first and last point, so curved branches are not eroded to an empty mask. The morphological **edge shrink** step would otherwise delete a thin polyline tube in a handful of iterations; branch runs therefore use a **balloon-first warmup** (inflate along the edge image with a low speed threshold, no shrink term), then the full MGAC loop.
3. **MGAC smoothing γ (steps):** skimage morphological curvature (`_curvop`) runs **after balloon+edge in every MGAC iteration**, not once at the end. Values ``> 0`` regularize but **thin** narrow masks each iteration; the UI default is **0** (tune upward per dataset). After each MGAC chunk a **binary closing** (one iteration, physical ball) fills typical **1-voxel stripe / checkerboard** gaps from the discrete edge update.
4. **Blockers:** Choose a labels layer (or **none**) in **Blocker mask (optional)**. Foreground voxels block **Plain** priority-queue growth (non-traversable). For **MGAC**, they zero the edge speed map, are cleared from the level set each iteration, and are stripped from the initial seed before the first displayed frame.

## Method overview

### Plain region growing

The plain mode is a min-heap front propagation method with multiple stopping
criteria:

1. Edge-weighted local cost.
   Local cost is derived from an edge indicator based on image gradients, so crossing strong edges becomes expensive.
2. Priority-queue expansion.
   Voxels are accepted in increasing accumulated cost order (Dijkstra-style, cheapest-first).
3. Flux penalty.
   Outward gradient flux is used as a soft penalty to discourage wall-to-background leakage while tolerating local wall roughness.
4. Adaptive intensity gate.
   Running region statistics reject candidates that fall too far below the current region intensity model.
5. Length constraint.
   Growth is clipped along the vessel axis implied by the polyline (chord-based margin), plus a margin.

### 3D active contour (MGAC)

The active contour mode initializes a tube around the seed centerline and
evolves it with Morphological Geodesic Active Contours on an inverse-gradient
edge image. A balloon force controls outward/inward bias. **Smoothing steps**
apply skimage morphological curvature **once per outer iteration** (after
balloon and edge updates), not as a final post-process—non-zero smoothing tends
to thin narrow tubes over many iterations. The length constraint clips extent
(polyline **corridor** for MGAC; plain mode uses the same polyline for statistics and axis margin).

### Shared post-processing

After you have merged labels on the working grid:

- Upsampling restores full resolution when the working grid is coarser than the finest image (e.g. **Grow** on a pyramid level), or from tracked metadata on a preprocessed layer.
- Morphological Dilation/Erosion (ball radius in voxels) refines mask shape.
- In anisotropic datasets, a common correction is one Erosion with radius 1
   to remove slight extra thickness along Z while preserving XY quality.

## Practical parameter tips

### Plain mode (parameters under **Segmentation → Plain parameters**)

- Use **Blocker mask** if plain growth leaks outside the organ (same optional layer as MGAC).
- Smoothing sigma: start at 2.0; increase for noisy images.
- Flux penalty: increase if leakage occurs; decrease if growth stalls too early.
- Intensity tolerance: increase if true vessel voxels are being rejected.
- Cost budget: keep auto first; increase only when growth stops prematurely.

### Active contour mode (MGAC in **Segmentation parameters**)

- Optional **Blocker mask** (same grid / pyramid level as the image): painted voxels block MGAC (see **Segmentation → Blocker mask**).
- **Seed tube radius**: default **20** voxels (same as MGAC ``radius``); reduce for very thin vessels.
- **Smoothing γ (steps):** default **0**; increase for stronger per-iteration smoothing (also thins narrow tubes).
- **Total iterations:** default **100**.
- Sigma: 1.5 to 3.0 is a good default range for most datasets.
- Balloon: 0.1 to 0.3 for thin vessels or strong edges; 0.5 to 1.0 for weak edges or smoother interiors.

### Morphological post-processing

- Dilation (radius 1 to 2) can fill tiny gaps or connect close fragments.
- Erosion (radius 1 to 2) can remove thin protrusions or boundary noise.
- For anisotropic voxel spacing, start with Erosion radius 1 to clean mild
   Z-direction over-segmentation (often one-voxel too thick in Z).
- Use larger radii cautiously because topology changes quickly in 3D.

## References

1. Dijkstra EW. A note on two problems in connexion with graphs. Numerische Mathematik. 1959;1:269-271.
2. Vasilevskiy A, Siddiqi K. Flux maximizing geometric flows. IEEE Trans Pattern Anal Mach Intell. 2002;24(12):1565-1578.
3. Marquez-Neila P, Baumela L, Alvarez L. A morphological approach to curvature-based evolution of curves and surfaces. IEEE Trans Pattern Anal Mach Intell. 2014;36(1):2-17.
4. Welford BP. Note on a method for calculating corrected sums of squares and products. Technometrics. 1962;4(3):419-420.

## License

BSD-3-Clause

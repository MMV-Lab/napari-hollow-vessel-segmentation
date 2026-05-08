import traceback
import shutil
from pathlib import Path

from bioio_ome_tiff.writers import OmeTiffWriter
from bioio_ome_zarr.writers import OMEZarrWriter, get_default_config_for_viz

from .config import XML_DIR, THUMBNAIL_DIR, TIFF_DIR, ZARR_DIR
from .image_utils import generate_quick_preview, generate_rgb_thumbnail


def process_images(
    input_path_str: str,
    uuid: str,
    adapter,
    overwrite: bool = False,
    scene_index: int | None = None,
) -> tuple[bool, Exception | None]:
    try:
        input_path = Path(input_path_str).resolve()
        if not input_path.exists() or not input_path.is_file():
            raise ValueError(f"{input_path} doesn't exist")

        _out_xml_path = XML_DIR / f"{uuid}.xml"
        _img_tn_path = THUMBNAIL_DIR / f"{uuid}.jpg"
        _img_ome_tiff_path = TIFF_DIR / f"{uuid}.ome.tiff"
        _img_ome_zarr = ZARR_DIR / f"{uuid}.ome.zarr"

        # Fast skips
        need_xml = overwrite or not _out_xml_path.exists()
        need_thumbnail = overwrite or not _img_tn_path.exists()
        need_tiff = overwrite or not _img_ome_tiff_path.exists()
        need_zarr = overwrite or not _img_ome_zarr.exists()

        if not (need_xml or need_thumbnail or need_tiff or need_zarr):
            return True, None

        img = (
            adapter.load(input_path, scene_index=scene_index)
            if scene_index is not None
            else adapter.load(input_path)
        )

        # # In-memory evaluation to avoid dual disk reads
        # if need_tiff or need_zarr:
        #     img_xr = img.xarray_data.squeeze(drop=True)
        # else:
        #     img_xr = img.xarray_dask_data.squeeze(drop=True)

        img_xr = img.xarray_dask_data.squeeze(drop=True)

        img_xr_dims = "".join(img_xr.dims)
        img_physical_pixel_sizes = img.physical_pixel_sizes

        # ------------------ FIX OME-XML ------------------
        if need_xml or need_tiff: 
            if hasattr(adapter, 'build_ome'):
                out_ome = adapter.build_ome(img, img_xr)
            else:
                out_ome = img.ome_metadata

            if need_xml:
                with open(_out_xml_path, "w") as f:
                    f.write(out_ome.to_xml())

        # ------------------ THUMBNAIL ------------------
        if need_thumbnail:
            thumbnail_xr = generate_quick_preview(img_xr)
            pil_thumbnail, _ = generate_rgb_thumbnail(thumbnail_xr)
            pil_thumbnail.save(_img_tn_path)

        # ------------------ TIFF ------------------
        if need_tiff:
            OmeTiffWriter.save(
                data=img_xr.data,
                uri=_img_ome_tiff_path,
                dim_order=img_xr_dims,
                ome_xml=out_ome,
                physical_pixel_sizes=img_physical_pixel_sizes,
            )

        # ------------------ ZARR ------------------
        if need_zarr:
            if overwrite and _img_ome_zarr.exists():
                shutil.rmtree(_img_ome_zarr, ignore_errors=True)

            # 1. Store upper-case dims for BioIO lookup, but use lower-case for OME-NGFF compliance
            # dims_upper = list(img_xr.dims)
            # axes_names = [d.lower() for d in dims_upper]

            # 2. Map physical pixel sizes natively via BioIO's Scale named tuple
            # Use `or 1.0` to catch properties that exist but are `None` (like Channel)
            physical_pixel_size = [
                getattr(img.scale, dim, 1.0) or 1.0
                for dim in list(img_xr.dims)
            ]

            # 3. Map units and types dynamically via BioIO's DimensionProperties
            # axes_units = []
            # axes_types = []
            # for dim in dims_upper:
            #     prop = getattr(img.dimension_properties, dim, None)
            #     axes_units.append(str(prop.unit) if prop and prop.unit else None)
            #     axes_types.append(prop.type if prop else None)

            # 4. Let the bioio_ome_zarr utility build the pyramid and optimal ~16MiB chunk shape
            # downsample_z=True will halve Z/Y/X. Set to False if only Y/X downsampling is desired.
            viz_config = get_default_config_for_viz(img_xr.data, downsample_z=True)

            ZARR_writer = OMEZarrWriter(
                store=_img_ome_zarr,
                zarr_format=3,
                level_shapes=viz_config["level_shapes"],
                chunk_shape=viz_config["chunk_shape"],
                dtype=viz_config["dtype"],
                # axes_names=axes_names,
                # axes_types=axes_types,
                # axes_units=axes_units,
                physical_pixel_size=physical_pixel_size,
            )

            ZARR_writer.write_full_volume(img_xr.data)

        return True, None
    except Exception as e:
        print(f"Error processing {input_path_str}, {uuid}:")
        traceback.print_exc()
        return False, e
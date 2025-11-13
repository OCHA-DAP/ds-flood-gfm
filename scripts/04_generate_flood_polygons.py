import argparse
import logging
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import ocha_stratus as stratus
from dotenv import load_dotenv

from ds_flood_gfm.datasources.gfm import (
    create_flood_composite,
    create_provenance_raster,
    export_polygons,
    query_gfm_stac,
    raster_to_polygons,
)
from ds_flood_gfm.geo_utils import generate_cache_key

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Generate flood polygons from GFM STAC data"
    )
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--n-latest",
        type=int,
        default=4,
        help="Number of days to look back (default: 4)",
    )
    parser.add_argument(
        "--n-search", type=int, default=15, help="Search window in days (default: 15)"
    )
    parser.add_argument(
        "--iso3", required=True, help="Country ISO3 code (JAM, HTI, CUB)"
    )
    parser.add_argument(
        "--flood-mode",
        choices=["latest", "cumulative"],
        default="latest",
        help="Flood composite mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/polygons"),
        help="Output directory",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GFM FLOOD POLYGON GENERATOR")
    logger.info("=" * 60)
    logger.info(f"Country: {args.iso3}")
    logger.info(f"End date: {args.end_date}")
    logger.info(f"Days back: {args.n_latest}")
    logger.info(f"Search window: {args.n_search}")
    logger.info(f"Mode: {args.flood_mode}")
    logger.info("=" * 60)

    gdf_admin = stratus.codab.load_codab_from_fieldmaps(args.iso3, 0)
    bbox = gdf_admin.total_bounds
    items = query_gfm_stac(bbox, args.end_date, args.n_search)

    if len(items) == 0:
        logger.error("No STAC items found for the specified criteria")
        return

    # Create flood composite with stack for provenance
    flood_composite, unique_dates, stack_flood_max = create_flood_composite(
        items, bbox, args.n_latest, mode=args.flood_mode, return_stack=True
    )

    logger.info("Converting flood raster to polygons...")
    try:
        flood_polygons = raster_to_polygons(flood_composite)
        logger.info(f"✅ Polygon conversion complete")
    except Exception as e:
        logger.error(f"❌ Polygon conversion failed: {e}")
        raise

    if len(flood_polygons) == 0:
        logger.warning("No polygons created")
        return

    date_str = args.end_date.replace("-", "")
    filename_base = f"{args.iso3.lower()}_flood_{args.flood_mode}_{date_str}"
    output_path = args.output_dir / filename_base

    output_path = generate_cache_key(args.iso3, unique_dates, None, args.flood_mode)
    export_polygons(flood_polygons, output_path, local=False, blob=True)

    # Generate and upload provenance raster
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING PROVENANCE RASTER")
    logger.info("=" * 60)

    provenance_idx, date_mapping = create_provenance_raster(
        stack_flood_max, unique_dates
    )

    # Compute the provenance raster
    logger.info("Computing provenance raster...")
    prov_computed = provenance_idx.compute()
    logger.info(f"✅ Provenance raster computed: {prov_computed.shape}")

    # Create filename based on cache key
    prov_filename = f"{output_path}_provenance.tif"
    blob_path = f"ds-flood-gfm/processed/provenance_raster/{prov_filename}"

    # Upload to blob as COG (stratus expects xarray DataArray)
    logger.info(f"Uploading provenance raster to: {blob_path}")
    stratus.upload_cog_to_blob(
        prov_computed, blob_path, container_name="projects", stage="dev"
    )
    logger.info(f"✅ Provenance raster uploaded")

    # Log date mapping for reference
    logger.info("\nProvenance date mapping:")
    for idx, date in date_mapping.items():
        logger.info(f"  {idx}: {date}")

    logger.info("\n" + "=" * 60)
    logger.info("FLOOD POLYGON GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Polygons created: {len(flood_polygons)}")
    logger.info(f"   Provenance raster saved to blob: {blob_path}")


if __name__ == "__main__":
    main()

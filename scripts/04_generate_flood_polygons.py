import argparse
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import ocha_stratus as stratus
from dotenv import load_dotenv

from ds_flood_gfm.datasources.gfm import (
    add_provenance_to_polygons,
    create_flood_composite,
    create_provenance_raster,
    export_polygons,
    process_country_tiled,
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
    parser.add_argument("--target-date", required=True, help="Reference date (YYYY-MM-DD)")
    parser.add_argument(
        "--n-images",
        type=int,
        default=4,
        help="Number of dates to use for composite (default: 4)",
    )
    parser.add_argument(
        "--n-search",
        type=int,
        default=-15,
        help="Search window in days. Positive = forward scan, Negative = backward scan (default: -15)"
    )

    # Geometry source: either ISO3 or custom geoparquet file
    geom_group = parser.add_mutually_exclusive_group(required=True)
    geom_group.add_argument(
        "--iso3",
        help="Country ISO3 code (JAM, HTI, CUB)"
    )
    geom_group.add_argument(
        "--aoi-geom-blob",
        help="Blob path to geoparquet file (.parquet)"
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
    parser.add_argument(
        "--use-tiling",
        action="store_true",
        help="Use spatial tiling for large countries (automatic for PHL)",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=2.0,
        help="Tile size in degrees (default: 2.0)",
    )

    args = parser.parse_args()

    # Auto-enable tiling for known large countries (only for ISO3 mode)
    if args.iso3:
        large_countries = ["PHL", "IDN", "BRA", "USA", "CAN", "RUS", "CHN", "AUS", "URY"]
        if args.iso3 in large_countries and not args.use_tiling:
            logger.info(f"⚠️  {args.iso3} is a large country - automatically enabling tiled processing")
            args.use_tiling = True

    scan_direction = "forward" if args.n_search > 0 else "backward"
    logger.info("=" * 60)
    logger.info("GFM FLOOD POLYGON GENERATOR")
    logger.info("=" * 60)

    # Load geometry from ISO3 or custom geoparquet file
    if args.iso3:
        logger.info(f"Loading admin boundaries for {args.iso3} via CODAB...")
        gdf_admin = stratus.codab.load_codab_from_fieldmaps(args.iso3, 0)
        aoi_name = args.iso3
    elif args.aoi_geom_blob:
        logger.info(f"Loading custom geometry from blob: {args.aoi_geom_blob}")
        gdf_admin = stratus.load_geoparquet_from_blob(args.aoi_geom_blob)
        aoi_name = Path(args.aoi_geom_blob).stem
        logger.info(f"  Loaded {len(gdf_admin)} features")

    bbox = gdf_admin.total_bounds
    logger.info(f"  Bounding box: {bbox}")

    logger.info(f"AOI: {aoi_name}")
    logger.info(f"Target date: {args.target_date}")
    logger.info(f"Number of images: {args.n_images}")
    logger.info(f"Search window: {args.n_search} days ({scan_direction})")
    logger.info(f"Mode: {args.flood_mode}")
    logger.info("=" * 60)

    # Use tiled processing for large countries
    if args.use_tiling:
        logger.info(f"Using tiled processing with {args.tile_size}° tiles")

        # Process polygons with tiling (memory-intensive operation)
        flood_polygons, unique_dates = process_country_tiled(
            bbox=bbox,
            target_date=args.target_date,
            n_images=args.n_images,
            n_search=args.n_search,
            mode=args.flood_mode,
            tile_size=args.tile_size,
            return_stack=False
        )

        # Generate provenance raster for full country (doesn't need tiling)
        # Why no tiling for provenance:
        # 1. Provenance generation stays LAZY - doesn't compute the flood composite
        # 2. Only computes a 2D raster (not 3D temporal stack) at the end
        # 3. Memory usage similar to one tile (~0.1-0.2GB vs 5GB for full composite)
        # 4. Rechunking strategy (time:-1, y:4096, x:4096) makes it efficient
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING PROVENANCE RASTER (FULL COUNTRY)")
        logger.info("=" * 60)
        logger.info("Note: Provenance doesn't use tiling - it's a lightweight operation")
        logger.info("      that stays lazy and only computes a 2D raster at the end.")

        # Query STAC for full country
        items = query_gfm_stac(bbox, args.target_date, args.n_search)

        if len(items) == 0:
            logger.warning("No STAC items found for provenance generation")
            stack_flood_max = None
        else:
            # Create stack for provenance (stays lazy, flood_composite not used)
            _, _, stack_flood_max = create_flood_composite(
                items, bbox, args.n_images,
                mode=args.flood_mode,
                n_search=args.n_search,
                return_stack=True
            )
            logger.info("✅ Stack created for provenance (still lazy)")

    else:
        # Standard processing for small countries
        items = query_gfm_stac(bbox, args.target_date, args.n_search)

        if len(items) == 0:
            logger.error("No STAC items found for the specified criteria")
            return

        # Create flood composite with stack for provenance
        flood_composite, unique_dates, stack_flood_max = create_flood_composite(
            items, bbox, args.n_images, mode=args.flood_mode, n_search=args.n_search, return_stack=True
        )

        logger.info("Converting flood raster to polygons...")
        try:
            flood_polygons = raster_to_polygons(flood_composite)
            logger.info(f"✅ Polygon conversion complete: {len(flood_polygons)} polygons")
        except Exception as e:
            logger.error(f"❌ Polygon conversion failed: {e}")
            raise

    if len(flood_polygons) == 0:
        logger.warning("No polygons created")
        return

    # Add provenance to flood polygons (only for non-tiled processing)
    if stack_flood_max is not None:
        logger.info("\n" + "=" * 60)
        logger.info("ADDING PROVENANCE TO FLOOD POLYGONS")
        logger.info("=" * 60)

        # Create provenance raster (lazy)
        provenance_idx, date_mapping = create_provenance_raster(
            stack_flood_max, unique_dates
        )

        # Compute provenance raster
        logger.info("Computing provenance raster...")
        prov_computed = provenance_idx.compute().astype(np.int16)
        logger.info(f"✅ Provenance raster computed: {prov_computed.shape}")

        # Add provenance dates to flood polygons
        flood_polygons = add_provenance_to_polygons(
            flood_polygons, prov_computed, date_mapping
        )
        logger.info(f"✅ Added provenance dates to flood polygons")

    output_path = generate_cache_key(aoi_name, unique_dates, None, args.flood_mode)
    export_polygons(flood_polygons, output_path, local=False, blob=True)

    # Upload provenance raster (already computed above)
    if stack_flood_max is not None:
        logger.info("\n" + "=" * 60)
        logger.info("UPLOADING PROVENANCE RASTER")
        logger.info("=" * 60)

        # Add metadata as DataArray attributes (stored as GeoTIFF tags)
        prov_computed.attrs['date_mapping'] = json.dumps(date_mapping)
        prov_computed.attrs['created_date'] = datetime.now().isoformat()
        logger.info("✅ Added metadata to provenance raster")

        # Create filename based on cache key
        prov_filename = f"{output_path}_provenance.tif"
        blob_path = f"ds-flood-gfm/processed/provenance_raster/{prov_filename}"

        # Upload to blob as COG (stratus expects xarray DataArray)
        logger.info(f"Uploading provenance raster to: {blob_path}")
        stratus.upload_cog_to_blob(
            prov_computed, blob_path, container_name="projects", stage="dev"
        )
        logger.info(f"✅ Provenance raster uploaded with metadata")

        # Log date mapping for reference
        logger.info("\nProvenance date mapping:")
        for idx, date in date_mapping.items():
            logger.info(f"  {idx}: {date}")

        logger.info("\n" + "=" * 60)
        logger.info("FLOOD POLYGON GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"   Polygons created: {len(flood_polygons)}")
        logger.info(f"   Provenance raster saved to blob: {blob_path}")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("FLOOD POLYGON GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"   Polygons created: {len(flood_polygons)}")
        logger.info("   ⚠️  Provenance raster skipped (no STAC items found)")


if __name__ == "__main__":
    main()

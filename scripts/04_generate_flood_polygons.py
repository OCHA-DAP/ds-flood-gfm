import argparse
import logging
from pathlib import Path
import ocha_stratus as stratus
from dotenv import load_dotenv

from ds_flood_gfm.datasources.gfm import (
    create_flood_composite,
    export_polygons,
    query_gfm_stac,
    raster_to_polygons,
)
from ds_flood_gfm.geo_utils import generate_cache_key

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description='Generate flood polygons from GFM STAC data'
    )
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--n-latest', type=int, default=4, help='Number of days to look back (default: 4)')
    parser.add_argument('--iso3', required=True, help='Country ISO3 code (JAM, HTI, CUB)')
    parser.add_argument('--flood-mode', choices=['latest', 'cumulative'], default='latest', help='Flood composite mode')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/polygons'), help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("GFM FLOOD POLYGON GENERATOR")
    logger.info("=" * 60)
    logger.info(f"Country: {args.iso3}")
    logger.info(f"End date: {args.end_date}")
    logger.info(f"Days back: {args.n_latest}")
    logger.info(f"Mode: {args.flood_mode}")
    logger.info("=" * 60)
    
    gdf_admin = stratus.codab.load_codab_from_fieldmaps(args.iso3, 0)
    bbox = gdf_admin.total_bounds
    items = query_gfm_stac(bbox, args.end_date)
    
    if len(items) == 0:
        logger.error("No STAC items found for the specified criteria")
        return
    
    flood_composite, unique_dates = create_flood_composite(
        items, bbox, args.n_latest, mode=args.flood_mode
    )

    flood_polygons = raster_to_polygons(flood_composite)
    
    if len(flood_polygons) == 0:
        logger.warning("No polygons created")
        return
    
    date_str = args.end_date.replace('-', '')
    filename_base = f"{args.iso3.lower()}_flood_{args.flood_mode}_{date_str}"
    output_path = args.output_dir / filename_base
    
    output_path = generate_cache_key(args.iso3, unique_dates, None, args.flood_mode)
    export_polygons(
        flood_polygons, 
        output_path,
        local=False,
        blob=True
    )
    
    logger.info("=" * 60)
    logger.info("FLOOD POLYGON GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Polygons created: {len(flood_polygons)}")



if __name__ == "__main__":
    main()
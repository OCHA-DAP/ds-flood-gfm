"""
Extract Songkhla Province (Thailand) admin boundaries and upload to blob storage.

This script:
1. Loads Thailand (THA) admin levels 1, 2, and 3 boundaries from CODAB
2. Filters for Songkhla Province admin units
3. Saves as geoparquet to blob storage

Usage:
    uv run python scripts/data_prep/extract_songkhla_to_blob.py
"""

import logging
import io
import ocha_stratus as stratus
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


def extract_and_upload_admin_level(admin_level: int):
    """Extract Songkhla Province boundaries for a specific admin level."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EXTRACTING SONGKHLA PROVINCE ADMIN LEVEL {admin_level}")
    logger.info("=" * 60)

    # Load Thailand admin boundaries
    logger.info(f"Loading Thailand (THA) admin level {admin_level} boundaries...")
    gdf_tha = stratus.codab.load_codab_from_fieldmaps("THA", admin_level)
    logger.info(f"  Loaded {len(gdf_tha)} admin units")

    # Filter for Songkhla Province using adm1_name
    # All admin levels have adm1_name column showing their parent province
    logger.info("\nFiltering for Songkhla Province admin units...")
    gdf_songkhla = gdf_tha[gdf_tha['adm1_name'].str.contains('Songkhla', case=False, na=False)]

    if len(gdf_songkhla) == 0:
        logger.warning("No Songkhla admin units found!")
        logger.info(f"Available columns: {gdf_tha.columns.tolist()}")
        if f'adm{admin_level}_name' in gdf_tha.columns:
            logger.info(f"Sample admin names: {gdf_tha[f'adm{admin_level}_name'].head().tolist()}")
        # Try alternative spellings/romanizations
        logger.info("\nTrying alternative filters...")
        gdf_songkhla = gdf_tha[gdf_tha['adm1_name'].str.contains('Songkla|Song Khla', case=False, na=False, regex=True)]
        if len(gdf_songkhla) > 0:
            logger.info(f"  Found {len(gdf_songkhla)} units with alternative spelling")

    if len(gdf_songkhla) == 0:
        logger.error("Still no Songkhla admin units found!")
        logger.info("\nAvailable province names:")
        if 'adm1_name' in gdf_tha.columns:
            for name in sorted(gdf_tha['adm1_name'].unique()):
                logger.info(f"  - {name}")
        return None

    logger.info(f"  Found {len(gdf_songkhla)} Songkhla admin units")

    # Display some example names
    if f'adm{admin_level}_name' in gdf_songkhla.columns:
        sample_names = gdf_songkhla[f'adm{admin_level}_name'].head(10).tolist()
        logger.info(f"  Sample names: {sample_names}")

    # Display bounding box
    bbox = gdf_songkhla.total_bounds
    logger.info(f"  Bounding box: {bbox}")

    # Save as geoparquet and upload to blob
    blob_path = f"ds-flood-gfm/raw/geom/songkhla_province_adm{admin_level}.parquet"
    logger.info(f"\nUploading to blob: {blob_path}")

    # Write to bytes buffer
    buffer = io.BytesIO()
    gdf_songkhla.to_parquet(buffer)
    buffer.seek(0)

    # Upload to blob
    stratus.upload_blob_data(
        buffer.read(),
        blob_path,
        container_name="projects",
        stage="dev"
    )

    logger.info(f"✅ Successfully uploaded admin level {admin_level}")
    logger.info(f"   Blob path: {blob_path}")
    logger.info(f"   Features: {len(gdf_songkhla)}")

    return gdf_songkhla


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SONGKHLA PROVINCE EXTRACTION")
    logger.info("=" * 60)

    # Extract all three admin levels
    for level in [1, 2, 3]:
        result = extract_and_upload_admin_level(level)
        if result is None:
            logger.error(f"Failed to extract admin level {level}")
            break

    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)

"""
Extract Gaza Strip admin boundaries and upload to blob storage.

This script:
1. Loads Palestine (PSE) admin levels 1, 2, and 3 boundaries from CODAB
2. Filters for Gaza Strip admin units
3. Saves as geoparquet to blob storage

Usage:
    uv run python scripts/data_prep/extract_gaza_to_blob.py
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
    """Extract Gaza Strip boundaries for a specific admin level."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EXTRACTING GAZA STRIP ADMIN LEVEL {admin_level}")
    logger.info("=" * 60)

    # Load Palestine admin boundaries
    logger.info(f"Loading Palestine (PSE) admin level {admin_level} boundaries...")
    gdf_pse = stratus.codab.load_codab_from_fieldmaps("PSE", admin_level)
    logger.info(f"  Loaded {len(gdf_pse)} admin units")

    # Filter for Gaza Strip using adm1_name (Gaza governorate)
    # All admin levels have adm1_name column showing their parent governorate
    logger.info("\nFiltering for Gaza Strip admin units...")
    gdf_gaza = gdf_pse[gdf_pse['adm1_name'].str.contains('Gaza', case=False, na=False)]

    if len(gdf_gaza) == 0:
        logger.warning("No Gaza admin units found!")
        logger.info(f"Available columns: {gdf_pse.columns.tolist()}")
        if f'adm{admin_level}_name' in gdf_pse.columns:
            logger.info(f"Sample admin names: {gdf_pse[f'adm{admin_level}_name'].head().tolist()}")
        return None

    logger.info(f"  Found {len(gdf_gaza)} Gaza admin units")

    # Display some example names
    if f'adm{admin_level}_name' in gdf_gaza.columns:
        sample_names = gdf_gaza[f'adm{admin_level}_name'].head(5).tolist()
        logger.info(f"  Sample names: {sample_names}")

    # Display bounding box
    bbox = gdf_gaza.total_bounds
    logger.info(f"  Bounding box: {bbox}")

    # Save as geoparquet and upload to blob
    blob_path = f"ds-flood-gfm/raw/geom/gaza_strip_adm{admin_level}.parquet"
    logger.info(f"\nUploading to blob: {blob_path}")

    # Write to bytes buffer
    buffer = io.BytesIO()
    gdf_gaza.to_parquet(buffer)
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
    logger.info(f"   Features: {len(gdf_gaza)}")
    logger.info(f"   CRS: {gdf_gaza.crs}")

    return blob_path


def main():
    logger.info("=" * 60)
    logger.info("EXTRACT GAZA STRIP TO BLOB (ALL ADMIN LEVELS)")
    logger.info("=" * 60)

    # Extract and upload all three admin levels
    admin_levels = [1, 2, 3]
    uploaded_paths = []

    for level in admin_levels:
        result = extract_and_upload_admin_level(level)
        if result:
            uploaded_paths.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info("Uploaded files:")
    for path in uploaded_paths:
        logger.info(f"  - {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Download CODAB administrative boundaries and upload to Azure Blob Storage.

Downloads admin levels 1-3 for Cuba, Jamaica, and Haiti from FieldMaps.io
and uploads them to Azure Blob Storage in parquet format.
"""
import pandas as pd

from ds_flood_gfm.geo_utils import load_fieldmaps_parquet
from ds_flood_gfm.constants import PROJECT_PREFIX
from ocha_stratus import upload_parquet_to_blob


def download_and_upload_codab(iso3_codes, adm_levels, stage="dev"):
    """Download CODAB boundaries and upload to blob storage.

    Parameters
    ----------
    iso3_codes : list of str
        ISO3 country codes to download (e.g., ["CUB", "JAM", "HTI"])
    adm_levels : list of int
        Administrative levels to download (e.g., [1, 2, 3])
    stage : str, optional
        Azure stage to upload to ("dev" or "prod"), by default "dev"
    """
    for iso3 in iso3_codes:
        iso3_lower = iso3.lower()
        print(f"\n{'='*60}")
        print(f"Processing {iso3}")
        print(f"{'='*60}")

        for adm_level in adm_levels:
            print(f"\nDownloading ADM{adm_level} for {iso3}...")

            try:
                # Download from FieldMaps
                gdf = load_fieldmaps_parquet(iso3, adm_level=adm_level)

                # Convert GeoDataFrame to pandas DataFrame
                # Convert geometry to WKB for storage
                df = gdf.copy()
                df['geometry'] = df['geometry'].to_wkb()

                # Create blob path
                blob_name = (
                    f"{PROJECT_PREFIX}/raw/codab/{iso3_lower}/"
                    f"{iso3_lower}_adm{adm_level}.parquet"
                )

                print(f"  Loaded {len(df)} features")
                print(f"  Uploading to blob: {blob_name}")

                # Upload to Azure Blob
                # (convert to regular pandas DataFrame)
                df_pandas = pd.DataFrame(df.drop(columns='geometry', errors='ignore'))
                df_pandas['geometry'] = df['geometry']

                upload_parquet_to_blob(
                    df=df_pandas,
                    blob_name=blob_name,
                    stage=stage,
                    container_name="projects"
                )

                print(f"  ✓ Successfully uploaded ADM{adm_level} for {iso3}")

            except ValueError as e:
                print(f"  ✗ Error: {e}")
                continue
            except Exception as e:
                print(f"  ✗ Unexpected error: {e}")
                continue

    print(f"\n{'='*60}")
    print("Download and upload complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Configuration
    ISO3_CODES = ["CUB", "JAM", "HTI"]
    ADM_LEVELS = [1, 2, 3]
    STAGE = "dev"  # Change to "prod" for production

    print("CODAB Download and Upload Script")
    print(f"Countries: {', '.join(ISO3_CODES)}")
    print(f"Admin levels: {', '.join(map(str, ADM_LEVELS))}")
    print(f"Azure stage: {STAGE}")

    download_and_upload_codab(ISO3_CODES, ADM_LEVELS, stage=STAGE)

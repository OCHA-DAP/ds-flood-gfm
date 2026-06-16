"""Utilities for uploading/downloading GFM cache and outputs to/from Azure Blob Storage.

This module provides functions to:
- Upload cache directories (flood_points.parquet, provenance.tif, metadata.json)
- Download cache from blob for reuse
- Upload choropleth PNGs
- List available caches in blob storage

Blob Structure:
    projects/ds-flood-gfm/
    ├── processed/cache/{ISO3}/{cache_key}/
    │   ├── flood_points.parquet
    │   ├── provenance.tif
    │   └── metadata.json
    └── outputs/choropleths/{ISO3}/{filename}.png
"""

from pathlib import Path
import json
from typing import Literal, Optional
from ocha_stratus import (
    upload_parquet_to_blob,
    load_parquet_from_blob,
    upload_cog_to_blob,
    upload_blob_data,
    load_blob_data,
    list_container_blobs
)
from ds_flood_gfm.constants import PROJECT_PREFIX


def upload_cache_to_blob(
    cache_dir: Path,
    iso3: str,
    cache_key: str,
    stage: Literal["dev", "prod"] = "dev",
    container_name: str = "projects"
) -> dict:
    """Upload a complete cache directory to Azure Blob Storage.

    Parameters
    ----------
    cache_dir : Path
        Local path to cache directory containing:
        - flood_points.parquet
        - provenance.tif
        - metadata.json
    iso3 : str
        ISO3 country code (e.g., 'JAM', 'HTI', 'CUB')
    cache_key : str
        Human-readable cache key (e.g., 'JAM_20241020_20241022_20241025_ghsl_cumulative')
    stage : Literal["dev", "prod"], optional
        Azure stage, by default "dev"
    container_name : str, optional
        Azure container name, by default "projects"

    Returns
    -------
    dict
        Upload status with blob paths

    Example
    -------
    >>> from pathlib import Path
    >>> upload_cache_to_blob(
    ...     cache_dir=Path("data/cache/JAM_20241020_20241022_20241025_ghsl_cumulative"),
    ...     iso3="JAM",
    ...     cache_key="JAM_20241020_20241022_20241025_ghsl_cumulative",
    ...     stage="dev"
    ... )
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    base_blob_path = f"{PROJECT_PREFIX}/processed/cache/{iso3}/{cache_key}"
    uploaded = {}

    # Upload flood_points.parquet
    flood_points_file = cache_dir / "flood_points.parquet"
    if flood_points_file.exists():
        import pandas as pd
        df = pd.read_parquet(flood_points_file)
        blob_name = f"{base_blob_path}/flood_points.parquet"
        upload_parquet_to_blob(df, blob_name, stage=stage, container_name=container_name)
        uploaded['flood_points'] = blob_name
        print(f"  ✓ Uploaded flood_points.parquet")

    # Upload provenance.tif
    provenance_file = cache_dir / "provenance.tif"
    if provenance_file.exists():
        with open(provenance_file, 'rb') as f:
            data = f.read()
        blob_name = f"{base_blob_path}/provenance.tif"
        upload_blob_data(
            data,
            blob_name,
            stage=stage,
            container_name=container_name
        )
        uploaded['provenance'] = blob_name
        print(f"  ✓ Uploaded provenance.tif")

    # Upload metadata.json
    metadata_file = cache_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = f.read()
        blob_name = f"{base_blob_path}/metadata.json"
        upload_blob_data(
            metadata.encode('utf-8'),
            blob_name,
            stage=stage,
            container_name=container_name
        )
        uploaded['metadata'] = blob_name
        print(f"  ✓ Uploaded metadata.json")

    print(f"\nCache uploaded to: {base_blob_path}")
    return uploaded


def download_cache_from_blob(
    iso3: str,
    cache_key: str,
    output_dir: Path,
    stage: Literal["dev", "prod"] = "dev",
    container_name: str = "projects"
) -> Path:
    """Download a cache directory from Azure Blob Storage.

    Parameters
    ----------
    iso3 : str
        ISO3 country code
    cache_key : str
        Cache key to download
    output_dir : Path
        Local directory to save cache
    stage : Literal["dev", "prod"], optional
        Azure stage, by default "dev"
    container_name : str, optional
        Azure container name, by default "projects"

    Returns
    -------
    Path
        Path to downloaded cache directory

    Example
    -------
    >>> cache_dir = download_cache_from_blob(
    ...     iso3="JAM",
    ...     cache_key="JAM_20241020_20241022_20241025_ghsl_cumulative",
    ...     output_dir=Path("data/cache"),
    ...     stage="dev"
    ... )
    """
    import pandas as pd

    base_blob_path = f"{PROJECT_PREFIX}/processed/cache/{iso3}/{cache_key}"
    cache_dir = Path(output_dir) / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download flood_points.parquet
    blob_name = f"{base_blob_path}/flood_points.parquet"
    try:
        df = load_parquet_from_blob(blob_name, stage=stage, container_name=container_name)
        df.to_parquet(cache_dir / "flood_points.parquet")
        print(f"  ✓ Downloaded flood_points.parquet")
    except Exception as e:
        print(f"  ⚠ Could not download flood_points.parquet: {e}")

    # Download provenance.tif
    blob_name = f"{base_blob_path}/provenance.tif"
    try:
        data = load_blob_data(blob_name, stage=stage, container_name=container_name)
        with open(cache_dir / "provenance.tif", 'wb') as f:
            f.write(data)
        print(f"  ✓ Downloaded provenance.tif")
    except Exception as e:
        print(f"  ⚠ Could not download provenance.tif: {e}")

    # Download metadata.json
    blob_name = f"{base_blob_path}/metadata.json"
    try:
        data = load_blob_data(blob_name, stage=stage, container_name=container_name)
        with open(cache_dir / "metadata.json", 'wb') as f:
            f.write(data)
        print(f"  ✓ Downloaded metadata.json")
    except Exception as e:
        print(f"  ⚠ Could not download metadata.json: {e}")

    print(f"\nCache downloaded to: {cache_dir}")
    return cache_dir


def upload_choropleth_to_blob(
    png_file: Path,
    iso3: str,
    stage: Literal["dev", "prod"] = "dev",
    container_name: str = "projects"
) -> str:
    """Upload a choropleth PNG to Azure Blob Storage.

    Parameters
    ----------
    png_file : Path
        Local path to PNG file
    iso3 : str
        ISO3 country code
    stage : Literal["dev", "prod"], optional
        Azure stage, by default "dev"
    container_name : str, optional
        Azure container name, by default "projects"

    Returns
    -------
    str
        Blob path

    Example
    -------
    >>> upload_choropleth_to_blob(
    ...     png_file=Path("experiments/JAM_choropleth_adm3_20251027.png"),
    ...     iso3="JAM",
    ...     stage="dev"
    ... )
    """
    png_file = Path(png_file)
    if not png_file.exists():
        raise FileNotFoundError(f"PNG file not found: {png_file}")

    filename = png_file.name
    blob_name = f"{PROJECT_PREFIX}/outputs/choropleths/{iso3}/{filename}"

    with open(png_file, 'rb') as f:
        data = f.read()

    upload_blob_data(
        data,
        blob_name,
        stage=stage,
        container_name=container_name,
        content_type="image/png"
    )

    print(f"✓ Uploaded choropleth: {blob_name}")
    return blob_name


def list_available_caches(
    iso3: Optional[str] = None,
    stage: Literal["dev", "prod"] = "dev",
    container_name: str = "projects"
) -> list:
    """List available caches in blob storage.

    Parameters
    ----------
    iso3 : str, optional
        Filter by ISO3 country code. If None, returns all countries.
    stage : Literal["dev", "prod"], optional
        Azure stage, by default "dev"
    container_name : str, optional
        Azure container name, by default "projects"

    Returns
    -------
    list
        List of available cache keys

    Example
    -------
    >>> # List all caches
    >>> list_available_caches()
    ['JAM/JAM_20241020_20241022_20241025_ghsl_cumulative', 'HTI/...']

    >>> # List caches for Jamaica
    >>> list_available_caches(iso3="JAM")
    ['JAM_20241020_20241022_20241025_ghsl_cumulative', 'JAM_...']
    """
    if iso3:
        prefix = f"{PROJECT_PREFIX}/processed/cache/{iso3}/"
    else:
        prefix = f"{PROJECT_PREFIX}/processed/cache/"

    blobs = list_container_blobs(
        prefix=prefix,
        stage=stage,
        container_name=container_name
    )

    # Extract cache keys from blob names
    caches = set()
    for blob in blobs:
        # blob name format: ds-flood-gfm/processed/cache/JAM/cache_key/file.ext
        parts = blob.split('/')
        if len(parts) >= 5:
            if iso3:
                cache_key = parts[4]  # Just the cache_key
            else:
                cache_key = f"{parts[3]}/{parts[4]}"  # ISO3/cache_key
            caches.add(cache_key)

    return sorted(list(caches))


def cache_exists_in_blob(
    iso3: str,
    cache_key: str,
    stage: Literal["dev", "prod"] = "dev",
    container_name: str = "projects"
) -> bool:
    """Check if a cache exists in blob storage.

    Parameters
    ----------
    iso3 : str
        ISO3 country code
    cache_key : str
        Cache key to check
    stage : Literal["dev", "prod"], optional
        Azure stage, by default "dev"
    container_name : str, optional
        Azure container name, by default "projects"

    Returns
    -------
    bool
        True if cache exists with all required files

    Example
    -------
    >>> cache_exists_in_blob("JAM", "JAM_20241020_20241022_20241025_ghsl_cumulative")
    True
    """
    base_path = f"{PROJECT_PREFIX}/processed/cache/{iso3}/{cache_key}"

    required_files = [
        f"{base_path}/flood_points.parquet",
        f"{base_path}/provenance.tif",
        f"{base_path}/metadata.json"
    ]

    blobs = list_container_blobs(
        prefix=base_path,
        stage=stage,
        container_name=container_name
    )

    # Check if all required files exist
    blob_set = set(blobs)
    return all(f in blob_set for f in required_files)

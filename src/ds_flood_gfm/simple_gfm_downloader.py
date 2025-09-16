"""
Simple GFM Historical Data Downloader for East Africa AOI

Just downloads historical GFM data for a given bounding box.
Compositing is a separate step.
"""

from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import logging

from ds_flood_gfm.gfm_downloader import GFMDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_gfm_historical_data(
    bbox: List[float],
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    download_dir: str = "./data/gfm/historical",
    limit: int = None
) -> Dict:
    """
    Download historical GFM data for a given AOI bounding box.

    Args:
        bbox: [west, south, east, north] bounding box
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        download_dir: Directory to save files
        limit: Maximum number of items to download (None for all available)

    Returns:
        Dictionary with download results
    """
    downloader = GFMDownloader()

    logger.info(f"Downloading GFM data for bbox: {bbox}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Download directory: {download_dir}")

    # Search for data
    items = downloader.search_flood_data(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        limit=limit if limit is not None else 10000
    )

    logger.info(f"Found {len(items)} GFM items")

    if len(items) == 0:
        return {
            'items_found': 0,
            'downloaded_files': {},
            'message': 'No items found for the specified criteria'
        }

    # Download the data
    downloaded_files = downloader.download_item_assets(
        items,
        download_dir=download_dir,
        asset_types=['ensemble_flood_extent'],
        create_subdirs=False
    )

    results = {
        'items_found': len(items),
        'items_downloaded': len(downloaded_files),
        'downloaded_files': downloaded_files,
        'download_dir': download_dir,
        'bbox': bbox,
        'date_range': f"{start_date} to {end_date}"
    }

    logger.info(f"Download complete: {len(downloaded_files)} items downloaded")

    return results


def main():
    """Example usage: Download for Somalia region Sept 2023 to Jan 2024."""

    # Somalia AOI - covering major river basins and flood-prone areas
    somalia_bbox = [40.0, -2.0, 51.0, 12.0]  # [west, south, east, north]

    print("Downloading GFM historical data for Somalia region...")

    results = download_gfm_historical_data(
        bbox=somalia_bbox,
        start_date="2023-09-01",   # September 2023
        end_date="2024-01-31",     # End of January 2024
        download_dir="./data/gfm/somalia_example/raw_gfm"
    )

    print(f"\nResults:")
    print(f"Items found: {results['items_found']}")
    print(f"Items downloaded: {results['items_downloaded']}")
    print(f"Download directory: {results['download_dir']}")

    if results['downloaded_files']:
        print(f"\nFirst few downloaded items:")
        for i, (item_id, files) in enumerate(list(results['downloaded_files'].items())[:5]):
            print(f"{i+1}. {item_id}: {len(files)} files")


if __name__ == "__main__":
    main()
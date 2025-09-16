"""
Global Flood Monitoring (GFM) data downloader using EODC STAC API.

This module provides functionality to search and download GFM flood data
from the EODC STAC API service.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import os
import urllib.request
from pathlib import Path
import json
from pystac_client import Client
from pystac import ItemCollection
from shapely.geometry import box, Polygon
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GFMDownloader:
    """Download Global Flood Monitoring data from EODC STAC API."""
    
    # EODC STAC API endpoint
    STAC_ENDPOINT = "https://stac.eodc.eu/api/v1"
    
    # GFM Collection ID
    COLLECTION_ID = "GFM"
    
    # Pakistan bounding box (approximate)
    PAKISTAN_BBOX = [60.872, 23.693, 77.840, 37.097]  # [west, south, east, north]
    
    # Available asset types in GFM collection
    ASSET_TYPES = [
        "ensemble_flood_extent",
        "tuw_flood_extent",
        "dlr_flood_extent", 
        "list_flood_extent",
        "tuw_likelihood",
        "dlr_likelihood",
        "list_likelihood",
        "advisory_flags",
        "exclusion_mask",
        "rendered_preview"
    ]
    
    def __init__(self):
        """Initialize GFM downloader with EODC STAC API."""
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize STAC client."""
        try:
            logger.info(f"Connecting to EODC STAC API: {self.STAC_ENDPOINT}")
            self.client = Client.open(self.STAC_ENDPOINT)
            logger.info("Successfully connected to EODC STAC API")
            
            # Verify GFM collection exists
            collections = [col.id for col in self.client.get_collections()]
            if self.COLLECTION_ID in collections:
                logger.info(f"GFM collection '{self.COLLECTION_ID}' found")
            else:
                logger.warning(f"GFM collection '{self.COLLECTION_ID}' not found. Available: {collections}")
                
        except Exception as e:
            logger.error(f"Failed to connect to EODC STAC API: {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """
        List available collections from the STAC API.
        
        Returns:
            List of collection IDs
        """
        if not self.client:
            logger.error("No STAC client available")
            return []
        
        try:
            collections = list(self.client.get_collections())
            collection_ids = [col.id for col in collections]
            logger.info(f"Found {len(collection_ids)} collections: {collection_ids}")
            return collection_ids
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def search_flood_data(
        self,
        bbox: Optional[List[float]] = None,
        geometry: Optional[Union[Polygon, dict]] = None,
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        datetime_range: Optional[str] = None,
        limit: int = 100
    ) -> ItemCollection:
        """
        Search for flood monitoring data.
        
        Args:
            bbox: Bounding box [west, south, east, north]
            geometry: Shapely geometry or GeoJSON dict for area of interest
            start_date: Start date (ISO format string or datetime)
            end_date: End date (ISO format string or datetime)
            datetime_range: Date range string in format 'YYYY-MM-DD/YYYY-MM-DD'
            limit: Maximum number of items to return
            
        Returns:
            STAC ItemCollection containing search results
        """
        if not self.client:
            raise RuntimeError("No STAC client available")
        
        # Handle datetime parameter
        if datetime_range:
            datetime_param = datetime_range
        elif start_date and end_date:
            # Convert dates to ISO format strings if needed
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            datetime_param = f"{start_date}/{end_date}"
        else:
            datetime_param = None
        
        search_params = {
            "collections": [self.COLLECTION_ID],
            "max_items": limit
        }
        
        # Add spatial filter
        if geometry:
            search_params["intersects"] = geometry
        elif bbox:
            search_params["bbox"] = bbox
        
        # Add temporal filter
        if datetime_param:
            search_params["datetime"] = datetime_param
        
        logger.info(f"Searching GFM collection with parameters: {search_params}")
        
        try:
            search = self.client.search(**search_params)
            items = search.item_collection()
            logger.info(f"Found {len(items)} items")
            return items
            
        except Exception as e:
            logger.error(f"Error searching for data: {e}")
            raise
    
    def search_pakistan_floods(
        self,
        start_date: Union[str, datetime] = "2023-06-01",
        end_date: Union[str, datetime] = "2023-09-30", 
        limit: int = 50
    ) -> ItemCollection:
        """
        Search for flood data over Pakistan for a given time period.
        Uses Pakistan bounding box for spatial filtering.
        
        Args:
            start_date: Start date for search (default: monsoon season start)
            end_date: End date for search (default: monsoon season end)
            limit: Maximum number of items to return
            
        Returns:
            STAC ItemCollection containing Pakistan flood data
        """
        # Create Pakistan AOI as shapely geometry
        pakistan_aoi = box(*self.PAKISTAN_BBOX)
        
        return self.search_flood_data(
            geometry=pakistan_aoi,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    
    def download_item_assets(
        self,
        items: ItemCollection,
        download_dir: Union[str, Path] = "./data",
        asset_types: Optional[List[str]] = None,
        create_subdirs: bool = True
    ) -> Dict[str, List[str]]:
        """
        Download assets from STAC items.
        
        Args:
            items: STAC ItemCollection to download
            download_dir: Directory to save downloaded files
            asset_types: Specific asset types to download. If None, downloads key assets.
            create_subdirs: Create subdirectories for each item
            
        Returns:
            Dictionary mapping item IDs to list of downloaded file paths
        """
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Default to key flood extent assets if none specified
        if asset_types is None:
            asset_types = ["ensemble_flood_extent", "tuw_flood_extent"]
        
        downloaded_files = {}
        
        for item in items:
            item_id = item.id
            logger.info(f"Processing item: {item_id}")
            
            if create_subdirs:
                item_dir = download_dir / item_id
                item_dir.mkdir(parents=True, exist_ok=True)
            else:
                item_dir = download_dir
            
            # Save item metadata
            metadata_file = item_dir / f"{item_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(item.to_dict(), f, indent=2)
            
            downloaded_files[item_id] = [str(metadata_file)]
            
            # Download specified assets
            logger.info(f"Available assets for {item_id}: {list(item.assets.keys())}")
            
            for asset_type in asset_types:
                if asset_type not in item.assets:
                    logger.warning(f"Asset '{asset_type}' not found in item {item_id}")
                    continue
                
                asset = item.assets[asset_type]
                
                # Skip assets without 'data' role (e.g., metadata, previews)
                if "data" not in asset.roles:
                    logger.info(f"Skipping asset '{asset_type}' (no 'data' role)")
                    continue
                
                # Generate filename from asset href
                filename = os.path.basename(asset.href)
                if not filename or filename == asset.href:
                    # Fallback filename if basename extraction fails
                    filename = f"{item_id}_{asset_type}.tif"
                
                filepath = item_dir / filename

                # Check if file already exists
                if filepath.exists():
                    logger.info(f"File already exists, skipping: {filepath}")
                    downloaded_files[item_id].append(str(filepath))
                    continue

                try:
                    logger.info(f"Downloading {asset_type} from {asset.href}")
                    urllib.request.urlretrieve(asset.href, str(filepath))

                    downloaded_files[item_id].append(str(filepath))
                    logger.info(f"Downloaded: {filepath}")

                except Exception as e:
                    logger.error(f"Error downloading {asset_type} for item {item_id}: {e}")
        
        return downloaded_files
    
    def get_collection_info(self) -> Dict:
        """
        Get detailed information about the GFM collection.
        
        Returns:
            Collection information as dictionary
        """
        if not self.client:
            raise RuntimeError("No STAC client available")
        
        try:
            collection = self.client.get_collection(self.COLLECTION_ID)
            return collection.to_dict()
        except Exception as e:
            logger.error(f"Error getting GFM collection info: {e}")
            raise
    
    def print_item_summary(self, items: ItemCollection) -> None:
        """
        Print a summary of items found in search.
        
        Args:
            items: STAC ItemCollection to summarize
        """
        print(f"\nFound {len(items)} GFM items:")
        print("-" * 80)
        
        for i, item in enumerate(items):
            print(f"{i+1}. Item ID: {item.id}")
            print(f"   Date: {item.datetime}")
            print(f"   Assets: {list(item.assets.keys())}")
            
            # Print bounding box if available
            if hasattr(item, 'bbox') and item.bbox:
                print(f"   Bbox: {item.bbox}")
            
            print()


def main():
    """Example usage of GFMDownloader for Pakistan floods."""
    try:
        # Initialize downloader
        downloader = GFMDownloader()
        
        # Get collection info
        print("GFM Collection Information:")
        collection_info = downloader.get_collection_info()
        print(f"Title: {collection_info.get('title', 'N/A')}")
        print(f"Description: {collection_info.get('description', 'N/A')}")
        print()
        
        # Search for Pakistan flood data (limited sample)
        print("Searching for Pakistan flood data...")
        items = downloader.search_pakistan_floods(
            start_date="2022-09-15",  # Historical flood event
            end_date="2022-09-16",
            limit=5  # Small sample for testing
        )
        
        # Print summary
        downloader.print_item_summary(items)
        
        # Download sample data (metadata + key assets)
        if len(items) > 0:
            print("Downloading sample data...")
            downloaded = downloader.download_item_assets(
                items[:2],  # Download only first 2 items
                download_dir="./data/gfm",
                asset_types=["ensemble_flood_extent"]  # Download main flood extent only
            )
            
            print("\nDownload Summary:")
            for item_id, files in downloaded.items():
                print(f"{item_id}: {len(files)} files")
                for file in files:
                    print(f"  - {file}")
        else:
            print("No items found for the specified criteria.")
    
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Full error details:")


if __name__ == "__main__":
    main()
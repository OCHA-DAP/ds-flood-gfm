"""
STAC-Aware Daily GFM Compositor

This module creates daily composite GFM flood extents using STAC metadata
instead of filename parsing. It properly handles STAC Item JSON files
to extract datetime, spatial information, and asset paths.
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.crs import CRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STACDailyCompositor:
    """Create daily GFM composites using STAC metadata."""

    def __init__(self):
        """Initialize the STAC-aware compositor."""
        pass

    def load_stac_metadata(self, metadata_dir: Path) -> List[Dict]:
        """
        Load all STAC metadata JSON files from directory.

        Args:
            metadata_dir: Directory containing *_metadata.json files

        Returns:
            List of parsed STAC Item dictionaries
        """
        metadata_files = list(metadata_dir.glob("*_metadata.json"))
        logger.info(f"Found {len(metadata_files)} STAC metadata files")

        stac_items = []
        failed_files = []

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    stac_item = json.load(f)

                # Basic STAC validation
                if not self._validate_stac_item(stac_item):
                    logger.warning(f"Invalid STAC item: {metadata_file}")
                    continue

                # Add file path for reference
                stac_item['_metadata_file'] = str(metadata_file)
                stac_items.append(stac_item)

            except Exception as e:
                logger.error(f"Error loading {metadata_file}: {e}")
                failed_files.append(str(metadata_file))

        logger.info(f"Successfully loaded {len(stac_items)} STAC items")
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files")

        return stac_items

    def _validate_stac_item(self, stac_item: Dict) -> bool:
        """
        Validate basic STAC Item structure.

        Args:
            stac_item: STAC Item dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['type', 'id', 'properties', 'assets']
        for field in required_fields:
            if field not in stac_item:
                return False

        # Check for datetime in properties
        if 'datetime' not in stac_item['properties']:
            return False

        # Check for ensemble_flood_extent asset
        if 'ensemble_flood_extent' not in stac_item['assets']:
            return False

        return True

    def extract_datetime_from_stac(self, stac_item: Dict) -> datetime:
        """
        Extract datetime from STAC Item properties.

        Args:
            stac_item: STAC Item dictionary

        Returns:
            Parsed datetime object
        """
        datetime_str = stac_item['properties']['datetime']

        # Handle different datetime formats
        if datetime_str.endswith('Z'):
            # Remove 'Z' and parse as UTC
            datetime_str = datetime_str[:-1] + '+00:00'

        return datetime.fromisoformat(datetime_str)

    def group_stac_items_by_date(self, stac_items: List[Dict]) -> Dict[date, List[Dict]]:
        """
        Group STAC items by observation date.

        Args:
            stac_items: List of STAC Item dictionaries

        Returns:
            Dictionary mapping dates to lists of STAC items
        """
        logger.info("Grouping STAC items by observation date")

        grouped_items = defaultdict(list)
        failed_items = []

        for stac_item in stac_items:
            try:
                item_datetime = self.extract_datetime_from_stac(stac_item)
                item_date = item_datetime.date()
                grouped_items[item_date].append(stac_item)

            except Exception as e:
                logger.warning(f"Failed to extract datetime from {stac_item['id']}: {e}")
                failed_items.append(stac_item['id'])

        # Convert to regular dict and sort dates
        grouped_items = dict(grouped_items)

        logger.info(f"Grouped items into {len(grouped_items)} distinct dates")
        for obs_date, items in grouped_items.items():
            logger.info(f"  {obs_date}: {len(items)} tiles")

        if failed_items:
            logger.warning(f"Failed to process {len(failed_items)} items")

        return grouped_items

    def get_geotiff_path_from_stac(self, stac_item: Dict, base_dir: Path) -> Optional[Path]:
        """
        Get the local path to the GeoTIFF file from STAC asset information.

        Args:
            stac_item: STAC Item dictionary
            base_dir: Base directory containing the downloaded files

        Returns:
            Path to the GeoTIFF file, or None if not found
        """
        item_id = stac_item['id']

        # The GeoTIFF should be in the same directory as the metadata
        # with the same name as the item ID
        geotiff_path = base_dir / f"{item_id}.tif"

        if geotiff_path.exists():
            return geotiff_path

        # Alternative: try extracting filename from asset href
        try:
            asset_href = stac_item['assets']['ensemble_flood_extent']['href']
            filename = Path(asset_href).name
            alt_path = base_dir / filename

            if alt_path.exists():
                return alt_path
        except Exception:
            pass

        logger.warning(f"GeoTIFF not found for {item_id}")
        return None

    def create_daily_composite(self, obs_date: date, stac_items: List[Dict],
                             base_dir: Path, output_dir: Path) -> Optional[Path]:
        """
        Create a daily composite from multiple GFM tiles for a single date.

        Args:
            obs_date: Observation date
            stac_items: List of STAC items for this date
            base_dir: Directory containing the GeoTIFF files
            output_dir: Directory for composite outputs

        Returns:
            Path to created composite file, or None if failed
        """
        logger.info(f"Creating daily composite for {obs_date} from {len(stac_items)} tiles")

        # Collect GeoTIFF paths
        geotiff_paths = []
        for stac_item in stac_items:
            geotiff_path = self.get_geotiff_path_from_stac(stac_item, base_dir)
            if geotiff_path:
                geotiff_paths.append(geotiff_path)
            else:
                logger.warning(f"Missing GeoTIFF for {stac_item['id']}")

        if len(geotiff_paths) < 1:
            logger.error(f"No valid GeoTIFF files found for {obs_date}")
            return None

        if len(geotiff_paths) == 1:
            logger.info(f"Only one tile for {obs_date}, copying instead of merging")
            # Could copy the single file or still process it through merge for consistency

        logger.info(f"Merging {len(geotiff_paths)} tiles for {obs_date}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open all datasets
        datasets = []
        try:
            for geotiff_path in geotiff_paths:
                dataset = rasterio.open(geotiff_path)
                datasets.append(dataset)

            # Merge datasets using maximum value (flood=1 takes priority)
            merged_data, merged_transform = merge(
                datasets,
                method='max',  # Take maximum value in overlapping areas
                dtype=rasterio.uint8
            )

            # Get CRS from first dataset (should be consistent across tiles)
            merged_crs = datasets[0].crs

            # Create output filename
            output_filename = f"somalia_gfm_composite_{obs_date.strftime('%Y-%m-%d')}.tif"
            output_path = output_dir / output_filename

            # Write composite
            profile = {
                'driver': 'GTiff',
                'height': merged_data.shape[1],
                'width': merged_data.shape[2],
                'count': 1,
                'dtype': rasterio.uint8,
                'crs': merged_crs,
                'transform': merged_transform,
                'compress': 'lzw',
                'nodata': 255
            }

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(merged_data[0], 1)

            # Log statistics
            flood_pixels = np.sum(merged_data[0] == 1)
            total_pixels = merged_data[0].size
            flood_percentage = (flood_pixels / total_pixels) * 100

            logger.info(f"Created composite: {output_path}")
            logger.info(f"  Flood pixels: {flood_pixels:,} ({flood_percentage:.3f}%)")
            logger.info(f"  Dimensions: {merged_data.shape[1]} x {merged_data.shape[2]}")

            return output_path

        except Exception as e:
            logger.error(f"Error creating composite for {obs_date}: {e}")
            return None

        finally:
            # Close all datasets
            for dataset in datasets:
                try:
                    dataset.close()
                except:
                    pass

    def process_all_dates(self, input_dir: Path, output_dir: Path,
                         max_dates: Optional[int] = None) -> Dict[str, Path]:
        """
        Process all dates in the input directory to create daily composites.

        Args:
            input_dir: Directory containing STAC metadata and GeoTIFF files
            output_dir: Directory for composite outputs
            max_dates: Maximum number of dates to process (None for all)

        Returns:
            Dictionary mapping date strings to composite file paths
        """
        logger.info("=" * 60)
        logger.info("STARTING STAC-AWARE DAILY COMPOSITING")
        logger.info("=" * 60)

        # Load STAC metadata
        stac_items = self.load_stac_metadata(input_dir)
        if not stac_items:
            logger.error("No valid STAC items found")
            return {}

        # Group by date
        grouped_items = self.group_stac_items_by_date(stac_items)
        if not grouped_items:
            logger.error("No items could be grouped by date")
            return {}

        # Sort dates and optionally limit
        sorted_dates = sorted(grouped_items.keys())
        if max_dates:
            sorted_dates = sorted_dates[:max_dates]
            logger.info(f"Processing first {max_dates} dates only")

        # Process each date
        composites = {}
        failed_dates = []

        for obs_date in sorted_dates:
            date_items = grouped_items[obs_date]
            logger.info(f"\nProcessing {obs_date}: {len(date_items)} tiles")

            composite_path = self.create_daily_composite(
                obs_date, date_items, input_dir, output_dir
            )

            if composite_path:
                composites[obs_date.strftime('%Y-%m-%d')] = composite_path
            else:
                failed_dates.append(obs_date)

        # Summary
        logger.info(f"\n" + "=" * 60)
        logger.info("COMPOSITING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Successfully processed: {len(composites)} dates")
        logger.info(f"Failed dates: {len(failed_dates)}")

        if composites:
            logger.info(f"\nCreated composites:")
            for date_str, path in composites.items():
                logger.info(f"  {date_str}: {path.name}")

        return composites


def main():
    """Example usage: Process Somalia GFM data into daily composites."""

    compositor = STACDailyCompositor()

    # Paths
    input_dir = Path("data/gfm/somalia_example/raw_gfm")
    output_dir = Path("data/gfm/somalia_example/daily_composites")

    print("Processing Somalia GFM data into daily composites...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Process all available dates
    composites = compositor.process_all_dates(
        input_dir=input_dir,
        output_dir=output_dir,
        max_dates=5  # Limit to first 5 dates for testing
    )

    print(f"\nResults:")
    print(f"Created {len(composites)} daily composites")

    if composites:
        print(f"\nComposite files:")
        for date_str, path in composites.items():
            print(f"  {date_str}: {path}")


if __name__ == "__main__":
    main()
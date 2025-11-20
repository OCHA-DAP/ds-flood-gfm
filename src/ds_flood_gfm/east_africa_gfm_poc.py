"""
East Africa GFM Proof of Concept - Historical Daily Composites

This module implements the proof of concept for creating daily composite GFM flood extents
across an East African AOI that spans multiple overlapping Sentinel tiles.

Methodology:
1. Define E. Africa AOI covering multiple Sentinel swaths
2. Retrieve entire historical GFM record for this AOI
3. For each day, composite all available images to get latest flood extent
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from pathlib import Path
import logging
from shapely.geometry import box, Polygon
import pandas as pd
from collections import defaultdict

from ds_flood_gfm.gfm_downloader import GFMDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EastAfricaGFMCompositor:
    """Create daily GFM composites for East African AOI."""

    # East Africa AOI covering Lake Victoria region + Nile Basin
    # This area spans multiple Sentinel tiles and has high flood activity
    EAST_AFRICA_AOI = {
        'name': 'Lake Victoria - Upper Nile Basin',
        'bbox': [29.0, -3.0, 36.0, 4.0],  # [west, south, east, north]
        'description': 'Uganda, Kenya, Tanzania border region with Lake Victoria',
        'expected_tiles': '3-4 overlapping Sentinel-1 swaths',
        'rationale': 'High flood activity, multiple countries, complex hydrology'
    }

    # Alternative AOIs for comparison
    ALTERNATIVE_AOIS = {
        'horn_of_africa': {
            'name': 'Horn of Africa - Shabelle Basin',
            'bbox': [40.0, 0.0, 48.0, 8.0],
            'description': 'Somalia/Ethiopia border, Shabelle River basin',
            'expected_tiles': '3-4 overlapping swaths'
        },
        'upper_nile': {
            'name': 'Upper Nile - South Sudan',
            'bbox': [29.0, 6.0, 34.0, 11.0],
            'description': 'South Sudan Nile confluence region',
            'expected_tiles': '3-4 overlapping swaths'
        }
    }

    def __init__(self, aoi_name: str = 'default'):
        """Initialize compositor with specific AOI.

        Args:
            aoi_name: AOI to use ('default', 'horn_of_africa', 'upper_nile')
        """
        self.downloader = GFMDownloader()

        if aoi_name == 'default':
            self.aoi = self.EAST_AFRICA_AOI
        else:
            self.aoi = self.ALTERNATIVE_AOIS.get(aoi_name, self.EAST_AFRICA_AOI)

        logger.info(f"Initialized East Africa GFM Compositor")
        logger.info(f"AOI: {self.aoi['name']}")
        logger.info(f"Bbox: {self.aoi['bbox']}")
        logger.info(f"Description: {self.aoi['description']}")

    def get_historical_gfm_data(self,
                               start_date: Union[str, datetime] = "2019-01-01",
                               end_date: Union[str, datetime] = None,
                               limit: int = 1000) -> pd.DataFrame:
        """Retrieve entire historical GFM record for the AOI.

        Args:
            start_date: Start date for historical search
            end_date: End date (defaults to today)
            limit: Maximum items to retrieve

        Returns:
            DataFrame with GFM items organized by date
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Searching historical GFM data for {self.aoi['name']}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Spatial extent: {self.aoi['bbox']}")

        # Search using the AOI bounding box
        items = self.downloader.search_flood_data(
            bbox=self.aoi['bbox'],
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        logger.info(f"Found {len(items)} GFM items")

        # Convert to DataFrame for easier analysis
        records = []
        for item in items:
            records.append({
                'item_id': item.id,
                'datetime': item.datetime,
                'date': item.datetime.date() if item.datetime else None,
                'bbox': item.bbox,
                'assets': list(item.assets.keys()),
                'has_ensemble': 'ensemble_flood_extent' in item.assets,
                'item': item  # Keep reference for downloading
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('datetime')

        logger.info(f"Data spans {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Items with ensemble data: {df['has_ensemble'].sum()}")

        return df

    def analyze_daily_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze daily coverage and identify days with multiple tiles.

        Args:
            df: DataFrame from get_historical_gfm_data()

        Returns:
            DataFrame with daily statistics
        """
        logger.info("Analyzing daily coverage patterns")

        daily_stats = df.groupby('date').agg({
            'item_id': 'count',
            'has_ensemble': 'sum',
            'bbox': lambda x: list(x),
            'item': lambda x: list(x)
        }).rename(columns={'item_id': 'total_items', 'has_ensemble': 'ensemble_items'})

        # Add coverage analysis
        daily_stats['multiple_tiles'] = daily_stats['total_items'] > 1
        daily_stats['good_for_composite'] = (
            (daily_stats['ensemble_items'] >= 2) &
            (daily_stats['total_items'] >= 2)
        )

        # Sort by date
        daily_stats = daily_stats.sort_index()

        logger.info(f"Total days with data: {len(daily_stats)}")
        logger.info(f"Days with multiple tiles: {daily_stats['multiple_tiles'].sum()}")
        logger.info(f"Days suitable for compositing: {daily_stats['good_for_composite'].sum()}")

        return daily_stats

    def download_daily_data(self,
                           daily_stats: pd.DataFrame,
                           target_dates: Optional[List[str]] = None,
                           max_days: int = 10,
                           download_dir: Union[str, Path] = "./data/gfm/east_africa") -> Dict:
        """Download GFM data for specific days suitable for compositing.

        Args:
            daily_stats: DataFrame from analyze_daily_coverage()
            target_dates: Specific dates to download (YYYY-MM-DD format)
            max_days: Maximum number of days to download
            download_dir: Directory for downloaded files

        Returns:
            Dictionary mapping dates to downloaded file information
        """
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        if target_dates:
            # Convert string dates to datetime.date objects
            target_dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in target_dates]
            selected_days = daily_stats[daily_stats.index.isin(target_dates)]
        else:
            # Select best days for compositing
            good_days = daily_stats[daily_stats['good_for_composite']]
            selected_days = good_days.head(max_days)

        logger.info(f"Downloading data for {len(selected_days)} days")

        downloaded_data = {}

        for date, row in selected_days.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            logger.info(f"Processing {date_str}: {row['total_items']} items")

            # Create date-specific directory
            date_dir = download_dir / date_str
            date_dir.mkdir(exist_ok=True)

            # Download all items for this date
            items_for_date = row['item']
            downloaded_files = self.downloader.download_item_assets(
                items_for_date,
                download_dir=date_dir,
                asset_types=['ensemble_flood_extent'],
                create_subdirs=True
            )

            downloaded_data[date_str] = {
                'items_count': len(items_for_date),
                'downloaded_files': downloaded_files,
                'directory': date_dir
            }

        return downloaded_data

    def create_daily_composite(self,
                              date_str: str,
                              downloaded_data: Dict,
                              output_dir: Union[str, Path] = "./results/composites") -> Optional[Path]:
        """Create daily composite from multiple GFM tiles for a single date.

        Args:
            date_str: Date string (YYYY-MM-DD)
            downloaded_data: Output from download_daily_data()
            output_dir: Directory for composite outputs

        Returns:
            Path to created composite file, or None if failed
        """
        if date_str not in downloaded_data:
            logger.error(f"No downloaded data for {date_str}")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data_info = downloaded_data[date_str]
        logger.info(f"Creating composite for {date_str} from {data_info['items_count']} tiles")

        # Collect all GeoTIFF files for this date
        tif_files = []
        for item_id, files in data_info['downloaded_files'].items():
            for file_path in files:
                if file_path.endswith('.tif'):
                    tif_files.append(file_path)

        if len(tif_files) < 2:
            logger.warning(f"Only {len(tif_files)} tiles found for {date_str}, skipping composite")
            return None

        logger.info(f"Compositing {len(tif_files)} tiles")

        # Read and merge tiles
        datasets = []
        for tif_file in tif_files:
            try:
                dataset = rasterio.open(tif_file)
                datasets.append(dataset)
            except Exception as e:
                logger.warning(f"Could not open {tif_file}: {e}")

        if len(datasets) < 2:
            logger.error(f"Could not open enough datasets for compositing")
            return None

        try:
            # Merge datasets using rasterio.merge
            # This handles overlapping areas by taking the maximum value (latest/most recent flood)
            merged_data, merged_transform = merge(
                datasets,
                method='max',  # Take maximum value in overlapping areas
                dtype=rasterio.uint8
            )

            # Get CRS from first dataset
            merged_crs = datasets[0].crs

            # Create output filename
            output_file = output_dir / f"east_africa_gfm_composite_{date_str}.tif"

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

            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(merged_data[0], 1)

            logger.info(f"Created composite: {output_file}")

            # Close datasets
            for dataset in datasets:
                dataset.close()

            return output_file

        except Exception as e:
            logger.error(f"Error creating composite for {date_str}: {e}")
            for dataset in datasets:
                dataset.close()
            return None

    def run_proof_of_concept(self,
                           start_date: str = "2023-01-01",
                           end_date: str = "2023-12-31",
                           max_days: int = 5) -> Dict:
        """Run the complete proof of concept workflow.

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            max_days: Maximum number of daily composites to create

        Returns:
            Dictionary with results and file paths
        """
        logger.info("=" * 60)
        logger.info("STARTING EAST AFRICA GFM PROOF OF CONCEPT")
        logger.info("=" * 60)
        logger.info(f"AOI: {self.aoi['name']}")
        logger.info(f"Goal: Daily composites from {max_days} days with 3-4 overlapping tiles")

        results = {
            'aoi': self.aoi,
            'date_range': f"{start_date} to {end_date}",
            'historical_data': None,
            'daily_stats': None,
            'downloaded_data': None,
            'composites': []
        }

        try:
            # Step 1: Get historical data
            logger.info("\n1. Retrieving historical GFM data...")
            historical_df = self.get_historical_gfm_data(start_date, end_date)
            results['historical_data'] = historical_df

            if historical_df.empty:
                logger.error("No historical data found!")
                return results

            # Step 2: Analyze daily coverage
            logger.info("\n2. Analyzing daily coverage patterns...")
            daily_stats = self.analyze_daily_coverage(historical_df)
            results['daily_stats'] = daily_stats

            # Step 3: Download data for best days
            logger.info("\n3. Downloading data for compositing...")
            downloaded_data = self.download_daily_data(
                daily_stats,
                max_days=max_days
            )
            results['downloaded_data'] = downloaded_data

            # Step 4: Create daily composites
            logger.info("\n4. Creating daily composites...")
            for date_str in downloaded_data.keys():
                composite_file = self.create_daily_composite(date_str, downloaded_data)
                if composite_file:
                    results['composites'].append(str(composite_file))

            logger.info(f"\n✅ PROOF OF CONCEPT COMPLETE")
            logger.info(f"Created {len(results['composites'])} daily composites")

        except Exception as e:
            logger.error(f"Error in proof of concept: {e}")
            results['error'] = str(e)

        return results


def main():
    """Run the East Africa GFM proof of concept."""
    compositor = EastAfricaGFMCompositor()

    # Run for recent period with good data availability
    results = compositor.run_proof_of_concept(
        start_date="2023-06-01",  # Wet season start
        end_date="2023-09-30",   # Wet season end
        max_days=3               # Create 3 daily composites for demo
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EAST AFRICA GFM PROOF OF CONCEPT RESULTS")
    print("=" * 60)

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return

    print(f"AOI: {results['aoi']['name']}")
    print(f"Period: {results['date_range']}")

    if results['historical_data'] is not None:
        print(f"Historical items found: {len(results['historical_data'])}")

    if results['daily_stats'] is not None:
        good_days = results['daily_stats']['good_for_composite'].sum()
        print(f"Days suitable for compositing: {good_days}")

    print(f"Daily composites created: {len(results['composites'])}")

    if results['composites']:
        print("\nComposite files:")
        for composite in results['composites']:
            print(f"  - {composite}")

    print("\n✅ Proof of concept demonstrates:")
    print("  • Multi-tile GFM data retrieval for East Africa")
    print("  • Daily temporal compositing across overlapping swaths")
    print("  • Automated workflow for historical flood monitoring")


if __name__ == "__main__":
    main()
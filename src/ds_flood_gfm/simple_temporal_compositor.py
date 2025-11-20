"""
Simple Temporal GFM Compositor

Creates temporal composites using a two-step approach:
1. Create daily composites using rasterio.merge
2. Build temporal composites where newer days overwrite older pixels
3. Track observation dates for each pixel
"""

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.merge import merge
import pystac
from pystac import Catalog, Collection, Item, Asset, Extent, SpatialExtent, TemporalExtent
from shapely.geometry import box, mapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTemporalCompositor:
    """Simple temporal compositor using daily merge approach."""

    def __init__(self, reference_date: date = None):
        """Initialize compositor."""
        self.reference_date = reference_date or date(2023, 1, 1)
        logger.info(f"Reference date: {self.reference_date}")

    def load_and_group_stac_items(self, metadata_dir: Path) -> Dict[date, List[Dict]]:
        """Load STAC items and group by date."""
        metadata_files = list(metadata_dir.glob("*_metadata.json"))
        logger.info(f"Found {len(metadata_files)} STAC metadata files")

        daily_groups = defaultdict(list)

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    stac_item = json.load(f)

                # Extract datetime
                datetime_str = stac_item['properties']['datetime']
                if datetime_str.endswith('Z'):
                    datetime_str = datetime_str[:-1] + '+00:00'
                item_datetime = datetime.fromisoformat(datetime_str)
                item_date = item_datetime.date()

                stac_item['_datetime'] = item_datetime
                stac_item['_metadata_file'] = str(metadata_file)
                daily_groups[item_date].append(stac_item)

            except Exception as e:
                logger.warning(f"Failed to load {metadata_file}: {e}")

        logger.info(f"Grouped into {len(daily_groups)} distinct dates")
        for obs_date, items in sorted(daily_groups.items()):
            logger.info(f"  {obs_date}: {len(items)} tiles")

        return dict(daily_groups)

    def create_daily_composite(self, obs_date: date, stac_items: List[Dict],
                             base_dir: Path, daily_output_dir: Path) -> Optional[Dict]:
        """Create daily composite for a single date."""

        # Create output directory and file path
        daily_output_dir.mkdir(parents=True, exist_ok=True)
        date_str = obs_date.strftime('%Y-%m-%d')
        daily_path = daily_output_dir / f"daily_{date_str}.tif"

        # Check if daily composite already exists
        if daily_path.exists():
            logger.info(f"Daily composite already exists, skipping: {daily_path.name}")

            # Read existing file stats for return info
            try:
                with rasterio.open(daily_path) as src:
                    data = src.read(1)
                    valid_pixels = np.sum(data != 255)
                    flood_pixels = np.sum(data == 1)

                    composite_info = {
                        'date': date_str,
                        'daily_path': str(daily_path),
                        'tiles_count': len(stac_items),  # Original tile count
                        'valid_pixels': int(valid_pixels),
                        'flood_pixels': int(flood_pixels),
                        'shape': data.shape,
                        'transform': src.transform,
                        'crs': src.crs,
                        'bounds': src.bounds
                    }

                    logger.info(f"  Valid pixels: {valid_pixels:,}, Flood pixels: {flood_pixels:,}")
                    return composite_info
            except Exception as e:
                logger.warning(f"Error reading existing file {daily_path}, will recreate: {e}")

        logger.info(f"Creating daily composite for {obs_date} ({len(stac_items)} tiles)")

        # Get GeoTIFF paths
        geotiff_paths = []
        for stac_item in stac_items:
            item_id = stac_item['id']
            geotiff_path = base_dir / f"{item_id}.tif"
            if geotiff_path.exists():
                geotiff_paths.append(geotiff_path)

        if not geotiff_paths:
            logger.warning(f"No GeoTIFF files found for {obs_date}")
            return None

        logger.info(f"Merging {len(geotiff_paths)} tiles")

        # Open datasets
        datasets = []
        try:
            for path in geotiff_paths:
                ds = rasterio.open(path)
                datasets.append(ds)

            # Merge using max (flood=1 takes priority over no-flood=0)
            merged_data, merged_transform = merge(datasets, method='max')

            # Output path already defined above

            # Save simple single-band daily composite
            profile = datasets[0].profile.copy()
            profile.update({
                'height': merged_data.shape[1],
                'width': merged_data.shape[2],
                'transform': merged_transform,
                'compress': 'lzw'
            })

            with rasterio.open(daily_path, 'w', **profile) as dst:
                dst.write(merged_data[0], 1)

            # Calculate statistics
            valid_pixels = np.sum(merged_data[0] != 255)
            flood_pixels = np.sum(merged_data[0] == 1)

            composite_info = {
                'date': date_str,
                'daily_path': str(daily_path),
                'tiles_count': len(geotiff_paths),
                'valid_pixels': int(valid_pixels),
                'flood_pixels': int(flood_pixels),
                'shape': merged_data.shape[1:],
                'transform': merged_transform,
                'crs': datasets[0].crs,
                'bounds': datasets[0].bounds
            }

            logger.info(f"Saved: {daily_path.name}")
            logger.info(f"  Valid pixels: {valid_pixels:,}, Flood pixels: {flood_pixels:,}")

            return composite_info

        finally:
            for ds in datasets:
                try:
                    ds.close()
                except:
                    pass

    def create_temporal_composites(self, daily_composites: List[Dict],
                                 output_dir: Path) -> List[Dict]:
        """Create cumulative temporal composites."""
        logger.info("Creating cumulative temporal composites")

        # Sort by date
        daily_composites.sort(key=lambda x: x['date'])

        temporal_composites = []
        cumulative_flood = None
        cumulative_dates = None

        for i, daily_comp in enumerate(daily_composites):
            obs_date = daily_comp['date']
            logger.info(f"Processing temporal composite {i+1}: {obs_date}")

            # Load daily composite (single-band flood extent)
            with rasterio.open(daily_comp['daily_path']) as src:
                daily_flood = src.read(1)
                profile = src.profile.copy()

            # Create DOY data for this date
            obs_date_obj = datetime.strptime(obs_date, '%Y-%m-%d').date()
            doy = obs_date_obj.timetuple().tm_yday
            daily_doy = np.full(daily_flood.shape, doy, dtype=np.uint8)
            # Set nodata areas to 0 (no observation)
            daily_doy[daily_flood == 255] = 0

            if cumulative_flood is None:
                # First day - initialize
                cumulative_flood = daily_flood.copy()
                cumulative_doy = daily_doy.copy()
            else:
                # Update cumulative composite
                # Where daily has valid data, update cumulative
                valid_daily = (daily_flood != 255)
                cumulative_flood[valid_daily] = daily_flood[valid_daily]
                cumulative_doy[valid_daily] = daily_doy[valid_daily]

            # Save temporal composite as multi-band file only
            temporal_multiband_path = output_dir / f"temporal_{obs_date}.tif"

            # Save multi-band file (Band 1=flood, Band 2=DOY)
            profile_multi = profile.copy()
            profile_multi.update({
                'count': 2,
                'dtype': rasterio.uint8,
                'nodata': None
            })

            with rasterio.open(temporal_multiband_path, 'w', **profile_multi) as dst:
                dst.write(cumulative_flood.astype(np.uint8), 1)
                dst.write(cumulative_doy.astype(np.uint8), 2)
                dst.set_band_description(1, 'Flood Extent (0=no flood, 1=flood, 255=nodata)')
                dst.set_band_description(2, 'Day of Year (1-365, 0=no observation)')

            # Calculate stats
            valid_pixels = np.sum(cumulative_flood != 255)
            flood_pixels = np.sum(cumulative_flood == 1)

            temporal_info = {
                'date': obs_date,
                'composite_path': str(temporal_multiband_path),
                'cumulative_tiles': sum(dc['tiles_count'] for dc in daily_composites[:i+1]),
                'valid_pixels': int(valid_pixels),
                'flood_pixels': int(flood_pixels),
                'shape': cumulative_flood.shape,
                'transform': profile['transform'],
                'crs': profile['crs'],
                'bounds': daily_comp['bounds']
            }

            temporal_composites.append(temporal_info)

            logger.info(f"Temporal composite {obs_date}: {flood_pixels:,} flood pixels")

        return temporal_composites

    def create_stac_catalog(self, temporal_composites: List[Dict],
                          output_dir: Path) -> Path:
        """Create STAC catalog."""
        logger.info("Creating STAC catalog")

        # Create catalog
        catalog = Catalog(
            id='somalia-temporal-composites',
            description='Somalia GFM temporal composites with provenance tracking',
            title='Somalia GFM Temporal Composites'
        )

        # Create collection
        if temporal_composites:
            min_date = datetime.fromisoformat(temporal_composites[0]['date'])
            max_date = datetime.fromisoformat(temporal_composites[-1]['date'])
            bounds = temporal_composites[0]['bounds']

            spatial_extent = SpatialExtent(bboxes=[list(bounds)])
            temporal_extent = TemporalExtent(intervals=[[min_date, max_date]])
            extent = Extent(spatial=spatial_extent, temporal=temporal_extent)

            collection = Collection(
                id='temporal-composites',
                description='Cumulative temporal composites with provenance tracking',
                extent=extent,
                title='Somalia Temporal Composites'
            )

            catalog.add_child(collection)

            # Add items
            for temp_comp in temporal_composites:
                item = self._create_stac_item(temp_comp)
                collection.add_item(item)

        # Save catalog
        catalog_path = output_dir / 'catalog.json'
        catalog.normalize_hrefs(str(output_dir))
        catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

        logger.info(f"STAC catalog saved: {catalog_path}")
        return catalog_path

    def _create_stac_item(self, temp_comp: Dict) -> Item:
        """Create STAC item."""
        date_str = temp_comp['date']
        item_datetime = datetime.fromisoformat(date_str)
        bounds = temp_comp['bounds']
        geometry = mapping(box(*bounds))

        item = Item(
            id=f"somalia-temporal-{date_str}",
            geometry=geometry,
            bbox=list(bounds),
            datetime=item_datetime,
            properties={
                'title': f'Somalia Temporal Composite {date_str}',
                'gfm:cumulative_tiles': temp_comp['cumulative_tiles'],
                'gfm:valid_pixels': temp_comp['valid_pixels'],
                'gfm:flood_pixels': temp_comp['flood_pixels'],
                'created': datetime.now().isoformat() + 'Z'
            }
        )

        # Add assets
        composite_asset = Asset(
            href=Path(temp_comp['composite_path']).name,
            media_type='image/tiff; application=geotiff',
            title='Cumulative Flood Composite',
            description='Multi-band: Band 1=Flood Extent, Band 2=Day of Year',
            roles=['data']
        )

        item.add_asset('composite', composite_asset)

        return item

    def process(self, input_dir: Path, output_dir: Path,
               max_dates: Optional[int] = None) -> Dict:
        """Complete processing workflow."""
        logger.info("=" * 60)
        logger.info("SIMPLE TEMPORAL COMPOSITING WORKFLOW")
        logger.info("=" * 60)

        output_dir.mkdir(parents=True, exist_ok=True)
        daily_output_dir = output_dir.parent / "daily_composites"

        # Step 1: Load and group by date
        daily_groups = self.load_and_group_stac_items(input_dir)

        # Limit dates if requested
        if max_dates:
            sorted_dates = sorted(daily_groups.keys())[:max_dates]
            daily_groups = {d: daily_groups[d] for d in sorted_dates}
            logger.info(f"Limited to first {max_dates} dates")

        # Step 2: Create daily composites
        logger.info("\nStep 1: Creating daily composites")
        daily_composites = []
        for obs_date, stac_items in sorted(daily_groups.items()):
            daily_comp = self.create_daily_composite(obs_date, stac_items, input_dir, daily_output_dir)
            if daily_comp:
                daily_composites.append(daily_comp)

        # Step 3: Create temporal composites
        logger.info("\nStep 2: Creating temporal composites")
        temporal_composites = self.create_temporal_composites(daily_composites, output_dir)

        # Step 4: Create STAC catalog
        logger.info("\nStep 3: Creating STAC catalog")
        catalog_path = self.create_stac_catalog(temporal_composites, output_dir)

        results = {
            'daily_composites': len(daily_composites),
            'temporal_composites': len(temporal_composites),
            'catalog_path': str(catalog_path),
            'output_dir': str(output_dir)
        }

        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Daily composites: {len(daily_composites)}")
        logger.info(f"Temporal composites: {len(temporal_composites)}")
        logger.info(f"STAC catalog: {catalog_path}")

        return results


def main():
    """Test the simple temporal compositor."""
    compositor = SimpleTemporalCompositor()

    input_dir = Path("data/gfm/somalia_example/raw_gfm")
    output_dir = Path("data/gfm/somalia_example/temporal_composites")

    # Process 5 dates for testing
    results = compositor.process(input_dir, output_dir, max_dates=5)

    print("Results:", results)


if __name__ == "__main__":
    main()
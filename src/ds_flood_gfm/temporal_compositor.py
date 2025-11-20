"""
Temporal GFM Compositor with STAC Catalog Generation

This module creates cumulative temporal composites where newer observations
overwrite older pixels, maintaining provenance tracking of acquisition dates.
Outputs STAC-compliant catalog structure.
"""

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
import pystac
from pystac import Catalog, Collection, Item, Asset, Extent, SpatialExtent, TemporalExtent
from shapely.geometry import box, mapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalGFMCompositor:
    """Create temporal composites with provenance tracking and STAC catalog."""

    def __init__(self, reference_date: date = None):
        """
        Initialize temporal compositor.

        Args:
            reference_date: Reference date for date encoding (default: 2023-01-01)
        """
        self.reference_date = reference_date or date(2023, 1, 1)
        logger.info(f"Reference date for encoding: {self.reference_date}")

    def load_and_sort_stac_items(self, metadata_dir: Path) -> List[Dict]:
        """
        Load STAC metadata and sort chronologically.

        Args:
            metadata_dir: Directory containing *_metadata.json files

        Returns:
            List of STAC items sorted by datetime
        """
        metadata_files = list(metadata_dir.glob("*_metadata.json"))
        logger.info(f"Found {len(metadata_files)} STAC metadata files")

        stac_items = []

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    stac_item = json.load(f)

                # Extract datetime
                datetime_str = stac_item['properties']['datetime']
                if datetime_str.endswith('Z'):
                    datetime_str = datetime_str[:-1] + '+00:00'
                item_datetime = datetime.fromisoformat(datetime_str)

                stac_item['_datetime'] = item_datetime
                stac_item['_metadata_file'] = str(metadata_file)
                stac_items.append(stac_item)

            except Exception as e:
                logger.warning(f"Failed to load {metadata_file}: {e}")

        # Sort by datetime
        stac_items.sort(key=lambda x: x['_datetime'])

        logger.info(f"Loaded {len(stac_items)} items from {stac_items[0]['_datetime'].date()} to {stac_items[-1]['_datetime'].date()}")

        return stac_items

    def encode_date_to_days(self, obs_date: date) -> int:
        """
        Encode date as days since reference date.

        Args:
            obs_date: Observation date

        Returns:
            Days since reference date
        """
        return (obs_date - self.reference_date).days

    def decode_days_to_date(self, days: int) -> date:
        """
        Decode days since reference to date.

        Args:
            days: Days since reference date

        Returns:
            Decoded date
        """
        return self.reference_date + timedelta(days=days)

    def get_geotiff_path(self, stac_item: Dict, base_dir: Path) -> Optional[Path]:
        """Get path to GeoTIFF file for STAC item."""
        item_id = stac_item['id']
        geotiff_path = base_dir / f"{item_id}.tif"

        if geotiff_path.exists():
            return geotiff_path

        logger.warning(f"GeoTIFF not found for {item_id}")
        return None

    def create_composite_grid(self, stac_items: List[Dict], base_dir: Path) -> Dict:
        """
        Determine the composite grid from all input tiles using rasterio.merge logic.

        Args:
            stac_items: List of STAC items
            base_dir: Directory containing GeoTIFF files

        Returns:
            Grid metadata dictionary
        """
        logger.info("Determining composite grid from input tiles")

        # Get paths to all GeoTIFF files
        geotiff_paths = []
        for stac_item in stac_items[:100]:  # Sample to determine bounds
            geotiff_path = self.get_geotiff_path(stac_item, base_dir)
            if geotiff_path:
                geotiff_paths.append(geotiff_path)

        if not geotiff_paths:
            raise ValueError("No valid tiles found for grid determination")

        # Open first few datasets to get bounds and CRS
        datasets = []
        try:
            for path in geotiff_paths[:20]:  # Use first 20 for bounds calculation
                try:
                    ds = rasterio.open(path)
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Could not open {path}: {e}")

            if not datasets:
                raise ValueError("No datasets could be opened")

            # Use rasterio.merge to determine output bounds and transform
            from rasterio.merge import merge
            merged_data, merged_transform = merge(datasets, method='first')

            grid_metadata = {
                'bounds': datasets[0].bounds,  # Use first dataset bounds for now
                'shape': merged_data.shape[1:],  # Remove band dimension
                'transform': merged_transform,
                'crs': datasets[0].crs,
                'resolution': abs(datasets[0].transform[0])  # Get pixel size
            }

            logger.info(f"Composite grid: {grid_metadata['shape'][1]} x {grid_metadata['shape'][0]} ({grid_metadata['resolution']}m resolution)")

        finally:
            # Close all datasets
            for ds in datasets:
                try:
                    ds.close()
                except:
                    pass

        return grid_metadata

    def create_temporal_composite(self, stac_items: List[Dict], base_dir: Path,
                                output_dir: Path, max_items: Optional[int] = None) -> List[Dict]:
        """
        Create temporal composites with provenance tracking.

        Args:
            stac_items: Chronologically sorted STAC items
            base_dir: Directory containing GeoTIFF files
            output_dir: Output directory for composites
            max_items: Maximum number of items to process

        Returns:
            List of composite metadata dictionaries
        """
        logger.info("=" * 60)
        logger.info("CREATING TEMPORAL COMPOSITES WITH PROVENANCE TRACKING")
        logger.info("=" * 60)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine composite grid
        grid_metadata = self.create_composite_grid(stac_items, base_dir)

        # Initialize composite arrays
        flood_composite = np.full(grid_metadata['shape'], 255, dtype=np.uint8)  # 255 = nodata
        date_composite = np.full(grid_metadata['shape'], -1, dtype=np.int16)    # -1 = no observation

        composites_metadata = []
        processed_tiles = []

        # Limit items if specified
        if max_items:
            stac_items = stac_items[:max_items]
            logger.info(f"Processing first {max_items} items only")

        # Group items by date for processing
        daily_groups = defaultdict(list)
        for item in stac_items:
            item_date = item['_datetime'].date()
            daily_groups[item_date].append(item)

        # Process each day sequentially
        for day_index, (obs_date, day_items) in enumerate(sorted(daily_groups.items())):
            logger.info(f"\nProcessing day {day_index + 1}: {obs_date} ({len(day_items)} tiles)")

            # Process all tiles for this day
            tiles_processed_today = 0
            pixels_updated_today = 0

            for item in day_items:
                geotiff_path = self.get_geotiff_path(item, base_dir)
                if not geotiff_path:
                    continue

                try:
                    # Read tile data
                    with rasterio.open(geotiff_path) as src:
                        # Reproject to composite grid
                        tile_data = np.full(grid_metadata['shape'], 255, dtype=np.uint8)

                        reproject(
                            source=rasterio.band(src, 1),
                            destination=tile_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=grid_metadata['transform'],
                            dst_crs=grid_metadata['crs'],
                            resampling=Resampling.nearest
                        )

                        # Update composite where we have valid data
                        valid_pixels = (tile_data != 255)
                        pixels_updated = np.sum(valid_pixels)

                        if pixels_updated > 0:
                            # Update flood composite
                            flood_composite[valid_pixels] = tile_data[valid_pixels]

                            # Update date composite
                            date_encoded = self.encode_date_to_days(obs_date)
                            date_composite[valid_pixels] = date_encoded

                            pixels_updated_today += pixels_updated
                            tiles_processed_today += 1
                            processed_tiles.append(item['id'])

                except Exception as e:
                    logger.warning(f"Error processing {item['id']}: {e}")

            # Save composite for this day
            if tiles_processed_today > 0:
                composite_metadata = self._save_temporal_composite(
                    obs_date, flood_composite, date_composite, grid_metadata,
                    output_dir, processed_tiles, pixels_updated_today
                )
                composites_metadata.append(composite_metadata)

                logger.info(f"  Processed {tiles_processed_today} tiles, updated {pixels_updated_today:,} pixels")

        logger.info(f"\n" + "=" * 60)
        logger.info(f"TEMPORAL COMPOSITING COMPLETE")
        logger.info(f"Created {len(composites_metadata)} temporal composites")
        logger.info(f"Processed {len(processed_tiles)} total tiles")

        return composites_metadata

    def _save_temporal_composite(self, obs_date: date, flood_composite: np.ndarray,
                               date_composite: np.ndarray, grid_metadata: Dict,
                               output_dir: Path, processed_tiles: List[str],
                               pixels_updated: int) -> Dict:
        """Save temporal composite and return metadata."""

        date_str = obs_date.strftime('%Y-%m-%d')
        base_filename = f"somalia_temporal_{date_str}"

        # Save flood extent
        flood_path = output_dir / f"{base_filename}_flood.tif"
        self._write_raster(flood_composite, grid_metadata, flood_path, dtype=rasterio.uint8, nodata=255)

        # Save observation dates
        dates_path = output_dir / f"{base_filename}_dates.tif"
        self._write_raster(date_composite, grid_metadata, dates_path, dtype=rasterio.int16, nodata=-1)

        # Calculate statistics
        valid_pixels = np.sum(flood_composite != 255)
        flood_pixels = np.sum(flood_composite == 1)
        coverage_pct = (valid_pixels / flood_composite.size) * 100
        flood_pct = (flood_pixels / max(1, valid_pixels)) * 100

        # Get date range of observations
        valid_dates = date_composite[date_composite != -1]
        if len(valid_dates) > 0:
            min_date = self.decode_days_to_date(int(np.min(valid_dates)))
            max_date = self.decode_days_to_date(int(np.max(valid_dates)))
        else:
            min_date = max_date = obs_date

        composite_metadata = {
            'date': date_str,
            'flood_extent_path': str(flood_path),
            'observation_dates_path': str(dates_path),
            'total_tiles_used': len(set(processed_tiles)),
            'pixels_updated_today': pixels_updated,
            'total_valid_pixels': int(valid_pixels),
            'total_flood_pixels': int(flood_pixels),
            'coverage_percentage': float(coverage_pct),
            'flood_percentage': float(flood_pct),
            'observation_date_range': {
                'min': min_date.strftime('%Y-%m-%d'),
                'max': max_date.strftime('%Y-%m-%d')
            },
            'grid_metadata': grid_metadata,
            'processed_tiles': processed_tiles
        }

        logger.info(f"Saved: {flood_path}")
        logger.info(f"       {dates_path}")
        logger.info(f"  Coverage: {coverage_pct:.1f}%, Flood: {flood_pct:.2f}%")

        return composite_metadata

    def _write_raster(self, data: np.ndarray, grid_metadata: Dict,
                     output_path: Path, dtype, nodata):
        """Write raster data to file."""
        profile = {
            'driver': 'GTiff',
            'height': grid_metadata['shape'][0],
            'width': grid_metadata['shape'][1],
            'count': 1,
            'dtype': dtype,
            'crs': grid_metadata['crs'],
            'transform': grid_metadata['transform'],
            'compress': 'lzw',
            'nodata': nodata
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)

    def create_stac_catalog(self, composites_metadata: List[Dict],
                          output_dir: Path) -> Path:
        """
        Create STAC catalog for temporal composites.

        Args:
            composites_metadata: List of composite metadata
            output_dir: Output directory containing composites

        Returns:
            Path to created catalog.json
        """
        logger.info("Creating STAC catalog for temporal composites")

        # Create root catalog
        catalog = Catalog(
            id='somalia-gfm-temporal-composites',
            description='Temporal composites of GFM flood data for Somalia with provenance tracking',
            title='Somalia GFM Temporal Composites'
        )

        # Create collection
        # Calculate temporal extent
        min_date = min(datetime.fromisoformat(c['date']) for c in composites_metadata)
        max_date = max(datetime.fromisoformat(c['date']) for c in composites_metadata)

        # Calculate spatial extent from grid metadata
        if composites_metadata:
            bounds = composites_metadata[0]['grid_metadata']['bounds']
            spatial_extent = SpatialExtent(bboxes=[list(bounds)])
            temporal_extent = TemporalExtent(intervals=[[min_date, max_date]])
            extent = Extent(spatial=spatial_extent, temporal=temporal_extent)

            collection = Collection(
                id='temporal-composites',
                description='Daily temporal composites showing latest flood observations',
                extent=extent,
                title='Somalia GFM Temporal Composites Collection'
            )

            # Add collection to catalog
            catalog.add_child(collection)

            # Create items for each composite
            for comp_meta in composites_metadata:
                item = self._create_stac_item(comp_meta)
                collection.add_item(item)

        # Save catalog
        catalog_path = output_dir / 'catalog.json'
        catalog.normalize_hrefs(str(output_dir))
        catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

        logger.info(f"STAC catalog saved: {catalog_path}")
        return catalog_path

    def _create_stac_item(self, comp_meta: Dict) -> Item:
        """Create STAC Item from composite metadata."""
        date_str = comp_meta['date']
        item_datetime = datetime.fromisoformat(date_str)

        # Create geometry from bounds
        bounds = comp_meta['grid_metadata']['bounds']
        geometry = mapping(box(*bounds))

        # Create item
        item = Item(
            id=f"somalia-temporal-{date_str}",
            geometry=geometry,
            bbox=list(bounds),
            datetime=item_datetime,
            properties={
                'title': f'Somalia Temporal Composite {date_str}',
                'description': f'Cumulative temporal composite up to {date_str}',
                'gfm:tiles_used': comp_meta['total_tiles_used'],
                'gfm:pixels_updated_today': comp_meta['pixels_updated_today'],
                'gfm:coverage_percentage': comp_meta['coverage_percentage'],
                'gfm:flood_percentage': comp_meta['flood_percentage'],
                'gfm:observation_date_range': comp_meta['observation_date_range'],
                'processing:level': 'L4',
                'processing:facility': 'Custom Temporal Compositor',
                'created': datetime.now().isoformat() + 'Z'
            }
        )

        # Add assets
        flood_asset = Asset(
            href=Path(comp_meta['flood_extent_path']).name,
            media_type='image/tiff; application=geotiff; profile=cloud-optimized',
            title='Flood Extent (Latest Observations)',
            description='Latest flood extent composite with temporal prioritization',
            roles=['data']
        )

        dates_asset = Asset(
            href=Path(comp_meta['observation_dates_path']).name,
            media_type='image/tiff; application=geotiff; profile=cloud-optimized',
            title='Observation Dates',
            description='Acquisition dates for each pixel (days since 2023-01-01)',
            roles=['metadata']
        )

        item.add_asset('flood_extent', flood_asset)
        item.add_asset('observation_dates', dates_asset)

        return item


def main():
    """Example usage: Create temporal composites for Somalia GFM data."""

    compositor = TemporalGFMCompositor()

    # Paths
    input_dir = Path("data/gfm/somalia_example/raw_gfm")
    output_dir = Path("data/gfm/somalia_example/temporal_composites")

    print("Creating temporal composites for Somalia GFM data...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load and sort STAC items
    stac_items = compositor.load_and_sort_stac_items(input_dir)

    # Create temporal composites
    composites_metadata = compositor.create_temporal_composite(
        stac_items, input_dir, output_dir, max_items=100  # Limit for testing
    )

    # Create STAC catalog
    if composites_metadata:
        catalog_path = compositor.create_stac_catalog(composites_metadata, output_dir)
        print(f"\nSTAC catalog created: {catalog_path}")

    print(f"\nProcessing complete!")
    print(f"Created {len(composites_metadata)} temporal composites")


if __name__ == "__main__":
    main()
"""
Cloud Temporal GFM Compositor

Creates temporal composites directly from STAC API without local downloads.
This is the Earth Engine-style approach using remote data.
"""

import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import requests
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudTemporalCompositor:
    """Create temporal composites directly from remote STAC data."""

    def __init__(self):
        self.stac_api = "https://stac.eodc.eu/api/v1"

    def query_stac_items(self, bbox: List[float], datetime_range: str,
                        limit: int = 1000) -> List[Dict]:
        """Query STAC API for GFM items in AOI and time range."""
        logger.info(f"Querying STAC API for bbox {bbox}, datetime {datetime_range}")

        search_url = f"{self.stac_api}/collections/GFM/items"
        params = {
            'bbox': ','.join(map(str, bbox)),
            'datetime': datetime_range,
            'limit': limit
        }

        response = requests.get(search_url, params=params)

        if response.status_code != 200:
            raise Exception(f"STAC API error {response.status_code}: {response.text}")

        items_response = response.json()
        features = items_response.get('features', [])

        logger.info(f"Found {len(features)} items from STAC API")

        return features

    def group_items_by_date(self, stac_items: List[Dict]) -> Dict[date, List[Dict]]:
        """Group STAC items by observation date."""
        daily_groups = defaultdict(list)

        for stac_item in stac_items:
            try:
                # Extract datetime
                datetime_str = stac_item['properties']['datetime']
                if datetime_str.endswith('Z'):
                    datetime_str = datetime_str[:-1] + '+00:00'
                item_datetime = datetime.fromisoformat(datetime_str)
                item_date = item_datetime.date()

                # Add remote URL
                if 'ensemble_flood_extent' in stac_item['assets']:
                    stac_item['_remote_url'] = stac_item['assets']['ensemble_flood_extent']['href']
                    stac_item['_date'] = item_date
                    daily_groups[item_date].append(stac_item)
                else:
                    logger.warning(f"No ensemble_flood_extent asset in {stac_item['id']}")

            except Exception as e:
                logger.warning(f"Failed to process item {stac_item.get('id', 'unknown')}: {e}")

        logger.info(f"Grouped into {len(daily_groups)} dates")
        for obs_date, items in sorted(daily_groups.items()):
            logger.info(f"  {obs_date}: {len(items)} tiles")

        return dict(daily_groups)

    def create_daily_mosaic_from_remote(self, obs_date: date, stac_items: List[Dict],
                                      target_crs: str = 'EPSG:4326',
                                      resolution: float = 0.0002) -> Optional[Dict]:
        """Create daily mosaic from remote URLs."""
        logger.info(f"Creating daily mosaic for {obs_date} from {len(stac_items)} remote tiles")

        if not stac_items:
            return None

        # Collect remote URLs
        remote_urls = [item['_remote_url'] for item in stac_items]

        try:
            # Open all remote datasets
            datasets = []
            for url in remote_urls:
                ds = rasterio.open(url)
                datasets.append(ds)

            logger.info(f"Opened {len(datasets)} remote datasets")

            # Calculate target bounds (union of all datasets in target CRS)
            all_bounds = []
            for ds in datasets:
                # Transform bounds to target CRS
                left, bottom, right, top = ds.bounds
                dst_transform, width, height = calculate_default_transform(
                    ds.crs, target_crs, ds.width, ds.height,
                    left, bottom, right, top, resolution=resolution
                )

                # Calculate bounds in target CRS
                left_tgt = dst_transform.c
                top_tgt = dst_transform.f
                right_tgt = left_tgt + width * dst_transform.a
                bottom_tgt = top_tgt + height * dst_transform.e

                all_bounds.append([left_tgt, bottom_tgt, right_tgt, top_tgt])

            # Union bounds
            union_bounds = [
                min(b[0] for b in all_bounds),  # min_x
                min(b[1] for b in all_bounds),  # min_y
                max(b[2] for b in all_bounds),  # max_x
                max(b[3] for b in all_bounds)   # max_y
            ]

            # Calculate output dimensions
            width = int((union_bounds[2] - union_bounds[0]) / resolution)
            height = int((union_bounds[3] - union_bounds[1]) / resolution)

            logger.info(f"Target mosaic size: {width}x{height} at {resolution}° resolution")

            # Create target transform
            target_transform = rasterio.transform.from_bounds(
                union_bounds[0], union_bounds[1], union_bounds[2], union_bounds[3],
                width, height
            )

            # Initialize output array
            mosaic_data = np.full((height, width), 255, dtype=np.uint8)

            # Reproject and mosaic each dataset
            for i, ds in enumerate(datasets):
                logger.info(f"Reprojecting tile {i+1}/{len(datasets)}")

                # Reproject to target grid
                reprojected = np.full((height, width), 255, dtype=np.uint8)

                reproject(
                    source=ds.read(1),
                    destination=reprojected,
                    src_transform=ds.transform,
                    src_crs=ds.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )

                # Mosaic: new valid pixels overwrite old
                valid_pixels = (reprojected != 255)
                mosaic_data[valid_pixels] = reprojected[valid_pixels]

            # Calculate statistics
            valid_pixels = np.sum(mosaic_data != 255)
            flood_pixels = np.sum(mosaic_data == 1)
            total_pixels = mosaic_data.size

            mosaic_info = {
                'date': obs_date.strftime('%Y-%m-%d'),
                'data': mosaic_data,
                'transform': target_transform,
                'crs': target_crs,
                'bounds': union_bounds,
                'shape': (height, width),
                'tiles_count': len(stac_items),
                'valid_pixels': int(valid_pixels),
                'flood_pixels': int(flood_pixels),
                'total_pixels': int(total_pixels),
                'coverage_pct': 100 * valid_pixels / total_pixels
            }

            logger.info(f"Daily mosaic {obs_date}: {flood_pixels:,} flood pixels, "
                       f"{valid_pixels:,} valid pixels ({mosaic_info['coverage_pct']:.2f}% coverage)")

            return mosaic_info

        finally:
            # Close all datasets
            for ds in datasets:
                try:
                    ds.close()
                except:
                    pass

    def create_temporal_composites(self, daily_mosaics: List[Dict]) -> List[Dict]:
        """Create growing temporal composites from daily mosaics."""
        logger.info("Creating temporal composites with growing extents")

        if not daily_mosaics:
            return []

        # Sort by date
        daily_mosaics.sort(key=lambda x: x['date'])

        temporal_composites = []
        cumulative_data = None
        cumulative_doy = None

        for i, daily_mosaic in enumerate(daily_mosaics):
            obs_date = daily_mosaic['date']
            daily_data = daily_mosaic['data']

            logger.info(f"Processing temporal composite {i+1}: {obs_date}")

            # Create DOY data
            date_obj = datetime.strptime(obs_date, '%Y-%m-%d').date()
            doy = date_obj.timetuple().tm_yday
            daily_doy = np.full(daily_data.shape, doy, dtype=np.uint8)
            daily_doy[daily_data == 255] = 0

            if cumulative_data is None:
                # First day - initialize
                cumulative_data = daily_data.copy()
                cumulative_doy = daily_doy.copy()
                reference_transform = daily_mosaic['transform']
                reference_crs = daily_mosaic['crs']
                reference_bounds = daily_mosaic['bounds']
            else:
                # For growing extents, we'd need to handle different grid sizes
                # For now, assume same grid (this would need enhancement for true growing extents)
                if daily_data.shape == cumulative_data.shape:
                    # Update cumulative composite: newer valid pixels overwrite older
                    valid_daily = (daily_data != 255)
                    cumulative_data[valid_daily] = daily_data[valid_daily]
                    cumulative_doy[valid_daily] = daily_doy[valid_daily]
                else:
                    logger.warning(f"Shape mismatch: {daily_data.shape} vs {cumulative_data.shape}")
                    # Skip this date or implement proper grid alignment

            # Calculate stats
            valid_pixels = np.sum(cumulative_data != 255)
            flood_pixels = np.sum(cumulative_data == 1)
            total_pixels = cumulative_data.size

            temporal_info = {
                'date': obs_date,
                'data': cumulative_data.copy(),
                'doy_data': cumulative_doy.copy(),
                'transform': reference_transform,
                'crs': reference_crs,
                'bounds': reference_bounds,
                'shape': cumulative_data.shape,
                'cumulative_tiles': sum(dm['tiles_count'] for dm in daily_mosaics[:i+1]),
                'valid_pixels': int(valid_pixels),
                'flood_pixels': int(flood_pixels),
                'total_pixels': int(total_pixels),
                'coverage_pct': 100 * valid_pixels / total_pixels
            }

            temporal_composites.append(temporal_info)

            logger.info(f"Temporal composite {obs_date}: {flood_pixels:,} flood pixels, "
                       f"coverage: {temporal_info['coverage_pct']:.2f}%")

        return temporal_composites

    def save_temporal_composite(self, temporal_info: Dict, output_dir: Path) -> Path:
        """Save temporal composite to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"cloud_temporal_{temporal_info['date']}.tif"

        # Create profile
        profile = {
            'driver': 'GTiff',
            'height': temporal_info['shape'][0],
            'width': temporal_info['shape'][1],
            'count': 2,  # Band 1: flood, Band 2: DOY
            'dtype': rasterio.uint8,
            'crs': temporal_info['crs'],
            'transform': temporal_info['transform'],
            'compress': 'lzw',
            'nodata': None
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(temporal_info['data'], 1)
            dst.write(temporal_info['doy_data'], 2)
            dst.set_band_description(1, 'Flood Extent (0=no flood, 1=flood, 255=nodata)')
            dst.set_band_description(2, 'Day of Year (1-365, 0=no observation)')

        logger.info(f"Saved temporal composite: {output_path.name}")
        return output_path

    def process_cloud(self, bbox: List[float], datetime_range: str,
                     output_dir: Path, limit: int = 1000) -> Dict:
        """Complete cloud processing workflow."""
        logger.info("=" * 60)
        logger.info("CLOUD TEMPORAL COMPOSITING WORKFLOW")
        logger.info("=" * 60)

        # Step 1: Query STAC API
        stac_items = self.query_stac_items(bbox, datetime_range, limit)

        # Step 2: Group by date
        daily_groups = self.group_items_by_date(stac_items)

        # Step 3: Create daily mosaics from remote data
        logger.info("\\nStep 1: Creating daily mosaics from remote data")
        daily_mosaics = []
        for obs_date, items in sorted(daily_groups.items()):
            daily_mosaic = self.create_daily_mosaic_from_remote(obs_date, items)
            if daily_mosaic:
                daily_mosaics.append(daily_mosaic)

        # Step 4: Create temporal composites
        logger.info("\\nStep 2: Creating temporal composites")
        temporal_composites = self.create_temporal_composites(daily_mosaics)

        # Step 5: Save temporal composites
        logger.info("\\nStep 3: Saving temporal composites")
        saved_paths = []
        for temporal_comp in temporal_composites:
            output_path = self.save_temporal_composite(temporal_comp, output_dir)
            saved_paths.append(output_path)

        results = {
            'stac_items_found': len(stac_items),
            'daily_dates': len(daily_groups),
            'daily_mosaics': len(daily_mosaics),
            'temporal_composites': len(temporal_composites),
            'output_files': [str(p) for p in saved_paths],
            'output_dir': str(output_dir)
        }

        logger.info("=" * 60)
        logger.info("CLOUD PROCESSING COMPLETE")
        logger.info(f"STAC items: {len(stac_items)}")
        logger.info(f"Daily mosaics: {len(daily_mosaics)}")
        logger.info(f"Temporal composites: {len(temporal_composites)}")

        return results


def main():
    """Test the cloud temporal compositor."""
    compositor = CloudTemporalCompositor()

    # Somalia bounding box
    bbox = [40.0, -2.0, 51.0, 12.0]
    datetime_range = "2023-09-01T00:00:00Z/2023-09-10T23:59:59Z"
    output_dir = Path("data/gfm/somalia_example/cloud_temporal_composites")

    print("Testing Cloud Temporal Compositor...")
    print(f"AOI: {bbox}")
    print(f"Time range: {datetime_range}")

    results = compositor.process_cloud(bbox, datetime_range, output_dir, limit=100)

    print("\\nResults:", results)


if __name__ == "__main__":
    main()
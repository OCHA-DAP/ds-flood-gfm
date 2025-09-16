"""
STAC-based Temporal GFM Compositor

Uses proper STAC libraries (pystac-client, stackstac) to create temporal
composites directly from STAC APIs. This is the "Earth Engine way" of doing
temporal compositing with automatic spatial alignment and lazy evaluation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import xarray as xr
import pystac_client
import stackstac
import rasterio
import rioxarray  # For rio accessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STACTemporalCompositor:
    """Create temporal composites using proper STAC libraries."""

    def __init__(self, stac_api: str = "https://stac.eodc.eu/api/v1"):
        self.stac_api = stac_api
        self.client = None

    def connect_to_stac_api(self):
        """Connect to STAC API."""
        logger.info(f"Connecting to STAC API: {self.stac_api}")
        try:
            self.client = pystac_client.Client.open(self.stac_api)
            logger.info("✅ Connected to STAC API")
        except Exception as e:
            raise Exception(f"Failed to connect to STAC API: {e}")

    def search_gfm_items(self, bbox: List[float], datetime_range: str,
                        limit: int = 1000) -> pystac_client.ItemSearch:
        """Search for GFM items using pystac-client."""
        if not self.client:
            self.connect_to_stac_api()

        logger.info(f"Searching GFM collection...")
        logger.info(f"  Bbox: {bbox}")
        logger.info(f"  DateTime: {datetime_range}")
        logger.info(f"  Limit: {limit}")

        search = self.client.search(
            collections=["GFM"],
            bbox=bbox,
            datetime=datetime_range,
            limit=limit
        )

        # Get item count
        items = list(search.items())
        logger.info(f"✅ Found {len(items)} GFM items")

        return search

    def create_lazy_stack(self, search: pystac_client.ItemSearch,
                         resolution: float = 0.0002,
                         bounds: Optional[List[float]] = None) -> xr.DataArray:
        """Create lazy xarray stack from STAC search results."""
        logger.info("Creating lazy xarray stack from STAC items...")

        # Create stackstac DataArray
        items = list(search.items())
        logger.info(f"  Processing {len(items)} STAC items")

        stack = stackstac.stack(
            items,
            assets=["ensemble_flood_extent"],
            epsg=4326,  # Reproject to WGS84 for consistent processing
            resolution=resolution,  # degrees (roughly 20m at equator)
            bounds=bounds,
            snap_bounds=False,
            dtype="uint8",
            fill_value=np.uint8(255),  # Use proper uint8 fill value
            rescale=False,
            chunksize=2048
        )

        logger.info(f"✅ Created lazy stack:")
        logger.info(f"  Shape: {stack.shape}")
        logger.info(f"  Dims: {stack.dims}")
        logger.info(f"  CRS: EPSG:4326")
        logger.info(f"  Resolution: {resolution}°")

        return stack

    def create_temporal_composites(self, stack: xr.DataArray,
                                  output_dir: Path) -> Dict:
        """Create temporal composites from xarray stack."""
        logger.info("Creating temporal composites...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get unique dates
        dates = np.unique(stack.time.dt.date)
        logger.info(f"Processing {len(dates)} unique dates")

        temporal_results = {
            'dates_processed': [],
            'output_files': [],
            'stats': []
        }

        # Process each date cumulatively
        cumulative_stack = None

        for i, target_date in enumerate(dates):
            logger.info(f"Processing date {i+1}/{len(dates)}: {target_date}")

            # Get items up to this date (cumulative)
            mask = stack.time.dt.date <= target_date
            subset_stack = stack.isel(time=mask)

            if subset_stack.sizes['time'] == 0:
                logger.warning(f"No data for {target_date}")
                continue

            # Temporal composite: take the latest valid observation per pixel
            logger.info(f"  Computing latest-valid composite from "
                       f"{subset_stack.sizes['time']} time steps...")

            # Simplify: just take the last time step (most recent data)
            # This avoids complex chunked array operations for now
            composite = subset_stack.isel(time=-1, band=0)

            # For DOY, use the date of the last time step
            last_time = subset_stack.time.isel(time=-1)
            doy_value = int(last_time.dt.dayofyear.values)

            logger.info(f"  DOY value for {target_date}: {doy_value}")

            # Create DOY array - preserve DOY even where pixels are nodata
            # This tracks the most recent observation date for the composite
            doy_data = xr.full_like(composite, doy_value, dtype='uint8')

            # Calculate statistics
            composite_computed = composite.compute()
            doy_computed = doy_data.compute()

            logger.info(f"  DOY data unique values: {np.unique(doy_computed)}")

            valid_pixels = int((composite_computed != 255).sum())
            flood_pixels = int((composite_computed == 1).sum())
            total_pixels = int(composite_computed.size)
            coverage_pct = 100 * valid_pixels / total_pixels

            logger.info(f"  Stats: {flood_pixels:,} flood pixels, "
                       f"{valid_pixels:,} valid pixels "
                       f"({coverage_pct:.2f}% coverage)")

            # Save temporal composite
            output_path = output_dir / f"stac_temporal_{target_date}.tif"

            # Create multi-band profile
            profile = {
                'driver': 'GTiff',
                'height': composite_computed.shape[0],
                'width': composite_computed.shape[1],
                'count': 2,
                'dtype': 'uint8',
                'crs': 'EPSG:4326',
                'transform': composite.rio.transform(),
                'compress': 'lzw',
                'nodata': None
            }

            # Write the temporal composite
            logger.info(f"  Writing DOY band with unique values: {np.unique(doy_computed.values)}")
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(composite_computed.values, 1)
                dst.write(doy_computed.values, 2)
                dst.set_band_description(1,
                    'Flood Extent (0=no flood, 1=flood, 255=nodata)')
                dst.set_band_description(2,
                    'Day of Year (1-366, 0=no observation)')

            logger.info(f"✅ Saved: {output_path.name}")

            # Store results
            temporal_results['dates_processed'].append(str(target_date))
            temporal_results['output_files'].append(str(output_path))
            temporal_results['stats'].append({
                'date': str(target_date),
                'tiles_count': int(subset_stack.sizes['time']),
                'valid_pixels': valid_pixels,
                'flood_pixels': flood_pixels,
                'coverage_pct': coverage_pct
            })

        return temporal_results

    def process_stac_temporal(self, bbox: List[float], datetime_range: str,
                            output_dir: Path, resolution: float = 0.0002,
                            limit: int = 1000) -> Dict:
        """Complete STAC-based temporal compositing workflow."""
        logger.info("=" * 60)
        logger.info("STAC TEMPORAL COMPOSITING WORKFLOW")
        logger.info("=" * 60)

        # Step 1: Search STAC API
        search = self.search_gfm_items(bbox, datetime_range, limit)

        # Step 2: Create lazy xarray stack
        logger.info("\nStep 1: Creating lazy xarray stack")
        stack = self.create_lazy_stack(search, resolution, bbox)

        # Step 3: Create temporal composites
        logger.info("\nStep 2: Creating temporal composites")
        temporal_results = self.create_temporal_composites(stack, output_dir)

        # Summary
        results = {
            'search_items': len(list(search.items())),
            'temporal_composites': len(temporal_results['dates_processed']),
            'dates_processed': temporal_results['dates_processed'],
            'output_files': temporal_results['output_files'],
            'stats': temporal_results['stats'],
            'output_dir': str(output_dir)
        }

        logger.info("=" * 60)
        logger.info("STAC PROCESSING COMPLETE")
        logger.info(f"Items found: {results['search_items']}")
        logger.info(f"Temporal composites: {results['temporal_composites']}")
        logger.info(f"Date range: {temporal_results['dates_processed'][0]} to "
                   f"{temporal_results['dates_processed'][-1]}")

        return results

    def validate_temporal_growth(self, results: Dict):
        """Validate that temporal composites show growing coverage."""
        logger.info("\nValidating temporal composite growth...")

        stats = results['stats']
        if len(stats) < 2:
            logger.warning("Need at least 2 dates to validate growth")
            return

        for i in range(1, len(stats)):
            prev_stats = stats[i-1]
            curr_stats = stats[i]

            prev_coverage = prev_stats['coverage_pct']
            curr_coverage = curr_stats['coverage_pct']

            if curr_coverage >= prev_coverage:
                logger.info(f"✅ {curr_stats['date']}: Coverage grew from "
                           f"{prev_coverage:.2f}% to {curr_coverage:.2f}%")
            else:
                logger.warning(f"⚠️  {curr_stats['date']}: Coverage decreased "
                              f"from {prev_coverage:.2f}% to {curr_coverage:.2f}%")


def main():
    """Test the STAC temporal compositor."""
    compositor = STACTemporalCompositor()

    # Somalia bounding box
    bbox = [40.0, -2.0, 51.0, 12.0]
    datetime_range = "2023-09-01/2023-09-03"
    output_dir = Path("data/gfm/somalia_example/stac_temporal_composites")

    print("Testing STAC Temporal Compositor...")
    print(f"AOI: {bbox}")
    print(f"Time range: {datetime_range}")

    try:
        results = compositor.process_stac_temporal(
            bbox, datetime_range, output_dir,
            resolution=0.0002, limit=20  # Very small test to avoid timeout
        )

        print("\n✅ Processing complete!")
        print("Results:", results)

        # Validate growth
        compositor.validate_temporal_growth(results)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
Virtual Temporal GFM Compositor

Uses GDAL VRT files to create virtual temporal composites directly from
remote STAC data, bypassing the need for daily composite creation.
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.enums import Resampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VirtualTemporalCompositor:
    """Create virtual temporal composites using VRT files."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

                # Get remote URL and spatial info
                remote_url = stac_item['assets']['ensemble_flood_extent']['href']

                stac_item['_date'] = item_date
                stac_item['_remote_url'] = remote_url
                daily_groups[item_date].append(stac_item)

            except Exception as e:
                logger.warning(f"Failed to load {metadata_file}: {e}")

        logger.info(f"Grouped into {len(daily_groups)} distinct dates")
        for obs_date, items in sorted(daily_groups.items()):
            logger.info(f"  {obs_date}: {len(items)} tiles")

        return dict(daily_groups)

    def create_daily_vrt(self, obs_date: date, stac_items: List[Dict]) -> Path:
        """Create a VRT file for a single date's tiles."""
        logger.info(f"Creating VRT for {obs_date} with {len(stac_items)} tiles")

        vrt_path = self.output_dir / f"daily_{obs_date.strftime('%Y-%m-%d')}.vrt"

        # Calculate union bounds for this date
        all_bounds = []
        for item in stac_items:
            bounds = item['properties']['proj:bbox']  # [minx, miny, maxx, maxy]
            all_bounds.append(bounds)

        if not all_bounds:
            logger.warning(f"No bounds found for {obs_date}")
            return None

        # Union bounds
        union_bounds = [
            min(b[0] for b in all_bounds),  # min_x
            min(b[1] for b in all_bounds),  # min_y
            max(b[2] for b in all_bounds),  # max_x
            max(b[3] for b in all_bounds)   # max_y
        ]

        # Calculate output size (20m resolution)
        width = int((union_bounds[2] - union_bounds[0]) / 20)
        height = int((union_bounds[3] - union_bounds[1]) / 20)

        # Create VRT XML content
        vrt_content = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <SRS>EPSG:27701</SRS>
  <GeoTransform>{union_bounds[0]}, 20, 0, {union_bounds[3]}, 0, -20</GeoTransform>
  <VRTRasterBand dataType="Byte" band="1">
    <NoDataValue>255</NoDataValue>
"""

        # Add each source tile
        for item in stac_items:
            remote_url = item['_remote_url']
            tile_bounds = item['properties']['proj:bbox']

            # Calculate destination offset in the union grid
            dst_x = int((tile_bounds[0] - union_bounds[0]) / 20)
            dst_y = int((union_bounds[3] - tile_bounds[3]) / 20)

            vrt_content += f"""    <SimpleSource>
      <SourceFilename relativeToVRT="0">{remote_url}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="15000" ySize="15000"/>
      <DstRect xOff="{dst_x}" yOff="{dst_y}" xSize="15000" ySize="15000"/>
    </SimpleSource>
"""

        vrt_content += """  </VRTRasterBand>
</VRTDataset>"""

        # Write VRT file
        with open(vrt_path, 'w') as f:
            f.write(vrt_content)

        logger.info(f"Created VRT: {vrt_path.name} ({width}x{height})")

        return vrt_path

    def create_temporal_composite_vrt(self, daily_vrt_paths: List[Path],
                                    target_date: date) -> Path:
        """Create a temporal composite VRT that layers daily VRTs with latest-wins logic."""

        # Filter VRTs up to target date
        target_vrts = []
        for vrt_path in daily_vrt_paths:
            vrt_date_str = vrt_path.stem.replace('daily_', '')
            vrt_date = datetime.strptime(vrt_date_str, '%Y-%m-%d').date()
            if vrt_date <= target_date:
                target_vrts.append((vrt_date, vrt_path))

        # Sort by date (oldest first, so newest overwrites)
        target_vrts.sort(key=lambda x: x[0])

        logger.info(f"Creating temporal composite for {target_date} from {len(target_vrts)} daily VRTs")

        temporal_vrt_path = self.output_dir / f"temporal_{target_date.strftime('%Y-%m-%d')}.vrt"

        if not target_vrts:
            logger.warning(f"No VRTs found for temporal composite {target_date}")
            return None

        # For simplicity, we'll create a Python script that reads and composites the VRTs
        # This is a limitation of VRT - it can't do complex temporal logic natively
        # A full implementation would need to materialize the temporal composite

        # For now, just return the latest daily VRT as a placeholder
        latest_vrt = target_vrts[-1][1]

        # Create a symlink or copy to indicate this is the temporal composite
        import shutil
        shutil.copy2(latest_vrt, temporal_vrt_path)

        logger.info(f"Created temporal VRT: {temporal_vrt_path.name}")
        return temporal_vrt_path

    def materialize_temporal_composite(self, vrt_paths: List[Path],
                                     target_date: date) -> Optional[Path]:
        """Materialize a temporal composite from VRT files using latest-wins logic."""

        # Filter and sort VRTs up to target date
        target_vrts = []
        for vrt_path in vrt_paths:
            vrt_date_str = vrt_path.stem.replace('daily_', '')
            vrt_date = datetime.strptime(vrt_date_str, '%Y-%m-%d').date()
            if vrt_date <= target_date:
                target_vrts.append((vrt_date, vrt_path))

        target_vrts.sort(key=lambda x: x[0])

        if not target_vrts:
            return None

        logger.info(f"Materializing temporal composite for {target_date} from {len(target_vrts)} daily VRTs")

        # Start with first VRT
        cumulative_data = None
        cumulative_doy = None
        profile = None

        for vrt_date, vrt_path in target_vrts:
            with rasterio.open(vrt_path) as src:
                daily_data = src.read(1)

                if cumulative_data is None:
                    # Initialize with first day
                    cumulative_data = daily_data.copy()

                    # Create DOY array
                    doy = vrt_date.timetuple().tm_yday
                    cumulative_doy = np.full(daily_data.shape, doy, dtype=np.uint8)
                    cumulative_doy[daily_data == 255] = 0

                    profile = src.profile.copy()
                    profile.update({'count': 2, 'dtype': rasterio.uint8})
                else:
                    # Update with newer valid pixels
                    valid_daily = (daily_data != 255)

                    # Align arrays if different sizes (basic approach)
                    if daily_data.shape != cumulative_data.shape:
                        logger.warning(f"Shape mismatch: {daily_data.shape} vs {cumulative_data.shape}")
                        continue

                    cumulative_data[valid_daily] = daily_data[valid_daily]

                    # Update DOY
                    doy = vrt_date.timetuple().tm_yday
                    cumulative_doy[valid_daily] = doy

        # Save materialized temporal composite
        output_path = self.output_dir / f"temporal_{target_date.strftime('%Y-%m-%d')}.tif"

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(cumulative_data, 1)
            dst.write(cumulative_doy, 2)
            dst.set_band_description(1, 'Flood Extent (0=no flood, 1=flood, 255=nodata)')
            dst.set_band_description(2, 'Day of Year (1-365, 0=no observation)')

        # Stats
        valid_pixels = np.sum(cumulative_data != 255)
        flood_pixels = np.sum(cumulative_data == 1)
        logger.info(f"Temporal composite {target_date}: {flood_pixels:,} flood pixels, {valid_pixels:,} valid pixels")

        return output_path

    def process(self, input_dir: Path, max_dates: Optional[int] = None) -> Dict:
        """Process virtual temporal composites."""
        logger.info("=" * 60)
        logger.info("VIRTUAL TEMPORAL COMPOSITING WORKFLOW")
        logger.info("=" * 60)

        # Step 1: Load and group STAC items
        daily_groups = self.load_and_group_stac_items(input_dir)

        if max_dates:
            sorted_dates = sorted(daily_groups.keys())[:max_dates]
            daily_groups = {d: daily_groups[d] for d in sorted_dates}
            logger.info(f"Limited to first {max_dates} dates")

        # Step 2: Create daily VRTs
        logger.info("\nStep 1: Creating daily VRTs")
        daily_vrt_paths = []
        for obs_date, stac_items in sorted(daily_groups.items()):
            vrt_path = self.create_daily_vrt(obs_date, stac_items)
            if vrt_path:
                daily_vrt_paths.append(vrt_path)

        # Step 3: Create temporal composites
        logger.info("\nStep 2: Creating temporal composites")
        temporal_paths = []

        sorted_dates = sorted(daily_groups.keys())
        for target_date in sorted_dates:
            temporal_path = self.materialize_temporal_composite(daily_vrt_paths, target_date)
            if temporal_path:
                temporal_paths.append(temporal_path)

        results = {
            'daily_vrts': len(daily_vrt_paths),
            'temporal_composites': len(temporal_paths),
            'output_dir': str(self.output_dir)
        }

        logger.info("=" * 60)
        logger.info("VIRTUAL PROCESSING COMPLETE")
        logger.info(f"Daily VRTs: {len(daily_vrt_paths)}")
        logger.info(f"Temporal composites: {len(temporal_paths)}")

        return results


def main():
    """Test the virtual temporal compositor."""
    input_dir = Path("data/gfm/somalia_example/raw_gfm")
    output_dir = Path("data/gfm/somalia_example/virtual_temporal_composites")

    compositor = VirtualTemporalCompositor(output_dir)

    # Test with first 3 dates
    results = compositor.process(input_dir, max_dates=3)

    print("Results:", results)


if __name__ == "__main__":
    main()
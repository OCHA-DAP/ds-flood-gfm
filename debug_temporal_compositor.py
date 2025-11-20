"""
Debug script for temporal compositor - focused test with flood data
"""

import rasterio
import numpy as np
from pathlib import Path
from src.ds_flood_gfm.temporal_compositor import TemporalGFMCompositor

def find_flood_tiles(input_dir: Path, max_check: int = 50) -> list:
    """Find tiles that actually contain flood data."""
    tif_files = list(input_dir.glob('*.tif'))
    flood_files = []

    print(f"Checking {min(max_check, len(tif_files))} tiles for flood data...")

    for i, tif_file in enumerate(tif_files[:max_check]):
        try:
            with rasterio.open(tif_file) as src:
                data = src.read(1)
                flood_count = np.sum(data == 1)

                if flood_count > 0:
                    total_count = data.size
                    pct = 100 * flood_count / total_count
                    print(f"  {tif_file.name}: {flood_count:,} flood pixels ({pct:.4f}%)")
                    flood_files.append(tif_file.name.replace('.tif', ''))

        except Exception as e:
            print(f"Error reading {tif_file.name}: {e}")

    print(f"Found {len(flood_files)} tiles with flood data")
    return flood_files

def test_temporal_compositor():
    """Test temporal compositor with a small focused dataset."""

    input_dir = Path("data/gfm/somalia_example/raw_gfm")
    output_dir = Path("data/gfm/somalia_example/temporal_composites")

    # Find tiles with flood data
    flood_tile_ids = find_flood_tiles(input_dir, max_check=100)

    if not flood_tile_ids:
        print("No flood data found! Cannot test compositor.")
        return

    # Initialize compositor
    compositor = TemporalGFMCompositor()

    # Load all STAC items
    print("\nLoading STAC metadata...")
    stac_items = compositor.load_and_sort_stac_items(input_dir)

    # Filter to only items with flood data
    print(f"\nFiltering to items with flood data...")
    flood_items = [item for item in stac_items if item['id'] in flood_tile_ids]
    print(f"Found {len(flood_items)} STAC items with flood data")

    if flood_items:
        print("Date range of flood items:")
        dates = [item['_datetime'].date() for item in flood_items]
        print(f"  {min(dates)} to {max(dates)}")

    # Process with smaller dataset
    print(f"\nCreating temporal composites...")
    composites_metadata = compositor.create_temporal_composite(
        flood_items[:10],  # Just first 10 flood items
        input_dir,
        output_dir,
        max_items=10
    )

    # Create STAC catalog
    if composites_metadata:
        print(f"\nCreating STAC catalog...")
        catalog_path = compositor.create_stac_catalog(composites_metadata, output_dir)
        print(f"STAC catalog created: {catalog_path}")

        # Validate composite has flood data
        print(f"\nValidating composite flood data...")
        for comp_meta in composites_metadata:
            flood_path = comp_meta['flood_extent_path']
            with rasterio.open(flood_path) as src:
                data = src.read(1)
                flood_pixels = np.sum(data == 1)
                total_pixels = data.size
                flood_pct = 100 * flood_pixels / total_pixels if total_pixels > 0 else 0

                print(f"  {Path(flood_path).name}: {flood_pixels:,} flood pixels ({flood_pct:.4f}%)")

    print(f"\nTest complete! Created {len(composites_metadata)} temporal composites")

if __name__ == "__main__":
    test_temporal_compositor()
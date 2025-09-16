#!/usr/bin/env python3
"""
Test DOY provenance logic in isolation
"""
import numpy as np
import xarray as xr
import rasterio
from pathlib import Path
from datetime import datetime

def test_doy_logic():
    """Test DOY provenance tracking logic."""
    print("Testing DOY provenance logic...")

    # Create test data similar to STAC compositor
    test_date = datetime(2023, 9, 1)
    expected_doy = test_date.timetuple().tm_yday
    print(f"Expected DOY for {test_date.date()}: {expected_doy}")

    # Create a small test composite (simulating what STAC compositor creates)
    height, width = 100, 100
    composite_data = np.full((height, width), 255, dtype=np.uint8)  # All nodata

    # Add some valid flood data in a small area
    composite_data[40:60, 40:60] = 0  # No flood
    composite_data[45:55, 45:55] = 1  # Flood

    # Create DOY array (this is the logic from STAC compositor)
    doy_data = np.full((height, width), expected_doy, dtype=np.uint8)

    print(f"DOY array unique values: {np.unique(doy_data)}")
    print(f"Composite unique values: {np.unique(composite_data)}")

    # Write test file
    output_path = Path("data/gfm/somalia_example/stac_temporal_composites/test_doy.tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 2,
        'dtype': rasterio.uint8,
        'crs': 'EPSG:4326',
        'transform': rasterio.transform.from_bounds(40, -2, 41, -1, width, height),
        'compress': 'lzw',
        'nodata': None
    }

    print(f"Writing test file to {output_path}")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(composite_data, 1)
        dst.write(doy_data, 2)
        dst.set_band_description(1, 'Flood Extent (0=no flood, 1=flood, 255=nodata)')
        dst.set_band_description(2, 'Day of Year (1-366, 0=no observation)')

    # Verify the file
    print("Verifying saved file...")
    with rasterio.open(output_path) as src:
        saved_composite = src.read(1)
        saved_doy = src.read(2)

        print(f"Saved composite unique values: {np.unique(saved_composite)}")
        print(f"Saved DOY unique values: {np.unique(saved_doy)}")
        print(f"Band descriptions: {src.descriptions}")

        # Check if DOY is preserved
        if expected_doy in np.unique(saved_doy):
            print(f"✅ DOY {expected_doy} correctly preserved in file")
        else:
            print(f"❌ DOY {expected_doy} not found in saved file")

        # Check valid flood pixels
        valid_pixels = np.sum(saved_composite != 255)
        flood_pixels = np.sum(saved_composite == 1)
        print(f"Valid pixels: {valid_pixels}, Flood pixels: {flood_pixels}")

if __name__ == "__main__":
    test_doy_logic()
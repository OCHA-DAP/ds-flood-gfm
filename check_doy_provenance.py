#!/usr/bin/env python3
"""
Check DOY provenance tracking in temporal composites
"""
import numpy as np
import rasterio
from pathlib import Path

def check_doy_provenance():
    """Check DOY provenance tracking in temporal composites."""
    output_dir = Path("data/gfm/somalia_example/stac_temporal_composites")

    # Find all temporal composite files
    tif_files = sorted(list(output_dir.glob("stac_temporal_*.tif")))

    print(f"Checking DOY provenance in {len(tif_files)} files:")
    print()

    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            print(f"File: {tif_file.name}")
            print(f"  Bands: {src.count}")
            print(f"  Band 1 description: {src.descriptions[0]}")
            print(f"  Band 2 description: {src.descriptions[1]}")

            # Read both bands
            flood_data = src.read(1)
            doy_data = src.read(2)

            # Analyze DOY values
            unique_doy = np.unique(doy_data)

            print(f"  DOY values found: {sorted(unique_doy)}")

            # Check DOY where we have valid flood data
            valid_flood_mask = (flood_data != 255)
            if np.any(valid_flood_mask):
                valid_doy_values = np.unique(doy_data[valid_flood_mask])
                print(f"  DOY values with valid flood data: {sorted(valid_doy_values)}")
            else:
                print(f"  No valid flood pixels found")

            # Expected DOY for this date
            date_str = tif_file.stem.replace('stac_temporal_', '')
            year, month, day = map(int, date_str.split('-'))

            from datetime import date
            expected_doy = date(year, month, day).timetuple().tm_yday
            print(f"  Expected DOY for {date_str}: {expected_doy}")

            # Check if expected DOY is present
            if expected_doy in unique_doy:
                print(f"  ✅ Expected DOY {expected_doy} is present")
            else:
                print(f"  ⚠️  Expected DOY {expected_doy} not found")

            print()

if __name__ == "__main__":
    check_doy_provenance()
#!/usr/bin/env python3
"""
Quick validation script for temporal composite growth
"""
import numpy as np
import rasterio
from pathlib import Path

def validate_temporal_growth():
    """Validate that temporal composites show growing coverage."""
    output_dir = Path("data/gfm/somalia_example/stac_temporal_composites")

    # Find all temporal composite files
    tif_files = sorted(list(output_dir.glob("stac_temporal_*.tif")))

    print(f"Found {len(tif_files)} temporal composite files:")

    results = []
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            # Read flood extent band
            flood_data = src.read(1)
            doy_data = src.read(2)

            # Calculate stats
            valid_pixels = int(np.sum(flood_data != 255))
            flood_pixels = int(np.sum(flood_data == 1))
            total_pixels = int(flood_data.size)
            coverage_pct = 100 * valid_pixels / total_pixels

            date_str = tif_file.stem.replace('stac_temporal_', '')

            result = {
                'date': date_str,
                'file': tif_file.name,
                'valid_pixels': valid_pixels,
                'flood_pixels': flood_pixels,
                'total_pixels': total_pixels,
                'coverage_pct': coverage_pct
            }
            results.append(result)

            print(f"  {date_str}: {flood_pixels:,} flood pixels, "
                  f"{valid_pixels:,} valid pixels ({coverage_pct:.3f}% coverage)")

    # Validate growth
    print("\nGrowth validation:")
    for i in range(1, len(results)):
        prev_result = results[i-1]
        curr_result = results[i]

        prev_coverage = prev_result['coverage_pct']
        curr_coverage = curr_result['coverage_pct']

        if curr_coverage >= prev_coverage:
            growth_pct = curr_coverage - prev_coverage
            print(f"✅ {curr_result['date']}: Coverage grew by {growth_pct:.3f}% "
                  f"(from {prev_coverage:.3f}% to {curr_coverage:.3f}%)")
        else:
            decline_pct = prev_coverage - curr_coverage
            print(f"⚠️  {curr_result['date']}: Coverage decreased by {decline_pct:.3f}% "
                  f"(from {prev_coverage:.3f}% to {curr_coverage:.3f}%)")

    return results

if __name__ == "__main__":
    results = validate_temporal_growth()
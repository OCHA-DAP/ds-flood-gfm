"""
GFM Latest 3 Days Pipeline

Orchestrates the full workflow for downloading and processing GFM flood extent data
for the latest 3 days for a specified country.

Workflow:
1. Load country boundary using ISO3 code
2. Query STAC API for latest 3 days of GFM data
3. Build stackstac array (EPSG:4326)
4. Clip to country bounding box
5. Create daily composites using max() - NO forward fill
6. Write each day's composite as a Cloud Optimized GeoTIFF (COG)
7. Print summary of results

Usage:
    uv run python src/ds_flood_gfm/pipeline_latest_3days.py
    uv run python src/ds_flood_gfm/pipeline_latest_3days.py --iso3 NGA
    uv run python src/ds_flood_gfm/pipeline_latest_3days.py --iso3 ETH --days 7 --output-dir data/gfm/ethiopia
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pystac_client
import stackstac
import xarray as xr

from ds_flood_gfm.geo_utils import load_adm0_fieldmaps


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and process latest GFM flood extent data for a country",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--iso3",
        type=str,
        default="JAM",
        help="ISO3 country code (e.g., JAM, NGA, ETH)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days to process (counting back from today)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for COG files (default: data/gfm/{iso3_lower}_latest)",
    )

    parser.add_argument(
        "--stac-url",
        type=str,
        default="https://stac.eodc.eu/api/v1",
        help="STAC API endpoint URL",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress messages",
    )

    return parser.parse_args()


def load_country_boundary(iso3: str, verbose: bool = False) -> gpd.GeoDataFrame:
    """
    Load country boundary from FieldMaps.io humanitarian data.

    Parameters
    ----------
    iso3 : str
        ISO3 country code
    verbose : bool
        Print progress messages

    Returns
    -------
    gpd.GeoDataFrame
        Country boundary
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 1: Loading Country Boundary")
        print(f"{'='*80}")
        print(f"  ISO3 code: {iso3}")

    try:
        gdf_country = load_adm0_fieldmaps(iso3)

        if verbose:
            country_name = gdf_country.adm0_name.iloc[0]
            bbox = gdf_country.total_bounds
            print(f"  Country: {country_name}")
            print(f"  Bounding box: {bbox}")
            print(f"  ✅ Country boundary loaded")

        return gdf_country

    except Exception as e:
        print(f"  ❌ ERROR: Failed to load country boundary: {e}")
        sys.exit(1)


def query_stac_api(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    stac_url: str,
    verbose: bool = False,
) -> List:
    """
    Query STAC API for GFM data.

    Parameters
    ----------
    bbox : tuple
        Bounding box (minx, miny, maxx, maxy) in EPSG:4326
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    stac_url : str
        STAC API endpoint
    verbose : bool
        Print progress messages

    Returns
    -------
    list
        STAC items
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 2: Querying STAC API")
        print(f"{'='*80}")
        print(f"  STAC endpoint: {stac_url}")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Bounding box: {bbox}")

    try:
        client = pystac_client.Client.open(stac_url)

        datetime_range = f"{start_date}/{end_date}"
        search = client.search(
            collections=["GFM"],
            bbox=bbox,
            datetime=datetime_range,
        )

        items = list(search.items())

        if verbose:
            print(f"  ✅ Found {len(items)} STAC items")

            # Show date breakdown
            if items:
                dates = {}
                for item in items:
                    date_str = item.datetime.strftime("%Y-%m-%d")
                    dates[date_str] = dates.get(date_str, 0) + 1

                print(f"\n  Items by date:")
                for date_str in sorted(dates.keys()):
                    print(f"    {date_str}: {dates[date_str]} items")

        return items

    except Exception as e:
        print(f"  ❌ ERROR: STAC query failed: {e}")
        sys.exit(1)


def build_stackstac_array(
    items: List,
    bbox: Tuple[float, float, float, float],
    verbose: bool = False,
) -> xr.DataArray:
    """
    Build stackstac data array from STAC items.

    Parameters
    ----------
    items : list
        STAC items
    bbox : tuple
        Bounding box to clip to
    verbose : bool
        Print progress messages

    Returns
    -------
    xr.DataArray
        Flood extent data array
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 3: Building Stackstac Array")
        print(f"{'='*80}")

    try:
        # Create stack
        if verbose:
            print(f"  Creating stack from {len(items)} items...")

        stack = stackstac.stack(items, epsg=4326)

        # Select flood extent band
        stack_flood = stack.sel(band="ensemble_flood_extent")

        # Clip to bbox
        if verbose:
            print(f"  Clipping to bounding box...")

        stack_flood_clipped = stack_flood.sel(
            x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1])
        )

        if verbose:
            print(f"  ✅ Stack shape: {stack_flood_clipped.shape}")
            print(f"     Time steps: {len(stack_flood_clipped.time)}")
            print(f"     Spatial: {stack_flood_clipped.shape[1]} x {stack_flood_clipped.shape[2]} pixels")

        return stack_flood_clipped

    except Exception as e:
        print(f"  ❌ ERROR: Failed to build stack: {e}")
        sys.exit(1)


def create_daily_composites(
    stack_flood: xr.DataArray, verbose: bool = False
) -> xr.DataArray:
    """
    Create daily composites using max() - NO forward fill.

    This groups multiple observations per day and takes the maximum value,
    which represents the union of all flood detections for that day.

    Parameters
    ----------
    stack_flood : xr.DataArray
        Flood extent data array
    verbose : bool
        Print progress messages

    Returns
    -------
    xr.DataArray
        Daily composited flood extent
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 4: Creating Daily Composites")
        print(f"{'='*80}")
        print(f"  Method: groupby('time.date').max()")
        print(f"  NO forward fill - only actual observations")

    try:
        # Group by date and take max
        stack_flood_max = stack_flood.groupby("time.date").max()

        # Rename dimension back to 'time'
        stack_flood_max = stack_flood_max.rename({"date": "time"})
        stack_flood_max["time"] = stack_flood_max.time.astype("datetime64[ns]")

        if verbose:
            print(f"  ✅ Created {len(stack_flood_max.time)} daily composites")
            print(f"\n  Composite dates:")
            for t in stack_flood_max.time.values:
                print(f"    - {str(t)[:10]}")

        return stack_flood_max

    except Exception as e:
        print(f"  ❌ ERROR: Failed to create composites: {e}")
        sys.exit(1)


def write_cog(
    data_array: xr.DataArray,
    output_path: Path,
    date_str: str,
    verbose: bool = False,
) -> Dict:
    """
    Write a daily composite as a Cloud Optimized GeoTIFF (COG).

    Parameters
    ----------
    data_array : xr.DataArray
        Data to write
    output_path : Path
        Output file path
    date_str : str
        Date string for metadata
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Metadata about the written file
    """
    import rioxarray  # Import here to add rio accessor

    try:
        # Ensure CRS is set
        if data_array.rio.crs is None:
            data_array = data_array.rio.write_crs("EPSG:4326")

        # Compute the data (trigger lazy evaluation)
        if verbose:
            print(f"    Computing data from EODC...")

        data_computed = data_array.compute()

        # Write as COG
        if verbose:
            print(f"    Writing COG to: {output_path.name}")

        data_computed.rio.to_raster(
            output_path,
            driver="COG",
            compress="deflate",
            tiled=True,
        )

        # Calculate statistics
        valid_pixels = int((~np.isnan(data_computed.values)).sum())
        flood_pixels = int((data_computed.values == 1).sum())
        flood_pct = (flood_pixels / valid_pixels * 100) if valid_pixels > 0 else 0

        metadata = {
            "date": date_str,
            "file_path": str(output_path),
            "file_size_mb": output_path.stat().st_size / (1024 * 1024),
            "valid_pixels": valid_pixels,
            "flood_pixels": flood_pixels,
            "flood_pct": flood_pct,
        }

        if verbose:
            print(f"    ✅ File size: {metadata['file_size_mb']:.2f} MB")
            print(f"       Valid pixels: {valid_pixels:,}")
            print(f"       Flood pixels: {flood_pixels:,} ({flood_pct:.2f}%)")

        return metadata

    except Exception as e:
        print(f"    ❌ ERROR writing COG: {e}")
        return None


def process_and_write_cogs(
    stack_flood_max: xr.DataArray,
    output_dir: Path,
    verbose: bool = False,
) -> List[Dict]:
    """
    Process daily composites and write as COGs.

    Parameters
    ----------
    stack_flood_max : xr.DataArray
        Daily composited flood extent
    output_dir : Path
        Output directory
    verbose : bool
        Print progress messages

    Returns
    -------
    list
        List of metadata dictionaries for written files
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 5: Writing Daily COGs")
        print(f"{'='*80}")
        print(f"  Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, time_step in enumerate(stack_flood_max.time.values):
        date_str = str(time_step)[:10]

        if verbose:
            print(f"\n  📅 Processing {date_str} ({i+1}/{len(stack_flood_max.time)})")

        # Get data for this date
        data_daily = stack_flood_max.sel(time=time_step)

        # Define output path
        output_path = output_dir / f"gfm_flood_extent_{date_str}.tif"

        # Write COG
        metadata = write_cog(data_daily, output_path, date_str, verbose)

        if metadata:
            results.append(metadata)

    return results


def print_summary(
    iso3: str,
    country_name: str,
    start_date: str,
    end_date: str,
    results: List[Dict],
    verbose: bool = False,
):
    """
    Print summary of processing results.

    Parameters
    ----------
    iso3 : str
        ISO3 country code
    country_name : str
        Country name
    start_date : str
        Start date
    end_date : str
        End date
    results : list
        List of result metadata dictionaries
    verbose : bool
        Print detailed summary
    """
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")

    print(f"\nCountry: {country_name} ({iso3})")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Files created: {len(results)}")

    if results:
        total_size = sum(r["file_size_mb"] for r in results)
        print(f"Total size: {total_size:.2f} MB")

        if verbose:
            print(f"\n{'Date':<12} {'Flood Pixels':>15} {'Flood %':>10} {'File Size (MB)':>15}")
            print("-" * 55)
            for r in results:
                print(
                    f"{r['date']:<12} {r['flood_pixels']:>15,} {r['flood_pct']:>9.2f}% {r['file_size_mb']:>14.2f}"
                )

        print(f"\nOutput files:")
        for r in results:
            print(f"  - {r['file_path']}")

    else:
        print("  ⚠️  No data found for the specified date range")

    print(f"\n{'='*80}")


def main():
    """Main pipeline execution."""
    args = parse_args()

    # Print header
    if args.verbose:
        print(f"\n{'='*80}")
        print("GFM LATEST DAYS PIPELINE")
        print(f"{'='*80}")
        print(f"ISO3: {args.iso3}")
        print(f"Days: {args.days}")
        print(f"STAC URL: {args.stac_url}")

    # 1. Load country boundary
    gdf_country = load_country_boundary(args.iso3, args.verbose)
    country_name = gdf_country.adm0_name.iloc[0]
    bbox = tuple(gdf_country.total_bounds)

    # 2. Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days - 1)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # 3. Query STAC API
    items = query_stac_api(bbox, start_date_str, end_date_str, args.stac_url, args.verbose)

    if not items:
        print("\n⚠️  No STAC items found for the specified date range")
        sys.exit(0)

    # 4. Build stackstac array
    stack_flood_clipped = build_stackstac_array(items, bbox, args.verbose)

    # 5. Create daily composites
    stack_flood_max = create_daily_composites(stack_flood_clipped, args.verbose)

    # 6. Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/gfm/{args.iso3.lower()}_latest")

    # 7. Process and write COGs
    results = process_and_write_cogs(stack_flood_max, output_dir, args.verbose)

    # 8. Print summary
    print_summary(args.iso3, country_name, start_date_str, end_date_str, results, args.verbose)

    # Success
    if results:
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Pipeline completed but no files were created")
        sys.exit(1)


if __name__ == "__main__":
    main()

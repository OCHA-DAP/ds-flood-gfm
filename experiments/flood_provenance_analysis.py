"""
Jamaica Flood Monitoring - Parameterized Provenance + Flood Density Analysis

General-purpose script that can be run for any date range.

Usage:
    python flood_provenance_analysis.py --end-date 2025-10-27 --days-back 7
    python flood_provenance_analysis.py --end-date 2024-07-15 --days-back 3

Key Logic for Flood Density:
- Only uses flood pixels from their LATEST provenance date
- If a newer observation covers an area, we ignore older data from that area
- This avoids counting "stale" flood pixels that may have receded

Output:
- Saves provenance + flood density map as PNG
"""

import argparse
from datetime import datetime, timedelta
import pystac_client
import stackstac
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from fsspec.implementations.http import HTTPFileSystem
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.features import geometry_mask
from pathlib import Path
import json
import hashlib


def generate_cache_key(iso3, dates_list, population_raster):
    """Generate unique cache key from parameters."""
    # Create string representation of dates
    dates_str = "_".join([str(d)[:10] for d in dates_list])
    pop_str = "pop" if population_raster else "nopos"
    cache_str = f"{iso3}_{dates_str}_{pop_str}"
    # Use hash for shorter filenames
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:8]
    return f"{iso3}_{cache_hash}"


def save_cache(cache_dir, cache_key, flood_points, provenance_indexed, provenance_target, unique_dates, metadata):
    """Save processed data to cache."""
    cache_path = Path(cache_dir) / cache_key
    cache_path.mkdir(parents=True, exist_ok=True)

    # Save flood points as GeoParquet
    if len(flood_points) > 0:
        gdf_floods = gpd.GeoDataFrame(flood_points, crs='EPSG:4326')
        gdf_floods.to_parquet(cache_path / "flood_points.parquet")

    # Save provenance raster as GeoTIFF
    from rasterio.transform import from_bounds
    transform = from_bounds(
        float(provenance_target.x.min()),
        float(provenance_target.y.min()),
        float(provenance_target.x.max()),
        float(provenance_target.y.max()),
        provenance_indexed.shape[1],
        provenance_indexed.shape[0]
    )

    with rasterio.open(
        cache_path / "provenance.tif",
        'w',
        driver='GTiff',
        height=provenance_indexed.shape[0],
        width=provenance_indexed.shape[1],
        count=1,
        dtype=provenance_indexed.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(provenance_indexed, 1)

    # Save metadata as JSON
    metadata_serializable = {
        'unique_dates': [str(d) for d in unique_dates],
        'x_min': float(provenance_target.x.min()),
        'x_max': float(provenance_target.x.max()),
        'y_min': float(provenance_target.y.min()),
        'y_max': float(provenance_target.y.max()),
        **metadata
    }

    with open(cache_path / "metadata.json", 'w') as f:
        json.dump(metadata_serializable, f, indent=2)

    print(f"\n✓ Saved cache to: {cache_path}")


def load_cache(cache_dir, cache_key):
    """Load processed data from cache."""
    cache_path = Path(cache_dir) / cache_key

    if not cache_path.exists():
        return None

    # Check all required files exist
    required_files = ["metadata.json", "provenance.tif"]
    if not all((cache_path / f).exists() for f in required_files):
        return None

    print(f"\n✓ Loading from cache: {cache_path}")

    # Load metadata
    with open(cache_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    # Load flood points if they exist
    flood_points_file = cache_path / "flood_points.parquet"
    if flood_points_file.exists():
        gdf_floods = gpd.read_parquet(flood_points_file)
        flood_points = gdf_floods.to_dict('records')
        # Restore geometry as Point objects
        for fp in flood_points:
            fp['geometry'] = Point(fp['lon'], fp['lat'])
    else:
        flood_points = []

    # Load provenance raster
    with rasterio.open(cache_path / "provenance.tif") as src:
        provenance_indexed = src.read(1)

    # Convert unique_dates back to datetime
    unique_dates = [np.datetime64(d) for d in metadata['unique_dates']]

    return {
        'flood_points': flood_points,
        'provenance_indexed': provenance_indexed,
        'unique_dates': unique_dates,
        'metadata': metadata
    }


def main(end_date_str, n_latest, iso3="JAM", population_raster=None, cache_dir="data/cache", use_cache=True):
    """
    Run flood provenance analysis using the N most recent observations before end_date.

    Parameters:
    -----------
    end_date_str : str
        End date in YYYY-MM-DD format
    n_latest : int
        Number of most recent observations to use
    iso3 : str
        ISO3 country code (default: JAM for Jamaica)
    population_raster : str, optional
        Path to GHSL population raster. If provided, creates affected population density map.
    """

    # Parse dates - look back 60 days to find available data
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    search_start_date = end_date - timedelta(days=60)

    start_date_str = search_start_date.strftime("%Y-%m-%d")

    print("="*80)
    print(f"FLOOD PROVENANCE ANALYSIS: {iso3}")
    print(f"End date: {end_date_str}")
    print(f"Searching for {n_latest} most recent observations (looking back 60 days)")
    print("="*80)

    # Configuration
    OUTPUT_DIR = "experiments"
    ADMIN_FILE = "experiments/claude-tests/jamaica_admin_cleaned.geojson"

    # Load cleaned boundaries
    gdf_aoi = gpd.read_file(ADMIN_FILE)
    bbox = gdf_aoi.total_bounds
    print(f"\nBounding box: {bbox}")

    # Load admin1 boundaries for internal divisions
    GLOBAL_ADM1 = "https://data.fieldmaps.io/edge-matched/humanitarian/intl/adm1_polygons.parquet"
    filesystem = HTTPFileSystem()
    filters = [("iso_3", "=", iso3)]
    gdf_admin1 = gpd.read_parquet(GLOBAL_ADM1, filesystem=filesystem, filters=filters)
    print(f"Loaded {len(gdf_admin1)} admin1 divisions")

    # Query STAC API
    stac_api = "https://stac.eodc.eu/api/v1"
    client = pystac_client.Client.open(stac_api)
    search = client.search(
        collections=["GFM"],
        bbox=bbox,
        datetime=f"{start_date_str}/{end_date_str}"
    )
    items = search.item_collection()
    print(f"Found {len(items)} STAC items")

    if len(items) == 0:
        print("ERROR: No STAC items found for this date range!")
        return

    # Build xarray stack
    print("\nBuilding xarray stack...")
    stack = stackstac.stack(items, epsg=4326)
    stack_flood = stack.sel(band="ensemble_flood_extent")
    stack_flood_clipped = stack_flood.sel(
        x=slice(bbox[0], bbox[2]),
        y=slice(bbox[3], bbox[1])
    )

    # Create daily composites
    print("Creating daily composites...")
    stack_flood_max = stack_flood_clipped.groupby("time.date").max()
    stack_flood_max = stack_flood_max.rename({"date": "time"})
    stack_flood_max["time"] = stack_flood_max.time.astype("datetime64[ns]")

    all_dates = stack_flood_max.time.values
    print(f"All dates found: {[str(d)[:10] for d in all_dates]}")

    if len(all_dates) == 0:
        print("ERROR: No valid dates after compositing!")
        return

    # Select only the N most recent dates
    all_dates_sorted = np.sort(all_dates)
    dates_to_use = all_dates_sorted[-n_latest:] if len(all_dates_sorted) >= n_latest else all_dates_sorted

    print(f"Using {len(dates_to_use)} most recent dates: {[str(d)[:10] for d in dates_to_use]}")

    # Filter stack to only use selected dates
    stack_flood_max = stack_flood_max.sel(time=dates_to_use)
    dates = stack_flood_max.time.values

    # Create 'ever_has_data' mask
    print("\nCreating 'ever_has_data' mask...")
    ever_has_data = xr.full_like(stack_flood_max.isel(time=0), fill_value=False, dtype=bool)
    for i in range(len(dates)):
        has_data_this_time = ~np.isnan(stack_flood_max.isel(time=i))
        ever_has_data = ever_has_data | has_data_this_time

    pixels_with_data = ever_has_data.sum().values
    pixels_no_data = (~ever_has_data).sum().values
    print(f"  Pixels with data: {pixels_with_data:,}")
    print(f"  Pixels without data: {pixels_no_data:,}")
    print(f"  Coverage: {100 * pixels_with_data / (pixels_with_data + pixels_no_data):.1f}%")

    # Track provenance
    print("\nTracking provenance...")
    provenance = xr.full_like(stack_flood_max, fill_value=np.datetime64('NaT'), dtype='datetime64[ns]')
    for i, time_val in enumerate(dates):
        has_data = ~np.isnan(stack_flood_max.isel(time=i))
        provenance[i] = xr.where(has_data, time_val, np.datetime64('NaT'))

    # Forward-fill
    print("Forward-filling...")
    flood_filled = stack_flood_max.ffill(dim="time")
    provenance_filled = provenance.ffill(dim="time")

    # Apply spatial mask
    print("Applying spatial mask to provenance...")
    for i in range(len(dates)):
        provenance_filled[i] = xr.where(ever_has_data, provenance_filled[i], np.datetime64('NaT'))

    # Use the latest available date instead of the requested end_date
    latest_date = str(dates[-1])[:10]
    print(f"\nCreating provenance + flood map for {latest_date} (latest available)...")
    create_map(
        latest_date,
        flood_filled,
        provenance_filled,
        stack_flood_max,
        dates,
        gdf_aoi,
        gdf_admin1,
        OUTPUT_DIR,
        population_raster
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


def create_map(target_date, flood_filled, provenance_filled, stack_flood_max,
               dates, gdf_aoi, gdf_admin1, output_dir, population_raster=None):
    """Create provenance + flood density map for target date.

    If population_raster is provided, creates affected population density instead of flood pixel density.
    """

    # Extract data for target date
    flood_target = flood_filled.sel(time=target_date).compute()
    provenance_target = provenance_filled.sel(time=target_date).compute()

    # Get unique provenance dates
    unique_dates = pd.unique(provenance_target.values.ravel())
    unique_dates = unique_dates[~pd.isna(unique_dates)]
    unique_dates = np.sort(unique_dates)

    print(f"\nProvenance breakdown:")
    total_valid = np.sum(~pd.isna(provenance_target.values))
    for date in unique_dates:
        count = np.sum(provenance_target.values == date)
        pct = 100 * count / total_valid
        print(f"  {str(pd.Timestamp(date))[:10]}: {count:,} pixels ({pct:.1f}%)")

    no_data_count = np.sum(pd.isna(provenance_target.values))
    pct_no_data = 100 * no_data_count / provenance_target.size
    print(f"  NO DATA: {no_data_count:,} pixels ({pct_no_data:.1f}% of bbox)")

    # Extract flood pixels by provenance date
    print("\nExtracting flood pixels by provenance date...")
    flood_points = []

    for date in unique_dates:
        # Get original flood data for this date
        original_flood = stack_flood_max.sel(time=date).compute()

        # Mask: provenance == date AND flood == 1
        is_this_provenance = (provenance_target.values == date)
        is_flooded = (original_flood.values == 1)

        flood_mask = is_this_provenance & is_flooded
        flood_count = np.sum(flood_mask)

        print(f"  {str(pd.Timestamp(date))[:10]}: {flood_count:,} flood pixels")

        # Convert to points
        if flood_count > 0:
            y_coords, x_coords = np.where(flood_mask)
            x_geo = provenance_target.x.values[x_coords]
            y_geo = provenance_target.y.values[y_coords]

            for x, y in zip(x_geo, y_geo):
                flood_points.append({
                    'geometry': Point(x, y),
                    'date': str(pd.Timestamp(date))[:10],
                    'lon': x,
                    'lat': y
                })

    print(f"\nTotal unique flood pixels: {len(flood_points):,}")

    # Sample population values at flood locations if raster provided
    if population_raster and len(flood_points) > 0:
        print(f"\nSampling population values from: {population_raster}")
        try:
            # Check if using blob storage (ocha_stratus) - assume blob if not a local file path
            if not population_raster.startswith("/") and not population_raster.startswith("./"):
                import ocha_stratus as stratus
                import rioxarray
                import xarray as xr

                print(f"  Accessing blob: {population_raster}")

                # Open blob COG directly (no "blob://" prefix needed)
                da_pop = stratus.open_blob_cog(population_raster, container_name='raster').squeeze(drop=True)

                # Clip to Jamaica bbox for faster processing
                min_x, min_y, max_x, max_y = gdf_aoi.total_bounds
                da_pop_clip = da_pop.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)

                # Sample at flood point locations using xarray selection
                total_affected_pop = 0
                for fp in flood_points:
                    lon, lat = fp['lon'], fp['lat']
                    try:
                        # Use sel with nearest neighbor
                        pop_val = float(da_pop_clip.sel(x=lon, y=lat, method='nearest').values)
                        if np.isnan(pop_val) or pop_val < 0:
                            pop_val = 0
                        fp['population'] = pop_val
                        total_affected_pop += pop_val
                    except:
                        fp['population'] = 0

                print(f"  Total affected population: {total_affected_pop:,.0f}")
                print(f"  Average population per flood pixel: {total_affected_pop/len(flood_points):.1f}")

            else:
                # Local file using rasterio
                with rasterio.open(population_raster) as pop_src:
                    # Extract coordinates
                    coords = [(fp['lon'], fp['lat']) for fp in flood_points]

                    # Sample population raster at flood point locations
                    pop_values = list(pop_src.sample(coords))

                    # Add population to flood_points
                    total_affected_pop = 0
                    for i, fp in enumerate(flood_points):
                        pop_val = float(pop_values[i][0])
                        if pop_val < 0:  # Handle nodata values
                            pop_val = 0
                        fp['population'] = pop_val
                        total_affected_pop += pop_val

                    print(f"  Total affected population: {total_affected_pop:,.0f}")
                    print(f"  Average population per flood pixel: {total_affected_pop/len(flood_points):.1f}")

        except Exception as e:
            print(f"  WARNING: Could not sample population raster: {e}")
            print(f"  Falling back to flood pixel density")
            population_raster = None  # Fall back to flood pixel density

    # Create indexed provenance array
    date_to_idx = {date: i for i, date in enumerate(unique_dates)}
    provenance_indexed = np.full(provenance_target.shape, -1, dtype=np.int32)
    for date, idx in date_to_idx.items():
        provenance_indexed[provenance_target.values == date] = idx

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Colors: oldest to most recent (green -> yellow -> orange)
    colors = ['#91cf60', '#ffffbf', '#fc8d59'][:len(unique_dates)]
    cmap_prov = ListedColormap(colors)

    # Plot grey background for no data
    grey_layer = np.ones(provenance_indexed.shape) * 0.5
    ax.imshow(
        grey_layer,
        cmap='Greys',
        extent=[
            float(provenance_target.x.min()),
            float(provenance_target.x.max()),
            float(provenance_target.y.min()),
            float(provenance_target.y.max())
        ],
        origin='upper',
        vmin=0,
        vmax=1,
        zorder=1
    )

    # Plot provenance
    prov_masked = np.ma.masked_where(provenance_indexed == -1, provenance_indexed)
    ax.imshow(
        prov_masked,
        cmap=cmap_prov,
        extent=[
            float(provenance_target.x.min()),
            float(provenance_target.x.max()),
            float(provenance_target.y.min()),
            float(provenance_target.y.max())
        ],
        origin='upper',
        interpolation='nearest',
        vmin=0,
        vmax=len(unique_dates)-1,
        zorder=2
    )

    # Create flood density heatmap (population-weighted if available)
    if len(flood_points) > 0:
        gdf_floods = gpd.GeoDataFrame(flood_points, crs='EPSG:4326')

        # Check if we have population data
        has_population = population_raster and 'population' in gdf_floods.columns

        if has_population:
            print(f"Creating AFFECTED POPULATION density heatmap from {len(gdf_floods)} points...")
        else:
            print(f"Creating FLOOD PIXEL density heatmap from {len(gdf_floods)} points...")

        # Extract coordinates
        x_coords = gdf_floods.geometry.x.values
        y_coords = gdf_floods.geometry.y.values

        # Define extent for histogram
        x_min_flood = float(provenance_target.x.min())
        x_max_flood = float(provenance_target.x.max())
        y_min_flood = float(provenance_target.y.min())
        y_max_flood = float(provenance_target.y.max())

        # Create 2D histogram (density map)
        bins_x = 200
        bins_y = 200

        if has_population:
            # Population-weighted histogram
            weights = gdf_floods['population'].values
            heatmap, xedges, yedges = np.histogram2d(
                x_coords, y_coords,
                bins=[bins_x, bins_y],
                range=[[x_min_flood, x_max_flood], [y_min_flood, y_max_flood]],
                weights=weights
            )
        else:
            # Simple pixel count histogram
            heatmap, xedges, yedges = np.histogram2d(
                x_coords, y_coords,
                bins=[bins_x, bins_y],
                range=[[x_min_flood, x_max_flood], [y_min_flood, y_max_flood]]
            )

        # Apply Gaussian smoothing
        heatmap_smooth = gaussian_filter(heatmap.T, sigma=2)

        # Mask zeros
        heatmap_smooth = np.ma.masked_where(heatmap_smooth == 0, heatmap_smooth)

        # Colormap: purple for population, blue for flood pixels
        if has_population:
            # Purple colormap for population density
            colors_density = ['#00000000', '#E8DAEF', '#BB8FCE', '#8E44AD', '#5B2C6F']
        else:
            # Electric blue colormap for flood pixel density
            colors_density = ['#00000000', '#00BFFF', '#0080FF', '#0040FF']

        cmap_density = LinearSegmentedColormap.from_list('density', colors_density, N=100)

        # Plot density heatmap
        im_density = ax.imshow(
            heatmap_smooth,
            extent=[x_min_flood, x_max_flood, y_min_flood, y_max_flood],
            origin='lower',
            cmap=cmap_density,
            alpha=0.7,
            interpolation='bilinear',
            zorder=4
        )

        # Add colorbar for density
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im_density, cax=cax)

        if has_population:
            cbar.set_label('Affected Population Density', rotation=270, labelpad=20, fontsize=10)
            print(f"Plotted affected population density heatmap")
        else:
            cbar.set_label('Flood Point Density', rotation=270, labelpad=20, fontsize=10)
            print(f"Plotted flood pixel density heatmap")

    # Add boundaries
    gdf_aoi.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.8, zorder=5)
    gdf_admin1.boundary.plot(ax=ax, color='black', linewidth=0.8, alpha=0.6, zorder=5)

    # Set extent
    x_min, x_max = float(provenance_target.x.min()), float(provenance_target.x.max())
    y_min, y_max = float(provenance_target.y.min()), float(provenance_target.y.max())
    x_buffer = (x_max - x_min) * 0.02
    y_buffer = (y_max - y_min) * 0.02
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    # Title
    ax.set_title(
        f"Data Provenance Map - {target_date}\n(Latest observation date per pixel)",
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=0)

    # Legend
    legend_elements = [Patch(facecolor='#808080', label='No data')]
    for i, date in enumerate(unique_dates):
        legend_elements.append(
            Patch(facecolor=colors[i], label=str(pd.Timestamp(date))[:10])
        )

    ax.legend(
        handles=legend_elements,
        title='Latest Observation Date',
        loc='upper right',
        fontsize=10,
        title_fontsize=11,
        framealpha=0.95,
        edgecolor='black',
        fancybox=False
    )

    plt.tight_layout()

    # Save with descriptive filename
    density_type = "population" if (population_raster and len(flood_points) > 0 and 'population' in gpd.GeoDataFrame(flood_points).columns) else "flood"
    output_filename = f"{density_type}_provenance_{target_date.replace('-', '')}.png"
    output_path = f"{output_dir}/{output_filename}"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate flood provenance + density maps using N most recent observations"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format (e.g., 2025-10-27 or 2024-07-15)"
    )
    parser.add_argument(
        "--n-latest",
        type=int,
        default=3,
        help="Number of most recent observations to use (default: 3)"
    )
    parser.add_argument(
        "--iso3",
        type=str,
        default="JAM",
        help="ISO3 country code (default: JAM)"
    )
    parser.add_argument(
        "--population-raster",
        type=str,
        default=None,
        help="Path to GHSL population raster for affected population density (optional)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Directory for caching processed data (default: data/cache)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recompute and bypass cache"
    )

    args = parser.parse_args()

    main(
        args.end_date,
        args.n_latest,
        args.iso3,
        args.population_raster,
        args.cache_dir,
        use_cache=not args.no_cache
    )

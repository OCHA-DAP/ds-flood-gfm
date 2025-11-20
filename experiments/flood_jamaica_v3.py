"""
Jamaica Flood Monitoring - Provenance + Flood Density Visualization v3

Adds flood density visualization on top of v2 provenance map.

Key Logic for Flood Density:
- Only use flood pixels from their LATEST provenance date
- If Oct 27 has provenance coverage (whether flood=0 or flood=1), ignore Oct 22 data from that area
- This avoids counting "stale" flood pixels that may have receded when newer observations are available

Example:
- Green areas (Oct 22 provenance): Use flood pixels from Oct 22 data
- Orange areas (Oct 27 provenance): Use flood pixels from Oct 27 data (ignore Oct 22 even if it had floods)
- This way each location uses its MOST RECENT observation

Output:
- provenance_flood_v3_oct24.png: Provenance + flood density for Oct 24
- provenance_flood_v3_oct27.png: Provenance + flood density for Oct 27
"""

import pystac_client
import stackstac
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point

# Configuration
ISO3 = "JAM"
START_DATE = "2025-10-20"
END_DATE = "2025-10-30"
OUTPUT_DIR = "experiments"
ADMIN_FILE = "experiments/claude-tests/jamaica_admin_cleaned.geojson"

print("="*80)
print("JAMAICA FLOOD MONITORING - PROVENANCE + FLOOD DENSITY V3")
print("="*80)

# Load cleaned Jamaica boundaries (removes southern islands that distort extent)
gdf_aoi = gpd.read_file(ADMIN_FILE)
bbox = gdf_aoi.total_bounds
print(f"Loaded Jamaica boundaries from: {ADMIN_FILE}")
print(f"Bounding box: {bbox}")

# Load full admin1 boundaries for internal divisions
from fsspec.implementations.http import HTTPFileSystem
GLOBAL_ADM1 = "https://data.fieldmaps.io/edge-matched/humanitarian/intl/adm1_polygons.parquet"
filesystem = HTTPFileSystem()
filters = [("iso_3", "=", ISO3)]
gdf_admin1 = gpd.read_parquet(GLOBAL_ADM1, filesystem=filesystem, filters=filters)
print(f"Loaded {len(gdf_admin1)} admin1 divisions")

# Query STAC API for GFM data
stac_api = "https://stac.eodc.eu/api/v1"
client = pystac_client.Client.open(stac_api)
search = client.search(collections=["GFM"], bbox=bbox, datetime=f"{START_DATE}/{END_DATE}")
items = search.item_collection()
print(f"Found {len(items)} STAC items")

# Build xarray stack from STAC items
print("Building xarray stack...")
stack = stackstac.stack(items, epsg=4326)
stack_flood = stack.sel(band="ensemble_flood_extent")
stack_flood_clipped = stack_flood.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1]))

# Create daily composites (max value across overlapping tiles)
print("Creating daily composites...")
stack_flood_max = stack_flood_clipped.groupby("time.date").max()
stack_flood_max = stack_flood_max.rename({"date": "time"})
stack_flood_max["time"] = stack_flood_max.time.astype("datetime64[ns]")

dates = stack_flood_max.time.values
print(f"Dates with data: {[str(d)[:10] for d in dates]}")

# CRITICAL: Create mask of pixels that have data at least once
# This prevents forward-fill from filling spatial gaps (areas with no coverage)
print("\nCreating 'ever_has_data' mask...")
ever_has_data = xr.full_like(stack_flood_max.isel(time=0), fill_value=False, dtype=bool)
for i in range(len(dates)):
    has_data_this_time = ~np.isnan(stack_flood_max.isel(time=i))
    ever_has_data = ever_has_data | has_data_this_time

pixels_with_data = ever_has_data.sum().values
pixels_no_data = (~ever_has_data).sum().values
print(f"  Pixels with data at least once: {pixels_with_data:,}")
print(f"  Pixels that NEVER have data: {pixels_no_data:,}")
print(f"  Coverage: {100 * pixels_with_data / (pixels_with_data + pixels_no_data):.1f}%")

# Track provenance (which date each pixel came from)
print("\nTracking provenance...")
provenance = xr.full_like(stack_flood_max, fill_value=np.datetime64('NaT'), dtype='datetime64[ns]')
for i, time_val in enumerate(dates):
    has_data = ~np.isnan(stack_flood_max.isel(time=i))
    provenance[i] = xr.where(has_data, time_val, np.datetime64('NaT'))

# Forward-fill to create complete coverage
print("Forward-filling flood data and provenance...")
flood_filled = stack_flood_max.ffill(dim="time")
provenance_filled = provenance.ffill(dim="time")

# CRITICAL FIX: Mask out pixels that never had data
# Without this, forward-fill incorrectly assigns dates to areas with no coverage
print("Applying spatial mask to provenance...")
for i in range(len(dates)):
    provenance_filled[i] = xr.where(ever_has_data, provenance_filled[i], np.datetime64('NaT'))

print("Setup complete. Ready to create provenance + flood density maps.\n")


def create_provenance_flood_map(target_date, output_name):
    """
    Create a provenance map with flood density overlay.

    Key: Only uses flood pixels from their LATEST provenance date.
    If Oct 27 covers an area, we ignore Oct 22 data from that area entirely.

    Parameters:
    -----------
    target_date : str
        Target date in YYYY-MM-DD format
    output_name : str
        Output filename (without path)
    """
    print(f"\n{'='*80}")
    print(f"Creating provenance + flood density map for {target_date}")
    print(f"{'='*80}")

    # Extract data for target date and compute
    flood_target = flood_filled.sel(time=target_date).compute()
    provenance_target = provenance_filled.sel(time=target_date).compute()

    # Get unique provenance dates
    unique_dates = pd.unique(provenance_target.values.ravel())
    unique_dates = unique_dates[~pd.isna(unique_dates)]
    unique_dates = np.sort(unique_dates)

    # Print provenance breakdown
    print(f"Provenance breakdown:")
    total_valid = np.sum(~pd.isna(provenance_target.values))
    for date in unique_dates:
        count = np.sum(provenance_target.values == date)
        pct = 100 * count / total_valid
        print(f"  {str(pd.Timestamp(date))[:10]}: {count:,} pixels ({pct:.1f}%)")

    no_data_count = np.sum(pd.isna(provenance_target.values))
    pct_no_data = 100 * no_data_count / provenance_target.size
    print(f"  NO DATA: {no_data_count:,} pixels ({pct_no_data:.1f}% of bbox)")

    # CRITICAL: Extract flood pixels only from their LATEST provenance date
    print("\nExtracting flood pixels by provenance date...")
    flood_points = []

    for date in unique_dates:
        # Get the original (non-forward-filled) flood data for this date
        original_flood = stack_flood_max.sel(time=date).compute()

        # Mask: pixels where provenance == this date AND flood == 1
        is_this_provenance = (provenance_target.values == date)
        is_flooded = (original_flood.values == 1)

        flood_mask = is_this_provenance & is_flooded
        flood_count = np.sum(flood_mask)

        print(f"  {str(pd.Timestamp(date))[:10]}: {flood_count:,} flood pixels")

        # Convert flood pixels to points
        if flood_count > 0:
            y_coords, x_coords = np.where(flood_mask)

            # Convert array indices to geographic coordinates
            x_geo = provenance_target.x.values[x_coords]
            y_geo = provenance_target.y.values[y_coords]

            # Create points with provenance date
            for x, y in zip(x_geo, y_geo):
                flood_points.append({
                    'geometry': Point(x, y),
                    'date': str(pd.Timestamp(date))[:10],
                    'lon': x,
                    'lat': y
                })

    print(f"\nTotal unique flood pixels (latest provenance only): {len(flood_points):,}")

    # Create indexed array for provenance visualization
    date_to_idx = {date: i for i, date in enumerate(unique_dates)}
    provenance_indexed = np.full(provenance_target.shape, -1, dtype=np.int32)
    for date, idx in date_to_idx.items():
        provenance_indexed[provenance_target.values == date] = idx

    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Define colors - oldest to most recent: green -> yellow -> orange
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

    # Plot provenance data
    prov_masked = np.ma.masked_where(provenance_indexed == -1, provenance_indexed)
    im = ax.imshow(
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

    # Plot flood points as small dots
    if len(flood_points) > 0:
        gdf_floods = gpd.GeoDataFrame(flood_points, crs='EPSG:4326')
        gdf_floods.plot(
            ax=ax,
            color='blue',
            markersize=0.5,
            alpha=0.6,
            zorder=4
        )
        print(f"Plotted {len(gdf_floods)} flood points")

    # Add Jamaica administrative boundaries in black
    gdf_aoi.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.8, zorder=5)

    # Add admin1 internal boundaries in black (lighter)
    gdf_admin1.boundary.plot(ax=ax, color='black', linewidth=0.8, alpha=0.6, zorder=5)

    # Set extent - tighter to the data
    x_min, x_max = float(provenance_target.x.min()), float(provenance_target.x.max())
    y_min, y_max = float(provenance_target.y.min()), float(provenance_target.y.max())
    x_buffer = (x_max - x_min) * 0.02
    y_buffer = (y_max - y_min) * 0.02
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    # Simple title
    ax.set_title(
        f"Data Provenance Map\n(Latest observation date per pixel)",
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=0)

    # Legend - clean design matching reference
    legend_elements = [Patch(facecolor='#808080', label='No data')]
    for i, date in enumerate(unique_dates):
        legend_elements.append(
            Patch(facecolor=colors[i], label=str(pd.Timestamp(date))[:10])
        )

    # Add flood indicator to legend
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=6, alpha=0.6, label='Flood pixels')
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

    # Save figure
    output_path = f"{OUTPUT_DIR}/{output_name}"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


# Create provenance + flood density maps for Oct 24 and Oct 27
create_provenance_flood_map("2025-10-24", "provenance_flood_v3_oct24.png")
create_provenance_flood_map("2025-10-27", "provenance_flood_v3_oct27.png")

print("\n" + "="*80)
print("COMPLETE - PROVENANCE + FLOOD DENSITY MAPS V3 CREATED")
print("="*80)
print("\nOutput files:")
print(f"  - {OUTPUT_DIR}/provenance_flood_v3_oct24.png")
print(f"  - {OUTPUT_DIR}/provenance_flood_v3_oct27.png")
print("\nV3: Adds flood pixel visualization (blue dots)")
print("    Only uses flood pixels from LATEST provenance date (no double-counting)")

"""
Jamaica Flood Monitoring - Provenance Visualization v1

This script generates provenance maps showing which date each pixel's flood data
came from when using forward-fill compositing for real-time monitoring.

Key Features:
- Downloads GFM flood data for Jamaica from STAC API
- Creates daily composites from overlapping satellite tiles
- Tracks data provenance (which date each pixel originated from)
- Forward-fills to create complete coverage for monitoring
- Correctly masks pixels that never had satellite coverage
- Uses cleaned Jamaica boundaries (removes distorting southern islands)

Critical Fix:
The 'ever_has_data' mask prevents forward-fill from incorrectly assigning
provenance dates to pixels that have NO satellite coverage on any date.

Output:
- provenance_oct24.png: Shows Oct 24 data provenance
- provenance_oct27.png: Shows Oct 27 data provenance
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
from shapely.geometry import shape
from rasterio import features
from rasterio.transform import from_bounds

# Configuration
ISO3 = "JAM"
START_DATE = "2025-10-20"
END_DATE = "2025-10-30"
OUTPUT_DIR = "experiments"
ADMIN_FILE = "experiments/claude-tests/jamaica_admin_cleaned.geojson"

print("="*80)
print("JAMAICA FLOOD MONITORING - PROVENANCE VISUALIZATION V1")
print("="*80)

# Load cleaned Jamaica boundaries (removes southern islands that distort extent)
gdf_aoi = gpd.read_file(ADMIN_FILE)
bbox = gdf_aoi.total_bounds
print(f"Loaded Jamaica boundaries from: {ADMIN_FILE}")
print(f"Bounding box: {bbox}")

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

print("Setup complete. Ready to create provenance maps.\n")


def create_provenance_map(target_date, output_name):
    """
    Create a provenance map showing which date each pixel's data originated from.

    Parameters:
    -----------
    target_date : str
        Target date in YYYY-MM-DD format
    output_name : str
        Output filename (without path)
    """
    print(f"\n{'='*80}")
    print(f"Creating provenance map for {target_date}")
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

    # Create indexed array for visualization
    date_to_idx = {date: i for i, date in enumerate(unique_dates)}
    provenance_indexed = np.full(provenance_target.shape, -1, dtype=np.int32)
    for date, idx in date_to_idx.items():
        provenance_indexed[provenance_target.values == date] = idx

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Define colors for each date
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(unique_dates)]
    cmap_prov = ListedColormap(colors)

    # Plot provenance - mask out -1 values (no data)
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
        vmax=len(unique_dates)-1
    )

    # Extract and plot polygons for each provenance date
    transform = from_bounds(
        float(provenance_target.x.min()),
        float(provenance_target.y.min()),
        float(provenance_target.x.max()),
        float(provenance_target.y.max()),
        provenance_indexed.shape[1],
        provenance_indexed.shape[0]
    )

    print("Extracting and labeling polygons...")
    for date, idx in date_to_idx.items():
        mask = (provenance_indexed == idx).astype(np.uint8)
        shapes_gen = features.shapes(mask, mask=mask, transform=transform)

        polys = [shape(geom) for geom, value in shapes_gen if value == 1]

        if polys:
            gdf_date = gpd.GeoDataFrame(
                {'date': [str(pd.Timestamp(date))[:10]] * len(polys)},
                geometry=polys,
                crs='EPSG:4326'
            )
            gdf_dissolved = gdf_date.dissolve(by='date')

            # Draw boundary around provenance region
            gdf_dissolved.boundary.plot(ax=ax, color='black', linewidth=3, alpha=0.8)

            # Add label with date and pixel count
            geom = gdf_dissolved.geometry.iloc[0]
            if geom.geom_type == 'MultiPolygon':
                largest_poly = max(geom.geoms, key=lambda p: p.area)
                centroid = largest_poly.centroid
            else:
                centroid = geom.centroid

            date_str = str(pd.Timestamp(date))[:10]
            pixel_count = np.sum(provenance_target.values == date)

            ax.text(
                centroid.x, centroid.y,
                f"{date_str}\n({pixel_count:,} px)",
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle='round,pad=0.6',
                    facecolor='white',
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.95
                )
            )

    # Add Jamaica administrative boundaries
    gdf_aoi.boundary.plot(ax=ax, color='grey', linewidth=1, alpha=0.5, linestyle='--')

    # Set extent with buffer
    x_min, x_max = float(provenance_target.x.min()), float(provenance_target.x.max())
    y_min, y_max = float(provenance_target.y.min()), float(provenance_target.y.max())
    x_buffer = (x_max - x_min) * 0.05
    y_buffer = (y_max - y_min) * 0.05
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    # Title
    ax.set_title(
        f"Data Provenance Map - {target_date}\n" +
        f"Shows acquisition date for each pixel\n" +
        f"White areas = No satellite coverage on any date",
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Legend
    legend_elements = [
        Patch(facecolor=colors[i], label=str(pd.Timestamp(date))[:10])
        for i, date in enumerate(unique_dates)
    ]
    legend_elements.append(Patch(facecolor='white', edgecolor='grey', label='No Data Coverage'))

    ax.legend(
        handles=legend_elements,
        title='Data Status',
        loc='upper left',
        fontsize=11,
        title_fontsize=12,
        framealpha=0.95
    )

    # Note box
    textstr = (
        f"White areas have NO satellite data\n"
        f"Colored areas show when flood data was acquired\n"
        f"Older dates = Forward-filled (no fresh observation)\n"
        f"Coverage: {100 - pct_no_data:.1f}% of bbox has data"
    )
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.8, edgecolor='orange', linewidth=2)
    ax.text(
        0.02, 0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=props
    )

    plt.tight_layout()

    # Save figure
    output_path = f"{OUTPUT_DIR}/{output_name}"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# Create provenance maps for Oct 24 and Oct 27
create_provenance_map("2025-10-24", "provenance_oct24.png")
create_provenance_map("2025-10-27", "provenance_oct27.png")

print("\n" + "="*80)
print("COMPLETE - PROVENANCE MAPS CREATED")
print("="*80)
print("\nOutput files:")
print(f"  - {OUTPUT_DIR}/provenance_oct24.png")
print(f"  - {OUTPUT_DIR}/provenance_oct27.png")
print("\nKey insight: White areas correctly show regions with NO satellite coverage")
print("            Forward-filled areas show old dates (not fresh observations)")

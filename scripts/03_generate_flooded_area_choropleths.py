#!/usr/bin/env python3
"""
Generate flooded area choropleths (in m²) from cached flood points.
Creates 2 maps:
1. Latest composite (343 pixels from latest provenance)
2. Cumulative (419 pixels from all 3 dates combined)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from pathlib import Path
from shapely.geometry import Point
from adjustText import adjust_text
import exactextract
import tempfile
import rasterio
from rasterio.transform import from_bounds
import os

# Import custom geo utils
from ds_flood_gfm.geo_utils import load_admin_from_blob

# Constants
PIXEL_AREA_M2 = 400  # 20m x 20m = 400 m²
ISO3 = "JAM"
ADM_LEVEL = 3
OUTPUT_DIR = "experiments"

def load_flood_points_from_cache(cache_path):
    """Load flood points from cache parquet file."""
    gdf = gpd.read_parquet(cache_path)
    flood_points = gdf.to_dict('records')
    for fp in flood_points:
        fp['geometry'] = Point(fp['lon'], fp['lat'])
    return flood_points

def load_provenance_raster_from_cache(cache_dir):
    """Load provenance raster and metadata from cache."""
    with rasterio.open(cache_dir / "provenance.tif") as src:
        provenance_indexed = src.read(1)
        transform = src.transform

    # Load metadata for dates
    import json
    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)
    unique_dates = [np.datetime64(d) for d in metadata['unique_dates']]

    return provenance_indexed, transform, unique_dates, metadata

def create_flooded_area_choropleth(flood_points, provenance_indexed, transform, unique_dates,
                                    metadata, mode_name, output_filename):
    """
    Create choropleth showing flooded area (m²) per ADM3 division.

    Parameters:
    -----------
    flood_points : list
        List of flood point dictionaries with geometry
    provenance_indexed : ndarray
        Provenance raster (indexed by date)
    transform : affine.Affine
        Raster transform
    unique_dates : list
        List of unique dates
    metadata : dict
        Cache metadata
    mode_name : str
        'latest' or 'cumulative'
    output_filename : str
        Output PNG filename
    """
    print(f"\n{'='*80}")
    print(f"GENERATING FLOODED AREA CHOROPLETH ({mode_name.upper()})")
    print(f"{'='*80}")

    # Load admin boundaries
    print(f"Loading ADM{ADM_LEVEL} boundaries from blob...")
    gdf_admin = load_admin_from_blob(ISO3, ADM_LEVEL, stage="dev")
    print(f"Using ADM{ADM_LEVEL} boundaries: {len(gdf_admin)} divisions")

    # Load ADM1 for overlay
    gdf_admin1 = load_admin_from_blob(ISO3, 1, stage="dev")
    print(f"Loading ADM1 for overlay: {len(gdf_admin1)} divisions")

    # Count flood pixels per ADM3
    print(f"Counting flood pixels per ADM{ADM_LEVEL} ({len(flood_points)} total pixels)...")

    # Create GeoDataFrame from flood points
    gdf_floods = gpd.GeoDataFrame(flood_points, crs='EPSG:4326')

    # Spatial join to find which ADM3 each flood pixel belongs to
    gdf_joined = gpd.sjoin(gdf_floods, gdf_admin, how='left', predicate='within')

    # Count pixels per ADM3
    adm_col = f'adm{ADM_LEVEL}_id'
    pixel_counts = gdf_joined.groupby(adm_col).size()

    # Calculate flooded area in km²
    flooded_area_km2 = (pixel_counts * PIXEL_AREA_M2) / 1_000_000  # Convert m² to km²

    # Merge with admin boundaries
    gdf_choropleth = gdf_admin.copy()
    gdf_choropleth['flooded_area_km2'] = gdf_choropleth[adm_col].map(flooded_area_km2).fillna(0)

    # Get divisions with flooding
    affected_divs = gdf_choropleth[gdf_choropleth['flooded_area_km2'] > 0].copy()
    print(f"Divisions with flooded area: {len(affected_divs)}/{len(gdf_admin)}")
    print(f"Total flooded area: {affected_divs['flooded_area_km2'].sum():.4f} km²")

    # Extract modal provenance for ADM3 boundaries using exactextract
    print(f"  Extracting modal provenance for ADM{ADM_LEVEL} boundaries (exactextract)...")

    # Map provenance index to colors (darker yellow for better visibility)
    colors = ['#91cf60', '#f0c040', '#fc8d59'][:len(unique_dates)]
    date_to_color = {i: colors[i] for i in range(len(unique_dates))}

    # Write provenance raster to temp file for exactextract
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        tmp_raster_path = tmp.name

    height, width = provenance_indexed.shape
    with rasterio.open(tmp_raster_path, 'w', driver='GTiff', height=height, width=width,
                     count=1, dtype='int16', crs='EPSG:4326', transform=transform) as dst:
        prov_data = provenance_indexed.astype('int16')
        dst.write(prov_data, 1)

    # Run exactextract
    modal_prov = exactextract.exact_extract(
        tmp_raster_path,
        gdf_admin,
        ['mode'],
        include_cols=[adm_col],
        output='pandas'
    )

    os.unlink(tmp_raster_path)

    # Map modal provenance to colors
    prov_colors = []
    for _, row in modal_prov.iterrows():
        mode_val = row['mode']
        if pd.isna(mode_val) or mode_val == -1:
            prov_colors.append('lightgray')
        else:
            prov_colors.append(date_to_color.get(int(mode_val), 'lightgray'))

    gdf_admin['prov_color'] = prov_colors

    # Dissolve by provenance color
    print("  Dissolving by provenance color...")
    dissolved_provenance = gdf_admin.dissolve(by='prov_color', as_index=False)

    # Create color mapping
    color_to_date = {color: unique_dates[i] for i, color in date_to_color.items()}

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Plot no-data areas (grey fill)
    no_data_polys = dissolved_provenance[dissolved_provenance['prov_color'] == 'lightgray']
    if len(no_data_polys) > 0:
        no_data_polys.plot(ax=ax, facecolor='lightgrey', edgecolor='none', alpha=0.7, zorder=1)

    # Plot choropleth (ONLY polygons with flooded area > 0 to avoid covering grey no-data)
    max_area = gdf_choropleth['flooded_area_km2'].max()

    if max_area > 0:
        # Filter to only show polygons with flooded area
        gdf_to_plot = gdf_choropleth[gdf_choropleth['flooded_area_km2'] > 0].copy()

        # Custom blue colormap with white for minimum (water/flood theme)
        colors_list = ['white', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
        cmap_custom = LinearSegmentedColormap.from_list('white_blues', colors_list, N=256)

        if max_area < 0.05:  # Use binned scale for very small areas (< 0.05 km²)
            bins = [0.0001, 0.001, 0.005, 0.01, 0.02, max_area + 0.001]
            norm = BoundaryNorm(bins, ncolors=256)

            gdf_to_plot.plot(
                column='flooded_area_km2',
                ax=ax,
                cmap=cmap_custom,
                edgecolor='#c0c0c0',
                linewidth=0.15,
                legend=True,
                norm=norm,
                legend_kwds={
                    'label': 'Flooded Area (km²)',
                    'orientation': 'vertical',
                    'shrink': 0.6
                },
                zorder=2
            )
        else:
            gdf_to_plot.plot(
                column='flooded_area_km2',
                ax=ax,
                cmap=cmap_custom,
                edgecolor='#c0c0c0',
                linewidth=0.15,
                legend=True,
                vmin=0,
                vmax=max_area,
                legend_kwds={
                    'label': 'Flooded Area (km²)',
                    'orientation': 'vertical',
                    'shrink': 0.6
                },
                zorder=2
            )

    # Plot dissolved provenance boundaries
    legend_elements = []
    for idx, row in dissolved_provenance.iterrows():
        color = row['prov_color']
        if color != 'lightgray':
            gpd.GeoSeries([row.geometry]).boundary.plot(
                ax=ax, edgecolor=color, linewidth=2.5, alpha=0.9, zorder=3
            )

    # Build legend in chronological order
    sorted_dates = sorted(unique_dates)
    for date in sorted_dates:
        date_idx = unique_dates.index(date)
        color = date_to_color.get(date_idx)
        if color:
            date_str = str(date)[:10]
            legend_elements.append(Patch(facecolor='none', edgecolor=color, linewidth=2.5, label=date_str))

    if len(no_data_polys) > 0:
        legend_elements.append(Patch(facecolor='lightgrey', alpha=0.7, edgecolor='none', label='No data'))

    # Plot all ADM3 boundaries
    gdf_admin.boundary.plot(ax=ax, edgecolor='#c0c0c0', linewidth=0.15, alpha=0.4, zorder=2)

    print(f"  Provenance boundaries (dissolved) and ADM{ADM_LEVEL} borders plotted")

    # Add ADM1 boundaries
    gdf_admin1.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add labels for top affected divisions (with halo effect)
    top_n = min(5, len(affected_divs))
    if top_n > 0:
        adm_name_col = f'adm{ADM_LEVEL}_name'
        top_affected = affected_divs.nlargest(top_n, 'flooded_area_km2')
        texts = []
        for idx, row in top_affected.iterrows():
            centroid = row.geometry.centroid
            text = ax.text(
                centroid.x, centroid.y,
                f"{row[adm_name_col]}\n({row['flooded_area_km2']:.2f} km²)",
                fontsize=8,
                weight='bold',
                ha='center', va='center',
                color='black',
                path_effects=[withStroke(linewidth=3, foreground='white', alpha=0.7)]
            )
            texts.append(text)

        adjust_text(texts, ax=ax,
                   arrowprops=dict(arrowstyle='->', color='black', lw=0.5, alpha=0.6))

    # Set axis limits
    x_min, x_max = metadata['x_min'], metadata['x_max']
    y_min, y_max = metadata['y_min'], metadata['y_max']
    x_buffer = (x_max - x_min) * 0.02
    y_buffer = (y_max - y_min) * 0.02
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    ax.set_title(f'Flooded Area by ADM{ADM_LEVEL} ({mode_name.capitalize()})', fontsize=14, pad=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Add provenance legend
    ax.legend(
        handles=legend_elements,
        title='Data Provenance',
        loc='lower right',
        fontsize=9,
        title_fontsize=10,
        framealpha=0.9,
        edgecolor='black'
    )

    plt.tight_layout()

    # Save
    output_path = f"{OUTPUT_DIR}/{output_filename}"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"{'='*80}\n")

def main():
    # Paths to cached data
    latest_cache = Path("data/cache/JAM_latest_cea15136")
    cumulative_cache = Path("data/cache/JAM_cumulative_30dc22f1")

    # Load latest flood points
    print("Loading LATEST flood points...")
    latest_points = load_flood_points_from_cache(latest_cache / "flood_points.parquet")
    print(f"  Loaded {len(latest_points)} flood pixels")

    # Load cumulative flood points
    print("\nLoading CUMULATIVE flood points...")
    cumulative_points = load_flood_points_from_cache(cumulative_cache / "flood_points.parquet")
    print(f"  Loaded {len(cumulative_points)} flood pixels")

    # Load provenance raster (same for both)
    prov_indexed, transform, unique_dates, metadata = load_provenance_raster_from_cache(latest_cache)

    # Generate flooded area choropleth (LATEST)
    # Get latest date from metadata for filename
    latest_date = metadata['unique_dates'][-1] if 'unique_dates' in metadata else "20251027"
    latest_date_str = latest_date.replace('-', '') if isinstance(latest_date, str) else str(latest_date)[:10].replace('-', '')

    create_flooded_area_choropleth(
        latest_points, prov_indexed, transform, unique_dates, metadata,
        mode_name="latest",
        output_filename=f"{ISO3}_choropleth_flooded_area_latest_{latest_date_str}.png"
    )

    # Generate flooded area choropleth (CUMULATIVE)
    create_flooded_area_choropleth(
        cumulative_points, prov_indexed, transform, unique_dates, metadata,
        mode_name="cumulative",
        output_filename=f"{ISO3}_choropleth_flooded_area_cumulative_{latest_date_str}.png"
    )

    print("\n" + "="*80)
    print("FLOODED AREA CHOROPLETHS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

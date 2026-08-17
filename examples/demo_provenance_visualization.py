"""
Demo script to visualize provenance tracking with GFM data.

This script demonstrates how the lightweight provenance approach works:
1. Query GFM STAC for a small AOI
2. Create flood composite (stays lazy)
3. Create provenance raster (stays lazy)
4. Compute and visualize provenance dates
5. Show how polygons can sample provenance

Usage:
    uv run python demo_provenance_visualization.py
"""

import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import geopandas as gpd
import ocha_stratus as stratus
from dotenv import load_dotenv

from src.ds_flood_gfm.datasources.gfm import (
    query_gfm_stac,
    create_flood_composite,
    create_provenance_raster,
    add_modal_provenance_to_admin,
    raster_to_polygons,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def generate_date_colors(n_dates):
    """Generate colors from red (oldest) to green (newest)."""
    cmap = plt.cm.RdYlGn
    return [mcolors.rgb2hex(cmap(i / (n_dates - 1))) for i in range(n_dates)]


def main():
    # ========== CONFIGURATION ==========
    # Using Jamaica for a smaller AOI
    ISO3 = "JAM"
    TARGET_DATE = "2025-10-28"
    N_IMAGES = 3
    N_SEARCH = -15  # Negative = backward scan

    scan_direction = "forward" if N_SEARCH > 0 else "backward"
    logger.info("=" * 70)
    logger.info("PROVENANCE VISUALIZATION DEMO")
    logger.info("=" * 70)
    logger.info(f"Country: {ISO3}")
    logger.info(f"Target date: {TARGET_DATE}")
    logger.info(f"Number of images: {N_IMAGES}")
    logger.info(f"Search window: {N_SEARCH} days ({scan_direction})")
    logger.info("=" * 70)

    # ========== STEP 1: Get AOI and Query STAC ==========
    logger.info("\n[STEP 1] Loading admin boundaries and querying STAC...")
    gdf_admin_country = stratus.codab.load_codab_from_fieldmaps(ISO3, 0)
    gdf_admin = stratus.codab.load_codab_from_fieldmaps(ISO3, 3)
    bbox = gdf_admin_country.total_bounds
    logger.info(f"  Bbox: {bbox}")

    items = query_gfm_stac(bbox, TARGET_DATE, N_SEARCH)
    if len(items) == 0:
        logger.error("No STAC items found!")
        return

    # ========== STEP 2: Create Flood Composite ==========
    logger.info("\n[STEP 2] Creating flood composite (lazy)...")
    flood_composite, unique_dates, stack_flood_max = create_flood_composite(
        items, bbox, N_IMAGES, mode='cumulative', n_search=N_SEARCH, return_stack=True
    )
    logger.info(f"  Flood composite shape: {flood_composite.shape}")
    logger.info(f"  Unique dates: {[str(d)[:10] for d in unique_dates]}")
    logger.info("  ⚠️  Still lazy - no computation yet!")

    # ========== STEP 3: Create Provenance Raster ==========
    logger.info("\n[STEP 3] Creating provenance raster (lazy)...")
    provenance_idx, date_mapping = create_provenance_raster(stack_flood_max, unique_dates)
    logger.info(f"  Provenance shape: {provenance_idx.shape}")
    logger.info(f"  Date mapping: {date_mapping}")
    logger.info("  ⚠️  Still lazy - no computation yet!")

    # ========== STEP 4: Compute and Visualize ==========
    logger.info("\n[STEP 4] Computing provenance raster...")
    logger.info("  ⚡ COMPUTATION HAPPENS HERE ⚡")
    prov_computed = provenance_idx.compute()
    logger.info(f"  ✅ Computed! Shape: {prov_computed.shape}")

    # Get statistics
    unique_values, counts = np.unique(prov_computed.values[~np.isnan(prov_computed.values)], return_counts=True)
    logger.info("\n  Provenance breakdown:")
    for val, count in zip(unique_values, counts):
        date_str = date_mapping.get(int(val), "Unknown")
        pct = 100 * count / counts.sum()
        logger.info(f"    {date_str}: {count:,} pixels ({pct:.1f}%)")

    # ========== STEP 5: Create Flood Polygons ==========
    logger.info("\n[STEP 5] Creating flood polygons...")
    logger.info("  ⚡ Flood composite computed here ⚡")
    flood_polygons = raster_to_polygons(flood_composite)
    logger.info(f"  ✅ Created {len(flood_polygons)} polygons")

    # ========== STEP 6: Sample Provenance at Polygon Centroids ==========
    logger.info("\n[STEP 6] Sampling provenance at polygon centroids...")
    provenance_dates = []
    for idx, row in flood_polygons.iterrows():
        centroid = row.geometry.centroid
        # Sample the already-computed provenance raster
        prov_val = int(prov_computed.sel(x=centroid.x, y=centroid.y, method='nearest').values)
        date_str = date_mapping.get(prov_val, "Unknown")
        provenance_dates.append(date_str)

    flood_polygons['prov_date'] = provenance_dates
    logger.info(f"  ✅ Added provenance to {len(flood_polygons)} polygons")

    # Show sample
    logger.info("\n  Sample polygons with provenance:")
    logger.info(flood_polygons[['prov_date', 'geometry']].head().to_string())

    # ========== STEP 7: Add Modal Provenance to Admin Boundaries ==========
    logger.info("\n[STEP 7] Adding modal provenance to admin boundaries...")
    gdf_admin_prov = add_modal_provenance_to_admin(
        gdf_admin,
        prov_computed,
        date_mapping
    )

    # ========== STEP 8: Visualize ==========
    logger.info("\n[STEP 8] Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # --- Subplot 1 (Top Left): Provenance Raster ---
    ax1 = axes[0, 0]
    n_dates = len(unique_dates)
    colors = generate_date_colors(n_dates)
    cmap = mcolors.ListedColormap(colors)

    # Mask out no-data areas
    prov_masked = np.ma.masked_where(prov_computed.values == -1, prov_computed.values)

    im = ax1.imshow(
        prov_masked,
        cmap=cmap,
        extent=[
            float(prov_computed.x.min()),
            float(prov_computed.x.max()),
            float(prov_computed.y.min()),
            float(prov_computed.y.max()),
        ],
        origin='upper',
        vmin=0,
        vmax=n_dates - 1,
        interpolation='nearest'
    )

    # Add admin boundary
    gdf_admin_country.boundary.plot(ax=ax1, color='black', linewidth=2)

    ax1.set_title(f'Provenance: Last Observation Date\n{ISO3}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # Create legend
    legend_elements = []
    for i, date in enumerate(unique_dates):
        date_str = str(pd.Timestamp(date))[:10]
        legend_elements.append(
            Patch(facecolor=colors[i], label=date_str)
        )
    ax1.legend(handles=legend_elements, loc='upper right', title='Last Observation')

    # --- Subplot 2 (Top Right): Admin Modal Provenance ---
    ax2 = axes[0, 1]

    # Plot admin boundaries colored by modal provenance
    for idx, row in gdf_admin_prov.iterrows():
        prov_idx = row['prov_idx']
        if prov_idx >= 0 and prov_idx < len(colors):
            color = colors[prov_idx]
        else:
            color = 'lightgray'

        gpd.GeoSeries([row.geometry]).plot(
            ax=ax2,
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.7
        )

    ax2.set_title(f'Modal Provenance by Admin (Level 3)\n{len(gdf_admin_prov)} admin units', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.legend(handles=legend_elements, loc='upper right', title='Modal Date')

    # --- Subplot 3 (Bottom Left): Flood Composite ---
    ax3 = axes[1, 0]
    flood_computed = flood_composite.compute()
    ax3.imshow(
        flood_computed.values,
        cmap='Blues',
        extent=[
            float(flood_computed.x.min()),
            float(flood_computed.x.max()),
            float(flood_computed.y.min()),
            float(flood_computed.y.max()),
        ],
        origin='upper',
        vmin=0,
        vmax=1,
        interpolation='nearest'
    )
    gdf_admin_country.boundary.plot(ax=ax3, color='black', linewidth=2)
    ax3.set_title(f'Flood Composite (Cumulative)\n{len(flood_polygons)} polygons', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')

    # --- Subplot 4 (Bottom Right): Flood Polygons with Provenance ---
    ax4 = axes[1, 1]

    # Plot admin boundary
    gdf_admin_country.boundary.plot(ax=ax4, color='black', linewidth=2, zorder=1)

    # Create color mapping for polygons
    date_to_color = {date_str: colors[i] for i, date_str in enumerate([date_mapping[i] for i in range(len(unique_dates))])}

    # Plot polygons colored by provenance
    for idx, row in flood_polygons.iterrows():
        color = date_to_color.get(row['prov_date'], 'gray')
        gpd.GeoSeries([row.geometry]).plot(
            ax=ax4,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7,
            zorder=2
        )

    ax4.set_title(f'Flood Polygons Colored by Provenance\n{len(flood_polygons)} polygons', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.legend(handles=legend_elements, loc='upper right', title='Provenance Date')

    plt.tight_layout()
    date_str = TARGET_DATE.replace('-', '')
    output_path = f"outputs/provenance_demo_{ISO3}_{date_str}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\n✅ Visualization saved to: {output_path}")
    logger.info("\nDone! Check the output image to see provenance tracking in action.")

    # ========== SUMMARY ==========
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total dates used: {len(unique_dates)}")
    logger.info(f"Flood polygons created: {len(flood_polygons)}")
    logger.info(f"Provenance raster shape: {prov_computed.shape}")
    logger.info(f"Total pixels computed: {prov_computed.size:,}")
    logger.info("\nKey insight: Provenance stayed LAZY until we called .compute()")
    logger.info("Only ~2 computation steps instead of many in old approach!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

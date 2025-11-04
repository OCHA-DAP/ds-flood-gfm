"""Shared GFM Processing Utilities.

Common functions for STAC querying, temporal compositing, and raster processing
that can be used across multiple scripts in the GFM pipeline.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import stackstac
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)


def query_gfm_stac(bbox: list, end_date: str, n_search: int = 15) -> list:
    """Query GFM STAC API for flood data.
    
    Parameters
    ----------
    bbox : list
        Bounding box [west, south, east, north].
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
        
    Returns
    -------
    list
        STAC items.
    """
    logger.info("Querying GFM STAC API...")
    logger.info(f"  Looking {n_search} days back from {end_date}")
    logger.info(f"  Bbox: {bbox}")
    
    stac_api = "https://stac.eodc.eu/api/v1"
    client = pystac_client.Client.open(stac_api)

    # Parse dates - look back 15 days to find available data
    _end_date = datetime.strptime(end_date, "%Y-%m-%d")
    search_start_date = _end_date - timedelta(days=15)
    start_date = search_start_date.strftime("%Y-%m-%d")
    
    search = client.search(
        collections=["GFM"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}"
    )
    
    items = search.item_collection()
    logger.info(f"  Found {len(items)} STAC items")
    
    return items


def create_flood_composite(items: list, bbox: list, n_latest: int, mode: str = "latest") -> tuple[xr.DataArray, list]:
    """Create flood composite from STAC items.
    
    Parameters
    ----------
    items : list
        STAC items.
    bbox : list
        Bounding box [west, south, east, north].
    mode : str, default "latest"
        Compositing mode ('latest' or 'cumulative').
        
    Returns
    -------
    tuple[xr.DataArray, list]
        Flood composite and unique dates.
    """
    logger.info(f"Creating {mode} flood composite with {n_latest} most recent dates...")
    
    # Extract unique dates from STAC items (fast - no raster loading!)
    all_dates_set = set()
    for item in items:
        dt = pd.to_datetime(item.datetime)
        all_dates_set.add(dt.date())

    all_dates = sorted(list(all_dates_set))
    print(f"All dates found: {[str(d) for d in all_dates]}")

    if len(all_dates) == 0:
        print("ERROR: No valid dates in STAC items!")
        return

    # Select only the N most recent dates
    dates_to_use = all_dates[-n_latest:] if len(all_dates) >= n_latest else all_dates
    print(
        f"Using {len(dates_to_use)} most recent dates: {[str(d) for d in dates_to_use]}"
    )

    # Convert to numpy datetime64 for cache key generation
    dates = np.array([np.datetime64(d) for d in dates_to_use])
    
    # Build xarray stack
    logger.info("  Building xarray stack...")
    stack = stackstac.stack(items, epsg=4326)
    stack_flood = stack.sel(band="ensemble_flood_extent")
    stack_flood_clipped = stack_flood.sel(
        x=slice(bbox[0], bbox[2]), 
        y=slice(bbox[3], bbox[1])
    )
    
    # Create daily composites
    logger.info("  Creating daily composites...")
    stack_flood_max = stack_flood_clipped.groupby("time.date").max()
    stack_flood_max = stack_flood_max.rename({"date": "time"})
    stack_flood_max["time"] = stack_flood_max.time.astype("datetime64[ns]")
    
    stack_flood_max = stack_flood_max.sel(time=dates)
    
    if mode == "latest":
        # Forward-fill with provenance tracking
        ever_has_data = (~stack_flood_max.isnull()).any(dim="time")
        flood_filled = stack_flood_max.ffill(dim="time")
        
        # Use latest time step
        flood_composite = flood_filled.isel(time=-1)
        
        # Mask areas that never had data
        flood_composite = flood_composite.where(ever_has_data)
        
    elif mode == "cumulative":
        # Union of all flood observations
        flood_composite = (stack_flood_max == 1).any(dim="time").astype(int)
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'latest' or 'cumulative'")
    
    logger.info(f"  Created {mode} composite")
    flood_pixels = int((flood_composite == 1).sum())
    logger.info(f"  Flood pixels: {flood_pixels:,}")
    
    return flood_composite


def raster_to_polygons(flood_raster: xr.DataArray) -> gpd.GeoDataFrame:
    """Convert flood raster to vector polygons."""
    from rasterio import features
    from shapely.geometry import shape
    
    # Get raster properties  
    transform = flood_raster.rio.transform()
    crs = flood_raster.rio.crs
    
    # Convert to binary array
    flood_array = (flood_raster.values == 1).astype(np.uint8)
    
    # Use rasterio.features.shapes (the standard GDAL-based method)
    shapes_gen = features.shapes(
        flood_array,
        mask=flood_array,
        transform=transform,
        connectivity=8
    )
    
    # Convert to geometries
    geometries = [shape(geom) for geom, value in shapes_gen if value == 1]
    
    return gpd.GeoDataFrame({'geometry': geometries}, crs=crs)


def export_polygons(gdf: gpd.GeoDataFrame, output_path: Path, formats: list = ['shapefile']) -> dict:
    """Export polygons to vector formats.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to export.
    output_path : Path
        Output file path (without extension).
    formats : list, default ['shapefile']
        Output formats ('shapefile', 'geojson').
        
    Returns
    -------
    dict
        Mapping of format to output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    for fmt in formats:
        if fmt == 'shapefile':
            file_path = output_path.with_suffix('.shp')
            gdf.to_file(file_path, driver='ESRI Shapefile')
            
        elif fmt == 'geojson':
            file_path = output_path.with_suffix('.geojson')
            gdf.to_file(file_path, driver='GeoJSON')
        
        output_files[fmt] = file_path
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Exported {fmt}: {file_path} ({file_size_mb:.1f} MB)")
    
    return output_files


def calculate_date_range(end_date: str, n_latest: int) -> tuple[str, str]:
    """Calculate start date from end date and number of days.
    
    Parameters
    ----------
    end_date : str
        End date (YYYY-MM-DD).
    n_latest : int
        Number of days to look back.
        
    Returns
    -------
    tuple[str, str]
        Start and end dates as YYYY-MM-DD strings.
    """
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=n_latest)
    start_date = start_dt.strftime('%Y-%m-%d')
    
    return start_date, end_date
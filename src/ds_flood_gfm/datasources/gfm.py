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
from rasterio import features
from shapely.geometry import shape
import ocha_stratus as stratus

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
    search_start_date = _end_date - timedelta(days=n_search)
    start_date = search_start_date.strftime("%Y-%m-%d")
    
    search = client.search(
        collections=["GFM"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}"
    )
    
    items = search.item_collection()
    logger.info(f"  Found {len(items)} STAC items")
    
    return items


def create_flood_composite(items: list, bbox: list, n_latest: int, mode: str = "latest", return_stack: bool = False) -> tuple:
    """Create flood composite from STAC items.

    Parameters
    ----------
    items : list
        STAC items.
    bbox : list
        Bounding box [west, south, east, north].
    n_latest : int
        Number of most recent dates to use.
    mode : str, default "latest"
        Compositing mode ('latest' or 'cumulative').
    return_stack : bool, default False
        If True, also return stack_flood_max for provenance calculation.

    Returns
    -------
    tuple
        If return_stack=False: (flood_composite, unique_dates)
        If return_stack=True: (flood_composite, unique_dates, stack_flood_max)
    """
    logger.info(f"Creating {mode} flood composite with {n_latest} most recent dates...")
    
    # Extract unique dates from STAC items (fast - no raster loading!)
    all_dates_set = set()
    for item in items:
        dt = pd.to_datetime(item.datetime)
        all_dates_set.add(dt.date())

    all_dates = sorted(list(all_dates_set))
    logger.info(f"All dates found: {[str(d) for d in all_dates]}")

    if len(all_dates) == 0:
        logger.info("ERROR: No valid dates in STAC items!")
        return

    # Select only the N most recent dates
    dates_to_use = all_dates[-n_latest:] if len(all_dates) >= n_latest else all_dates
    logger.info(
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

    if return_stack:
        return flood_composite, dates, stack_flood_max
    else:
        return flood_composite, dates


def raster_to_polygons(flood_raster: xr.DataArray) -> gpd.GeoDataFrame:
    """Convert flood raster to vector polygons."""
    # Get raster properties  
    transform = flood_raster.rio.transform()
    
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
    
    return gpd.GeoDataFrame({'geometry': geometries}, crs="EPSG:4326")


def export_polygons(gdf: gpd.GeoDataFrame, output_path: Path, local=False, blob=True) -> dict:
    """Export polygons to vector formats.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to export.
    output_path : Path
        Output file path (without extension).
        
    Returns
    -------
    dict
        Mapping of format to output file path.
    """

    if local:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = output_path.with_suffix('.shp')
        gdf.to_file(file_path, driver='ESRI Shapefile')
        logger.info(f"Output shapefile locally: {file_path}")
    if blob:
        blob_name = f"ds-flood-gfm/processed/polygon/{output_path}.shp.zip"
        stratus.upload_shp_to_blob(gdf, blob_name)
        logger.info(f"Output shapefile to blob: {blob_name}")


def create_provenance_raster(stack_flood_max: xr.DataArray, unique_dates: np.ndarray) -> tuple[xr.DataArray, dict]:
    """Create provenance raster showing last observation date per pixel.

    Parameters
    ----------
    stack_flood_max : xr.DataArray
        3D array (time, y, x) of flood data.
    unique_dates : np.ndarray
        Array of datetime64 dates corresponding to time dimension.

    Returns
    -------
    tuple[xr.DataArray, dict]
        - provenance_idx: 2D lazy DataArray with integer indices (NOT computed)
        - date_mapping: Dictionary mapping indices to date strings
    """
    logger.info("Creating provenance raster (lazy)...")

    # Create mask of where we have valid observations (regardless of flood value)
    has_data = ~stack_flood_max.isnull()

    # Find the LAST time index where we had data for each pixel
    # argmax finds first True, so we reverse the time dimension first
    # This stays LAZY until .compute() is called
    has_data_reversed = has_data.isel(time=slice(None, None, -1))
    provenance_idx_reversed = has_data_reversed.argmax(dim='time', skipna=True)

    # Convert back to original time indexing
    n_times = len(unique_dates)
    provenance_idx = n_times - 1 - provenance_idx_reversed

    # Mask pixels that never had data
    never_had_data = ~has_data.any(dim='time')
    provenance_idx = provenance_idx.where(~never_had_data, -1)

    # Create date mapping for lookup
    date_mapping = {
        int(i): str(pd.Timestamp(date))[:10]
        for i, date in enumerate(unique_dates)
    }
    date_mapping[-1] = "No Data"

    logger.info(f"  Provenance raster created (lazy, shape will be {provenance_idx.shape})")
    logger.info(f"  Date mapping: {date_mapping}")

    return provenance_idx, date_mapping
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import rasterio
from rasterio.transform import from_bounds
import stackstac
import xarray as xr
import exactextract
from rasterio import features
from shapely.geometry import shape
import ocha_stratus as stratus

logger = logging.getLogger(__name__)


def create_spatial_tiles(bbox: list, tile_size_degrees: float = 2.0) -> list[list]:
    """Create spatial tiles from a bounding box.

    Parameters
    ----------
    bbox : list
        Bounding box [west, south, east, north].
    tile_size_degrees : float, default 2.0
        Size of each tile in degrees.

    Returns
    -------
    list[list]
        List of tile bboxes [[west, south, east, north], ...].
    """
    west, south, east, north = bbox

    tiles = []
    current_west = west
    while current_west < east:
        tile_east = min(current_west + tile_size_degrees, east)

        current_south = south
        while current_south < north:
            tile_north = min(current_south + tile_size_degrees, north)

            tiles.append([current_west, current_south, tile_east, tile_north])

            current_south = tile_north

        current_west = tile_east

    logger.info(f"Created {len(tiles)} tiles of ~{tile_size_degrees}° each")
    return tiles


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
    
    # Build xarray stack with optimized settings for sparse GFM data
    # NOTE: stackstac validation requires float64 for nan fill_value
    # Using float64 with larger chunks and rechunking strategy instead
    logger.info("  Building xarray stack...")
    stack = stackstac.stack(
        items,
        epsg=4326,
        # dtype defaults to float64 - accepted as necessary for nan support
        rescale=False,         # Don't rescale - values are already 0/1/255
        chunksize=2048         # Larger chunks = fewer dask tasks (vs default 1024)
    )
    logger.info(f"    Stack is lazy: {hasattr(stack.data, 'dask')}")
    logger.info(f"    Optimized: dtype={stack.dtype}, chunksize=2048, rescale=False")
    stack_flood = stack.sel(band="ensemble_flood_extent")
    stack_flood_clipped = stack_flood.sel(
        x=slice(bbox[0], bbox[2]),
        y=slice(bbox[3], bbox[1])
    )
    logger.info(f"    Clipped stack shape: {stack_flood_clipped.shape}, still lazy: {hasattr(stack_flood_clipped.data, 'dask')}")

    # Create daily composites
    logger.info("  Creating daily composites...")
    stack_flood_max = stack_flood_clipped.groupby("time.date").max()
    logger.info(f"    After groupby.max(), still lazy: {hasattr(stack_flood_max.data, 'dask')}")
    stack_flood_max = stack_flood_max.rename({"date": "time"})
    stack_flood_max["time"] = stack_flood_max.time.astype("datetime64[ns]")

    stack_flood_max = stack_flood_max.sel(time=dates)

    # RECHUNK: Consolidate time dimension and use larger spatial chunks
    # This dramatically reduces task count for temporal operations
    logger.info("  Rechunking for optimal temporal operations...")
    stack_flood_max = stack_flood_max.chunk({
        'time': -1,    # Single chunk for time (only 4 dates typically)
        'y': 4096,     # Larger spatial chunks (4x reduction in tasks)
        'x': 4096
    })
    logger.info(f"    Rechunked to time:-1, y:4096, x:4096")
    
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
        flood_composite = (stack_flood_max == 1).any(dim="time").astype("uint8")

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'latest' or 'cumulative'")

    logger.info(f"  Created {mode} composite (lazy)")
    logger.info(f"    Composite shape: {flood_composite.shape}, still lazy: {hasattr(flood_composite.data, 'dask')}")
    logger.info(f"  ⚠️  Composite will be computed when converting to polygons")

    if return_stack:
        return flood_composite, dates, stack_flood_max
    else:
        return flood_composite, dates


def raster_to_polygons(flood_raster: xr.DataArray) -> gpd.GeoDataFrame:
    """Convert flood raster to vector polygons."""
    logger.info("  Getting raster properties...")
    # Get raster properties
    transform = flood_raster.rio.transform()

    import time
    # Check task graph size before computation
    if hasattr(flood_raster.data, 'dask'):
        n_tasks = len(flood_raster.data.__dask_graph__())
        logger.info(f"  Dask computation graph has {n_tasks:,} tasks")
        logger.info(f"  Estimated memory: {flood_raster.nbytes / 1e9:.2f} GB")
        logger.info(f"  Chunks: {flood_raster.chunks}")
        logger.info(f"  Shape: {flood_raster.shape}")
    else:
        logger.info(f"  Data is not lazy (already computed)")
        logger.info(f"  Shape: {flood_raster.shape}")

    logger.info("  Computing flood array (this may take a while for large areas)...")
    start_time = time.time()
    # Convert to binary array
    flood_array = (flood_raster.values == 1).astype(np.uint8)
    elapsed = time.time() - start_time
    actual_memory_gb = flood_array.nbytes / 1e9
    logger.info(f"  ✅ Flood array computed in {elapsed:.1f}s - shape: {flood_array.shape}")
    logger.info(f"  Actual memory used: {actual_memory_gb:.2f} GB")

    logger.info("  Extracting polygon shapes from raster...")
    # Use rasterio.features.shapes (the standard GDAL-based method)
    shapes_gen = features.shapes(
        flood_array,
        mask=flood_array,
        transform=transform,
        connectivity=8
    )

    logger.info("  Converting shapes to geometries...")
    # Convert to geometries
    geometries = [shape(geom) for geom, value in shapes_gen if value == 1]
    logger.info(f"  Created {len(geometries)} polygons")

    return gpd.GeoDataFrame({'geometry': geometries}, crs="EPSG:4326")


def process_country_tiled(
    bbox: list,
    end_date: str,
    n_latest: int = 4,
    n_search: int = 15,
    mode: str = "latest",
    tile_size: float = 2.0,
    return_stack: bool = False
) -> tuple:
    """Process large country in tiles to avoid memory issues.

    Parameters
    ----------
    bbox : list
        Full country bounding box [west, south, east, north].
    end_date : str
        End date (YYYY-MM-DD).
    n_latest : int, default 4
        Number of most recent dates to use.
    n_search : int, default 15
        Search window in days.
    mode : str, default "latest"
        Compositing mode ('latest' or 'cumulative').
    tile_size : float, default 2.0
        Size of each tile in degrees.
    return_stack : bool, default False
        If True, also return combined stack for provenance.

    Returns
    -------
    tuple
        If return_stack=False: (combined_polygons, unique_dates)
        If return_stack=True: (combined_polygons, unique_dates, combined_stack)
    """
    logger.info("=" * 60)
    logger.info("TILED PROCESSING FOR LARGE COUNTRY")
    logger.info("=" * 60)

    tiles = create_spatial_tiles(bbox, tile_size)
    logger.info(f"Processing {len(tiles)} tiles...")

    tile_polygons = []
    tile_stacks = []
    all_dates = None

    for i, tile_bbox in enumerate(tiles, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"TILE {i}/{len(tiles)}")
        logger.info(f"{'='*60}")
        logger.info(f"Bbox: {tile_bbox}")

        try:
            # Query STAC for this tile
            items = query_gfm_stac(tile_bbox, end_date, n_search)

            if len(items) == 0:
                logger.info(f"  No data for tile {i}, skipping...")
                continue

            # Create composite for this tile
            if return_stack:
                flood_composite, unique_dates, stack_flood_max = create_flood_composite(
                    items, tile_bbox, n_latest, mode=mode, return_stack=True
                )
                tile_stacks.append(stack_flood_max)
            else:
                flood_composite, unique_dates = create_flood_composite(
                    items, tile_bbox, n_latest, mode=mode, return_stack=False
                )

            # Store dates from first tile
            if all_dates is None:
                all_dates = unique_dates

            # Convert to polygons
            logger.info(f"Converting tile {i} to polygons...")
            tile_polys = raster_to_polygons(flood_composite)

            if len(tile_polys) > 0:
                tile_polygons.append(tile_polys)
                logger.info(f"  ✅ Tile {i} complete: {len(tile_polys)} polygons")
            else:
                logger.info(f"  ⚠️  Tile {i} had no flood polygons")

        except Exception as e:
            logger.error(f"  ❌ Tile {i} failed: {e}")
            continue

    # Combine all tile polygons
    logger.info("\n" + "=" * 60)
    logger.info("COMBINING TILES")
    logger.info("=" * 60)

    if len(tile_polygons) == 0:
        logger.warning("No polygons from any tile!")
        return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326"), all_dates

    combined_polygons = gpd.GeoDataFrame(
        pd.concat(tile_polygons, ignore_index=True),
        crs="EPSG:4326"
    )
    logger.info(f"✅ Combined {len(combined_polygons)} total polygons from {len(tile_polygons)} tiles")

    if return_stack:
        # For provenance, we'd need to mosaic the tile stacks - not implemented yet
        logger.warning("⚠️  Provenance raster not yet supported for tiled processing")
        return combined_polygons, all_dates, None
    else:
        return combined_polygons, all_dates


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


def add_modal_provenance_to_admin(gdf_admin: gpd.GeoDataFrame,
                                   provenance_idx: xr.DataArray,
                                   date_mapping: dict,
                                   admin_level: int = 3) -> gpd.GeoDataFrame:
    """Add modal (most common) provenance date to admin boundaries.

    Uses exactextract to efficiently calculate the most common provenance date
    within each administrative boundary.

    Parameters
    ----------
    gdf_admin : gpd.GeoDataFrame
        Administrative boundaries.
    provenance_idx : xr.DataArray
        Provenance raster with integer indices (should be computed, not lazy).
    date_mapping : dict
        Mapping from integer indices to date strings.
    admin_level : int, default 0
        Administrative level (0, 1, 2, 3) for column naming.

    Returns
    -------
    gpd.GeoDataFrame
        Admin boundaries with 'prov_date' and 'prov_idx' columns added.
    """
    logger.info(f"Adding modal provenance to admin level {admin_level} boundaries...")

    # Write provenance raster to temporary file for exactextract
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_raster_path = tmp.name

    try:
        # Get raster properties
        height, width = provenance_idx.shape
        x_min, x_max = float(provenance_idx.x.min()), float(provenance_idx.x.max())
        y_min, y_max = float(provenance_idx.y.min()), float(provenance_idx.y.max())
        transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

        # Write to temporary GeoTIFF
        with rasterio.open(
            tmp_raster_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="int16",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(provenance_idx.values.astype(np.int16), 1)

        # Use exactextract to get mode of provenance for each polygon
        logger.info("  Running exactextract for modal provenance...")
        modal_prov = exactextract.exact_extract(
            tmp_raster_path,
            gdf_admin,
            ["mode", "count"],
            include_cols=[f"adm{admin_level}_id"],
            output="pandas",
        )

        # Merge back to admin
        gdf_result = gdf_admin.merge(
            modal_prov,
            left_on=f"adm{admin_level}_id",
            right_on=f"adm{admin_level}_id",
            how="left"
        )

        # Map mode values to actual dates
        gdf_result["prov_idx"] = gdf_result["mode"].fillna(-1).astype(int)
        gdf_result["prov_date"] = gdf_result["prov_idx"].map(date_mapping)

        logger.info(f"  ✅ Added provenance to {len(gdf_result)} admin units")

        # Log summary
        prov_summary = gdf_result["prov_date"].value_counts()
        logger.info("  Provenance summary by admin unit:")
        for date, count in prov_summary.items():
            logger.info(f"    {date}: {count} admin units")

        return gdf_result

    finally:
        # Clean up temp file
        if os.path.exists(tmp_raster_path):
            os.unlink(tmp_raster_path)
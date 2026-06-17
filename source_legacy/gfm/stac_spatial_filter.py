"""
Utilities for filtering STAC search results to remove spatial metadata false positives.

The EODC GFM STAC catalog sometimes returns items that don't actually intersect
with the requested bounding box. This module provides efficient filtering to remove
those false positives.
"""
import numpy as np
import xarray as xr
import rioxarray
from typing import List, Tuple
from pystac import Item, ItemCollection
from shapely.geometry import box
from pyproj import Transformer
import warnings


def get_tile_actual_bounds_wgs84(item: Item, asset_key: str = 'ensemble_flood_extent') -> Tuple[float, float, float, float]:
    """
    Get the actual geographic bounds of a COG tile in WGS84 coordinates.

    This loads just the metadata (not the full raster data) to determine
    the true spatial extent of the tile.

    Parameters
    ----------
    item : pystac.Item
        STAC item to check
    asset_key : str
        Asset key to check (default: 'ensemble_flood_extent')

    Returns
    -------
    tuple
        (min_lon, min_lat, max_lon, max_lat) in WGS84
    """
    try:
        cog_url = item.assets[asset_key].href

        # Open with chunks to avoid loading full array
        with rioxarray.open_rasterio(cog_url, masked=True, chunks='auto') as da:
            bounds = da.rio.bounds()
            crs = da.rio.crs

            # If already in WGS84, return directly
            if crs.to_epsg() == 4326:
                return bounds

            # Transform to WGS84
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            min_x, min_y, max_x, max_y = bounds

            # Transform corners
            lon_min, lat_min = transformer.transform(min_x, min_y)
            lon_max, lat_max = transformer.transform(max_x, max_y)

            return (lon_min, lat_min, lon_max, lat_max)

    except Exception as e:
        warnings.warn(f"Could not get bounds for item {item.id}: {e}")
        # Fall back to STAC bbox if available
        if hasattr(item, 'bbox') and item.bbox:
            return tuple(item.bbox)
        return None


def item_intersects_bbox(item: Item, bbox: List[float], asset_key: str = 'ensemble_flood_extent') -> bool:
    """
    Check if a STAC item's actual data footprint intersects with a bounding box.

    This verifies the actual COG bounds (not just STAC metadata) to catch
    cases where STAC spatial indexing is incorrect.

    Parameters
    ----------
    item : pystac.Item
        STAC item to check
    bbox : list
        Bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84
    asset_key : str
        Asset key to check (default: 'ensemble_flood_extent')

    Returns
    -------
    bool
        True if item intersects bbox, False otherwise
    """
    item_bounds = get_tile_actual_bounds_wgs84(item, asset_key)

    if item_bounds is None:
        warnings.warn(f"Could not verify bounds for item {item.id}, keeping it")
        return True

    min_lon, min_lat, max_lon, max_lat = bbox
    item_min_lon, item_min_lat, item_max_lon, item_max_lat = item_bounds

    # Check for intersection (NOT operation = no overlap)
    no_overlap = (
        item_max_lon < min_lon or  # item is west of bbox
        item_min_lon > max_lon or  # item is east of bbox
        item_max_lat < min_lat or  # item is south of bbox
        item_min_lat > max_lat     # item is north of bbox
    )

    return not no_overlap


def filter_stac_results(items: ItemCollection, bbox: List[float],
                       asset_key: str = 'ensemble_flood_extent',
                       verbose: bool = False) -> List[Item]:
    """
    Filter STAC search results to only include items that actually intersect the bbox.

    This is a workaround for STAC catalogs with incorrect spatial metadata.
    It verifies each item's actual COG footprint against the requested bbox.

    Parameters
    ----------
    items : ItemCollection or list
        STAC items to filter
    bbox : list
        Bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84
    asset_key : str
        Asset key to check (default: 'ensemble_flood_extent')
    verbose : bool
        Print filtering statistics (default: False)

    Returns
    -------
    list
        Filtered list of STAC items that actually intersect the bbox

    Examples
    --------
    >>> import pystac_client
    >>> from ds_flood_gfm.stac_spatial_filter import filter_stac_results
    >>>
    >>> # Search STAC (may return false positives)
    >>> catalog = pystac_client.Client.open("https://stac.eodc.eu/api/v1")
    >>> haiti_bbox = [-74.5, 18.0, -71.6, 20.1]
    >>> search = catalog.search(collections=["GFM"], bbox=haiti_bbox,
    ...                         datetime="2025-10-27")
    >>> items = search.item_collection()
    >>>
    >>> # Filter to only items that actually cover Haiti
    >>> filtered_items = filter_stac_results(items, haiti_bbox, verbose=True)
    """
    items_list = list(items)

    if verbose:
        print(f"Filtering {len(items_list)} STAC items for bbox {bbox}...")

    filtered = []
    removed = []

    for item in items_list:
        if item_intersects_bbox(item, bbox, asset_key):
            filtered.append(item)
        else:
            removed.append(item)
            if verbose:
                print(f"  Removed (no intersection): {item.id}")

    if verbose:
        print(f"  Kept: {len(filtered)} items")
        print(f"  Removed: {len(removed)} items (false positives)")

    return filtered


def get_dates_with_actual_coverage(items: ItemCollection, bbox: List[float],
                                   asset_key: str = 'ensemble_flood_extent',
                                   verbose: bool = False) -> List[str]:
    """
    Extract unique dates from STAC items, filtering out dates with no actual coverage.

    This is useful for getting the true list of available dates for a region,
    accounting for STAC spatial metadata issues.

    Parameters
    ----------
    items : ItemCollection or list
        STAC items to process
    bbox : list
        Bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84
    asset_key : str
        Asset key to check (default: 'ensemble_flood_extent')
    verbose : bool
        Print filtering statistics (default: False)

    Returns
    -------
    list
        Sorted list of date strings (YYYY-MM-DD) that have actual coverage

    Examples
    --------
    >>> import pystac_client
    >>> from ds_flood_gfm.stac_spatial_filter import get_dates_with_actual_coverage
    >>>
    >>> # Search STAC
    >>> catalog = pystac_client.Client.open("https://stac.eodc.eu/api/v1")
    >>> haiti_bbox = [-74.5, 18.0, -71.6, 20.1]
    >>> search = catalog.search(collections=["GFM"], bbox=haiti_bbox,
    ...                         datetime="2025-10-13/2025-10-28")
    >>> items = search.item_collection()
    >>>
    >>> # Get only dates with actual coverage
    >>> dates = get_dates_with_actual_coverage(items, haiti_bbox, verbose=True)
    >>> print(f"Available dates: {dates}")
    """
    import pandas as pd

    # Filter items first
    filtered_items = filter_stac_results(items, bbox, asset_key, verbose)

    # Extract dates
    dates_set = set()
    for item in filtered_items:
        dt = pd.to_datetime(item.datetime)
        dates_set.add(str(dt.date()))

    return sorted(list(dates_set))

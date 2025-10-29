# STAC Spatial Metadata Issue - Reproducible Example

This folder contains reproducible examples demonstrating a spatial metadata issue with the EODC GFM STAC catalog.

## Issue Summary

The STAC catalog at `https://stac.eodc.eu/api/v1` returns items that do not actually intersect with the requested bounding box. Specifically, when searching for Haiti data (Caribbean), the catalog returns tiles that are physically located in Asia (~12,000 km away).

## Files

### Documentation
- **[stac_spatial_metadata_issue.md](stac_spatial_metadata_issue.md)** - Detailed issue report for sharing with data provider

### Demonstration Scripts
- **[demonstrate_stac_issue.py](demonstrate_stac_issue.py)** - Shows the problem with actual STAC queries
- **[test_spatial_filter.py](test_spatial_filter.py)** - Demonstrates the workaround solution

## Running the Examples

### 1. Demonstrate the Issue

```bash
uv run python reprex/demonstrate_stac_issue.py
```

This script:
- Searches STAC for Haiti data on 2025-10-27
- Shows that returned tiles are actually in Asia (E105°, N015°)
- Calculates distance between Haiti and the incorrect tiles
- Provides detailed output showing the mismatch

### 2. Test the Workaround Solution

```bash
uv run python reprex/test_spatial_filter.py
```

This script:
- Searches STAC for Haiti data over a date range
- Shows dates found WITHOUT spatial filtering (includes false positives)
- Shows dates found WITH spatial filtering (verified coverage only)
- Demonstrates which dates are false positives

## The Solution

We've implemented a spatial filtering module to work around this issue:

**Module**: `src/ds_flood_gfm/stac_spatial_filter.py`

### Basic Usage

```python
from ds_flood_gfm.stac_spatial_filter import get_dates_with_actual_coverage
import pystac_client

# Search STAC (may return false positives)
catalog = pystac_client.Client.open("https://stac.eodc.eu/api/v1")
bbox = [-74.5, 18.0, -71.6, 20.1]  # Haiti
search = catalog.search(collections=["GFM"], bbox=bbox,
                       datetime="2025-10-20/2025-10-28")
items = search.item_collection()

# Get only dates with actual coverage (removes false positives)
dates = get_dates_with_actual_coverage(items, bbox, verbose=True)
```

### Available Functions

1. **`filter_stac_results(items, bbox)`** - Filter items to only those that intersect bbox
2. **`get_dates_with_actual_coverage(items, bbox)`** - Get list of dates with verified coverage
3. **`item_intersects_bbox(item, bbox)`** - Check if a single item intersects bbox
4. **`get_tile_actual_bounds_wgs84(item)`** - Get actual geographic bounds of a tile

## How the Filter Works

The spatial filter:
1. Opens each COG file (metadata only, not full data)
2. Reads the actual geographic bounds from the raster
3. Transforms bounds to WGS84 if needed
4. Checks for intersection with requested bbox
5. Filters out items that don't intersect

This is fast because it only reads metadata, not the full raster arrays.

## Performance Considerations

- **Metadata-only reads**: Only COG headers are loaded (very fast)
- **Lazy evaluation**: Uses rioxarray with `chunks='auto'`
- **Minimal overhead**: Adds ~0.1-0.5 seconds per item to verify bounds
- **Worth it**: Prevents loading and processing gigabytes of irrelevant data

## Impact of the STAC Issue

Without spatial filtering, this issue causes:

1. ❌ **False date availability** - Shows dates that don't actually cover your AOI
2. ❌ **Wasted processing** - Downloads and processes irrelevant tiles
3. ❌ **Misleading titles** - Maps titled with dates that have no actual coverage
4. ❌ **Cache pollution** - Stores empty/irrelevant data in cache

With spatial filtering:

1. ✅ **Accurate date lists** - Only shows dates with real coverage
2. ✅ **Efficient processing** - Only loads relevant tiles
3. ✅ **Correct metadata** - Titles reflect actual data availability
4. ✅ **Clean caches** - Only real data is cached

## Recommendation

**For all STAC searches**, use the spatial filter until the catalog metadata is fixed:

```python
# OLD (may include false positives):
dates = sorted({str(pd.to_datetime(item.datetime).date()) for item in items})

# NEW (verified coverage only):
from ds_flood_gfm.stac_spatial_filter import get_dates_with_actual_coverage
dates = get_dates_with_actual_coverage(items, bbox, verbose=True)
```

## Contact

This issue was identified by the CHD Data Science team during development of flood monitoring workflows.

For questions about this reprex, contact: [Your contact info]

For questions about the STAC catalog, contact: EODC GFM support

# STAC Spatial Metadata Issue: False Positives in Bounding Box Search

## Summary

The EODC GFM STAC catalog (`https://stac.eodc.eu/api/v1`) returns items that do not actually intersect with the requested bounding box. When searching for data over Haiti using bbox `[-74.5, 18.0, -71.6, 20.1]`, the catalog returns tiles that are physically located in Asia (around E105°, N015°).

## Issue Details

- **STAC Catalog**: `https://stac.eodc.eu/api/v1`
- **Collection**: `GFM`
- **Search Region**: Haiti (Caribbean)
- **Search BBox**: `[-74.5, 18.0, -71.6, 20.1]` (W74.5°, N18°, W71.6°, N20.1°)
- **Date**: 2025-10-27
- **Problem**: STAC search returns 3 items, but all are located in Asia (not Haiti)

## Expected vs Actual Behavior

### Expected
When searching with Haiti's bounding box, the STAC API should only return items whose actual data footprint intersects with Haiti.

### Actual
The STAC API returns items for date 2025-10-27 that are named `E105N015T3`, indicating they cover coordinates around E105°, N015° (Southeast Asia), which is approximately 12,000 km away from Haiti.

## Reproducible Example

### Python Code

```python
import pystac_client
import rioxarray

# Haiti bounding box (Caribbean: ~W74.5°-W71.6°, N18°-N20°)
HAITI_BBOX = [-74.5, 18.0, -71.6, 20.1]
TARGET_DATE = "2025-10-27"

# Search STAC catalog
catalog = pystac_client.Client.open("https://stac.eodc.eu/api/v1")
search = catalog.search(
    collections=["GFM"],
    bbox=HAITI_BBOX,
    datetime=f"{TARGET_DATE}T00:00:00Z/{TARGET_DATE}T23:59:59Z",
    max_items=100
)
items = list(search.items())

print(f"Search parameters:")
print(f"  BBox: {HAITI_BBOX} (Haiti, Caribbean)")
print(f"  Date: {TARGET_DATE}")
print(f"\nSTAC search returned {len(items)} items:")

for i, item in enumerate(items, 1):
    cog_url = item.assets['ensemble_flood_extent'].href
    filename = cog_url.split('/')[-1]
    print(f"\n  Item {i}:")
    print(f"    Filename: {filename}")

    # Load the actual data to check its coordinates
    da = rioxarray.open_rasterio(cog_url, masked=True)
    actual_bounds = da.rio.bounds()
    print(f"    Actual bounds (projected): {actual_bounds}")
    print(f"    CRS: {da.rio.crs}")

    # Check if tile name indicates location
    if 'E105N015' in filename:
        print(f"    ⚠️  WARNING: Tile name indicates Asia (E105°, N015°), NOT Haiti!")
```

### Expected Output
```
Search parameters:
  BBox: [-74.5, 18.0, -71.6, 20.1] (Haiti, Caribbean)
  Date: 2025-10-27

STAC search returned 0 items:

(Or, if data exists for Haiti on 2025-10-27, return only tiles that actually cover Haiti)
```

### Actual Output
```
Search parameters:
  BBox: [-74.5, 18.0, -71.6, 20.1] (Haiti, Caribbean)
  Date: 2025-10-27

STAC search returned 3 items:

  Item 1:
    Filename: ENSEMBLE_FLOOD_20251027T110407_VV_NA020M_E105N015T3.tif
    Actual bounds (projected): (10500000.0, 1500000.0, 10800000.0, 1800000.0)
    CRS: EPSG:6933
    ⚠️  WARNING: Tile name indicates Asia (E105°, N015°), NOT Haiti!

  Item 2:
    Filename: ENSEMBLE_FLOOD_20251027T110342_VV_NA020M_E105N015T3.tif
    Actual bounds (projected): (10500000.0, 1500000.0, 10800000.0, 1800000.0)
    CRS: EPSG:6933
    ⚠️  WARNING: Tile name indicates Asia (E105°, N015°), NOT Haiti!

  Item 3:
    Filename: ENSEMBLE_FLOOD_20251027T110317_VV_NA020M_E105N015T3.tif
    Actual bounds (projected): (10500000.0, 1500000.0, 10800000.0, 1800000.0)
    CRS: EPSG:6933
    ⚠️  WARNING: Tile name indicates Asia (E105°, N015°), NOT Haiti!
```

## Impact

This issue causes:

1. **False positives in date availability**: Users believe data exists for a date when it actually doesn't cover their area of interest
2. **Wasted processing time**: Applications download and process tiles that don't contain relevant data
3. **Misleading metadata**: Date lists include dates that don't actually have coverage for the requested region
4. **Incorrect analysis**: When compositing data, the "latest available date" may appear to be more recent than it actually is for the region

## Technical Details

### Item Metadata Example

When inspecting one of the returned STAC items:

```python
item = items[0]
print(f"Item ID: {item.id}")
print(f"Item bbox (from STAC): {item.bbox}")
print(f"Item geometry: {item.geometry}")
```

The item's STAC metadata (bbox/geometry) likely claims to intersect with the Haiti bounding box, but the actual COG file does not.

### Verification

To verify this is a STAC metadata issue (not a data processing issue), we confirmed:

1. ✅ The STAC search query is correct (bbox format, coordinate order)
2. ✅ The returned items have `E105N015T3` in their filenames (Asia, not Caribbean)
3. ✅ When loaded, the actual data bounds are in projected coordinates centered around 10.5-10.8M meters (consistent with E105° in EPSG:6933)
4. ✅ Other dates (2025-10-24, 2025-10-26) correctly return tiles that DO cover Haiti

## Suggested Fixes

1. **Reindex STAC catalog**: Update spatial metadata to match actual tile footprints
2. **Validate tile footprints**: Check that all GFM tiles have correct bbox/geometry in STAC metadata
3. **Add tile naming validation**: Ensure tile grid names (e.g., `E105N015T3`) align with geographic coordinates in STAC metadata

## Environment

- **STAC Catalog**: https://stac.eodc.eu/api/v1
- **Collection**: GFM
- **Python**: 3.12.4
- **pystac-client**: (version from requirements)
- **rioxarray**: (version from requirements)
- **Date tested**: 2025-10-28

## Additional Notes

This issue appears to be intermittent or date-specific. Not all dates exhibit this problem - for example, 2025-10-24 and 2025-10-26 correctly return only Haiti tiles. This suggests the issue may be related to specific processing runs or data ingestion batches rather than a systemic catalog problem.

---

**Prepared by**: CHD Data Science Team
**Date**: 2025-10-28
**Contact**: [Your contact info]

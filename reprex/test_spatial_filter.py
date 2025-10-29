"""
Test script demonstrating the spatial filter to remove STAC false positives.

This shows how to use the stac_spatial_filter module to automatically
detect and remove tiles that don't actually intersect with your AOI.

Run: uv run python reprex/test_spatial_filter.py
"""
import pystac_client
from ds_flood_gfm.stac_spatial_filter import filter_stac_results, get_dates_with_actual_coverage

# Haiti bounding box
HAITI_BBOX = [-74.5, 18.0, -71.6, 20.1]
DATE_RANGE = "2025-10-20/2025-10-28"

print("="*80)
print("TESTING STAC SPATIAL FILTER")
print("="*80)

print(f"\nSearch parameters:")
print(f"  Region: Haiti")
print(f"  BBox: {HAITI_BBOX}")
print(f"  Date range: {DATE_RANGE}")

# Search STAC catalog (may return false positives)
print(f"\nSearching STAC catalog...")
catalog = pystac_client.Client.open("https://stac.eodc.eu/api/v1")
search = catalog.search(
    collections=["GFM"],
    bbox=HAITI_BBOX,
    datetime=DATE_RANGE
)
items = search.item_collection()

print(f"\nSTAC returned {len(items)} items")

# Extract dates WITHOUT filtering (includes false positives)
print(f"\n{'='*80}")
print("DATES WITHOUT SPATIAL FILTERING (may include false positives):")
print(f"{'='*80}")

import pandas as pd
dates_unfiltered = set()
for item in items:
    dt = pd.to_datetime(item.datetime)
    dates_unfiltered.add(str(dt.date()))

dates_unfiltered = sorted(list(dates_unfiltered))
print(f"Dates: {dates_unfiltered}")
print(f"Total: {len(dates_unfiltered)} dates")

# Extract dates WITH filtering (removes false positives)
print(f"\n{'='*80}")
print("DATES WITH SPATIAL FILTERING (verified to actually cover Haiti):")
print(f"{'='*80}")

dates_filtered = get_dates_with_actual_coverage(
    items,
    HAITI_BBOX,
    verbose=True
)

print(f"\nDates: {dates_filtered}")
print(f"Total: {len(dates_filtered)} dates")

# Show which dates were false positives
false_positives = set(dates_unfiltered) - set(dates_filtered)
if false_positives:
    print(f"\n{'='*80}")
    print(f"FALSE POSITIVES DETECTED:")
    print(f"{'='*80}")
    print(f"These dates were in STAC but don't actually cover Haiti:")
    for date in sorted(false_positives):
        print(f"  - {date}")
else:
    print(f"\n✓ No false positives detected - STAC metadata is correct!")

print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}")
print(f"\nTo avoid false positives in your scripts, use:")
print(f"")
print(f"  from ds_flood_gfm.stac_spatial_filter import get_dates_with_actual_coverage")
print(f"")
print(f"  # Instead of extracting dates directly from STAC items:")
print(f"  dates = get_dates_with_actual_coverage(items, bbox, verbose=True)")
print(f"")
print(f"This will automatically filter out tiles that don't actually intersect")
print(f"with your bounding box, ensuring you only work with real data.")
print(f"\n{'='*80}\n")

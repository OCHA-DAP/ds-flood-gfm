"""
Reproducible example demonstrating STAC spatial metadata issue.

This script searches for GFM data over Haiti and shows that the STAC catalog
returns tiles that are actually located in Asia (not Haiti).

Run: uv run python reprex/demonstrate_stac_issue.py
"""
import pystac_client
import rioxarray
from pyproj import Transformer

# Haiti bounding box (Caribbean: ~W74.5°-W71.6°, N18°-N20°)
HAITI_BBOX = [-74.5, 18.0, -71.6, 20.1]
TARGET_DATE = "2025-10-27"

print("="*80)
print("STAC SPATIAL METADATA ISSUE DEMONSTRATION")
print("="*80)

print(f"\nSearch parameters:")
print(f"  STAC Catalog: https://stac.eodc.eu/api/v1")
print(f"  Collection: GFM")
print(f"  BBox: {HAITI_BBOX}")
print(f"    (Haiti, Caribbean: W74.5° to W71.6°, N18° to N20°)")
print(f"  Date: {TARGET_DATE}")

# Search STAC catalog
print(f"\nSearching STAC catalog...")
catalog = pystac_client.Client.open("https://stac.eodc.eu/api/v1")
search = catalog.search(
    collections=["GFM"],
    bbox=HAITI_BBOX,
    datetime=f"{TARGET_DATE}T00:00:00Z/{TARGET_DATE}T23:59:59Z",
    max_items=100
)
items = list(search.items())

print(f"\n{'='*80}")
print(f"STAC SEARCH RESULTS: {len(items)} items returned")
print(f"{'='*80}")

if len(items) == 0:
    print("\n✓ CORRECT: No items returned (either no data exists, or STAC is working correctly)")
    exit(0)

# Check each returned item
print(f"\nAnalyzing returned items:\n")

# Create transformer to convert from EPSG:6933 to WGS84
transformer = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)

for i, item in enumerate(items, 1):
    print(f"{'─'*80}")
    print(f"Item {i}/{len(items)}:")
    print(f"{'─'*80}")

    # Get item metadata
    print(f"  STAC Item ID: {item.id}")
    print(f"  STAC Item bbox: {item.bbox}")

    # Get COG URL and filename
    cog_url = item.assets['ensemble_flood_extent'].href
    filename = cog_url.split('/')[-1]
    print(f"  Filename: {filename}")

    # Check tile grid name from filename
    if 'E105N015' in filename:
        print(f"  🚨 ISSUE DETECTED: Tile grid name is 'E105N015T3'")
        print(f"     This indicates location around E105°, N015° (Southeast Asia)")
        print(f"     Haiti is at W74.5° to W71.6° (~12,000 km away!)")

    # Load actual COG to check real coordinates
    print(f"\n  Loading COG to verify actual bounds...")
    try:
        da = rioxarray.open_rasterio(cog_url, masked=True)
        actual_bounds = da.rio.bounds()
        crs = da.rio.crs

        print(f"  Actual bounds (projected CRS): {actual_bounds}")
        print(f"  CRS: {crs}")

        # Convert bounds to lat/lon for comparison
        if crs.to_string() == "EPSG:6933":
            # Transform corners to lat/lon
            min_x, min_y, max_x, max_y = actual_bounds
            lon_min, lat_min = transformer.transform(min_x, min_y)
            lon_max, lat_max = transformer.transform(max_x, max_y)

            print(f"  Actual bounds (WGS84): [{lon_min:.2f}, {lat_min:.2f}, {lon_max:.2f}, {lat_max:.2f}]")
            print(f"    (E{lon_min:.1f}° to E{lon_max:.1f}°, N{lat_min:.1f}° to N{lat_max:.1f}°)")

            # Check if bounds intersect with Haiti
            haiti_lon_min, haiti_lat_min, haiti_lon_max, haiti_lat_max = HAITI_BBOX

            intersects = not (
                lon_max < haiti_lon_min or  # tile is west of Haiti
                lon_min > haiti_lon_max or  # tile is east of Haiti
                lat_max < haiti_lat_min or  # tile is south of Haiti
                lat_min > haiti_lat_max     # tile is north of Haiti
            )

            if intersects:
                print(f"  ✓ CORRECT: Tile actually intersects Haiti bounding box")
            else:
                print(f"  🚨 PROBLEM: Tile does NOT intersect Haiti bounding box!")
                print(f"     Haiti is at: W74.5° to W71.6°, N18° to N20°")
                print(f"     This tile is at: E{lon_min:.1f}° to E{lon_max:.1f}°, N{lat_min:.1f}° to N{lat_max:.1f}°")

                # Calculate approximate distance
                haiti_center_lon = (haiti_lon_min + haiti_lon_max) / 2
                haiti_center_lat = (haiti_lat_min + haiti_lat_max) / 2
                tile_center_lon = (lon_min + lon_max) / 2
                tile_center_lat = (lat_min + lat_max) / 2

                # Rough distance calculation (deg to km at equator)
                lon_diff_km = abs(tile_center_lon - haiti_center_lon) * 111
                lat_diff_km = abs(tile_center_lat - haiti_center_lat) * 111
                distance_km = (lon_diff_km**2 + lat_diff_km**2)**0.5

                print(f"     Approximate distance from Haiti: {distance_km:,.0f} km")

        da.close()

    except Exception as e:
        print(f"  ⚠️  Could not load COG: {e}")

    print()

print(f"{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"\nThe STAC catalog returned {len(items)} items for Haiti on {TARGET_DATE},")
print(f"but these tiles are located in Asia (around E105°, N015°), NOT Haiti.")
print(f"\nThis is a STAC catalog metadata issue where the spatial index")
print(f"does not correctly match the actual tile footprints.")
print(f"\n{'='*80}\n")

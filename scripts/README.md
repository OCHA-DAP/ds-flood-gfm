# GFM Flood Analysis Scripts

This directory contains production scripts for generating flood analysis choropleths with data provenance tracking.

Scripts are numbered in execution order (01, 02, 03).

## Scripts Overview

### 01. `01_download_codab_to_blob.py`
Download COD-AB (Common Operational Datasets - Administrative Boundaries) to Azure blob storage.

**Purpose:** One-time setup to cache admin boundaries in blob storage for fast access.

**Usage:**
```bash
uv run python scripts/01_download_codab_to_blob.py
```

---

### 02. `02_generate_affected_population_choropleths.py`
Main analysis script that queries STAC API, processes flood data, and creates affected population visualizations.

**Features:**
- Queries GFM STAC API for satellite flood observations
- Builds xarray stack with temporal compositing
- Tracks data provenance (which observation date each pixel came from)
- Extracts flood points (latest OR cumulative mode)
- Samples GHSL population at flood locations with 3-tier adjustment:
  - `population_raw` - Raw GHSL value (100m pixel = 10,000 m²)
  - `population_adjusted_raw` - Raw ÷ 25 (fractional, accounts for 20m GFM vs 100m GHSL)
  - `population_adjusted` - Rounded up to nearest integer (default, no fractional people)
- Smart caching system (only rebuilds when satellite data changes)
- Fast blob storage integration for admin boundaries

**Outputs:**
- Density heatmap (purple/blue gradient)
- Affected population choropleth (white→red, ADM3-level)

---

### 03. `03_generate_flooded_area_choropleths.py`
Standalone script that reads cached flood points and generates flooded area visualizations.

**Features:**
- Loads cached flood points from script #1
- Counts flood pixels per ADM3 division
- Calculates flooded area: pixels × 400 m² ÷ 1,000,000 = km²
- Generates both latest and cumulative choropleths
- No population data required

**Outputs:**
- Flooded area choropleth (latest) - white→dark blue
- Flooded area choropleth (cumulative) - white→dark blue

---

## Workflow: Generate All 4 Choropleths

### Prerequisites
- Python 3.12.4 with `uv` for dependency management
- Azure blob storage access (for admin boundaries and population data)
- GFM STAC API access

### Step 1: Generate Cumulative Affected Population
This builds the cumulative cache with all flood points from all 3 dates combined.

```bash
uv run python scripts/02_generate_affected_population_choropleths.py \
  --end-date 2025-10-27 \
  --n-latest 3 \
  --population-raster "ghsl/pop/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif" \
  --flood-mode cumulative
```

**Runtime:** ~4-5 minutes (first run), ~46 seconds (cached)

**Outputs:**
- `experiments/population_provenance_20251027.png` - Density heatmap
- `experiments/choropleth_adm3_20251027.png` - Affected pop choropleth (cumulative)

**What's cached:**
- `data/cache/JAM_cumulative_<hash>/flood_points.parquet` - 419 flood pixels with population
- `data/cache/JAM_cumulative_<hash>/provenance.tif` - Provenance raster
- `data/cache/JAM_cumulative_<hash>/metadata.json` - Dates and extents

---

### Step 2: Generate Latest Affected Population
This builds the latest cache with only the most recent provenance pixels.

```bash
uv run python scripts/02_generate_affected_population_choropleths.py \
  --end-date 2025-10-27 \
  --n-latest 3 \
  --population-raster "ghsl/pop/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif" \
  --flood-mode latest
```

**Runtime:** ~4-5 minutes (first run), ~46 seconds (cached)

**Outputs:**
- `experiments/population_provenance_20251027.png` - Density heatmap (overwrites cumulative version)
- `experiments/choropleth_adm3_20251027.png` - Affected pop choropleth (latest)

**What's cached:**
- `data/cache/JAM_latest_<hash>/flood_points.parquet` - 343 flood pixels with population
- `data/cache/JAM_latest_<hash>/provenance.tif` - Provenance raster
- `data/cache/JAM_latest_<hash>/metadata.json` - Dates and extents

---

### Step 3: Generate Both Flooded Area Choropleths
This reads both caches and generates flooded area maps in one go.

```bash
uv run python scripts/03_generate_flooded_area_choropleths.py
```

**Runtime:** ~30-40 seconds

**Outputs:**
- `experiments/choropleth_flooded_area_latest_20251027.png` - Latest flooded area (0.13 km²)
- `experiments/choropleth_flooded_area_cumulative_20251027.png` - Cumulative flooded area (0.16 km²)

---

## Understanding the Cache System

### Cache Key Generation
Cache keys are generated from:
- ISO3 country code
- Actual observation dates from STAC (not query end-date!)
- Population raster path
- Flood mode (latest/cumulative)

**Example:** `JAM_cumulative_19f12424`

### Why Smart Caching Matters

**Scenario A: Same satellite data**
```bash
# Run 1: Query up to 2025-10-27
# Found dates: [2025-10-22, 2025-10-24, 2025-10-27]
# Cache key: JAM_cumulative_19f12424
# Result: Builds cache (~4 min)

# Run 2: Query up to 2025-10-28 (next day)
# Found dates: [2025-10-22, 2025-10-24, 2025-10-27]  ← SAME!
# Cache key: JAM_cumulative_19f12424  ← SAME!
# Result: Uses cache (~46 sec) ✅ 93% faster!
```

**Scenario B: New satellite observation**
```bash
# Run 3: Query up to 2025-10-29
# Found dates: [2025-10-24, 2025-10-27, 2025-10-29]  ← NEW DATE!
# Cache key: JAM_cumulative_a7b3c5d9  ← DIFFERENT!
# Result: Rebuilds cache with new data (~4 min)
```

### Cache Invalidation
The cache ONLY rebuilds when:
- ✅ New satellite observation date detected
- ✅ Different flood mode (latest ↔ cumulative)
- ✅ Different population raster
- ✅ Different country (ISO3)
- ✅ Cache manually deleted

The cache DOES NOT rebuild when:
- ❌ Query end-date changes but same observations found
- ❌ Script re-run with identical parameters
- ❌ Creating different visualizations from same data

---

## Map Features

All choropleths include:
- **Provenance boundaries** - Thick colored lines showing which observation date each area came from
  - Green = Oldest observation
  - Darker yellow = Middle observation
  - Orange = Newest observation
- **Chronological legend** - Dates sorted oldest to newest (top to bottom)
- **ADM3 boundaries** - Light grey, subtle lines (`#c0c0c0`, linewidth 0.15)
- **Labels** - Semi-transparent white halos (no bounding boxes)
- **No-data areas** - Transparent grey fill (70% opacity)

### Choropleth-Specific Features

**Affected Population Maps:**
- Color ramp: White → Light Yellow → Orange → Red
- Uses `population_adjusted` (rounded up integer values)
- Label format: `Name\n(XX)` - integer population

**Flooded Area Maps:**
- Color ramp: White → Light Blue → Dark Blue
- Units: km² (pixels × 400 m² ÷ 1,000,000)
- Label format: `Name\n(X.XX km²)` - 2 decimal places
- Only plots ADM3 polygons with flooding (allows grey no-data to show)

---

## Population Adjustment Methodology

### The Problem
- **GHSL resolution:** 100m × 100m pixels = 10,000 m² per pixel
- **GFM resolution:** 20m × 20m pixels = 400 m² per pixel
- **Pixel area ratio:** (100/20)² = **25**

When we sample GHSL population at a 20m flood pixel, we get the population for an area **25× larger** than the actual flood pixel.

### The Solution: 3-Tier System

Each flood point stores three population values:

1. **`population_raw`** (float)
   - Raw GHSL value from 100m pixel
   - Represents population in 10,000 m²
   - Example: 63.3 people

2. **`population_adjusted_raw`** (float)
   - `population_raw ÷ 25`
   - Proportional estimate for 20m pixel (400 m²)
   - Example: 63.3 ÷ 25 = 2.53 people

3. **`population_adjusted`** (integer) ← **Used by default**
   - `math.ceil(population_adjusted_raw)`
   - Rounded up to nearest whole person
   - Example: ceil(2.53) = **3 people**
   - Rationale: A fraction of a person cannot be affected

### Example Calculation

For Jamaica with 419 cumulative flood pixels:
```
Raw GHSL total:           63.3 people
Adjusted raw total:        2.53 people (÷25)
Adjusted rounded total:   64 people (ceil each pixel, then sum)
```

The choropleths use `population_adjusted` to display whole numbers: "59 people affected" instead of "2.53 people affected".

---

## Output File Naming

| File | Description |
|------|-------------|
| `population_provenance_YYYYMMDD.png` | Density heatmap (latest or cumulative, gets overwritten) |
| `choropleth_adm3_YYYYMMDD.png` | Affected pop choropleth (latest or cumulative, gets overwritten) |
| `choropleth_flooded_area_latest_YYYYMMDD.png` | Flooded area km² (latest mode) |
| `choropleth_flooded_area_cumulative_YYYYMMDD.png` | Flooded area km² (cumulative mode) |

**Note:** For production workflows, rename affected pop outputs to distinguish latest vs cumulative modes.

---

## Troubleshooting

### Cache not invalidating when it should
- Check that observation dates actually changed (script prints "All dates found")
- Verify cache key is different between runs
- Manually delete cache: `rm -rf data/cache/JAM_*`

### "WARNING: Could not sample population raster"
- Check blob storage connection
- Verify population raster path exists in Azure blob
- Check that `ocha_stratus` is configured correctly

### ADM3 boundaries not loading
- Ensure admin boundaries uploaded to blob: `projects/ds-flood-gfm/raw/codab/jam/`
- Check blob storage stage (`dev` vs `prod`)
- Verify `load_admin_from_blob()` function in `src/ds_flood_gfm/geo_utils.py`

### Choropleths showing all white
- Check that flood pixels actually have population > 0
- Verify population sampling succeeded (check console output)
- Ensure colormap vmin/vmax are set correctly

---

## Performance Benchmarks

**With all optimizations (cache + blob storage + exactextract):**
- First run (cache miss): ~4-5 minutes
- Subsequent runs (cache hit): ~46 seconds
- Flooded area script: ~30-40 seconds

**Speed breakdown (cache hit):**
- Load cache: ~5 seconds
- Load admin boundaries from blob: ~2.5 seconds
- Exactextract zonal stats (777 polygons): ~3 seconds
- Create visualization: ~35 seconds

**vs. Without optimizations:**
- HTTP admin boundaries: ~91 seconds (40× slower)
- Rasterstats zonal stats: ~5+ minutes (100× slower)
- No cache: ~4-5 minutes every run

---

## Dependencies

Key libraries:
- `stackstac` - Build xarray stacks from STAC items
- `pystac-client` - Query STAC API
- `geopandas` - Spatial data handling
- `exactextract` - Fast zonal statistics
- `ocha_stratus` - Azure blob storage integration
- `matplotlib` - Plotting and visualization
- `adjustText` - Non-overlapping labels (like R's ggrepel)

---

## Future Enhancements

Potential improvements:
- [ ] Unified script to generate all 4 maps in one command
- [ ] Command-line option to specify output directory
- [ ] Support for multiple countries in batch mode
- [ ] Automated file naming to distinguish latest vs cumulative
- [ ] Optional PDF export for reports
- [ ] Interactive HTML maps with hover tooltips


## Gotchas 

- If geometry footprint of STAC intersects AOI it is returned by the STAC search call. But as the STAC footprint is bigger thant the actual sentinel tile it does not mean that the actual flood data intersects. So if I run: 

```
 source ~/.zshrc && PYTHONUNBUFFERED=1 uv run python scripts/02_generate_affected_population_choropleths.py --end-date 2025-10-29 --n-latest 3 --iso3 JAM --flood-mode cumulative
```

what happens is that the STAC search returns 3 dates (2025-10-24, 2025-10-27, 2025-10-29) as the footprints intersect the AOI, but when building the xarray stack only 2 dates have actual flood data intersecting the AOI (2025-10-24, 2025-10-27). So the cache key is built with 3 dates but the actual data only has 2 dates. This causes confusion as the cache key indicates there should be 3 dates but the data only has 2. Causes confusion in provenance date colors as well.
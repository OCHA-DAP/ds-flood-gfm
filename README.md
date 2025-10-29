# GFM Flood Monitoring

Global Flood Monitoring (GFM) analysis pipeline for generating flood extent maps and affected population estimates.

## Quick Start

### Prerequisites
- Python 3.12.4 with `uv` for dependency management
- Azure blob storage access (for admin boundaries and population data)

### Setup
```bash
# Install dependencies
uv sync

# Install Jupyter kernel (optional)
uv run python -m ipykernel install --user --name=ds-flood-gfm --display-name="Python (ds-flood-gfm)"
```

### Run Daily Updates

For Jamaica (JAM), Haiti (HTI), or Cuba (CUB):

```bash
# Generate affected population maps (latest mode)
uv run python scripts/02_generate_affected_population_choropleths.py \
  --end-date 2025-10-29 \
  --n-latest 4 \
  --iso3 JAM \
  --flood-mode latest

# Generate affected population maps (cumulative mode)
uv run python scripts/02_generate_affected_population_choropleths.py \
  --end-date 2025-10-29 \
  --n-latest 4 \
  --iso3 JAM \
  --flood-mode cumulative
```

**Parameters:**
- `--end-date`: Query satellite data up to this date (format: YYYY-MM-DD)
- `--n-latest`: Number of most recent observations to include
- `--iso3`: Country code (JAM, HTI, or CUB)
- `--flood-mode`: Either `latest` (most recent provenance) or `cumulative` (ever flooded)

**Runtime:**
- First run: ~4-5 minutes for Jamaica (downloads and processes satellite data)
  - Haiti: ~5-10 minutes
  - Cuba: ~45-50 minutes (143M pixels)
- Subsequent runs: ~46 seconds (uses cached data if satellite observations unchanged)

**Outputs:**
Maps saved to `experiments/` directory:
- `population_provenance_YYYYMMDD.png` - Flood density heatmap with data provenance
- `choropleth_adm3_YYYYMMDD.png` - Affected population by admin division

### Generate Flooded Area Maps (Optional)

After running script 02 in **both** flood modes, generate flooded area choropleths:

```bash
uv run python scripts/03_generate_flooded_area_choropleths.py \       
  --end-date 2025-10-29 \
  --n-latest 4 \
  --iso3 JAM
```

**Prerequisites:**
- Must have run script 02 in both `latest` AND `cumulative` modes first
- Script reads cached data from both modes to generate flooded area maps

**Runtime:** ~30-40 seconds

**Outputs:**
- `choropleth_flooded_area_latest_YYYYMMDD.png` - Flooded area (km²) from latest mode
- `choropleth_flooded_area_cumulative_YYYYMMDD.png` - Flooded area (km²) from cumulative mode

### Understanding the Cache

The script caches processed data based on actual satellite observation dates (not query end-date). If you run with `--end-date 2025-10-29` but the latest satellite image is still from 2025-10-27, the cache from yesterday will be reused.

Cache rebuilds automatically when:
- New satellite observations are available
- Different flood mode selected (latest vs cumulative)
- Different country selected

**Manual cache clearing:**
```bash
# Clear cache for specific country
rm -rf data/cache/JAM_*
rm -rf data/cache/HTI_*
rm -rf data/cache/CUB_*
```

### Flood Modes Explained

**Latest Mode:**
- Shows only pixels from their most recent observation date
- Ignores older data if area was re-observed
- Best for "current flooding" estimates
- Avoids counting stale flood pixels that may have receded

**Cumulative Mode:**
- Shows all pixels ever flooded across all dates
- Spatial union of all observations
- Best for "total flood extent" during event
- Useful for disaster impact assessment

### Color Coding

All maps use **data provenance colors** showing which satellite observation each area came from:
- Red: Oldest observation
- Orange/Yellow: Middle observations
- Green: Newest observation

This helps identify data currency - green areas have the freshest information.

## Country Configuration

Supported countries in [src/ds_flood_gfm/country_config.py](src/ds_flood_gfm/country_config.py):
- JAM (Jamaica)
- HTI (Haiti)
- CUB (Cuba)

Each country has pre-configured:
- Bounding box for STAC queries
- Admin level for choropleths (ADM2 or ADM3)
- Legend placement on maps
- COD-AB admin boundary paths

## Project Structure

```
ds-flood-gfm/
├── scripts/
│   ├── 01_download_codab_to_blob.py     # One-time admin boundary setup
│   ├── 02_generate_affected_population_choropleths.py  # Main analysis script
│   └── 03_generate_flooded_area_choropleths.py  # Generate flooded area maps
├── src/ds_flood_gfm/                    # Python modules
│   ├── country_config.py                # Country-specific settings
│   ├── geo_utils.py                     # Geospatial utilities
│   └── ...
├── data/
│   ├── cache/                           # Cached flood data (auto-generated)
│   ├── gfm/                             # Raw GFM data
│   └── ghsl/                            # Population data
├── experiments/                         # Output maps and analysis
└── book_gfm/                            # Quarto documentation
```

## Troubleshooting

**No new data showing up:**
- Check if satellite has actually acquired new imagery
- Verify `--end-date` is set to today's date or later
- Look for "All dates found:" in console output to see actual observation dates

**Out of memory (Cuba/large areas):**
- Cuba processing requires ~16GB RAM
- Memory optimizations already implemented (512×512 chunking, vectorized operations)
- If still failing, reduce `--n-latest` from 3 to 2

**Blob storage errors:**
- Ensure Azure credentials are configured
- Check `ocha_stratus` package is installed: `uv add ocha_stratus`
- Verify blob paths in `country_config.py`

## Documentation

Detailed documentation available in:
- [scripts/README.md](scripts/README.md) - Script usage, caching, performance benchmarks
- [CACHING_GUIDE.md](CACHING_GUIDE.md) - Deep dive into cache system
- [book_gfm/](book_gfm/) - Quarto analysis notebooks

## Data Sources

- **Flood data:** GFM STAC API (https://services.eodc.eu/browser/#/v1/collections/GFM)
- **Population:** GHSL 2025 (100m resolution, stored in Azure blob)
- **Admin boundaries:** COD-AB via HDX (stored in Azure blob)

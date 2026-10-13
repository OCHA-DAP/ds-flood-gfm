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
  - Haiti: ~15-20 minutes
  - Cuba: ~45-50 minutes (143M pixels)
- Subsequent runs: ~46 seconds (uses cached data if satellite observations unchanged)

**Outputs:**
Maps saved to `outputs/plots/` directory:
- `{ISO3}_population_provenance_{YYYYMMDD}.png` - Flood density heatmap with data provenance
- `{ISO3}_population_{mode}_adm3_{YYYYMMDD}.png` - Affected population by admin division

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
Maps saved to `outputs/plots/` directory:
- `{ISO3}_choropleth_flooded_area_latest_{YYYYMMDD}.png` - Flooded area (km²) from latest mode
- `{ISO3}_choropleth_flooded_area_cumulative_{YYYYMMDD}.png` - Flooded area (km²) from cumulative mode

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

## Flood fusion v1 (daily P(flooded), 0.01°)

A daily global-capable raster product on a **0.01° (~1 km) WGS84 grid**, restricted to the floodable domain (HAND < 15 m, permanent water removed). Three layers per day:

1. **P(flooded)**, from 0 to 1.
2. **Uncertainty**: the absolute log-odds, plus the count of sources that observed the cell that day.
3. **Days since the last direct observation.**

It fuses three independent sensor families:
- **Copernicus GFM** (Sentinel-1 SAR).
- **FloodScan OR GFDS**, merged into one passive-microwave vote.
- **NOAA VIIRS flood maps** (optical). These stand in for MODIS, which needs an Earthdata login.

The fusion is per-stratum naive Bayes in log-odds, with sensitivity and specificity learned from Copernicus EMS and UNOSAT flood maps. Yesterday's log-odds carry over to today with a 0.9 decay toward the prior.

> **Resolution.** v1 is deliberately coarser than GFM's 20 m. A cell counts as "flooded" only when at least 30% of its area is flooded, so small or narrow floods are not represented. Use GFM directly when you need detail; use this product when you need a daily, gap-filled estimate. v1's probabilities are not calibrated: read them as a ranking score.
>
> **Licence.** FloodScan is a commercial input, and outputs derived from it may not be shared outside OCHA without AER approval.

**v1 status:**
- **Validation (49 held-out gold events):** the product is built and tested end to end, but as specified it does **not** beat the best single source in any stratum. Its probabilities are strongly over-confident: reliability slope about 0.01 against a target of 0.8–1.2.
- **Where a single product observed, it beats fusion.** Recall · precision on GFM-observed cells: GFM 0.45 · 0.94 vs fusion 0.34 · 0.63. On VIIRS-observed cells: VIIRS 0.75 · 0.60 vs fusion 0.46 · 0.57.
- **One lead, not yet established.** In open, snow-free land, a decision threshold learned on training events (P ≥ 0.02) raised held-out recall from 0.38 to 0.53 at precision 0.56 (VIIRS: 0.42 and 0.62). That rests on one run, with no confidence interval. A per-stratum recalibration improved calibration there but did not reach the 0.8–1.2 target.
- **Why it is over-confident (measured 2026-10-13).** On cells the gold maps call dry, the sources' false alarms coincide 4–18 times more often than the model's independence assumption allows. An agreement-based accuracy table (plan v2, ADR 0002) fails for the same reason, so the model keeps the gold table.
- **Inputs are ending.** GFDS stopped on 2026-09-14, and FloodScan outside Africa ends on 2026-03-31 on our blob.

Where fusion is better than single products, and where it is not, is written up in **[docs/final_report.md](docs/final_report.md)**.

### Running it (a junior's walk-through)

```bash
uv sync --extra dev            # environment
make test                      # 45 synthetic tests: must pass before anything touches real data
make phase0                    # snow climatology (CDS key in ~/.cdsapirc) + static tile indexes
make phase1                    # gold events -> event cubes -> per-source skill, oracle + coverage gate
make phase2                    # stratified random sample cube + static rasters
make phase3                    # accuracy table (results/accuracy_table.csv) + prior (results/prior_v1.csv); agreement table + comparison
make phase4                    # engine tests (gate)
make phase5                    # held-out validation, daily COGs for every held-out event window
make phase6                    # mirror to dev blob + MANIFEST.json
make daily DATE=2026-09-10     # one operational day for the AOIs in configs/fusion.yaml
```

- **Timing.** Every step is resumable. The network-bound steps take hours on a laptop:
  - the GFDS prefetch respects a limit of 1 request per second;
  - GFM, VIIRS and FloodScan reads are windowed.
- **Logs** go to `data/fusion/logs/`.
- **Credentials:** `DSCI_AZ_BLOB_*` environment variables (ocha-stratus) and `~/.cdsapirc`. Nothing is in code.

| what | file |
|---|---|
| every threshold and path | `configs/fusion.yaml` |
| why each choice was made | `docs/decisions.md`, `docs/assumptions.md`, `docs/decisions/0001-*.md` |
| what data exists and its limits | `docs/inventory.md` |
| what was in the repo before | `docs/repo_audit.md` |
| gate result | `docs/feasibility.md` |
| the trained model | `results/accuracy_table.csv`, `results/prior_v1.csv` |
| validation | `results/phase5_*.csv`, `results/reliability_*.png`, `docs/final_report.md` |
| scheduled job | `databricks.yml` (job "Flood Fusion Daily"), `scripts/fusion/daily_job.py` |

### Reading an output

```python
import rioxarray
da = rioxarray.open_rasterio("data/fusion/out/daily/<window>/<window>_2024-09-15.tif")
p_flood, abs_logodds, n_sources, days_since_obs = da  # NaN outside the floodable domain
```

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

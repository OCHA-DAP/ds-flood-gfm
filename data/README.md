# Data Directory

This directory contains datasets for flood monitoring analysis:

## Structure

```
data/
├── gfm/          # Global Flood Monitoring data from EODC STAC API
│   └── <item_id>/
│       ├── <item_id>.tif         # Ensemble flood extent GeoTIFF
│       └── <item_id>_metadata.json # STAC item metadata
├── ghsl/         # Global Human Settlement Layer population data
│   ├── <tile_name>.zip           # Downloaded GHSL tiles
│   ├── <tile_name>/              # Extracted GHSL data
│   └── population_equi7_20m.tif  # Processed population raster
└── README.md     # This file
```

## Data Sources

### GFM (Global Flood Monitoring)
- **Source**: Copernicus Emergency Management Service
- **API**: https://stac.eodc.eu/api/v1
- **Collection**: GFM
- **Resolution**: 20m
- **Coordinate System**: Equi7Grid (e.g., EPSG:27703 for Asia)
- **Assets**: ensemble_flood_extent, tuw_flood_extent, dlr_flood_extent, etc.

### GHSL (Global Human Settlement Layer)
- **Source**: European Commission Joint Research Centre
- **Dataset**: GHS-POP R2022A (2020 population)
- **Original Resolution**: 100m
- **Processed Resolution**: 20m (resampled with /25 division)
- **URL**: https://human-settlement.emergency.copernicus.eu/

## Usage Notes

1. **Data Directory**: This entire `data/` directory is git-ignored
2. **Population Data**: Real GHSL data should replace sample/test files
3. **Coordinate Systems**: All data is harmonized to Equi7Grid for analysis
4. **File Sizes**: GFM tiles are typically 300×300 km (15000×15000 pixels)

## Getting Started

Run the downloaders to populate this directory:
```python
# Download GFM flood data
from src.gfm_downloader import GFMDownloader
downloader = GFMDownloader()
items = downloader.search_pakistan_floods()
downloader.download_item_assets(items, "./data/gfm")

# Download GHSL population data
from src.ghsl_downloader import GHSLDownloader
ghsl_downloader = GHSLDownloader()
ghsl_downloader.download_pakistan_population()
```
# Legacy GFM modules

Pre-STAC / "latest N days" era modules, kept for reference. **None are imported by
the production pipeline** (`scripts/01`–`04`), which now runs entirely through
`src/ds_flood_gfm/datasources/gfm.py` plus `constants`, `country_config`, and
`geo_utils`.

These were moved here out of the importable package (`src/ds_flood_gfm/`) so they
no longer sit on the import path, while remaining available if any pre-STAC
behaviour ever needs to be revisited.

| file | original location | superseded by |
|------|-------------------|---------------|
| `download_latest.py` | `src/ds_flood_gfm/download_latest.py` | STAC-based `datasources/gfm.py` (replaces the `LatestGFMDownloader` "latest N days" download path) |
| `gfm_rest_client.py` | `src/ds_flood_gfm/gfm_rest_client.py` | the move to the GFM **STAC** API; this is the older REST auth/download client |
| `pipeline_latest_3days.py` | `src/ds_flood_gfm/pipeline_latest_3days.py` | `scripts/04_generate_flood_polygons.py` + `datasources/gfm.py` + `write_cog.py` |
| `stac_gfm_processor.py` | `src/ds_flood_gfm/stac_gfm_processor.py` | the functional rewrite in `datasources/gfm.py` (replaces the `STACGFMProcessor` class) |
| `stac_spatial_filter.py` | `src/ds_flood_gfm/stac_spatial_filter.py` | spatial intersection folded into `gfm.query_gfm_stac()` |
| `blob_utils.py` | `src/ds_flood_gfm/blob_utils.py` | `ocha-stratus` (`stratus.*`) for blob upload/download |
| `test_spatial_filter.py` | `reprex/test_spatial_filter.py` | reprex for `stac_spatial_filter.py`; moved alongside it (import updated to the co-located module) |

> `gfm_rest_client.py` is the only non-STAC data path. Retained as a fallback in
> case the STAC catalog ever lacks auth/history/full-res assets the REST API exposes.

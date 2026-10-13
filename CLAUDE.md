# ds-flood-gfm — read this first

This repo has two parts.

1. **GFM exposure pipeline** (`scripts/01..04`, `src/ds_flood_gfm/*.py` outside `fusion/`). It turns Copernicus GFM flood extents into affected-population maps per country. See README "Quick Start".
2. **Flood fusion v1** (`src/ds_flood_gfm/fusion/`, `scripts/fusion/`, `configs/fusion.yaml`, `tests/`). This is a daily 0.01° P(flooded) product combining GFM (SAR), FloodScan or GFDS (microwave) and VIIRS (optical), validated against Copernicus EMS and UNOSAT maps. Start with `docs/final_report.md`, then `docs/decisions.md`.

## Flood fusion: how to work in it

- **Config, not code.** Every threshold, window, path and source switch is in `configs/fusion.yaml`. The reasons are in `docs/decisions.md` (D-NN numbers are cited in docstrings).
- **One entry point per phase:** `make phase0` … `make phase6`, plus `make test` and `make daily`. Each runs `scripts/fusion/phase*.py`, and each step is resumable: re-running skips work whose outputs exist.
- **Value rule (never break it):** a product raster cell is 1 = observed flood, 0 = observed dry, 255 = could not observe. Unobserved is never 0. Each adapter in `fusion/sources/` documents where its NoData comes from.
- **Fail loudly.** Three states are kept apart everywhere:
  - upstream absence (no scene, or GDACS 403 for a day) is recorded;
  - a fetch failure raises;
  - a real dry is 0.

  No broad `except`. Opt-in exceptions are named flags (`--continue-on-error`, `--cold-start`, `on_missing_asset: drop_scene`) and are recorded in outputs.
- **Tests before real data.** `make test` (37 synthetic tests) must pass, and `phase4_engine.py` refuses to run if they fail.
- **Held-out events are sacred.** `results/event_split.csv` is frozen. Phases 1–3 use `split == train` only. Never fit anything on `holdout`.

## Data locations

| what | where |
|---|---|
| local cache (gitignored) | `data/fusion/` (cubes, static, GFDS day store, outputs, logs) |
| blob outputs | dev account, container `projects`, `ds-flood-gfm/processed/flood_fusion_v1/` (daily COGs, static, cubes, results, `MANIFEST.json`) |
| blob raw inputs | `ds-flood-gfm/raw/{era5_land,gfds,hydrobasins,koppen,natural_earth,tile_index}/` |
| gold labels (read only) | dev `global/copernicus_ems/flood/gold/`, `global/unosat/gold/` (built in ds-geospatial-impact-estimates) |
| FloodScan | Africa: prod `raster/floodscan/daily/v5/processed/`. Asia-Australia/Americas: **dev** `raster/floodscan/daily/v5/raw/*_no_ndt.nc` (to 2026-03-31) |
| credentials | `DSCI_AZ_BLOB_{DEV,PROD}_SAS[_WRITE]` env vars (ocha-stratus); CDS key in `~/.cdsapirc` (snow climatology only) |

## Gotchas learned building v1

- **GDACS (GFDS) returns HTTP 403, not 404, for missing days.** The feed stopped after **2026-09-14**. The adapter treats a 403 as "absent" only after a known-good file reads fine. Keep requests at ≤ 1 per second.
- **GFM:**
  - EODC STAC returns items outside the query bbox, so always re-check each item's bbox.
  - `exclusion_mask` pixels read 0 in the flood layer, so they must be dropped, not counted as dry.
  - Newer `advisory_flags` files have no overviews, and some scenes have no advisory asset at all.
  - Some asset files 404 although STAC lists them (`on_missing_asset`).
- **xarray picks the wrong NetCDF engine.** Without h5py it falls back to scipy/netCDF3, which cannot write uint8. Always pass `engine="netcdf4"`.
- **VIIRS:**
  - The flood map archive has no files for 2021-01 → 2023-07.
  - Its files are uncompressed with one-row strips: read windows, never whole files.
- **Long runs are network-bound.** On the user's laptop: `nice -n 19`, at most 2–3 worker processes, small or cheap units first, and confirm the first unit in the log after launch.
- **Killing a job with a process pool:** kill the whole process tree. Pool workers survive their parent (they reparent to PID 1); check for orphans with `ps -axo pid,ppid,command | grep multiprocessing`.
- **`pgrep -f "<pattern>"` inside a wait loop matches the loop's own shell.** Wait on a PID or a file instead.
- **DAB development mode does not pause an explicit schedule.** Keep `pause_status: PAUSED` in the base definition (D-27).
- **Uploads:** the upload rate fell to about 30 KB/s for part of the v1 session (cause not measured). `phase6_package.py` is resumable (it skips files whose size already matches) and uploads in priority order.

## v1 result in one line

Fusion as specified is mid-pack and over-confident. Where a single product observed, it beats fusion. A threshold learned on training events is a lead in open land, not an established fix (docs/final_report.md §4, §7).

## Not done in v1 (see docs/final_report.md "Open issues")

- Global static layers: build them with `scripts/fusion/static_build.py` or the `flood_fusion_static` Databricks job.
- Prior v2 (GloFAS).
- MODIS (needs an Earthdata login).
- GSW v1.5.
- The 5 giant UNOSAT events and the 166 events beyond the v1 quotas.

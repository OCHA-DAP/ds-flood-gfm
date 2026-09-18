# Flood fusion: learned weighting of global flood layers on a common grid

**Date:** 2026-09-18
**Status:** proposed (design discussion 2026-09-17 → 2026-09-18)
**Driver:** the team has several global flood signals (GFM 20 m SAR extent,
FloodScan and GFDS ~10 km passive microwave, more to come) and no principled
way to combine them. This spec sets up the substrate — a common grid, a
tiered label corpus, one adapter interface per input layer, one feature table
— so that the weights can be learned, every claim of skill is measured
against independent labels, and adding a layer is one adapter plus one
column. The first model is deliberately simple; the substrate is the product.

## Decisions taken during design

- **Prediction grain: fixed 30 arcsec grid (1/120°, ~0.93 km), EPSG:4326**,
  integer row/col identity. Nests exactly 10× inside FloodScan's 300 arcsec
  cells; GHSL 3 arcsec nests 10× inside it. Rejected: GFM-native 20 m (the
  coarse layers are constant over 10 km blocks, 2,500× more rows) and admin
  units (destroys the label information, too few rows).
- **Primary target: water present at acquisition, not flood.** Every product
  subtracts a *different* baseline (FloodScan: permanent lakes only, seasonal
  water counted; GFDS: the pixel's 2009 annual mean; GFM: its own S1 reference
  mask; CEMS: the pre-event image). They agree on Lake Victoria and disagree
  on the Sudd, Lake Chad, the inner Niger delta and the Zambezi floodplain,
  which is where floods matter to us. Training on flood makes the model learn
  baseline differences instead of water signal. Water is also the common
  denominator of the label sources (UNOSAT, Sen1Floods11, GSW, Dynamic World
  all label water). Flood becomes a post-processing step with an explicit,
  configurable baseline (GSW occurrence for permanent, GSW seasonality for
  seasonal-aware). The label table keeps `flood_frac` so a flood-target model
  can be trained for comparison; which is better on the flood metric is an
  experiment, not a decision made here.
- **Label tiers, assigned by sensor class, not by source.** Tier A: analyst-made
  dated extents from SAR or optical imagery of tens of metres or better (CEMS
  gold, UNOSAT gold with `sensor_class` ∈ {sar, optical_vhr, optical_hr},
  Sen1Floods11) train the model. Tier B: automated or coarse water products
  (GFM `ensemble_water_extent`; UNOSAT layers derived from VIIRS/MODIS at
  375 m+, which are 1,829 of its label layers; later Dynamic World, GSW
  monthly history) calibrate the coarse layers only — GFM is never scored
  against itself. Tier C: event
  footprints (groundsource, Global Flood Database, DFO) evaluate at event
  level only. Sources that repackage CEMS (WorldFloods, Kuro Siwo, MMFlood,
  SEN12-FLOOD) are excluded as duplicates.
- **PoC region: Africa** (the only region where all three PoC layers exist in
  our holdings; FloodScan on blob covers −30…60°E, −50…40°N only). The design
  treats a missing layer as `observed = false`, so a global run needs no
  redesign.
- **First model: weighted logistic stacking.** Readable weights, missing
  layers as flags, new layer = one column. Per-layer reliability curves are a
  required diagnostic. Rejected for the PoC: per-layer calibration with
  log-odds sum (assumes conditional independence FloodScan and GFDS violate:
  same radiometers) and gradient boosting (learns geography; try it later on
  the same table).
- **Cross-validation grouped by event code**, so no event is split across
  folds. **Circularity hold-out**: label sets digitised from Sentinel-1 are
  held out to measure GFM's unfair advantage.
- **Non-destructive placement.** New sub-package `src/ds_flood_gfm/fusion/`,
  new `src/ds_flood_gfm/datasources/gfds.py`, new `tests/`. Existing modules
  untouched; the only edit to a shared file is adding scikit-learn to
  `pyproject.toml`.

## Evidence (verified 2026-09-17/18)

| input layer | resolution / archive | permanent water | holdings |
|---|---|---|---|
| GFM (Copernicus, Sentinel-1) | 20 m, 2015 → | `ensemble_water_extent` = flood ∪ reference water (verified: on a Lake Victoria scene 21,560 of 21,733 water pixels were reference water); `ensemble_flood_extent` excludes it; `exclusion_mask` = 1 marks terrain GFM does not evaluate (HAND-based non-floodable plus unreliable areas; 86 % of that hilly scene), and excluded pixels read water = 0, so they are *unevaluated*, not dry; `ensemble_likelihood` is a flood likelihood 0–100 (255 nodata), zero over reference water; `advisory_flags`, three member extents and likelihoods | STAC `stac.eodc.eu`, collection `GFM`; all assets uint8, nodata 255 |
| FloodScan (AER) | 300 arcsec, 1998 → 2026-09-15 | lakes 0.000 (Victoria, Tanganyika); seasonal water 0.04–0.35 (Chad, Mopti) | prod blob `raster/floodscan/daily/v5/processed/`, 10,474 daily 2-band COGs (SFED, MFED), **Africa grid only** |
| GFDS (JRC/DFO) | 0.09°, 1997 → | anomaly vs 2009 annual mean | HTTP, no auth, end-of-life feed; fetch code only in `experiments/` |

| label source | sets | dated to the day | notes |
|---|---|---|---|
| CEMS gold (`global/copernicus_ems/flood/gold`) | 2,715 | 2,495 | 71 % Europe; Africa 172 / Americas 190 / Asia-Oceania 376 precise 2015+; 55 % SAR, 22 % Sentinel-1; flood-only today, valid mask included |
| UNOSAT (HDX catalogue, files on UNOSAT/CERN hosts; own spec in impact-estimates) | 226 flood/cyclone event codes; 11,842 shapefile layers of which ~2,000 flood, ~1,050 water, ~1,400 pre-flood/permanent water, ~1,440 cumulative | date or window; 94 % filename/attribute agreement | South Sudan, Somalia, Madagascar, Sudan, Mozambique, Chad, Bangladesh, Pakistan; VIIRS is the most common label sensor (Tier B), then Sentinel-2, Sentinel-1, Pléiades, WorldView-3 |
| Sen1Floods11 | 446 hand-labelled 10 m chips, 11 events | one S1 date per event (12-feature metadata GeoJSON, verified) | public GCS bucket; Ghana 53, Somalia 26, Nigeria 18, Pakistan 28; water labels; Sentinel-1 chips → GFM hold-out only |
| groundsource_2026 (`global/vector/raw`) | 2.6 M polygons | start/end | median 5 vertices, 376 M km² summed: event footprints, Tier C |

## Out of scope (hooks left in place)

- Inference over arbitrary bbox/day and a served product (the adapters are the
  inference path; nothing else is built now).
- Groundsource / GFD / DFO event-level evaluation (Tier C reader defined,
  not run).
- DEM / HAND post-processing, gradient boosting, Dynamic World and GSW
  monthly labels, Asia and the Americas.
- The UNOSAT archive itself (impact-estimates spec of the same date).

## 1. Grid (`fusion/grid.py`)

`Grid(res_arcsec=30)`: `cell_id(lon, lat) → (row, col)`, `bounds(row, col)`,
`window(bbox) → (row0, row1, col0, col1)`, `transform(window)` for rasterio,
and `to_xarray(window)` coordinates. Pure functions, no I/O, fully unit-tested
including the FloodScan and GHSL nesting invariants.

## 2. Label adapters (`fusion/labels/`)

Interface: `LabelSource.iter_sets(region, date_range) → LabelSet`, where a
`LabelSet` carries `label_source`, `code`, `aoi`, `acq_start`, `acq_end`,
`acq_precision`, `sensor`, `tier`, and three nullable geometries
(`geom_water`, `geom_flood`, `geom_possible`) plus `geom_valid` with
`valid_basis`. Adapters:

- `cems_gold` — reads the CEMS gold on blob; accepts v1 (columns `geometry`,
  `valid_geometry`, `valid_basis`, flood-only; verified 283 partitions) and
  v2, and records which it got. `geom_water` is null on v1.
- `unosat_gold` — reads the UNOSAT gold v2 once it exists.
- `sen1floods11` — reads the hand-labelled chips and their metadata GeoJSON;
  label is water at 10 m, valid = chip footprint minus no-data; tier A but
  tagged `sensor = Sentinel-1`, so it falls in the circularity hold-out.
- `gfm_water` (Tier B) — for a bbox/day, `ensemble_water_extent` as water,
  `exclusion_mask` complement as valid. Only ever used to calibrate FloodScan
  and GFDS.

**Rasteriser** (`fusion/labels/rasterize.py`): area-weighted coverage of each
geometry onto the grid window (exactextract `cell_id` + `coverage` ops over a
cell-id raster, already a dependency; verified to return exact fractions) → `water_frac`, `flood_frac`, `possible_frac`, `valid_frac` per
cell, with the fraction defined over the *valid* area of the cell. Cells with
`valid_frac < min_valid` (default 0.5, recorded in the manifest) are dropped.
Label sets whose acquisition is a window wider than `max_window_days` (default
1) are excluded from training and kept filterable.

## 3. Layer adapters (`fusion/layers/`)

Interface: `Layer.extract(window, day) → xr.Dataset` on the grid window with
the layer's variables and a boolean `observed`. Fetch failure raises with the
URL; upstream absence (no scene, 404 for a day) sets `observed = false` and is
logged as such. Three adapters:

- **GFM** (`layers/gfm.py`, built on the existing STAC query in
  `datasources/gfm.py`): nearest scene(s) within `±gfm_window_days` (default
  2); `gfm_dt_days`; `gfm_water_frac` (from `ensemble_water_extent`, which
  already includes reference water) and `gfm_flood_frac`, both averaged over
  *evaluated* 20 m pixels only; `gfm_likelihood_mean` (a flood likelihood,
  zero over reference water, so it is combined with `gfm_refwater_frac`, not
  read as a water likelihood); `gfm_evaluated_frac` = share of the cell with
  `exclusion_mask == 0` inside the swath. `observed` = evaluated fraction ≥
  0.5. Because the exclusion mask carries HAND-based non-floodable terrain,
  GFM's observed fraction will be low in hilly areas; that is a property of
  the product and is reported per label set, not patched.
- **FloodScan** (`layers/floodscan.py`): `fs_sfed`, `fs_mfed` at the day
  (exact block value thanks to nesting), `fs_sfed_3d_max`. Reads the prod
  COGs via ocha-stratus; a missing day raises (the archive is complete, a
  gap is a real problem).
- **GFDS** (`layers/gfds.py`, on a new `datasources/gfds.py` following
  `docs/superpowers/plans/2026-08-17-gfds-prototype.md`: URL builder, both
  nodata sentinels, scale factors, windowed HTTP range reads, 1 req/s,
  resumable cache): `gfds_signal`, `gfds_anom_sigma` against the static
  baseline, `gfds_anom_4d` trailing mean; 404 → `observed = false` recorded
  with the date.
- **Static context** (`layers/gsw.py`): JRC Global Surface Water v1.4
  occurrence and seasonality, block-averaged to the grid from the public 10°
  tiles on `storage.googleapis.com/global-surface-water/downloads2021/`
  (verified reachable; no Earth Engine needed): `gsw_occurrence_mean`, `gsw_seasonality_mean`,
  `gsw_permanent_frac` (occurrence ≥ 90 %). Used as features and as the
  stratification key for evaluation.

## 4. Feature table (`fusion/features.py`)

One Parquet table partitioned by `label_source` and `code`, one row per
(cell, label set): grid keys and centroid; label columns (`label_source`,
`code`, `aoi`, `label_day`, `acq_start`, `acq_end`, `acq_precision`,
`label_sensor`, `label_tier`, `water_frac`, `flood_frac`, `possible_frac`,
`valid_frac`); per-layer features and `*_observed` flags as listed above;
GSW context; `run_id`. A manifest JSON per run records adapter versions,
parameters (`min_valid`, `gfm_window_days`, …), label sets attempted, label
sets skipped and why, and per-layer observed rates. Written under
`projects/ds-flood-gfm/fusion/features/run={run_id}/` (dev) with a local
cache under `data/fusion/` (gitignored).

Extraction is idempotent per label set (skip if the partition exists for the
same manifest hash) and resumable; a failed label set stops the run with the
code named unless `--continue-on-error` is passed, in which case failures are
listed in the manifest, never omitted.

## 5. Model and evaluation (`fusion/model.py`, `fusion/evaluate.py`)

Training rows: Tier A, `acq_precision ∈ {minute, date}`, all PoC layers
observed. Binomial logistic regression via the two-row expansion (y = 1 with
weight `water_frac × valid_frac`, y = 0 with weight `(1 − water_frac) ×
valid_frac`), standardised features, optional interaction of each layer with
`gsw_permanent_frac`. Grouped K-fold by `code`. Outputs: coefficients with
confidence intervals, and predictions per held-out cell.

Baselines that must be beaten for any claim of "better than its parts": each
layer alone after isotonic calibration, and the unweighted mean of calibrated
layers. Metrics: Brier score, AUC and PR-AUC at cell level; flooded-area bias
per label set; all reported **stratified by GSW class** (permanent ≥ 90 %,
seasonal 5–90 %, never < 5 %) so lakes cannot inflate the numbers. Two
required diagnostics: per-layer reliability curves, and the Sentinel-1
hold-out comparison for GFM. A flood-target model (same features,
`flood_frac`) is trained alongside and compared on the flood metric after the
water model's baseline subtraction.

Coarse-layer calibration (Tier B): FloodScan and GFDS reliability curves
against `gfm_water` over all GFM scenes in the PoC region and period,
reported separately, never mixed into Tier A training.

## 6. Reporting

A book chapter `book_gfm/07_fusion_poc.qmd`: the weights, the stratified
metrics, the reliability curves, the circularity result, and the water-vs-
flood-target comparison, all computed from the feature table. Plots via the
team dataviz conventions.

## Code layout

```
src/ds_flood_gfm/fusion/__init__.py
src/ds_flood_gfm/fusion/grid.py
src/ds_flood_gfm/fusion/labels/{base,cems_gold,unosat_gold,sen1floods11,gfm_water,rasterize}.py
src/ds_flood_gfm/fusion/layers/{base,gfm,floodscan,gfds,gsw}.py
src/ds_flood_gfm/fusion/features.py
src/ds_flood_gfm/fusion/model.py
src/ds_flood_gfm/fusion/evaluate.py
src/ds_flood_gfm/datasources/gfds.py
scripts/fusion/{01_build_labels,02_extract_features,03_fit_evaluate}.py
tests/{conftest,test_grid,test_rasterize,test_labels_*,test_layers_*,test_features,test_model}.py
book_gfm/07_fusion_poc.qmd
docs/decisions/0001-...  (ADRs below; bootstraps docs/decisions/ in this repo)
```

## Fail-loud rules

Three states, never conflated: **upstream absence** (no GFM scene in the
window, GFDS 404) → `observed = false`, counted in the manifest; **fetch
failure** → raise with the resource named; **empty content** (a valid cell
with zero water) → a real zero. No broad `except`; no fallback values. A
layer that is *misconfigured* (bad credentials, wrong container) fails at
adapter construction, before any extraction starts.

## Testing

`pytest`, first tests in this repo. Grid: nesting invariants, round-trips,
window maths at the antimeridian and the equator. Rasteriser: synthetic
polygons with known coverage fractions, valid-mask subtraction, threshold.
Layer adapters: synthetic COGs and STAC-item fixtures for GFM aggregation,
FloodScan block lookup, GFDS sentinel masking and scaling; the three states
above each have a test. Features: idempotency and manifest contents. Model:
two-row expansion recovers known weights on synthetic data; grouped folds
never leak a code.

## Phasing

1. Grid + rasteriser + `cems_gold` adapter + label table for Africa (tests
   first). Deliverable: label cells on blob with a manifest.
2. `datasources/gfds.py` + the three layer adapters + GSW context + feature
   table over the Africa CEMS sets.
3. Model, baselines, stratified evaluation, book chapter. Go/no-go on the
   "better than its parts" claim with the CEMS-only labels.
4. `unosat_gold` and `sen1floods11` adapters once the UNOSAT gold exists;
   re-run 2–3. Tier B coarse-layer calibration.
5. Later: Dynamic World / GSW monthly, Tier C event evaluation, boosting,
   inference path, other regions.

## ADRs to write with this work (bootstrap `docs/decisions/` here)

- `0001-fusion-grid-30-arcsec.md` — grain and nesting rationale; rejected 20 m
  and admin.
- `0002-water-first-target-flood-by-baseline.md` — the baseline-divergence
  evidence; flood as configurable post-processing; rejected flood-direct as
  primary.
- `0003-label-tiers-and-circularity.md` — A/B/C tiers, GFM-as-label for the
  coarse layers only, Sentinel-1 hold-out; excluded CEMS-derived datasets.

## Open questions

- The GHSL 3 arcsec population COG named in `country_config.py`
  (`ghsl/pop/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif`) returned 404 on
  the prod `raster` container on 2026-09-18. Not needed for the PoC, but the
  nesting claim is asserted by a grid unit test against the GHSL product
  definition, and the exposure pipeline that reads it may be broken.

- `min_valid` and `gfm_window_days` defaults are guesses; phase 1–2 report
  their sensitivity.
- Negative-cell volume: label footprints are mostly dry; whether to cap cells
  per label set (stratified) or weight, decided on phase-1 counts.
- FloodScan Africa grid edge: label sets straddling the grid boundary get
  `fs_observed = false` for the outside cells; confirm no PoC set straddles.

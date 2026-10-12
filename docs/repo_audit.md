# Repository audit for flood fusion v1

Date: 2026-10-12. Branch audited: `docs/flood-fusion-spec` at `638a403`. Fusion work continues on `feat/flood-fusion-v1`.

This audit covers what the repo held before any fusion code was written. It records what runs, what is half-done, and what conflicts with the fixed fusion design. For each item it says whether fusion v1 reuses it, extends it or replaces it.

## 1. What exists

### Production GFM exposure pipeline (runs; unrelated to fusion)
| path | what it does | state |
|---|---|---|
| `src/ds_flood_gfm/datasources/gfm.py` | STAC query (`https://stac.eodc.eu/api/v1`, collection `GFM`). `stackstac` daily `max` composite in two modes, `latest` and `cumulative`. Tiled processing, polygonising and a provenance raster. | Runs. It is the only live GFM path. Used by `scripts/02`/`04`, book ch02 and `experiments/gfds_downscaling_poc/03`. |
| `scripts/01..04_*.py` | Pipeline that turns CODAB plus GFM plus GHSL into affected-population and flooded-area choropleths for JAM/HTI/CUB. | Runs, per the README and book ch04. `02` (1,846 lines) is a copy of `experiments/flood_provenance_analysis_cached.py`. |
| `src/ds_flood_gfm/geo_utils.py`, `country_config.py`, `constants.py` | Admin loaders, per-country settings, `PROJECT_PREFIX = "ds-flood-gfm"`. | Used. |
| `flood_exposure.py` | Marimo exposure app. | ARCHITECTURE.md records a hard-coded EPSG:32618 bug. |

### Dead or duplicated code (not imported by anything live)
- Seven temporal-compositor implementations: `cloud_`, `stac_`, `simple_`, `stac_daily_`, `virtual_temporal_compositor.py`, `temporal_compositor.py` and `east_africa_gfm_poc.py`. All are superseded by `datasources/gfm.create_flood_composite`. Their `main()`s point at `data/gfm/somalia_example/`, which does not exist.
- `gfm_downloader.py`, `simple_gfm_downloader.py`, `ghsl_downloader.py`, `affected_population.py` (its docstring has wrong GFM value semantics), and `write_cog.py` (imported only from `source_legacy/`).
- `source_legacy/gfm/*`: retired. Its README claims spatial filtering was folded into `query_gfm_stac()`, which is not true. `reprex/` documents that EODC STAC returns items far outside the query bbox.
- `experiments/flood_jamaica_v1..v3.py` and `flood_provenance_analysis*.py` are iterative drafts of `scripts/02`.

### GFDS / FloodScan research (runs from local caches; the base for the microwave adapters)
- `experiments/gfds_nga2022/fetch_gfds_nga.py`: windowed `/vsicurl/` reads of GFDS daily `signal` and `mag` GeoTIFFs. It retries on failure and keeps a resumable `.npy` cache.
  - A 404 writes a `.missing` marker. The live server actually answers **403** for days that do not exist (verified 2026-10-12), so that marker never fires.
- `experiments/gfds_nga2022/fetch_sfed_nga.py` reads FloodScan from **dev** `global/raster/cogs/aer_area_300s_{YYYYMMDD}_v05r01.tif`. The KB and this run use the **prod** `raster/floodscan/daily/v5/processed/` archive instead.
- `experiments/gfds_downscaling_poc/01..07`: calibrated and FloodScan-independent GFDS fractions, plus downscaling to 83 m.
  - `06_independent_fractions.py` holds the **R1 physics fraction**, `f = f_perm + (s_dry − s)/(s_dry·K)` with `K = 0.35`. Fusion v1 reuses it.
  - Scripts `05` and `07` `exec()` the body of `04`, so they cannot be imported.
- The local `data/gdacs_gfds/nga2022/` cache holds 157 signal days, the baselines and 214 SFED days. It was used for the R1 check in `docs/decisions.md` (D-07).

### Documents
- `docs/superpowers/specs/2026-09-18-flood-fusion-design.md` is the earlier fusion design, status *proposed*, PR #33. Its conflicts are listed in §3.
- `docs/superpowers/plans/2026-08-17-gfds-prototype.md` was never executed (there is no `datasources/gfds.py`). Its facts are reused: sentinels `-32000` and int32-min, signal scale 1e6, flood lowers the signal, and the 1 req/s politeness limit.
- `docs/gdacs-gfds-assessment.md`, `docs/gfds-nga2022-findings.md`, `docs/gfds-downscaling-poc.md`, and book chapters 05 and 06 hold the GFDS evaluation. Its standing verdicts are that GFDS is a FloodScan analogue at admin level, that it is end-of-life, and that the 10 km calibrated fraction is the useful product.
- `ARCHITECTURE.md` (exposure monitoring, not implemented) and `OPTIMIZATION_STRATEGY.md` (Philippines memory fixes) have nothing to do with fusion.

### Configuration, tooling, tests
- Before this work the only config was `pyproject.toml`, `book_gfm/_quarto.yml` and the `country_config.py` dict. There was **no YAML config, no `tests/` directory and no pytest configuration**.
- `black` is configured and `ruff` is not, while the KB's `python-tooling.md` recommends ruff. Fusion v1 does not change the formatter; that is out of scope.
- `.gitignore` ignores `*.csv`, `*.json`, `*.parquet`, `CLAUDE.md` and `uv.lock`. Fusion v1 adds narrow negations so its deliverables can be committed (D-02).
- `.env.example` names only `GFM_API_*`. The `DSCI_AZ_BLOB_*` variables that `ocha-stratus` reads are environment variables, per the KB.

## 2. What runs, what is half-done

| component | runs? | notes |
|---|---|---|
| GFM exposure scripts 01–04 | yes | Not touched by fusion. |
| GFDS Nigeria experiments | yes, from cache | The cache has silent gaps (Jul 19/31 and Aug 1–14 have no `.missing` markers), so that fetch never finished. |
| GFDS downscaling POC | yes, from cache | `exec()` coupling; single AOI. |
| Fusion (spec 2026-09-18) | **no code existed** | The spec itself says "No fusion code exists yet". |
| `datasources/gfds.py` (plan 2026-08-17) | never written | Fusion v1 writes `fusion/sources/gfds.py` instead (§4). |

## 3. Conflicts with the fixed v1 design, and the resolution

| # | existing artefact | conflict | resolution |
|---|---|---|---|
| C1 | Spec §1: 30-arcsec grid, chosen because it nests in FloodScan 300″ and GHSL 3″ | Fixed decision: 0.01° grid. 0.01° nests 9×9 in GFDS 0.09° (cell edges align: `j = 9k − 5`) but **not** in FloodScan (8.33×). | **Replaced.** FloodScan is sampled at the containing 300″ cell for each 0.01° cell centre. Recorded as ADR 0001 and D-03. |
| C2 | Spec: primary target *water*, flood as post-processing | Fixed: 1 = observed flood. | **Replaced.** All sources and gold are flood-only (GFM `ensemble_flood_extent` excludes reference water; CEMS and UNOSAT flood classes). Permanent water is removed by the GSW ≥ 90 % mask. The spec's baseline-divergence argument survives as a stated limitation in the final report. |
| C3 | Spec: weighted logistic stacking; explicitly rejected log-odds sum because FloodScan and GFDS share radiometers | Fixed: naive Bayes in log-odds with an OR-merged microwave vote. | **Replaced.** The OR-merge is the answer to the spec's independence objection. Independence is now assumed only across SAR, microwave and optical. |
| C4 | Spec: label fractions via exactextract; `observed` = evaluated fraction ≥ 0.5 | Fixed: binarise flooded fraction at 0.30. | **Extended.** Fractions are still computed (rasterised at 0.001° and block-averaged, the 24 Sep leaning) and then binarised at 0.30 for gold, GFM and FloodScan. |
| C5 | Spec: Africa PoC, GSW-class strata | Fixed: global, land cover × snow season. | **Replaced.** |
| C6 | `create_flood_composite(mode="cumulative")` writes 0 for unobserved pixels; no code applies `exclusion_mask` | Fixed: never store 0 for an unobserved cell. | **Not reused for fusion.** The fusion GFM adapter reads `exclusion_mask` and nodata 255 directly. The exposure pipeline is left as is (out of scope; noted as an open issue). |
| C7 | GSW permanent water at occurrence > 80 (`02_build_prior.py`) | Fixed: ≥ 90. | Fusion uses ≥ 90. The POC is left untouched. |
| C8 | GFDS has no native flooded fraction; existing calibrations either use FloodScan (R0, circular) or a 4-day mean | The fixed OR-merge needs a daily flood/dry call. | **Extended.** Daily R1 fraction against GFDS's own static baseline, binarised at 0.30 (D-07). |
| C9 | No YAML config, no tests | Fixed: config-driven, tested. | **New.** `configs/fusion.yaml`, `tests/`, `Makefile`. |

## 4. What v1 reuses and what it adds

- **Reused:**
  - The STAC endpoint, collection and asset names from `datasources/gfm.py` (reimplemented as per-scene windowed reads; `stackstac` is not used because it hides nodata and exclusion semantics).
  - GFDS URLs, sentinels and scale from the experiments.
  - The R1 physics formula.
  - `PROJECT_PREFIX` from `constants.py`.
  - `ocha-stratus` for all blob access.
- **New package:** `src/ds_flood_gfm/fusion/`. Existing modules are untouched. The only shared-file edits are `pyproject.toml` (dependencies), `.gitignore` (negations), `README.md` (a fusion section) and the new `CLAUDE.md`.
- **Left alone:** the exposure scripts, the experiments and the book. The dead compositor modules are listed here for a later cleanup; deleting them is outside the scope of this work.

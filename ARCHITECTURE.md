# Architecture & Direction

> Status: **working draft** — captures decisions and open questions as of the
> 2026 architecture review. Sections marked **[OPEN]** are not yet decided.

## Purpose

This document records *where the project is going* and *why*, so that changes can
be reviewed against an agreed direction rather than evaluated one diff at a time.
It is the big-picture companion to `OPTIMIZATION_STRATEGY.md` (which covers
pipeline performance).

---

## 1. Framing: from one-off tool to live monitoring

The project has so far been used as a **one-off tool** — run by hand for a country
and date, producing maps and CSVs. The intent is to evolve it into a **live flood
monitoring system**: AOIs are processed repeatedly as new GFM observations arrive,
results accumulate into a time series, and consumers (DS team today, a dashboard
soon) **pull** from a stable store rather than re-running a notebook.

### The core realization

Most of the value users actually see is **population exposed per admin unit** — and
today that number is *not* produced by the pipeline. The pipeline stops at flood
polygons + a provenance raster; the exposure calculation lives **outside** the
pipeline, in an interactive marimo app (`flood_exposure.py`). That makes the most
valuable output non-reproducible, non-automatable, and untested.

**Direction:** make exposure a first-class, headless pipeline output written to a
stable store. The map/app becomes *a view* over that store, not *the tool* that
computes it.

---

## 2. Exposure methodology audit

There are currently **three** implementations of "affected population," and they
are **not** cosmetic variants — they use different methods and would produce
different numbers. There is no single source of truth.

| | **A. marimo `flood_exposure.py`** | **B. `affected_population.py`** | **C. `scripts/02`** |
|---|---|---|---|
| Flood input | pipeline **polygons** (`.shp.zip`) | a flood **GeoTIFF** | **recomputes** flood from STAC |
| Population | GHSL ~90 m (`3ss`) | GHSL GeoTIFF (local) | GHSL 100 m |
| Admin breakdown | per-admin | **none** (AOI total only) | per-admin |
| CRS handling | reproject all → **hardcoded EPSG:32618** | keep flood grid; reproject pop to it | works in EPSG:4326 |
| Resolution / ÷25 | **none** | **÷25** | **÷25** |
| Buffer | user buffer (default **100 m**) | none | none |
| Overlay method | rasterize flood → binary mask → exactextract zonal sum | flood_mask × pop → `np.sum` | flood pixels → **points** → sample pop → sjoin to admin |
| Rounding | `ceil` per admin | none | `ceil` **per flood point** |
| Wired to product? | de-facto product path | standalone POC (Pakistan example) | choropleth images |

### What makes the numbers diverge

1. **The ÷25 factor.** B/C divide GHSL by 25 (spreading a 100 m pixel onto 20 m
   flood sub-pixels); A does not divide and uses ~90 m pop. Incompatible
   population accounting.
2. **All-or-nothing vs fractional.** A's `geometry_mask` is binary — any pop pixel
   *touched* by flood contributes its **full** population (over-count). None of the
   three do coverage-weighted (fractional) overlay, which is the areal-correct method.
3. **Buffering.** Only A buffers flood (+100 m), inflating exposure. The buffer is a
   human-tuned slider with no fixed justification.
4. **Rounding bias.** C `ceil`s **per flood point** — a large systematic upward bias
   when points are sparse.

### Verdict

- **B is effectively dead** for the product (no admin breakdown, never touches blob,
  Pakistan-example `main()`) → retire to `source_legacy/`.
- **A is the de-facto product path** but carries the CRS bug, no resolution
  adjustment, binary over-count, and buffer inflation.
- **C is the most GFM-faithful** per-admin path but bypasses the pipeline's polygon
  outputs and has the per-point rounding bias.

**P0 consolidation is therefore a *methodology decision*, not a refactor.** It
changes every published number, so it must be decided deliberately (see §6 OPEN).

---

## 3. Recommended exposure method

**One function**, owning the calculation, imported by both the pipeline and any
viewer. Computed entirely in **raster space** — polygons are a *display deliverable,
not a dependency*.

```
flood extent raster (20 m, native)
      │  average-resample → population grid
      ▼
flood_fraction raster (0–1 on the pop grid)
      │  exact_extract(values=pop, weights=flood_fraction,
      │                ops=["weighted_sum"], zones=admin)
      ▼
exposed population per admin unit
```

### Why this shape

- **No raster→vector→raster round trip.** Flood starts as a raster; polygonizing it
  only to re-rasterize for stats is wasted work *and* hands `exactextract` thousands
  of jagged polygons (the expensive input). Keeping flood as a weight raster gives
  `exactextract` clean input (a few smooth admin zones) — it's the *faster* call.
- **Admin is the only vector.** `exactextract` earns its place handling partial-pixel
  coverage at admin boundaries.
- **Coverage-weighted = areal-correct.** `exposed = Σ pop × flooded_fraction`, not
  full-pixel-if-touched. No binary over-count.
- **Downsample flood (20 → 90 m), not upsample pop (90 → 20 m).** The two are
  *mathematically identical* (`Σ pop × flood_fraction`) when done correctly, but
  downsampling flood is strictly better in practice:
  - ~20× fewer cells (cheaper);
  - no population-conservation footgun (upsampling counts requires sum-preserving
    resampling or you inflate population ~20× — the hazard the ÷25 constant was
    patching);
  - honest about resolution (population is only known at ~90 m; a 20 m grid is false
    precision);
  - robust to the geographic CRS (a fixed area-ratio constant is latitude-dependent;
    fractional overlap is not).
- **This dissolves the ÷25 question.** Average-resampling the binary flood to the pop
  grid *is* the principled resolution reconciliation — the ÷25 idea done exactly,
  without per-point rounding loss.

### Performance note

For adm0/adm1 over one country/date, both the resample (one downsampling pass) and
the `exactextract` call are seconds-scale; the GFM STAC fetch + dask composite
dominate. The resample is a **prerequisite**, not a competitor to `exactextract`
(`exact_extract` requires value and weight grids to align — `grid_compat_tol`).

---

## 4. Provenance → incremental & cumulative exposure

"Cumulative exposure" is **not** the sum of daily exposures — a location flooded in
week 1 and again in week 3 must be counted once. The clean decomposition is to store
**incremental newly-exposed population per observation date**:

> exposure attributed to date *d* = population in pixels that *first* became flooded
> on *d* (within the window), by admin unit.

Then **cumulative = Σ increments** (additive, no double-counting), and daily/event
views fall out of the same table.

The structure that makes this computable already exists in the pipeline: the
**provenance raster** (pixel → observation date). The exposure method in §3
generalizes directly — for each date `d`, `flood_fraction_d` = average-resample of
`(provenance == d)`, then `weighted_sum`. ~4 dates → 4 cheap passes → an incremental
exposure time series. (Note: `create_provenance_raster` currently tracks the *most
recent* date per pixel; "first flooded" increments would flip that — same machinery.)

This reframes provenance from a visualization into **the core data structure of the
product.**

---

## 5. Data & storage contract

Replace the current ad-hoc, filename-encoded blob layout (where dates are regex-parsed
out of `.shp.zip` names and AOIs are hardcoded `if/elif`) with explicit contracts.

- **Exposure results → one partitioned parquet table on blob** (tidy/long format):
  `aoi, adm_level, adm_id, obs_date, run_date, mode, pop_exposed, ...`. Analysts read
  it with pandas/duckdb; a dashboard queries it with duckdb/Polars. **No API/database
  server yet** — a table is enough for DS + dashboard; an API can front it later.
- **Run manifest.** A run = `(aoi, target_date, mode, n_images, params)` emits its
  artifacts plus a small manifest (JSON/STAC item) carrying dates, bbox, params, and
  artifact paths. **Metadata lives in the manifest, never parsed back from filenames.**
- **Admin source = the edge-matched humanitarian DB.**
  `global / dev / fieldmaps/edge-matched/humanitarian/intl/{adm0|adm1}/{ISO3}.parquet`.
  Clean per-country-per-level parquet, WGS84, stable versioned key `adm1_id`
  (e.g. `HTI-20230404-01`). Edge-matched → supports regional/multi-country views.
  **adm0/adm1 only** — finer levels remain on the legacy CODAB path.
- **AOI registry.** One config-driven registry (ISO3 *or* custom geom blob), shared by
  pipeline and viewer — removes the hardcoded `if/elif` and the dropdown that drifts
  from the CLI.
- **Viewer.** A new, thin marimo (or similar) that **imports the exposure function and
  reads the table** — zero recompute, zero duplicated logic.

---

## 6. Open decisions [OPEN]

These gate the methodology and schema and need explicit sign-off:

1. **Population layer + resolution.** Standardize on which GHSL (~90 m `3ss` vs 100 m
   vs 1 km). Fixes whether any resolution constant is even meaningful.
2. **Buffer.** Keep it (off by default, documented) or drop it entirely?
3. **Cumulative window semantics.** Rolling N days, event-anchored, or season-to-date?
   Determines what `obs_date`/window columns the table needs.
4. **Product scope.** Is v1 monitoring **adm0/adm1 only** (clean, global, edge-matched)
   with finer levels deferred? (Leaning yes.)

---

## 7. Roadmap

- **P0 — value chain, made reproducible.** Decide the exposure method (§6), implement
  the single raster-weight function (§3), wire it as a pipeline stage, define the run
  manifest + tidy exposure table. After this, "give me HTI exposure for date X" runs
  headless and the marimo app is demoted to a viewer. Retire impl B; reconcile/retire C.
- **P1 — storage/contract hardening.** AOI registry, run index, stable paths; backfill
  existing ad-hoc blobs into the new layout.
- **P2 — live monitoring.** Per-AOI state ("last processed date"), incremental/idempotent
  runs, scheduling, alerts that read the table.
- **Defer.** `source_legacy/` modules (done), early-POC book chapters, GEE/east-africa
  code, standalone choropleth-image scripts — confirm before touching.

# Architecture & Direction

> Status: **working draft** — captures decisions and open questions as of the
> 2026 architecture review. Sections marked **[OPEN]** are not yet decided.

## Purpose

This document records *where the project is going* and *why*, so that changes can
be reviewed against an agreed direction rather than evaluated one diff at a time.
It is the big-picture companion to `OPTIMIZATION_STRATEGY.md` (pipeline performance).

---

## 1. Framing: from one-off tool to live monitoring

The project has so far been a **one-off tool** — run by hand for a country and date,
producing maps and CSVs. The intent is a **live flood monitoring system**: AOIs are
processed repeatedly as new GFM observations arrive, results accumulate into a time
series, and consumers (DS team today, a dashboard soon) **pull** from a stable store
rather than re-running a notebook.

### The core realization

Most of the value users see is **population exposed per admin unit** — and today that
number is *not* produced by the pipeline. The pipeline stops at flood polygons + a
provenance raster; the exposure calculation lives **outside** it, in an interactive
marimo app (`flood_exposure.py`). That makes the most valuable output
non-reproducible, non-automatable, and untested.

**Direction:** make exposure a first-class, headless pipeline output written to a
stable store. The map/app becomes *a view* over that store, not *the tool* that
computes it.

---

## 2. Exposure methodology audit

Three implementations of "affected population" exist, and they use **different
methods that produce different numbers** — there is no single source of truth.

| | **A. marimo `flood_exposure.py`** | **B. `affected_population.py`** | **C. `scripts/02`** |
|---|---|---|---|
| Flood input | pipeline **polygons** | a flood **GeoTIFF** | **recomputes** from STAC |
| Population | GHSL ~90 m (`3ss`) | GHSL GeoTIFF (local) | GHSL 100 m |
| Admin breakdown | per-admin | **none** | per-admin |
| CRS | reproject → **hardcoded EPSG:32618** | keep flood grid | EPSG:4326 |
| Resolution / ÷25 | **none** | **÷25** | **÷25** |
| Buffer | user buffer (default 100 m) | none | none |
| Overlay | rasterize flood → binary mask → exactextract sum | flood_mask × pop → `np.sum` | flood→points→sample→sjoin |
| Rounding | `ceil` per admin | none | `ceil` **per point** |
| Wired to product? | de-facto path | POC (Pakistan) | choropleth images |

**Divergences that change the number:** the ÷25 factor (B/C vs A); all-or-nothing vs
fractional overlay (none do coverage-weighting); buffering (A only); per-point rounding
(C). **Verdict:** B is dead (retire to `source_legacy/`); A is the de-facto path but
carries the CRS bug + over-count + buffer inflation; C is most GFM-faithful but bypasses
the polygon outputs and has rounding bias. **P0 consolidation is a *methodology
decision*, not a refactor** — it changes every published number.

---

## 3. Recommended exposure method

**One function**, owning the calculation, imported by both pipeline and viewer.
Computed entirely in **raster space** — polygons are a *display deliverable, not a
dependency*.

```
flood extent raster (20 m, native)
   │  average-resample → population grid      (fractional flood coverage per pop pixel)
   ▼
exact_extract(values=pop, weights=flood_fraction, ops=["weighted_sum"], zones=admin)
   ▼
exposed population per admin unit
```

**Why:**
- **No raster→vector→raster round trip.** Polygonizing flood only to re-rasterize for
  stats is wasted work *and* hands `exactextract` thousands of jagged polygons (its
  expensive input). Flood-as-weight-raster gives it clean input (few smooth admin
  zones) — the *faster* call. Admin is the only vector; `exactextract` earns its place
  handling partial-pixel coverage at admin boundaries.
- **Coverage-weighted = areal-correct.** `exposed = Σ pop × flooded_fraction`, not
  full-pixel-if-touched.
- **Downsample flood (20→90 m), not upsample pop (90→20 m).** Mathematically identical
  (`Σ pop × flood_fraction`), but downsampling flood is ~20× cheaper, avoids the
  population-conservation footgun (upsampling counts needs sum-preserving resampling or
  you inflate pop ~20× — what the ÷25 constant was patching), is honest about resolution
  (pop is only known at ~90 m), and is robust to the geographic CRS. **This dissolves
  the ÷25 question** — averaging the binary flood to the pop grid *is* the principled
  resolution reconciliation.
- `exact_extract` requires value+weight grids aligned (`grid_compat_tol`), so the
  resample is a prerequisite, not a competitor — and it's the cheap part.

---

## 4. Provenance raster as the substrate

The provenance raster (per pixel, **most-recent flood date**) is not a visualization —
it is **the core retained artifact**, because end-anchored rolling windows are just
thresholds on it:

```
pixel flooded in [d−N, d]   ⟺   provenance_date(pixel) ≥ d − N
```

One compact raster → **all** rolling windows by thresholding. No per-date stack needed,
and **no increment decomposition** (fixed windows compute each union directly; the union
handles overlap/double-counting internally, so we store the cumulative value per window).

### Three-state encoding (handles the swath/footprint issue)

A flood pixel is in one of **three** states, not two — and "not observed" ≠ "not
flooded." SAR swaths are oblique, non-rectangular, with interior nodata (permanent
water, layover/shadow, dense vegetation, urban, sensitivity masks). Encode all three in
the **same** raster:

```
value ≥ 0   → most-recent flood date-index   (observed & flooded)
sentinel    → observed, never flooded         (dry)
nodata      → never observed                  (coverage gap)
```

The observed extent is therefore **derived from real valid pixels, never from tile
footprints/bboxes** (which overstate — the historical `stac_spatial_filter` bug). A
polygon-with-holes *could* represent the extent exactly; whether raster or polygon is
more compact is empirical (depends on nodata fragmentation) and **moot** — we keep the
provenance raster regardless, so the extent comes free in its nodata mask.

- **Run-level coverage** ("observed at all this run") → free from the 3-state raster.
- **Sub-window coverage** ("% observed in last N days") → needs a `last_observed_date`
  value too; **optional / deferrable** confidence layer. [OPEN]
- **Event-anchored windows** (arbitrary start, e.g. "since landfall") → not derivable
  from one run's provenance; keep per-date detail only for **active-event AOIs** as an
  escape hatch.

> TODO: verify what `create_provenance_raster` *currently* encodes (most-recent
> *observed* vs *flooded* date) before finalizing the 3-state encoding.

---

## 5. Data & storage contract

Replace the ad-hoc, filename-encoded blob layout (dates regex-parsed from `.shp.zip`
names; AOIs hardcoded `if/elif`) with explicit contracts. We are a **thin derived layer
over GFM's STAC** — STAC is the raw source-of-record; we never mirror it. We own the
**transform** (tile-mosaic + temporal composite + grid/CRS normalization — the
genuinely hard part) and cache only its *processed* output where latency or
reproducibility demands it.

### Served product: the rolling-window exposure table

One partitioned parquet table (analysts read with pandas/duckdb; dashboard via
duckdb/Polars — **no API/DB server yet**). Fixed rolling windows ending at each
observation date cover most use cases:

```
key:   aoi, adm_level, adm_id, obs_date, window, pop_source
value: pop_exposed   (+ optional observed_frac for coverage)
meta:  run_id, run_date, mode, pop_version, admin_version
windows: 1, 3, 5, 10, 15, 30 days
```

- **Windows are nested → monotone** (`1d ≤ 3d ≤ … ≤ 30d`): a free QA check.
- **`pop_source` is a KEY dimension**, not metadata — multiple population masters coexist
  per row, and backfilling a new master writes new rows without touching old ones.
- **wide vs long** [OPEN]: long (`window_days` column) is extensible/SQL-friendly; wide
  (6 columns) is compact/dashboard-friendly. Lean long for the store, pivot on read.

### The (a)/(b) split — why backfill/pop-swap is cheap

- **(a) flood footprint** — `STAC → mosaic → composite → provenance`. Expensive,
  **pop-independent.**
- **(b) population overlay** — `threshold provenance → resample → exact_extract`. Cheap,
  **pop-dependent.**

Swapping a population master re-runs **only (b)** — no GFM re-fetch, no re-compositing.

### Provenance / manifests
- **Pixel-level provenance → the provenance raster**, `date_mapping` in COG metadata
  (this is where the `map_date`/metadata work earns its place).
- **Run/temporal provenance → the manifest** (JSON/STAC item): contributing obs dates,
  bbox, params, artifact paths. Identical across admin rows, so **store once per run**
  and reference by `run_id` — keeps the table skinny.

### Admin source
`global / dev / fieldmaps/edge-matched/humanitarian/intl/{adm0|adm1}/{ISO3}.parquet` —
per-country parquet, WGS84, stable versioned key `adm1_id` (e.g. `HTI-20230404-01`),
edge-matched (supports regional views). **adm0/adm1 only**; finer levels stay on legacy
CODAB. One **AOI registry** (config-driven: ISO3 or custom geom blob) shared by pipeline
and viewer.

### Storage budget (it is not the constraint)

Single-band int16 rasters; flood/provenance is **sparse** → compresses heavily.

| Scope | Provenance @20 m | Cost |
|---|---|---|
| 1 national AOI (~Haiti), weekly | ~0.3 GB/yr | pennies |
| 50 national AOIs, weekly (+ obs layer) | ~57 GB/yr | ~$13/yr |
| **Global**, provenance stored **only where flood** | ~11 GB/yr | **~$2/yr** |
| Global provenance @20 m, blanket | ~1 TB/yr | ~$227/yr |
| Global **observation** layer @20 m (the one big item) | ~3.5 TB/yr | ~$800/yr |
| Global exposure **table** (adm1, weekly) | ~86 MB/yr | ~free |

**Implications:** storage never forces bounded-dynamism — persist the exact substrate.
The real cost driver is **object count**, not bytes (50 AOIs × weekly × 2 layers ≈
5k+ objects/yr) → favor a **per-AOI / per-tile temporal cube** (zarr or multi-band COG)
over thousands of loose files; a window query is then one read. Resolution (~20×) and
cadence (~30×) are the levers; even the worst case is lunch money.

---

## 6. Open decisions [OPEN]

1. **Population layer + vintage.** Which GHSL (~90 m `3ss` / 100 m / 1 km), and **pin
   the epoch** — a long series is only comparable if pop vintage is held fixed
   (re-baseline deliberately, never silently). `pop_source` is a key dimension so
   multiple can coexist.
2. **Buffer.** Keep (off by default, documented) or drop? Lean **drop for v1** — an
   unattended run has no human to set it, and a fixed defensible number beats a slider.
3. **Cumulative window semantics.** Resolved toward a **fixed window set** (§5);
   windows are query-time views over event-time records, not baked into storage. Confirm
   the set {1,3,5,10,15,30}.
4. **Product scope.** v1 = **adm0/adm1 only** (clean, global, edge-matched)? (Leaning yes.)
5. **Fetch depth.** Rolling windows up to 30 d require each run to fetch **all obs in
   [d−30, d]** (no `n_images` cap) and build provenance over them.
6. **Coverage layer scope.** Run-level (free) vs sub-window (`last_observed_date`,
   optional) vs global (@90 m ~$39/yr).
7. **Trigger model (global).** **Event-driven** (react to GFM flood flags — makes
   provenance sparse, tames compute) vs scheduled-blanket. Lean **event-driven**.

---

## 7. Roadmap

- **P0 — value chain, reproducible.** Decide the method (§6.1–2), implement the single
  raster-weight exposure function (§3), wire it as a pipeline stage, define the run
  manifest + the **final-shape** rolling-window table (§5). *Schema must be
  monitoring-shaped now even though scheduling is P2, or P0 is throwaway.* Retire impl B;
  reconcile/retire C. After this, "give me HTI exposure for date X" runs headless.
- **P1 — storage/contract hardening.** AOI registry, run index, per-AOI/tile cubes,
  stable paths; idempotent **overwrite-by-partition** (not naive append — reruns/late
  data would duplicate); backfill existing ad-hoc blobs.
- **P2 — live monitoring.** Per-AOI state (last-processed date), incremental/idempotent
  runs, **event-driven trigger**, scheduling, alerts reading the table.
- **Defer.** `source_legacy/` (done), early-POC chapters, GEE/east-africa code,
  standalone choropleth scripts — confirm before touching.

---

## 8. Event rarity / return period [OPEN, P2+]

**The hard limit:** RP is extreme-value statistics over a long record. GFM is **~5–6
years** — far too short to estimate rare RP (1-in-50/100). Any system printing such
numbers off GFM alone is lying. So the question is *where the long record comes from.*

**Scoped ambition: "≤6 yr vs >6 yr"** — "have we ever seen one this big, or is this a
new record" — is *exactly* what a 6-year record honestly supports, and needs **no
external model, no EVT fit, no declustering**: just a **per-admin annual maximum** of a
hazard metric. New event exceeds the historical max → ">6 yr (unprecedented in the GFM
era)"; otherwise rank gives an empirical RP in 1–6.

**Reuses the architecture:** this is **the monitoring pipeline run over full history**
once + a tiny climatology table — RP is a downstream analytic, not a redesign.

**Key conceptual split:** compute RP on a **hazard metric** (per-admin peak flood
*extent/fraction*), **not exposure** (which drifts with population). Report
"1-in-X-year flood affecting N people" — rarity from hazard, N from the exposure layer.

**Higher RP (1-in-50/100)** requires **model anchoring** — overlay observed flood against
pre-computed RP hazard layers (GloFAS / JRC Global Flood Hazard / Fathom) built from
decades of modeled hydrology. Plus stationarity caveats (climate non-stationarity).

**Cost of the global ≤6/>6 backfill:** one-time **~$300–850** cloud (full) or ~$60–170
(event-driven); realistically **~$1–3k** if pipeline throughput is slow (per the dask
bottleneck history). Egress is the swing factor — **co-locate compute with the data**.
Ongoing storage negligible (climatology table <1 MB adm1 / ~34 MB adm2). **The real cost
is engineering effort + whether GFM's STAC will serve a bulk 6-yr global history without
rate-limiting — not compute or storage.** A feasibility spike on bulk-archive access
should precede committing.

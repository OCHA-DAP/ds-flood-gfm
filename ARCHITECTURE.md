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

## 4. Flood windows and the provenance raster

### The core operation: rolling max over the chosen window

A window's cumulative flood extent is just a **rolling max over the stacked daily flood
rasters** — `(stack == 1).any()` over the dates in `[d−N, d]`. The window union is
*determined by which dates you include*, so each rolling window (1/3/5/10/15/30 d) is
computed directly during a run from the 30-day stack. **No special flood-date artifact is
needed to produce the windows** — and **no increment decomposition** (each window's union
handles overlap internally, so we store the cumulative value per window).

### What we keep: `last_observed_date` (the current provenance raster)

The *current* `create_provenance_raster` computes **`last_observed_date`** — most-recent
date a pixel had a valid observation, `has_data = ~stack_flood_max.isnull()` ("regardless
of flood value"). (Confirmed by code: `create_flood_composite` tests `stack_flood_max == 1`
for flood, which is only meaningful because dry `0` is a real value. A live pixel read was
attempted but GFM STAC timed out — itself a data point for the §8 bulk-access risk.)

This is exactly the **coverage substrate**, and we keep it. Thresholding it gives the
observed extent per window:

```
pixel observed in [d−N, d]   ⟺   last_observed_date ≥ d − N
```

→ drives `pct_unit_covered` / `pct_unit_pop_covered` (§5) — what makes the exposure number
honest about gaps. **Three states, not two:** "not observed" ≠ "not flooded" (SAR swaths
are oblique, non-rectangular, with interior nodata — permanent water, layover/shadow,
vegetation, urban, sensitivity masks). The observed extent comes from **real valid pixels,
never tile footprints/bboxes** (which overstate — the historical `stac_spatial_filter`
bug); it falls out of `last_observed_date`'s nodata mask for free.

### Future experiment (not a priority): a `last_flooded_date` raster

Storing a per-pixel `last_flooded_date` (`argmax` over `stack == 1`) would let us threshold
out all windows from one raster (`flooded in [d−N,d] ⟺ last_flooded_date ≥ d−N`, equivalent
to the rolling max). It is **not needed to compute the product** — its only value is as a
*cache*:

1. **Local pop-backfill** — re-overlay a new population master without re-fetching/
   re-compositing from the (slow) GFM STAC.
2. **Correct per-pixel flood date in cumulative mode** — today's flood-polygon date
   attribution uses `last_observed_date`, so a pixel flooded day 1 but imaged dry day 5 is
   tagged day 5. (This mismatch is **cumulative-mode only**; in *latest* mode the last
   observation *is* the flood, so the dates coincide.)

Both are optimizations worth a future experiment, **not a P0 priority**. For now: compute
windows via rolling max, keep `last_observed_date` for coverage, and re-derive from STAC if
a backfill is needed.

- **Event-anchored windows** (arbitrary start, e.g. "since landfall") → not derivable from
  one run's rasters; keep per-date detail only for **active-event AOIs** as an escape hatch.

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
value: pop_exposed
       pct_unit_pop_covered   (pop in observed pixels / total pop — primary confidence metric)
       pct_unit_covered       (observed area / total area — secondary)
meta:  run_id, run_date, mode, pop_version, admin_version
windows: 1, 3, 5, 10, 15, 30 days
```

- **Coverage columns make the number honest.** `pct_unit_pop_covered` turns "12,400
  exposed" into "12,400 exposed, 85% of the unit's population observed this window" — the
  difference between a real figure and "we mostly didn't look." Both come for free by
  thresholding `last_observed_date` (§4): area-coverage via `exact_extract` mean of the
  observed mask; pop-coverage via `weighted_sum(pop, weight=observed) / sum(pop)`. Pop-
  coverage is pop-dependent → a step-(b) output, recomputed cheaply on a pop-master swap.
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
6. **Coverage layer scope.** Per-window coverage is now first-class (`pct_unit_pop_covered`
   / `pct_unit_covered`, §5) via `last_observed_date`. Remaining choice: store the
   observation raster at 20 m (exact) or 90 m (~$39/yr global) — and whether to compute
   global coverage at all or only for monitored/event AOIs.
7. **Trigger model (global).** **Event-driven** (react to GFM flood flags — makes
   provenance sparse, tames compute) vs scheduled-blanket. Lean **event-driven**.

---

## 7. Roadmap

- **P0 — value chain, reproducible.** Decide the method (§6.1–2), implement the single
  raster-weight exposure function (§3), compute windows via rolling max over the stack
  (§4), emit the coverage columns from `last_observed_date` (§5), wire it as a pipeline
  stage, define the run manifest + the **final-shape** rolling-window table (§5).
  *Schema must be monitoring-shaped now even though scheduling is P2, or P0 is throwaway.*
  Retire impl B; reconcile/retire C. After this, "give me HTI exposure for date X" runs
  headless. (A stored `last_flooded_date` raster is a **future experiment**, not P0 — §4.)
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

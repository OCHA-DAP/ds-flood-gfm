# GFDS × FloodScan: Nigeria 2022 Validation Findings

*2026-08-18. Companion to [gdacs-gfds-assessment.md](gdacs-gfds-assessment.md).
Code: `experiments/gfds_nga2022/`. Outputs: `outputs/gfds_validation/` (local).*

## Setup

- **Event:** Nigeria 2022 floods — Lagdo Dam releases from 13 Sep, Kogi/confluence
  peak early–mid Oct, Bayelsa/delta peaking late Oct–Nov. Reference LGAs:
  Lokoja, Ajaokuta, Ibaji, Bassa (Kogi); Sagbama, Yenegoa, Ekeremor (Bayelsa).
- **Data:** GFDS daily `signal` (merged `ALL`), Nigeria bbox (2.7–14.7°E,
  4–14°N, 111×133 px @0.09°), Jun–Nov 2022; GFDS static baselines
  (`bt_signal_avg/sd`, frozen 2009); FloodScan SFED daily COGs from team blob
  (band 1), bilinearly regridded to the GFDS grid.
- **GFDS anomaly:** `(avg − signal) / max(sd, 0.005)` (flood-positive), plus a
  4-day trailing mean to tame swath noise. This is the DIY anomaly — JRC's own
  `mag_signal` product was not needed for the core result.

## Results

### Pixel level (Jun–Nov, ~11.3k pixels scored)

| Pixel population | median r vs SFED |
|---|---|
| All pixels | 0.16 |
| Flooded pixels (SFED max > 0.2) | 0.50 |
| Flooded pixels, 4-day-mean anomaly | 0.54 |

The correlation map reproduces the **entire Niger–Benue river network** at
r ≈ 0.75–0.9 — spatial agreement with FloodScan's flood corridors is
feature-for-feature (see `nga_pixel_corr_map.png`).

### LGA level (the common-data-model test; 774 LGAs, mean per LGA per day)

> **Note (2026-08-19):** the book chapter (`book_gfm/05_alternative_flood_sources.qmd`)
> is the canonical, reproducible version of this comparison. It scores on the
> complete Aug 15–Nov 30 window and replaced the fragile argmax peak-lag with a
> cross-correlation best-lag (argmax lag flipped by up to 16 days under
> different smoothing choices for the same LGA). On that basis the top severity
> bands score higher than the table below (r = 0.88 for SFED max 0.10–0.20 and
> 0.92 for >0.20, median |lag| 2–3 days). The table below is the original
> exploratory run (Jun–Dec series, argmax lag), kept for provenance.

Dose-response by flood severity (SFED LGA-mean seasonal max):

| SFED max band | median r | median \|peak lag\| | n LGAs |
|---|---|---|---|
| 0.02–0.05 | 0.26 | 32 d | 160 |
| 0.05–0.10 | 0.44 | 20 d | 152 |
| 0.10–0.20 | 0.59 | 3 d | 148 |
| > 0.20 | **0.75** | **4 d** | 59 |

Among LGAs with SFED max > 0.10 (n=207): 58% reach r ≥ 0.6, 32% reach r ≥ 0.8.

### Reference LGAs (known ground truth)

| LGA | State | r | peak lag (d) |
|---|---|---|---|
| Ibaji | Kogi | **0.97** | +1 |
| Sagbama | Bayelsa | **0.96** | +10 |
| Kogi | Kogi | 0.95 | +7 |
| Lokoja | Kogi | 0.93 | −1 |
| Bassa | Kogi | 0.92 | −2 |
| Ajaokuta | Kogi | 0.89 | −1 |
| Ekeremor | Bayelsa | 0.38 | −28 |

Time-series overlays (`nga_reference_lgas.png`) show GFDS tracking SFED
curve-for-curve through onset, peak, and recession at the confluence LGAs,
including the post-Lagdo surge. Ekeremor is the instructive failure: a coastal
delta LGA where the 10 km microwave footprint is contaminated by ocean/estuary
— consistent with GFDS physics (needs a dry land calibration contrast).

## What this settles

1. **GFDS raw `signal` + a baseline is the product.** Everything else on the
   server is derivable or legacy: JRC's `mag_signal` is just signal scored
   against their frozen 2009 baseline; the `Avg*` folders are 4-day trailing
   means we compute in one line; `dn_*`/`deltaT_*` are legacy formulations.
   An ingest pipeline needs **one layer: daily signal** (~13 MB/day as uint16
   COG), plus the two static baseline files fetched once.
2. **Even the stale 2009 static baseline is good enough for within-season
   tracking** at admin level in strongly flooded riverine areas. (Absolute σ
   levels are inflated/noisy — cross-season return periods still need our own
   DOY climatology, but that's an enhancement, not a prerequisite.)
3. **Common-data-model fit is direct:** daily zonal mean of the (4-day
   smoothed) anomaly per admin unit — structurally identical to the
   `floodscan_exposure` per-LGA pattern the team already stores. Same key
   (pcode, date), one value column.
4. **Where GFDS is trustworthy:** substantial riverine flooding (SFED-max >
   0.10 LGAs; big-river corridors). **Where it is not:** coastal/estuarine
   admin units, marginal flooding, and single-day values (use the 4-day mean).
5. **Operational caveat measured during this work:** GDACS server throughput
   swung from ~2 s to ~140+ s per file within a day. Ingest must be resumable
   with generous timeouts and no same-day SLA.

## Open items

- JRC `mag_signal` head-to-head at LGA level (fetch was still completing;
  expected to match the DIY anomaly modulo scale — Pakistan spot-check gave
  |r| = 0.84 at pixel level).
- DOY climatology (2015+) vs static baseline: quantify the improvement on
  this same Nigeria setup before designing the climatology job.
- A second event/country (e.g. Pakistan 2022 bbox already cached) to confirm
  transferability.

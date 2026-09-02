# GFDS Downscaling Proof of Concept: Findings

*2026-09-01. Follows the "what would it take to downscale GFDS like FloodScan"
question. Code: `experiments/gfds_downscaling_poc/` (five scripts, run in
order; all inputs cached or fetched via HTTP range reads). Companion docs:
[gdacs-gfds-assessment.md](gdacs-gfds-assessment.md),
[gfds-nga2022-findings.md](gfds-nga2022-findings.md).*

## What was built

The two-step recipe, end to end, on the Nigeria 2022 cache:

1. **Signal → fraction calibration (10 km).** Per-pixel monotone quantile
   mapping from the 4-day GFDS anomaly to FloodScan SFED, fit on alternating
   days and scored on the held-out days, so every number below is
   out-of-sample. (`01_calibrate_fraction.py`)
2. **Prior-based disaggregation (~83 m).** JRC Global Surface Water
   occurrence (~28 m, fetched by HTTP range reads, aggregated 3×3) ranks the
   cells inside each 0.09° pixel (occurrence desc, then distance to
   historical water asc); the calibrated fraction is flood-filled top-down,
   excluding permanent water. AOI: Niger–Benue confluence (6.0–7.8°E,
   6.3–8.3°N). (`02_build_prior.py`, `04_downscale_validate.py`)
3. **Independent validation.** GFM Sentinel-1 cumulative extent for
   2022-10-06/08/13 (different sensor family from both microwave products),
   fetched with this repo's own pipeline. Controls: the same flood-fill fed
   with licensed SFED (upper bound), and a **prior-only map with the same
   total area but no 10 km information** — the critical control for whether
   downscaling adds anything. (`03_fetch_gfm_extent.py`,
   `05_multiscale_eval.py`)

## Results

### Step 1: calibration works

Out-of-sample (78 held-out days, pixels with SFED max > 0.05):

- RMSE 0.040 vs 0.048 for the predict-the-pixel-mean baseline; bias +0.001
- Basin-scale flooded-area time series: r = 0.80 vs SFED on held-out days
- At the SFED peak day, GFDS-derived total flooded area = **84% of SFED's**

A unitless anomaly becomes a usable physical fraction with one monotone map
per pixel.

### Step 2: downscaling is input-agnostic, and mostly prior

Against GFM at 83 m (POD / FAR / CSI): GFDS 0.63/0.80/0.180, SFED
0.69/0.80/0.185, prior-only 0.60/0.79/0.184. Two readings:

- **Free GFDS ≈ licensed SFED through the identical pipeline.** The input
  costs almost nothing in skill (ΔCSI 0.005).
- **At 83 m, the static prior carries all the skill.** The prior-only control
  ties both. The painted fine detail is cartography, not observation.

### The scale ladder is the real finding

Correlation of wet fraction vs GFM by evaluation scale
(`multiscale_corr.csv`):

| scale | GFDS | SFED | prior-only |
|---|---|---|---|
| 83 m | 0.33 | 0.35 | 0.33 |
| 0.5 km | 0.47 | 0.49 | 0.46 |
| 1 km | 0.55 | 0.56 | 0.52 |
| 3 km | 0.68 | 0.68 | 0.63 |
| 6 km | 0.79 | 0.79 | **0.72** |

The 10 km microwave information adds nothing at 83 m, a little at 1 km, and a
clear margin (+0.05–0.07 r) from 3 km up. GFDS tracks SFED within 0.01–0.02
at every scale.

Interpretation caveats: GFM is imperfect truth (SAR under-detects in
vegetation; only 3 observation dates vs continuous microwave coverage), which
inflates FAR for all three maps and likely compresses the differences; and
this is one event in one basin.

## What this means

1. **The valuable product is the calibrated 10 km fraction.** It converts
   GFDS into FloodScan-comparable physical units at zero data cost, and it is
   what admin-level monitoring and exposure aggregation actually consume.
2. **90 m downscaling is presentation, not information.** Defensible for
   dasymetric population overlay (where sub-pixel water placement changes who
   is counted — the prior is a reasonable allocator), but it should never be
   read as observed extent. If fine-scale maps are the goal, GFM is the tool.
3. **The free/licensed gap through this pipeline is ~nil for this event.**
   Combined with the LGA-level r = 0.89–0.97 findings, the case that GFDS +
   our own scoring can stand in for FloodScan in admin-level riverine
   monitoring got stronger.

## Addendum (2026-09-02): FloodScan-free calibration shoot-out

Three routes from GFDS signal to flooded fraction with zero FloodScan input,
scored against SFED as a fair external benchmark
(`06_independent_fractions.py`, confluence AOI):

| route | r vs SFED (wet cells) | area ratio at SFED peak |
|---|---|---|
| R1 physics (literature emissivity contrast K=0.35) | **0.84** | **0.99** |
| R1 + MDFF-style threshold (0.10) | 0.83 | 0.86 (June floor drops 658→118 km², SFED ~100) |
| R2 optical anchor (GSW history) | 0.42 | 0.19 |
| R3 radar anchor (GFM peak window) | 0.74 | 0.24 |
| R0 SFED-trained lookup (ceiling, not independent) | 0.94 | 0.83 |

Verdict: the simple mixing-equation inversion wins decisively. Anchor routes
fail structurally: 2022 exceeded anything in their anchors' record, so their
amplitudes cap out low. The physics route's dry-season floor reproduces the
exact problem AER's Users Guide documents solving with its MDFF threshold
(v05R01 §1.2); applying the same idea fixes it here too. This makes an
independent open-source product plausible, with a to-do list: per-pixel K,
seasonal dry reference, threshold tuning, multi-year training.

## Before anything large-scale

- Replicate calibration + scale ladder on a second event/basin (Pakistan 2022
  window is already cached at repo level) and ideally a multi-year Nigeria
  record, so the quantile maps are not trained and tested inside one wet
  season.
- Calibrate against GFM-aggregated fractions instead of SFED to break the
  shared-radiometer circularity, then re-score against held-out GFM dates.
- If downscaling is pursued at all: add HAND/DEM to the prior (the GSW-only
  prior has water history on just 2.7% of cells, leaving ties broken by
  distance alone) and re-run the prior-only control — the bar it sets is the
  whole question.

# Flood Fusion: Plan

2026-10-12

## What the product is

A daily, global, ~1 km map of the probability that each floodable cell is inundated, with an uncertainty layer and a days-since-last-observation layer. It is a best-current-estimate product that fills the gaps between single-sensor products. It is not a replacement for GFM's 20 m detail.

- Inputs: Copernicus GFM (Sentinel-1 SAR); FloodScan and GFDS merged into one passive-microwave vote; NASA MODIS NRT flood product. VIIRS flood product is optional if accessible.
- Reference (gold) data: Copernicus EMS rapid-mapping activations and UNOSAT flood maps. These drive training and are the only test set.
- Supporting layers: HAND, land cover, snow-season flag, permanent-water mask. GloFAS return period is a version-2 upgrade to the prior, not a dependency.
- Outputs per day: probability raster, uncertainty raster, observation-age raster. Plus one static accuracy table (products x strata) that is the whole trained model.

"Better than any single product" is defined up front so the run can test it: daily coverage of the floodable domain higher than any single product; CSI per stratum at least equal to the best single product in that stratum on held-out gold events; calibrated probabilities (a cell labelled 70% floods about 70% of the time). Resolution below GFM is accepted and will be stated in the docs.

## Plan: phases and the gate

The run starts by auditing what is already in the repo and reading the knowledge base, then decides from gold data alone whether the full fusion is worth building before any pipeline exists. Percentages are the effort budget handed to Claude Code; scope is cut at phase boundaries, never by skipping tests or validation.

| Phase | Effort | What happens |
| --- | --- | --- |
| 0 Inventory | 5% | Repo audit, knowledge base, which products and years are reachable |
| 1 Feasibility | 15% | Gold only: accuracy by stratum, oracle test, coverage test |
| Gate | | Yes if oracle gain >= 3 CSI points or 3-day coverage gain >= 15 points. No: 5-basin prototype plus a feasibility report explaining why fusion loses |
| 2 Data cube | 20% | Gold footprints with dry windows plus a stratified global sample |
| 3 Accuracy table | 10% | Sens and spec per source per stratum, with 95% intervals |
| 4 Fusion engine | 20% | Tiled daily job, proven on synthetic truth before real data |
| 5 Validation | 20% | Held-out events: CSI, reliability, coverage, timing, ablation |
| 6 Packaging | 10% | Register dataset, schedule job, docs, final report |

## Design decisions fixed before the run

These are settled so the autonomous run cannot wander. Each one maps to a numbered rule in the prompt.

1. Grid: 0.01° (~1 km) WGS84, global, restricted to the floodable domain (HAND < 15 m) with permanent water removed (JRC Global Surface Water occurrence >= 90%).
2. Value rule: every product raster holds 1 = observed flood, 0 = observed dry, NoData = could not observe (cloud, no overpass, snow or frozen ground, the product's own exclusion mask). Unobserved is never 0.
3. Strata: 6 classes = land cover {open, vegetated, urban} x snow season {yes, no}. One static raster per month.
4. FloodScan and GFDS share a sensor family, so they become one microwave vote. Three sources enter the model: SAR, microwave, optical.
5. Training is gold-first. The accuracy table (sensitivity and specificity per source per stratum) comes from confusion matrices against EMS and UNOSAT maps, including matched dry days. EM fills only strata with too little gold.
6. Prior v1 is the base flood rate per stratum. GloFAS enters only as prior v2, only if access is quick.
7. Daily combination is naive Bayes in log-odds; NoData adds nothing. Yesterday's log-odds carry forward with a 0.9 decay toward the prior.
8. Validation holds out whole gold events and basins, never individual cells.
9. Stack: Python, xarray, dask, rioxarray, numpy, pandas. Config-driven, tiled, idempotent, resumable. Storage, compute and credentials follow the DS knowledge base.

## Risks and what the run does about them

The biggest risk is building a fusion that is no better than GFM alone, so the run tests that first and scales itself down if the answer is no.

| Risk | What the run does |
| --- | --- |
| Fusion not better than the best single product | Phase 1 feasibility gate on gold only: oracle test (best product per stratum with hindsight) and coverage test. Below threshold: 5-basin prototype plus a report explaining why, instead of the full build. |
| Gold maps are biased (big events, certain regions, mostly Sentinel-derived) | Metrics reported per stratum and per continent; strata with thin gold flagged with wide confidence intervals; agreement with GFM on Sentinel-derived gold reported as a caveat. |
| Correlated errors double-counted | FloodScan and GFDS merged before modelling; EM assumes independence only across SAR, microwave, optical. |
| A product, year or region is inaccessible (FloodScan licence, GFM API limits, archive gaps) | Recorded in the Phase 0 inventory; run proceeds with what exists; ablation in Phase 5 shows the cost of each missing input. Nothing is fabricated. |
| Wrong NoData handling silently ruins the accuracy table | Unit tests on synthetic data with known truth must recover the planted sens and spec before any real data is used. |
| Probabilities look right but are miscalibrated | Reliability diagrams per stratum are a required deliverable; slope outside 0.8-1.2 is flagged. |
| Scope creep or running out of time | Effort budget per phase; scope is cut at phase boundaries, never by skipping tests or validation. |
| Storage and compute conventions ignored | Knowledge base is read first; every path, dataset registration and job follows it. |

## The prompt

See claude_code_prompt.md. Paste it as the single opening message in Claude Code, inside the repo.

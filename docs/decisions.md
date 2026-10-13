# Decisions log (flood fusion v1)

This is the running log of every choice the fixed design left open. The prompt's eight fixed decisions are not repeated or reopened here. Where something was ambiguous, the entry says what was picked, why it is the closest reading of the fixed decisions, and what was rejected. The one decision with lasting architectural weight, superseding the 2026-09-18 spec, also has an ADR: [`docs/decisions/0001-flood-fusion-v1-supersedes-2026-09-18-spec.md`](decisions/0001-flood-fusion-v1-supersedes-2026-09-18-spec.md).

Each entry is numbered D-NN and is cited from code docstrings and the config.

## Phase 0: setup and inventory

**D-01 Branch and scope.** Work happens on `feat/flood-fusion-v1`, cut from `docs/flood-fusion-spec` (PR #33) so the earlier spec remains readable next to what replaces it. All new code lives in `src/ds_flood_gfm/fusion/`, and existing modules are untouched (`docs/repo_audit.md` §4).

**D-02 Committing deliverables.** `.gitignore` ignored `*.csv`, `*.json` and `CLAUDE.md`, which would have kept `accuracy_table.csv`, the split and `CLAUDE.md` out of git. Narrow negations were added (`!/CLAUDE.md`, `!results/`, `!results/**`). Blanket ignores were left alone everywhere else. Commits carry no AI attribution lines, per the user's global CLAUDE.md, which overrides the harness default.

**D-03 Grid and FloodScan resampling.** The grid is 0.01° EPSG:4326 with integer cell indices (`fusion/grid.py`). GFDS's 0.09° pixels nest exactly 9×9 in it (pixel edge k lies at −180 + 0.01·(9k − 5)), so GFDS needs no resampling. FloodScan's 300″ grid does **not** nest (0.0833/0.01 = 8.33), so each 0.01° cell takes the SFED of the FloodScan cell containing its centre. Area-weighting across two FloodScan cells was rejected: it would invent sub-cell structure the product does not have.

**D-04 Floodable domain from 30 m HAND.** A cell is floodable iff at least **30%** of its area has HAND < 15 m (`static.hand.floodable_min_frac`).
- **Why 30%:** a cell can only be labelled flood if at least 30% of it is flooded (fixed decision 1). If floods occur only where HAND < 15 m, a cell with less low ground than that can never reach the flood threshold, so the domain cut and the flood threshold use the same number.
- **Rejected:** majority (≥ 50%), which drops valley cells that can genuinely be flood cells, and "any low pixel", which makes almost every cell next to a stream floodable.
- **How the fraction is measured:** by sampling every 4th pixel of the 30 m tiles, about 80 samples per cell.

**D-05 Snow season from ERA5-Land, not MODIS.** The fixed decision names a monthly MODIS snow-cover climatology. MOD10CM is reachable only with a NASA Earthdata login, which the team has not configured and the KB does not cover, so the equivalent used is the **ERA5-Land monthly mean snow cover, 2014–2023**.
- **Resolution:** the CDS regrids it to 0.25°, and each 0.01° cell takes the nearest 0.25° value.
- **Threshold:** a month is snow season iff its mean snow cover is ≥ **50%**, meaning snow lies over most of the cell for most of the month.
- **Units:** ERA5-Land's documentation contradicts itself ("%" vs a 0–1 fraction). The data are 0–100: the Alps in January exceed 50%.
- **Tested values:** Lokoja (Nigeria) is never snow season. The Alps are 82% snow season in January and 0% in July. The Canadian prairie is 100% in January, 30% in April and 0% in May.

**D-06 Permanent water.** A cell is permanent water, and is removed from the domain, iff at least **50%** of it has GSW occurrence ≥ 90%. A smaller permanent-water share stays in the domain, because every source and the gold exclude normal water from "flood" pixel by pixel. GSW v1.4 is used (D-28 records why not v1.5).

**D-07 GFDS flood/dry rule (R1 physics).** GFDS has no flooded fraction of its own, yet the fixed OR-merge needs a daily flood or dry call from it. The rule used is the repo's FloodScan-independent R1 route, `f = (s_ref − s)/(s_ref·K)` with K = 0.35, where `s_ref` is GFDS's own static baseline mean. A cell is flood iff f ≥ **0.30**, the same threshold FloodScan uses.
- **Rejected:**
  - R0, which is calibrated on FloodScan and would make the "independent" second microwave vote circular.
  - A σ-anomaly threshold, which is not a fraction and has no shared threshold with FloodScan.
  - The 4-day trailing mean, which is not a daily observation.
- **Local check on Nigeria 2022** (157 days, cached): the sign is right, with the signal 0.08–0.12 below the baseline over SFED ≥ 0.30 cells. R1 ≥ 0.30 flags 1.85% of cell-days against 0.94% for SFED ≥ 0.30. It recovers 37% of SFED's flood cells, and 19% of R1's flags agree with SFED.
- **Implication:** R1 at 0.30 is noisy. The OR-merge therefore inherits GFDS false alarms, and the Phase 5 ablation measures the cost (FloodScan-only vs GFDS-only vs OR).

**D-08 GFM read at the 80 m overview.** The 20 m COGs carry overviews at 2× to 32×. On a flooded Pakistan scene (20220903, E018N030T3), flooded fractions over 400 blocks of 2 km compared between the level-1 overview (80 m, 4×) and full resolution as follows:
- correlation 0.9997
- mean absolute error 0.005
- bias +0.0025
- 99.5% agreement on the 0.30 flood/dry call

The overview is about 16 times cheaper to read and was adopted.

**D-09 GFM observed cell.** A pixel counts as observed iff `ensemble_flood_extent ≠ 255`, `exclusion_mask = 0` and `advisory_flags ∉ {1, 3}` (D-17). A cell is observed iff observed pixels cover at least **50%** of it, where the expected pixel count comes from the cell's area and the 80 m pixel size. It is then flood iff flooded pixels make up at least 30% of its observed pixels. Excluded pixels read 0 in the flood layer, so treating them as dry would break fixed decision 1. They are dropped explicitly.

## Gold labels and events

**D-10 Which gold label sets are used.** A label set is kept only if all of the following hold. Every excluded set is listed with its reason in `results/label_sets_excluded_summary.csv`.
- **One acquisition day.** `acq_precision ∈ {date, minute}`, because same-day matching needs a day.
- **Year ≥ 2015**, the start of the GFM archive.
- **A valid mask exists.** Without one, dry cells cannot be known.
- **Not coarse optical.** UNOSAT `sensor_class` is not `optical_coarse`: those are VIIRS/MODIS-derived maps, and VIIRS is now an input.
- **A flood layer exists.** UNOSAT sets with only "water", "pre-flood water" or "flood possible" are excluded.

**D-11 Gold truth per cell.** Gold polygons are burned at 0.001° and averaged in 10×10 blocks. That gives fractions on the grid; it is the "fine burn plus block-sum" option leaned towards on 24 Sep and avoids exactextract's per-geometry cost. A cell has a label iff at least 50% of it lies inside the valid mask. It is flood iff flooded area is at least 30% of the valid area (the same threshold as the sources). It is unlabelled iff "flood possible" covers at least 30% of it. When several sets cover the same day, flood beats dry. CEMS v1 geometries are forced to 2-D polygons first, since they are 3-D with line fragments. Cells outside the floodable domain are never truth.

**D-12 Dry windows: literal reading, diagnostic added.** "30 days before event start, 60 days after event end" is read literally as the windows [start − 30, start − 1] and [end + 1, end + 60]. Every footprint cell in those windows is dry truth. Event start and end are the first and last gold days of the event.
- **Rejected:** offsetting the post window to start later. It would have avoided counting flood recession as false alarms, but it is not what the decision says.
- **The risk is measured instead.** The accuracy table step reports each source's false-alarm rate in the post window split by days after the event end (1–30 vs 31–60). The three numbers (`dry_pre_days`, `dry_post_gap_days = 0`, `dry_post_days`) are in the config.

**D-13 Events and windows.** An event is one activation code. Its label-set bboxes are merged when they come within 0.05° of each other, and each merged box becomes one processing window. This keeps empty space between distant AOIs out of the reads. The full window period is [start − 30, end + 60], and every day in it is built; nothing is subsampled.

**D-14 Event selection for v1.** Events are capped by a label partition of at most 300 MB and windows of at most 80,000 cells. Within each continent they are taken cheapest first (cells × days), up to these quotas:

| continent | quota |
|---|---|
| Africa | 45 |
| Asia | 45 |
| Europe | 50 |
| North America | all |
| South America | all |
| Oceania | all |

That gives 187 events on six continents, against the prompt's aim of at least 150 events on at least 5 continents. Excluded events and reasons are in `results/events_excluded.csv`. The caps remove the largest events (for example Pakistan 2022 and Bangladesh 2020), which biases v1 toward smaller events. This is recorded as an open issue.

**D-14b Snow supplement.** The cost-ordered quotas left the two snow strata with almost no gold. Across the first 135 events there were **2** gold flood cells in snow strata and 0 in urban-snow and vegetated-snow. So every other usable event was added if it meets both conditions:
- at least 5% of its window is snow season in a month that has a gold day;
- it fits the size caps.

That added 11 events, all in Europe (EMSR268, 280, 283, 421, 442, 511, 557, 644, 668, 783, 856), for **198 events** in total; the selection column in `results/event_split.csv` marks them `snow_supplement`. EMSR427 qualified on snow but has 170,000 cells and was left out. Nothing had been fitted when this was done, and the split was regenerated with the same seed: 42 of the original 187 events changed side, which is harmless before any fitting.

**D-15 Split.** Each event is assigned to the HydroBASINS level-4 basin nearest its centroid. Per continent, whole basins are drawn at random (seed 20261012) until 25% of that continent's events are held out. No basin is on both sides; the code asserts this. Result after D-14b: **149 training events and 49 held-out events**. The split is frozen in `results/event_split.csv`. Phase 1 uses **training events only**, so the held-out set stays unseen until Phase 5.

## Sources

**D-16 Optical slot: VIIRS, not MODIS.** The MODIS NRT flood product cannot be reached in v1. `MCDWD_L3_NRT` starts in 2026 (LAADS lists 2025 days 346–365 and 2026), the reprocessed `MCDWD_L3` covers 2003–2025, and both need an Earthdata login. The fixed design allows VIIRS "only if already archived". NOAA's public archive (`s3://noaa-jpss/.../VFM_1day_GLB`) is such an archive, so VIIRS v2r0 takes the optical slot, and the model keeps its three independent sensor families (SAR, microwave, optical).
- **Gap:** VIIRS has no files for 2021-01 → 2023-07.
- **Mapping to the value rule:**
  - NoData: no data, cloud, shadow, snow/ice, river ice, supra-snow water.
  - Dry: land, vegetation, normal water.
  - Flooded fraction: 15 counts as 100%; codes 100–200 are (v − 100)%.
  - Cell rule: the same 50% observed / 30% flood rule as GFM.

**D-17 GFM snow and frozen ground.** GFM's exclusion mask does not exclude snow. Its `advisory_flags` value 1 ("low regional backscatter: snow-covered, frozen or dry soil") and value 3 (1 plus rough water) are GFM's own snow/frozen-ground flag, so fixed decision 1 makes them NoData. Note that flag 1 also covers dry soil, so some arid pixels become NoData. The rough-water flag on its own (2) is left observed.

**D-18 FloodScan regions.** Africa uses the prod daily COGs, which are current. Asia-Australia and the Americas use the dev raw NetCDFs, which end 2026-03-31. Where grids overlap (50–60°E) the Africa product wins. Outside every region, or after a region's archive ends, cells are NoData; Europe is never covered. FloodScan has no unobserved state inside its regions.

**D-19 Compute and politeness.**
- **Where it runs:** on the user's laptop under `nice -n 19`, with at most 2–3 worker processes, small or cheap units first, resumable per event.
- **Databricks** was not used for the backfill, because it would need the code merged first (D-27).
- **GFDS reads** are fetched once per day, as row bands covering every window active that day; the per-event alternative would have meant about 40,000 requests.
- **GFDS politeness was revised after measuring the server** (2026-10-12). Each range request costs about 1.1 s, and bandwidth is about 0.3 MB/s per connection. That made the planned single connection at 1 request/s a 10+ hour prefetch (60 s/day once the sample blocks were added).
- **Current rule:** at most 4 connections, at least 1 s between requests on each (so the server sees ≤ 4 requests/s), and needed rows merged across gaps of ≤ 16 rows.
- **Rejected:**
  - whole-file downloads: 64 s/day;
  - merging into huge bands: the bytes dominate.
- **403 handling:** a 403 from GDACS counts as "day absent" only after a known-good file reads. A block therefore cannot pass as a gap.
- **Fetch order:** sample-period days (2023–2025) first, so that the Phase 2 sample cubes can build while the rest downloads. Stage C follows the same order.

## Metrics

**D-20 Metric definitions.** These are the formulas in `fusion/metrics.py`:
- **Hit rate:** TP/(TP+FN).
- **False-alarm rate (POFD):** FP/(FP+TN).
- **False-alarm ratio:** FP/(TP+FP), also reported.
- **CSI:** TP/(TP+FP+FN).

How the cells are chosen:
- **Conditional scores** use only cells where both the truth and the source observed.
- **"All-cells" scores** count a source's NoData as "not detected", which is what a map user sees.
- **CSI and hit rate in Phase 1** use gold days only, so the map is compared with the map. Specificity is reported both on gold days alone and with the dry windows.
- **The oracle gate uses the conditional CSI**, which is accuracy where the source looked. Coverage is tested separately. A source needs at least 200 observed gold flood cells in a stratum to be chosen there.
- **Coverage gain** is "any source within the last 3 days" minus "GFM alone, same day", as written. "Minus GFM within 3 days" is also reported.

**D-21 Prior v2 not built.** The GloFAS reanalysis and forecast sit on EWDS. Using them would mean a separate `.cdsapirc` endpoint, accepting licences per dataset, and a request schema that changed in July 2026. On top of that, prior v2 needs return-period thresholds, a nearest-river-cell mapping and a regression. All of that together is well over 5% of the effort, so v1 uses prior v1 only, and prior v2 is ranked as a next improvement.

## Phase 2–6 decisions

**D-22 Registration and scheduling follow the KB's actual mechanisms.**
- **No dataset registry.** The KB has no registry for team-produced datasets. Its registration routes are:
  - a `pipelines/*.md` page with `deployment.jobs`, `inputs` and `outputs` frontmatter;
  - the generated `assets/<project>/` blob inventory;
  - the pipeline registry, which discovers Databricks jobs automatically.
- **What v1 adds:**
  - the dataset's own `MANIFEST.json` on blob;
  - a KB page, `pipelines/flood-fusion.md`, opened as a draft PR from a worktree, as the KB asks;
  - a Databricks Asset Bundle (`databricks.yml`) using the team's `run_task.py` wrapper and the Job Compute policy.

**D-23 Stratified random sample (Phase 2).**
- **Blocks:** 0.25° blocks drawn at random, one per Köppen main group (Beck et al. 2018) × continent cell, plus 12 city-centred 0.5° blocks (6 in Köppen D/E climates, 6 elsewhere), because urban strata are rare.
- **Top-up:** reserve blocks are added until each stratum has at least 2 million cell-days over 2023–2025, capped at 25 extra blocks. Any shortfall is reported in `results/phase2_sample_cell_days_by_stratum.csv`, not hidden.
- **Why 2023–2025:** the three most recent full years. GFM has two satellites again there (S1A, plus S1C from 2025), which is closest to how the product runs today.
- **Prior v1** is the per-stratum prevalence MLE on this sample with sensitivity and specificity fixed from the accuracy table. A one-parameter EM is identifiable, unlike a joint fit. It needs the gold table, so `phase3_accuracy.py` computes it straight after the table.

**D-24 Dawid–Skene EM.**
- **When it runs:** for any stratum with fewer than 50,000 observed gold cell-days (the maximum over the three sources), **or** where any model source has zero observed gold flood cells. In that second case sensitivity would be 0/0, which is not an estimate, so the 50,000 rule cannot be read as requiring it.
- **Data:** gold cells enter labelled and sample cells unlabelled (semi-supervised).
- **Initialisation and prior:** values come from the nearest stratum with complete gold estimates: same land cover with the other snow state, otherwise the same snow state. The fit is MAP with Beta priors of 200 pseudo-counts centred on those values. Plain maximum likelihood with three binary sources and sparse joint observations can drift to degenerate solutions; MAP keeps it near the donor unless the data say otherwise.
- **No data at all:** a stratum with neither gold nor sample observations gets the donor's values, `method = no_data_donor_copy(...)`, and a forced flag. It never passes as an estimate.
- **Hard checks:** `Model.from_tables` raises on any non-finite sensitivity, specificity or base rate.

**D-25 Storage.** Following the KB path convention, outputs go to the dev account, container `projects`:
- `ds-flood-gfm/processed/flood_fusion_v1/` with `daily/`, `static/`, `cubes/`, `results/` and `MANIFEST.json`;
- inputs we downloaded, under `ds-flood-gfm/raw/{datasource}/`.

The account and container follow the flood-gfm precedent; the KB has no rule for experimental products. FloodScan-derived outputs stay internal because of the licence.

**D-26 Operational scope of v1.** The brief asks for a daily global product.
- **What is built:** the engine and job are tiled and global-capable. The static layers (HAND, GSW, WorldCover) are built on demand per 1° tile and cached.
- **What is not:** a one-off **global static build** (about 15,000 land tiles) is too heavy for the user's laptop. It is provided as the `flood_fusion_static` job (`scripts/fusion/static_build.py`) and was **not run**.
- **What the daily job covers:** two AOIs where the team runs flood frameworks (Nigeria Niger–Benue, Chad Logone–Chari). All held-out event windows are produced for validation.
- **Extending coverage:** add AOIs or run the static job; no code change is needed.

**D-27 Prod deployment is gated on review.**
- **Prod:** the team convention (KB `conventions.md`) is that code reaches `main` through a reviewed PR, and DAB prod jobs pull `main`. A prod job therefore cannot run this code before review, and v1 does not deploy to the prod target.
- **Dev:** the bundle is validated for both targets and deployed to the **dev** target, where development mode pauses the schedule. The prod deploy is a single documented command for after the merge.

This is a constraint of the team's merge convention, not a step skipped.

**D-29 GFM scene-level absences.** Two cases:
- **Flood or exclusion layer missing.** Either the asset is listed in STAC but its file returns 404, or the item does not list the flood or exclusion layer at all; both occur in the archive. Without the exclusion mask, excluded pixels (which read 0) cannot be told apart from dry ones. So the scene is dropped and its id recorded in the cube attributes (`gfm_dropped_scenes`). Config: `on_missing_asset: drop_scene`.
- **Advisory layer missing.** Some product versions have no advisory layer at all, and many 2015–2016 archive scenes list an advisory file that returns 404 while their flood and exclusion files are fine (8 of 8 dropped scenes checked for EMSR174). Either way there is no GFM snow flag to apply, so the scene is read without it and recorded in `gfm_scenes_without_advisory`. Config: `on_unlisted_advisory: read_without_flag`. The first version of this rule dropped those scenes; cubes built that way were rebuilt. Zero-byte files exist upstream too; one advisory file has a 1970 timestamp. An empty advisory file is treated as missing. An empty flood or exclusion file makes the scene unusable, so it is dropped and recorded.

**D-30 Metric details.**
- **Oracle candidates.** In each stratum the candidates are sources with at least 200 observed gold flood cells, plus the overall best source. This keeps a well-sampled source from losing to one with almost no evidence. The flood-weighted per-stratum gain is also reported; it is ≥ 0 by construction.
- **Time to detection** has two forms:
  - first day on or after the first gold day when at least max(5, 10%) of the gold flood footprint is flagged;
  - first such day anywhere in the window. Negative means before the first gold map: either the flood had already begun or it is a false alarm.
- **Reliability** has two forms:
  - slope from weighted least squares of observed frequency on mean predicted probability across 10 bins (the criterion);
  - logistic calibration slope from a binomial MLE on the bin counts (reported too).

  A stratum whose predictions have no spread gets NaN ("not estimable"), not a number.

**D-31 Stage C failures are opt-in to skip.** `phase1 sources --continue-on-error` writes failing events to `results/phase1_stage_c_failures.csv` and the log. The metrics step then excludes them with a warning that lists them. Without the flag, the first failure stops the run.

## Static layer versions

**D-32 Jeffreys smoothing, and flags that say why a number is weak.**
- **Smoothing.** Gold sensitivity and specificity are Beta(½, ½) posterior means, (k + 0.5)/(n + 1), and the bootstrap intervals use the same estimator. Raw proportions are kept as `sens_raw`/`spec_raw`. In the dry run, microwave sensitivity in vegetated/no-snow came out at exactly 0 from 46 flood cells. Clipped to 0.001, that turns a microwave "flood" into a strong *dry* signal. Smoothed, it is 0.011, a near-neutral weight. This is an estimator choice inside fixed decision 5, not a change to it.
- **`prior-dominated` rows.** An EM row is marked this way, and flagged, when the effective number of flood or dry cells the source contributed under the EM posterior is below the Beta prior's 200 pseudo-counts. Such a row is the donor's value, not an estimate.
- **`flag_small_n`.** A gold row is flagged when it has fewer than 30 observed flood or dry cells, because a bootstrap interval from one event can collapse to zero width.

**D-33 Ablation: evidence only on change.** Fixed decision 6 fixes the recursion; it does not define what counts as a new observation. A source that reports the same value at the same cell on consecutive days is not independent evidence under the model's own assumption if its errors persist. GFDS false alarms tied to a static baseline do persist. Phase 5 therefore adds two **ablation-only** variants:
- `microwave_on_change`: microwave evidence counts only on days its value differs from the day before;
- `all_on_change`: the same for all three sources.

NoData breaks a run, so the first observation after a gap counts. The product itself stays exactly as specified.

**D-34 Numerical guards.** The two clips do different jobs and are kept separate:
- **Sensitivity and specificity** are clipped to [0.001, 0.999], so no single observation can contribute a weight beyond about ±7 log-odds.
- **The prior base rate** is clipped to [1e-5, 0.999]. The sample MLE puts several strata near 1e-5 to 1e-4; clipping the prior at 0.001 would have raised those priors as much as 100-fold without saying so.

**D-28 GSW v1.4 kept.** JRC now publishes v1.5 (1984–2024) and states that occurrence in v1.2–1.4 has consistency errors. v1 keeps v1.4 because:
- the ≥ 90% permanent-water cut is the least sensitive part of the occurrence range;
- v1.5 is on a different endpoint, mixes Landsat collections, and has a co-registration offset of ≥ 1 pixel.

Moving to v1.5 is an open issue.

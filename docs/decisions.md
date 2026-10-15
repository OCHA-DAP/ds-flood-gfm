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
- **Prior v1** was planned as the per-stratum prevalence MLE on this sample, with sensitivity and specificity fixed from the accuracy table. That estimator collapsed, and D-35 replaces it. The step runs in `phase3_accuracy.py` straight after the table.

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

**Gotcha.** DAB development mode does **not** pause a schedule whose `pause_status` is set explicitly. The first dev deploy had `UNPAUSED` and created a live daily schedule; it was caught by reading the job back two minutes later, before any trigger fired. The bundle now sets `PAUSED` in the base definition and `UNPAUSED` only in the prod target.

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

**D-35 Prior v1 estimator: GFM's flood rate in the random sample.**
- **Why the planned estimator failed.** That estimator (D-23) was the prevalence MLE on the sample with sensitivity and specificity fixed from the gold table. It hit its lower bound (1e-7) in the three no-snow strata, with zero-width intervals. The gold specificities come from dry windows next to flood events, where false alarms run higher than on random days (`results/phase3_post_window_pofd.csv`). Fixed at those values, the likelihood explains every flagged sample cell as a false alarm.
- **Estimator used.** Prior v1 is the share of GFM-observed sample cell-days that GFM flags as flood, per stratum, with block-bootstrap intervals. GFM has the lowest false-alarm rate of the three sources. This is the reading closest to "base flood rate per stratum, estimated from the training cube": the sample is the unbiased part of that cube.
- **What is kept.** The MLE is kept as `base_rate_mle` in `results/prior_v1.csv`.
- **Bias.** This rate includes GFM's own false alarms, so it is if anything an upper estimate.

**D-28 GSW v1.4 kept.** JRC now publishes v1.5 (1984–2024) and states that occurrence in v1.2–1.4 has consistency errors. v1 keeps v1.4 because:
- the ≥ 90% permanent-water cut is the least sensitive part of the occurrence range;
- v1.5 is on a different endpoint, mixes Landsat collections, and has a co-registration offset of ≥ 1 pixel.

Moving to v1.5 is an open issue.

**D-36 Agreement accuracy table (plan v2, ADR 0002): built; it does not identify flood, and gold shows why.**
- **What was built.** `scripts/fusion/phase3_agreement.py` (`make phase3`), with the logic in `fusion/agreement.py`:
  - unsupervised Dawid–Skene on the random sample, per stratum, from three starts; best log-likelihood wins; flood is the minority class;
  - a comparison with gold on evidence weights (ADR 0002);
  - a check on gold cells of the independence assumption the agreement table rests on.

  Thresholds are under `accuracy.agreement` in the config.
- **Even the best case recovers only specificity.** `scripts/fusion/experiment_agreement_identifiability.py` plants the gold accuracies, the sample's real observation rates and base rates, and independent errors. Over 8 synthetic samples (5th–95th percentile):
  - specificity comes back within about ±0.0005 in every stratum;
  - sensitivity does not. Open land: GFM 0.24–0.41, VIIRS 0.45–0.93. Vegetated and urban land: close to the whole 0–1 range (VIIRS 0.08–1.0, 0.01–1.0).

  Flood is too rare on ordinary days for agreement alone to pin down how often a source catches it.
- **On the real sample the "flood" class is something else.**
  - Open/no-snow: the class covers 2.4% of cell-days, 15 times the base rate. VIIRS sees 53% of it, microwave 19%, GFM 4%. That fits the products' different water baselines (VIIRS and FloodScan count seasonal water; GFM subtracts its own reference water), not gold flood.
  - Vegetated/no-snow does not converge.
  - Urban/no-snow is degenerate: two sources at sensitivity 1.0.
- **Comparison.** Per row: 1 agree, 2 diverge, 6 unstable, 9 with no gold to check (snow strata).
  - **Rule added during the build:** a stratum uses the agreement values only if *every* source agrees, because all three come from one joint fit. The single "agree" (microwave, open/no-snow) sits in a fit that diverges for GFM and VIIRS.
  - **Result:** every stratum keeps gold, and `results/accuracy_table.csv` is unchanged.
- **Why.** On gold-dry training cells the sources' false alarms coincide far more often than independence predicts (`results/phase3_gold_error_dependence.csv`):
  - **open/no-snow:** microwave + VIIRS 141,077 joint false alarms, lift 5.2; GFM + VIIRS lift 7.9;
  - **vegetated/no-snow:** lifts 4.8–17.7;
  - **within the same event:** still 2.2–9.1 in both strata, so the same cells, not just wetter events;
  - **misses on flood cells:** barely coincide (lift 1.0–1.5).

  This fails the agreement table's assumption and is the measured cause of v1's over-confidence: naive Bayes adds two coinciding false alarms as two independent votes.
- **Not done.** No bootstrap intervals for the agreement table: one fit takes 10–30 s, so 200 replicates × 6 strata would take hours. The status calls do not use intervals.

**D-37 MODIS flood product (MCDWD) reachable; scored against gold before any model use.** This supersedes D-16's "cannot be reached".
- **Access.** The source is LAADS DAAC, with an Earthdata **token** read from the environment variable `EARTHDATA_TOKEN`, never from code. LANCE and LAADS accept only a token for scripted downloads.
  - The token expires (the profile page shows the date).
  - For a scheduled job, use the team's shared `dsci` Earthdata account from secrets, as the IMERG pipeline does (KB `pipelines/imerg.md`), and add a token renewal step.
- **Archive.**
  - `MCDWD_L3`: reprocessed, complete 2000-01-01 to 2025-12-31.
  - `MCDWD_L3_NRT`: archived near-real-time files from 2026-01-01, same layout.
  - Both were checked on LAADS on 2026-10-13.
- **Format.** HDF4 tiles of 10° × 10° on a geographic grid, 4800 × 4800 pixels of 0.0020833°. The `rasterio` wheel's GDAL has no HDF4 driver, so files are read with `pyhdf`, an added dependency.
- **Flag values** come from each layer's own description attribute: 255 insufficient data, 0 no water, 1 normal (reference) water, 2 recurring seasonal flood, 3 unusual flood. Under the value rule:
  - 255 → NoData;
  - 1 → not flood;
  - 3 → flood;
  - 2 → flood or not, per `recurring_as`. Both are scored, because code 2 is NASA's own seasonal baseline.
- **Composite.** As daily model evidence, only the 1-day layers qualify. The 2- and 3-day composites reuse the same satellite passes on consecutive days, which would be counted again under the carry-over. The default is the cloud-shadow-screened 1-day layer.
- **Status.** MODIS is not a model source (`sources.modis.enabled: false`). `scripts/fusion/experiment_modis_gold.py` scores it against gold beside VIIRS, first for the events in the VIIRS archive gap (2021-01 to 2023-07). If it enters the model, it shares one optical vote with VIIRS, not a vote of its own (D-36).

**D-38 Seasonal "normally wet" baseline from JRC GSW monthly recurrence (experiment).**
- **Why.** On gold-dry cells the sources' false alarms coincide 4–18 times more often than independence allows (D-36), and the agreement fit's "wet" class looks like seasonal water. A per-cell, per-month baseline should remove those calls.
- **Source.** `gs://global-surface-water/downloads2021/monthlyRecurrence2021/`: 12 calendar months, Landsat 1984–2021, 30 m. It's a public bucket, the same one v1 reads occurrence from, and needs no Earth Engine or login.
  - Tiles are 10° and 40,000 × 40,000 pixels of 0.00025°, so a 0.01° cell is exactly 40 × 40 pixels.
  - Code: `fusion/baseline.py`.
- **Rule, fixed before any result was seen:**
  - a pixel is normally wet in month m if it was water in at least 50% of the observed years;
  - a cell is normally wet if at least 30% of its pixels are.
  - Masking turns a source's *flood* call in a normally-wet cell into NoData. Dry calls are unchanged.
- **Caveat.** In this export the `has_observations` layers are the same for every month and match "occurrence > 0" for 99% of pixels (checked on the Inner Niger Delta). They mark "water ever seen", not "this month was observed". A persistently cloudy month reads 0, so the baseline can miss cloudy-season water: it errs towards masking too little.
- **Test.** `scripts/fusion/experiment_seasonal_baseline.py` reports, on training and held-out events separately:
  - how much gold flood falls on normally-wet cells;
  - per-source recall and precision before vs after masking;
  - the coinciding-false-alarm lift before vs after.

  Nothing is fitted. If the baseline helps, the next step (rebuilding the accuracy table and rerunning fusion) needs a decision, since it changes what each source counts as evidence.

**D-39 Dry-window labels: measured, and a corrected variant (proposed, not adopted).**
- **What the literal reading does.** Fixed decision 5, read literally (D-12), labels every gold-footprint cell dry for 30 days before and 60 days after the gold days. That includes cells flooded on the gold days, so rising and receding floodwater is scored as a false alarm. `scripts/fusion/diagnostic_dry_windows.py` measured it on held-out events:
  - only **2%** of cell-days with P ≥ 0.5 fall on gold days; the rest are dry-window days labelled dry;
  - in open land, cells flooded on a gold day sit at P ≥ 0.5 on 48% of days 1–30 after the event, 25% of days 31–60 and 14% of the days before;
  - never-flooded cells do so on 5%, 4% and 2% of those days.
- **Consequences for v1's results:**
  - **Criterion (c).** The reported slope (about 0.01) was measured over gold days plus literal dry windows. On gold days only it is 0.47 in open land, 0.34 vegetated and 0.27 urban: still failing 0.8–1.2, but by much less. On gold days the mean P is at or below the observed flood frequency (open land: 0.15 vs 0.22).
  - **Accuracy table.** False-alarm rates were inflated. With formerly flooded cells excluded from the dry windows (training events), open-land 1 − specificity falls: VIIRS 5.5% → 2.6%, GFM 1.3% → 0.44%, microwave 5.1% → 3.9%. Flood reports were therefore given too little weight.
  - **Recalibration** (D-34, experiment). Its fit pooled the same dry windows, so its "about 12 times too extreme" was largely fitting label noise.
  - **Dependence** (D-36). Most joint false alarms are dry-window days (54% are days 1–30 after the event). But the lift is about the same on gold days (microwave + VIIRS 4.3, within-event 2.2), so the dependence itself is real.
- **Corrected variant.** `scripts/fusion/experiment_dry_window_fix.py`: a cell flooded on any gold day of its window is NOTRUTH, not dry, in the dry windows. Specificity is re-estimated on training events (sensitivity cannot change) and held-out events are re-run in memory. Held-out gold days at P ≥ 0.5:
  - **all strata:** recall 0.37 → 0.44, precision 0.55 → 0.55, CSI 0.285 → 0.322;
  - **open land:** recall 0.38 → 0.46, precision 0.58 → 0.58, CSI 0.300 → 0.341, above VIIRS's 0.333;
  - **vegetated and urban:** about unchanged.

  Reliability with corrected labels is still about 0.03. Never-flooded footprint cells reach P ≥ 0.5 on 2–7% of dry-window days and outnumber gold-day cells about 50 to 1. Whether those are false alarms or water the analysts did not map cannot be told with these labels, so criterion (c) cannot be measured cleanly with them.
- **Status.** Proposed on 2026-10-13; adopted for the cross-validated re-run on 2026-10-14 (D-45, ADR 0006).

**D-40 "Super-GFM" (learn GFM's map from daily inputs): first test does not beat GFM persistence.**
- **Idea (user, 2026-10-13).** Train on GFM's own calls as labels to get a GFM-like map on days without a Sentinel-1 pass.
- **Test.** `scripts/fusion/experiment_super_gfm.py`:
  - **Labels:** gradient-boosted trees on 4.2 M GFM-labelled cell-days from training events.
  - **Features:** microwave today and its last 3/7 days; VIIRS today, last value and age; GFM last value and age; static fractions; GSW monthly normally-wet share; season; latitude.
  - **Product scored:** "GFM if observed, else the model", on held-out gold days.
- **Results**, recall · precision (CSI):

  | product | all held-out gold days | events with non-Sentinel-1 gold (8,032 flood cells) |
  |---|---|---|
  | super-GFM | 0.25 · 0.77 (0.233) | 0.12 · 0.86 |
  | GFM persistence | 0.25 · 0.81 (0.239) | 0.14 · 0.68 |
  | prior-only control | 0.19 · 0.86 | |
  | GFM alone | 0.19 · 0.94 | 0.02 · 0.91 |
  | fusion v1 | 0.37 · 0.55 (0.285) | **0.38 · 0.50**, the best |
  | VIIRS | 0.41 · 0.60 (0.319) | 0.19 · 0.54 |

  - On days GFM observed, the model (which never sees GFM that day) reproduces GFM's flood at recall 0.30 and precision 0.58.
  - Permutation importance: GFM's last value dominates (0.21). The microwave features are at or below zero.
- **Reading.**
  - The daily inputs carry too little information to reproduce GFM between passes. The model collapses to persistence.
  - An emulator also inherits GFM's ceiling against gold: recall 0.19 overall, 0.02 on events with non-Sentinel-1 gold.
  - "GFM, else last GFM" is a high-precision, low-recall layer and may suit uses that need precision. Fusion is the higher-recall option.
  - Untested: a trained threshold (0.5 is conservative for a 1.3% class), and richer temporal features.
- **Result (2026-10-13).** Scored on the 52 events in the VIIRS archive gap: 249 event-days, 30,688 gold flood cell-days. Outputs: `results/experiment_modis_gold_gap.csv` and `_context.csv`.
  - **MODIS by composite** (recall · precision, all labelled cells; recall where observed in brackets): 1-day 0.19 · 0.73 (0.46); cloud-shadow-screened 1-day 0.17 · 0.75 (0.45); 3-day 0.12 · 0.87 (0.43). Counting the recurring (seasonal) class as flood is slightly better for every composite.
  - **The other sources on the same event-days:** fusion v1, which has no optical input there, scores 0.13 · 0.45 (CSI 0.112); GFM 0.10 · 0.87; the microwave vote 0.28 · 0.37. MODIS OR GFM scores **0.27 · 0.76 (CSI 0.247)**, the best.
  - **Reading.** MODIS fills the optical gap well. The next step is to make it part of the optical vote (with VIIRS where both exist). That needs MODIS over whole event windows, not only gold days: about 100+ GB of tiles, which belongs on Databricks rather than this laptop.
- **Subsetting (checked 2026-10-13).** NASA also publishes the composites as GeoTIFFs (`MCDWD_L3_{F1,F1C,F2,F3}_NRT`).
  - They are internally tiled (256 × 256, deflate; no overviews, not marked COG), so a window can be read remotely with the token. The daily job can use them without whole-tile downloads.
  - They are kept only for a rolling window of about 7 days. On 2026-10-13 the earliest in CMR was 2026-10-06, with nothing for 2022.
  - History (2000–2025) exists only as the HDF4 reprocessed archive. HDF4 cannot be range-read, so every needed tile-day must be downloaded whole (5–26 MB).
  - NASA GIBS serves the product as rendered imagery for any date. That is not the reprocessed data, and is not used.

**D-41 MODIS joins VIIRS in one optical vote; MODIS counts for every event window come from a Databricks job.**
- **Rule.** MODIS (MCDWD, D-37) and VIIRS are the same sensor family: optical, and blocked by cloud. They are merged into one optical vote the way FloodScan and GFDS form the microwave vote (fixed decision 2): flood if either says flood, dry if at least one observed and neither says flood. They are not added as two independent sources, which would double-count shared cloud gaps and shared errors (D-36).
- **Data.** Every event window, every day, needs about 25,000 MODIS tile-days (about 376 GB of HDF4 tiles, which cannot be read in part). `scripts/fusion/modis_windows.py` downloads each tile-day once, keeps only per-cell pixel counts of the 1-day layers, and writes one NetCDF per window to dev blob `processed/flood_fusion_v1/modis/`. It runs as the on-demand Databricks job `flood_fusion_modis`.
- **Test.** `scripts/fusion/experiment_optical_vote.py` scores four models on held-out events, with the v1 or the corrected dry-window labels, each with and without the optical vote, and splits the results by whether the event falls in the VIIRS gap.
- **Status.** Blocked: the job needs an Earthdata login it can use. The team login in the dsci secret scope is rejected by NASA's token API (HTTP 401). Options are the user's personal token in a Databricks secret, or fixing the team account.

**D-42 Scope: Africa, Asia, South America and Oceania, with every usable event in them (user decision, 2026-10-14).**
- **Scope.** Europe and North America are set aside for now (`scope.continents` in `configs/fusion.yaml`). FloodScan does not cover Europe, and pooling regions whose sources differ hides how each one performs.
- **Every usable event.** Within scope, every event that passes the gold rules (D-10) is used. v1's per-continent quotas and size caps (D-14) and the snow top-up (D-14b, all in Europe) no longer apply. That adds 51 events to the 119 in-scope v1 events, for 170:
  - 30 left out by the quotas (27 Africa, 3 Asia; 1.2 M cells);
  - 21 left out by the size caps (13 Asia, 8 Africa; 2.9 M cells, 4.9 GB of label data; Bangladesh 2020 alone is 2.1 GB).
- **Building them.** `phase1_feasibility.py extra` writes the two build lists `data/fusion/gold/events_extra_{small,large}.parquet`. The `gold`, `gfds` and `sources` steps take `--events-file` to build a list, skipping windows already built. Their failures go to `results/phase1_stage_c_failures_<list>.csv`, so v1's record is kept. The v1 selection files and `results/event_split.csv` are not touched; the five-fold design that replaces the split comes next.
- **Order.** The 30 smaller events first on the laptop, to measure the rate; the 21 large ones after.

**D-43 Five-fold cross-validation by river basin replaces the single split (user decision, 2026-10-14).**
- **Why.** 49 test events leave per-continent results thin, and those 49 have already been used to compare variants. With five folds, every in-scope event is tested once, by a model fitted without it. Fitting is counting, so five fits cost little.
- **Rule.** Whole HydroBASINS level-4 basins go to one fold (the D-15 basin rule). Basins holding at least 2% of all gold flood are placed first, each to the fold with the least flood so far; every other basin goes to the fold with the fewest events of its continent. A seeded shuffle breaks ties. Config: `gold.n_folds`, `gold.fold_seed`, `gold.fold_big_basin_share`; code: `events.assign_folds`, tests in `tests/test_folds.py`.
- **Why flood area first.** A few events dominate the flooded area: five hold 45% of the gold flood in the 119 built in-scope events. Balancing event counts alone gave one fold 34% of the open-land flood and another 6%; placing the big basins first gives 18–22%.
- **Output.** `scripts/fusion/make_folds.py` writes `results/event_folds.csv` and `results/event_folds_balance.csv`. It runs only once every in-scope event is built, and the fold file is frozen like the split (a different assignment is refused without `--overwrite`).
- **Discipline.** Anything tuned is tuned inside each fold. No event set stays untouched, so every rule is fixed before the re-run's results are seen. `results/event_split.csv` stays as v1's record.

**D-44 Three land types replace the six strata; the snow split is dropped (user decision, 2026-10-14). ADR 0005.**
- **Why.** 518 of the 520 gold flooded cell-days in snow strata are in Europe, now out of scope; in scope the snow strata would be donor copies (D-24).
- **How.** The cubes keep the 6-class code; land type is `code // 2`. Rates are fitted per source and land type and written to both snow codes, so `engine.Model` is unchanged. The prior is the GFM flood share in the 30 in-scope sample blocks, per land type. No EM fallback: a land type with no observed gold flood for a source stops the run.
- **Not changed.** The operational job still reads v1's six-row table.

**D-45 Dry-window labels: the corrected rule is adopted for the re-run (decided under delegation, 2026-10-14). ADR 0006.**
- **Rule.** In the 30 days before and 60 after an event, cells flooded on any map day of their window get no label; other examined cells stay dry. `gold.dry_window_rule: corrected`; applied at read time by `gold.corrected_dry_windows`.
- **Why.** D-39: the literal rule scores draining floodwater as false alarms, inflating false-alarm rates and the measured over-confidence.
- **Who decided.** The user asked on 2026-10-14 for the work to be finished without further questions; this was decided then, before any cross-validated score existed. The literal rule is reported as one sensitivity row and can be restored in the config.
- **Also.** The unused `gold.dry_day_step` key was removed: every dry-window day has always been used, despite docs/methods.md saying every 3rd (corrected).

**D-46 Event-level scoring rules (decided under delegation, 2026-10-14). ADR 0007.**
- **Caught:** on some day from 3 days before the first map to 3 days after the last, at least 10% of the event's gold footprint (floodable cells flooded on any map day), and at least 5 cells, flagged. **Days to detection:** first such day, from the first map. **False event:** the same rule on days 30 to 10 before the first map. Config `event_level`.
- **Deferred:** false events on the global random sample (no footprint to apply the rule to).


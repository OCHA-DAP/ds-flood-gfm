# Flood fusion: work log and reproduction ledger

One entry per step, in order. Each gives the question, the command that reproduces it, where its outputs land, the result in one line, and the commit. Read `docs/methods.md` for what the numbers mean, `docs/decisions.md` and `docs/decisions/` for why each rule was chosen, `docs/final_report.md` for the full write-up. Branch `feat/flood-fusion-v1` (not pushed as of 2026-10-21).

Conventions used throughout:
- Every rule is fixed and committed **before** the script that uses it runs; follow-ups that change a rule after seeing results are labelled as such.
- Every number in the docs comes from a file under `results/`; nothing is typed by hand.
- Scripts are resumable (outputs that exist are skipped) and fail loudly (a missing input raises; no event is dropped silently).
- Long runs on a laptop: `nice -n 19`, 1–2 workers.

## Checkpoints (stock-takes)

The review doc is where the analysis is developed; `docs/book/` (a Quarto book) is the record. A stock-take is: read the doc's tabs through the assistant, save the XML, run `scripts/fusion/doc_to_book.py`, `cd docs/book && quarto render`, commit, and tag `checkpoint-YYYY-MM-DD`. Render locally with `quarto preview docs/book`. Tags so far: `checkpoint-2026-10-21`.

## Reproduce from scratch

Inputs needed: blob credentials (`DSCI_AZ_BLOB_DEV_SAS`), the Earthdata token in `EARTHDATA_TOKEN` (MODIS and CYGNSS only), `~/.cdsapirc` (snow climatology only). Everything downloads into `data/fusion/` (gitignored, ~10 GB).

```
make test                                                      # 75 synthetic tests
make phase0                                                    # tile indexes, snow climatology
uv run python scripts/fusion/phase1_feasibility.py select      # v1 event selection (198 events, 6 continents)
uv run python scripts/fusion/phase1_feasibility.py extra       # the 51 in-scope events v1 left out (two lists)
# build event cubes: gold labels, GFDS day store, then sources, for the v1 selection and both extra lists
uv run python scripts/fusion/phase1_feasibility.py gold --workers 2 [--events-file data/fusion/gold/events_extra_small.parquet]
uv run python scripts/fusion/phase1_feasibility.py gfds        [--events-file ...]   # restart loop needed over a weak link
uv run python scripts/fusion/phase1_feasibility.py sources --workers 2 --continue-on-error [--events-file ...]
make phase2                                                    # global random sample cubes (prior)
make folds                                                     # results/event_folds.csv (frozen)
make crossval                                                  # results/cv/: the main cross-validated result
uv run python scripts/fusion/cv_breakdowns.py                  # without the largest basin; VIIRS period; VIIRS ablation
uv run python scripts/fusion/cv_normally_wet.py                # water vs flood split (needs JRC monthly masks, fetched on demand)
uv run python scripts/fusion/cv_microwave_components.py        # FloodScan / GFDS alone in the microwave slot
uv run python scripts/fusion/cv_baselines_and_views.py         # cascade baseline + fixed scoring views
uv run python scripts/fusion/cv_pattern_model.py --extended-grid --sweep   # learned combinations x memory x threshold
uv run python scripts/fusion/figure_tradeoff.py
uv run python scripts/fusion/figure_water_vs_flood.py --count && ... --draw FL20170815BGD 2017-08-22
uv run --with h5py --with fsspec --with aiohttp python scripts/fusion/experiment_cygnss.py
```

## Ledger

### 2026-10-12: v1 built end to end (phases 0–6)
- Question: can a naive-Bayes fusion of GFM, FloodScan/GFDS and VIIRS beat each single product against gold?
- How: `make phase0 … phase6` (`scripts/fusion/phase*.py`); 198 events on six continents; one split, 149 fit / 49 held out, six strata, every examined cell dry before and after events.
- Outputs: `results/phase5_*.csv`, `results/accuracy_table.csv`, `results/prior_v1.csv`, daily COGs on dev blob.
- Result: held-out CSI fusion 0.285 vs VIIRS 0.319; criterion (b) fails everywhere; reliability slope ~0.01.
- Commits: 07651c2 … fd0c8d6, 6b56bb6 (overstated recalibration claim corrected).

### 2026-10-13: why v1 was over-confident
- D-36 agreement table (`phase3_agreement.py`): false alarms of different sources coincide 4–18x more than independence predicts; the gold-free table does not identify flood. 61cf986, 3350ba1.
- D-39 dry-window labels (`diagnostic_dry_windows.py`, `experiment_dry_window_fix.py`): cells still flooded after the last map were labelled dry; this produced most of the over-confidence; corrected variant recall 0.37 → 0.44. 2133bd6, fc3cb9a.
- D-38 seasonal mask (`experiment_seasonal_baseline.py`): rejected; gold counts seasonal water as flood. aba5244, a6deb13.
- D-40 super-GFM (`experiment_super_gfm.py`): collapses to GFM persistence. 0dde253.
- D-37 MODIS (`experiment_modis_gold.py`, `modis_windows.py`): fills the VIIRS archive gap (2021-01..2023-07); full-window extraction is a Databricks job blocked on an Earthdata login the job can use. 3c4ee38, 382605e, d3dd4df.
- `docs/methods.md` written: pixels to scores. 747734a.

### 2026-10-14: the re-run design (ADRs 0003–0007)
- Scope: Africa, Asia, South America, Oceania; every usable event (170). `configs/fusion.yaml: scope`. 2aebe4c.
- Gold burn fix: burn polygon parts instead of unioning (the union took hours). 525104f. Fail-fast pools. eaa0f4f.
- Five folds by HydroBASINS level-4 basin, big-flood basins placed first: `events.assign_folds`, `make_folds.py`, `tests/test_folds.py`. 4042ef2.
- Rules fixed before any score (`docs/decisions/0003–0007`): three land types (snow split dropped); corrected dry-window rule adopted (`gold.dry_window_rule`); event-level rules (`event_level` in config: 10% + 5 cells, ±3 days, false-event days −30..−10). a0461a8. The unused `gold.dry_day_step` key removed (docs had said "every 3rd day"; code always used every day).
- Cross-validation code: `fusion/crossval.py`, `scripts/fusion/phase5_crossval.py` (`make crossval`), `tests/test_crossval.py`. c97b019, 802bd33. `docs/methods.md` §10. 66ae80b.
- Build of the 51 added events: 30 smaller ones 14 Oct; 21 large ones 14–15 Oct (laptop, restart loops; `data/fusion/resume/finish_build.sh`). No stage-C failures.

### 2026-10-15: the cross-validated result
- `make folds` → `results/event_folds.csv` (170 events, 99 basins). One basin (four UNOSAT Bangladesh floods) holds 40% of the gold flood; its fold holds 43% of open-land flood. ADR 0004 outcome.
- `make crossval` → `results/cv/skill_*.csv`, `criteria.csv`, `reliability_*`, `event_level_*`, `f1_by_*.png`. Fusion F1 0.44 (0.34–0.50) vs microwave vote 0.38, FloodScan 0.37, GFM 0.24, VIIRS 0.22; (b) passes on point estimate open/vegetated, fails urban; (c) fails. de2afda, 0292338, 361145d.
- `cv_breakdowns.py` → `skill_without_largest_basin.csv` (fusion 0.35 vs 0.34 vs 0.33), `skill_by_viirs_period.csv`, `skill_viirs_ablation.csv` (VIIRS adds 0.08 F1 where it has files; without VIIRS fusion = microwave vote). ba0b8e6.
- `cv_normally_wet.py` → `normally_wet_*.csv`. Normally-wet cells: 7% of labelled cells, 38% of gold flood; false-alarm rates 3–11x higher there; fusion's lead sits there; reliability flat within both groups (slopes 0.29/0.34), so the engine, not the target, is the calibration problem. 48a7c95, 100a54d, 3daa6c5.
- `figure_water_vs_flood.py --count/--draw` → `docs/figures/water_vs_flood_cells.png`, `data/fusion/metrics/water_vs_flood_cells.parquet`. Gold labels normally-wet cells 52% flood / 48% dry; 19% of fusion's flood calls on gold-dry cells are there; masking them drops F1 0.44 → 0.27. 374c07e, d6aab7f.

### 2026-10-16: the cascade bar and what separates fusion from it
- `cv_microwave_components.py` → `microwave_components_*.csv`. FloodScan alone costs 0.011 F1; GFDS alone 0.056; no microwave 0.106. beed811.
- `cv_baselines_and_views.py` (rules committed first, 3850538) → `baselines_views_*.csv`. Cascade "VIIRS, else GFM, else microwave": pooled 0.46 vs fusion 0.44 (ns); basin-weighted 0.40 vs 0.31; normally-wet excluded 0.46 vs 0.36; fusion false events 34% vs cascade 64%. c2cb793. **Headline changed: fusion as built does not beat the simplest combination of its inputs on cell-level scores.**
- `fusion/pattern.py` + `cv_pattern_model.py` (design committed first, f174daa) → `pattern_model_*.csv`. Evidence {independent, learned per combination} × memory {0.9, 0} × threshold {0.5, in fold}. No version passes the pre-registered test (pooled AND basin-weighted above the cascade AND fewer false events). Grid floor 0.02 failed its check → labelled follow-up to 0.001 (`*_grid_extended.csv`). 7911e32, e8ff1a1.
- `--sweep` → `pattern_model_sweep.csv`, `figure_tradeoff.py` → `docs/figures/tradeoff_f1_false_events.png`. With memory the threshold trades false events (25% → 63%) for F1 (0.41 → 0.49); without memory fusion lands on the cascade. Operating point is a use decision (doc Next steps 10). e77cb59.
- `experiment_cygnss.py` (rules committed first, 74c93ef) → `results/experiment_cygnss_*.csv`. UC Berkeley CYGNSS watermask is a rolling multi-week composite (6% of cells change per day, 24% over 30 days); on 8 events by rule, best single-product F1 0.47 (recall 0.67) but precision 0.24 on non-normally-wet cells. Not integrated. e77cb59.

### 2026-10-21: basics section in the review doc; this ledger
- Review doc (Claude Docs, https://claude.ai/artifact/EzpeTJ8mnRVPaw6rAE7Xop): "The basics" section added up front (cell vs event level; what is classified; 12% = 355,927/2,856,112 flooded share of labelled open-land cell-days on map days; 0.2% = 1,493/647,443 GFM flood share on ordinary days in the random sample).

### 2026-10-21: published
- GitHub Pages site restructured under the team's landing-page convention (KB `methods/static-data-apps.md`): landing page at <https://ocha-dap.github.io/ds-flood-gfm/>, the fusion book at `/fusion/`, the 2024 GFM exploration book moved from the root to `/gfm-exploration/` with forwarding stubs for its old URLs. `scripts/publish_pages.sh` (gh-pages worktree; only its own paths; refuses otherwise). KB: `surfaces:` on `pipelines/flood-fusion.md` and `flood-gfm.md`, in PR OCHA-DAP/ds-knowledge-base#757. Commits 4a69b75, f6c9936; gh-pages 4ac0c97.

### 2026-10-26: ordinary days (Next steps 6)
- `cv_ordinary_days.py` (rules committed first, 23f69de) → `results/cv/ordinary_days_{dry_windows,sample,sample_blocks,implied_precision}.csv` (d14d285). Dry-window false-alarm rates: fusion 3.0%, cascade 2.65%, GFM 0.03%. Random sample (30 blocks, GFM referee): fusion flags 1.06% of floodable cell-days, 1.7% agreed by GFM; implied precision ~0.03 (GFM 0.46). GFDS is the noisy half of the microwave vote on ordinary days (2.4% vs FloodScan 0.1%). Needed the remaining 194 window-months of JRC masks (`data/fusion/resume/prefetch_baseline.py`, all months).
- CYGNSS on all 148 events in coverage started (`experiment_cygnss.py --all`, resumable; 74c93ef + follow-up).

## Open decisions and next steps (as of 2026-10-21)
1. Operating point: catch floods (low threshold / no memory ≈ cascade) or avoid false alarms (fusion as built or higher threshold). User's call.
2. ~~Ordinary-day precision~~ done 26 Oct. New: drop GFDS from the microwave vote? (doc Next steps 12).
3. Normally-wet stratum (separate rates per land type × normally wet). Masks cached for map-day months; ~195 more window-months to fetch.
4. CYGNSS on all 148 events in coverage (~4.3 GB of windowed reads), then as a background layer (base rate) rather than daily evidence.
5. MODIS for the VIIRS gap: Databricks job blocked on an Earthdata login the job can use (user's consent for a personal token as a secret, or the team account fixed).
6. FloodScan licence outside Africa unconfirmed; GFDS feed ended 2026-09-14.
7. Water-first redesign (fuse water, subtract one normal-water baseline): a target-clarity change, not a calibration fix.

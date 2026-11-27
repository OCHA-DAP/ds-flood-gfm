# FloodScan reproduction: provenance of every number in the notes

Every table or figure quoted in `docs/floodscan-algorithm-notes.md` comes from a
script in this repo run on data that is backed up to the team's dev blob
(`projects/ds-flood-gfm/raw/floodscan_repro/`, `.../processed/floodscan_repro/`;
manifest with sha256 per file: `experiments/backup_rebuild_data.py` →
`data/backup_manifest.csv`). The scripts here were first written in a session
scratchpad; the scratchpad was wiped a month later and they were recovered from
the session transcript on 2026-11-27, given repo-relative paths, and rerun.

Run everything from the repo root. The chain scripts (`01`–`10`) use
`uv run --no-project --with numpy --with scipy --with rasterio python …` unless
noted; `06_score_2024.py` and `peak_check.py` need the project venv
(`uv run python …`) for `ocha_stratus`. FloodScan itself is read from the main
checkout's cache `data/fusion/raw/floodscan/africa_daily/` (AER's data, not
backed up by us).

| Notes section | Produces | Script(s) | Inputs | Committed output | Verified 2026-11-27 |
|---|---|---|---|---|---|
| Step 0 (Nigeria 2022) | core retrieval scores | `01_core_l3.py`, `core_excess_where.py` | `data/floodscan_repro/nga2022_core_l3.npz`, main checkout `data/gdacs_gfds/nga2022/` | `results/floodscan_repro/nga2022_core_l3_lga.csv` | not rerun |
| Step 1 tables and diagnostics | history tables, MDFF by years, f_hop gaps, Q_pow check | `04_static_tables.py`, `mdff_years.py`, `debug_fhop.py`, `peak_check.py` | `data/floodscan_repro/csu_amsr2_grid/{2023,2025}` | `data/floodscan_repro/static_tables.npz` (blob) | not rerun |
| Step 2 results (2024, tables 2023+2025) | variant table, LGA table | `05_ff_chain.py`, `06_score_2024.py` | `static_tables.npz`, `ff_2024.npz` | `results/floodscan_repro/nga2024_*.csv` | `regen_results.sh`: identical |
| Atmospheric correction | same, corrected Tb | `atmos.py`, `09_gee_merra2.py`, `04 … atmos`, `05 … {"atmos": true}` | `data/floodscan_repro/merra2/`, `static_tables_atmos.npz`, `ff_2024_atmos.npz` | `results/floodscan_repro/atmos/` | `regen_results.sh`: identical |
| Parameter sweep | knob table | `07_fit_params.py` | as Step 2 | `results/floodscan_repro/fit_sweep_atmos.csv` (the non-atmos sweep table is in the notes only) | not rerun |
| Gridding-noise test | σ7 grid vs footprints | `10_swath_noise_test.py`, `swath.py` | CSU orbit files (streamed), cache `data/floodscan_repro/swath_proto/` | printed; cache on blob | not rerun |
| Hold-out (2023, tables 2024+2025) | variant table | `holdout_2023.sh` (04 → 05 → 06) | `csu_amsr2_grid/{2024,2025}`, `static_tables_2425.npz`, `ff_2023.npz` | `results/floodscan_repro/holdout_2023/` | `regen_results.sh`: identical |
| Hold-out diagnosis | yearly Q levels; Sahel-band references | `q_level_by_year.py`, `sahel_tables_diag.py` | the three `csu_amsr2_grid` years, both table sets | printed (numbers in the notes) | rerun: identical to the notes |
| Expectation-weight test | w_exp table | `wexp_test.sh` | both table sets, `ff_{2023,2024}_wexp*.npz` | `results/floodscan_repro/wexp/` | `regen_results.sh`: identical |
| CSU bucket facts | files per sensor/year, variable layout, swath chunking | `bucket_census.py`, `csu_vars.py`, `csu_swath_inspect.py` | the NOAA CDR bucket (live) | memory/notes text | not rerun |

`regen_results.sh` reruns `06` on every flooded-fraction file and prints
`git diff --stat results/`; an empty diff means the committed tables are what the
code produces from the data on disk.

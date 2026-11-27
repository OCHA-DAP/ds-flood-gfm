# GFDS rebuild: provenance of every number in the evaluation

Every table quoted in `docs/gfds-rebuild-evaluation.md` (and
`docs/gdacs-gfds-assessment.md`, `docs/gfds-nga2022-findings.md`) comes from a
script in this repo run on data backed up to the team's dev blob
(`projects/ds-flood-gfm/raw/gfds/rebuild/`, `.../processed/gfds_rebuild/`;
manifest with sha256 per file: `experiments/backup_rebuild_data.py` →
`data/backup_manifest.csv`). The scripts in this folder were recovered from the
session transcript on 2026-11-27 after the scratchpad they lived in was wiped,
and given repo-relative paths. Run everything from the repo root.

JRC's archive at `www.gdacs.org/flooddetection/DATA` stopped updating on
2026-09-14 and its listings were already empty on 2026-10-26, so the copies
under `data/gdacs_gfds/rebuild/` (and on the blob) may be the only ones we will
ever have. G-Portal moved to `sftp.gportal.jaxa.jp:22` with key-only login on
2026-10-01; the probes below that name the old host document what was true
before that.

| Evaluation section | Produces | Script(s) | Inputs | Committed output | Status |
|---|---|---|---|---|---|
| §1 timeline, archive size | files/months, GB per folder | `gfds_archive_size.py`, `gdacs_check.py` | GDACS directory listings (live, now empty) | numbers in the doc | listings gone; not repeatable |
| §3 G-Portal inputs | paths, sizes, delays | `gportal_probe.py`, `gportal_probe2.py`, `gportal_l1b_list.py` | old host, password login | numbers in the doc | old host shut down 2026-11-04 |
| §3 sensors agree in level, coverage | AMSR2 vs GPM vs ALL | `gfds_coverage.py` | `data/gdacs_gfds/rebuild/{AMSR2,GPM,ALL}` samples | numbers in the doc | rerunnable |
| §7 gate 1a (grid route) | signal from JAXA L3 vs JRC | `02_fetch_amsr2_l3_tb36.py`, `04_signal_from_l3.py`, `05_compare_jrc.py`, `diff_where.py`, `temporal_4day.py` | `amsr2_l3_tb36/`, `jrc_amsr2_signal/`, `ALL/oceanmask.tif` → `signal_l3/` | printed tables in the doc | rerunnable from blob copies |
| Nigeria 2022 LGA comparison | r by source | `06_nigeria_lga_vs_floodscan.py` | `signal_l3/`, `jrc_amsr2_signal/`, FloodScan cache | `results/gfds_rebuild/nga2022_lga_r_by_source.csv` | rerunnable |
| Exactness test (2026-10-26) | T_M, C, s, m vs JRC answer keys | `07_signal_from_swath.py`, `08_gridding_schemes.py`, `09_window_convention.py`, `gdacs_key_2016.py` | CSU orbit files (streamed), `answer_key_2016/20180820/`, `AMSR2/AveragesAndSd/` → `swath_signal/` | printed tables in the doc | rerunnable from blob copies |
| Exactness test, JAXA calibration | same from JAXA L1B | `10_signal_from_jaxa_l1b.py`, `gportal_new_host_probe.py` | G-Portal L1B (new host, needs `GPORTAL_KEY`) | none yet | blocked on key registration |

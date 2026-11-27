# Operational helpers (rescued from the gitignored `data/fusion/resume/` on 2026-11-27)

One-off launchers and doc helpers used while building and validating v1 on the user's laptop. They are kept
as run, with their absolute paths, because `docs/log.md` cites them; logs go to `data/fusion/resume/logs/`.

| file | what it did | cited in |
|---|---|---|
| `finish_build.sh` | resumable restart loop that finished the 170-event build after a shutdown (gold, sources, GFDS days, cubes) | log.md 14–15 Oct |
| `prefetch_baseline.py` / `.sh` | fetched every JRC monthly normally-wet mask the windows need (D-38), smallest first, resumable | log.md 26 Oct |
| `s2_referee.sh` | 8-attempt restart loop around `scripts/fusion/experiment_s2_referee.py` | log.md 27 Oct |
| `cv_doc_tables.py` | markdown tables for the review doc straight from `results/cv/*.csv` (the seed of the data-driven book) | checkpoint 2026-10-21 |
| `doc_events_update.py` | the Events tab table from the event table + folds | checkpoint 2026-10-21 |
| `day_kind_counts.py` | counts of map days, dry-window days and sample days behind "Events, days and how counts add up" | doc section |

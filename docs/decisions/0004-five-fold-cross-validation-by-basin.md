---
status: accepted
date: 2026-10-14
---

# Five-fold cross-validation by river basin, big flood basins spread first

## Context and problem statement

v1 split events once: 149 for fitting, 49 for testing. In scope (ADR 0003) that leaves 29 test events, 3 of them in Oceania. Those 49 have also been used to compare variants (D-39, D-40, D-41). Detail: D-43 in `docs/decisions.md`.

## Considered options

1. **Keep the single split.** Against: thin per-continent results; the test set is no longer untouched.
2. **Five folds, events placed to balance continent counts.** Against: five events hold 45% of the gold flood; one fold got 34% of the open-land flood and another 6%.
3. **Five folds, whole HydroBASINS level-4 basins; basins with ≥ 2% of the gold flood placed first, each to the fold with the least flood, then the rest to balance continents** (chosen).
4. **Leave-one-basin-out.** Against: no gain in honesty over five folds, and many more fits to report.

## Decision outcome

Option 3 (user decision, 2026-10-14). Every event is scored once, by a model fitted on the other four folds. Code: `events.assign_folds`, `scripts/fusion/make_folds.py` (writes the frozen `results/event_folds.csv`), tests in `tests/test_folds.py`. Config: `gold.n_folds`, `gold.fold_seed`, `gold.fold_big_basin_share`.

### Consequences

- Fitting is counting, so five fits cost little.
- No event set stays untouched, so every rule used to score the re-run (ADRs 0005–0007) is fixed and committed before the first score is computed.
- `results/event_split.csv` stays as v1's record.

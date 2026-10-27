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

## Outcome (2026-10-15)

The rule balanced what it could; one basin cannot be split. Four UNOSAT floods in Bangladesh (FL20170815BGD, FL20180619BGD, FL20200713BGD, FL20220525BGD) share HydroBASINS level-4 basin 4040975160 and hold 40% of all gold flooded cell-days in the 170 events. Their fold holds 43% of the open-land flood and 62% of the vegetated flood; the other folds hold 8–17% each (`results/event_folds_balance.csv`). The 18–22% balance measured on the 119 events built before the large ones does not hold for the full set.

Consequences: that fold's rates are fitted on the remaining 57% (open) and 38% (vegetated) of the flood, and pooled scores lean on those four floods. The results are therefore also reported without that basin (`scripts/fusion/cv_breakdowns.py`, `results/cv/skill_without_largest_basin.csv`).

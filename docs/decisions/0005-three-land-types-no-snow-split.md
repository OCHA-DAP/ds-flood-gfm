---
status: accepted
date: 2026-10-14
---

# Three land types (open, vegetated, urban); the snow split is dropped

## Context and problem statement

v1 fitted each source's hit and false-alarm rates per six strata: three land types, each split by snow season (ERA5-Land monthly snow cover ≥ 50%). Of the 520 gold flooded cell-days in snow strata, 518 are in Europe, now out of scope (ADR 0003). The snow strata had too little gold and were filled by EM from a neighbouring stratum (D-24).

## Considered options

1. **Keep six strata.** Against: in scope, the snow strata have no gold to fit or test; their rows would be donor copies.
2. **Three land types, cubes rebuilt with three codes.** Against: rebuilding every cube costs hours and changes nothing measurable.
3. **Three land types at read time** (chosen). The cubes keep the 6-class code (`landcover × 2 + snow`); land type is `code // 2`. Rates are fitted per source and land type, and each row is written to both snow codes of its land type, so the engine (`engine.Model`, which indexes by the 6 codes) is unchanged.

## Decision outcome

Option 3 (user decision on the snow split, 2026-10-14; implementation chosen here). Code: `fusion/crossval.py`.

- **Prior** (base flood rate, D-35): the GFM flood share of GFM-observed cell-days in the global random sample, per land type, pooled over snow, from the 30 sample blocks in the four continents (of 53).
- **No EM fallback.** With three land types and about 136 fitting events per fold, every source should have observed gold flood in every land type. If one has none in some fold, the run stops and names it, rather than filling the row.

### Consequences

- Snow-season cells in scope (mountains in Asia, Patagonia) use their land type's rates.
- The snow layer stays in the cubes and in `configs/fusion.yaml` (`static.snow`) for the operational job, which still reads v1's six-row table.

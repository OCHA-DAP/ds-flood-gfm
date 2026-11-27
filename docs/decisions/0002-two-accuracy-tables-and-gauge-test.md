---
status: proposed
date: 2026-10-13
supersedes: fixed decision 5 of the v1 build brief (`docs/flood_fusion_plan.md` as of commit dd416a7)
---

# Two accuracy tables (agreement and gold) and a river-gauge test

## Context and problem statement

v1 learned each source's sensitivity and specificity from gold maps (EMS, UNOSAT) only. EM filled just the three snow strata, which had too little gold. Validation then found the product over-confident and below the best single product in every stratum (`docs/final_report.md`).

The gold sample has three limits:
- 149 training events, Europe-heavy;
- labels only inside event footprints;
- dry days only in the weeks around each flood.

There is also a direct sign of trouble. The base flood rate fitted on the random global sample, with gold accuracies held fixed, collapsed to its floor in all three no-snow strata (D-35). The gold accuracies do not explain how the sources agree on ordinary days.

The updated plan (2026-10-13) asks for two additions:
- a second, gold-free estimate of the accuracies, compared with gold;
- a test set independent of every satellite product.

## Considered options

1. **Gold only (v1).**
   - For: checked against analyst maps.
   - Against: small and biased, and measured in the weeks around floods, not on the ordinary days the daily product mostly covers.
2. **Agreement only** (Dawid–Skene EM on the global random sample, no gold).
   - For: large, global, ordinary days.
   - Against: with three sources and two classes the model is exactly identified (7 unknowns, 7 independent pattern frequencies). It can only be fitted by assuming the sources err independently, and agreement data cannot test that assumption.
   - Against: the hidden class may be "what all three call wet" (seasonal water, wetlands under each product's own baseline), not flood as gold defines it.
   - Against: VIIRS's cloud gaps are not random, because floods come with cloud.
3. **Both, compared per stratum** (chosen).
   - Where the tables agree, the larger-sample agreement table is used.
   - Where they diverge, gold is used, the stratum is flagged and the divergence explained.
   - Gold remains the only check on the agreement table's independence assumption.

## Decision outcome

Option 3, as in `docs/flood_fusion_plan.md`. The implementation rules below are this repo's reading of the plan. Each is configurable under `accuracy.agreement` in `configs/fusion.yaml`.

- **Agreement is judged on the engine's own scale:** the evidence weights.
  - A flood report adds w₁ = log(sens / (1 − spec)) to the log-odds.
  - A dry report adds w₀ = log((1 − sens) / spec).
  - The tables agree for a source when both weights differ by at most log 1.5. That means a report's effect on the odds differs by at most a factor of 1.5.
  - Sensitivity and specificity are not compared directly. A 0.01 difference in specificity near 0.99 doubles the false-alarm rate, while the same difference near 0.5 means little.
- **An agreement estimate is "uninformative"** in three cases, and gold is then used:
  - its 95% interval is wider than 0.15 (the v1 table's own flag);
  - the random starts reach different solutions;
  - the fit is degenerate (some source no better than chance).
- **Strata without adequate gold** keep their v1 rows. These are the three snow strata. Their status is `no_gold_to_check`, and the agreement value is shown beside the v1 row as information. An unchecked agreement estimate is exactly what this design exists to prevent.
- **Each divergence records its direction.** Gold's false-alarm rates come from the weeks around floods, while the sample covers random days. So a specificity divergence is partly expected from that difference in population. It is not, by itself, evidence that the sources share a blind spot.
- **River-gauge test:** the gauge source is not chosen yet. Observed-gauge candidates are listed in `docs/gauge_sources.md`; the choice is the user's.

### Consequences

- v1 outputs (`results/accuracy_table.csv`, the daily COGs) are unchanged.
- The agreement table, the comparison and the v2 table are separate files. The product switches to v2 only after a held-out comparison with v1.
- Gold cannot be retired. It is the only thing that can show the agreement table is wrong.

## Outcome (2026-10-13)

**The agreement table was built and does not identify flood. Gold shows why, and this also gives the measured cause of v1's over-confidence.** Detail: D-36 in `docs/decisions.md`; outputs `results/accuracy_table_agreement.csv`, `results/accuracy_table_comparison.csv`, `results/phase3_gold_error_dependence.csv`.

- **Even in the best case it cannot recover sensitivity.** On synthetic data with independent errors, at the sample's real rarity and observation rates, specificity is recovered but sensitivity is not (`results/experiment_agreement_identifiability.csv`).
- **On the real sample, its "flood" class is something else.**
  - In open land without snow it covers 2.4% of cell-days, 15 times the base flood rate. VIIRS sees 53% of it, microwave 19%, GFM 4%. That fits the products' different water baselines, not flood as gold defines it.
  - In vegetated and urban land the fit is unstable.
- **Comparison with gold:**
  - per row: 1 agree, 2 diverge, 6 unstable, 9 with no gold to check;
  - every stratum keeps gold, so the model table is unchanged.

  One rule was added during the build: a stratum uses the agreement values only if *all* its sources agree, because they come from one joint fit.
- **Why:** on gold-dry cells, the sources' false alarms coincide 4–18 times more often than independence predicts, and 2–9 times within the same event. The independence assumption fails, both for the agreement table and for the naive-Bayes engine.

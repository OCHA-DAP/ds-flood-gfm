---
status: proposed
date: 2026-10-12
supersedes: docs/superpowers/specs/2026-09-18-flood-fusion-design.md (sections 1, 4, 5 and the Africa PoC scope)
---

# Flood fusion v1: 0.01° flood-only naive-Bayes fusion replaces the 30″ water-target stacking spec

## Context and problem statement

The 2026-09-18 spec (PR #33) designed fusion as follows:
- a 30-arcsec grid;
- a *water* target, with flood as post-processing;
- weighted logistic stacking;
- an Africa proof of concept.

The v1 build brief (2026-10-12, `flood_fusion_plan.md`) fixes a different design:
- a global 0.01° grid;
- flood/dry/NoData per product;
- an OR-merged microwave vote;
- naive Bayes in log-odds with per-stratum sensitivity and specificity;
- a 0.9 carry-over.

One of the two had to govern the code.

## Considered options

1. **Keep the spec.** The 30″ grid nests in FloodScan and GHSL, the water target avoids comparing four products' different flood baselines, and the stacked weights are readable. Against it: it conflicts with every fixed decision in the brief, and it has no temporal model.
2. **Follow the brief and record what the spec got right.** The 0.01° grid nests 9×9 in GFDS. The flood target matches what every input and both gold corpora actually label. Naive Bayes per stratum is the whole model in one small table. The OR-merge of FloodScan and GFDS answers the spec's correct objection that the two share radiometers, so they are not independent.
3. **A hybrid**, such as the water target with naive Bayes. Rejected: it would need water-labelled gold, and CEMS gold v1 is flood-only.

## Decision outcome

Option 2.

The spec's still-valid points carry over as follows:

| spec point | where it lives now |
|---|---|
| Its baseline-divergence argument | Stated as a limitation in `docs/final_report.md` |
| Its circularity concern (GFM vs Sentinel-1-derived gold) | A required split in every GFM metric |
| Its fail-loud three states (absent / failed / real zero) | Implemented in every source adapter |
| Grouped validation by event | Extended to grouping by basin |

### Consequences

- FloodScan does not nest in 0.01°, so it is sampled at the containing cell (D-03).
- The OR-merge and per-stratum naive Bayes assume conditional independence across SAR, microwave and optical only. The Phase 5 ablation and the reliability slope are the checks on that assumption.
- The spec's open questions about gold v3 (binary vs fraction) and CEMS `sensor_class` are resolved for this repo:
  - Gold is burned at 0.001° and block-averaged into fractions, then binarised at 0.30.
  - `sensor_class` is derived from the CEMS `sensor` string.

  Neither change is made in ds-geospatial-impact-estimates.

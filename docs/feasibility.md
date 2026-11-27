# Phase 1 feasibility gate

**Gate rule (from the brief):** shrink phases 2–5 to a 5-basin prototype if
**oracle gain < 3 CSI points AND 3-day coverage gain < 15 percentage points**; otherwise continue in full.

**Data:** training events only (the held-out 25% stays unseen until Phase 5), gold days for skill, every window day for coverage.

## Written before the numbers (2026-10-12, before `phase1 metrics` ran)

These expectations were recorded before any Phase 1 numbers existed, so the result can be read against them rather than rationalised after the fact.

1. **The coverage half of the gate will pass by construction, and that says little.** The microwave vote is observed almost everywhere every day: FloodScan has no NoData inside its regions, and GFDS is daily outside its sentinels. "Any source within the last 3 days" will therefore sit near 100% of the floodable domain, against GFM's roughly 5–15% per day. A coverage gain far above 15 pp is guaranteed whether or not the microwave vote is any good, so passing the gate on coverage is **not** evidence that fusion is worth building.
2. **Microwave sensitivity at the 0.30 threshold should be close to zero.** A 0.01° cell takes the value of its ~9 km FloodScan or GFDS cell. Only floods covering ≥ 30% of that ~80 km² cell register, which means mega-floods. The likelihood ratio for a microwave "dry" will be near 1, so the microwave vote adds little evidence in either direction despite its coverage. Lowering the threshold was ruled out: the 0.30 rule is a fixed decision.
3. **The real test is the oracle (accuracy where each source looks).** The question is whether GFM, microwave and VIIRS each win in different strata. If GFM wins every stratum, fusion's value reduces to filling GFM's gaps with weak evidence.

## Result (148 training events with gold-day labels, 57,699 gold flood cells; `results/phase1_gate.json`)

| test | threshold | result | passes? |
|---|---|---|---|
| Oracle gain, CSI on each source's observed cells | ≥ 3 points | **+1.6** (VIIRS 0.367 → hindsight-best 0.384) | no |
| Oracle gain, CSI over all cells (NoData = not detected) | (reported) | +1.2 (microwave 0.151 → 0.163) | — |
| 3-day coverage gain over GFM alone | ≥ 15 pp | **+93.3 pp** (GFM 6.4% per day → any source within 3 days 99.7%) | yes |

**Gate decision: continue in full**, because the coverage criterion passed (the rule shrinks only when *both* criteria fail). As predicted above, it passed on coverage, which is almost entirely the microwave vote: microwave observes 99.1% of floodable cell-days.

### Per source, on gold days (training events)

| source | gold flood cells observed | hit rate | false-alarm ratio | false-alarm rate (incl. dry windows) | CSI (observed) | CSI (all cells) |
|---|---|---|---|---|---|---|
| GFM | 23,081 of 57,699 (40%) | 0.30 | 0.13 | 0.005 | 0.28 | 0.12 |
| microwave (FloodScan OR GFDS) | 57,690 (100%) | 0.26 | 0.73 | 0.044 | 0.15 | 0.15 |
| VIIRS | 14,844 (26%) | 0.66 | 0.55 | 0.032 | 0.37 | 0.14 |
| FloodScan alone | 56,787 | 0.16 | 0.66 | 0.020 | 0.12 | 0.12 |
| GFDS alone | 42,844 | 0.19 | 0.76 | 0.036 | 0.12 | 0.10 |

Per stratum (`results/phase1_skill_by_source_stratum.csv`):
- **Hindsight-best source:** GFM in vegetated/no-snow, VIIRS in open/no-snow and urban/no-snow.
- **Snow strata:** 83 (open), 81 (vegetated) and 0 (urban) gold flood cells. Too few to rank sources.
- **GFM circularity:** no advantage on Sentinel-1-derived gold maps (hit rate 0.29 on S1 gold vs 0.32 on other gold).

### What this means

1. **Accuracy complementarity is weak.** Choosing the best source per stratum with hindsight adds only 1.6 CSI points. No single source dominates by a large margin either: the best CSI on observed cells is 0.37.
2. **Coverage complementarity is large but low-grade.** The only source that observes almost everywhere every day is the microwave vote. At 1 km it has a high false-alarm ratio (0.73) and a modest hit rate (0.26). Item 2 above predicted a hit rate near zero. It came out higher because a few very large floods do fill 9-km cells. Eight training events give 52% of all microwave hits: EMSR479 (North America, 12%), the Australian EMSR184 and EMSR637, EMSR314 (Africa), a Philippine typhoon (TC20201111PHL), and floods in Nepal and Bangladesh. So the microwave vote's skill rests on mega-floods.
3. **So the full build goes ahead, on the gate as written.** Its value depends on whether naive Bayes can turn many low-grade microwave observations plus sparse SAR and optical observations into something better than the best single map. That is what Phase 5 tests. The 5-basin prototype path was not triggered.

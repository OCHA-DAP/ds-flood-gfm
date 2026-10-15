---
status: accepted
date: 2026-10-14
---

# Scope: Africa, Asia, South America and Oceania, with every usable event in them

## Context and problem statement

v1 selected 198 events on all six continents, with per-continent quotas and size caps to save compute (D-14). FloodScan does not cover Europe, so a pooled score mixes regions where fusion has different inputs. The user also wants to focus on the regions the team serves. Detail: D-42 in `docs/decisions.md`.

## Considered options

1. **Keep v1's selection, all continents.** Against: pooled scores mix regions with different sources; Europe-heavy gold dominates.
2. **Four continents, v1's quotas and caps.** Against: 119 events, thin per continent (Oceania 13, South America 16).
3. **Four continents, every event that passes the gold rules** (chosen). 170 events, with 51 more than option 2.

## Decision outcome

Option 3 (user decision, 2026-10-14). `scope.continents` in `configs/fusion.yaml` lists the continents. `phase1_feasibility.py extra` writes the two build lists of events v1 left out. Europe and North America are set aside, not discarded; their cubes are kept.

### Consequences

- Building the 51 events costs hours of downloads (GFDS re-reads every day for new rows), done once.
- Every result is reported per continent, because Asia holds most of the flooded area.

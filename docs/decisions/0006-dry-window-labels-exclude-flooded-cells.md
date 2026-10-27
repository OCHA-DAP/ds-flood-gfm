---
status: accepted
date: 2026-10-14
supersedes: fixed decision 5's literal dry-window rule (D-12), for the cross-validated re-run
---

# Days before and after an event: cells flooded on any map day get no label

## Context and problem statement

Gold maps exist only on flood days. To fit false-alarm rates, v1 labelled every examined cell of an event dry for the 30 days before its first map and the 60 days after its last (fixed decision 5, read literally, D-12). That includes cells flooded on the map days, so rising and draining floodwater is scored as a false alarm. In open land, fusion calls flood in those cells on 48% of days 1–30 after the event, against 5% in cells that never flooded (D-39).

## Considered options

1. **Literal rule (v1).** Against: inflates false-alarm rates (open-land VIIRS 5.5% against 2.6%), so flood reports get too little weight; most of v1's measured over-confidence came from it.
2. **Corrected rule** (chosen). In the days before and after, cells flooded on any map day of their window get no label; every other examined cell stays dry. On v1's held-out events: recall 0.37 → 0.44 at the same precision 0.55 (D-39).
3. **Drop the dry days entirely; fit false-alarm rates on map days only.** Against: map days have few dry cells in some events, and the dry days are the only labels away from the flood peak.

## Decision outcome

Option 2. `gold.dry_window_rule: corrected` in `configs/fusion.yaml`; applied at read time by `gold.corrected_dry_windows`, so the cubes are unchanged.

The user had not signed this off and asked on 2026-10-14 for the work to be finished without further questions. It was decided under that delegation, before any cross-validated score was computed. It is reversible: set `gold.dry_window_rule: literal`. The re-run also reports the literal rule as one sensitivity row.

### Consequences

- Hit rates do not change (flood labels exist only on map days); false-alarm rates fall.
- Recall, precision and F1 are scored on map days only, so the rule changes them only through the fitted rates.
- Reliability is shown on map days alone and on map days plus dry days under this rule.
- What the rule cannot fix: unmapped water inside the examined area is still labelled dry.

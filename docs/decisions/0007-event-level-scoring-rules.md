---
status: accepted
date: 2026-10-14
---

# Event-level scoring: was the event caught, how early, and false events

## Context and problem statement

Cell-level scores say how well a product maps a flood. Many uses only need to know whether it noticed the flood, and how early. The first run counted a window "detected" on the first day at least 10% of its gold footprint was flagged, on any day of the window. Microwave then "caught" floods a median 24 days before the first map: persistent false alarms or seasonal water, not detection.

## Considered options

1. **v1's rule** (any day, no false-event check). Against: a product that always says flood catches everything.
2. **Rule with a time limit and a false-event check** (chosen), as proposed in the review doc.
3. **Per-window instead of per-event.** Against: large events with many windows would count many times; the question is about the event.

## Decision outcome

Option 2. Config: `event_level` in `configs/fusion.yaml`; code: `crossval.event_level`.

- **Footprint.** The floodable-domain cells flooded on any map day in any of the event's windows. Every window covers the same days, so per-day counts add across windows. Events with fewer than 5 footprint cells are left out of event-level results, and counted.
- **Flagged.** A product flags a cell on a day if it says flood (no data = not flagged); fusion flags it at P ≥ 0.5.
- **Caught.** On some day from 3 days before the first map to 3 days after the last, the product flags at least 10% of the footprint, and at least 5 cells.
- **Days to detection.** The first day the caught rule holds, counted from the first map (−3 at the earliest).
- **False event.** The same rule on the same cells on ordinary days: days 30 to 10 before the first map. The 10-day buffer allows for flooding that began before mapping.
- **Per land type.** The same rules with the footprint restricted to one land type (events with at least 5 such cells).
- **Reported** per product and fusion, overall and per continent: share caught, share with a false event, median days to detection, with intervals from resampling events.
- **Derived column** (added 2026-10-14, before the run): "caught, no false event", the share of events caught and quiet on days 30 to 10 before. It separates detection from a product flagging the same cells for weeks.

Decided under the user's delegation of 2026-10-14, before any cross-validated score was computed.

**Deferred:** "false events away from floods" on the global random sample. A sample block has no gold footprint, so the rule above has no cells to apply to; defining one now, under time pressure, risks a rule tuned after the fact. It belongs with the ordinary-day precision work.

### Consequences

- Every event counts equally here, unlike the cell-level scores, where large floods dominate.
- The false-event share is measured on the event's own footprint, so it says nothing about false events in places that never flood.

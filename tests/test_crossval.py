"""Cross-validated re-run pieces (fusion/crossval.py; ADRs 0005-0007).

Why these cases:
- land type is read from v1's 6-class stratum code; a wrong mapping would silently mix land types;
- the dry-window rule must remove formerly flooded cells from the dry days (and only those), and
  cells outside the floodable domain must never be counted;
- a fold's rates must come from its training events only; a leak would make every score optimistic;
- each fitted row must reach both snow codes, because the engine indexes by the six codes;
- a land type with nothing to fit must stop the run, not get a made-up rate;
- the event-level rules have edges (the +-3-day limit, the 10% share, the 5-cell floor, the
  false-event period) that decide a yes or no per event, so each edge is tested on both sides.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ds_flood_gfm.fusion import crossval


def _cube():
    """5 days, map on day 3; cells: flooded on the map day, dry on the map day, outside the domain."""
    days = pd.date_range("2024-01-01", periods=5)
    truth = np.zeros((5, 1, 3), "uint8")
    truth[2] = [[1, 0, 1]]
    gfm = np.zeros((5, 1, 3), "uint8")
    gfm[:, 0, 0] = 1
    gfm[:, 0, 2] = 1
    nod = np.full((5, 1, 3), 255, "uint8")
    dims = ("day", "lat", "lon")
    return xr.Dataset(
        {"truth": (dims, truth), "gfm": (dims, gfm), "microwave": (dims, nod), "viirs": (dims, nod),
         "stratum": (dims, np.zeros((5, 1, 3), "uint8")), "domain": (("lat", "lon"), np.array([[1, 1, 0]], "uint8"))},
        coords={"day": days}, attrs={"event_start": "2024-01-03", "event_end": "2024-01-03"})


def test_landtype_codes():
    assert crossval.landtype_codes(np.array([0, 1, 2, 3, 4, 5, 255])).tolist() == [0, 0, 1, 1, 2, 2, 255]


def test_dry_rule_drops_formerly_flooded_cells_and_ignores_non_domain():
    ds = _cube()
    gfm = lambda c: c[(c.source == "gfm") & (c.landtype == "open")].set_index("day_type")  # noqa: E731
    lit = gfm(crossval.fit_counts(ds, crossval.dry_truth(ds, "literal"), "E1"))
    cor = gfm(crossval.fit_counts(ds, crossval.dry_truth(ds, "corrected"), "E1"))
    # map day: the flooded cell is a hit, the dry cell a correct dry; the non-domain cell is not counted
    assert lit.loc["map", ["tp", "fn", "fp", "tn"]].tolist() == [1, 0, 0, 1]
    # dry days: the literal rule scores the formerly flooded cell (still flagged) as 4 false alarms
    assert lit.loc["dry", ["fp", "tn"]].tolist() == [4, 4]
    assert cor.loc["dry", ["fp", "tn"]].tolist() == [0, 4]
    assert cor.loc["map"].equals(lit.loc["map"])


def test_unknown_dry_rule_raises():
    with pytest.raises(ValueError):
        crossval.dry_truth(_cube(), "lenient")


def _counts(code, tp, fn, fp, tn):
    rows = []
    for s in ["gfm", "microwave", "viirs"]:
        for lt in ["open", "vegetated", "urban"]:
            rows.append({"code": code, "day_type": "map", "landtype": lt, "source": s, "tp": tp, "fn": fn, "fp": fp, "tn": tn})
    return pd.DataFrame(rows)


def test_fit_table_uses_training_events_only_and_fills_both_snow_codes():
    c = pd.concat([_counts("TRAIN", 30, 70, 10, 990), _counts("HELD_OUT", 1000, 0, 1000, 0)])
    t = crossval.fit_table(c, ["TRAIN"])
    assert sorted(t.stratum_code.unique()) == [0, 1, 2, 3, 4, 5]
    g = t[t.source == "gfm"].set_index("stratum_code")
    assert g.loc[0, "sens"] == pytest.approx(30.5 / 101) and g.loc[0, "spec"] == pytest.approx(990.5 / 1001)
    assert g.loc[1, ["sens", "spec"]].tolist() == g.loc[0, ["sens", "spec"]].tolist()  # snow code = no-snow code
    assert (t.n_flood_obs == 100).all()  # the held-out event's 1000 flood cells never enter


def test_fit_table_raises_when_a_landtype_has_no_flood():
    c = _counts("TRAIN", 30, 70, 10, 990)
    c.loc[(c.source == "viirs") & (c.landtype == "urban"), ["tp", "fn"]] = 0
    with pytest.raises(ValueError, match="viirs/urban"):
        crossval.fit_table(c, ["TRAIN"])


EV = dict(min_share=0.10, min_cells=5, days_around=3, false_days=(-30, -10))
FIRST, LAST = pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-05")
DAYS = pd.date_range(FIRST - pd.Timedelta(days=30), LAST + pd.Timedelta(days=60))


def _flagged(**by_offset):
    """Flagged-cell series over the window; keys like d_m3 = 3 days before the first map, d_p8 = 8 after."""
    s = pd.Series(0, index=DAYS)
    for k, v in by_offset.items():
        off = int(k[3:]) * (-1 if k[2] == "m" else 1)
        s[FIRST + pd.Timedelta(days=off)] = v
    return s


def test_caught_within_three_days_of_the_maps_only():
    # last map is day +4, so the window closes at +7
    assert crossval.event_level(_flagged(d_p7=5), 40, FIRST, LAST, **EV)["caught"] is True
    assert crossval.event_level(_flagged(d_p8=5), 40, FIRST, LAST, **EV)["caught"] is False
    r = crossval.event_level(_flagged(d_m3=5, d_p2=40), 40, FIRST, LAST, **EV)
    assert r["caught"] is True and r["days_to_detection"] == -3
    assert crossval.event_level(_flagged(d_m4=40), 40, FIRST, LAST, **EV)["caught"] is False


def test_caught_needs_ten_percent_and_five_cells():
    assert crossval.event_level(_flagged(d_p0=4), 40, FIRST, LAST, **EV)["caught"] is False  # 10% of 40 is 4, floor is 5
    assert crossval.event_level(_flagged(d_p0=19), 200, FIRST, LAST, **EV)["caught"] is False  # need 20
    assert crossval.event_level(_flagged(d_p0=20), 200, FIRST, LAST, **EV)["caught"] is True


def test_false_event_only_on_days_30_to_10_before():
    assert crossval.event_level(_flagged(d_m10=5), 40, FIRST, LAST, **EV)["false_event"] is True
    assert crossval.event_level(_flagged(d_m30=5), 40, FIRST, LAST, **EV)["false_event"] is True
    assert crossval.event_level(_flagged(d_m9=5), 40, FIRST, LAST, **EV)["false_event"] is False
    assert crossval.event_level(_flagged(d_p20=5), 40, FIRST, LAST, **EV)["false_event"] is False


def test_small_footprint_not_eligible():
    r = crossval.event_level(_flagged(d_p0=4), 4, FIRST, LAST, **EV)
    assert r["eligible"] is False and r["caught"] is None

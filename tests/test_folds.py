"""Cross-validation folds (events.assign_folds, D-43).

Why these cases:
- a basin's events must never be split across folds (otherwise a neighbouring flood on the same river
  helps score its neighbour);
- the few basins that dominate the flooded area must land in different folds (otherwise one fold's model
  is fitted without most of the flood data);
- among the rest, each continent's events must be spread evenly;
- the assignment must be reproducible from the seed;
- an event without a basin must be rejected, not silently dropped.
"""

import numpy as np
import pandas as pd
import pytest

from ds_flood_gfm.fusion.events import assign_folds


def _events(basins, continents, floods):
    return pd.DataFrame({"code": [f"E{i}" for i in range(len(basins))], "basin_id": basins, "continent": continents, "flood": floods})


def test_basins_stay_whole():
    rng = np.random.default_rng(1)
    ev = _events(rng.integers(0, 30, 80), rng.choice(["Africa", "Asia", "Oceania"], 80), rng.integers(0, 500, 80))
    fold = assign_folds(ev, 5, seed=7, big_share=0.02)
    assert fold.between(0, 4).all() and len(fold) == len(ev)
    assert (ev.assign(fold=fold).groupby("basin_id").fold.nunique() == 1).all()


def test_big_basins_spread_one_per_fold():
    big = _events([100, 101, 102, 103, 104], ["Asia"] * 5, [15_000] * 5)
    small = _events(list(range(40)), ["Africa"] * 40, [50] * 40)
    ev = pd.concat([big, small.assign(code=lambda d: "S" + d.code)], ignore_index=True)
    fold = assign_folds(ev, 5, seed=7, big_share=0.02)
    assert sorted(fold[:5]) == [0, 1, 2, 3, 4]


def test_continents_balanced_among_small_basins():
    conts = ["Africa"] * 23 + ["Asia"] * 17 + ["Oceania"] * 10
    ev = _events(list(range(50)), conts, [10] * 50)
    fold = assign_folds(ev, 5, seed=7, big_share=0.5)  # no basin is "big" here
    counts = pd.crosstab(ev.continent, fold)
    assert (counts.max(axis=1) - counts.min(axis=1) <= 1).all()


def test_same_seed_same_folds():
    rng = np.random.default_rng(2)
    ev = _events(rng.integers(0, 25, 60), rng.choice(["Africa", "Asia"], 60), rng.integers(0, 900, 60))
    assert assign_folds(ev, 5, 11, 0.02).equals(assign_folds(ev, 5, 11, 0.02))


def test_missing_basin_raises():
    ev = _events([1, np.nan], ["Africa", "Asia"], [5, 5])
    with pytest.raises(ValueError):
        assign_folds(ev, 5, 7, 0.02)

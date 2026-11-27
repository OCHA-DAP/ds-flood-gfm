"""Learned evidence per combination of reports (fusion/pattern.py).

Why these cases:
- the pattern code must be exact, or every learned ratio lands on the wrong combination;
- with sources that really are independent, the learned evidence must equal the engine's summed weights
  (the pattern model nests fusion as built, so any difference in results comes from dependence);
- "couldn't see" must carry no evidence even when one class is observed less often (cloud on flood days);
- heavy shrinkage must fall back to the independence values;
- a fold's table must come from its training events only;
- an empty land type must stop the fit, not get a made-up table;
- the cascade "VIIRS, else GFM, else microwave" must be representable: with labels that follow it, every
  combination the cascade calls flood gets positive evidence and every other negative evidence.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from ds_flood_gfm.fusion import engine, pattern

SRC = ["gfm", "microwave", "viirs"]


def test_pattern_ids():
    obs = {"gfm": np.array([1, 255, 0]), "microwave": np.array([0, 255, 1]), "viirs": np.array([255, 255, 1])}
    assert pattern.pattern_ids(obs).tolist() == [9 * 1 + 3 * 0 + 2, 26, 9 * 0 + 3 * 1 + 1]
    with pytest.raises(ValueError):
        pattern.pattern_ids({"gfm": np.array([3]), "microwave": np.array([0]), "viirs": np.array([0])})


def _independent(rate, seen, n=1e9):
    """27 pattern counts for one class: each source observes with prob seen[j], reports flood with rate[j]."""
    out = np.zeros(27)
    for p, codes in enumerate(itertools.product(range(3), repeat=3)):
        q = 1.0
        for j, c in enumerate(codes):
            q *= (1 - seen[j]) if c == 2 else seen[j] * (rate[j] if c == 1 else 1 - rate[j])
        out[p] = n * q
    return out


def test_independent_sources_reproduce_the_engine():
    sens, fa = np.array([0.3, 0.4, 0.6]), np.array([0.005, 0.04, 0.03])
    # flood days are cloudier: VIIRS observes less often on flood days than on dry days
    nf, nd = _independent(sens, [0.3, 0.99, 0.2]), _independent(fa, [0.3, 0.99, 0.5])
    lr = pattern.log_lr_one(nf, nd, alpha=100)
    acc = pd.DataFrame([{"source": s, "stratum_code": k, "sens": sens[j], "spec": 1 - fa[j]} for j, s in enumerate(SRC) for k in range(6)])
    prior = pd.DataFrame({"stratum_code": range(6), "base_rate": 0.01})
    model = engine.Model.from_tables(acc, prior, SRC, 0.9, clip=(1e-6, 1 - 1e-6))
    for p, codes in enumerate(itertools.product(range(3), repeat=3)):
        obs = {s: np.array([[255 if c == 2 else c]], "uint8") for s, c in zip(SRC, codes)}
        expected = engine.evidence(obs, np.zeros((1, 1), "uint8"), model)[0, 0]
        assert lr[p] == pytest.approx(expected, abs=1e-3), codes  # incl. missingness adding nothing


def test_heavy_shrinkage_falls_back_to_independence():
    rng = np.random.default_rng(3)
    nf, nd = rng.integers(0, 500, 27).astype(float), rng.integers(0, 50000, 27).astype(float)
    lr_big = pattern.log_lr_one(nf, nd, alpha=1e12)
    rf, rd = pattern._source_rates(nf), pattern._source_rates(nd)
    for p, codes in enumerate(itertools.product(range(3), repeat=3)):
        exp = sum(np.log(rf[j] / rd[j]) if c == 1 else np.log((1 - rf[j]) / (1 - rd[j])) for j, c in enumerate(codes) if c != 2)
        assert lr_big[p] == pytest.approx(exp, abs=1e-6)
    assert lr_big[pattern.ALL_MISSING] == 0


def _counts(code, nf, nd):
    return pd.DataFrame([{"code": code, "landtype": lt, "pattern": p, "n_flood": nf[p], "n_dry": nd[p]}
                         for lt in ["open", "vegetated", "urban"] for p in range(27)])


def test_fit_uses_training_events_only_and_fills_both_snow_codes():
    rng = np.random.default_rng(4)
    nf, nd = rng.integers(1, 500, 27), rng.integers(1, 5000, 27)
    c = pd.concat([_counts("TRAIN", nf, nd), _counts("HELD_OUT", nd * 100, nf)])
    t = pattern.fit_log_lr(c, ["TRAIN"], alpha=100)
    assert np.allclose(t[0], pattern.log_lr_one(nf.astype(float), nd.astype(float), 100))
    assert np.array_equal(t[0], t[1]) and np.array_equal(t[4], t[5])


def test_empty_landtype_raises():
    c = _counts("TRAIN", np.ones(27, int), np.ones(27, int))
    c.loc[c.landtype == "urban", "n_flood"] = 0
    with pytest.raises(ValueError, match="urban"):
        pattern.fit_log_lr(c, ["TRAIN"], alpha=100)


def test_cascade_is_representable():
    nf, nd = np.zeros(27), np.zeros(27)
    for p, (g, m, v) in enumerate(itertools.product(range(3), repeat=3)):
        first = next((c for c in (v, g, m) if c != 2), None)  # VIIRS, else GFM, else microwave
        if first == 1:
            nf[p] = 1000
        elif first == 0:
            nd[p] = 1000
    lr = pattern.log_lr_one(nf, nd, alpha=1)
    for p, (g, m, v) in enumerate(itertools.product(range(3), repeat=3)):
        first = next((c for c in (v, g, m) if c != 2), None)
        if first is not None:
            assert (lr[p] > 0) == (first == 1), (g, m, v)

"""Tests for Dawid-Skene EM and the prevalence MLE on synthetic data with planted parameters.

Why these cases:
- Unsupervised 3-source EM must recover planted sens/spec/prevalence. This is the
  identifiable case the fixed design assumes (three independent sensor families).
- Fully labelled data must reproduce plain confusion-matrix counting exactly.
  EM for thin strata is then a strict generalisation of the gold table.
- NoData must be uninformative. Adding all-NoData cells must not move the estimates.
- The prevalence MLE (prior v1) must recover the planted base rate with sens/spec fixed.
- Pattern counting must round-trip.
"""

import numpy as np

from ds_flood_gfm.fusion import accuracy

SENS = np.array([0.85, 0.4, 0.7])
SPEC = np.array([0.97, 0.9, 0.94])
P_OBS = np.array([0.5, 0.95, 0.4])


def simulate(rng, n, prev):
    z = (rng.random(n) < prev).astype(int)
    obs = np.empty((n, 3), "int64")
    for j in range(3):
        says = np.where(z == 1, rng.random(n) < SENS[j], rng.random(n) > SPEC[j]).astype(int)
        says[rng.random(n) >= P_OBS[j]] = 255
        obs[:, j] = says
    return z, obs


def to_patterns(obs, z=None):
    d = {f"s{j}": obs[:, j] for j in range(3)}
    mask = np.ones(len(obs), bool)
    return accuracy.pattern_counts(d, ["s0", "s1", "s2"], mask, z)


def test_unsupervised_em_recovers_planted_parameters():
    rng = np.random.default_rng(1)
    _, obs = simulate(rng, 400_000, 0.15)
    pc = to_patterns(obs)
    res = accuracy.dawid_skene(pc[["s0", "s1", "s2"]].values, pc.n.values, np.zeros((0, 3), "int64"), np.zeros(0), np.zeros(0),
                               init_sens=np.array([0.7, 0.5, 0.6]), init_spec=np.array([0.9, 0.85, 0.9]), init_prev=0.1)
    assert res.converged
    np.testing.assert_allclose(res.sens, SENS, atol=0.02)
    np.testing.assert_allclose(res.spec, SPEC, atol=0.01)
    assert abs(res.prevalence - 0.15) < 0.01


def test_fully_labelled_equals_counting():
    rng = np.random.default_rng(2)
    z, obs = simulate(rng, 50_000, 0.3)
    pc = to_patterns(obs, z)
    res = accuracy.dawid_skene(np.zeros((0, 3), "int64"), np.zeros(0), pc[["s0", "s1", "s2"]].values, pc.n.values, pc.truth.values,
                               init_sens=np.array([0.5] * 3), init_spec=np.array([0.5] * 3), init_prev=0.5, prior_strength=0.0)
    for j in range(3):
        o = obs[:, j]
        sens = ((o == 1) & (z == 1)).sum() / ((o != 255) & (z == 1)).sum()
        spec = ((o == 0) & (z == 0)).sum() / ((o != 255) & (z == 0)).sum()
        assert abs(res.sens[j] - sens) < 1e-9 and abs(res.spec[j] - spec) < 1e-9


def test_all_nodata_cells_do_not_move_estimates():
    rng = np.random.default_rng(3)
    _, obs = simulate(rng, 100_000, 0.2)
    pc = to_patterns(obs)
    kw = dict(init_sens=np.array([0.7, 0.5, 0.6]), init_spec=np.array([0.9, 0.85, 0.9]), init_prev=0.1)
    a = accuracy.dawid_skene(pc[["s0", "s1", "s2"]].values, pc.n.values, np.zeros((0, 3), "int64"), np.zeros(0), np.zeros(0), **kw)
    extra = np.vstack([pc[["s0", "s1", "s2"]].values, [[255, 255, 255]]])
    n_extra = np.concatenate([pc.n.values, [10_000_000]])
    b = accuracy.dawid_skene(extra, n_extra, np.zeros((0, 3), "int64"), np.zeros(0), np.zeros(0), **kw)
    np.testing.assert_allclose(a.sens, b.sens, atol=1e-6)
    np.testing.assert_allclose(a.spec, b.spec, atol=1e-6)
    # prevalence is defined over all cells, all-NoData cells sit at the prior mean; sens/spec must not move


def test_prevalence_mle_recovers_base_rate():
    rng = np.random.default_rng(4)
    _, obs = simulate(rng, 300_000, 0.04)
    pc = to_patterns(obs)
    p = accuracy.prevalence_mle(pc[["s0", "s1", "s2"]].values, pc.n.values, SENS, SPEC)
    assert abs(p - 0.04) < 0.005


def test_pattern_counts_round_trip():
    obs = {"a": np.array([0, 1, 255, 1]), "b": np.array([1, 1, 255, 1])}
    pc = accuracy.pattern_counts(obs, ["a", "b"], np.ones(4, bool))
    rows = {(r.a, r.b): r.n for r in pc.itertuples()}
    assert rows == {(0, 1): 1, (1, 1): 2, (255, 255): 1}

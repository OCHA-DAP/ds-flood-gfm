"""Tests for the log-odds fusion engine on synthetic data with planted accuracies.

What these prove, and why each matters:
- NoData adds exactly zero evidence: an unobserved cell must not drift (fixed decision 1/6).
- With planted sens/spec and a known truth, thresholding the fused P at 0.5 recovers
  truth at least as well as the best single source, and the predicted probabilities
  are calibrated when the model's parameters are the true ones.
- The carry-over recursion is exactly prior + decay*(L_prev - prior) + evidence.
- days_since_obs resets on any observation and counts up otherwise.
- Bad inputs (values outside {0,1,255}, missing strata) raise instead of passing silently.
"""

import numpy as np
import pandas as pd
import pytest

from ds_flood_gfm.fusion import engine

SOURCES = ["gfm", "microwave", "viirs"]


def make_model(sens, spec, base_rate, decay=0.9):
    acc = pd.DataFrame(
        [{"source": s, "stratum_code": k, "sens": sens[s], "spec": spec[s]} for s in SOURCES for k in range(6)]
    )
    prior = pd.DataFrame({"stratum_code": range(6), "base_rate": [base_rate] * 6})
    return engine.Model.from_tables(acc, prior, SOURCES, decay)


def simulate(rng, truth, sens, spec, p_obs):
    obs = {}
    for s in SOURCES:
        says_flood = np.where(truth == 1, rng.random(truth.shape) < sens[s], rng.random(truth.shape) > spec[s])
        o = says_flood.astype("uint8")
        o[rng.random(truth.shape) >= p_obs[s]] = 255
        obs[s] = o
    return obs


def test_nodata_adds_zero_evidence():
    m = make_model({s: 0.8 for s in SOURCES}, {s: 0.9 for s in SOURCES}, 0.05)
    shape = (4, 5)
    obs = {s: np.full(shape, 255, "uint8") for s in SOURCES}
    strat = np.zeros(shape, "uint8")
    assert np.all(engine.evidence(obs, strat, m) == 0.0)
    # with no observations the state stays exactly at the prior day after day
    dom = np.ones(shape, bool)
    out = engine.run_sequence({s: np.stack([obs[s]] * 10) for s in SOURCES}, np.stack([strat] * 10), dom, m)
    np.testing.assert_allclose(out["p_flood"], 0.05, rtol=1e-6)
    assert (out["n_sources"] == 0).all()
    assert (out["days_since_obs"] == engine.NEVER).all()


def test_single_observation_evidence_values():
    m = make_model({"gfm": 0.7, "microwave": 0.4, "viirs": 0.6}, {"gfm": 0.95, "microwave": 0.8, "viirs": 0.9}, 0.1)
    strat = np.zeros((1, 3), "uint8")
    obs = {"gfm": np.array([[1, 0, 255]], "uint8"),
           "microwave": np.full((1, 3), 255, "uint8"), "viirs": np.full((1, 3), 255, "uint8")}
    ev = engine.evidence(obs, strat, m)
    np.testing.assert_allclose(ev[0], [np.log(0.7 / 0.05), np.log(0.3 / 0.95), 0.0])


def test_recovers_truth_with_planted_accuracies():
    rng = np.random.default_rng(0)
    sens = {"gfm": 0.85, "microwave": 0.35, "viirs": 0.7}
    spec = {"gfm": 0.97, "microwave": 0.9, "viirs": 0.95}
    p_obs = {"gfm": 0.6, "microwave": 0.95, "viirs": 0.5}
    base = 0.2
    shape = (400, 400)
    truth = (rng.random(shape) < base).astype("uint8")
    obs = simulate(rng, truth, sens, spec, p_obs)
    m = make_model(sens, spec, base, decay=0.0)  # single day: no carry-over
    r = engine.run_day(obs, np.zeros(shape, "uint8"), np.ones(shape, bool), m, None, None)
    pred = r.p_flood >= 0.5

    def csi(p):
        tp = (p & (truth == 1)).sum()
        return tp / (tp + (p & (truth == 0)).sum() + (~p & (truth == 1)).sum())

    fused = csi(pred)
    singles = {s: csi(obs[s] == 1) for s in SOURCES}  # NoData counted as no detection
    assert fused >= max(singles.values()), (fused, singles)
    # calibration: with the true parameters, mean truth in each probability bin matches the bin
    p = r.p_flood.ravel()
    t = truth.ravel()
    for lo, hi in [(0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0)]:
        sel = (p >= lo) & (p < hi)
        if sel.sum() > 500:
            assert abs(t[sel].mean() - p[sel].mean()) < 0.03, (lo, hi, t[sel].mean(), p[sel].mean())


def test_carry_over_recursion_exact():
    m = make_model({s: 0.8 for s in SOURCES}, {s: 0.9 for s in SOURCES}, 0.05, decay=0.9)
    strat = np.zeros((1, 1), "uint8")
    prior = engine.logit(0.05)
    flood = {"gfm": np.ones((1, 1), "uint8"), "microwave": np.full((1, 1), 255, "uint8"), "viirs": np.full((1, 1), 255, "uint8")}
    none = {s: np.full((1, 1), 255, "uint8") for s in SOURCES}
    w = np.log(0.8 / 0.1)
    L1 = engine.step(None, strat, engine.evidence(flood, strat, m), m)
    np.testing.assert_allclose(L1, prior + w)
    L2 = engine.step(L1, strat, engine.evidence(none, strat, m), m)
    np.testing.assert_allclose(L2, prior + 0.9 * w)
    L3 = engine.step(L2, strat, engine.evidence(none, strat, m), m)
    np.testing.assert_allclose(L3, prior + 0.81 * w)


def test_days_since_obs():
    d = engine.days_since(None, np.array([True, False]))
    assert d.tolist() == [0, engine.NEVER]
    d = engine.days_since(d, np.array([False, False]))
    assert d.tolist() == [1, engine.NEVER]
    d = engine.days_since(d, np.array([False, True]))
    assert d.tolist() == [2, 0]


def test_outside_domain_is_nan():
    m = make_model({s: 0.8 for s in SOURCES}, {s: 0.9 for s in SOURCES}, 0.05)
    obs = {s: np.ones((2, 2), "uint8") for s in SOURCES}
    dom = np.array([[True, False], [True, True]])
    strat = np.array([[0, 0], [255, 3]], "uint8")
    r = engine.run_day(obs, strat, dom, m, None, None)
    assert np.isnan(r.p_flood[0, 1]) and np.isnan(r.p_flood[1, 0])
    assert np.isfinite(r.p_flood[0, 0]) and np.isfinite(r.p_flood[1, 1])
    assert r.n_sources[0, 1] == 255 and r.n_sources[1, 1] == 3


def test_invalid_values_raise():
    m = make_model({s: 0.8 for s in SOURCES}, {s: 0.9 for s in SOURCES}, 0.05)
    obs = {s: np.zeros((1, 2), "uint8") for s in SOURCES}
    obs["gfm"][0, 1] = 7
    with pytest.raises(ValueError, match="outside"):
        engine.evidence(obs, np.zeros((1, 2), "uint8"), m)


def test_missing_stratum_raises():
    acc = pd.DataFrame([{"source": s, "stratum_code": k, "sens": 0.8, "spec": 0.9} for s in SOURCES for k in range(5)])
    prior = pd.DataFrame({"stratum_code": range(6), "base_rate": [0.05] * 6})
    with pytest.raises(ValueError, match="lacks strata"):
        engine.Model.from_tables(acc, prior, SOURCES, 0.9)


def test_nan_sensitivity_rejected():
    # a stratum with no gold flood cells gives sens = 0/0; it must never become a NaN weight
    acc = pd.DataFrame([{"source": s, "stratum_code": k, "sens": 0.8, "spec": 0.9} for s in SOURCES for k in range(6)])
    acc.loc[(acc.source == "viirs") & (acc.stratum_code == 3), "sens"] = np.nan
    prior = pd.DataFrame({"stratum_code": range(6), "base_rate": [0.05] * 6})
    with pytest.raises(ValueError, match="non-finite"):
        engine.Model.from_tables(acc, prior, SOURCES, 0.9)

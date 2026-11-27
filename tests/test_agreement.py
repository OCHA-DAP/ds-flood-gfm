"""Tests for the agreement table, its comparison with gold, and the error-dependence check (ADR 0002).

Why these cases:
- The log-likelihood picks the winner among EM starts, so it must equal the textbook sum.
- Urban flood rates in the random sample (~4e-5) sit below the old 1e-4 prevalence floor. The new
  `prev_clip` must let EM reach them, and the old floor must visibly stop it.
- Flood must always be the minority class. Swapping the classes must not change the likelihood
  (the two labellings are the same model), or the start comparison would be biased.
- A sample with no class structure (every source reports independently of the truth) must come
  out `unstable`, never as a confident table.
- The comparison's status rules decide which table the model would use. Only a stratum where every
  source is `agree` may select the agreement values. One source matching by coincidence, inside a
  joint fit that diverges for the others, must not switch the stratum.
- The dependence check must give lift 1 for independent errors and 4 for planted shared false
  alarms. Its within-event lift must be 1 when the pooled lift comes only from one event being
  wetter than another for every source.
"""

import numpy as np
import pandas as pd

from ds_flood_gfm.fusion import accuracy, agreement

PATS = accuracy.all_patterns(3)
A = {"max_iter": 200_000, "tol": 1e-12, "prev_clip_low": 1e-8, "start_ll_tolerance": 1.0, "param_tolerance": 0.02, "boundary": 0.001}


def expected_counts(sens, spec, prev, p_obs, n_total):
    """Exact expected pattern counts under the two-class model (no sampling noise)."""
    out = np.empty(len(PATS))
    for k, pt in enumerate(PATS):
        a, b = prev, 1 - prev
        for j, o in enumerate(pt):
            if o == 255:
                a *= 1 - p_obs[j]; b *= 1 - p_obs[j]
            elif o == 1:
                a *= p_obs[j] * sens[j]; b *= p_obs[j] * (1 - spec[j])
            else:
                a *= p_obs[j] * (1 - sens[j]); b *= p_obs[j] * spec[j]
        out[k] = a + b
    return out * n_total


def test_observed_loglik_matches_brute_force():
    sens, spec, prev = np.array([0.6, 0.3, 0.8]), np.array([0.98, 0.9, 0.95]), 0.07
    n = np.arange(1, len(PATS) + 1, dtype=float)
    brute = 0.0
    for pt, c in zip(PATS, n):
        p1, p0 = prev, 1 - prev
        for j, o in enumerate(pt):
            if o == 1:
                p1 *= sens[j]; p0 *= 1 - spec[j]
            elif o == 0:
                p1 *= 1 - sens[j]; p0 *= spec[j]
        brute += c * np.log(p1 + p0)
    assert abs(accuracy.observed_loglik(PATS, n, sens, spec, prev) - brute) < 1e-6


def test_prev_clip_lets_a_rare_class_through():
    sens, spec, prev = np.array([0.9, 0.85, 0.8]), np.array([0.999, 0.998, 0.999]), 5e-5
    n = expected_counts(sens, spec, prev, np.ones(3), 50_000_000)
    kw = dict(init_sens=np.array([0.7] * 3), init_spec=np.array([0.99] * 3), init_prev=1e-3, max_iter=200_000, tol=1e-12)
    empty = (np.zeros((0, 3), "int64"), np.zeros(0), np.zeros(0))
    low = accuracy.dawid_skene(PATS, n, *empty, prev_clip=(1e-8, 1 - 1e-8), **kw)
    old = accuracy.dawid_skene(PATS, n, *empty, **kw)
    assert abs(low.prevalence - prev) / prev < 0.05
    assert old.prevalence == 1e-4  # held at the old floor, twice the planted rate


def test_orient_swaps_classes_without_changing_likelihood():
    r = accuracy.EMResult(np.array([0.1, 0.2, 0.3]), np.array([0.6, 0.7, 0.8]), 0.9, 10, True)
    o = agreement.orient(r)
    assert abs(o.prevalence - 0.1) < 1e-12
    np.testing.assert_allclose(o.sens, 1 - r.spec)
    np.testing.assert_allclose(o.spec, 1 - r.sens)
    n = np.arange(1, len(PATS) + 1, dtype=float)
    assert abs(accuracy.observed_loglik(PATS, n, r.sens, r.spec, r.prevalence)
               - accuracy.observed_loglik(PATS, n, o.sens, o.spec, o.prevalence)) < 1e-6


def test_clean_sample_is_stable_and_structureless_sample_is_unstable():
    starts = [([0.5] * 3, [0.95] * 3, 0.01), ([0.8] * 3, [0.99] * 3, 0.001), ([0.3] * 3, [0.9] * 3, 0.2)]
    clean = expected_counts(np.array([0.85, 0.4, 0.7]), np.array([0.97, 0.9, 0.94]), 0.15, np.array([0.5, 0.95, 0.4]), 400_000)
    f = agreement.fit(PATS, clean, starts, A)
    assert not f["unstable"]
    np.testing.assert_allclose(f["result"].sens, [0.85, 0.4, 0.7], atol=1e-3)
    # every source says "flood" at a fixed rate whatever the truth: nothing for EM to find
    rates = np.array([0.05, 0.1, 0.2])
    flat = expected_counts(rates, 1 - rates, 0.1, np.array([0.5, 0.95, 0.4]), 400_000)
    assert agreement.fit(PATS, flat, starts, A)["unstable"]


def _gold(rows):
    """rows: (code, gfm, microwave, n) for dry gold cells; VIIRS unobserved."""
    return pd.DataFrame([{"stratum": "open_nosnow", "truth": 0, "code": c, "gfm": a, "microwave": b, "viirs": 255, "n": n}
                         for c, a, b, n in rows])


def _lift(df):
    d = agreement.error_dependence(df)
    return d[d.pair == "gfm+microwave"].iloc[0]


def test_error_dependence_lift_is_one_when_independent_and_four_when_shared():
    # false-alarm rates 0.1 and 0.2 over 10,000 dry cells
    indep = _gold([("E1", 1, 1, 200), ("E1", 1, 0, 800), ("E1", 0, 1, 1800), ("E1", 0, 0, 7200)])
    shared = _gold([("E1", 1, 1, 800), ("E1", 1, 0, 200), ("E1", 0, 1, 1200), ("E1", 0, 0, 7800)])
    assert abs(_lift(indep).lift - 1.0) < 1e-9
    assert abs(_lift(shared).lift - 4.0) < 1e-9


def test_within_event_lift_removes_between_event_differences():
    # each event independent inside, but E2 is wetter for both sources (0.3 vs 0.02)
    rows = [("E1", 1, 1, 4), ("E1", 1, 0, 196), ("E1", 0, 1, 196), ("E1", 0, 0, 9604),
            ("E2", 1, 1, 900), ("E2", 1, 0, 2100), ("E2", 0, 1, 2100), ("E2", 0, 0, 4900)]
    r = _lift(_gold(rows))
    assert r.lift > 1.5
    assert abs(r.lift_within_event - 1.0) < 1e-9


def test_compare_status_rules():
    gold = pd.DataFrame([
        {"source": "gfm", "stratum": "s_agree", "method": "gold", "sens": 0.3, "spec": 0.99},
        {"source": "gfm", "stratum": "s_diverge", "method": "gold", "sens": 0.3, "spec": 0.99},
        {"source": "gfm", "stratum": "s_nogold", "method": "em(init=s_agree)", "sens": 0.3, "spec": 0.99},
        {"source": "gfm", "stratum": "s_unstable", "method": "gold", "sens": 0.3, "spec": 0.99},
    ]).assign(sens_lo=np.nan, sens_hi=np.nan, spec_lo=np.nan, spec_hi=np.nan)
    agr = pd.DataFrame([
        {"source": "gfm", "stratum": "s_agree", "sens": 0.31, "spec": 0.99, "unstable": False},
        {"source": "gfm", "stratum": "s_diverge", "sens": 0.05, "spec": 0.9996, "unstable": False},
        {"source": "gfm", "stratum": "s_nogold", "sens": 0.3, "spec": 0.99, "unstable": False},
        {"source": "gfm", "stratum": "s_unstable", "sens": 0.3, "spec": 0.99, "unstable": True},
    ]).assign(prevalence=0.01, flag_rate=0.01)
    c = agreement.compare(agr, gold, np.log(1.5)).set_index("stratum")
    assert c.loc["s_agree", "status"] == "agree" and c.loc["s_agree", "table_used"] == "agreement"
    assert c.loc["s_diverge", "status"] == "diverge" and c.loc["s_diverge", "table_used"] == "gold"
    assert c.loc["s_nogold", "status"] == "no_gold_to_check" and c.loc["s_nogold", "table_used"] == "gold"
    assert c.loc["s_unstable", "status"] == "agreement_unstable" and c.loc["s_unstable", "table_used"] == "gold"


def test_one_agreeing_source_does_not_switch_a_diverging_stratum():
    # one joint fit per stratum: microwave matches gold by coincidence while gfm diverges
    gold = pd.DataFrame([
        {"source": "gfm", "stratum": "s", "method": "gold", "sens": 0.3, "spec": 0.99},
        {"source": "microwave", "stratum": "s", "method": "gold", "sens": 0.26, "spec": 0.95},
    ]).assign(sens_lo=np.nan, sens_hi=np.nan, spec_lo=np.nan, spec_hi=np.nan)
    agr = pd.DataFrame([
        {"source": "gfm", "stratum": "s", "sens": 0.04, "spec": 0.9996, "unstable": False},
        {"source": "microwave", "stratum": "s", "sens": 0.25, "spec": 0.95, "unstable": False},
    ]).assign(prevalence=0.02, flag_rate=0.03)
    c = agreement.compare(agr, gold, np.log(1.5)).set_index("source")
    assert c.loc["microwave", "status"] == "agree" and c.loc["gfm", "status"] == "diverge"
    assert (c.table_used == "gold").all()

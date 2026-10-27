"""Validation statistics on synthetic data.

Why these cases: the reliability slope is one of the three "better than any single
product" criteria (it must lie in 0.8-1.2), so the estimator must return about 1 for
calibrated probabilities, below 1 for over-confident ones, and "not estimable"
(NaN) rather than a number when predictions have no spread.
"""

import numpy as np

from ds_flood_gfm.fusion import validate


def _bins(p, y, nb=10):
    edges = np.linspace(0, 1, nb + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, nb - 1)
    n = np.bincount(b, minlength=nb)
    k = np.bincount(b, weights=y, minlength=nb)
    mp = np.bincount(b, weights=p, minlength=nb) / np.maximum(n, 1)
    ok = n > 0
    return mp[ok], k[ok], n[ok]


def test_calibrated_slope_is_one():
    rng = np.random.default_rng(0)
    p = rng.beta(0.3, 5, 200_000)
    y = rng.random(p.size) < p
    _, b = validate.logistic_slope(*_bins(p, y))
    assert 0.9 < b < 1.1


def test_overconfident_slope_below_one():
    rng = np.random.default_rng(1)
    true_p = rng.beta(1, 3, 200_000)
    y = rng.random(true_p.size) < true_p
    logit = np.log(true_p) - np.log1p(-true_p)
    over = 1 / (1 + np.exp(-2.5 * logit))  # stretched log-odds = over-confident
    _, b = validate.logistic_slope(*_bins(over, y))
    assert b < 0.6


def test_no_spread_is_nan():
    a, b = validate.logistic_slope(np.array([0.2]), np.array([3.0]), np.array([10.0]))
    assert np.isnan(a) and np.isnan(b)

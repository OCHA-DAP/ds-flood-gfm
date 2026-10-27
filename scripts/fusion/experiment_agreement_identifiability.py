"""Experiment (ADR 0002): can the agreement table (unsupervised Dawid-Skene on the random sample)
recover sensitivity and specificity at the sample's real flood rarity and observation rates?

This is the best case for the method. For each no-snow stratum it plants:
  - the gold table's sens/spec (results/accuracy_table.csv);
  - each source's real observation rate in the sample (data/fusion/metrics/sample_patterns.parquet);
  - the stratum's base flood rate (results/prior_v1.csv);
  - errors independent given the truth, and "couldn't see" independent of the truth.
It then draws the stratum's real number of cell-days as 27-pattern counts and fits
unsupervised EM from three starts (best log-likelihood kept, flood = minority class).
Repeated `--reps` times; the spread of the recovered values is the method's precision.

Output: results/experiment_agreement_identifiability.csv (planted, median, 5th/95th
percentiles of the recovered values, per stratum x source x parameter).

Run: nice -n 19 uv run python scripts/fusion/experiment_agreement_identifiability.py [--reps 8]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ds_flood_gfm.fusion import accuracy
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES

PATS = accuracy.all_patterns(3)


def pattern_probs(sens: np.ndarray, spec: np.ndarray, prev: float, p_obs: np.ndarray) -> np.ndarray:
    """Probability of each of the 27 patterns under the two-class model with independent errors."""
    out = np.empty(len(PATS))
    for k, pt in enumerate(PATS):
        a, b = prev, 1 - prev
        for j, o in enumerate(pt):
            if o == accuracy.NODATA:
                a *= 1 - p_obs[j]
                b *= 1 - p_obs[j]
            elif o == 1:
                a *= p_obs[j] * sens[j]
                b *= p_obs[j] * (1 - spec[j])
            else:
                a *= p_obs[j] * (1 - sens[j])
                b *= p_obs[j] * spec[j]
        out[k] = a + b
    return out / out.sum()


def fit_unsupervised(n: np.ndarray, starts: list[tuple]) -> accuracy.EMResult:
    best = None
    for s0, p0, q0 in starts:
        r = accuracy.dawid_skene(PATS, n, np.zeros((0, 3), "int64"), np.zeros(0), np.zeros(0), np.asarray(s0, float),
                                 np.asarray(p0, float), q0, max_iter=200_000, tol=1e-10, prev_clip=(1e-8, 1 - 1e-8))
        if r.prevalence > 0.5:  # flood is the minority class
            r = accuracy.EMResult(1 - r.spec, 1 - r.sens, 1 - r.prevalence, r.iterations, r.converged)
        ll = accuracy.observed_loglik(PATS, n, r.sens, r.spec, r.prevalence)
        if best is None or ll > best[0]:
            best = (ll, r)
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config)
    samp = pd.read_parquet(cfg.local("metrics", "sample_patterns.parquet"))
    acc = pd.read_csv(cfg.results_dir / "accuracy_table.csv")
    prior = pd.read_csv(cfg.results_dir / "prior_v1.csv").set_index("stratum")
    rng = np.random.default_rng(0)
    rows = []
    for st in ["open_nosnow", "vegetated_nosnow", "urban_nosnow"]:
        g = acc[acc.stratum == st].set_index("source").loc[MODEL_SOURCES]
        sens, spec, prev = g.sens.values, g.spec.values, float(prior.loc[st, "base_rate"])
        sp = samp[samp.stratum == st]
        n_total = int(sp.n.sum())
        p_obs = np.array([sp[sp[s] != accuracy.NODATA].n.sum() / n_total for s in MODEL_SOURCES])
        p = pattern_probs(sens, spec, prev, p_obs)
        starts = [([0.5] * 3, [0.95] * 3, 0.01), ([0.8] * 3, [0.99] * 3, 0.001), (list(sens), list(spec), prev)]
        reps = [fit_unsupervised(rng.multinomial(n_total, p).astype(float), starts) for _ in range(a.reps)]
        for j, s in enumerate(MODEL_SOURCES):
            for par, planted, vals in [("sens", sens[j], [r.sens[j] for r in reps]), ("spec", spec[j], [r.spec[j] for r in reps])]:
                rows.append({"stratum": st, "source": s, "parameter": par, "planted": planted, "median": np.median(vals),
                             "p05": np.quantile(vals, 0.05), "p95": np.quantile(vals, 0.95), "reps": a.reps,
                             "n_cell_days": n_total, "p_observed": p_obs[j], "planted_base_rate": prev})
        q = [r.prevalence for r in reps]
        rows.append({"stratum": st, "source": "all", "parameter": "base_rate", "planted": prev, "median": np.median(q),
                     "p05": np.quantile(q, 0.05), "p95": np.quantile(q, 0.95), "reps": a.reps, "n_cell_days": n_total,
                     "planted_base_rate": prev})
        print(f"{st} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(cfg.results_dir / "experiment_agreement_identifiability.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

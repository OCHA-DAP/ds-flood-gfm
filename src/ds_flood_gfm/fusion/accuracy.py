"""Accuracy-table estimation: Dawid-Skene EM and the per-stratum prevalence (prior v1).

Observations are reduced to *pattern counts*: with three sources each in
{0, 1, NoData} there are 27 joint patterns per stratum, so EM runs on a
27-row table instead of millions of cell-days.

Dawid-Skene (fixed decision 5): latent truth Z in {0, 1}; sources independent
given Z (independence is assumed only across SAR, microwave, optical);
P(source says 1 | Z=1) = sens, P(says 0 | Z=0) = spec, NoData uninformative.
Labelled gold cells enter with Z fixed (semi-supervised). Estimates are MAP
under Beta priors centred on the initial values with `prior_strength`
pseudo-counts, which keeps EM away from degenerate solutions when a stratum
has few joint observations (D-24).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

NODATA = 255
PATTERN_VALUES = (0, 1, NODATA)


def all_patterns(n_sources: int) -> np.ndarray:
    return np.array(list(itertools.product(PATTERN_VALUES, repeat=n_sources)), dtype="int64")


def pattern_counts(obs: dict[str, np.ndarray], sources: list[str], mask: np.ndarray, truth: np.ndarray | None = None) -> pd.DataFrame:
    """Count joint observation patterns over cells where `mask` is True.

    Returns columns [*sources, (truth), n]. Cells where every source is NoData are kept
    (they carry no evidence but count towards sample sizes).
    """
    cols = [obs[s][mask].astype("int64") for s in sources]
    if truth is not None:
        cols.append(truth[mask].astype("int64"))
    if not cols[0].size:
        return pd.DataFrame(columns=[*sources, *(["truth"] if truth is not None else []), "n"])
    code = np.zeros(cols[0].shape, "int64")
    for c in cols:
        code = code * 256 + c
    uniq, n = np.unique(code, return_counts=True)
    names = [*sources, *(["truth"] if truth is not None else [])]
    out = {}
    rem = uniq.copy()
    for name in reversed(names):
        out[name] = rem % 256
        rem //= 256
    df = pd.DataFrame(out)[names]
    df["n"] = n
    return df


def _loglik_terms(pat: np.ndarray, sens: np.ndarray, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log P(pattern | Z=1), log P(pattern | Z=0) for each pattern row."""
    l1 = np.zeros(len(pat))
    l0 = np.zeros(len(pat))
    for j in range(pat.shape[1]):
        o = pat[:, j]
        l1 += np.where(o == 1, np.log(sens[j]), np.where(o == 0, np.log1p(-sens[j]), 0.0))
        l0 += np.where(o == 1, np.log1p(-spec[j]), np.where(o == 0, np.log(spec[j]), 0.0))
    return l1, l0


@dataclass
class EMResult:
    sens: np.ndarray
    spec: np.ndarray
    prevalence: float
    iterations: int
    converged: bool


def dawid_skene(unlab: np.ndarray, n_unlab: np.ndarray, lab: np.ndarray, n_lab: np.ndarray, z_lab: np.ndarray,
                init_sens: np.ndarray, init_spec: np.ndarray, init_prev: float, prior_strength: float = 0.0,
                max_iter: int = 20000, tol: float = 1e-7, clip: tuple[float, float] = (1e-4, 1 - 1e-4),
                prev_clip: tuple[float, float] | None = None) -> EMResult:
    """Semi-supervised two-class Dawid-Skene EM on pattern counts.

    unlab: (P, S) patterns with counts n_unlab; lab: (Q, S) patterns with counts n_lab and known z_lab.
    Beta(a, b) priors centred on the initial values: a = strength*init, b = strength*(1-init).
    prev_clip bounds the prevalence separately from sens/spec (default: `clip`). The agreement
    table needs a lower bound than 1e-4: urban flood rates in the random sample are ~4e-5 (ADR 0002).
    """
    prev_clip = clip if prev_clip is None else prev_clip
    # all-NoData patterns carry no information about any parameter (their likelihood is 1
    # for both classes); keeping them only slows the prevalence update to a crawl
    keep = (unlab != NODATA).any(axis=1) if len(unlab) else np.zeros(0, bool)
    unlab, n_unlab = unlab[keep], n_unlab[keep]
    sens, spec, prev = init_sens.astype(float).copy(), init_spec.astype(float).copy(), float(init_prev)
    a_se, b_se = prior_strength * init_sens, prior_strength * (1 - init_sens)
    a_sp, b_sp = prior_strength * init_spec, prior_strength * (1 - init_spec)
    S = init_sens.size
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # E-step on unlabelled patterns
        if len(unlab):
            l1, l0 = _loglik_terms(unlab, sens, spec)
            a = np.log(prev) + l1
            b = np.log1p(-prev) + l0
            q = 1.0 / (1.0 + np.exp(b - a))
        else:
            q = np.zeros(0)
        # M-step: pool labelled (q = z) and unlabelled (q) weights
        pats = np.vstack([unlab, lab]) if len(lab) else unlab
        w = np.concatenate([n_unlab, n_lab]).astype(float) if len(lab) else n_unlab.astype(float)
        qq = np.concatenate([q, z_lab.astype(float)]) if len(lab) else q
        w1, w0 = w * qq, w * (1 - qq)
        new_sens = np.empty(S)
        new_spec = np.empty(S)
        for j in range(S):
            o = pats[:, j]
            new_sens[j] = (w1[o == 1].sum() + a_se[j]) / (w1[o != NODATA].sum() + a_se[j] + b_se[j])
            new_spec[j] = (w0[o == 0].sum() + a_sp[j]) / (w0[o != NODATA].sum() + a_sp[j] + b_sp[j])
        new_prev = (w1.sum()) / w.sum()
        new_sens = np.clip(new_sens, *clip)
        new_spec = np.clip(new_spec, *clip)
        new_prev = float(np.clip(new_prev, *prev_clip))
        delta = max(np.abs(new_sens - sens).max(), np.abs(new_spec - spec).max(), abs(new_prev - prev))
        sens, spec, prev = new_sens, new_spec, new_prev
        if delta < tol:
            converged = True
            break
    return EMResult(sens, spec, prev, it, converged)


def observed_loglik(pat: np.ndarray, n: np.ndarray, sens: np.ndarray, spec: np.ndarray, prev: float) -> float:
    """Observed-data log-likelihood of pattern counts under the two-class model.

    Used to choose between EM runs from different starts (the agreement table, ADR 0002).
    All-NoData patterns contribute log(1) = 0.
    """
    l1, l0 = _loglik_terms(pat, sens, spec)
    return float(np.sum(np.asarray(n, float) * np.logaddexp(np.log(prev) + l1, np.log1p(-prev) + l0)))


def prevalence_mle(patterns: np.ndarray, n: np.ndarray, sens: np.ndarray, spec: np.ndarray,
                   max_iter: int = 1000, tol: float = 1e-10, init: float = 0.05) -> float:
    """MLE of P(Z=1) from unlabelled pattern counts with sens/spec held fixed (prior v1)."""
    informative = (patterns != NODATA).any(axis=1)
    pat, cnt = patterns[informative], n[informative].astype(float)
    if cnt.sum() == 0:
        raise ValueError("no informative observations to estimate prevalence from")
    l1, l0 = _loglik_terms(pat, sens, spec)
    prev = init
    for _ in range(max_iter):
        a = np.log(prev) + l1
        b = np.log1p(-prev) + l0
        q = 1.0 / (1.0 + np.exp(b - a))
        new = float((cnt * q).sum() / cnt.sum())
        new = min(max(new, 1e-7), 1 - 1e-7)
        if abs(new - prev) < tol:
            return new
        prev = new
    return prev

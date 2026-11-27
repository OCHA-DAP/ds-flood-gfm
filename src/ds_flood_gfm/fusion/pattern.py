"""Learned evidence per combination of reports (Next steps 9, 2026-10-16).

Fusion as built adds each source's evidence as if the sources were independent (naive Bayes). Their errors
are not independent (D-36). Here the evidence for a day is learned for each combination of the reports
that exist, per land type, from the training folds:

  - A pattern is the three sources' values on one cell-day (dry, flood, couldn't see): 27 patterns.
    pattern_ids encodes it as 9 * gfm + 3 * microwave + viirs, with 0 dry, 1 flood, 2 couldn't see.
  - Missingness carries no evidence. For each set S of sources that observed, the likelihood ratio is
    learned over the combinations of their values: LR = P(values | flood, S observed) / P(values | dry, S
    observed). Nothing observed: log LR = 0, as in the engine. (On event days, cloud and flood come
    together; learning "couldn't see" as evidence would not carry over to ordinary days.)
  - Shrinkage: each class's distribution within S is pulled towards the independence product of the
    per-source rates, with `alpha` pseudo-counts: (n + alpha * q) / (N_S + alpha). With independent
    sources this gives exactly the engine's summed weights; a combination seen rarely leans on them.
  - The day-to-day recursion, prior and threshold are the engine's (engine.step), unchanged.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import crossval, engine, gold
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.static import LANDCOVER_NAMES as LANDTYPES

N_PATTERNS = 27
ALL_MISSING = 26  # (2, 2, 2)
# per pattern: each source's code (0 dry, 1 flood, 2 couldn't see), in MODEL_SOURCES order
CODES = np.array(list(itertools.product(range(3), repeat=3)))


def pattern_ids(obs: dict[str, np.ndarray], sources=MODEL_SOURCES) -> np.ndarray:
    """0..26 per cell from the sources' 0 / 1 / 255 values (9 * first + 3 * second + third)."""
    out = np.zeros(obs[sources[0]].shape, "int16")
    for s in sources:
        v = obs[s]
        bad = ~np.isin(v, (0, 1, NODATA))
        if bad.any():
            raise ValueError(f"{s}: values outside {{0, 1, 255}}: {np.unique(v[bad])[:5]}")
        out = out * 3 + np.where(v == NODATA, 2, v).astype("int16")
    return out


def pattern_counts(ds: xr.Dataset, truth: np.ndarray, code: str) -> pd.DataFrame:
    """Labelled floodable cell-days per (land type, pattern, class): flood on map days, dry on map days and
    dry days, the same population as crossval.fit_counts."""
    dom = ds.domain.values.astype(bool)[None]
    lt = crossval.landtype_codes(ds.stratum.values)
    ids = pattern_ids({s: ds[s].values for s in MODEL_SOURCES})
    lab = ((truth == gold.DRY) | (truth == gold.FLOOD)) & dom & (lt != NODATA)
    rows = []
    for k, name in LANDTYPES.items():
        m = lab & (lt == k)
        nf = np.bincount(ids[m & (truth == gold.FLOOD)], minlength=N_PATTERNS)
        nd = np.bincount(ids[m & (truth == gold.DRY)], minlength=N_PATTERNS)
        rows += [{"code": code, "landtype": name, "pattern": p, "n_flood": int(nf[p]), "n_dry": int(nd[p])} for p in range(N_PATTERNS)]
    return pd.DataFrame(rows)


def _source_rates(n: np.ndarray) -> np.ndarray:
    """Per source: P(flood report | observed) for one class, Jeffreys-smoothed, from 27 pattern counts."""
    out = np.zeros(len(MODEL_SOURCES))
    for j in range(len(MODEL_SOURCES)):
        k1 = n[CODES[:, j] == 1].sum()
        kobs = n[CODES[:, j] != 2].sum()
        out[j] = (k1 + 0.5) / (kobs + 1.0)
    return out


def log_lr_one(nf: np.ndarray, nd: np.ndarray, alpha: float) -> np.ndarray:
    """27 log likelihood ratios for one land type from its flood and dry pattern counts."""
    rf, rd = _source_rates(nf), _source_rates(nd)
    out = np.zeros(N_PATTERNS)
    observed = CODES != 2
    for mask in {tuple(r) for r in observed}:
        if not any(mask):
            continue  # nothing observed: no evidence
        group = np.where((observed == np.array(mask)).all(axis=1))[0]
        for c, n, r in ((0, nf, rf), (1, nd, rd)):
            q = np.ones(len(group))
            for j in np.where(mask)[0]:
                fl = CODES[group, j] == 1
                q *= np.where(fl, r[j], 1 - r[j])
            p = (n[group] + alpha * q) / (n[group].sum() + alpha)
            out[group] += np.log(p) if c == 0 else -np.log(p)
    out[ALL_MISSING] = 0.0
    return out


def fit_log_lr(counts: pd.DataFrame, train_codes, alpha: float) -> np.ndarray:
    """(6 stratum codes, 27) log likelihood ratios from the training events; each land type's row is written
    to both snow codes, as in crossval.fit_table. A land type with no flood or no dry cell-days raises."""
    c = counts[counts.code.isin(set(train_codes))].groupby(["landtype", "pattern"])[["n_flood", "n_dry"]].sum()
    out = np.zeros((6, N_PATTERNS))
    for k, name in LANDTYPES.items():
        if name not in c.index.get_level_values(0):
            raise ValueError(f"no training cell-days in {name} land")
        g = c.loc[name].reindex(range(N_PATTERNS), fill_value=0)
        nf, nd = g.n_flood.to_numpy(float), g.n_dry.to_numpy(float)
        if nf.sum() == 0 or nd.sum() == 0:
            raise ValueError(f"{name} land: flood {nf.sum():.0f}, dry {nd.sum():.0f} training cell-days; cannot fit")
        out[2 * k] = out[2 * k + 1] = log_lr_one(nf, nd, alpha)
    return out


def run_sequence(obs_seq: dict[str, np.ndarray], stratum_seq: np.ndarray, domain: np.ndarray, log_lr: np.ndarray,
                 model: engine.Model) -> np.ndarray:
    """P(flood) per (day, cell): the engine's recursion (model.prior_logodds, model.decay) with each day's
    evidence looked up from the pattern table. NaN outside the domain or where land type is unknown."""
    p = np.empty(stratum_seq.shape, "float32")
    L = None
    for t in range(stratum_seq.shape[0]):
        strat = np.where(domain, stratum_seq[t], NODATA).astype("uint8")
        ok = strat != NODATA
        ids = pattern_ids({s: obs_seq[s][t] for s in MODEL_SOURCES})
        ev = np.where(ok, log_lr[np.where(ok, strat, 0).astype(int), ids], np.nan)
        L = engine.step(L, strat, ev, model)
        p[t] = np.where(ok, engine.sigmoid(L), np.nan)
    return p

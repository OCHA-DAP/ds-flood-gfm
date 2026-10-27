"""Daily log-odds fusion (fixed decision 6). Pure array functions; no I/O.

For each cell and day:
    evidence_t = sum over sources s of
                   log(sens_s / (1 - spec_s))      if s says flood
                   log((1 - sens_s) / spec_s)      if s says dry
                   0                               if s is NoData
    L_t = prior + decay * (L_{t-1} - prior) + evidence_t
    P_t = 1 / (1 + exp(-L_t))
sens/spec and the prior are looked up by the cell's stratum (static per month).
Cells outside the floodable domain, or with unknown stratum, get NaN.

Outputs per day: P(flooded), |L| and the number of sources observed today
(uncertainty), and days since the last direct observation by any source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NODATA = 255
NEVER = np.uint16(65535)  # days_since_obs value for "never observed in this run"


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.asarray(p, dtype="float64")
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype="float64")))


@dataclass
class Model:
    """Accuracy table + prior as dense lookup arrays indexed by stratum code (0..5)."""

    sources: list[str]
    w_flood: np.ndarray  # (n_sources, 6)
    w_dry: np.ndarray  # (n_sources, 6)
    prior_logodds: np.ndarray  # (6,)
    decay: float

    @classmethod
    def from_tables(cls, accuracy: pd.DataFrame, prior: pd.DataFrame, sources: list[str], decay: float,
                    clip: tuple[float, float] = (0.001, 0.999), prior_clip: tuple[float, float] = (1e-5, 0.999)) -> Model:
        """accuracy: columns source, stratum_code, sens, spec. prior: columns stratum_code, base_rate."""
        for col in ("sens", "spec"):
            sub = accuracy[accuracy.source.isin(sources)][col]
            if not np.isfinite(sub.astype(float)).all():
                raise ValueError(f"accuracy table has non-finite {col} for sources {sources}: a 0/0 is not an estimate")
        if not np.isfinite(prior.base_rate.astype(float)).all():
            raise ValueError("prior has non-finite base_rate")
        n = len(sources)
        wf = np.zeros((n, 6))
        wd = np.zeros((n, 6))
        for i, s in enumerate(sources):
            a = accuracy[accuracy.source == s].set_index("stratum_code")
            missing = set(range(6)) - set(a.index)
            if missing:
                raise ValueError(f"accuracy table lacks strata {sorted(missing)} for source {s}")
            for k in range(6):
                se = float(np.clip(a.loc[k, "sens"], *clip))
                sp = float(np.clip(a.loc[k, "spec"], *clip))
                wf[i, k] = np.log(se / (1 - sp))
                wd[i, k] = np.log((1 - se) / sp)
        pr = prior.set_index("stratum_code").base_rate
        if set(range(6)) - set(pr.index):
            raise ValueError("prior lacks strata")
        # the prior has its own (wider) clip: a base flood rate of 1e-4 is plausible and must not be
        # raised to the sens/spec clip of 1e-3
        prior_lo = logit(np.clip(pr.loc[list(range(6))].values.astype(float), *prior_clip))
        return cls(sources, wf, wd, prior_lo, decay)


def evidence(obs: dict[str, np.ndarray], stratum: np.ndarray, model: Model) -> np.ndarray:
    """Summed log-likelihood ratio for one day. NoData adds exactly 0."""
    ev = np.zeros(stratum.shape)
    ok = stratum != NODATA
    st = np.where(ok, stratum, 0).astype(int)
    for i, s in enumerate(model.sources):
        o = obs[s]
        if o.shape != stratum.shape:
            raise ValueError(f"{s}: shape {o.shape} != stratum {stratum.shape}")
        bad = ~np.isin(o, (0, 1, NODATA))
        if bad.any():
            raise ValueError(f"{s}: values outside {{0, 1, 255}}: {np.unique(o[bad])[:5]}")
        ev += np.where(o == 1, model.w_flood[i, st], 0.0)
        ev += np.where(o == 0, model.w_dry[i, st], 0.0)
    return np.where(ok, ev, np.nan)


def step(prev_logodds: np.ndarray | None, stratum: np.ndarray, ev: np.ndarray, model: Model) -> np.ndarray:
    """One day of the carry-over recursion. prev None = cold start at the prior."""
    ok = stratum != NODATA
    prior = np.where(ok, model.prior_logodds[np.where(ok, stratum, 0).astype(int)], np.nan)
    prev = prior if prev_logodds is None else np.where(np.isfinite(prev_logodds), prev_logodds, prior)
    return prior + model.decay * (prev - prior) + ev


def days_since(prev: np.ndarray | None, observed_today: np.ndarray) -> np.ndarray:
    if prev is None:
        prev = np.full(observed_today.shape, NEVER, "uint16")
    out = np.where(prev == NEVER, NEVER, np.minimum(prev.astype("int64") + 1, NEVER - 1)).astype("uint16")
    out[observed_today] = 0
    return out


@dataclass
class DayOutput:
    p_flood: np.ndarray  # float32, NaN outside domain
    abs_logodds: np.ndarray  # float32
    n_sources: np.ndarray  # uint8 (255 outside domain)
    days_since_obs: np.ndarray  # uint16 (65535 never / outside domain)
    logodds: np.ndarray  # float64 state for tomorrow


def run_day(obs: dict[str, np.ndarray], stratum: np.ndarray, domain: np.ndarray, model: Model,
            prev_logodds: np.ndarray | None, prev_days: np.ndarray | None) -> DayOutput:
    strat = np.where(domain, stratum, NODATA).astype("uint8")
    ev = evidence(obs, strat, model)
    L = step(prev_logodds, strat, ev, model)
    observed = np.zeros(strat.shape, bool)
    n = np.zeros(strat.shape, "uint8")
    for s in model.sources:
        o = (obs[s] != NODATA) & domain
        observed |= o
        n += o.astype("uint8")
    ds_ = days_since(prev_days, observed)
    in_dom = domain & (strat != NODATA)
    return DayOutput(
        p_flood=np.where(in_dom, sigmoid(L), np.nan).astype("float32"),
        abs_logodds=np.where(in_dom, np.abs(L), np.nan).astype("float32"),
        n_sources=np.where(in_dom, n, NODATA).astype("uint8"),
        days_since_obs=np.where(in_dom, ds_, NEVER).astype("uint16"),
        logodds=np.where(in_dom, L, np.nan),
    )


def run_sequence(obs_seq: dict[str, np.ndarray], stratum_seq: np.ndarray, domain: np.ndarray, model: Model,
                 prev_logodds: np.ndarray | None = None, prev_days: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Run consecutive days. obs_seq[s] and stratum_seq have shape (days, rows, cols)."""
    nd = stratum_seq.shape[0]
    out = {
        "p_flood": np.empty(stratum_seq.shape, "float32"),
        "abs_logodds": np.empty(stratum_seq.shape, "float32"),
        "n_sources": np.empty(stratum_seq.shape, "uint8"),
        "days_since_obs": np.empty(stratum_seq.shape, "uint16"),
    }
    L, D = prev_logodds, prev_days
    for t in range(nd):
        r = run_day({s: obs_seq[s][t] for s in model.sources}, stratum_seq[t], domain, model, L, D)
        out["p_flood"][t], out["abs_logodds"][t] = r.p_flood, r.abs_logodds
        out["n_sources"][t], out["days_since_obs"][t] = r.n_sources, r.days_since_obs
        L, D = r.logodds, r.days_since_obs
    out["final_logodds"] = L
    out["final_days"] = D
    return out

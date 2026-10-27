"""Confusion counts, skill scores and coverage from event cubes.

Everything is first reduced to integer counts per
(event, window, source, stratum, day_type, label_s1), so that every later
statistic (pooled scores, per-stratum tables, event-cluster bootstrap) is a
sum over rows of one table and is reproducible from it.

Definitions (docs/decisions.md D-20):
  hit rate (sensitivity) = TP / (TP + FN)
  specificity            = TN / (TN + FP)
  false-alarm rate (POFD)= FP / (FP + TN) = 1 - specificity
  false-alarm ratio (FAR)= FP / (TP + FP)
  CSI                    = TP / (TP + FP + FN)
  precision              = TP / (TP + FP) = 1 - FAR
  F1                     = 2 TP / (2 TP + FP + FN), the headline score from 2026-10-14
Counts use only cells where both truth and the source are observed
("conditional" scores). `*_all` variants treat source NoData as "no flood
detected" over every truth cell (what a map user sees).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion.static import STRATUM_NAMES

SOURCES = ["gfm", "microwave", "viirs", "floodscan", "gfds"]
MODEL_SOURCES = ["gfm", "microwave", "viirs"]


def _counts(pred: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    tf, td = truth == 1, truth == 0
    obs = pred != 255
    pf, pd_ = pred == 1, pred == 0
    return {
        "tp": int((pf & tf).sum()),
        "fp": int((pf & td).sum()),
        "fn": int((pd_ & tf).sum()),
        "tn": int((pd_ & td).sum()),
        "truth_flood": int(tf.sum()),
        "truth_dry": int(td.sum()),
        "obs_truth_flood": int((obs & tf).sum()),
        "obs_truth_dry": int((obs & td).sum()),
    }


def event_counts(ds: xr.Dataset, code: str, continent: str, split: str, s1_days: set, sources=SOURCES) -> pd.DataFrame:
    """Confusion counts for one event-window cube."""
    truth = ds.truth.values
    strat = ds.stratum.values
    days = pd.to_datetime(ds.day.values)
    start, end = pd.Timestamp(ds.attrs["event_start"]), pd.Timestamp(ds.attrs["event_end"])
    rows = []
    for i, d in enumerate(days):
        t = truth[i]
        if not ((t == 0) | (t == 1)).any():
            continue
        if start <= d <= end:
            day_type = "gold"
            days_after_end = np.nan
        else:
            day_type = "dry_pre" if d < start else "dry_post"
            days_after_end = (d - end).days if d > end else np.nan
        for s in np.unique(strat[i][(t == 0) | (t == 1)]):
            if s == 255:
                continue
            m = strat[i] == s
            tt = np.where(m, t, 255)
            for src in sources:
                c = _counts(ds[src].values[i], tt)
                rows.append({"code": code, "window": ds.attrs["window_idx"], "continent": continent, "split": split,
                             "day": d, "day_type": day_type, "days_after_end": days_after_end,
                             "label_s1": d.normalize() in s1_days, "stratum": STRATUM_NAMES[int(s)], "source": src, **c})
    return pd.DataFrame(rows)


def event_coverage(ds: xr.Dataset, code: str, continent: str, split: str, sources=SOURCES) -> pd.DataFrame:
    """Per day and stratum: domain cells, and cells observed by each source / any / any within 3 days."""
    dom = ds.domain.values.astype(bool)
    strat = ds.stratum.values
    obs = {s: ds[s].values != 255 for s in sources}
    any_obs = obs["gfm"] | obs["microwave"] | obs["viirs"]
    nday = any_obs.shape[0]
    any3 = np.zeros_like(any_obs)
    gfm3 = np.zeros_like(any_obs)
    for i in range(nday):
        lo = max(0, i - 2)
        any3[i] = any_obs[lo : i + 1].any(axis=0)
        gfm3[i] = obs["gfm"][lo : i + 1].any(axis=0)
    rows = []
    days = pd.to_datetime(ds.day.values)
    for i, d in enumerate(days):
        for s in range(6):
            m = dom & (strat[i] == s)
            n = int(m.sum())
            if n == 0:
                continue
            r = {"code": code, "window": ds.attrs["window_idx"], "continent": continent, "split": split, "day": d,
                 "day_index": i, "stratum": STRATUM_NAMES[s], "n_domain": n,
                 "any": int((any_obs[i] & m).sum()), "any_3d": int((any3[i] & m).sum()), "gfm_3d": int((gfm3[i] & m).sum())}
            for src in sources:
                r[src] = int((obs[src][i] & m).sum())
            rows.append(r)
    return pd.DataFrame(rows)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add score columns to a frame of summed counts."""
    out = df.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["sens"] = out.tp / (out.tp + out.fn)
        out["spec"] = out.tn / (out.tn + out.fp)
        # Jeffreys (Beta(1/2, 1/2)) posterior means: never exactly 0 or 1, so a source that was
        # never right (or never wrong) on a handful of cells cannot become a near-infinite weight (D-32)
        out["sens_j"] = (out.tp + 0.5) / (out.tp + out.fn + 1.0)
        out["spec_j"] = (out.tn + 0.5) / (out.tn + out.fp + 1.0)
        out["pofd"] = out.fp / (out.fp + out.tn)
        out["far"] = out.fp / (out.tp + out.fp)
        out["csi"] = out.tp / (out.tp + out.fp + out.fn)
        out["precision"] = out.tp / (out.tp + out.fp)
        out["f1"] = 2 * out.tp / (2 * out.tp + out.fp + out.fn)
        # all-cells variants: source NoData counted as "not detected"
        out["sens_all"] = out.tp / out.truth_flood
        out["csi_all"] = out.tp / (out.truth_flood + out.fp)
        out["coverage_flood"] = out.obs_truth_flood / out.truth_flood
        out["coverage_dry"] = out.obs_truth_dry / out.truth_dry
    return out


COUNT_COLS = ["tp", "fp", "fn", "tn", "truth_flood", "truth_dry", "obs_truth_flood", "obs_truth_dry"]


def pooled(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return add_scores(df.groupby(by, as_index=False)[COUNT_COLS].sum())


def bootstrap_ci(df: pd.DataFrame, by: list[str], stats: list[str], reps: int, seed: int, cluster: str = "code") -> pd.DataFrame:
    """Event-cluster bootstrap 95% intervals for `stats` within each `by` group.

    Events are resampled with replacement (whole events, all their cells and
    days), so within-event spatial and temporal dependence is respected.
    """
    rng = np.random.default_rng(seed)
    per_event = df.groupby([*by, cluster], as_index=False)[COUNT_COLS].sum()
    codes = per_event[cluster].unique()
    idx = {c: i for i, c in enumerate(codes)}
    per_event["_ci"] = per_event[cluster].map(idx)
    groups = per_event.groupby(by)
    results = []
    for key, g in groups:
        mat = np.zeros((len(codes), len(COUNT_COLS)))
        mat[g._ci.values] = g[COUNT_COLS].values
        draws = rng.integers(0, len(codes), size=(reps, len(codes)))
        sums = np.stack([mat[d].sum(axis=0) for d in draws])
        boot = add_scores(pd.DataFrame(sums, columns=COUNT_COLS))
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        for s in stats:
            v = boot[s].dropna().values
            row[f"{s}_lo"] = float(np.quantile(v, 0.025)) if len(v) else np.nan
            row[f"{s}_hi"] = float(np.quantile(v, 0.975)) if len(v) else np.nan
        row["n_events"] = int((mat.sum(axis=1) > 0).sum())
        results.append(row)
    return pd.DataFrame(results)

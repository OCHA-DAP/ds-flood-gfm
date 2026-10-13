"""Phase 5 validation on held-out events (never used for the accuracy table or prior).

For every held-out event window the daily job runs over the whole window period
(cold start at window_start, so the 30-day pre-window is the spin-up). Every
daily output is written as a COG. Then:

  skill        fused (P >= 0.5) vs each single source, per stratum and per continent,
               conditional (cells the source observed; fusion scored on the same cells)
               and all-cells (source NoData = not detected; fusion scored everywhere)
  reliability  binned P vs observed flood frequency per stratum; slope by weighted
               least squares on the bins and by logistic calibration (IRLS)
  coverage     share of domain cells observed today (fusion = any source) and within 3 days,
               and the distribution of days since last observation on gold days
  timing       days from event start (first gold day) to first detection inside the
               gold flood footprint, fusion vs each source
  ablation     the same, with each source (and each microwave component) dropped
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import cube, engine, gold, grid, job
from ds_flood_gfm.fusion.config import Config
from ds_flood_gfm.fusion.metrics import add_scores
from ds_flood_gfm.fusion.static import STRATUM_NAMES

log = logging.getLogger("phase5")
SINGLE = ["gfm", "microwave", "viirs", "floodscan", "gfds"]


def _conf(pred: np.ndarray, truth: np.ndarray) -> dict:
    tf, td = truth == 1, truth == 0
    return {"tp": int((pred & tf).sum()), "fp": int((pred & td).sum()), "fn": int((~pred & tf).sum()), "tn": int((~pred & td).sum()),
            "truth_flood": int(tf.sum()), "truth_dry": int(td.sum()), "obs_truth_flood": int(tf.sum()), "obs_truth_dry": int(td.sum())}


def on_change(seq: np.ndarray) -> np.ndarray:
    """Keep a source's value only on days it differs from its previous day's value at that cell.

    NoData breaks a run (the first observation after a gap counts). Ablation only (D-33):
    a strict reading of the model's conditional-independence assumption for sources whose
    errors persist from day to day at the same place.
    """
    out = seq.copy()
    same = np.zeros(seq.shape, bool)
    same[1:] = (seq[1:] == seq[:-1]) & (seq[1:] != 255)
    out[same] = 255
    return out


def fused_sequence(cfg: Config, ds: xr.Dataset, model: engine.Model, key: str, write_cogs: bool,
                   transform: dict | None = None) -> dict[str, np.ndarray]:
    """Run the job over the cube's days; returns arrays (day, lat, lon) of P, |L|, n_sources, days_since."""
    win = grid.Window(*json.loads(ds.attrs["window"]))
    days = [pd.Timestamp(d).date() for d in ds.day.values]
    stratum_by_day = ds.stratum.values
    dom = ds.domain.values.astype(bool)
    if write_cogs:
        months = {}
        for i, d in enumerate(days):
            months.setdefault(d.month, stratum_by_day[i])
        st = job.StaticForWindow(dom, {m: months.get(m, np.full(dom.shape, 255, "uint8")) for m in range(1, 13)})
        job.run_window(cfg, key, win, days, job.CubeProvider(ds, model.sources), model, st, cold_start=True)
    obs = {s: ds[s].values for s in model.sources}
    for s, f in (transform or {}).items():
        obs[s] = f(obs[s])
    return engine.run_sequence(obs, stratum_by_day, dom, model)


def event_rows(ds: xr.Dataset, fused: dict, code: str, continent: str, threshold: float, label: str) -> list[dict]:
    """Confusion counts on gold days (and dry days) for fusion and single sources, by stratum."""
    t = ds.truth.values
    strat = ds.stratum.values
    days = pd.to_datetime(ds.day.values)
    start, end = pd.Timestamp(ds.attrs["event_start"]), pd.Timestamp(ds.attrs["event_end"])
    p = fused["p_flood"]
    rows = []
    for i, d in enumerate(days):
        lab = (t[i] == 0) | (t[i] == 1)
        if not lab.any():
            continue
        day_type = "gold" if start <= d <= end else "dry"
        fp_ = np.where(np.isfinite(p[i]), p[i] >= threshold, False)
        for code_s, name in STRATUM_NAMES.items():
            m = lab & (strat[i] == code_s) & np.isfinite(p[i])
            if not m.any():
                continue
            base = {"code": code, "continent": continent, "day": d, "day_type": day_type, "stratum": name, "model": label}
            rows.append({**base, "product": "fusion", "mode": "all", **_conf(fp_[m], t[i][m])})
            for s in SINGLE:
                v = ds[s].values[i]
                obs = (v != 255) & m
                rows.append({**base, "product": s, "mode": "all", **_conf((v == 1)[m], t[i][m])})
                if obs.any():
                    rows.append({**base, "product": s, "mode": "conditional", **_conf((v == 1)[obs], t[i][obs])})
                    rows.append({**base, "product": f"fusion|{s}", "mode": "conditional", **_conf(fp_[obs], t[i][obs])})
    if not rows:
        return []
    # aggregate over days within the event (skill tables never need the day)
    df = pd.DataFrame(rows)
    keys = ["code", "continent", "day_type", "stratum", "model", "product", "mode"]
    return df.groupby(keys, as_index=False)[["tp", "fp", "fn", "tn", "truth_flood", "truth_dry", "obs_truth_flood", "obs_truth_dry"]].sum().to_dict("records")


def reliability_rows(ds: xr.Dataset, fused: dict, code: str, bins: int) -> pd.DataFrame:
    t = ds.truth.values
    p = fused["p_flood"]
    strat = ds.stratum.values
    lab = ((t == 0) | (t == 1)) & np.isfinite(p)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for code_s, name in STRATUM_NAMES.items():
        m = lab & (strat == code_s)
        if not m.any():
            continue
        pp, tt = p[m], t[m]
        b = np.clip(np.digitize(pp, edges) - 1, 0, bins - 1)
        for k in range(bins):
            s = b == k
            if s.any():
                out.append({"code": code, "stratum": name, "bin": k, "n": int(s.sum()), "sum_p": float(pp[s].sum()), "n_flood": int(tt[s].sum())})
    return pd.DataFrame(out)


def coverage_rows(ds: xr.Dataset, fused: dict, code: str, continent: str) -> pd.DataFrame:
    dom = ds.domain.values.astype(bool)
    n = int(dom.sum())
    rows = []
    days = pd.to_datetime(ds.day.values)
    t = ds.truth.values
    for i, d in enumerate(days):
        ns = fused["n_sources"][i][dom]
        age = fused["days_since_obs"][i][dom]
        r = {"code": code, "continent": continent, "day": d, "n_domain": n,
             "fusion_today": int((ns > 0).sum()), "fusion_within_3d": int((age <= 2).sum())}
        for s in SINGLE:
            r[s] = int((ds[s].values[i][dom] != 255).sum())
        gold_day = ((t[i] == 0) | (t[i] == 1)).any() and (pd.Timestamp(ds.attrs["event_start"]) <= d <= pd.Timestamp(ds.attrs["event_end"]))
        r["gold_day"] = bool(gold_day)
        if gold_day:
            a = age[age != engine.NEVER]
            r["age_median"] = float(np.median(a)) if a.size else np.nan
            r["age_p90"] = float(np.percentile(a, 90)) if a.size else np.nan
            r["never_observed"] = int((age == engine.NEVER).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def timing_row(ds: xr.Dataset, fused: dict, code: str, threshold: float) -> dict:
    """Days from event start to the first day >= max(5, 10%) of the gold flood footprint is flagged."""
    t = ds.truth.values
    fp = (t == 1).any(axis=0)
    n_fp = int(fp.sum())
    days = pd.to_datetime(ds.day.values)
    start = pd.Timestamp(ds.attrs["event_start"])
    out = {"code": code, "gold_flood_footprint_cells": n_fp}
    if n_fp == 0:
        return out
    need = max(5, int(np.ceil(0.1 * n_fp))) if n_fp >= 5 else 1
    series = {"fusion": np.where(np.isfinite(fused["p_flood"]), fused["p_flood"] >= threshold, False)}
    for s in SINGLE:
        series[s] = ds[s].values == 1
    after = np.asarray(days >= start)
    for name, arr in series.items():
        hits = arr[:, fp].sum(axis=1) >= need
        idx = np.where(hits)[0]
        idx_after = np.where(hits & after)[0]
        # first detection anywhere in the window (negative = before the first gold map; the flood may
        # already have begun, or it is a false alarm) and first detection on/after the first gold day
        out[f"{name}_first_any_days"] = int((days[idx[0]] - start).days) if idx.size else np.nan
        out[f"{name}_lag_days"] = int((days[idx_after[0]] - start).days) if idx_after.size else np.nan
    return out


def logistic_slope(p: np.ndarray, k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """Calibration intercept a and slope b: logit P(flood) = a + b * logit(p), binomial MLE on bin counts.

    k flood cells out of n in each bin with mean prediction p. Slope 1 and intercept 0 = calibrated.
    Returns NaN when predictions have no spread (not estimable).
    """
    from scipy.optimize import minimize

    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    x = np.log(p) - np.log1p(-p)
    k, n = np.asarray(k, float), np.asarray(n, float)
    if len(np.unique(np.round(x, 9))) < 2 or n.sum() <= 0:
        return float("nan"), float("nan")

    def nll(beta):
        eta = beta[0] + beta[1] * x
        # log(1 + exp(eta)) computed stably
        return float(np.sum(n * np.logaddexp(0.0, eta) - k * eta))

    r = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    if not r.success and not np.isfinite(r.fun):
        raise RuntimeError(f"calibration fit failed: {r.message}")
    return float(r.x[0]), float(r.x[1])


def summarise_reliability(rel: pd.DataFrame, bins: int) -> pd.DataFrame:
    out = []
    for st, g in rel.groupby("stratum"):
        a = g.groupby("bin")[["n", "sum_p", "n_flood"]].sum()
        mp = a.sum_p / a.n
        of = a.n_flood / a.n
        w = a.n.values.astype(float)
        if len(a) >= 2 and np.ptp(mp.values) > 0:
            A = np.column_stack([np.ones(len(a)), mp.values])
            coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], of.values * np.sqrt(w), rcond=None)[0]
            slope_wls = float(coef[1])
        else:
            slope_wls = np.nan
        # logistic calibration on bin aggregates: y = flood share per bin with weight n
        ai, bi = logistic_slope(mp.values, a.n_flood.values, a.n.values)
        out.append({"stratum": st, "n": int(a.n.sum()), "n_flood": int(a.n_flood.sum()), "bins_used": len(a),
                    "reliability_slope_wls": slope_wls, "calibration_slope_logistic": bi, "calibration_intercept_logistic": ai,
                    "mean_p": float(a.sum_p.sum() / a.n.sum()), "obs_freq": float(a.n_flood.sum() / a.n.sum())})
    return pd.DataFrame(out)


def holdout_keys(cfg: Config, sel: pd.DataFrame) -> list[tuple[pd.Series, gold.EventWindow]]:
    out = []
    for _, r in sel[sel.split == "holdout"].iterrows():
        for ew in gold.event_windows(r):
            out.append((r, ew))
    return out


def run(cfg: Config, sel: pd.DataFrame, model: engine.Model, label: str, write_cogs: bool,
        transform: dict | None = None) -> dict[str, pd.DataFrame]:
    thr = cfg["fusion"]["prob_threshold"]
    bins = cfg["validation"]["reliability_bins"]
    skill, rel, cov, tim = [], [], [], []
    for r, ew in holdout_keys(cfg, sel):
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        if write_cogs and transform:
            raise ValueError("transformed (ablation) runs never write the product COGs")
        fused = fused_sequence(cfg, ds, model, ew.key, write_cogs, transform)
        skill += event_rows(ds, fused, r.code, r.continent, thr, label)
        rel.append(reliability_rows(ds, fused, r.code, bins))
        cov.append(coverage_rows(ds, fused, r.code, r.continent))
        tim.append(timing_row(ds, fused, r.code, thr) | {"window": ew.idx, "continent": r.continent})
    return {"skill": pd.DataFrame(skill), "reliability": pd.concat(rel, ignore_index=True),
            "coverage": pd.concat(cov, ignore_index=True), "timing": pd.DataFrame(tim)}


def skill_table(skill: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = skill.groupby(["model", "product", "mode", *by], as_index=False)[["tp", "fp", "fn", "tn", "truth_flood", "truth_dry", "obs_truth_flood", "obs_truth_dry"]].sum()
    return add_scores(g)

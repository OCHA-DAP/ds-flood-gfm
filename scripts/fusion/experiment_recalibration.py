"""Experiment (not the product): per-stratum recalibration of the fused log-odds.

The v1 product follows fixed decision 6 exactly and is over-confident (Phase 5). This
experiment asks how much a post-hoc calibration would fix, without touching held-out data:

  1. run the v1 fusion over every TRAINING event window (same code as Phase 5);
  2. per stratum, fit logit P(flood) = a + b * L by binomial MLE on labelled cell-days
     (gold days + dry windows), aggregated over 0.25-wide bins of L;
  3. apply (a, b) to the HELD-OUT fused log-odds; report reliability slope (10 quantile bins,
     since calibrated probabilities concentrate near zero) and CSI at 0.5;
  4. per stratum, pick the P threshold that maximises CSI on TRAINING gold days and apply it to
     held-out gold days (recalibration is monotone, so the map is the same for both variants).

Output: results/experiment_recalibration.csv (and _coefficients.csv). Ranked as a next
improvement in docs/final_report.md; not part of the v1 product.

Run: nice -n 19 uv run python scripts/fusion/experiment_recalibration.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize

from ds_flood_gfm.fusion import cube, gold, job, validate
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES, add_scores
from ds_flood_gfm.fusion.static import STRATUM_NAMES


def collect(cfg, sel, split, model):
    """Per stratum: histogram of fused log-odds L for flood and dry labelled cell-days (gold + dry), and raw arrays for scoring."""
    rows = []
    for _, r in sel[sel.split == split].iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            obs = {s: ds[s].values for s in model.sources}
            dom = ds.domain.values.astype(bool)
            out = validate.engine.run_sequence(obs, ds.stratum.values, dom, model)
            p = out["p_flood"].astype("float64")
            t = ds.truth.values
            days = pd.to_datetime(ds.day.values)
            gold_day = ((days >= pd.Timestamp(ds.attrs["event_start"])) & (days <= pd.Timestamp(ds.attrs["event_end"])))[:, None, None]
            lab = ((t == 0) | (t == 1)) & np.isfinite(p)
            L = np.log(np.clip(p, 1e-12, 1)) - np.log1p(-np.clip(p, 0, 1 - 1e-12))
            st = ds.stratum.values
            for code, name in STRATUM_NAMES.items():
                m = lab & (st == code)
                if m.any():
                    rows.append(pd.DataFrame({"stratum": name, "L": L[m], "y": t[m].astype(int),
                                              "gold_day": np.broadcast_to(gold_day, t.shape)[m]}))
    return pd.concat(rows, ignore_index=True)


def fit(df):
    bins = np.round(df.L.clip(-30, 30) * 4) / 4
    g = df.assign(b=bins).groupby("b").y.agg(["sum", "count"]).reset_index()
    x, k, n = g.b.values, g["sum"].values.astype(float), g["count"].values.astype(float)

    def nll(beta):
        eta = beta[0] + beta[1] * x
        return float(np.sum(n * np.logaddexp(0.0, eta) - k * eta))

    r = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    return float(r.x[0]), float(r.x[1])


def main():
    cfg = load_config()
    res = cfg.results_dir
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    model = job.load_model(cfg)
    train = collect(cfg, sel, "train", model)
    test = collect(cfg, sel, "holdout", model)
    coefs, rows = [], []
    bins = cfg["validation"]["reliability_bins"]
    for st in STRATUM_NAMES.values():
        tr = train[train.stratum == st]
        te = test[test.stratum == st]
        if tr.y.sum() < 20 or te.empty:
            coefs.append({"stratum": st, "a": np.nan, "b": np.nan, "n_train_flood": int(tr.y.sum()), "note": "too few training flood cells"})
            continue
        a, b = fit(tr)
        coefs.append({"stratum": st, "a": a, "b": b, "n_train_flood": int(tr.y.sum()), "note": ""})
        # threshold that maximises training CSI on gold days (v1 P; monotone => same map after recalibration)
        trg = tr[tr.gold_day]
        p_tr = 1 / (1 + np.exp(-trg.L.values))
        best_thr, best_csi = 0.5, -1.0
        for thr in np.round(np.arange(0.02, 0.99, 0.02), 2):
            pr = p_tr >= thr
            tp = (pr & (trg.y.values == 1)).sum()
            den = tp + (pr & (trg.y.values == 0)).sum() + (~pr & (trg.y.values == 1)).sum()
            c = tp / den if den else 0.0
            if c > best_csi:
                best_thr, best_csi = float(thr), float(c)
        for label, L, thr in [("v1 (as specified), P>=0.5", te.L.values, 0.5),
                              ("recalibrated, P>=0.5", a + b * te.L.values, 0.5),
                              (f"v1, trained threshold P>={best_thr:.2f}", te.L.values, best_thr)]:
            p = 1 / (1 + np.exp(-L))
            y = te.y.values
            qs = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
            bi = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, max(len(qs) - 2, 0))
            nbin = max(len(qs) - 1, 1)
            nb = np.bincount(bi, minlength=nbin)
            kb = np.bincount(bi, weights=y, minlength=nbin)
            sp_ = np.bincount(bi, weights=p, minlength=nbin)
            rel = pd.DataFrame([{"stratum": st, "bin": i, "n": int(nb[i]), "sum_p": float(sp_[i]), "n_flood": int(kb[i])} for i in range(nbin) if nb[i] > 0])
            summ = validate.summarise_reliability(rel, nbin).iloc[0]
            gd = te.gold_day.values
            pred = p >= thr
            c = add_scores(pd.DataFrame([{"tp": int((pred & (y == 1) & gd).sum()), "fp": int((pred & (y == 0) & gd).sum()),
                                          "fn": int((~pred & (y == 1) & gd).sum()), "tn": int((~pred & (y == 0) & gd).sum()),
                                          "truth_flood": int(((y == 1) & gd).sum()), "truth_dry": int(((y == 0) & gd).sum()),
                                          "obs_truth_flood": 0, "obs_truth_dry": 0}])).iloc[0]
            rows.append({"stratum": st, "variant": label, "reliability_slope_wls_quantile_bins": summ.reliability_slope_wls,
                         "calibration_slope_logistic": summ.calibration_slope_logistic, "mean_p": summ.mean_p, "obs_freq": summ.obs_freq,
                         "csi_gold_days": c.csi, "sens": c.sens, "far": c.far, "n_heldout_flood_gold": int(c.truth_flood),
                         "train_best_threshold": best_thr, "train_best_csi": best_csi})
    pd.DataFrame(coefs).to_csv(res / "experiment_recalibration_coefficients.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(res / "experiment_recalibration.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(coefs).to_string(index=False))
    print(out.round(3).to_string(index=False))
    _ = MODEL_SOURCES


if __name__ == "__main__":
    main()

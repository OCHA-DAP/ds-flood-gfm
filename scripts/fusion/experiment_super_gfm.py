"""Experiment: "super-GFM". Learn GFM's flood map from inputs that exist on days GFM does not observe;
judge the result on independent gold.

Labels: GFM's own call (1 flood / 0 dry) wherever GFM observed, training events only.
Features per cell-day, all available without a Sentinel-1 pass that day:
  microwave (OR vote), FloodScan, GFDS today; microwave flood days in the previous 3 and 7 days;
  VIIRS today; VIIRS last observed value before today and its age; GFM last observed value before
  today and its age (persistence); static layers (HAND-low, permanent-water, tree, built fractions,
  stratum); the month's normally-wet share (GSW monthly recurrence, D-38); season; latitude.
Model: HistGradientBoostingClassifier (missing values = "not observed"). A second model sees only
the static, baseline, season and latitude features: the "prior-only" control.

The product judged is "GFM where GFM observed that day, else the model". On held-out gold days,
against gold, over every labelled cell (no observation = miss for single products), it is compared
with: GFM persistence (last GFM observation carried forward), the static-only model, fusion v1
(P >= 0.5), VIIRS, and GFM alone. All rows are also given for events whose gold maps were not drawn
from Sentinel-1 (frac_s1_sets == 0), so that GFM is not graded against its own satellite.

Outputs (results/): experiment_super_gfm_skill.csv, experiment_super_gfm_vs_gfm.csv (how well the
model reproduces GFM on held-out GFM-observed cells), experiment_super_gfm_importance.csv
Run: nice -n 19 uv run python scripts/fusion/experiment_super_gfm.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from ds_flood_gfm.fusion import baseline, cube, engine, gold
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA

log = logging.getLogger("super_gfm")
STATIC = ["hand_low_frac", "perm_water_frac", "tree_frac", "built_frac", "stratum", "normally_wet_share", "month_sin", "month_cos", "abs_lat"]
DYNAMIC = ["mw_today", "floodscan_today", "gfds_today", "mw_flood_prev3", "mw_flood_prev7", "viirs_today", "viirs_last", "viirs_age",
           "gfm_last", "gfm_age"]


def as_float(v: np.ndarray) -> np.ndarray:
    return np.where(v == NODATA, np.nan, v).astype("float32")


def last_observed(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each day: the source's last observed value strictly before that day, and its age in days (NaN if none)."""
    nd = v.shape[0]
    last = np.full(v.shape, np.nan, "float32")
    age = np.full(v.shape, np.nan, "float32")
    cur_v = np.full(v.shape[1:], np.nan, "float32")
    cur_d = np.full(v.shape[1:], np.nan, "float32")
    for d in range(nd):
        last[d], age[d] = cur_v, d - cur_d
        obs = v[d] != NODATA
        cur_v = np.where(obs, v[d], cur_v)
        cur_d = np.where(obs, d, cur_d)
    return last, age


def month_share(cfg, win, month: int) -> np.ndarray:
    """Normally-wet share per cell for a calendar month (fusion/baseline.py caches it on first use)."""
    path = cfg.local("static", "baseline", f"{win.key()}_m{month:02d}.npy")
    if not path.exists():
        baseline.normally_wet(cfg, win, month)
    return np.load(path)


def window_features(cfg, ew, ds: xr.Dataset) -> dict[str, np.ndarray]:
    """Feature arrays (day, lat, lon) for one event window."""
    nd = ds.sizes["day"]
    days = pd.to_datetime(ds.day.values)
    st = xr.load_dataset(cfg.local("static", f"{ew.window.key()}.nc"), engine="netcdf4")
    f = {}
    for k in ("hand_low_frac", "perm_water_frac", "tree_frac", "built_frac"):
        f[k] = np.broadcast_to(st[k].values.astype("float32"), (nd, *st[k].shape))
    f["stratum"] = ds.stratum.values.astype("float32")
    share = {m: month_share(cfg, ew.window, m) for m in sorted(set(days.month))}
    f["normally_wet_share"] = np.stack([share[d.month] for d in days]).astype("float32")
    mon = np.array([d.month for d in days], "float32")[:, None, None]
    f["month_sin"] = np.broadcast_to(np.sin(2 * np.pi * mon / 12), f["stratum"].shape)
    f["month_cos"] = np.broadcast_to(np.cos(2 * np.pi * mon / 12), f["stratum"].shape)
    f["abs_lat"] = np.broadcast_to(np.abs(ds.lat.values.astype("float32"))[None, :, None], f["stratum"].shape)
    mw = ds.microwave.values
    f["mw_today"], f["floodscan_today"], f["gfds_today"] = as_float(mw), as_float(ds.floodscan.values), as_float(ds.gfds.values)
    fl = (mw == 1).astype("float32")
    cs = np.concatenate([np.zeros((1, *fl.shape[1:]), "float32"), np.cumsum(fl, axis=0)])
    idx = np.arange(nd)
    f["mw_flood_prev3"] = cs[idx] - cs[np.maximum(idx - 3, 0)]
    f["mw_flood_prev7"] = cs[idx] - cs[np.maximum(idx - 7, 0)]
    f["viirs_today"] = as_float(ds.viirs.values)
    f["viirs_last"], f["viirs_age"] = last_observed(ds.viirs.values)
    f["gfm_last"], f["gfm_age"] = last_observed(ds.gfm.values)
    return f


def stack(f: dict[str, np.ndarray], cols: list[str], mask: np.ndarray) -> np.ndarray:
    return np.column_stack([np.asarray(f[c])[mask] for c in cols]).astype("float32")


def main() -> None:
    cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "experiment_super_gfm.log")), logging.StreamHandler(sys.stdout)])
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    sp = pd.read_csv(cfg.results_dir / "event_split.csv").set_index("code")
    thr = cfg["fusion"]["prob_threshold"]
    rng = np.random.default_rng(20261013)

    # training rows: GFM-observed domain cells of training windows (all days of the window)
    Xs, ys = [], []
    for _, r in sel[sel.code.map(sp.split) == "train"].iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            f = window_features(cfg, ew, ds)
            m = (ds.gfm.values != NODATA) & ds.domain.values.astype(bool)[None]
            if m.any():
                Xs.append(stack(f, DYNAMIC + STATIC, m))
                ys.append((ds.gfm.values[m] == 1).astype("int8"))
    X, y = np.vstack(Xs), np.concatenate(ys)
    log.info("training rows (GFM-observed cell-days, training events): %d, GFM flood share %.4f", len(y), y.mean())
    cap = 3_000_000
    if len(y) > cap:  # keep every flood row, subsample dry rows; reweight so probabilities stay unbiased
        fl, dr = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        keep_dr = rng.choice(dr, cap - len(fl), replace=False)
        w_dry = len(dr) / len(keep_dr)
        idx = np.concatenate([fl, keep_dr])
        X, y, w = X[idx], y[idx], np.where(y[idx] == 1, 1.0, w_dry)
    else:
        w = np.ones(len(y))
    full = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X, y, sample_weight=w)
    ns = len(DYNAMIC)
    prior_only = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X[:, ns:], y, sample_weight=w)
    log.info("models trained")

    # held-out evaluation
    acc = pd.read_csv(cfg.results_dir / "accuracy_table.csv")
    prior = pd.read_csv(cfg.results_dir / "prior_v1.csv")
    fus = engine.Model.from_tables(acc, prior, MODEL_SOURCES, cfg["fusion"]["decay"], tuple(cfg["fusion"]["sens_spec_clip"]))
    rows, emu = [], []
    imp_X, imp_y = [], []
    for _, r in sel[sel.code.map(sp.split) == "holdout"].iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            f = window_features(cfg, ew, ds)
            dom = ds.domain.values.astype(bool)
            days = pd.to_datetime(ds.day.values)
            gd = (days >= pd.Timestamp(ds.attrs["event_start"])) & (days <= pd.Timestamp(ds.attrs["event_end"]))
            P_fus = engine.run_sequence({s: ds[s].values for s in MODEL_SOURCES}, ds.stratum.values, dom, fus)["p_flood"]
            allm = dom[None] & np.ones(ds.gfm.shape, bool)
            p_full = np.full(ds.gfm.shape, np.nan, "float32")
            p_prior = np.full(ds.gfm.shape, np.nan, "float32")
            Xa = stack(f, DYNAMIC + STATIC, allm)
            p_full[allm] = full.predict_proba(Xa)[:, 1]
            p_prior[allm] = prior_only.predict_proba(Xa[:, ns:])[:, 1]
            gfm = ds.gfm.values
            gobs = gfm != NODATA
            # how well the model reproduces GFM where GFM did observe (it never sees GFM that day)
            em = gobs & dom[None]
            emu.append({"code": r.code, "n": int(em.sum()), "gfm_flood": int((gfm[em] == 1).sum()),
                        "tp": int(((p_full[em] >= thr) & (gfm[em] == 1)).sum()), "fp": int(((p_full[em] >= thr) & (gfm[em] == 0)).sum()),
                        "fn": int(((p_full[em] < thr) & (gfm[em] == 1)).sum())})
            if len(imp_y) < 40 and em.any():
                imp_X.append(stack(f, DYNAMIC + STATIC, em)); imp_y.append((gfm[em] == 1).astype("int8"))
            t = ds.truth.values
            persist = np.where(gobs, gfm, f["gfm_last"])          # GFM today, else last GFM, else NaN
            preds = {
                "super-GFM (GFM if observed, else model)": np.where(gobs, gfm == 1, p_full >= thr),
                "model alone": p_full >= thr,
                "prior-only control (GFM if observed, else static model)": np.where(gobs, gfm == 1, p_prior >= thr),
                "GFM persistence (GFM if observed, else last GFM)": persist == 1,
                "fusion v1": P_fus >= thr,
                "VIIRS": ds.viirs.values == 1,
                "GFM alone": gfm == 1,
            }
            s1 = "non-S1 gold" if sp.loc[r.code, "frac_s1_sets"] == 0 else "S1-derived gold"
            for k in np.flatnonzero(gd):
                lab = ((t[k] == 0) | (t[k] == 1)) & dom
                if not lab.any():
                    continue
                y_ = t[k][lab] == 1
                for name, pr in preds.items():
                    pk = pr[k][lab]
                    rows.append({"code": r.code, "gold": s1, "product": name, "tp": int((pk & y_).sum()),
                                 "fp": int((pk & ~y_).sum()), "fn": int((~pk & y_).sum())})
    sk = pd.DataFrame(rows)
    res = cfg.results_dir
    sk.to_csv(res / "experiment_super_gfm_skill.csv", index=False)
    em = pd.DataFrame(emu)
    em.to_csv(res / "experiment_super_gfm_vs_gfm.csv", index=False)
    pi = permutation_importance(full, np.vstack(imp_X), np.concatenate(imp_y), n_repeats=3, random_state=0, scoring="average_precision")
    imp = pd.DataFrame({"feature": DYNAMIC + STATIC, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    imp.to_csv(res / "experiment_super_gfm_importance.csv", index=False)

    pd.set_option("display.width", 220)
    def summ(g):
        tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
        return pd.Series({"recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else np.nan, "csi": tp / (tp + fp + fn), "flood_cells": tp + fn})
    for label, g in [("all held-out gold days", sk), ("non-S1 gold only", sk[sk.gold == "non-S1 gold"])]:
        log.info("%s:\n%s", label, g.groupby("product").apply(summ, include_groups=False).sort_values("csi", ascending=False).round(3).to_string())
    e = em[["tp", "fp", "fn"]].sum()
    log.info("model vs GFM on held-out GFM-observed cells (model never sees GFM that day): recall %.3f precision %.3f",
             e.tp / (e.tp + e.fn), e.tp / (e.tp + e.fp))
    log.info("permutation importance (average precision, held-out GFM-observed cells):\n%s", imp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

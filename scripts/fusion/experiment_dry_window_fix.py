"""Experiment (D-39): corrected dry windows. A cell flooded on any gold day of its window is no longer
assumed dry in the dry windows (it becomes NOTRUTH there); every other footprint cell stays dry.

This changes fixed decision 5's literal reading (D-12). It is measured here; the product is unchanged.
  1. Accuracy table, training events. Sensitivity cannot change (flood truth exists only on gold
     days), so only specificity is re-estimated, as Jeffreys point estimates without bootstrap
     intervals. Rows that v1 estimated by EM (the snow strata, D-24) are kept as they are.
  2. Held-out events. The v1 model and the variant model are both run in memory over every window
     and scored the same way:
       - recall, precision and CSI on gold days (P >= prob_threshold);
       - reliability on three label sets: gold days only; gold days + corrected dry windows;
         gold days + literal dry windows (v1's Phase 5 labels).

Outputs (results/): experiment_dry_window_fix_table.csv, experiment_dry_window_fix_skill.csv,
experiment_dry_window_fix_reliability.csv
Run: nice -n 19 uv run python scripts/fusion/experiment_dry_window_fix.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import cube, engine, gold, validate
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.static import STRATUM_NAMES

log = logging.getLogger("dry_window_fix")
CODE_OF = {v: k for k, v in STRATUM_NAMES.items()}


def corrected_truth(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """(truth with ever-flooded cells NOTRUTH outside gold days, gold-day mask per day)."""
    days = pd.to_datetime(ds.day.values)
    gd = (days >= pd.Timestamp(ds.attrs["event_start"])) & (days <= pd.Timestamp(ds.attrs["event_end"]))
    t = ds.truth.values.copy()
    ever = (t[gd] == gold.FLOOD).any(axis=0)
    t[~gd] = np.where(ever[None, :, :], gold.NOTRUTH, t[~gd])
    return t, gd


def main() -> None:
    cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "experiment_dry_window_fix.log")), logging.StreamHandler(sys.stdout)])
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    split = pd.read_csv(cfg.results_dir / "event_split.csv").set_index("code").split
    acc_v1 = pd.read_csv(cfg.results_dir / "accuracy_table.csv")
    prior = pd.read_csv(cfg.results_dir / "prior_v1.csv")
    f = cfg["fusion"]
    thr, bins = f["prob_threshold"], cfg["validation"]["reliability_bins"]

    # 1. specificity with corrected dry windows, training events
    cnt = {}
    for _, r in sel[sel.code.map(split) == "train"].iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            t, _ = corrected_truth(ds)
            dom, st = ds.domain.values.astype(bool)[None], ds.stratum.values
            for s in MODEL_SOURCES:
                v = ds[s].values
                for code_st, name in STRATUM_NAMES.items():
                    m = dom & (st == code_st) & (v != NODATA) & (t == gold.DRY)
                    c = cnt.setdefault((s, name), [0, 0])
                    c[0] += int((m & (v == 0)).sum())
                    c[1] += int((m & (v == 1)).sum())
    acc = acc_v1.copy()
    acc["spec_v1"] = acc.spec
    acc["n_dry_corrected"] = np.nan
    for i, row in acc.iterrows():
        if row.method != "gold" or row.source not in MODEL_SOURCES:
            continue
        tn, fp = cnt[(row.source, row.stratum)]
        acc.loc[i, "spec"] = (tn + 0.5) / (tn + fp + 1)
        acc.loc[i, "n_dry_corrected"] = tn + fp
    acc[["source", "stratum", "method", "sens", "spec_v1", "spec", "n_dry_obs", "n_dry_corrected"]].to_csv(
        cfg.results_dir / "experiment_dry_window_fix_table.csv", index=False)
    log.info("specificity, v1 vs corrected dry windows:\n%s",
             acc[["source", "stratum", "method", "sens", "spec_v1", "spec"]].round(4).to_string(index=False))

    # 2. held-out: both models, same in-memory path
    clip = tuple(f["sens_spec_clip"])
    models = {"v1": engine.Model.from_tables(acc_v1, prior, MODEL_SOURCES, f["decay"], clip),
              "corrected_dry_windows": engine.Model.from_tables(acc, prior, MODEL_SOURCES, f["decay"], clip)}
    skill, rel = [], {}
    for n, (r, ew) in enumerate(validate.holdout_keys(cfg, sel), 1):
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        t_lit = ds.truth.values
        t_cor, gd = corrected_truth(ds)
        dom, st = ds.domain.values.astype(bool), ds.stratum.values
        obs = {s: ds[s].values for s in MODEL_SOURCES}
        for name, model in models.items():
            P = engine.run_sequence(obs, st, dom, model)["p_flood"].astype("float64")
            fin = np.isfinite(P) & dom[None]
            for code_st, st_name in STRATUM_NAMES.items():
                sm = fin & (st == code_st)
                g = sm & gd[:, None, None] & ((t_lit == 0) | (t_lit == 1))
                p, y = P[g] >= thr, t_lit[g] == 1
                skill.append({"model": name, "stratum": st_name, "tp": int((p & y).sum()), "fp": int((p & ~y).sum()),
                              "fn": int((~p & y).sum()), "tn": int((~p & ~y).sum())})
                for labels, tt, extra in (("gold days only", t_lit, gd[:, None, None]), ("corrected dry windows", t_cor, True),
                                          ("literal dry windows (v1)", t_lit, True)):
                    m = sm & extra & ((tt == 0) | (tt == 1))
                    if not m.any():
                        continue
                    b = np.clip((P[m] * bins).astype(int), 0, bins - 1)
                    a = rel.setdefault((name, labels, st_name), np.zeros((bins, 3)))
                    a[:, 0] += np.bincount(b, minlength=bins)
                    a[:, 1] += np.bincount(b, weights=P[m], minlength=bins)
                    a[:, 2] += np.bincount(b, weights=(tt[m] == 1), minlength=bins)
        if n % 20 == 0:
            log.info("held-out window %d done", n)

    sk = pd.DataFrame(skill).groupby(["model", "stratum"], as_index=False)[["tp", "fp", "fn", "tn"]].sum()
    allst = pd.DataFrame(skill).groupby("model", as_index=False)[["tp", "fp", "fn", "tn"]].sum().assign(stratum="all")
    sk = pd.concat([sk, allst], ignore_index=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        sk["recall"], sk["precision"] = sk.tp / (sk.tp + sk.fn), sk.tp / (sk.tp + sk.fp)
        sk["csi"] = sk.tp / (sk.tp + sk.fn + sk.fp)
    sk.to_csv(cfg.results_dir / "experiment_dry_window_fix_skill.csv", index=False)
    rows = []
    for (name, labels, st_name), a in rel.items():
        for i in range(bins):
            if a[i, 0]:
                rows.append({"model": name, "labels": labels, "stratum": st_name, "bin": i, "n": int(a[i, 0]),
                             "sum_p": a[i, 1], "n_flood": a[i, 2]})
    rl = pd.DataFrame(rows)
    summ = pd.concat([validate.summarise_reliability(g.drop(columns=["model", "labels"]), bins).assign(model=k[0], labels=k[1])
                      for k, g in rl.groupby(["model", "labels"])], ignore_index=True)
    summ.to_csv(cfg.results_dir / "experiment_dry_window_fix_reliability.csv", index=False)
    pd.set_option("display.width", 220)
    log.info("gold-day skill (P >= %.2f):\n%s", thr, sk[["model", "stratum", "tp", "fp", "fn", "recall", "precision", "csi"]].round(3).to_string(index=False))
    log.info("reliability:\n%s", summ[["model", "labels", "stratum", "n", "reliability_slope_wls", "calibration_slope_logistic", "mean_p", "obs_freq"]]
             .sort_values(["stratum", "labels", "model"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

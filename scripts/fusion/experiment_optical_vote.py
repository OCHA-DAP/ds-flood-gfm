"""Experiment (D-41): MODIS joins VIIRS in one optical vote (VIIRS OR MODIS), as FloodScan and GFDS form the
microwave vote (fixed decision 2). The two are the same sensor family (optical, cloud-blocked), so
they are merged, not added as independent votes (D-36).

Inputs: event cubes (data/fusion/cube/events/) and the per-window MODIS counts written by
modis_windows.py (data/fusion/modis/{window}.nc; --pull fetches them from dev blob). MODIS layer,
recurring-flood mapping and cell rule come from configs/fusion.yaml sources.modis.

  1. Training events: sensitivity and specificity (Jeffreys) of VIIRS, MODIS and the optical vote
     per stratum, under the literal dry windows (v1) and the corrected ones (D-39). Strata that v1
     estimated by EM keep their v1 VIIRS row for "optical", flagged.
  2. Held-out events: four models run in memory with the same code. Each uses
     {gfm, microwave, optical slot}: v1; v1 + optical vote; corrected labels; corrected labels +
     optical vote. Recall, precision and CSI on gold days (P >= prob_threshold), overall, per
     stratum, and split by whether the event's gold days fall in the VIIRS archive gap.

Outputs (results/): experiment_optical_vote_table.csv, experiment_optical_vote_skill.csv
Run: nice -n 19 uv run python scripts/fusion/experiment_optical_vote.py [--pull]
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import cube, engine, gold, validate
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.sources import NODATA, merge_microwave
from ds_flood_gfm.fusion.sources import modis
from ds_flood_gfm.fusion.static import STRATUM_NAMES

log = logging.getLogger("optical_vote")
GAP = (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-07-31"))
CODE_OF = {v: k for k, v in STRATUM_NAMES.items()}


def modis_values(cfg, window: str, n_days: int) -> np.ndarray:
    s = cfg["sources"]["modis"]
    p = cfg.local("modis", f"{window}.nc")
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run modis_windows.py (Databricks job flood_fusion_modis), then --pull")
    m = xr.load_dataset(p, engine="netcdf4")
    if m.sizes["day"] != n_days:
        raise ValueError(f"{window}: MODIS has {m.sizes['day']} days, the cube {n_days}")
    out = np.empty((n_days, m.sizes["lat"], m.sizes["lon"]), "uint8")
    for i in range(n_days):
        c = {k: m[f"{s['layer']}__{k}"].values[i].astype(float) for k in ("n_all", "n_obs", "n_flood", "n_recurring")}
        out[i], _ = modis.values_from_counts(c, s["recurring_as"], s["observed_min_frac"], s["flood_threshold"])
    return out


def pull(cfg) -> None:
    import ocha_stratus as stratus

    b = cfg["paths"]["blob"]
    cc = stratus.get_container_client(container_name=b["container"], stage=b["stage"])
    prefix = cfg.blob_path("processed", "modis")
    n = 0
    for blob in cc.list_blobs(name_starts_with=prefix):
        dest = cfg.local("modis", blob.name.rsplit("/", 1)[-1])
        if dest.exists() and dest.stat().st_size == blob.size:
            continue
        dest.write_bytes(cc.download_blob(blob.name).readall())
        n += 1
    log.info("pulled %d MODIS window files", n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--pull", action="store_true", help="download the MODIS window files from dev blob first")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "experiment_optical_vote.log")), logging.StreamHandler(sys.stdout)])
    logging.getLogger("azure").setLevel(logging.WARNING)
    if a.pull:
        pull(cfg)
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    split = pd.read_csv(cfg.results_dir / "event_split.csv").set_index("code").split
    acc_v1 = pd.read_csv(cfg.results_dir / "accuracy_table.csv")
    prior = pd.read_csv(cfg.results_dir / "prior_v1.csv")
    f = cfg["fusion"]
    thr, clip = f["prob_threshold"], tuple(f["sens_spec_clip"])

    def load(ew):
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        mo = modis_values(cfg, ew.key, ds.sizes["day"])
        return ds, mo, merge_microwave(ds.viirs.values, mo)

    # 1. accuracy on training events
    cnt = {}
    for _, r in sel[sel.code.map(split) == "train"].iterrows():
        for ew in gold.event_windows(r):
            ds, mo, opt = load(ew)
            t_cor, _ = gold.corrected_dry_windows(ds)
            dom, st = ds.domain.values.astype(bool)[None], ds.stratum.values
            for labels, t in (("literal", ds.truth.values), ("corrected", t_cor)):
                for name, v in (("viirs", ds.viirs.values), ("modis", mo), ("optical", opt)):
                    for code_st, st_name in STRATUM_NAMES.items():
                        m = dom & (st == code_st) & (v != NODATA)
                        c = cnt.setdefault((labels, name, st_name), np.zeros(4))
                        c += [(m & (t == 1) & (v == 1)).sum(), (m & (t == 1) & (v == 0)).sum(),
                              (m & (t == 0) & (v == 0)).sum(), (m & (t == 0) & (v == 1)).sum()]
    tab = pd.DataFrame([{"labels": k[0], "source": k[1], "stratum": k[2], "tp": v[0], "fn": v[1], "tn": v[2], "fp": v[3],
                         "sens": (v[0] + 0.5) / (v[0] + v[1] + 1), "spec": (v[2] + 0.5) / (v[2] + v[3] + 1)} for k, v in cnt.items()])
    tab.to_csv(cfg.results_dir / "experiment_optical_vote_table.csv", index=False)

    def model_table(labels: str, optical: bool) -> pd.DataFrame:
        a = acc_v1.copy()
        if labels == "corrected":  # D-39 specificities for gfm, microwave, viirs (gold rows only)
            fix = pd.read_csv(cfg.results_dir / "experiment_dry_window_fix_table.csv").set_index(["source", "stratum"])
            for i, row in a.iterrows():
                if row.method == "gold" and (row.source, row.stratum) in fix.index:
                    a.loc[i, "spec"] = float(fix.loc[(row.source, row.stratum), "spec"])
        if optical:
            o = a[a.source == "viirs"].copy()
            o["source"] = "optical"
            for i, row in o.iterrows():
                if row.method == "gold":
                    t = tab[(tab.labels == labels) & (tab.source == "optical") & (tab.stratum == row.stratum)]
                    o.loc[i, ["sens", "spec"]] = [float(t.sens.iloc[0]), float(t.spec.iloc[0])]
                else:
                    o.loc[i, "method"] = f"{row.method},copied_from_viirs"
            a = pd.concat([a[a.source != "viirs"], o], ignore_index=True)
        return a

    models = {}
    for labels in ("literal", "corrected"):
        for optical in (False, True):
            src = ["gfm", "microwave", "optical" if optical else "viirs"]
            name = ("v1" if labels == "literal" else "corrected dry windows") + (" + optical vote" if optical else "")
            models[name] = (engine.Model.from_tables(model_table(labels, optical), prior, src, f["decay"], clip), src)
    for name, (mdl, src) in models.items():
        log.info("model %s sources %s", name, src)

    # 2. held-out
    rows = []
    for n, (r, ew) in enumerate(validate.holdout_keys(cfg, sel), 1):
        ds, mo, opt = load(ew)
        obs = {"gfm": ds.gfm.values, "microwave": ds.microwave.values, "viirs": ds.viirs.values, "optical": opt}
        dom, st, t = ds.domain.values.astype(bool), ds.stratum.values, ds.truth.values
        days = pd.to_datetime(ds.day.values)
        gd = (days >= pd.Timestamp(ds.attrs["event_start"])) & (days <= pd.Timestamp(ds.attrs["event_end"]))
        in_gap = bool(((days[gd] >= GAP[0]) & (days[gd] <= GAP[1])).any())
        singles = {"VIIRS": ds.viirs.values == 1, "MODIS": mo == 1, "optical vote (VIIRS OR MODIS)": opt == 1}
        for name, (mdl, src) in models.items():
            P = engine.run_sequence({s: obs[s] for s in src}, st, dom, mdl)["p_flood"]
            singles[f"fusion: {name}"] = P >= thr
        for code_st, st_name in STRATUM_NAMES.items():
            g = dom[None] & (st == code_st) & gd[:, None, None] & ((t == 0) | (t == 1))
            if not g.any():
                continue
            y = t[g] == 1
            for name, pr in singles.items():
                p = pr[g]
                rows.append({"code": r.code, "in_viirs_gap": in_gap, "stratum": st_name, "product": name,
                             "tp": int((p & y).sum()), "fp": int((p & ~y).sum()), "fn": int((~p & y).sum())})
        if n % 20 == 0:
            log.info("held-out window %d done", n)
    sk = pd.DataFrame(rows)
    sk.to_csv(cfg.results_dir / "experiment_optical_vote_skill.csv", index=False)

    def summ(g):
        tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
        return pd.Series({"recall": tp / (tp + fn) if tp + fn else np.nan, "precision": tp / (tp + fp) if tp + fp else np.nan,
                          "csi": tp / (tp + fp + fn) if tp + fp + fn else np.nan, "flood_cells": tp + fn})
    pd.set_option("display.width", 220)
    for label, g in (("all held-out", sk), ("held-out events in the VIIRS gap", sk[sk.in_viirs_gap]),
                     ("held-out events outside the gap", sk[~sk.in_viirs_gap])):
        log.info("%s, gold days:\n%s", label, g.groupby("product").apply(summ, include_groups=False).sort_values("csi", ascending=False).round(3).to_string())
    log.info("per stratum (all held-out):\n%s", sk.groupby(["stratum", "product"]).apply(summ, include_groups=False).round(3).to_string())


if __name__ == "__main__":
    main()

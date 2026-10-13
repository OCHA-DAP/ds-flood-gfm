"""Diagnostic (D-39): how much of v1's measured over-confidence and false-alarm dependence comes from
the dry-window labels?

Fixed decision 5 (D-12, A-02) labels every gold-footprint cell dry for 30 days before the first
and 60 days after the last gold day. That includes cells flooded on the gold days, so receding (or
rising) flood water is scored as a false alarm by construction. This script measures it from data
already on disk:

  reliability  held-out events: fused P (daily COGs, band 1) vs truth, per stratum, day type
               (pre, gold days, post 1-30, post 31-60) and P bin
  highp        held-out events, dry-window cell-days only: how many have P >= 0.5 / 0.9, split by
               whether the cell was flooded on any gold day of its window
  dependence   training events, gold-dry cells: coinciding false alarms of each source pair (lift,
               within-event lift) per day type

Outputs: results/diagnostic_dry_windows_{reliability,highp,dependence}.csv
Run: nice -n 19 uv run python scripts/fusion/diagnostic_dry_windows.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from ds_flood_gfm.fusion import cube, gold, validate
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.static import STRATUM_NAMES

log = logging.getLogger("dry_windows")
ORDER = ["pre", "gold days", "post 1-30", "post 31-60"]


def day_types(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    days = pd.to_datetime(ds.day.values)
    s0, e0 = pd.Timestamp(ds.attrs["event_start"]), pd.Timestamp(ds.attrs["event_end"])
    after = (days - e0).days
    dt_ = np.where(days < s0, "pre", np.where(days <= e0, "gold days", np.where(after <= 30, "post 1-30", "post 31-60")))
    return dt_, (days >= s0) & (days <= e0)


def main() -> None:
    cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "diagnostic_dry_windows.log")), logging.StreamHandler(sys.stdout)])
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    split = pd.read_csv(cfg.results_dir / "event_split.csv").set_index("code").split
    bins = cfg["validation"]["reliability_bins"]

    # held-out: reliability and high-P dry-window cell-days, from the stored daily COGs
    rel, hip = {}, {}
    hk = list(validate.holdout_keys(cfg, sel))
    for n, (r, ew) in enumerate(hk, 1):
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        dtype, gd = day_types(ds)
        t, st, dom = ds.truth.values, ds.stratum.values, ds.domain.values.astype(bool)
        ever = (t[gd] == 1).any(axis=0)
        for k, d in enumerate(pd.to_datetime(ds.day.values)):
            p = cfg.local_root / "out" / "daily" / ew.key / f"{ew.key}_{d:%Y-%m-%d}.tif"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run make phase5 first")
            with rasterio.open(p) as src:
                P = src.read(1).astype("float64")
            lab = ((t[k] == 0) | (t[k] == 1)) & dom & np.isfinite(P)
            for code_st, st_name in STRATUM_NAMES.items():
                m = lab & (st[k] == code_st)
                if not m.any():
                    continue
                b = np.clip((P[m] * bins).astype(int), 0, bins - 1)
                y = t[k][m] == 1
                a = rel.setdefault((st_name, dtype[k]), np.zeros((bins, 3)))
                a[:, 0] += np.bincount(b, minlength=bins)
                a[:, 1] += np.bincount(b, weights=P[m], minlength=bins)
                a[:, 2] += np.bincount(b, weights=y, minlength=bins)
                if dtype[k] != "gold days":
                    for grp, gm in (("flooded on a gold day", ever), ("never flooded on a gold day", ~ever)):
                        mm = m & gm & (t[k] == 0)
                        h = hip.setdefault((st_name, dtype[k], grp), np.zeros(3))
                        h += [mm.sum(), (mm & (P >= 0.5)).sum(), (mm & (P >= 0.9)).sum()]
        if n % 20 == 0 or n == len(hk):
            log.info("held-out window %d/%d", n, len(hk))
    rel_rows = [{"stratum": s, "day_type": d, "bin": i, "n": int(a[i, 0]), "mean_p": a[i, 1] / a[i, 0] if a[i, 0] else np.nan,
                 "obs_freq": a[i, 2] / a[i, 0] if a[i, 0] else np.nan} for (s, d), a in rel.items() for i in range(bins)]
    pd.DataFrame(rel_rows).to_csv(cfg.results_dir / "diagnostic_dry_windows_reliability.csv", index=False)
    hp = pd.DataFrame([{"stratum": s, "day_type": d, "cells": g, "cell_days": int(v[0]), "p_ge_0.5": int(v[1]), "p_ge_0.9": int(v[2])}
                       for (s, d, g), v in hip.items()])
    hp.to_csv(cfg.results_dir / "diagnostic_dry_windows_highp.csv", index=False)

    # training: coinciding false alarms per day type, from the cubes
    pairs = [(a, b) for i, a in enumerate(MODEL_SOURCES) for b in MODEL_SOURCES[i + 1:]]
    dep = []
    tr = sel[sel.code.map(split) == "train"]
    for _, r in tr.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            dtype, _ = day_types(ds)
            t, st, dom = ds.truth.values, ds.stratum.values, ds.domain.values.astype(bool)
            o = {s: ds[s].values for s in MODEL_SOURCES}
            for code_st, st_name in STRATUM_NAMES.items():
                for dname in ORDER:
                    kk = dtype == dname
                    m = (t[kk] == 0) & dom[None] & (st[kk] == code_st)
                    if not m.any():
                        continue
                    for a, b in pairs:
                        both = m & (o[a][kk] != NODATA) & (o[b][kk] != NODATA)
                        if both.any():
                            fa, fb = both & (o[a][kk] == 1), both & (o[b][kk] == 1)
                            dep.append({"code": r.code, "stratum": st_name, "day_type": dname, "pair": f"{a}+{b}",
                                        "n": int(both.sum()), "a": int(fa.sum()), "b": int(fb.sum()), "ab": int((fa & fb).sum())})
    d = pd.DataFrame(dep)
    out = []
    for (s, dname, pair), g in d.groupby(["stratum", "day_type", "pair"]):
        N, A, B, AB = g.n.sum(), g.a.sum(), g.b.sum(), g.ab.sum()
        ev = g.groupby("code")[["n", "a", "b", "ab"]].sum()
        ew_ = (ev.a * ev.b / ev.n).sum()
        out.append({"stratum": s, "day_type": dname, "pair": pair, "dry_cells_both_observed": int(N), "joint_false_alarms": int(AB),
                    "rate_a": A / N, "rate_b": B / N, "lift": AB * N / (A * B) if A * B else np.nan,
                    "lift_within_event": AB / ew_ if ew_ else np.nan})
    dep_df = pd.DataFrame(out)
    dep_df["share_of_joint"] = dep_df.joint_false_alarms / dep_df.groupby(["stratum", "pair"]).joint_false_alarms.transform("sum")
    dep_df.to_csv(cfg.results_dir / "diagnostic_dry_windows_dependence.csv", index=False)

    pd.set_option("display.width", 220)
    r = pd.DataFrame(rel_rows)
    summ = r.groupby(["stratum", "day_type"]).apply(lambda g: pd.Series({
        "cell_days": g.n.sum(), "mean_p": (g.mean_p * g.n).sum() / g.n.sum(), "obs_freq": (g.obs_freq * g.n).sum() / g.n.sum(),
        "share_p_ge_0.5": g[g.bin >= bins // 2].n.sum() / g.n.sum()}), include_groups=False)
    log.info("held-out reliability by day type:\n%s", summ.round(4).to_string())
    log.info("held-out dry-window high-P cell-days:\n%s", hp.to_string(index=False))
    log.info("training coinciding false alarms by day type (open, no snow):\n%s",
             dep_df[dep_df.stratum == "open_nosnow"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

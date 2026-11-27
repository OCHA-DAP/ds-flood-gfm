"""Ordinary-day false alarms on labelled cells far from the flood (option 1 of the ordinary-day work, 2026-10-26).

The expert maps examine areas much larger than the flood and label the rest dry. Those cells, long after the
event and away from the flooded area, are the closest thing we have to labelled ordinary days. Rules, fixed
before running:
  - days: +31..+60 after the last map (the quietest labelled period; see Ordinary days);
  - cells: labelled dry under the adopted rule, floodable, and at least 5 cells (about 5 km) from any cell
    flooded on a map day of the window (so draining water next to the flood is left out);
  - fusion with each fold's model (as built, and with FloodScan alone in the microwave slot); cascade; products;
  - false-alarm rate = flood calls / those cells (all cells, no data = not flagged), per land type and
    normally-wet status; implied ordinary-day precision from it with the sample base rate (a formula);
  - also the same on cells within 5 cells of the flood, so the two can be compared.
Still inside mapped events in the same places; unmapped water counts as dry. An upper bound on false alarms.

Outputs: results/cv/far_from_flood.csv. Run: nice -n 19 uv run python -u scripts/fusion/cv_far_from_flood.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation

from ds_flood_gfm.fusion import baseline, crossval, cube, engine, gold
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.validate import SINGLE

CASCADE = "cascade: VIIRS, else GFM, else microwave"
FS_ONLY = "fusion, FloodScan alone"
DIST = 5


def main() -> None:
    cfg = load_config()
    f, thr, rule = cfg["fusion"], cfg["fusion"]["prob_threshold"], cfg["gold"]["dry_window_rule"]
    res = cfg.results_dir / "cv"
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    ev = ev[ev.code.isin(folds.code)].merge(folds[["code", "fold"]], on="code").reset_index(drop=True)
    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = counts[counts.rule == rule].drop(columns="rule")
    comp = pd.read_parquet(cfg.local("metrics", "cv_fit_counts_components.parquet"))
    counts_fs = pd.concat([counts, comp[comp.source == "floodscan"]], ignore_index=True)
    prior = pd.read_csv(res / "prior.csv")
    clip = tuple(f["sens_spec_clip"])
    K = sorted(ev.fold.unique())
    models = {("fusion", k): engine.Model.from_tables(crossval.fit_table(counts, ev.code[ev.fold != k]), prior, MODEL_SOURCES, f["decay"], clip) for k in K}
    fs_srcs = ["gfm", "floodscan", "viirs"]
    models |= {(FS_ONLY, k): engine.Model.from_tables(crossval.fit_table(counts_fs, ev.code[ev.fold != k], sources=fs_srcs), prior, fs_srcs, f["decay"], clip) for k in K}
    rows, t0 = [], time.time()
    struct = np.ones((2 * DIST + 1, 2 * DIST + 1), bool)
    for _, r in ev.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            days = pd.to_datetime(ds.day.values)
            truth = crossval.dry_truth(ds, rule)
            gd = crossval.gold_day_mask(ds)
            end = pd.Timestamp(ds.attrs["event_end"])
            dom = ds.domain.values.astype(bool)
            lt = crossval.landtype_codes(ds.stratum.values)
            fp = (ds.truth.values[gd] == gold.FLOOD).any(axis=0)
            near = binary_dilation(fp, structure=struct)
            v = {s: ds[s].values for s in SINGLE}
            p = {"fusion": engine.run_sequence({s: v[s] for s in MODEL_SOURCES}, ds.stratum.values, dom, models[("fusion", r.fold)])["p_flood"],
                 FS_ONLY: engine.run_sequence({s: v[s] for s in fs_srcs}, ds.stratum.values, dom, models[(FS_ONLY, r.fold)])["p_flood"]}
            cas = np.full(v["viirs"].shape, NODATA, "uint8")
            for s in ["microwave", "gfm", "viirs"]:
                cas = np.where(v[s] != NODATA, v[s], cas)
            calls = {"fusion": np.isfinite(p["fusion"]) & (p["fusion"] >= thr), FS_ONLY: np.isfinite(p[FS_ONLY]) & (p[FS_ONLY] >= thr), CASCADE: cas == 1} | {s: v[s] == 1 for s in SINGLE}
            late = np.asarray((days > end) & ((days - end).days > 30))
            nw_cache = {}
            for i in np.where(late)[0]:
                dry = (truth[i] == gold.DRY) & dom & (lt[i] != NODATA)
                if not dry.any():
                    continue
                m = days[i].month
                if m not in nw_cache:
                    nw_cache[m] = baseline.normally_wet(cfg, ew.window, m)
                nw = nw_cache[m]
                for zone, zm in (("far (>5 cells from the flood)", ~near), ("near (within 5 cells)", near)):
                    for k, ltn in crossval.LANDTYPES.items():
                        for wet, wm in (("normally wet", nw), ("other", ~nw)):
                            mm = dry & zm & (lt[i] == k) & wm
                            if not mm.any():
                                continue
                            for name, c in calls.items():
                                rows.append({"code": r.code, "zone": zone, "landtype": ltn, "wet": wet, "product": name, "calls": int((c[i] & mm).sum()), "dry": int(mm.sum())})
        print(f"{r.code} done, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(cfg.local("metrics", "far_from_flood_rows.parquet"))
    pr = prior.drop_duplicates("landtype").set_index("landtype").base_rate
    sk = pd.read_csv(res / "skill_by_landtype.csv")
    sk = sk[sk["mode"] == "all"].set_index(["product", "landtype"]).recall
    mc = pd.read_csv(res / "microwave_components_skill.csv")
    rec_fs = mc[(mc.group == "landtype") & (mc["product"] == "fusion with FloodScan only")].set_index("value").recall
    out = []
    for by in (["zone"], ["zone", "landtype"], ["zone", "wet"], ["zone", "landtype", "wet"]):
        g = df.groupby(["product", *by], as_index=False)[["calls", "dry"]].sum()
        g["false_alarm_rate"] = g.calls / g.dry
        g["view"] = ",".join(by)
        if "landtype" in by:
            def implied(row):
                rec = rec_fs.get(row.landtype) if row["product"] == FS_ONLY else sk.get((row["product"], row.landtype))
                b = pr[row.landtype]
                return rec * b / (rec * b + row.false_alarm_rate * (1 - b)) if rec is not None and row.false_alarm_rate > 0 else np.nan
            g["implied_ordinary_day_precision"] = g.apply(implied, axis=1)
        out.append(g)
    out = pd.concat(out, ignore_index=True)
    out.to_csv(res / "far_from_flood.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[out.view == "zone"].pivot(index="product", columns="zone", values="false_alarm_rate").round(4).to_string(), "\n")
    o = out[(out.view == "zone,landtype") & (out.landtype == "open")]
    print(o[["product", "zone", "dry", "false_alarm_rate", "implied_ordinary_day_precision"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

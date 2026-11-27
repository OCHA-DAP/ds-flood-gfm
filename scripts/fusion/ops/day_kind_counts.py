"""Counts for the doc's 'kinds of day' table over the 170 fold events (adopted dry-window rule, floodable cells)."""
import numpy as np, pandas as pd, xarray as xr
from ds_flood_gfm.fusion import cube, gold, crossval
from ds_flood_gfm.fusion.config import load_config
cfg = load_config()
folds = pd.read_csv("results/event_folds.csv")
ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet")); ev = ev[ev.code.isin(folds.code)]
kept = pd.read_parquet(cfg.local("gold", "label_sets_kept.parquet")); kept = kept[kept.code.isin(folds.code)]
acc = {k: [0, 0, 0] for k in ("before", "map", "after")}  # window-days, flooded, dry
nwin = empty = 0; lengths = []
for _, r in ev.iterrows():
    for ew in gold.event_windows(r):
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        t = crossval.dry_truth(ds, "corrected"); dom = ds.domain.values.astype(bool)[None]
        gd = crossval.gold_day_mask(ds); days = pd.to_datetime(ds.day.values)
        start = pd.Timestamp(ds.attrs["event_start"])
        lab = ((t == 0) | (t == 1)) & dom
        nwin += 1; lengths.append(len(days))
        if not lab.any():
            empty += 1; continue
        anyl = lab.reshape(len(days), -1).any(axis=1)
        for name, sel in (("before", np.asarray(days < start)), ("map", gd & anyl), ("after", (~gd) & np.asarray(days > start))):
            acc[name][0] += int(sel.sum())
            acc[name][1] += int(((t == 1) & lab)[sel].sum()); acc[name][2] += int(((t == 0) & lab)[sel].sum())
mapdays = kept.groupby("code").label_day.apply(lambda s: s.dt.normalize().nunique())
print("windows", nwin, "without labelled floodable cell", empty)
print("gold maps (label sets)", len(kept), "map days", int(mapdays.sum()))
print("events by map days:", mapdays.value_counts().sort_index().head(4).to_dict(), "max", int(mapdays.max()))
L = np.array(lengths); print("window days min", L.min(), "median", int(np.median(L)), "p10-p90", np.percentile(L, [10, 90]), "max", L.max())
for k, v in acc.items():
    print(k, f"{v[0]:,} window-days; {v[1]:,} flooded, {v[2]:,} dry cell-days")

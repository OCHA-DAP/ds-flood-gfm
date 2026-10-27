"""A best-available cascade baseline, and four fixed scoring views (colleague's suggestions A and E, 2026-10-16).

A. Cascade baseline, no learning. Each cell-day takes the first product that observed it:
   VIIRS, else GFM, else the microwave vote (the colleague's order, the pre-registered baseline).
   "GFM first" (GFM, else VIIRS, else microwave) is reported only as a sensitivity: the order is a choice,
   and picking the better order on these events would tune on the scored data.
E. Scoring views, all fixed before any of them was computed for the cascade, reported side by side for
   fusion (cross-validated, as built), both cascades and every product. The pooled score stays the reference:
   1. pooled: counts summed over cells, days and events (as in make crossval);
   2. basin-weighted: each HydroBASINS level-4 basin's counts scaled by 1 / its gold flooded cell-days, so
      each basin's flood counts once; basins with fewer than 50 gold flooded cell-days are left out (counted);
      intervals resample basins;
   3. by the gold map's sensor on that day: Sentinel-1 (every label set that day), other radar, optical
      (any optical class), or mixed / unknown;
   4. normally-wet cells as ambiguous: cells normally wet in that month (D-38 rule) are not scored at all.
   Gold-map days, all cells (no data = not flagged), as everywhere else.
Also: the ADR 0007 event-level rules for the two cascades and fusion, and paired intervals for fusion minus
the cascade in every view (the same resampled events, or basins, for both).

Outputs, results/cv/: baselines_views_skill.csv, baselines_views_diff.csv, baselines_event_level.csv.
Run: nice -n 19 uv run python scripts/fusion/cv_baselines_and_views.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import baseline, crossval, cube, engine, gold
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA

CASCADE = "cascade: VIIRS, else GFM, else microwave"
CASCADE_GFM = "cascade: GFM, else VIIRS, else microwave (sensitivity)"
PRODUCTS = ["fusion", CASCADE, CASCADE_GFM, "viirs", "gfm", "microwave", "floodscan", "gfds"]
MIN_BASIN_FLOOD = 50
OPTICAL = {"optical", "optical_hr", "optical_vhr"}


def cascade(order: list[np.ndarray]) -> np.ndarray:
    """First observed value in `order` (arrays of 0 / 1 / 255); 255 where none observed."""
    out = np.full(order[0].shape, NODATA, "uint8")
    for v in reversed(order):
        out = np.where(v != NODATA, v, out)
    return out


def day_classes(cfg) -> dict:
    kept = pd.read_parquet(cfg.local("gold", "label_sets_kept.parquet"))
    out = {}
    for (code, day), g in kept.groupby(["code", kept.label_day.dt.normalize()]):
        if g.is_s1.all():
            c = "Sentinel-1"
        elif (g.sensor_class == "sar").all():
            c = "other radar"
        elif g.sensor_class.isin(OPTICAL).all():
            c = "optical"
        else:
            c = "mixed or unknown"
        out[(code, day)] = c
    return out


def f1(m):
    return 2 * m[..., 0] / (2 * m[..., 0] + m[..., 1] + m[..., 2])


def summarise(df: pd.DataFrame, unit: str, weight: str | None, reps: int, seed: int) -> pd.DataFrame:
    """Per product: recall, precision, F1 from (weighted) sums, interval resampling `unit`; plus the paired
    difference fusion minus the cascade."""
    rng = np.random.default_rng(seed)
    w = df[weight].to_numpy() if weight else np.ones(len(df))
    d = df.assign(tp=df.tp * w, fp=df.fp * w, fn=df.fn * w)
    e = d.groupby([unit, "product"])[["tp", "fp", "fn"]].sum().unstack("product").fillna(0)
    units = e.index.to_numpy()
    idx = rng.integers(0, len(units), (reps, len(units)))
    rows, mats = [], {}
    for p in PRODUCTS:
        if ("tp", p) not in e.columns:
            continue
        m = np.stack([e[("tp", p)], e[("fp", p)], e[("fn", p)]], axis=1).astype(float)
        mats[p] = m
        s = m.sum(0)
        b = f1(m[idx].sum(1))
        rows.append({"product": p, "units": len(units), "recall": s[0] / (s[0] + s[2]), "precision": s[0] / (s[0] + s[1]),
                     "f1": float(f1(s)), "f1_lo": float(np.nanquantile(b, 0.025)), "f1_hi": float(np.nanquantile(b, 0.975))})
    out = pd.DataFrame(rows)
    dd = f1(mats["fusion"][idx].sum(1)) - f1(mats[CASCADE][idx].sum(1))
    out.attrs["diff"] = {"fusion_minus_cascade": float(f1(mats["fusion"].sum(0)) - f1(mats[CASCADE].sum(0))),
                         "lo": float(np.nanquantile(dd, 0.025)), "hi": float(np.nanquantile(dd, 0.975))}
    return out


def main() -> None:
    cfg = load_config()
    f, thr, e = cfg["fusion"], cfg["fusion"]["prob_threshold"], cfg["event_level"]
    reps, rule = cfg["accuracy"]["bootstrap_reps"], cfg["gold"]["dry_window_rule"]
    res = cfg.results_dir / "cv"
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    ev = ev[ev.code.isin(folds.code)].merge(folds[["code", "fold", "basin_id"]], on="code").reset_index(drop=True)
    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = counts[counts.rule == rule]
    prior = pd.read_csv(res / "prior.csv")
    models = {k: engine.Model.from_tables(crossval.fit_table(counts, ev.code[ev.fold != k]), prior, MODEL_SOURCES,
                                          f["decay"], tuple(f["sens_spec_clip"])) for k in sorted(ev.fold.unique())}
    dclass = day_classes(cfg)

    rows, flags, sizes = [], [], defaultdict(int)
    for _, r in ev.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            days = pd.to_datetime(ds.day.values)
            gd = crossval.gold_day_mask(ds)
            t, dom = ds.truth.values, ds.domain.values.astype(bool)
            lt = crossval.landtype_codes(ds.stratum.values)
            p = engine.run_sequence({s: ds[s].values for s in MODEL_SOURCES}, ds.stratum.values, dom, models[r.fold])["p_flood"]
            v = {s: ds[s].values for s in ["viirs", "gfm", "microwave", "floodscan", "gfds"]}
            calls = {"fusion": np.isfinite(p) & (p >= thr),
                     CASCADE: cascade([v["viirs"], v["gfm"], v["microwave"]]) == 1,
                     CASCADE_GFM: cascade([v["gfm"], v["viirs"], v["microwave"]]) == 1}
            calls |= {s: v[s] == 1 for s in ["viirs", "gfm", "microwave", "floodscan", "gfds"]}
            for i in np.where(gd)[0]:
                lab = ((t[i] == gold.DRY) | (t[i] == gold.FLOOD)) & np.isfinite(p[i])
                if not lab.any():
                    continue
                nw = baseline.normally_wet(cfg, ew.window, days[i].month)
                base = {"code": r.code, "basin": r.basin_id, "continent": r.continent,
                        "day_class": dclass.get((r.code, days[i].normalize()), "mixed or unknown")}
                for g, gm in (("normally wet", nw), ("other", ~nw)):
                    for k, ltn in crossval.LANDTYPES.items():
                        m = lab & gm & (lt[i] == k)
                        if not m.any():
                            continue
                        tf = t[i][m] == gold.FLOOD
                        for prod, c in calls.items():
                            cc = c[i][m]
                            rows.append(base | {"group": g, "landtype": ltn, "product": prod, "tp": int((cc & tf).sum()),
                                                "fp": int((cc & ~tf).sum()), "fn": int((~cc & tf).sum())})
            ff, sz = crossval.footprint_flags(ds, {k: calls[k] for k in ["fusion", CASCADE, CASCADE_GFM]})
            if len(ff):
                flags.append(ff[ff.landtype == "all"].assign(code=r.code))
            sizes[r.code] += sz.get("all", 0)

    sk = pd.DataFrame(rows).groupby(["code", "basin", "continent", "day_class", "group", "landtype", "product"], as_index=False)[["tp", "fp", "fn"]].sum()
    flood_by_basin = sk[sk["product"] == "fusion"].assign(fl=lambda d: d.tp + d.fn).groupby("basin").fl.sum()
    keep_basins = set(flood_by_basin[flood_by_basin >= MIN_BASIN_FLOOD].index)
    tabs, diffs = [], []

    def add(view, df, unit="code", weight=None, seed=20261016):
        s = summarise(df, unit, weight, reps, seed)
        tabs.append(s.assign(view=view))
        diffs.append({"view": view, **s.attrs["diff"], "units": int(s.units.iloc[0])})

    add("pooled", sk)
    bw = sk[sk.basin.isin(keep_basins)]
    bw = bw.assign(w=bw.basin.map(1.0 / flood_by_basin))
    add(f"basin-weighted ({len(keep_basins)} basins with >= {MIN_BASIN_FLOOD} flooded cell-days; "
        f"{len(flood_by_basin) - len(keep_basins)} left out)", bw, unit="basin", weight="w")
    for c in ["Sentinel-1", "other radar", "optical", "mixed or unknown"]:
        add(f"gold map from {c}", sk[sk.day_class == c])
    add("normally-wet cells not scored", sk[sk.group == "other"])
    for c in ["Africa", "Asia", "Oceania", "South America"]:
        add(f"{c}, pooled", sk[sk.continent == c])
    for ltn in crossval.LANDTYPES.values():
        add(f"{ltn} land, pooled", sk[sk.landtype == ltn])
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(res / "baselines_views_skill.csv", index=False)
    pd.DataFrame(diffs).to_csv(res / "baselines_views_diff.csv", index=False)

    fl = pd.concat(flags, ignore_index=True).groupby(["code", "product", "day"], as_index=False).flagged.sum()
    info = ev.set_index("code")
    erows = []
    for (code, prod), g in fl.groupby(["code", "product"]):
        o = crossval.event_level(g.set_index("day").flagged, sizes[code], info.loc[code, "start"], info.loc[code, "end"],
                                 e["min_share"], e["min_cells"], e["days_around_maps"], tuple(e["false_event_days"]))
        erows.append({"code": code, "product": prod, **o})
    el = pd.DataFrame(erows)
    el = el[el.eligible.astype(bool)]
    el["caught_clean"] = el.caught.astype(bool) & ~el.false_event.astype(bool)
    es = el.groupby("product").agg(events=("code", "nunique"), caught=("caught", "mean"), false_event=("false_event", "mean"),
                                   caught_clean=("caught_clean", "mean"), median_days=("days_to_detection", "median")).reset_index()
    es.to_csv(res / "baselines_event_level.csv", index=False)

    pd.set_option("display.width", 230)
    gold_flood = sk[sk["product"] == "fusion"].assign(fl=lambda d: d.tp + d.fn)
    print("gold flooded cell-days by map-day sensor:", gold_flood.groupby("day_class").fl.sum().to_dict())
    print(tab[["view", "product", "units", "recall", "precision", "f1", "f1_lo", "f1_hi"]].round(3).to_string(index=False), "\n")
    print(pd.DataFrame(diffs).round(3).to_string(index=False), "\n")
    print(es.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

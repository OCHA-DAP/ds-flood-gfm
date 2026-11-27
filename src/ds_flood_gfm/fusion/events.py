"""Event selection for v1 and the held-out split (fixed decision 7).

Selection is cost-ordered within continent quotas (small events first), with
hard caps on label-partition size and window cells; every event left out is
listed with its reason. The split holds out `holdout_frac` of events,
stratified by continent, with all events in the same HydroBASINS level-4
basin kept on the same side: no basin contributes to both training and
validation.
"""

from __future__ import annotations

import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ds_flood_gfm.fusion.config import Config

# continent -> max events in v1 (cost-ordered); None = take all that pass the caps
CONTINENT_QUOTA = {
    "Africa": 45,
    "Asia": 45,
    "Europe": 50,
    "North America": None,
    "South America": None,
    "Oceania": None,
}
MAX_CELLS = 80_000


def select_events(ev: pd.DataFrame, cfg: Config, quota: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    quota = quota or CONTINENT_QUOTA
    g = cfg["gold"]
    ev = ev.copy()
    ev["n_window_days"] = (ev.window_end - ev.window_start).dt.days + 1
    ev["cost"] = ev.cells * ev.n_window_days
    reason = pd.Series("", index=ev.index)
    reason[ev.label_bytes > g["max_label_bytes"]] = f"label partition > {g['max_label_bytes'] / 1e6:.0f} MB"
    reason[(reason == "") & (ev.cells > MAX_CELLS)] = f"windows > {MAX_CELLS} cells"
    reason[(reason == "") & ~ev.continent.isin(list(quota))] = "continent not in quota table"
    ok = ev[reason == ""].sort_values("cost")
    chosen = []
    for cont, q in quota.items():
        c = ok[ok.continent == cont]
        chosen.append(c if q is None else c.head(q))
    sel = pd.concat(chosen)
    over = ok.index.difference(sel.index)
    reason[over] = "beyond continent quota (cost-ordered)"
    excluded = ev.loc[reason != ""].assign(reason=reason[reason != ""])
    return sel.sort_values("cost").reset_index(drop=True), excluded.reset_index(drop=True)


def snow_supplement(ev: pd.DataFrame, kept_sets: pd.DataFrame, cfg: Config, already: set[str], min_snow_frac: float = 0.05) -> pd.DataFrame:
    """Usable events whose windows are >= min_snow_frac snow season in a gold-day month (D-14b).

    The cost-ordered quotas leave the two snow strata with almost no gold flood cells,
    so every usable snow-season event within the size caps is added.
    """
    from ds_flood_gfm.fusion import grid, static

    snow = static.SnowClimatology(cfg)
    g = cfg["gold"]
    rows = []
    for code, sets in kept_sets.groupby("code"):
        if code in already:
            continue
        e = ev[ev.code == code]
        if e.empty or e.iloc[0].cells > MAX_CELLS or e.iloc[0].label_bytes > g["max_label_bytes"]:
            continue
        w = grid.window_from_bounds(sets.minx.min(), sets.miny.min(), sets.maxx.max(), sets.maxy.max())
        frac = max(snow.snow_season(int(m), w).mean() for m in set(sets.label_day.dt.month))
        if frac >= min_snow_frac:
            rows.append(e.iloc[0].to_dict() | {"snow_frac": float(frac)})
    out = pd.DataFrame(rows)
    if len(out):
        out["n_window_days"] = (out.window_end - out.window_start).dt.days + 1
        out["cost"] = out.cells * out.n_window_days
    return out


def assign_basins(ev: pd.DataFrame, cfg: Config) -> pd.Series:
    lvl = cfg["gold"]["basins_level"]
    files = sorted(glob.glob(str(cfg.local("raw", "hydrobasins", "x").parent / f"hybas_*_lev{lvl:02d}_v1c.zip")))
    if not files:
        raise FileNotFoundError(f"no HydroBASINS level {lvl} files under data/fusion/raw/hydrobasins")
    basins = pd.concat([gpd.read_file(f)[["HYBAS_ID", "geometry"]] for f in files], ignore_index=True)
    basins = gpd.GeoDataFrame(basins, crs="EPSG:4326")
    pts = gpd.GeoDataFrame({"code": ev.code.values}, geometry=[shapely.Point(x, y) for x, y in zip(ev.lon, ev.lat)], crs="EPSG:4326")
    j = gpd.sjoin_nearest(pts.to_crs(8857), basins.to_crs(8857), how="left").drop_duplicates("code")
    return pd.Series(j.HYBAS_ID.values, index=ev.index, name="basin_id").astype("int64")


def split_events(ev: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add `basin_id` and `split` in {train, holdout}. Deterministic given the seed."""
    g = cfg["gold"]
    rng = np.random.default_rng(g["split_seed"])
    ev = ev.copy()
    ev["basin_id"] = assign_basins(ev, cfg)
    ev["split"] = "train"
    for cont, grp in ev.groupby("continent"):
        target = int(round(g["holdout_frac"] * len(grp)))
        basins = grp.groupby("basin_id").size()
        order = rng.permutation(basins.index.values)
        taken = 0
        for b in order:
            n = int(basins[b])
            if taken >= target:
                break
            # take a basin if it does not overshoot the target by more than it undershoots
            if taken + n - target <= target - taken:
                ev.loc[(ev.continent == cont) & (ev.basin_id == b), "split"] = "holdout"
                taken += n
    leak = ev.groupby("basin_id").split.nunique()
    if (leak > 1).any():
        raise AssertionError(f"basins on both sides of the split: {leak[leak > 1].index.tolist()}")
    return ev


def assign_folds(ev: pd.DataFrame, n_folds: int, seed: int, big_share: float) -> pd.Series:
    """Cross-validation fold (0 .. n_folds-1) per event; all events of a basin share a fold (D-43).

    Needs columns code, continent, basin_id and flood (gold flooded cell-days). Basins are placed one
    at a time, most flooded first:
      * a basin holding at least `big_share` of all gold flood goes to the fold with the least flood so
        far, so the few events that dominate the flooded area are spread across folds;
      * any other basin goes to the fold with the fewest events of its continent(s), then the fewest
        events, then the lowest number.
    A seeded shuffle orders basins that tie, so the result is fixed by the seed.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    missing = [c for c in ("code", "continent", "basin_id", "flood") if c not in ev]
    if missing:
        raise KeyError(f"assign_folds needs columns {missing}")
    if ev[["continent", "basin_id", "flood"]].isna().any().any() or (ev.flood < 0).any():
        raise ValueError("every event needs a continent, a basin_id and a non-negative flood count")
    rng = np.random.default_rng(seed)
    b = ev.groupby("basin_id").agg(n=("code", "size"), flood=("flood", "sum"))
    b["tiebreak"] = rng.permutation(len(b))
    b = b.sort_values(["flood", "tiebreak"], ascending=[False, True])
    big = b.flood >= big_share * b.flood.sum()
    load, n_all = [0.0] * n_folds, [0] * n_folds
    n_cont = [dict.fromkeys(ev.continent.unique(), 0) for _ in range(n_folds)]
    fold_of = {}
    for basin, row in b.iterrows():
        conts = ev.loc[ev.basin_id == basin, "continent"].value_counts()
        if big[basin]:
            f = min(range(n_folds), key=lambda k: (load[k], k))
        else:
            f = min(range(n_folds), key=lambda k: (sum(n_cont[k][c] for c in conts.index), n_all[k], k))
        fold_of[basin] = f
        load[f] += float(row.flood)
        n_all[f] += int(row.n)
        for c, k in conts.items():
            n_cont[f][c] += int(k)
    return ev.basin_id.map(fold_of).astype(int).rename("fold")

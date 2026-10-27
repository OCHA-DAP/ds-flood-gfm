"""Assign every in-scope event to a cross-validation fold (D-43) and report how balanced the folds are.

Events: the v1 selection in `scope.continents` plus the two build lists from `phase1_feasibility.py extra`
(D-42). Each event's basin comes from events.assign_basins (HydroBASINS, the same rule as the v1 split), and
its gold flooded cell-days, per land type, from its stage-A gold files. Every event must be built: a missing
gold file raises and names the events, so folds are never assigned over a partial set.

Writes results/event_folds.csv (code, source, continent, basin_id, flood_cell_days, fold), frozen like the
split: if it exists and differs, this refuses unless --overwrite. Balance report: results/event_folds_balance.csv.

Run: uv run python scripts/fusion/make_folds.py [--overwrite]
"""

from __future__ import annotations

import argparse

import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import cube, events, gold
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.static import LANDCOVER_NAMES


def in_scope_events(cfg) -> pd.DataFrame:
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    extra = [pd.read_parquet(cfg.local("gold", f"events_extra_{n}.parquet")).code for n in ("small", "large")]
    codes = set(sel[sel.continent.isin(cfg["scope"]["continents"])].code).union(*map(set, extra))
    return ev[ev.code.isin(codes)].reset_index(drop=True)


def flood_by_landcover(cfg, row: pd.Series) -> dict[str, int]:
    out = dict.fromkeys(LANDCOVER_NAMES.values(), 0)
    for ew in gold.event_windows(row):
        ds = xr.load_dataset(cube.gold_path(cfg, ew.key), engine="netcdf4")
        flooded = (ds.truth.values == gold.FLOOD).sum(axis=0)
        for k, name in LANDCOVER_NAMES.items():
            out[name] += int(flooded[ds.landcover.values == k].sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing, different results/event_folds.csv")
    a = ap.parse_args()
    cfg = load_config(a.config)
    g = cfg["gold"]
    ev = in_scope_events(cfg)
    unbuilt = sorted({r.code for _, r in ev.iterrows() for ew in gold.event_windows(r) if not cube.gold_path(cfg, ew.key).exists()})
    if unbuilt:
        raise FileNotFoundError(f"{len(unbuilt)} in-scope events have no stage-A gold files yet (build them first): {unbuilt}")
    ev["basin_id"] = events.assign_basins(ev, cfg)
    fl = pd.DataFrame([flood_by_landcover(cfg, r) for _, r in ev.iterrows()], index=ev.index)
    ev["flood"] = fl.sum(axis=1)
    ev["fold"] = events.assign_folds(ev, g["n_folds"], g["fold_seed"], g["fold_big_basin_share"])
    if (ev.groupby("basin_id").fold.nunique() > 1).any():
        raise AssertionError("a basin landed in two folds")
    out = ev[["code", "source", "continent", "basin_id", "flood", "fold"]].rename(columns={"flood": "flood_cell_days"})
    out = out.sort_values(["fold", "continent", "code"]).reset_index(drop=True)
    path = cfg.results_dir / "event_folds.csv"
    if path.exists() and not a.overwrite:
        if not pd.read_csv(path).equals(out):
            raise FileExistsError(f"{path} exists and differs from this assignment; pass --overwrite to replace it")
        print(f"{path} unchanged")
    else:
        out.to_csv(path, index=False)
        print(f"wrote {path}: {len(out)} events, {out.basin_id.nunique()} basins, {g['n_folds']} folds")

    by_cont = pd.crosstab(ev.fold, ev.continent)
    share = fl.groupby(ev.fold).sum()
    share = (share / share.sum()).round(3).add_prefix("flood_share_")
    bal = by_cont.add_prefix("events_").join(share)
    bal.insert(0, "events", by_cont.sum(axis=1))
    bal.rename_axis("fold").to_csv(cfg.results_dir / "event_folds_balance.csv")
    pd.set_option("display.width", 200)
    print(bal.to_string())
    print("largest flood share in one fold, per land type:", share.max().round(2).to_dict(), "(even split:", round(1 / g["n_folds"], 2), ")")


if __name__ == "__main__":
    main()

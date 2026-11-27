"""Experiment (D-37): score NASA's MODIS flood product (MCDWD) against gold, beside VIIRS.

For each selected event window and each gold day (event_start..event_end with labelled cells),
read the MCDWD tiles covering the window, count pixels per 0.01-degree cell for every composite,
and score against the cube's truth. Variants: composite (1-day, 1-day cloud-shadow screened,
3-day) x recurring (seasonal) flood counted as flood or not. VIIRS (from the cube) is scored on
the same cells and days, and both on the cells where both observed.

Modes per score row:
  all          every labelled domain cell; a product's NoData counts as a miss
  conditional  only cells the product observed
  both         only cells MODIS (that variant) and VIIRS both observed

Output: results/experiment_modis_gold.csv (tp/fp/fn/tn per event x stratum x product x mode).
Per window-day pixel counts are cached (data/fusion/raw/modis_counts/), so a rerun downloads nothing.
Needs the Earthdata token in the environment (configs/fusion.yaml sources.modis.token_env).

Run: nice -n 19 uv run python scripts/fusion/experiment_modis_gold.py [--set gap|all] [--plan] [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import cube, gold
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.sources import modis
from ds_flood_gfm.fusion.static import STRATUM_NAMES

log = logging.getLogger("modis_gold")
GAP = (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-07-31"))  # no NOAA VIIRS files on AWS (docs/inventory.md)
VARIANTS = [(layer, rec) for layer in ("Flood_1Day_250m", "FloodCS_1Day_250m", "Flood_3Day_250m") for rec in ("flood", "dry")]


def gold_days(ds: xr.Dataset) -> list[int]:
    days = pd.to_datetime(ds.day.values)
    t = ds.truth.values
    in_event = (days >= pd.Timestamp(ds.attrs["event_start"])) & (days <= pd.Timestamp(ds.attrs["event_end"]))
    return [i for i in np.flatnonzero(in_event) if ((t[i] == 0) | (t[i] == 1)).any()]


def score(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    p, y = pred[mask] == 1, truth[mask] == 1
    return {"tp": int((p & y).sum()), "fp": int((p & ~y).sum()), "fn": int((~p & y).sum()), "tn": int((~p & ~y).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", choices=["gap", "all"], default="gap", help="gap = events with gold days in 2021-01..2023-07")
    ap.add_argument("--plan", action="store_true", help="count window-days and tile downloads, fetch nothing")
    ap.add_argument("--limit", type=int, default=None, help="first N events only (smoke test)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "experiment_modis_gold.log")), logging.StreamHandler(sys.stdout)])
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    split = pd.read_csv(cfg.results_dir / "event_split.csv").set_index("code").split
    s = cfg["sources"]["modis"]

    todo = []
    for _, r in sel.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            for i in gold_days(ds):
                d = pd.Timestamp(ds.day.values[i])
                if a.set == "gap" and not (GAP[0] <= d <= GAP[1]):
                    continue
                todo.append((r.code, ew, i, d.date()))
    if a.limit:
        keep = list(dict.fromkeys(t[0] for t in todo))[: a.limit]
        todo = [t for t in todo if t[0] in keep]
    n_tiles = sum(len(modis.tiles_for_bounds(*ew.window.bounds)) for _, ew, _, _ in todo)
    log.info("%d events, %d window-days, %d tile downloads (~15 MB each, cached counts skip them)",
             len({t[0] for t in todo}), len(todo), n_tiles)
    if a.plan:
        return

    src = modis.MODISFloodSource(cfg)
    rows = []
    cache_dir = cfg.local("raw", "modis_counts", "x").parent
    cache_dir.mkdir(exist_ok=True)
    for k, (code, ew, i, day) in enumerate(todo, 1):
        cpath = cache_dir / f"{ew.key}_{day:%Y%m%d}.npz"
        if cpath.exists():
            z = np.load(cpath)
            counts = {layer: {c: z[f"{layer}:{c}"] for c in ("n_all", "n_obs", "n_flood", "n_recurring")} for layer in modis.LAYERS}
        else:
            counts = src.counts_day(ew.window, day, modis.LAYERS)
            np.savez_compressed(cpath, **{f"{layer}:{c}": v for layer, d in counts.items() for c, v in d.items()})
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        t, st, vi = ds.truth.values[i], ds.stratum.values[i], ds.viirs.values[i]
        lab = ((t == 0) | (t == 1)) & ds.domain.values.astype(bool)
        base = {"code": code, "window": ew.key, "day": day.isoformat(), "split": split.get(code, "unknown"),
                "in_viirs_gap": bool(GAP[0] <= pd.Timestamp(day) <= GAP[1])}
        for code_st, st_name in STRATUM_NAMES.items():
            m = lab & (st == code_st)
            if not m.any():
                continue
            vi_obs = vi != NODATA
            rows.append(base | {"stratum": st_name, "product": "viirs", "mode": "all"} | score(vi, t, m))
            rows.append(base | {"stratum": st_name, "product": "viirs", "mode": "conditional"} | score(vi, t, m & vi_obs))
            for layer, rec in VARIANTS:
                v, _ = modis.values_from_counts(counts[layer], rec, s["observed_min_frac"], s["flood_threshold"])
                name = f"modis:{layer}:recurring_{rec}"
                mo = v != NODATA
                rows.append(base | {"stratum": st_name, "product": name, "mode": "all"} | score(v, t, m))
                rows.append(base | {"stratum": st_name, "product": name, "mode": "conditional"} | score(v, t, m & mo))
                rows.append(base | {"stratum": st_name, "product": name, "mode": "both"} | score(v, t, m & mo & vi_obs))
                rows.append(base | {"stratum": st_name, "product": f"viirs|{layer}", "mode": "both"} | score(vi, t, m & mo & vi_obs))
        if k % 10 == 0 or k == len(todo):
            log.info("window-day %d/%d done", k, len(todo))
    out = pd.DataFrame(rows).drop_duplicates()
    out.to_csv(cfg.results_dir / f"experiment_modis_gold_{a.set}.csv", index=False)
    if src.absent:
        log.info("tile-days with no MODIS file upstream (NoData, not an error): %d, e.g. %s", len(src.absent), src.absent[:5])
    g = out.groupby(["product", "mode"])[["tp", "fp", "fn"]].sum()
    g["recall"] = g.tp / (g.tp + g.fn)
    g["precision"] = g.tp / (g.tp + g.fp)
    pd.set_option("display.width", 200)
    log.info("recall and precision, all strata:\n%s", g.round(3).to_string())


if __name__ == "__main__":
    main()

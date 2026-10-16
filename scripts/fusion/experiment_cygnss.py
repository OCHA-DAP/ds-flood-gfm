"""Experiment: the UC Berkeley CYGNSS daily watermask against gold, on a few events (2026-10-16).

Product: CYGNSS_L3_UC_BERKELEY_WATERMASK_DAILY_V3.2 (NASA PO.DAAC): 0 land, 1 surface water, -99 no data or
ocean; 0.01 degree cells centred on multiples of 0.01 (half a cell off our grid), latitude -37.4 to 37.4,
from August 2018. Read with the Earthdata token in the environment variable named by
sources.modis.token_env (EARTHDATA_TOKEN); only the rows each window needs are fetched (the variable is
stored uncompressed, row by row). Checked first (2026-10-16): a "daily" map changes by about 6% of cells from
one day to the next and by about 24% over 30 days, so it behaves like a rolling multi-week composite.

Rules, fixed before running:
  - events: the two with the most gold flooded cell-days in each continent among the fold events inside
    CYGNSS's dates and latitudes (8 events);
  - onto our grid: each of our cells overlaps four CYGNSS cells by a quarter each; observed if at least
    half of that area has data, water if at least 30% of the observed area is water (the rules every
    source uses: observed_min_frac 0.5, flood_threshold 0.30);
  - scored on gold-map days against gold, labelled floodable cells, all cells and observed cells, beside
    GFM, VIIRS, the microwave vote, FloodScan, the cascade and cross-validated fusion as built on the same
    cells; also split by normally-wet cells (D-38), because CYGNSS maps water, not flood.
Nothing is fitted; CYGNSS is not added to fusion here.

Outputs: results/experiment_cygnss_skill.csv. Needs h5py, fsspec and aiohttp, which are not project
dependencies:
  nice -n 19 uv run --with h5py --with fsspec --with aiohttp python scripts/fusion/experiment_cygnss.py
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr

from ds_flood_gfm.fusion import baseline, crossval, cube, engine, gold, grid
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA

SHORT_NAME = "CYGNSS_L3_UC_BERKELEY_WATERMASK_DAILY_V3.2"
START, LAT_MAX = pd.Timestamp("2018-08-01"), 37.4


def granule_url(day: pd.Timestamp) -> str:
    d = day.strftime("%Y-%m-%d")
    r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json",
                     params={"short_name": SHORT_NAME, "temporal": f"{d}T00:00:00Z,{d}T23:59:59Z", "page_size": 5}, timeout=60)
    r.raise_for_status()
    hits = [lk["href"] for g in r.json()["feed"]["entry"] for lk in g["links"]
            if lk["href"].startswith("https") and lk["href"].endswith(".nc") and d in lk["href"]]
    if not hits:
        raise FileNotFoundError(f"no CYGNSS watermask granule for {d} in CMR")
    return hits[0]


def read_window(fs, url: str, win: grid.Window) -> np.ndarray:
    """CYGNSS values on our grid for the window: 1 water, 0 dry, 255 couldn't see."""
    import h5py

    minx, miny, maxx, maxy = win.bounds
    with fs.open(url, "rb") as fh, h5py.File(fh, "r") as f:
        lat, lon = f["lat"][:].astype("float64"), f["lon"][:].astype("float64")
        # CYGNSS centres at k * 0.01; our cell [a, a + 0.01] overlaps centres a and a + 0.01 in each axis
        r0 = int(np.argmin(np.abs(lat - miny))); r1 = int(np.argmin(np.abs(lat - maxy)))
        c0 = int(np.argmin(np.abs(lon - minx))); c1 = int(np.argmin(np.abs(lon - maxx)))
        w = f["watermask"][r0:r1 + 1, c0:c1 + 1].astype("int16")  # south to north
    if not (np.isclose(lat[r0], miny, atol=1e-3) and np.isclose(lon[c0], minx, atol=1e-3)):
        raise ValueError(f"CYGNSS grid does not line up with the window edge ({lat[r0]}, {lon[c0]}) vs ({miny}, {minx})")
    w = w[::-1]  # north to south, like our windows
    obs = (w != -99).astype(float)
    wet = (w == 1).astype(float)
    # each of our cells = mean of the 2 x 2 CYGNSS centres at its corners (a quarter of each in the cell)
    o = (obs[:-1, :-1] + obs[1:, :-1] + obs[:-1, 1:] + obs[1:, 1:]) / 4
    v = (wet[:-1, :-1] + wet[1:, :-1] + wet[:-1, 1:] + wet[1:, 1:]) / 4
    if o.shape != win.shape:
        raise ValueError(f"resampled shape {o.shape} != window {win.shape}")
    frac = np.divide(v, o, out=np.zeros_like(v), where=o > 0)
    return np.where(o >= 0.5, (frac >= 0.30).astype("uint8"), NODATA).astype("uint8")


def main() -> None:
    import fsspec

    cfg = load_config()
    tok = os.environ.get(cfg["sources"]["modis"]["token_env"])
    if not tok:
        raise RuntimeError(f"set {cfg['sources']['modis']['token_env']} (Earthdata token) in the environment")
    fs = fsspec.filesystem("https", headers={"Authorization": f"Bearer {tok}"}, block_size=2**22)
    f = cfg["fusion"]
    thr = f["prob_threshold"]
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    ev = ev[ev.code.isin(folds.code)].merge(folds[["code", "fold", "flood_cell_days"]], on="code")
    inside = ev[(ev.start >= START) & (ev.lat.abs() <= LAT_MAX)]
    pick = inside.sort_values("flood_cell_days", ascending=False).groupby("continent").head(2).sort_values(["continent", "code"])
    print("events:", pick[["code", "continent", "flood_cell_days"]].to_string(index=False), flush=True)
    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = counts[counts.rule == cfg["gold"]["dry_window_rule"]]
    prior = pd.read_csv(cfg.results_dir / "cv" / "prior.csv")
    models = {k: engine.Model.from_tables(crossval.fit_table(counts, ev.code[ev.fold != k]), prior, MODEL_SOURCES, f["decay"],
                                          tuple(f["sens_spec_clip"])) for k in sorted(ev.fold.unique())}
    rows, t0 = [], time.time()
    for _, r in pick.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            days = pd.to_datetime(ds.day.values)
            win = grid.Window(*json.loads(ds.attrs["window"]))
            p = engine.run_sequence({s: ds[s].values for s in MODEL_SOURCES}, ds.stratum.values, ds.domain.values.astype(bool),
                                    models[r.fold])["p_flood"]
            t = ds.truth.values
            for i in np.where(crossval.gold_day_mask(ds))[0]:
                lab = ((t[i] == gold.DRY) | (t[i] == gold.FLOOD)) & np.isfinite(p[i])
                if not lab.any():
                    continue
                cyg = read_window(fs, granule_url(days[i]), win)
                nw = baseline.normally_wet(cfg, ew.window, days[i].month)
                v = {s: ds[s].values[i] for s in ["gfm", "viirs", "microwave", "floodscan"]}
                casc = np.full(cyg.shape, NODATA, "uint8")
                for s in ["microwave", "gfm", "viirs"]:
                    casc = np.where(v[s] != NODATA, v[s], casc)
                prods = {"CYGNSS": cyg, "fusion (as built)": np.where(p[i] >= thr, 1, 0).astype("uint8"), "cascade": casc,
                         "GFM": v["gfm"], "VIIRS": v["viirs"], "microwave vote": v["microwave"], "FloodScan": v["floodscan"]}
                tf = t[i] == gold.FLOOD
                for g, gm in (("all cells", np.ones_like(nw)), ("normally wet", nw), ("other", ~nw)):
                    m = lab & gm
                    for name, val in prods.items():
                        fl = val == 1
                        seen = val != NODATA
                        rows.append({"code": r.code, "continent": r.continent, "day": days[i], "group": g, "product": name,
                                     "tp": int((fl & tf & m).sum()), "fp": int((fl & ~tf & m).sum()), "fn": int((~fl & tf & m).sum()),
                                     "tp_obs": int((fl & tf & m & seen).sum()), "fp_obs": int((fl & ~tf & m & seen).sum()),
                                     "fn_obs": int((~fl & tf & m & seen).sum()), "cells": int(m.sum()), "seen": int((m & seen).sum()),
                                     "flood_seen": int((m & seen & tf).sum()), "flood": int((m & tf).sum())})
                print(f"{r.code} {ew.key} {days[i]:%Y-%m-%d} done, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.results_dir / "experiment_cygnss_rows.csv", index=False)
    s = df.groupby(["group", "product"])[["tp", "fp", "fn", "tp_obs", "fp_obs", "fn_obs", "seen", "cells", "flood_seen", "flood"]].sum().reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        s["recall"], s["precision"] = s.tp / (s.tp + s.fn), s.tp / (s.tp + s.fp)
        s["f1"] = 2 * s.tp / (2 * s.tp + s.fp + s.fn)
        s["f1_observed_cells"] = 2 * s.tp_obs / (2 * s.tp_obs + s.fp_obs + s.fn_obs)
        s["share_of_flood_seen"] = s.flood_seen / s.flood
    s.to_csv(cfg.results_dir / "experiment_cygnss_skill.csv", index=False)
    pd.set_option("display.width", 220)
    print(s[["group", "product", "recall", "precision", "f1", "f1_observed_cells", "share_of_flood_seen", "flood"]].round(3).to_string(index=False))
    per = df[df.group == "all cells"].groupby(["code", "product"])[["tp", "fp", "fn"]].sum()
    per["f1"] = 2 * per.tp / (2 * per.tp + per.fp + per.fn)
    print(per.f1.unstack("product").round(2).to_string())


if __name__ == "__main__":
    main()

"""MODIS flood product (MCDWD) for every event window, every day: per-cell pixel counts (D-37, D-41).

Built to run on Databricks (`flood_fusion_modis` job): the full set is ~25,000 tile-days
(~376 GB of HDF4 tiles), far beyond a laptop uplink. Only the counts are kept.

For each event (resumable: events whose window files all exist are skipped):
  1. list every (day, tile) its windows need, and fetch each day's LAADS listing once;
  2. download each tile-day once (thread pool), and for every window it touches add the
     per-cell counts (all, observed, flood code 3, recurring code 2) of the 1-day layers;
  3. write one NetCDF per window, dims (day, lat, lon), variables {layer}__{count} as uint16,
     and upload it to blob processed/flood_fusion_v1/modis/{window}.nc (or keep it local).

Window list: results/event_windows.csv (written with --write-window-list from the local event
table; committed, so the job needs no local data). Token: EARTHDATA_TOKEN, or the team login in
IMERG_USERNAME / IMERG_PASSWORD (dsci scope, passed by run_task.py --secret).

Run:   nice -n 19 uv run python scripts/fusion/modis_windows.py [--events EMSR551,...] [--local-only]
Job:   databricks bundle run flood_fusion_modis -t dev
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.sources import modis

log = logging.getLogger("modis_windows")
LAYERS = ("Flood_1Day_250m", "FloodCS_1Day_250m")
COUNTS = ("n_all", "n_obs", "n_flood", "n_recurring")


def write_window_list(cfg) -> None:
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    split = pd.read_csv(cfg.results_dir / "event_split.csv").set_index("code").split
    rows = []
    for _, r in sel.iterrows():
        for i, w in enumerate(json.loads(r.windows)):
            win = grid.Window(*w)
            rows.append({"code": r.code, "split": split[r.code], "window": f"{r.code}_w{i}", "row0": win.row0, "row1": win.row1,
                         "col0": win.col0, "col1": win.col1, "first_day": pd.Timestamp(r.window_start).date(),
                         "last_day": pd.Timestamp(r.window_end).date()})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.results_dir / "event_windows.csv", index=False)
    log.info("wrote %d windows of %d events to results/event_windows.csv", len(out), out.code.nunique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--events", default=None, help="comma-separated event codes (default: all)")
    ap.add_argument("--local-only", action="store_true", help="write under data/fusion/modis/, do not upload")
    ap.add_argument("--only-day", default=None, help="smoke test: process this one day (YYYY-MM-DD) of each window only")
    ap.add_argument("--write-window-list", action="store_true", help="write results/event_windows.csv from the local event table and exit")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("azure").setLevel(logging.WARNING)
    if a.write_window_list:
        write_window_list(cfg)
        return
    wl = pd.read_csv(cfg.results_dir / "event_windows.csv", parse_dates=["first_day", "last_day"])
    if a.events:
        wl = wl[wl.code.isin(a.events.split(","))]
        if wl.empty:
            raise ValueError(f"none of {a.events} is in results/event_windows.csv")
    src = modis.MODISFloodSource(cfg)
    threads = cfg["sources"]["modis"]["threads"]
    cc = None
    if not a.local_only:
        import ocha_stratus as stratus

        b = cfg["paths"]["blob"]
        cc = stratus.get_container_client(container_name=b["container"], stage=b["stage"], write=True)
        prefix = cfg.blob_path("processed", "modis")
        done = {x.name.rsplit("/", 1)[-1] for x in cc.list_blobs(name_starts_with=prefix)}
    else:
        done = {p.name for p in cfg.local_root.joinpath("modis").glob("*.nc")}
    events = list(dict.fromkeys(wl.code))
    log.info("%d events, %d windows; %d window files already exist", len(events), len(wl), len(done))
    for k, code in enumerate(events, 1):
        ws = wl[wl.code == code]
        if all(f"{w}.nc" in done for w in ws.window):
            continue
        wins = {r.window: grid.Window(r.row0, r.row1, r.col0, r.col1) for r in ws.itertuples()}
        days = {r.window: pd.date_range(r.first_day, r.last_day).date for r in ws.itertuples()}
        if a.only_day:
            od = dt.date.fromisoformat(a.only_day)
            days = {w: np.array([od]) for w, d in days.items() if od in set(d)}
            wins = {w: wins[w] for w in days}
        acc = {w: {f"{layer}__{c}": np.zeros((len(days[w]), *wins[w].shape), "uint16") for layer in LAYERS for c in COUNTS} for w in wins}
        need: dict[tuple[dt.date, int, int], list[tuple[str, int]]] = {}
        for w, win in wins.items():
            for i, d in enumerate(days[w]):
                for h, v in modis.tiles_for_bounds(*win.bounds):
                    need.setdefault((d, h, v), []).append((w, i))
        for d in sorted({key[0] for key in need}):
            src.day_listing(d)  # fetch each day's listing once, before the threads

        def work(key):
            d, h, v = key
            arrays = src.read_tile(d, h, v, LAYERS)
            if arrays is None:
                return key, None
            out = []
            for w, i in need[key]:
                for layer, codes in arrays.items():
                    c = modis.tile_counts(codes, h, v, wins[w])
                    if c is not None:
                        out.append((w, i, layer, c))
            return key, out

        with ThreadPoolExecutor(threads) as ex:
            for _, res in ex.map(work, list(need)):
                for w, i, layer, c in res or []:
                    for name in COUNTS:
                        acc[w][f"{layer}__{name}"][i] += c[name].astype("uint16")
        for w, win in wins.items():
            ds = xr.Dataset({k2: (("day", "lat", "lon"), v2) for k2, v2 in acc[w].items()},
                            coords={"day": pd.to_datetime(days[w]), "lat": win.lat(), "lon": win.lon()},
                            attrs={"window": json.dumps([win.row0, win.row1, win.col0, win.col1]), "source": "NASA LANCE MCDWD (LAADS)",
                                   "layers": ",".join(LAYERS), "counts": ",".join(COUNTS), "created": dt.datetime.now(dt.timezone.utc).isoformat()})
            local = cfg.local("modis_smoke" if a.only_day else "modis", f"{w}.nc")
            ds.to_netcdf(local, engine="netcdf4", encoding={k2: {"zlib": True, "complevel": 4} for k2 in acc[w]})
            if cc is not None and not a.only_day:
                with open(local, "rb") as f:
                    cc.upload_blob(f"{prefix}/{w}.nc", f, overwrite=True)
        log.info("event %d/%d %s: %d windows, %d tile-days; absent upstream so far: %d",
                 k, len(events), code, len(wins), len(need), len(src.absent))
    if src.absent:
        log.info("tile-days with no file upstream (NoData, recorded): %d, e.g. %s", len(src.absent), src.absent[:10])


if __name__ == "__main__":
    main()

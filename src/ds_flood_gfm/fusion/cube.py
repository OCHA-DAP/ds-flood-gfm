"""Per-event-window data cube: static layers, gold truth and the three sources, daily.

Stage A (`build_gold`): static layers + gold truth on gold days -> gold/{key}.nc
Stage B (`gfds_requests`): the (day -> windows) map for the shared GFDS prefetch
Stage C (`build_sources`): GFM, FloodScan, GFDS, microwave, VIIRS for every window day -> cube/{key}.nc

The full cube for one window has dims (day, lat, lon) over
[start - dry_pre_days, end + dry_post_gap_days + dry_post_days]:
  truth     1 flood / 0 dry / 255 none; dry-window days are dry over the footprint
  gfm, floodscan, gfds, microwave, viirs   1 / 0 / 255 per fixed decision 1
  stratum   (month-dependent) 0..5 / 255; domain (floodable & not permanent water)
Each stage writes atomically (.part then rename) and is skipped if its output exists.
"""

from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import gold, grid, static
from ds_flood_gfm.fusion.config import Config
from ds_flood_gfm.fusion.sources import NODATA, merge_microwave

ENC_U8 = {"dtype": "uint8", "zlib": True, "complevel": 4}
DAY_THREADS = 4  # default; build_sources uses cfg["fetch"]["day_threads"] when set


def window_days(row: pd.Series) -> list[dt.date]:
    return [d.date() for d in pd.date_range(row.window_start, row.window_end, freq="D")]


def _atomic_write(ds: xr.Dataset, path, encoding=None) -> None:
    tmp = path.with_suffix(".part.nc")
    ds.to_netcdf(tmp, encoding=encoding, engine="netcdf4")
    tmp.rename(path)


def gold_path(cfg: Config, key: str):
    return cfg.local("cube", "gold", f"{key}.nc")


def cube_path(cfg: Config, key: str):
    return cfg.local("cube", "events", f"{key}.nc")


def build_gold(cfg: Config, row: pd.Series, kept_sets: pd.DataFrame) -> list[str]:
    """Stage A for every window of one event. Returns the window keys written or already present."""
    keys = []
    label_rows = None
    for ew in gold.event_windows(row):
        path = gold_path(cfg, ew.key)
        keys.append(ew.key)
        if path.exists():
            continue
        if label_rows is None:
            label_rows = gold.read_label_rows(cfg, row.source, row.code)
        st = static.static_for_window(cfg, ew.window)
        days, truth, footprint = gold.truth_for_window(cfg, ew, label_rows, kept_sets)
        domain = st.domain.values
        truth[:, ~domain] = gold.NOTRUTH
        ds = xr.Dataset(
            {
                "truth": (("gold_day", "lat", "lon"), truth),
                "footprint": (("lat", "lon"), (footprint & domain).astype("uint8")),
                "domain": (("lat", "lon"), domain.astype("uint8")),
                "landcover": (("lat", "lon"), st.landcover.values),
                "hand_low_frac": (("lat", "lon"), st.hand_low_frac.values),
                "built_frac": (("lat", "lon"), st.built_frac.values),
                "tree_frac": (("lat", "lon"), st.tree_frac.values),
            },
            coords={"gold_day": days.values, "lat": ew.window.lat(), "lon": ew.window.lon()},
            attrs={"code": ew.code, "window_idx": ew.idx, "window": json.dumps([ew.window.row0, ew.window.row1, ew.window.col0, ew.window.col1])},
        )
        _atomic_write(ds, path)
    return keys


def gfds_requests(cfg: Config, events: pd.DataFrame) -> dict[dt.date, list[grid.Window]]:
    req: dict[dt.date, list[grid.Window]] = {}
    for _, row in events.iterrows():
        wins = [ew.window for ew in gold.event_windows(row)]
        for d in window_days(row):
            req.setdefault(d, []).extend(wins)
    return req


def build_sources(cfg: Config, row: pd.Series, gfm_src, fs_src, gfds_src, viirs_src, snow: static.SnowClimatology) -> list[str]:
    """Stage C for every window of one event (requires stage A and the GFDS prefetch)."""
    keys = []
    days = window_days(row)
    day_index = {d: i for i, d in enumerate(days)}
    g = cfg["gold"]
    n_threads = cfg.raw.get("fetch", {}).get("day_threads", DAY_THREADS)
    start, end = row.start.date(), row.end.date()
    for ew in gold.event_windows(row):
        path = cube_path(cfg, ew.key)
        keys.append(ew.key)
        if path.exists():
            continue
        gds = xr.load_dataset(gold_path(cfg, ew.key), engine="netcdf4")
        win = ew.window
        shape = (len(days), *win.shape)
        # --- truth over all window days
        truth = np.full(shape, gold.NOTRUTH, "uint8")
        fp = gds.footprint.values.astype(bool)
        pre_end = start - dt.timedelta(days=1)
        post_start = end + dt.timedelta(days=1 + g["dry_post_gap_days"])
        for d, i in day_index.items():
            if d <= pre_end or d >= post_start:
                truth[i][fp] = gold.DRY
        for k, gd in enumerate(pd.to_datetime(gds.gold_day.values)):
            truth[day_index[gd.date()]] = gds.truth.values[k]
        # --- sources
        gfm = np.full(shape, NODATA, "uint8")
        gfm_frac = np.full(shape, np.nan, "float32")
        n_dropped0 = len(gfm_src.dropped_scenes)
        n_noadv0 = len(gfm_src.scenes_without_advisory)
        by_day = gfm_src.search(win, days[0], days[-1])
        gfm_days = [d for d in by_day if d in day_index]
        # days are independent network reads: run them concurrently (I/O bound)
        with ThreadPoolExecutor(n_threads) as ex:
            for d, (v, f) in zip(gfm_days, ex.map(lambda d: gfm_src.observe_day(win, by_day[d]), gfm_days)):
                gfm[day_index[d]] = v
                gfm_frac[day_index[d]] = f
        fs, sfed = fs_src.observe_days(win, days)
        gf, r1, absent = gfds_src.observe_days(win, days)
        mw = merge_microwave(fs, gf)
        vi = np.full(shape, NODATA, "uint8")
        with ThreadPoolExecutor(n_threads) as ex:
            for i, (v, _) in enumerate(ex.map(lambda d: viirs_src.observe_day(win, d), days)):
                vi[i] = v
        # --- strata per day (snow season by calendar month)
        lc = gds.landcover.values
        months = sorted({d.month for d in days})
        strat_by_month = {m: static.stratum(lc, snow.snow_season(m, win)) for m in months}
        stratum = np.stack([strat_by_month[d.month] for d in days])
        dims = ("day", "lat", "lon")
        out = xr.Dataset(
            {
                "truth": (dims, truth),
                "gfm": (dims, gfm),
                "floodscan": (dims, fs),
                "gfds": (dims, gf),
                "microwave": (dims, mw),
                "viirs": (dims, vi),
                "stratum": (dims, stratum),
                "gfm_frac_pct": (dims, np.where(np.isfinite(gfm_frac), np.clip(np.round(gfm_frac * 100), 0, 100), 255).astype("uint8")),
                "domain": (("lat", "lon"), gds.domain.values),
                "footprint": (("lat", "lon"), gds.footprint.values),
                "landcover": (("lat", "lon"), lc),
            },
            coords={"day": pd.to_datetime(days).values, "lat": win.lat(), "lon": win.lon()},
            attrs={
                "code": ew.code,
                "window_idx": ew.idx,
                "window": gds.attrs["window"],
                "event_start": str(start),
                "event_end": str(end),
                "gfds_absent_days": json.dumps([str(d) for d in absent]),
                "gfm_scene_days": json.dumps(sorted(str(d) for d in by_day if d in day_index)),
                "gfm_dropped_scenes": json.dumps(gfm_src.dropped_scenes[n_dropped0:]),
                "gfm_scenes_without_advisory": json.dumps(gfm_src.scenes_without_advisory[n_noadv0:]),
            },
        )
        enc = {v: ENC_U8 for v in ["truth", "gfm", "floodscan", "gfds", "microwave", "viirs", "stratum", "gfm_frac_pct"]}
        _atomic_write(out, path, encoding=enc)
    return keys

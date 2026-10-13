"""The daily fusion job: one function shared by the operational run, the backfill and validation.

`run_window(provider, ...)` walks consecutive days on one grid window. For each
day it:
  1. takes the three source rasters from a provider;
  2. runs `engine.run_day` with yesterday's state;
  3. writes the daily COG and today's state.

A day whose COG and state both exist is skipped, so the job is idempotent and
resumable.

Providers:
  LiveProvider  fetches GFM / FloodScan / GFDS / VIIRS through the adapters (operational)
  CubeProvider  replays an event cube built in phase 1 (validation, no re-fetch)

Cold start: if yesterday's state is missing, the run refuses to start unless
`cold_start=True` is passed explicitly. In that case the state starts at the
prior (backfill spin-up). The choice is recorded in the COG tags.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import engine, grid, io, static
from ds_flood_gfm.fusion.config import Config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA, merge_microwave


def load_model(cfg: Config, accuracy_csv: Path | None = None, prior_csv: Path | None = None, sources: list[str] | None = None) -> engine.Model:
    acc = pd.read_csv(accuracy_csv or cfg.results_dir / "accuracy_table.csv")
    prior = pd.read_csv(prior_csv or cfg.results_dir / "prior_v1.csv")
    f = cfg["fusion"]
    return engine.Model.from_tables(acc, prior, sources or MODEL_SOURCES, f["decay"], tuple(f["sens_spec_clip"]))


class CubeProvider:
    """Observations from an event cube (dims day, lat, lon)."""

    def __init__(self, ds: xr.Dataset, sources: list[str] = MODEL_SOURCES) -> None:
        self.ds = ds
        self.sources = sources
        self.index = {pd.Timestamp(d).date(): i for i, d in enumerate(ds.day.values)}

    def get(self, day: dt.date) -> tuple[dict[str, np.ndarray], list[str]]:
        i = self.index[day]
        obs = {s: self.ds[s].values[i] for s in self.sources}
        present = [s for s in self.sources if (obs[s] != NODATA).any()]
        return obs, present


class LiveProvider:
    """Observations fetched live for one window (operational mode)."""

    def __init__(self, cfg: Config, win: grid.Window) -> None:
        from ds_flood_gfm.fusion.sources.floodscan import FloodScanSource
        from ds_flood_gfm.fusion.sources.gfds import GFDSSource
        from ds_flood_gfm.fusion.sources.gfm import GFMSource
        from ds_flood_gfm.fusion.sources.viirs import VIIRSSource

        self.cfg, self.win = cfg, win
        self.gfm, self.fs, self.gfds, self.viirs = GFMSource(cfg), FloodScanSource(cfg), GFDSSource(cfg), VIIRSSource(cfg)
        self.absent: dict[str, list[str]] = {"gfds": []}

    def get(self, day: dt.date) -> tuple[dict[str, np.ndarray], list[str]]:
        w = self.win
        items = self.gfm.search(w, day, day).get(day, [])
        g, _ = self.gfm.observe_day(w, items) if items else (np.full(w.shape, NODATA, "uint8"), None)
        fs, _ = self.fs.observe_days(w, [day])
        self.gfds.prefetch({day: [w]})
        gf, _, absent = self.gfds.observe_days(w, [day])
        if absent:
            self.absent["gfds"].append(day.isoformat())
        mw = merge_microwave(fs[0], gf[0])
        vi, _ = self.viirs.observe_day(w, day)
        obs = {"gfm": g, "microwave": mw, "viirs": vi}
        present = [s for s in obs if (obs[s] != NODATA).any()]
        return obs, present


@dataclass
class StaticForWindow:
    domain: np.ndarray
    strata_by_month: dict[int, np.ndarray]


def static_layers(cfg: Config, win: grid.Window) -> StaticForWindow:
    st = static.static_for_window(cfg, win)
    snow = static.SnowClimatology(cfg)
    return StaticForWindow(st.domain.values.astype(bool),
                           {m: static.stratum(st.landcover.values, snow.snow_season(m, win)) for m in range(1, 13)})


def state_path(cfg: Config, key: str, day: dt.date) -> Path:
    return cfg.local("out", "state", key, f"{day.isoformat()}.npz")


def run_window(cfg: Config, key: str, win: grid.Window, days: list[dt.date], provider, model: engine.Model,
               st: StaticForWindow, cold_start: bool = False, write_cogs: bool = True, overwrite: bool = False) -> dict:
    """Run consecutive `days` on one window. Returns a summary dict."""
    days = sorted(days)
    if any((b - a).days != 1 for a, b in zip(days, days[1:])):
        raise ValueError("days must be consecutive (the carry-over needs every day)")
    def complete(day: dt.date) -> bool:
        return not overwrite and state_path(cfg, key, day).exists() and (io.daily_cog_path(cfg, key, day).exists() or not write_cogs)

    todo = [d for d in days if not complete(d)]
    if not todo:
        return {"key": key, "days": len(days), "written": 0, "skipped": len(days), "start": "none-needed",
                "first": days[0].isoformat(), "last": days[-1].isoformat()}
    first = todo[0]
    # every day from the first incomplete one onward is recomputed in order (the carry-over
    # makes later days depend on it), so later partial gaps are refilled consistently
    prev_day = first - dt.timedelta(days=1)
    sp = state_path(cfg, key, prev_day)
    if sp.exists():
        with np.load(sp) as z:
            L, D = z["logodds"], z["days_since"]
        started = "warm"
    elif cold_start and first == days[0]:
        L, D = None, None
        started = "cold"
    else:
        raise FileNotFoundError(f"no state for {key} on {prev_day} ({sp}); pass cold_start=True to start at the prior")
    written = 0
    skipped = days.index(first)
    for day in days[days.index(first):]:
        spath = state_path(cfg, key, day)
        obs, present = provider.get(day)
        out = engine.run_day(obs, st.strata_by_month[day.month], st.domain, model, L, D)
        L, D = out.logodds, out.days_since_obs
        tmp = spath.with_suffix(".part.npz")
        np.savez_compressed(tmp, logodds=L.astype("float32"), days_since=D)
        tmp.rename(spath)
        if write_cogs:
            io.write_daily_cog(cfg, key, day, win, out, present)
        written += 1
    return {"key": key, "days": len(days), "written": written, "skipped": skipped, "start": started,
            "first": days[0].isoformat(), "last": days[-1].isoformat()}


def summary_json(d: dict) -> str:
    return json.dumps(d, default=str)

"""Phase 2: training cube — stratified random sample, static rasters, prior v1.

The gold part of the training cube is built by phase1 (stage A + C); this
script adds the stratified random sample of floodable cells and the static
rasters, then estimates prior v1 once Phase 3's gold accuracy table exists.

Subcommands (each resumable):
  design   candidate blocks (Koppen group x continent, + city blocks), static layers,
           cell-days per stratum, block selection to >= target per stratum
  gfds     GFDS prefetch for every sample day x block
  sources  per block: GFM, FloodScan, GFDS, microwave, VIIRS for every sample day
  static   write static rasters (domain, HAND-low fraction, permanent water, monthly
           stratum) as COGs for every event window and sample block
Prior v1 (per-stratum prevalence MLE on this sample) needs the gold accuracy
table, so it is computed by phase3_accuracy.py right after the table (D-23).

Run: nice -n 19 uv run python scripts/fusion/phase2_cube.py <subcommand>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import cube, grid, sample, static
from ds_flood_gfm.fusion.config import load_config

log = logging.getLogger("phase2")
ENC = cube.ENC_U8


def _setup(cfg, name):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", f"phase2_{name}.log")), logging.StreamHandler(sys.stdout)])
    logging.getLogger("azure").setLevel(logging.WARNING)


def _static_one(cfg_path, bid, lon, lat, city):
    cfg = load_config(cfg_path)
    win = sample.block_window(cfg, lon, lat, city=city)
    st = static.static_for_window(cfg, win)
    snow = static.SnowClimatology(cfg)
    return bid, win, sample.stratum_cell_days(cfg, win, st, snow)


def cmd_design(cfg, workers):
    s = cfg["sample"]
    cand = sample.candidate_blocks(cfg, per_cell=s["blocks_per_cell"], seed=s["seed"])
    cities = sample.city_blocks(cfg, n_snow=s["city_blocks_snow"], n_other=s["city_blocks_other"])
    # snow top-up pool (D-23): the snow strata are rare, so draw extra reserve candidates there
    snow_extra = sample.snow_reserve_blocks(cfg, n=s["snow_reserve_random"], seed=s["seed"] + 1)
    snow_cities = sample.city_blocks(cfg, n_snow=s["snow_reserve_cities"], n_other=0)
    snow_cities = snow_cities[snow_cities["rank"] >= s["city_blocks_snow"] * 2].assign(kind="city_reserve")
    allb = pd.concat([cand.assign(city=""), cities, snow_extra.assign(city=""), snow_cities], ignore_index=True)
    allb = allb.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    allb["block_id"] = [f"b{i:04d}_{k}" for i, k in enumerate(allb.kind)]
    log.info("candidates: %s", allb.kind.value_counts().to_dict())
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_static_one, str(cfg.path), r.block_id, r.lon, r.lat, r.kind.startswith("city")) for r in allb.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            bid, win, cd = f.result()
            rows.append({"block_id": bid, "row0": win.row0, "row1": win.row1, "col0": win.col0, "col1": win.col1, **cd})
            if i % 20 == 0:
                log.info("static %d/%d", i, len(futs))
    tally = allb.merge(pd.DataFrame(rows), on="block_id")
    strata = list(static.STRATUM_NAMES.values())
    target = s["min_cell_days_per_stratum"]
    chosen = tally[tally.kind.isin(["random", "city"])].copy()
    reserve = tally[tally.kind.isin(["reserve", "city_reserve"])].copy()
    totals = chosen[strata].sum()
    # top up any stratum below target from the reserve, best contributor first
    added = 0
    for st_name in strata:
        while totals[st_name] < target and len(reserve) and added < s["max_topup_blocks"]:
            best = reserve[st_name].idxmax()
            if reserve.loc[best, st_name] == 0:
                break
            chosen = pd.concat([chosen, reserve.loc[[best]]])
            reserve = reserve.drop(best)
            added += 1
            totals = chosen[strata].sum()
    report = pd.DataFrame({"stratum": strata, "cell_days": [int(totals[x]) for x in strata], "target": target})
    report["meets_target"] = report.cell_days >= target
    report.to_csv(cfg.results_dir / "phase2_sample_cell_days_by_stratum.csv", index=False)
    chosen.to_parquet(cfg.local("sample", "blocks.parquet"))
    chosen[["block_id", "kind", "city", "lon", "lat", "koppen", "kgroup", "continent", *strata]].to_csv(cfg.results_dir / "phase2_sample_blocks.csv", index=False)
    log.info("chosen %d blocks; cell-days by stratum:\n%s", len(chosen), report)
    if not report.meets_target.all():
        log.warning("strata below target (reported in results/phase2_sample_cell_days_by_stratum.csv): %s",
                    report[~report.meets_target].stratum.tolist())


def _blocks(cfg):
    b = pd.read_parquet(cfg.local("sample", "blocks.parquet"))
    b["window"] = [grid.Window(int(r.row0), int(r.row1), int(r.col0), int(r.col1)) for r in b.itertuples()]
    return b


def cmd_gfds(cfg):
    from ds_flood_gfm.fusion.sources.gfds import GFDSSource

    b = _blocks(cfg)
    days = sample.sample_days(cfg)
    req = {d: list(b.window) for d in days}
    src = GFDSSource(cfg)
    src.reference()
    log.info("GFDS prefetch for sample: %d days x %d blocks, first=%s", len(days), len(b), days[0])
    for i in range(0, len(days), 50):
        stats = src.prefetch({d: req[d] for d in days[i : i + 50]})
        log.info("gfds sample days %d-%d: %s", i + 1, min(i + 50, len(days)), stats)


def sample_cube_path(cfg, bid):
    return cfg.local("cube", "sample", f"{bid}.nc")


def _sources_block(cfg_path, bid, win_t):
    """Build one sample cube. Returns (bid, seconds, status) with status in {done, pending}.

    GFM, FloodScan and VIIRS are fetched first and saved as a partial file; if the
    GFDS store does not yet cover this block's days the block is left "pending"
    (no blocking), and a later round finishes it from the partial file.
    """
    from concurrent.futures import ThreadPoolExecutor

    from ds_flood_gfm.fusion.sources import NODATA, merge_microwave
    from ds_flood_gfm.fusion.sources.floodscan import FloodScanSource
    from ds_flood_gfm.fusion.sources.gfds import GFDSSource, wait_for_store
    from ds_flood_gfm.fusion.sources.gfm import GFMSource
    from ds_flood_gfm.fusion.sources.viirs import VIIRSSource

    cfg = load_config(cfg_path)
    path = sample_cube_path(cfg, bid)
    if path.exists():
        return bid, 0.0, "done"
    t = time.time()
    win = grid.Window(*win_t)
    days = sample.sample_days(cfg)
    partial = cfg.local("sample", "partial", f"{bid}.npz")
    n_threads = cfg.raw.get("fetch", {}).get("day_threads", cube.DAY_THREADS)
    if partial.exists():
        with np.load(partial, allow_pickle=False) as z:
            gfm, fs, vi = z["gfm"], z["fs"], z["vi"]
            meta = json.loads(str(z["meta"]))
    else:
        di = {d: i for i, d in enumerate(days)}
        shape = (len(days), *win.shape)
        gfm_src, fs_src, vi_src = GFMSource(cfg, threads=3), FloodScanSource(cfg), VIIRSSource(cfg, threads=3)
        gfm = np.full(shape, NODATA, "uint8")
        by_day = gfm_src.search(win, days[0], days[-1])
        gdays = [d for d in by_day if d in di]
        with ThreadPoolExecutor(n_threads) as ex:
            for d, (v, _) in zip(gdays, ex.map(lambda d: gfm_src.observe_day(win, by_day[d]), gdays)):
                gfm[di[d]] = v
        fs, _ = fs_src.observe_days(win, days)
        vi = np.full(shape, NODATA, "uint8")
        with ThreadPoolExecutor(n_threads) as ex:
            for i, (v, _) in enumerate(ex.map(lambda d: vi_src.observe_day(win, d), days)):
                vi[i] = v
        meta = {"gfm_dropped_scenes": gfm_src.dropped_scenes, "gfm_scenes_without_advisory": gfm_src.scenes_without_advisory}
        tmp = partial.with_name(f"{bid}.{os.getpid()}.part.npz")
        np.savez_compressed(tmp, gfm=gfm, fs=fs, vi=vi, meta=np.array(json.dumps(meta)))
        tmp.rename(partial)
    try:
        wait_for_store(cfg.local("gfds", "days", "x").parent, [win], days, timeout_s=0, poll_s=0)
    except TimeoutError:
        return bid, time.time() - t, "pending"
    gf, _, absent = GFDSSource(cfg).observe_days(win, days)
    st = static.static_for_window(cfg, win)
    snow = static.SnowClimatology(cfg)
    lc = st.landcover.values
    smon = {m: static.stratum(lc, snow.snow_season(m, win)) for m in range(1, 13)}
    stratum = np.stack([smon[d.month] for d in days])
    dims = ("day", "lat", "lon")
    ds = xr.Dataset(
        {"gfm": (dims, gfm), "floodscan": (dims, fs), "gfds": (dims, gf), "microwave": (dims, merge_microwave(fs, gf)),
         "viirs": (dims, vi), "stratum": (dims, stratum), "domain": (("lat", "lon"), st.domain.values.astype("uint8"))},
        coords={"day": pd.to_datetime(days).values, "lat": win.lat(), "lon": win.lon()},
        attrs={"block_id": bid, "gfds_absent_days": json.dumps([str(d) for d in absent]),
               "gfm_dropped_scenes": json.dumps(meta["gfm_dropped_scenes"]),
               "gfm_scenes_without_advisory": json.dumps(meta["gfm_scenes_without_advisory"])},
    )
    tmp = path.with_suffix(".part.nc")
    ds.to_netcdf(tmp, engine="netcdf4", encoding={v: ENC for v in ["gfm", "floodscan", "gfds", "microwave", "viirs", "stratum"]})
    tmp.rename(path)
    return bid, time.time() - t, "done"


def cmd_sources(cfg, workers):
    b = _blocks(cfg)
    b["cells"] = [w.shape[0] * w.shape[1] for w in b.window]
    b = b.sort_values("cells")
    log.info("sample sources: %d blocks, first=%s", len(b), b.iloc[0].block_id)
    todo = list(b.itertuples())
    rnd = 0
    failures = []
    while todo:
        rnd += 1
        pending = []
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(_sources_block, str(cfg.path), r.block_id, (r.window.row0, r.window.row1, r.window.col0, r.window.col1)): r for r in todo}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    bid, secs, status = f.result()
                except Exception as e:  # recorded and listed; a failed block is not retried silently
                    import traceback

                    failures.append({"block_id": futs[f].block_id, "error": f"{type(e).__name__}: {e}"})
                    log.error("round %d block %s FAILED: %s\n%s", rnd, futs[f].block_id, e, traceback.format_exc())
                    continue
                log.info("round %d block %d/%d %s %s %.0fs", rnd, i, len(futs), bid, status, secs)
                if status == "pending":
                    pending.append(futs[f])
        pd.DataFrame(failures, columns=["block_id", "error"]).to_csv(cfg.results_dir / "phase2_sample_failures.csv", index=False)
        todo = pending
        if todo:
            log.info("%d blocks wait for the GFDS prefetch; next round in 5 min", len(todo))
            time.sleep(300)


def cmd_static(cfg):
    """Static rasters as COGs (domain, HAND-low fraction, permanent water, landcover, monthly stratum)."""
    from ds_flood_gfm.fusion import io

    snow = static.SnowClimatology(cfg)
    wins = {}
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    from ds_flood_gfm.fusion import gold

    for _, r in sel.iterrows():
        for ew in gold.event_windows(r):
            wins[ew.key] = ew.window
    if cfg.local("sample", "blocks.parquet").exists():
        for r in _blocks(cfg).itertuples():
            wins[r.block_id] = r.window
    n = 0
    for key, win in wins.items():
        if io.static_cog_exists(cfg, key):
            continue
        st = static.static_for_window(cfg, win)
        months = {m: static.stratum(st.landcover.values, snow.snow_season(m, win)) for m in range(1, 13)}
        io.write_static_cogs(cfg, key, win, st, months)
        n += 1
    log.info("static COGs written for %d windows (%d total)", n, len(wins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "gfds", "sources", "static"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config(a.config)
    _setup(cfg, a.cmd)
    {"design": lambda: cmd_design(cfg, a.workers), "gfds": lambda: cmd_gfds(cfg), "sources": lambda: cmd_sources(cfg, a.workers),
     "static": lambda: cmd_static(cfg)}[a.cmd]()


if __name__ == "__main__":
    main()

"""Phase 1: gold-only feasibility (event selection, gold cube, sources, metrics, gate).

Subcommands run in order and are each resumable:
  select   gold indexes -> filtered label sets -> event table -> v1 selection -> basin split
  gold     stage A per event window: static layers + gold truth           (data/fusion/cube/gold/)
  tally    gold cells per stratum x continent (decides whether selection covers all 6 strata)
  gfds     shared GFDS prefetch for every (day, window) pair              (data/fusion/gfds/days/)
  sources  stage C per event window: GFM, FloodScan, GFDS, microwave      (data/fusion/cube/events/)
  metrics  per-source per-stratum skill, oracle test, coverage test, gate (results/phase1_*.csv)

Run: nice -n 19 uv run python scripts/fusion/phase1_feasibility.py <subcommand> [--workers N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from ds_flood_gfm.fusion import cube, events, gold
from ds_flood_gfm.fusion.config import load_config

log = logging.getLogger("phase1")


def _setup_logging(cfg, name: str) -> None:
    path = cfg.local("logs", f"phase1_{name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("azure").setLevel(logging.WARNING)


def _fetch_order(cfg, days: list) -> list:
    """Sample-period days first (so phase-2 sample cubes can build early), then the rest chronologically."""
    y0, y1 = cfg["sample"]["years"]
    first = [d for d in days if y0 <= d.year <= y1]
    return first + [d for d in days if not (y0 <= d.year <= y1)]


def cmd_select(cfg) -> None:
    idx = gold.load_label_index(cfg)
    kept, excluded = gold.select_label_sets(idx, cfg)
    sizes = gold.label_partition_sizes(cfg)
    ev = gold.event_table(kept, cfg, sizes)
    sel, ev_excluded = events.select_events(ev, cfg)
    snow_extra = events.snow_supplement(ev, kept, cfg, set(sel.code))
    log.info("snow supplement: %d events %s", len(snow_extra), snow_extra.code.tolist() if len(snow_extra) else [])
    sel = pd.concat([sel.assign(selection="quota"), snow_extra.assign(selection="snow_supplement")], ignore_index=True)
    ev_excluded = ev_excluded[~ev_excluded.code.isin(sel.code)]
    sel = events.split_events(sel, cfg)
    kept.to_parquet(cfg.local("gold", "label_sets_kept.parquet"))
    excluded.to_parquet(cfg.local("gold", "label_sets_excluded.parquet"))
    ev.to_parquet(cfg.local("gold", "event_table_all.parquet"))
    sel.to_parquet(cfg.local("gold", "events_selected.parquet"))
    cfg.results_dir.mkdir(exist_ok=True)
    cols = ["code", "source", "continent", "selection", "basin_id", "split", "start", "end", "window_start", "window_end", "n_sets", "n_days", "n_windows", "cells", "frac_s1_sets", "lon", "lat"]
    sel[cols].to_csv(cfg.results_dir / "event_split.csv", index=False)
    ev_excluded[["code", "source", "continent", "cells", "label_bytes", "reason"]].to_csv(cfg.results_dir / "events_excluded.csv", index=False)
    excluded.groupby(["source", "reason"]).size().rename("n_label_sets").reset_index().to_csv(cfg.results_dir / "label_sets_excluded_summary.csv", index=False)
    log.info("selected %d events (%s); split %s", len(sel), sel.continent.value_counts().to_dict(), sel.split.value_counts().to_dict())


def _load_selected(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    kept = pd.read_parquet(cfg.local("gold", "label_sets_kept.parquet"))
    return sel, kept


def _gold_one(cfg_path: str, row: pd.Series, kept: pd.DataFrame) -> tuple[str, float]:
    cfg = load_config(cfg_path)
    t = time.time()
    cube.build_gold(cfg, row, kept)
    return row.code, time.time() - t


def cmd_gold(cfg, workers: int, codes: list[str] | None = None) -> None:
    sel, kept = _load_selected(cfg)
    if codes:
        sel = sel[sel.code.isin(codes)]
    sel = sel.sort_values("cells")  # small first (cost proxy)
    log.info("stage A: %d events, first=%s (%d cells)", len(sel), sel.iloc[0].code, sel.iloc[0].cells)
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_gold_one, str(cfg.path), row, kept[kept.code == row.code]): row.code for _, row in sel.iterrows()}
        for i, f in enumerate(as_completed(futs), 1):
            code, secs = f.result()  # a failure raises here and stops the run, naming the event
            log.info("gold %d/%d %s %.1fs", i, len(futs), code, secs)


def cmd_tally(cfg) -> None:
    import numpy as np
    import xarray as xr

    from ds_flood_gfm.fusion import static

    sel, _ = _load_selected(cfg)
    snow = static.SnowClimatology(cfg)
    rows = []
    for _, r in sel.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.gold_path(cfg, ew.key))
            lc = ds.landcover.values
            for k, gd in enumerate(pd.to_datetime(ds.gold_day.values)):
                st = static.stratum(lc, snow.snow_season(gd.month, ew.window))
                t = ds.truth.values[k]
                for s in range(6):
                    m = st == s
                    rows.append({"code": r.code, "continent": r.continent, "split": r.split, "stratum": static.STRATUM_NAMES[s],
                                 "flood": int(((t == 1) & m).sum()), "dry": int(((t == 0) & m).sum())})
            fp = ds.footprint.values.astype(bool)
            rows.append({"code": r.code, "continent": r.continent, "split": r.split, "stratum": "_footprint_cells", "flood": 0, "dry": int(fp.sum())})
    df = pd.DataFrame(rows)
    tab = df.groupby(["stratum", "split"])[["flood", "dry"]].sum().unstack()
    df.to_csv(cfg.local("gold", "gold_tally_long.csv"), index=False)
    tab.to_csv(cfg.results_dir / "phase1_gold_cells_by_stratum.csv")
    log.info("gold cells on gold days by stratum:\n%s", tab)
    log.info("by continent:\n%s", df[df.stratum != "_footprint_cells"].groupby("continent")[["flood", "dry"]].sum())
    log.info("events with zero gold flood cells: %d", int((df[df.stratum != "_footprint_cells"].groupby("code").flood.sum() == 0).sum()))
    _ = np


def cmd_gfds(cfg) -> None:
    from ds_flood_gfm.fusion.sources.gfds import GFDSSource

    sel, _ = _load_selected(cfg)
    req = cube.gfds_requests(cfg, sel)
    # fold in the phase-2 sample blocks (if designed) so each day is fetched once for both
    blocks_path = cfg.local("sample", "blocks.parquet")
    if blocks_path.exists():
        from ds_flood_gfm.fusion import grid, sample

        b = pd.read_parquet(blocks_path)
        wins = [grid.Window(int(r.row0), int(r.row1), int(r.col0), int(r.col1)) for r in b.itertuples()]
        for d in sample.sample_days(cfg):
            req.setdefault(d, []).extend(wins)
        log.info("including %d sample blocks for %d sample days", len(wins), len(sample.sample_days(cfg)))
    src = GFDSSource(cfg)
    src.reference()
    days = _fetch_order(cfg, sorted(req))
    log.info("GFDS prefetch: %d days, first=%s", len(days), days[0])
    batch = 100
    for i in range(0, len(days), batch):
        part = days[i : i + batch]
        stats = src.prefetch({d: req[d] for d in part}, order=part)
        log.info("gfds days %d-%d of %d: %s", i + 1, min(i + batch, len(days)), len(days), stats)


def _wait_for_gfds(cfg, row: pd.Series, timeout_s: float = 8 * 3600) -> None:
    """Block until stage A output and the GFDS store hold this event (both run ahead of stage C)."""
    days = cube.window_days(row)
    store = cfg.local("gfds", "days", "x").parent
    t0 = time.time()
    import numpy as np

    from ds_flood_gfm.fusion.sources.gfds import gfds_rows

    gold_files = [cube.gold_path(cfg, ew.key) for ew in gold.event_windows(row)]
    need = set().union(*[set(gfds_rows(ew.window).tolist()) for ew in gold.event_windows(row)])
    covered: set = set()

    def ok(d) -> bool:
        if d in covered:
            return True
        p = store / f"{d:%Y%m%d}.npz"
        if not p.exists():
            return False
        with np.load(p) as z:
            good = "absent" in z.files or need <= set(z["rows"].tolist())
        if good:
            covered.add(d)
        return good

    while True:
        missing = [d for d in days if not ok(d)]
        missing_gold = [p.name for p in gold_files if not p.exists()]
        if not missing and not missing_gold:
            return
        if missing_gold and not missing and time.time() - t0 > timeout_s:
            raise TimeoutError(f"{row.code}: stage A output missing: {missing_gold}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"{row.code}: GFDS store still lacks {len(missing)} days (first {missing[0]}) after {timeout_s/3600:.1f} h")
        time.sleep(60)


def _sources_one(cfg_path: str, row: pd.Series) -> tuple[str, float]:
    from ds_flood_gfm.fusion import static
    from ds_flood_gfm.fusion.sources.floodscan import FloodScanSource
    from ds_flood_gfm.fusion.sources.gfds import GFDSSource
    from ds_flood_gfm.fusion.sources.gfm import GFMSource
    from ds_flood_gfm.fusion.sources.viirs import VIIRSSource

    cfg = load_config(cfg_path)
    _wait_for_gfds(cfg, row)
    t = time.time()
    g = _sources_one.cache = getattr(_sources_one, "cache", None) or (GFMSource(cfg, threads=3), FloodScanSource(cfg), GFDSSource(cfg), VIIRSSource(cfg, threads=4), static.SnowClimatology(cfg))
    cube.build_sources(cfg, row, *g)
    return row.code, time.time() - t


def cmd_sources(cfg, workers: int, continue_on_error: bool = False) -> None:
    sel, _ = _load_selected(cfg)
    # same order as the GFDS prefetch (sample-period events first, then chronological), so the
    # run trails the prefetch instead of waiting for all of it
    y0, y1 = cfg["sample"]["years"]
    in_sample = (sel.window_start.dt.year >= y0) & (sel.window_end.dt.year <= y1)
    sel = sel.assign(_p=(~in_sample).astype(int)).sort_values(["_p", "window_end"]).drop(columns="_p")
    log.info("stage C: %d events, first=%s (%d cells)", len(sel), sel.iloc[0].code, sel.iloc[0].cells)
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_sources_one, str(cfg.path), row): row.code for _, row in sel.iterrows()}
        failures = []
        for i, f in enumerate(as_completed(futs), 1):
            try:
                code, secs = f.result()
            except Exception as e:  # only reached with --continue-on-error; recorded, never dropped
                if not continue_on_error:
                    raise
                failures.append({"code": futs[f], "error": f"{type(e).__name__}: {e}"})
                log.error("sources %d/%d %s FAILED: %s", i, len(futs), futs[f], e)
                continue
            log.info("sources %d/%d %s %.1fs", i, len(futs), code, secs)
    fpath = cfg.results_dir / "phase1_stage_c_failures.csv"
    pd.DataFrame(failures, columns=["code", "error"]).to_csv(fpath, index=False)
    if failures:
        log.error("%d events failed stage C; listed in %s", len(failures), fpath)


def cmd_metrics(cfg) -> None:
    from ds_flood_gfm.fusion import feasibility

    sel, _ = _load_selected(cfg)
    fpath = cfg.results_dir / "phase1_stage_c_failures.csv"
    if fpath.exists():
        failed = pd.read_csv(fpath)
        if len(failed):
            log.warning("excluding %d events that failed stage C (listed in %s): %s", len(failed), fpath, failed.code.tolist())
            sel = sel[~sel.code.isin(failed.code)]
            sel.to_parquet(cfg.local("gold", "events_selected.parquet").with_name("events_evaluated.parquet"))
    feasibility.run(cfg, sel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "gold", "tally", "gfds", "sources", "metrics"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--codes", default="", help="gold: comma-separated subset of event codes")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="stage C: record failing events in results/phase1_stage_c_failures.csv and go on")
    a = ap.parse_args()
    cfg = load_config(a.config)
    _setup_logging(cfg, a.cmd)
    if a.cmd == "select":
        cmd_select(cfg)
    elif a.cmd == "gold":
        cmd_gold(cfg, a.workers, [c for c in a.codes.split(",") if c])
    elif a.cmd == "tally":
        cmd_tally(cfg)
    elif a.cmd == "gfds":
        cmd_gfds(cfg)
    elif a.cmd == "sources":
        cmd_sources(cfg, a.workers, a.continue_on_error)
    elif a.cmd == "metrics":
        cmd_metrics(cfg)


if __name__ == "__main__":
    main()

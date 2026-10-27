"""Phase 4: the fusion engine — tests first, then the tiled daily job and backfill.

  test      run the synthetic-data unit tests (engine, EM, job). Every other
            subcommand refuses to run unless these pass (prompt: tests before real data).
  run       one day on one window, live sources:  --date 2026-09-10 --bbox minx,miny,maxx,maxy
  backfill  consecutive days on one window:      --start ... --end ... --bbox ... [--cold-start]
  tiles     consecutive days on every 1-degree tile intersecting a bbox (the tiled form of `backfill`)

Windows are snapped to the 0.01-degree grid; `--key` names the output folder
(default: the window's index key). Yesterday's state must exist unless
--cold-start is given (the state then starts at the prior; recorded in the COG tags).

Run: nice -n 19 uv run python scripts/fusion/phase4_engine.py <subcommand> ...
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import subprocess
import sys

from ds_flood_gfm.fusion import grid, job
from ds_flood_gfm.fusion.config import REPO_ROOT, load_config

log = logging.getLogger("phase4")
TESTS = ["tests/test_engine.py", "tests/test_accuracy.py", "tests/test_job.py", "tests/test_sources.py", "tests/test_grid.py", "tests/test_validate.py", "tests/test_net.py"]


def run_tests() -> None:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", *TESTS], cwd=REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit("engine tests failed: refusing to run on real data")


def _window(bbox: str) -> grid.Window:
    minx, miny, maxx, maxy = (float(x) for x in bbox.split(","))
    return grid.window_from_bounds(minx, miny, maxx, maxy)


def _days(start: str, end: str) -> list[dt.date]:
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if b < a:
        raise ValueError("end before start")
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


def run_range(cfg, win: grid.Window, key: str, days: list[dt.date], cold_start: bool) -> dict:
    model = job.load_model(cfg)
    st = job.static_layers(cfg, win)
    prov = job.LiveProvider(cfg, win)
    summary = job.run_window(cfg, key, win, days, prov, model, st, cold_start=cold_start)
    summary["gfds_absent_days"] = prov.absent["gfds"]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "run", "backfill", "tiles"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--date")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--bbox")
    ap.add_argument("--key")
    ap.add_argument("--cold-start", action="store_true")
    ap.add_argument("--skip-tests", action="store_true", help="only for re-runs after `test` passed in this checkout")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "phase4.log")), logging.StreamHandler(sys.stdout)])
    logging.getLogger("azure").setLevel(logging.WARNING)
    if a.cmd == "test":
        run_tests()
        log.info("engine tests passed")
        return
    if not a.skip_tests:
        run_tests()
    if not a.bbox:
        raise SystemExit("--bbox is required")
    win = _window(a.bbox)
    if a.cmd == "run":
        days = [dt.date.fromisoformat(a.date)]
    else:
        days = _days(a.start, a.end)
    if a.cmd == "tiles":
        for t in grid.tiles_for_window(win, 1):
            log.info("%s", job.summary_json(run_range(cfg, t, t.key(), days, a.cold_start)))
    else:
        log.info("%s", job.summary_json(run_range(cfg, win, a.key or win.key(), days, a.cold_start)))


if __name__ == "__main__":
    main()

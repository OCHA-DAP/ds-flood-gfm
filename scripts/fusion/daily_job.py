"""Operational daily fusion run (the scheduled job): every configured AOI, 1-degree tiles.

For the run date (default: UTC today - operational.lag_days), for each tile:
  1. pull yesterday's state and the tile's cached static layers from blob (if there);
  2. run the job for that one day with live sources (GFM, FloodScan, GFDS, VIIRS);
  3. push the daily COG, today's state and (if new) the static cache to blob;
then write a run manifest (sources present, GFDS absence, tiles) to blob.

Blob layout (dev, `projects`): ds-flood-gfm/processed/flood_fusion_v1/
  daily/{aoi}_{tile}/{aoi}_{tile}_{date}.tif  state/{aoi}_{tile}/{date}.npz
  static_cache/{window}.nc  runs/{date}.json

A tile with no state for yesterday refuses to run unless --cold-start is passed
(first deployment). The model tables come from results/ in the checked-out repo.

Run: nice -n 19 uv run python scripts/fusion/daily_job.py [--date YYYY-MM-DD] [--cold-start] [--no-upload]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

from ds_flood_gfm.fusion import grid, io, job
from ds_flood_gfm.fusion.config import load_config

log = logging.getLogger("daily")


class Blob:
    def __init__(self, cfg) -> None:
        import ocha_stratus as stratus

        b = cfg["paths"]["blob"]
        self.cfg = cfg
        self.cc = stratus.get_container_client(container_name=b["container"], stage=b["stage"], write=True)

    def name(self, *parts: str) -> str:
        return self.cfg.blob_path("processed", *parts)

    def pull(self, name: str, local: Path) -> bool:
        """Download if the blob exists. Returns False only for a blob that does not exist."""
        from azure.core.exceptions import ResourceNotFoundError

        try:
            data = self.cc.get_blob_client(name).download_blob().readall()
        except ResourceNotFoundError:
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        return True

    def push(self, local: Path, name: str) -> None:
        with open(local, "rb") as f:
            self.cc.upload_blob(name, f, overwrite=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--date")
    ap.add_argument("--cold-start", action="store_true", help="start tiles with no saved state at the prior (first run)")
    ap.add_argument("--no-upload", action="store_true", help="local only (testing)")
    # Databricks passes unset job parameters as empty strings ("--date=", "")
    a = ap.parse_args([x for x in sys.argv[1:] if x != ""])
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "daily.log")), logging.StreamHandler(sys.stdout)])
    logging.getLogger("azure").setLevel(logging.WARNING)
    op = cfg["operational"]
    day = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(dt.UTC).date() - dt.timedelta(days=op["lag_days"])
    prev = day - dt.timedelta(days=1)
    blob = None if a.no_upload else Blob(cfg)
    model = job.load_model(cfg)
    manifest = {"valid_time": day.isoformat(), "version": cfg["version"], "tiles": [], "cold_started": [],
                "gfds_absent": False, "floodscan_africa_absent": False}
    for aoi in op["aois"]:
        win = grid.window_from_bounds(*aoi["bbox"])
        for tile in grid.tiles_for_window(win, op["tile_deg"]):
            key = f"{aoi['name']}_{tile.key()}"
            sp = job.state_path(cfg, key, prev)
            static_nc = cfg.local("static", f"{tile.key()}.nc")
            if blob:
                if not sp.exists():
                    blob.pull(blob.name("state", key, f"{prev.isoformat()}.npz"), sp)
                if not static_nc.exists():
                    blob.pull(blob.name("static_cache", f"{tile.key()}.nc"), static_nc)
            had_static = static_nc.exists()
            cold = a.cold_start and not sp.exists()
            st = job.static_layers(cfg, tile)
            prov = job.LiveProvider(cfg, tile)
            s = job.run_window(cfg, key, tile, [day], prov, model, st, cold_start=cold)
            manifest["tiles"].append({"key": key, **s})
            if cold:
                manifest["cold_started"].append(key)
            manifest["gfds_absent"] = manifest["gfds_absent"] or bool(prov.absent["gfds"])
            manifest["floodscan_africa_absent"] = manifest["floodscan_africa_absent"] or bool(prov.absent["floodscan_africa"])
            if blob:
                blob.push(io.daily_cog_path(cfg, key, day), blob.name("daily", key, f"{key}_{day.isoformat()}.tif"))
                blob.push(job.state_path(cfg, key, day), blob.name("state", key, f"{day.isoformat()}.npz"))
                if not had_static:
                    blob.push(static_nc, blob.name("static_cache", f"{tile.key()}.nc"))
            log.info("%s", json.dumps(s))
    mpath = cfg.local("out", "runs", f"{day.isoformat()}.json")
    mpath.write_text(json.dumps(manifest, indent=2))
    if blob:
        blob.push(mpath, blob.name("runs", f"{day.isoformat()}.json"))
    if manifest["gfds_absent"]:
        log.warning("GFDS had no file for %s (feed stopped 2026-09-14): microwave = FloodScan only where it covers", day)
    if manifest["floodscan_africa_absent"]:
        log.warning("FloodScan Africa has no file yet for %s (archive lag?): microwave = GFDS only in Africa, if any", day)
    log.info("daily run %s: %d tiles", day, len(manifest["tiles"]))


if __name__ == "__main__":
    main()

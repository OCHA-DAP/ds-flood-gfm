"""Build and cache static layers (domain, land cover, ...) for every 1-degree land tile in a bbox.

This is how the daily job's coverage grows beyond the configured AOIs: the
global product needs this run once over the globe (~15,000 land tiles). It is
resumable: tiles already cached on blob are skipped. Ocean tiles (no HAND tile
upstream) are skipped and listed.

Run: nice -n 19 uv run python scripts/fusion/static_build.py --bbox minx,miny,maxx,maxy [--no-upload]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

from ds_flood_gfm.fusion import grid, static
from ds_flood_gfm.fusion.config import load_config

log = logging.getLogger("static_build")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("azure").setLevel(logging.WARNING)
    minx, miny, maxx, maxy = (float(x) for x in a.bbox.split(","))
    hand_tiles = static.tile_index(cfg, "hand")
    blob = None
    if not a.no_upload:
        import ocha_stratus as stratus

        b = cfg["paths"]["blob"]
        blob = stratus.get_container_client(container_name=b["container"], stage=b["stage"], write=True)
        cached = {x.name.rsplit("/", 1)[-1] for x in blob.list_blobs(name_starts_with=cfg.blob_path("processed", "flood_fusion_v1", "static_cache"))}
    else:
        cached = set()
    done = skipped_ocean = 0
    for lat in range(math.floor(miny), math.ceil(maxy)):
        for lon in range(math.floor(minx), math.ceil(maxx)):
            la = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            lo = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            if f"Copernicus_DSM_COG_10_{la}_00_{lo}_00_HAND.tif" not in hand_tiles:
                skipped_ocean += 1
                continue
            tile = grid.window_from_bounds(lon, lat, lon + 1, lat + 1)
            name = f"{tile.key()}.nc"
            if name in cached:
                continue
            static.static_for_window(cfg, tile)
            if blob is not None:
                with open(cfg.local("static", name), "rb") as f:
                    blob.upload_blob(cfg.blob_path("processed", "flood_fusion_v1", "static_cache", name), f, overwrite=True)
            done += 1
            log.info("static tile %s built", tile.key())
    log.info("built %d tiles; %d ocean tiles (no HAND tile upstream) skipped", done, skipped_ocean)


if __name__ == "__main__":
    main()

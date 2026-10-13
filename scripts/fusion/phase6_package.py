"""Phase 6: mirror inputs and outputs to blob (dev, `projects`) and write the dataset manifest.

Raw inputs (KB: {PROJECT_PREFIX}/raw/{datasource}/...) — files we downloaded that are
not already on team blob:
  raw/era5_land/   snow-cover climatology input (CDS)
  raw/gfds/        GFDS static baseline + the daily signal row bands we fetched (the feed has ended)
  raw/hydrobasins/ raw/koppen/ raw/natural_earth/ raw/tile_index/
Processed outputs ({PROJECT_PREFIX}/processed/flood_fusion_v1/...):
  daily/{window}/...tif     daily COGs for every held-out event window (P, |L|, n_sources, age)
  static/{window}.tif       static rasters (domain, HAND fraction, perm. water, land cover, monthly strata)
  cubes/events/*.nc         per-event-window training/validation cubes; cubes/sample/*.nc
  results/                  every results/ table and figure (incl. accuracy_table.csv)
  MANIFEST.json             what the dataset is, bands, versions, licence limits, how to read it
Every upload is verified by listing the prefix afterwards; a count mismatch raises.

Run: nice -n 19 uv run python scripts/fusion/phase6_package.py [--skip-raw] [--skip-cubes]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import ocha_stratus as stratus

from ds_flood_gfm.fusion.config import load_config

log = logging.getLogger("phase6")


def _cc(cfg):
    b = cfg["paths"]["blob"]
    return stratus.get_container_client(container_name=b["container"], stage=b["stage"], write=True)


def mirror(cfg, cc, local: Path, blob_prefix: str, suffixes: tuple[str, ...]) -> int:
    files = [p for p in sorted(local.rglob("*")) if p.is_file() and p.suffix in suffixes and ".part" not in p.name]
    have = {x.name: x.size for x in cc.list_blobs(name_starts_with=blob_prefix)}
    n = 0
    from ds_flood_gfm.fusion.net import retry

    for p in files:
        name = f"{blob_prefix}/{'/'.join(p.relative_to(local).parts)}"
        if have.get(name) == p.stat().st_size:
            continue

        def _up(p=p, name=name):
            with open(p, "rb") as f:
                # slow uplinks (~1 MB/s measured) time out the SDK's default single PUT: long timeout
                cc.upload_blob(name, f, overwrite=True, max_concurrency=2, timeout=1800)

        retry(_up, f"upload {name}")
        n += 1
    after = {x.name: x.size for x in cc.list_blobs(name_starts_with=blob_prefix)}
    missing = [p for p in files if after.get(f"{blob_prefix}/{'/'.join(p.relative_to(local).parts)}") != p.stat().st_size]
    if missing:
        raise RuntimeError(f"{len(missing)} files did not land under {blob_prefix} (first: {missing[0]})")
    log.info("%s: %d files present (%d uploaded now)", blob_prefix, len(files), n)
    return len(files)


def manifest(cfg) -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "name": "flood_fusion_v1",
        "description": "Daily P(flooded) on a 0.01-degree EPSG:4326 grid over the floodable domain (HAND<15 m, GSW occurrence>=90% removed), "
                       "fusing Copernicus GFM (SAR), FloodScan OR GFDS (passive microwave) and NOAA VIIRS flood maps (optical) by per-stratum naive Bayes "
                       "in log-odds with 0.9 daily carry-over.",
        "version": cfg["version"],
        "code": {"repo": "OCHA-DAP/ds-flood-gfm", "commit": sha, "config": "configs/fusion.yaml"},
        "grid": {"crs": "EPSG:4326", "res_deg": 0.01, "cell_edges": "multiples of 0.01 deg from -180/90"},
        "daily_bands": {"1": "p_flood (0-1)", "2": "abs_logodds", "3": "n_sources_observed (0-3)", "4": "days_since_obs (65535 = never)"},
        "nodata": "NaN outside the floodable domain",
        "valid_time": "the band date in the file name; GDAL tag valid_time",
        "coverage_v1": "held-out gold event windows (validation) + the operational AOIs in configs/fusion.yaml",
        "resolution_note": "0.01 deg (~1 km) is coarser than GFM's 20 m; small floods below 30% of a cell are not represented",
        "licence_limits": [
            "FloodScan (AER) is a commercial input: products derived from it may not be shared outside OCHA without AER approval.",
            "GFM/CEMS, VIIRS (NOAA), WorldCover (CC BY 4.0), GSW, GLO-30 HAND (CC0), ERA5-Land (CC BY 4.0) require attribution.",
            "GDACS/GFDS publishes no reuse licence (disclaimer only).",
        ],
        "model": {"accuracy_table": "results/accuracy_table.csv", "prior": "results/prior_v1.csv"},
        "validation": "results/phase5_*.csv and docs/final_report.md",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--skip-cubes", action="store_true")
    ap.add_argument("--only-gfds-bands", action="store_true", help="upload only the GFDS row-band archive (5.5 GB)")
    ap.add_argument("--skip-gfds-bands", action="store_true", help="everything except the GFDS row-band archive")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("azure").setLevel(logging.WARNING)
    cc = _cc(cfg)
    loc = cfg.local_root
    if a.only_gfds_bands:
        mirror(cfg, cc, loc / "gfds" / "days", cfg.blob_path("raw", "gfds", "signal_row_bands"), (".npz",))
        return
    proc = cfg.blob_path("processed", "flood_fusion_v1")
    # priority order (slow uplinks): what the scheduled job needs, then the deliverables, then archives
    if not a.skip_raw:
        mirror(cfg, cc, loc / "raw" / "era5_land", cfg.blob_path("raw", "era5_land"), (".nc",))
    m = cfg.local("out", "MANIFEST.json")
    m.write_text(json.dumps(manifest(cfg), indent=2))
    with open(m, "rb") as f:
        cc.upload_blob(f"{proc}/MANIFEST.json", f, overwrite=True)
    got = json.loads(cc.get_blob_client(f"{proc}/MANIFEST.json").download_blob().readall())
    if got.get("name") != "flood_fusion_v1":
        raise RuntimeError("MANIFEST.json did not round-trip")
    log.info("MANIFEST.json written and read back (%s)", f"{proc}/MANIFEST.json")
    mirror(cfg, cc, cfg.results_dir, f"{proc}/results", (".csv", ".json", ".png", ".md"))
    mirror(cfg, cc, loc / "out" / "static", f"{proc}/static", (".tif",))
    mirror(cfg, cc, loc / "out" / "daily", f"{proc}/daily", (".tif",))
    if not a.skip_cubes:
        mirror(cfg, cc, loc / "cube" / "events", f"{proc}/cubes/events", (".nc",))
        mirror(cfg, cc, loc / "cube" / "sample", f"{proc}/cubes/sample", (".nc",))
    if not a.skip_raw:
        mirror(cfg, cc, loc / "raw" / "gfds", cfg.blob_path("raw", "gfds"), (".tif",))
        mirror(cfg, cc, loc / "raw" / "tile_index", cfg.blob_path("raw", "tile_index"), (".txt", ".json"))
        mirror(cfg, cc, loc / "raw" / "natural_earth", cfg.blob_path("raw", "natural_earth"), (".zip",))
        mirror(cfg, cc, loc / "raw" / "hydrobasins", cfg.blob_path("raw", "hydrobasins"), (".zip",))
        mirror(cfg, cc, loc / "raw" / "koppen", cfg.blob_path("raw", "koppen"), (".zip",))
        if not a.skip_gfds_bands:
            mirror(cfg, cc, loc / "gfds" / "days", cfg.blob_path("raw", "gfds", "signal_row_bands"), (".npz",))


if __name__ == "__main__":
    main()

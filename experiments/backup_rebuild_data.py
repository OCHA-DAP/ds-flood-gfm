"""Back up the GFDS-rebuild and FloodScan-reproduction data to the team's dev blob storage, with a manifest.

Inputs downloaded from outside (JRC archive, JAXA L3, NOAA/CSU grids, MERRA-2 from GEE) go under
projects/ds-flood-gfm/raw/...; files our scripts computed (signals, static tables, flooded fractions) go under
projects/ds-flood-gfm/processed/.... Uploads are skipped when a blob of the same name and size exists.
The manifest (path, bytes, sha256, blob name, origin) is written to data/backup_manifest.csv and printed, so
every file in the docs can be traced to its blob copy and its source.

    uv run python experiments/backup_rebuild_data.py            # upload + manifest
    uv run python experiments/backup_rebuild_data.py --dry-run  # manifest only
"""

import csv
import hashlib
import sys
import time
from pathlib import Path

import ocha_stratus as stratus

PREFIX = "ds-flood-gfm"
# (local directory or file, blob prefix, origin). Order = irreplaceability: JRC's archive first.
SETS = [
    ("data/gdacs_gfds/rebuild/jrc_amsr2_signal", "raw/gfds/rebuild/jrc_amsr2_signal", "JRC GDACS archive AMSR2/SignalTiffs (feed ended 2026-09-14)"),
    ("data/gdacs_gfds/rebuild/answer_key_20160820", "raw/gfds/rebuild/answer_key_20160820", "JRC GDACS archive per-sensor Source/Calibration/Position/Signal/Mag tiffs, 2016-08-20"),
    ("data/gdacs_gfds/rebuild/answer_key_20180820", "raw/gfds/rebuild/answer_key_20180820", "JRC GDACS archive per-sensor tiffs, 2018-08-20"),
    ("data/gdacs_gfds/rebuild/AMSR2", "raw/gfds/rebuild/AMSR2", "JRC GDACS archive AMSR2/AveragesAndSd baseline rasters"),
    ("data/gdacs_gfds/rebuild/ALL", "raw/gfds/rebuild/ALL", "JRC GDACS archive ALL/ (oceanmask, AveragesAndSd, SpecialTiffs)"),
    ("data/gdacs_gfds/rebuild/GPM", "raw/gfds/rebuild/GPM", "JRC GDACS archive GPM/ samples"),
    ("data/gdacs_gfds/rebuild/amsr2_l3_tb36", "raw/gfds/rebuild/amsr2_l3_tb36", "JAXA G-Portal AMSR2 L3 Tb 36 GHz 0.1 deg (02_fetch_amsr2_l3_tb36.py)"),
    ("data/floodscan_repro/csu_amsr2_grid", "raw/floodscan_repro/csu_amsr2_grid", "NOAA CSU AMSR2 FCDR/ICDR-GRID 0.25 deg, West Africa box, as npz (03_fetch_csu_amsr2_grid.py)"),
    ("data/floodscan_repro/merra2", "raw/floodscan_repro/merra2", "MERRA-2 TQV/T2M/PS from GEE (09_gee_merra2.py)"),
    ("data/floodscan_repro/gee_tiles", "raw/floodscan_repro/gee_tiles", "GEE static layers (NDVI, water) for 04_static_tables.py"),
    ("data/floodscan_repro/static_0p25.tif", "raw/floodscan_repro/static_0p25.tif", "static layers on the 0.25 deg box"),
    ("data/gdacs_gfds/rebuild/signal_l3", "processed/gfds_rebuild/signal_l3", "our GFDS signal from JAXA L3 (04_signal_from_l3.py)"),
    ("data/gdacs_gfds/rebuild/swath_signal", "processed/gfds_rebuild/swath_signal", "our GFDS signal from CSU swath footprints (07_signal_from_swath.py)"),
    ("data/floodscan_repro/static_tables.npz", "processed/floodscan_repro/static_tables.npz", "04_static_tables.py, years 2023+2025"),
    ("data/floodscan_repro/static_tables_2425.npz", "processed/floodscan_repro/static_tables_2425.npz", "04_static_tables.py, years 2024+2025 (hold-out)"),
    ("data/floodscan_repro/static_tables_atmos.npz", "processed/floodscan_repro/static_tables_atmos.npz", "04_static_tables.py with atmospheric correction"),
    ("data/floodscan_repro/nga2022_core_l3.npz", "processed/floodscan_repro/nga2022_core_l3.npz", "01_core_l3.py step 0"),
    ("data/floodscan_repro/swath_proto", "processed/floodscan_repro/swath_proto", "08/10 swath prototype and noise test cache"),
]
SETS += [(str(p), f"processed/floodscan_repro/{p.name}", "05_ff_chain.py flooded fractions") for p in sorted(Path("data/floodscan_repro").glob("ff_*.npz"))]


def sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def main(dry_run):
    client = None if dry_run else stratus.get_container_client(stage="dev", container_name="projects", write=True)
    rows, total, uploaded = [], 0, 0
    for local, blob_prefix, origin in SETS:
        local = Path(local)
        if not local.exists():
            raise FileNotFoundError(f"{local} missing; it is listed for backup")
        files = sorted(p for p in local.rglob("*") if p.is_file()) if local.is_dir() else [local]
        for p in files:
            rel = p.relative_to(local) if local.is_dir() else Path(p.name)
            name = f"{PREFIX}/{blob_prefix}/{rel}" if local.is_dir() else f"{PREFIX}/{blob_prefix}"
            size = p.stat().st_size
            total += size
            status = "dry-run"
            if client is not None:
                bc = client.get_blob_client(name)
                if bc.exists() and bc.get_blob_properties().size == size:
                    status = "exists"
                else:
                    t0 = time.time()
                    with open(p, "rb") as f:
                        bc.upload_blob(f, overwrite=True, max_concurrency=4)
                    status = f"uploaded {size / 1e6:.0f} MB in {time.time() - t0:.0f} s"
                    uploaded += size
            rows.append({"local": str(p), "bytes": size, "sha256": sha256(p), "blob": f"projects/{name}", "origin": origin, "status": status})
            print(f"{status:28s} {p}", flush=True)
    out = Path("data/backup_manifest.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} files, {total / 1e9:.2f} GB listed, {uploaded / 1e9:.2f} GB uploaded; manifest {out}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)

"""Download ERA5-Land monthly snow cover for the snow-season climatology.

Access route: Copernicus Climate Data Store API, credentials in ~/.cdsapirc
(the CDS's own mechanism; the team KB does not cover CDS). The request is
idempotent: an existing complete file is not re-downloaded.

Run: uv run python scripts/fusion/fetch_snow_climatology.py
"""

from __future__ import annotations

import argparse

import cdsapi

from ds_flood_gfm.fusion.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    s = cfg["static"]["snow"]
    y0, y1 = s["years"]
    out = cfg.local("raw", "era5_land", f"era5land_snow_cover_monthly_{y0}_{y1}_{s['grid_deg']}deg.nc")
    if out.exists() and out.stat().st_size > 0:
        print(f"exists, skipping: {out}")
        return
    tmp = out.with_suffix(".part")
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [s["variable"]],
        "year": [str(y) for y in range(y0, y1 + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "grid": [s["grid_deg"], s["grid_deg"]],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    print(f"requesting {s['cds_dataset']}: {request}")
    cdsapi.Client().retrieve(s["cds_dataset"], request, str(tmp))
    tmp.rename(out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

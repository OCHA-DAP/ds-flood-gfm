"""POC step 3: independent validation target — GFM Sentinel-1 flood extent.

Cumulative (max) GFM ensemble flood extent over the Oct 2022 peak window for
the confluence AOI, plus the observation dates used. Cached to netcdf; skip
if present.
Run: uv run python experiments/gfds_downscaling_poc/03_fetch_gfm_extent.py
"""
import json
from pathlib import Path

import numpy as np

from ds_flood_gfm.datasources.gfm import create_flood_composite, query_gfm_stac

AOI = [6.0, 6.3, 7.8, 8.3]
TARGET = "2022-10-14"
N_SEARCH = -10  # Oct 4 - Oct 14
OUT = Path("outputs/gfds_downscaling_poc")
OUT.mkdir(parents=True, exist_ok=True)

if (OUT / "gfm_extent_peak.nc").exists():
    print("already cached:", OUT / "gfm_extent_peak.nc")
    raise SystemExit(0)

items = query_gfm_stac(AOI, TARGET, n_search=N_SEARCH)
if len(items) == 0:
    raise RuntimeError(
        f"GFM STAC returned zero items for {AOI} around {TARGET}. "
        "That is unexpected for 2022 — check collection name / API status "
        "before concluding there is no data."
    )
composite, unique_dates = create_flood_composite(
    items, AOI, n_images=12, mode="cumulative", n_search=N_SEARCH
)
composite = composite.astype("float32")
# drop object-typed coords/attrs (e.g. flood_members) that netcdf can't store
composite = composite.reset_coords(drop=True)
composite.attrs = {}
composite = composite.rename("gfm_flood")
composite.rio.write_crs("EPSG:4326", inplace=True)
composite.to_netcdf(OUT / "gfm_extent_peak.nc")
(OUT / "gfm_extent_dates.json").write_text(
    json.dumps([str(d) for d in np.asarray(unique_dates).tolist()])
)
print("dates used:", unique_dates)
print("composite:", dict(composite.sizes), "| wet px:",
      int(np.nansum(composite.values > 0)))

"""Fetch FloodScan SFED daily windows over Nigeria, Jun-Dec 2022, from team blob."""
import sys
import time
from pathlib import Path

import numpy as np
import ocha_stratus as stratus
import pandas as pd

BBOX = (2.7, 4.0, 14.7, 14.0)
CACHE = Path("data/gdacs_gfds/nga2022")
CACHE.mkdir(parents=True, exist_ok=True)

dates = pd.date_range("2022-06-01", "2022-12-31", freq="D")
n = 0
for ts in dates:
    d = ts.date()
    npy = CACHE / f"sfed_{d:%Y%m%d}.npy"
    miss = CACHE / f"sfed_{d:%Y%m%d}.missing"
    if npy.exists() or miss.exists():
        continue
    name = f"raster/cogs/aer_area_300s_{d:%Y%m%d}_v05r01.tif"
    sub = None
    for i in range(3):
        try:
            da = stratus.open_blob_cog(name, container_name="global")
            sub = da.sel(band=1).rio.clip_box(*BBOX)
            sub.load()
            break
        except Exception as e:
            if "BlobNotFound" in str(e) or "404" in str(e):
                miss.touch()
                print(f"MISSING in blob: sfed {d}", flush=True)
                break
            wait = 15 * (i + 1)
            print(f"retry {i+1}/3 on {name}: {e} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"SFED fetch failed after 3 attempts: {name}")
    if sub is None:
        continue
    if not (CACHE / "sfed_coords.npz").exists():
        np.savez(CACHE / "sfed_coords.npz", x=sub.x.values, y=sub.y.values)
    np.save(npy, sub.values.astype("float32"))
    n += 1
    if n % 25 == 0:
        print(f"{n} SFED days fetched (latest {d})", flush=True)

print(f"DONE: {len(list(CACHE.glob('sfed_*.npy')))} SFED days, "
      f"{len(list(CACHE.glob('sfed_*.missing')))} missing", flush=True)

"""POC step 2: build the downscaling prior for the confluence AOI.

Reads JRC Global Surface Water occurrence (1984-2021, ~28 m) for the AOI via
HTTP range requests, aggregates 3x3 to 0.00075 deg (~83 m) so that each GFDS
0.09 deg cell contains exactly 120x120 prior cells, and saves:
- occ (mean occurrence %, float32)
- permanent water mask (occurrence > 80%)
Run: uv run python experiments/gfds_downscaling_poc/02_build_prior.py
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

AOI = (6.0, 6.3, 7.8, 8.3)  # lon_min, lat_min, lon_max, lat_max
OUT = Path("outputs/gfds_downscaling_poc")
OUT.mkdir(parents=True, exist_ok=True)
URL = ("/vsicurl/https://storage.googleapis.com/global-surface-water/"
       "downloads2021/occurrence/occurrence_0E_10Nv1_4_2021.tif")

with rasterio.open(URL) as src:
    win = from_bounds(*AOI, src.transform)
    occ = src.read(1, window=win)
    tr = src.window_transform(win)
print("native window:", occ.shape)

# GSW encoding: 0-100 = occurrence %, 255 = nodata over land/ocean
occf = occ.astype("float32")
occf[occ == 255] = 0.0  # never-observed-as-water -> prior 0

ny, nx = occf.shape
ny3, nx3 = ny // 3 * 3, nx // 3 * 3
agg = occf[:ny3, :nx3].reshape(ny3 // 3, 3, nx3 // 3, 3).mean(axis=(1, 3))
xs = tr.c + (np.arange(nx3 // 3) * 3 + 1.5) * tr.a
ys = tr.f + (np.arange(ny3 // 3) * 3 + 1.5) * tr.e

perm = agg > 80.0
print(f"prior grid: {agg.shape} at 0.00075 deg | permanent-water px: {perm.sum():,} "
      f"({100 * perm.mean():.2f}%) | occ>0 px: {(agg > 0).sum():,}")
np.savez_compressed(OUT / "gsw_prior_aoi.npz", occ=agg.astype("float32"),
                    perm=perm, x=xs, y=ys)
print("wrote", OUT / "gsw_prior_aoi.npz")

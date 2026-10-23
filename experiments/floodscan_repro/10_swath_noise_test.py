"""Does the 0.25-deg daily grid add day-to-day Q noise that AER's fixed 22 km footprints on a 5' grid do not?

MDFF is 3 sigma of the 7-day Q variance (ADD s. 4.20). The CSU daily grid averages a different mix of
footprints into each 0.25-deg cell every day; footprint matching onto a fixed grid should not. Two weeks of
orbit files (swath.py) give Q at 5' and, aggregated 3x3, at 0.25 deg; the CSU grid gives Q at 0.25 deg for
the same days. Compared per pass over inland land cells (f_pow <= 0.5, no ocean), cells valid on >= 4 days:
  var7 of Q at 0.25 deg: CSU grid vs 3x3 mean of our 5' grid    (same cells, same days -> gridding noise)
  var7 of Q at 5'                                                 (the noise the chain would see at 5')
Output: data/floodscan_repro/swath_proto/q36_5min_<start>_<end>.npz and a printed table.

    uv run --no-project --with h5py --with fsspec --with aiohttp --with numpy --with scipy --with rasterio \
        python experiments/floodscan_repro/10_swath_noise_test.py 2024-09-01 2024-09-14
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import rasterio

from swath import NX, NY, grid_day, q_of

GRID_SRC = Path("data/floodscan_repro/csu_amsr2_grid")
STATIC = Path("data/floodscan_repro/static_0p25.tif")
OUT = Path("data/floodscan_repro/swath_proto")
d0, d1 = (dt.date.fromisoformat(a) for a in sys.argv[1:3])
days = [d0 + dt.timedelta(k) for k in range((d1 - d0).days + 1)]

with rasterio.open(STATIC) as ds:
    names = list(ds.descriptions)
    fpow, focean = ds.read(names.index("fpow") + 1), ds.read(names.index("focean") + 1)
land = (fpow <= 0.5) & (focean <= 0.001)

OUT.mkdir(parents=True, exist_ok=True)
q5 = {p: np.full((len(days), NY, NX), np.nan) for p in ("asc", "dsc")}
qg = {p: np.full((len(days), 80, 80), np.nan) for p in ("asc", "dsc")}
total_mb = 0.0
for i, d in enumerate(days):
    cache = OUT / f"q36_5min_{d:%Y%m%d}.npz"
    if cache.exists() and all(f"q_{p}" in np.load(cache).files for p in ("asc", "dsc")):
        z = np.load(cache)
        for p in ("asc", "dsc"):
            q5[p][i] = z[f"q_{p}"]
    else:
        g, mb = grid_day(d)
        total_mb += mb
        for p in ("asc", "dsc"):
            q5[p][i] = q_of(*g[p]["36"])
        np.savez_compressed(cache, **{f"q_{p}": q5[p][i] for p in ("asc", "dsc")})
        print(f"{d}: {mb:.0f} MB, 5' valid asc {np.isfinite(q5['asc'][i]).mean():.2f} dsc {np.isfinite(q5['dsc'][i]).mean():.2f}", flush=True)
    z = np.load(GRID_SRC / f"{d:%Y}" / f"{d:%Y%m%d}.npz")
    for p in ("asc", "dsc"):
        qg[p][i] = q_of(np.flipud(z[f"fcdr_tb36v_{p}"]), np.flipud(z[f"fcdr_tb36h_{p}"]))
print(f"streamed {total_mb:.0f} MB for {len(days)} days")


def var7(x):
    """Variance over the whole window per cell, cells with >= 4 valid days (a 14-day window: two 7-day values)."""
    out = []
    for k in range(0, len(x) - 6, 7):
        w = x[k : k + 7]
        with np.errstate(all="ignore"):
            out.append(np.where(np.isfinite(w).sum(0) >= 4, np.nanvar(w, 0), np.nan))
    return np.stack(out)


for p in ("asc", "dsc"):
    q3 = np.nanmean(q5[p].reshape(len(days), 80, 3, 80, 3), (2, 4))
    vg, v3 = var7(qg[p]), var7(q3)
    v5 = var7(q5[p])
    land5 = np.repeat(np.repeat(land, 3, 0), 3, 1)
    ok = np.isfinite(vg) & np.isfinite(v3) & land[None]
    print(f"\n{p}: cells x weeks compared {int(ok.sum())}")
    print(f"  sigma7 (sqrt of median var7) at 0.25 deg: CSU grid {np.sqrt(np.nanmedian(vg[ok])):.4f}, "
          f"ours from 5' {np.sqrt(np.nanmedian(v3[ok])):.4f}; ratio of medians {np.nanmedian(vg[ok]) / np.nanmedian(v3[ok]):.2f}")
    ok5 = np.isfinite(v5) & land5[None]
    print(f"  sigma7 at 5': {np.sqrt(np.nanmedian(v5[ok5])):.4f} (n {int(ok5.sum())}); p75 ratio grid/ours at 0.25: "
          f"{np.nanpercentile(vg[ok], 75) / np.nanpercentile(v3[ok], 75):.2f}")
    # same-day level agreement
    okd = np.isfinite(qg[p]) & np.isfinite(q3) & land[None]
    print(f"  same-day Q, 0.25 deg: r {np.corrcoef(qg[p][okd], q3[okd])[0, 1]:.4f}, bias {np.mean(q3[okd] - qg[p][okd]):+.4f}, "
          f"grid valid {np.isfinite(qg[p])[:, land].mean():.2f}, ours {np.isfinite(q3)[:, land].mean():.2f}")
np.savez_compressed(OUT / f"q36_5min_{d0:%Y%m%d}_{d1:%Y%m%d}.npz", days=np.array([d.isoformat() for d in days]), **{f"q_{p}": q5[p] for p in q5})

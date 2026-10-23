"""Swath-route prototype, one day: CSU AMSR2 orbit files -> 36 GHz Q on AER's 5-arcminute grid (swath.py),
checked against the CSU 0.25-deg grid for the same day and pass. Result 2024-09-01: 46 MB streamed,
r 0.99 and no bias after 3x3 aggregation, coverage 0.55 / 0.94 (asc / dsc) vs the grid's 0.52 / 0.75.

    uv run --no-project --with h5py --with fsspec --with aiohttp --with numpy --with scipy \
        python experiments/floodscan_repro/08_swath_prototype.py 2024-09-01
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np

from swath import GLAT, GLON, grid_day, q_of

GRID_SRC = Path("data/floodscan_repro/csu_amsr2_grid")
OUT = Path("data/floodscan_repro/swath_proto")
day = dt.date.fromisoformat(sys.argv[1])

g, mb = grid_day(day)
print(f"{day}: streamed {mb:.1f} MB")
q5 = {p: q_of(*g[p]["36"]) for p in ("asc", "dsc")}
z = np.load(GRID_SRC / f"{day:%Y}" / f"{day:%Y%m%d}.npz")
for p, q in q5.items():
    qg = q_of(np.flipud(z[f"fcdr_tb36v_{p}"]), np.flipud(z[f"fcdr_tb36h_{p}"]))
    q3 = np.nanmean(q.reshape(80, 3, 80, 3), (1, 3))
    ok = np.isfinite(q3) & np.isfinite(qg)
    print(f"{p}: 5' valid {np.isfinite(q).mean():.3f}; 5'->0.25 vs CSU grid: n {int(ok.sum())}, "
          f"r {np.corrcoef(q3[ok], qg[ok])[0, 1]:.4f}, bias {np.mean(q3[ok] - qg[ok]):+.4f}, "
          f"mad {np.median(np.abs(q3[ok] - qg[ok])):.4f}; grid valid {np.isfinite(qg).mean():.3f}, ours {np.isfinite(q3).mean():.3f}")
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / f"q36_5min_{day:%Y%m%d}.npz", **{f"q_{p}": q for p, q in q5.items()}, lat=GLAT, lon=GLON)
print(f"wrote {OUT / f'q36_5min_{day:%Y%m%d}.npz'}")

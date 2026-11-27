"""Where do our L3-grid signal and JRC's AMSR2 signal disagree? One day, land only."""
import sys
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

day = sys.argv[1]
W = "data/gdacs_gfds/rebuild"  # run from the repo root
lat09 = 90 - 0.09 * (np.arange(2000) + 0.5)
lon09 = -180 + 0.09 * (np.arange(4000) + 0.5)
r10 = np.clip(((90 - lat09) / 0.1).astype(int), 0, 1799)
c10 = np.clip(((lon09 + 180) / 0.1).astype(int), 0, 3599)
with rasterio.open(f"{W}/ALL/oceanmask.tif") as ds:
    land = ds.read(1) == 0
with rasterio.open(f"{W}/jrc_amsr2_signal/signal_{day}_HD5.tif") as ds:
    a = ds.read(1)
j = a / 1e6
j[(a == -32000) | (a == np.iinfo("int32").min) | (j <= 0)] = np.nan
with rasterio.open(f"{W}/signal_l3/{day[:4]}/s_{day}.tif") as ds:
    o = ds.read(1).astype("float64")[np.ix_(r10, c10)]
ok = land & np.isfinite(o) & np.isfinite(j)
d = np.abs(o - j)
coast = binary_dilation(~land, iterations=3) & land  # land within ~3 cells (~27 km) of ocean
lat2 = np.repeat(lat09[:, None], 4000, 1)
groups = {
    "all land": ok,
    "coastal (<~27 km)": ok & coast,
    "inland": ok & ~coast,
    "|lat| < 30": ok & (np.abs(lat2) < 30),
    "30-55": ok & (np.abs(lat2) >= 30) & (np.abs(lat2) < 55),
    "> 55": ok & (np.abs(lat2) >= 55),
    "JRC s < 0.95 (wetter)": ok & (j < 0.95),
    "JRC s >= 0.95": ok & (j >= 0.95),
}
print(f"{day}: median |ours - JRC| and median (ours - JRC) by group")
for k, m in groups.items():
    print(f"  {k:24s} n={m.sum():9,d}  med|d|={np.median(d[m]):.5f}  med d={np.median((o - j)[m]):+.5f}  r={np.corrcoef(o[m], j[m])[0, 1]:.3f}")

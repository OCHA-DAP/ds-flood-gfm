"""Compare one day of GFDS per-sensor signal coverage (AMSR2 vs GPM vs ALL merge).

Valid = not a sentinel (-32000, int32-min) and signal > 0, as in fusion/sources/gfds.py.
Reports valid pixel counts by latitude band and AMSR2-vs-GPM agreement where both are valid.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio

SENTINELS = (-32000, np.iinfo("int32").min)
d = Path(sys.argv[1])


def load(name):
    with rasterio.open(d / name) as ds:
        a = ds.read(1)
        assert a.shape == (2000, 4000), a.shape
        lat = ds.transform.f + (np.arange(a.shape[0]) + 0.5) * ds.transform.e
    v = a.astype("float64") / 1e6
    for s in SENTINELS:
        v[a == s] = np.nan
    v[v <= 0] = np.nan
    return v, lat


amsr2, lat = load("amsr2_20250815.tif")
gpm, _ = load("gpm_20250815.tif")
merged, _ = load("all_20250815.tif")

bands = {
    "> 65N": lat > 65,
    "65S-65N": (lat >= -65) & (lat <= 65),
    "< 65S": lat < -65,
}
print(f"{'band':10s} {'AMSR2':>10s} {'GPM':>10s} {'ALL':>10s}")
for name, rows in bands.items():
    c = [int(np.isfinite(x[rows]).sum()) for x in (amsr2, gpm, merged)]
    print(f"{name:10s} {c[0]:10d} {c[1]:10d} {c[2]:10d}")

mid = bands["65S-65N"][:, None] & np.ones((1, 4000), bool)
both = np.isfinite(amsr2) & np.isfinite(gpm) & mid
only_a = np.isfinite(amsr2) & ~np.isfinite(gpm) & mid
only_g = ~np.isfinite(amsr2) & np.isfinite(gpm) & mid
print(f"\n65S-65N: both valid {both.sum()}, AMSR2 only {only_a.sum()}, GPM only {only_g.sum()}")
r = np.corrcoef(amsr2[both], gpm[both])[0, 1]
diff = gpm[both] - amsr2[both]
print(f"AMSR2 vs GPM signal where both valid: r = {r:.3f}, median diff (GPM-AMSR2) = {np.median(diff):+.4f}, "
      f"IQR of diff = {np.percentile(diff, 25):+.4f}..{np.percentile(diff, 75):+.4f}")
print(f"median signal AMSR2 {np.median(amsr2[both]):.4f}, GPM {np.median(gpm[both]):.4f}")

# Land only: GFDS oceanmask.tif, 1 = ocean, 0 = land (checked: rows south of 70S are ~all 0)
with rasterio.open(d / "oceanmask.tif") as ds:
    om = ds.read(1)
land = om == 0
print(f"\nsanity: land share of rows south of 70S = {land[lat < -70].mean():.3f}; of rows 0-10N = {land[(lat > 0) & (lat < 10)].mean():.3f}")
print(f"{'land band':10s} {'pixels':>9s} {'AMSR2':>7s} {'GPM':>7s} {'ALL':>7s}  (share of land pixels valid)")
for name, rows in bands.items():
    m = land & rows[:, None]
    n = int(m.sum())
    shares = [np.isfinite(x)[m].mean() for x in (amsr2, gpm, merged)]
    print(f"{name:10s} {n:9d} {shares[0]:7.3f} {shares[1]:7.3f} {shares[2]:7.3f}")
lb = both & land
rl = np.corrcoef(amsr2[lb], gpm[lb])[0, 1]
dl = gpm[lb] - amsr2[lb]
print(f"land, 65S-65N, both valid: n = {lb.sum()}, r = {rl:.3f}, median diff = {np.median(dl):+.4f}, "
      f"IQR {np.percentile(dl, 25):+.4f}..{np.percentile(dl, 75):+.4f}, medians AMSR2 {np.median(amsr2[lb]):.4f} GPM {np.median(gpm[lb]):.4f}")

# Land only: GFDS oceanmask.tif, 1 = ocean, 0 = land (checked: rows south of 70S are ~all 0)
with rasterio.open(d / "oceanmask.tif") as ds:
    om = ds.read(1)
land = om == 0
print(f"\nsanity: land share of rows south of 70S = {land[lat < -70].mean():.3f}; of rows 0-10N = {land[(lat > 0) & (lat < 10)].mean():.3f}")
print(f"{'land band':10s} {'pixels':>9s} {'AMSR2':>7s} {'GPM':>7s} {'ALL':>7s}  (share of land pixels valid)")
for name, rows in bands.items():
    m = land & rows[:, None]
    n = int(m.sum())
    shares = [np.isfinite(x)[m].mean() for x in (amsr2, gpm, merged)]
    print(f"{name:10s} {n:9d} {shares[0]:7.3f} {shares[1]:7.3f} {shares[2]:7.3f}")
lb = both & land
rl = np.corrcoef(amsr2[lb], gpm[lb])[0, 1]
dl = gpm[lb] - amsr2[lb]
print(f"land, 65S-65N, both valid: n = {lb.sum()}, r = {rl:.3f}, median diff = {np.median(dl):+.4f}, "
      f"IQR {np.percentile(dl, 25):+.4f}..{np.percentile(dl, 75):+.4f}, medians AMSR2 {np.median(amsr2[lb]):.4f} GPM {np.median(gpm[lb]):.4f}")

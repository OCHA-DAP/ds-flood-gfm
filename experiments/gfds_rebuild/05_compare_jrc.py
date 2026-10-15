"""Gate 1a: our L3-grid AMSR2 signal (04) against JRC's own AMSR2 signal (03), land only.

Our 0.1-deg cells are sampled at JRC's 0.09-deg cell centres (nearest containing cell).
Land = GFDS oceanmask.tif == 0. JRC invalid = sentinels (-32000, int32-min) or signal <= 0.

Reports:
1. pooled pixel-days: r, median (ours - JRC), median |diff|  -> gate: r >= 0.95, median |diff| < 0.005
2. per-pixel temporal r (pixels with >= MIN_DAYS shared days), overall and for the 10% most
   variable pixels by JRC temporal sd (floodplains, where the signal moves)
3. Niger-Benue confluence AOI (6.0-7.8E, 6.3-8.3N): daily AOI-mean series, r and bias
4. coverage: share of land cells observed, ours vs JRC

    uv run --no-project --with numpy --with rasterio \
        python experiments/gfds_rebuild/05_compare_jrc.py 2022-10-01 2022-10-31
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import rasterio

OURS = Path("data/gdacs_gfds/rebuild/signal_l3")
JRC = Path("data/gdacs_gfds/rebuild/jrc_amsr2_signal")
MASK = Path("data/gdacs_gfds/rebuild/ALL/oceanmask.tif")
SENTINELS = (-32000, np.iinfo("int32").min)
MIN_DAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 20  # lower only for a quick partial look
AOI = (6.0, 6.3, 7.8, 8.3)  # lon0, lat0, lon1, lat1

start, end = (dt.date.fromisoformat(a) for a in sys.argv[1:3])
days = [start + dt.timedelta(k) for k in range((end - start).days + 1)]

lat09 = 90 - 0.09 * (np.arange(2000) + 0.5)
lon09 = -180 + 0.09 * (np.arange(4000) + 0.5)
r10 = np.clip(((90 - lat09) / 0.1).astype(int), 0, 1799)
c10 = np.clip(((lon09 + 180) / 0.1).astype(int), 0, 3599)
with rasterio.open(MASK) as ds:
    land = ds.read(1) == 0
aoi = np.zeros_like(land)
aoi[np.ix_((lat09 >= AOI[1]) & (lat09 <= AOI[3]), (lon09 >= AOI[0]) & (lon09 <= AOI[2]))] = True

ours, jrc, used = [], [], []
for d in days:
    pj = JRC / f"signal_{d:%Y%m%d}_HD5.tif"
    po = OURS / f"{d:%Y}" / f"s_{d:%Y%m%d}.tif"
    if not pj.exists():
        print(f"{d}: no JRC file (see {JRC}/missing.txt)")
        continue
    if not po.exists():
        raise FileNotFoundError(f"{d}: our signal missing ({po}); run 04 first")
    with rasterio.open(pj) as ds:
        a = ds.read(1)
    j = a.astype("float64") / 1e6
    for s in SENTINELS:
        j[a == s] = np.nan
    j[j <= 0] = np.nan
    with rasterio.open(po) as ds:
        o = ds.read(1).astype("float64")[np.ix_(r10, c10)]
    o[~land] = np.nan
    j[~land] = np.nan
    ours.append(o.astype("float32"))
    jrc.append(j.astype("float32"))
    used.append(d)

O, J = np.stack(ours), np.stack(jrc)
both = np.isfinite(O) & np.isfinite(J)
diff = (O - J)[both]
r_pool = np.corrcoef(O[both], J[both])[0, 1]
print(f"\ndays compared: {len(used)} ({used[0]} .. {used[-1]})")
print(f"coverage of land cells: ours {np.isfinite(O)[:, land].mean():.3f}, JRC {np.isfinite(J)[:, land].mean():.3f}")
print("\n1. pooled pixel-days (land)")
print(f"   n = {both.sum():,}  r = {r_pool:.4f}  median diff = {np.median(diff):+.5f}  "
      f"median |diff| = {np.median(np.abs(diff)):.5f}  IQR {np.percentile(diff, 25):+.5f}..{np.percentile(diff, 75):+.5f}")
print(f"   GATE 1a (r >= 0.95 and median |diff| < 0.005): "
      f"{'PASS' if r_pool >= 0.95 and np.median(np.abs(diff)) < 0.005 else 'FAIL'}")

# 1b. same comparison after block-averaging both (on the shared both-valid land cells):
# 5x5 JRC cells = 0.45 deg, 10x10 = 0.9 deg (0.9 is the common multiple of 0.09 and 0.1).
# Separates small-scale noise (averages out) from real disagreement (survives).
print("\n1b. pooled after block-averaging (block mean over shared valid land cells, >= 50% of block)")
for k, label in ((5, "0.45 deg"), (10, "0.9 deg")):
    sh = (O.shape[0], 2000 // k, k, 4000 // k, k)
    w = both.reshape(sh).sum((2, 4))
    with np.errstate(invalid="ignore"):
        ob = np.where(both, O, 0).reshape(sh).sum((2, 4)) / w
        jb = np.where(both, J, 0).reshape(sh).sum((2, 4)) / w
    keep = w >= 0.5 * k * k
    db = (ob - jb)[keep]
    print(f"   {label}: n = {keep.sum():,}  r = {np.corrcoef(ob[keep], jb[keep])[0, 1]:.4f}  "
          f"median |diff| = {np.median(np.abs(db)):.5f}  IQR {np.percentile(db, 25):+.5f}..{np.percentile(db, 75):+.5f}")

# 1c. is the flood contrast preserved? Inland land (>~27 km from ocean), both in 0.7..1.05.
# Binning on one noisy variable fakes compression (regression to the mean), so use the sd ratio,
# the orthogonal (total least squares) slope, and bins on the mean of both.
from scipy.ndimage import binary_dilation

inland = land & ~binary_dilation(~land, iterations=3)
sel = both & inland[None] & (O > 0.7) & (O < 1.05) & (J > 0.7) & (J < 1.05)
o1, j1 = O[sel].astype("float64"), J[sel].astype("float64")
w_, v_ = np.linalg.eigh(np.cov(np.vstack([j1 - j1.mean(), o1 - o1.mean()])))
print(f"\n1c. flood contrast, inland pixel-days (n = {sel.sum():,}): sd ours / sd JRC = {o1.std() / j1.std():.3f}, "
      f"orthogonal slope = {v_[1, 1] / v_[0, 1]:.3f}")
mb = (o1 + j1) / 2
edges = [0.80, 0.85, 0.90, 0.93, 0.96, 0.98, 1.00]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (mb >= lo) & (mb < hi)
    print(f"   mean-of-both {lo:.2f}-{hi:.2f}: n = {m.sum():9,d}  mean(ours - JRC) = {np.mean(o1[m] - j1[m]):+.4f}")
del o1, j1, mb

# 2. per-pixel temporal correlation
n = both.sum(0)
ok = n >= MIN_DAYS
Om = np.where(both, O, np.nan)
Jm = np.where(both, J, np.nan)
with np.errstate(invalid="ignore", divide="ignore"):
    oa = Om - np.nanmean(Om, 0)
    ja = Jm - np.nanmean(Jm, 0)
    rt = np.nansum(oa * ja, 0) / np.sqrt(np.nansum(oa**2, 0) * np.nansum(ja**2, 0))
    jsd = np.nanstd(Jm, 0)
rt_ok = rt[ok & np.isfinite(rt)]
thr = np.nanpercentile(jsd[ok], 90)
dyn = ok & (jsd >= thr) & np.isfinite(rt)
print(f"\n2. per-pixel temporal r (land pixels with >= {MIN_DAYS} shared days: {ok.sum():,})")
print(f"   all pixels:          median {np.median(rt_ok):.3f}  (p25 {np.percentile(rt_ok, 25):.3f}, p75 {np.percentile(rt_ok, 75):.3f})")
print(f"   top-10% variable px: median {np.median(rt[dyn]):.3f}  (p25 {np.percentile(rt[dyn], 25):.3f}, p75 {np.percentile(rt[dyn], 75):.3f}); "
      f"JRC temporal sd >= {thr:.4f}, n = {dyn.sum():,}  (selecting on JRC's sd partly selects JRC noise)")


def trail4(x):
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - 3) : i + 1]
        k = np.isfinite(w).sum(0)
        out[i] = np.where(k > 0, np.nansum(w, 0) / np.maximum(k, 1), np.nan)
    return out


O4, J4 = trail4(Om), trail4(Jm)  # same population as above: shared valid land pixel-days
with np.errstate(invalid="ignore", divide="ignore"):
    oa4, ja4 = O4 - np.nanmean(O4, 0), J4 - np.nanmean(J4, 0)
    rt4 = np.nansum(oa4 * ja4, 0) / np.sqrt(np.nansum(oa4**2, 0) * np.nansum(ja4**2, 0))
ok4 = ok & np.isfinite(rt4)
print(f"   4-day trailing means, same pixels: median {np.median(rt4[ok4]):.3f}  (n = {ok4.sum():,})")

# 3. Niger-Benue confluence AOI
ao = np.array([np.nanmean(np.where(both[k] & aoi, O[k], np.nan)) for k in range(len(used))])
aj = np.array([np.nanmean(np.where(both[k] & aoi, J[k], np.nan)) for k in range(len(used))])
m = np.isfinite(ao) & np.isfinite(aj)
print(f"\n3. Niger-Benue confluence AOI mean series ({m.sum()} days): r = {np.corrcoef(ao[m], aj[m])[0, 1]:.3f}, "
      f"mean bias (ours - JRC) = {np.mean(ao[m] - aj[m]):+.5f}")
for d, x, y in zip(used, ao, aj):
    print(f"   {d}  ours {x:.4f}  JRC {y:.4f}")

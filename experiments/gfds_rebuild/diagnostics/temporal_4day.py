"""Per-pixel temporal r, ours vs JRC AMSR2, October 2022: daily vs 4-day trailing mean,
and pixels selected by the MEAN of both sds (not JRC's alone, which selects JRC noise)."""
import datetime as dt
import numpy as np
import rasterio

W = "data/gdacs_gfds/rebuild"  # run from the repo root
lat09 = 90 - 0.09 * (np.arange(2000) + 0.5)
lon09 = -180 + 0.09 * (np.arange(4000) + 0.5)
r10 = np.clip(((90 - lat09) / 0.1).astype(int), 0, 1799)
c10 = np.clip(((lon09 + 180) / 0.1).astype(int), 0, 3599)
land = rasterio.open(f"{W}/ALL/oceanmask.tif").read(1) == 0
mid = (np.abs(lat09) < 60)[:, None] & land  # skip polar/snow
O, J = [], []
for k in range(31):
    d = dt.date(2022, 10, 1) + dt.timedelta(k)
    a = rasterio.open(f"{W}/jrc_amsr2_signal/signal_{d:%Y%m%d}_HD5.tif").read(1)
    j = a / 1e6
    j[(a == -32000) | (a == np.iinfo("int32").min) | (j <= 0)] = np.nan
    o = rasterio.open(f"{W}/signal_l3/2022/s_{d:%Y%m%d}.tif").read(1).astype("float64")[np.ix_(r10, c10)]
    o[~mid] = np.nan
    j[~mid] = np.nan
    O.append(o.astype("float32")); J.append(j.astype("float32"))
O, J = np.stack(O), np.stack(J)


def trail4(x):
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - 3): i + 1]
        n = np.isfinite(w).sum(0)
        out[i] = np.where(n > 0, np.nansum(w, 0) / np.maximum(n, 1), np.nan)
    return out


def temporal_r(o, j, min_days=20):
    both = np.isfinite(o) & np.isfinite(j)
    om, jm = np.where(both, o, np.nan), np.where(both, j, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        oa, ja = om - np.nanmean(om, 0), jm - np.nanmean(jm, 0)
        r = np.nansum(oa * ja, 0) / np.sqrt(np.nansum(oa**2, 0) * np.nansum(ja**2, 0))
        sd = (np.nanstd(om, 0) + np.nanstd(jm, 0)) / 2
    ok = (both.sum(0) >= min_days) & np.isfinite(r)
    return r, sd, ok


for label, (o, j) in {"daily": (O, J), "4-day mean": (trail4(O), trail4(J))}.items():
    r, sd, ok = temporal_r(o, j)
    thr = np.nanpercentile(sd[ok], 90)
    top = ok & (sd >= thr)
    print(f"{label:11s}: all land 60S-60N median r {np.median(r[ok]):.3f} (n {ok.sum():,});  "
          f"top-10% by mean sd of both: median r {np.median(r[top]):.3f} (n {top.sum():,})")

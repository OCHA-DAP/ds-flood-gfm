"""Which footprint-to-cell scheme and day window does JRC's SourceTiff (T_M) use? Offline, from the per-footprint
samples saved by 07 and JRC's source file for the day.

Schemes for the cell value: mean of footprints whose centre falls in the cell; nearest footprint to the cell
centre; inverse-distance mean within a radius; Gaussian within a radius. Day windows: all footprints of the
UTC day; ascending only; descending only; descending before/after 12 UTC. JRC's grid is 100% filled over
land, so the scheme must also fill cells without a footprint centre (nearest / IDW do; the mean does not).

    uv run --no-project --with numpy --with scipy --with rasterio python experiments/gfds_rebuild/08_gridding_schemes.py 2018-08-20
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

RES = 0.09
LON0, LON1, LAT0, LAT1 = 0.0, 20.0, 0.0, 20.0
R0, R1 = int((90 - LAT1) / RES), int(np.ceil((90 - LAT0) / RES))
C0, C1 = int((LON0 + 180) / RES), int(np.ceil((LON1 + 180) / RES))
NR, NC = R1 - R0, C1 - C0
day = dt.date.fromisoformat(sys.argv[1])
KEY = Path(f"data/gdacs_gfds/rebuild/answer_key_{day:%Y%m%d}")
z = np.load(f"data/gdacs_gfds/rebuild/swath_signal/{day:%Y%m%d}.npz")
fp = z["footprints"]  # lat, lon, M, C_lin, C_rank, seconds since 00 UTC, ascending
lat, lon, M, tsec, asc = fp[:, 0], fp[:, 1], fp[:, 2], fp[:, 5], fp[:, 6] > 0.5


def crop(path):
    with rasterio.open(path) as ds:
        a = ds.read(1, window=Window(C0, R0, NC, NR)).astype("float64")
    a[(a == -32000) | (a == np.iinfo("int32").min)] = np.nan
    return a


src = next(KEY.glob(f"{day:%Y%m%d}_HD5*.tif"))
jrc = crop(src) * 0.01
with rasterio.open("data/gdacs_gfds/rebuild/ALL/oceanmask.tif") as ds:
    land = ds.read(1, window=Window(C0, R0, NC, NR)) == 0
glat = 90 - RES * (np.arange(R0, R1) + 0.5)
glon = -180 + RES * (np.arange(C0, C1) + 0.5)
GY, GX = np.meshgrid(glat, glon, indexing="ij")
KM = 111.2
cos10 = np.cos(np.deg2rad(10.0))
gpts = np.column_stack([GY.ravel() * KM, GX.ravel() * KM * cos10])


def scheme(sel, kind, radius_km=10.0):
    pts = np.column_stack([lat[sel] * KM, lon[sel] * KM * cos10])
    vals = M[sel]
    out = np.full(NR * NC, np.nan)
    if kind == "cellmean":
        rr = ((90 - lat[sel]) / RES).astype(int) - R0
        cc = ((lon[sel] + 180) / RES).astype(int) - C0
        ok = (rr >= 0) & (rr < NR) & (cc >= 0) & (cc < NC)
        s, n = np.zeros((NR, NC)), np.zeros((NR, NC))
        np.add.at(s, (rr[ok], cc[ok]), vals[ok]); np.add.at(n, (rr[ok], cc[ok]), 1)
        return np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
    tree = cKDTree(pts)
    if kind == "nearest":
        d, i = tree.query(gpts, distance_upper_bound=radius_km)
        ok = np.isfinite(d)
        out[ok] = vals[i[ok]]
        return out.reshape(NR, NC)
    groups = tree.query_ball_point(gpts, r=radius_km)
    for gi, idx in enumerate(groups):
        if not idx:
            continue
        idx = np.asarray(idx)
        d2 = ((pts[idx] - gpts[gi]) ** 2).sum(1)
        w = 1 / np.maximum(np.sqrt(d2), 0.5) if kind == "idw" else np.exp(-0.5 * d2 / (radius_km / 2) ** 2)
        out[gi] = (w * vals[idx]).sum() / w.sum()
    return out.reshape(NR, NC)


def report(name, a):
    ok = np.isfinite(a) & np.isfinite(jrc) & land
    d = a[ok] - jrc[ok]
    print(f"{name:44s} n {int(ok.sum()):6d} cov {np.isfinite(a)[land].mean():.3f}  r {np.corrcoef(a[ok], jrc[ok])[0, 1]:.4f}  "
          f"bias {d.mean():+.3f}  mad {np.median(np.abs(d)):.3f}  |d|<0.05K {(np.abs(d) < 0.05).mean():.3f}")


hour = tsec / 3600  # hours since 00 UTC of the day; the saved span is [-12, 36)
print(f"{day}: {len(M)} footprints in the span; asc hours {np.unique(np.floor(hour[asc]))}; dsc hours {np.unique(np.floor(hour[~asc]))}")
# day windows: a 24 h window starting at H0 (UTC) on the day, H0 negative = the day before
for h0 in (-12, -6, 0, 6, 12):
    sel = (hour >= h0) & (hour < h0 + 24)
    for kind in ("cellmean", "idw"):
        report(f"day = [{h0:+d} h, {h0 + 24:+d} h) / {kind}", scheme(sel, kind))
# single passes, to see which pass JRC's value follows where both exist
for wname, sel in (("asc of the day (12-24 h)", asc & (hour >= 12) & (hour < 24)), ("dsc night before (-12-12 h)", ~asc & (hour >= -12) & (hour < 12)),
                   ("dsc night after (12-36 h)", ~asc & (hour >= 12) & (hour < 36))):
    if sel.sum() < 100:
        continue
    report(f"{wname} / idw", scheme(sel, "idw"))
# JRC's T_M follows the daytime (ascending) pass; candidates for how it fills cells the ascending pass missed
day_sel = (hour >= 0) & (hour < 24)
asc_day = asc & (hour >= 12) & (hour < 24)
a_idw = scheme(asc_day, "idw")
d_idw = scheme(~asc & (hour >= -12) & (hour < 12), "idw")
report("asc idw, else dsc night before", np.where(np.isfinite(a_idw), a_idw, d_idw))
report("asc idw, radius 15 km", scheme(asc_day, "idw", radius_km=15))
report("asc nearest, radius 25 km", scheme(asc_day, "nearest", radius_km=25))
# per-cell maximum over every sample of the UTC day (day > night, so a max would also look 'daytime')
rr = ((90 - lat[day_sel]) / RES).astype(int) - R0
cc = ((lon[day_sel] + 180) / RES).astype(int) - C0
ok = (rr >= 0) & (rr < NR) & (cc >= 0) & (cc < NC)
mx = np.full((NR, NC), -np.inf)
np.maximum.at(mx, (rr[ok], cc[ok]), M[day_sel][ok])
report("UTC day / cell max", np.where(np.isfinite(mx), mx, np.nan))
# which pass does JRC follow where both exist?
both = np.isfinite(a_idw) & np.isfinite(d_idw) & np.isfinite(jrc) & land
da, dd = np.abs(a_idw[both] - jrc[both]), np.abs(d_idw[both] - jrc[both])
print(f"cells with both passes: {int(both.sum())}; JRC closer to asc in {np.mean(da < dd):.3f}, to dsc in {np.mean(dd < da):.3f}; "
      f"asc coverage of JRC-valid land {np.isfinite(a_idw)[np.isfinite(jrc) & land].mean():.3f}")

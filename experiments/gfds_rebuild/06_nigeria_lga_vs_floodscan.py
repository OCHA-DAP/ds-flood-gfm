"""Does the rebuilt signal track the Nigeria 2022 floods as well as JRC's? Scored against FloodScan SFED.

Three signals, same treatment, 2022-08-15 .. 2022-11-30 (the book chapter 05 window):
- jrc_all:   JRC GFDS merged signal (AMSR2 + GPM), from the experiments/gfds_nga2022 cache
- jrc_amsr2: JRC GFDS AMSR2-only signal (03_fetch_jrc_reference.py)
- ours:      our L3-grid AMSR2 signal (04_signal_from_l3.py)

Grid: the gfds_nga2022 cache grid (0.09 deg, origin 2.7E 14.0N, 111 x 133), on which the earlier
analysis and the regridded SFED (outputs/gfds_validation/nga_sfed_on_gfds.nc) live. That cache is
offset ~0.44 px from JRC's native rows (its fetch read a fractional window), so jrc_amsr2 and ours
are both sampled onto it by nearest cell centre: identical treatment for the two.

Anomaly, as in the book: a = (avg - s) / max(sd, 0.005) with JRC's static baseline (flood-positive),
then a 4-day trailing mean. LGA means over cells touching each LGA (all_touched, as before).
Scores: per-LGA Pearson r (zero lag) of the 4-day anomaly vs SFED, by SFED seasonal-max band;
reference LGAs; per-pixel r for flooded pixels (SFED max > 0.2).

    uv run python experiments/gfds_rebuild/06_nigeria_lga_vs_floodscan.py
"""

import datetime as dt
from pathlib import Path

import numpy as np
import ocha_stratus as stratus
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.transform import Affine

MAIN = Path("/Users/zackarno/Documents/CHD/repos/ds-flood-gfm")  # gitignored caches live in the main checkout
NGA = MAIN / "data/gdacs_gfds/nga2022"
SFED_NC = MAIN / "outputs/gfds_validation/nga_sfed_on_gfds.nc"
OURS = Path("data/gdacs_gfds/rebuild/signal_l3")
JRC_AMSR2 = Path("data/gdacs_gfds/rebuild/jrc_amsr2_signal")
START, END = dt.date(2022, 8, 15), dt.date(2022, 11, 30)
NY, NX, RES, X0, Y0 = 111, 133, 0.09, 2.7, 14.0
SENTINELS = (-32000, np.iinfo("int32").min)
BANDS = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9)]
REF = ["Ibaji", "Sagbama", "Kogi", "Lokoja", "Bassa", "Ajaokuta", "Ekeremor"]

days = [START + dt.timedelta(k) for k in range((END - START).days + 1)]
lat = Y0 - RES * (np.arange(NY) + 0.5)
lon = X0 + RES * (np.arange(NX) + 0.5)
affine = Affine(RES, 0, X0, 0, -RES, Y0)


def scaled(a):
    v = a.astype("float64") / 1e6
    for s in SENTINELS:
        v[a == s] = np.nan
    v[v <= 0] = np.nan
    return v


# nearest-cell sampling from the two global grids onto the cache grid
r09, c09 = ((90 - lat) / 0.09).astype(int), ((lon + 180) / 0.09).astype(int)
r10, c10 = ((90 - lat) / 0.1).astype(int), ((lon + 180) / 0.1).astype(int)

stacks = {"jrc_all": [], "jrc_amsr2": [], "ours": []}
for d in days:
    p = NGA / f"signal_{d:%Y%m%d}.npy"
    stacks["jrc_all"].append(scaled(np.load(p)) if p.exists() else np.full((NY, NX), np.nan))
    p = JRC_AMSR2 / f"signal_{d:%Y%m%d}_HD5.tif"
    if p.exists():
        with rasterio.open(p) as ds:
            stacks["jrc_amsr2"].append(scaled(ds.read(1))[np.ix_(r09, c09)])
    else:
        stacks["jrc_amsr2"].append(np.full((NY, NX), np.nan))
    p = OURS / f"{d:%Y}" / f"s_{d:%Y%m%d}.tif"
    if not p.exists():
        raise FileNotFoundError(f"our signal missing for {d}: run 02 and 04 for the window first")
    with rasterio.open(p) as ds:
        stacks["ours"].append(ds.read(1).astype("float64")[np.ix_(r10, c10)])
S = {k: np.stack(v) for k, v in stacks.items()}
for k, v in S.items():
    n_days = int(np.isfinite(v).any((1, 2)).sum())
    print(f"{k:10s}: {n_days} of {len(days)} days with data")

avg = scaled(np.load(NGA / "baseline_avg.npy"))
sd = np.maximum(scaled(np.load(NGA / "baseline_sd.npy")), 0.005)
sfed_da = xr.open_dataarray(SFED_NC)
sfed_da = sfed_da.sel(time=slice(str(START), str(END)))
if sfed_da.y.values[0] < sfed_da.y.values[-1]:
    sfed_da = sfed_da.isel(y=slice(None, None, -1))
if sfed_da.shape[1:] != (NY, NX):
    raise ValueError(f"SFED grid {sfed_da.shape[1:]} != cache grid {(NY, NX)}")
sfed_days = [pd.Timestamp(t).date() for t in sfed_da.time.values]
SF = np.full((len(days), NY, NX), np.nan)
for i, d in enumerate(sfed_days):
    SF[days.index(d)] = sfed_da.values[i]


def anom4(s):
    a = (avg - s) / sd
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        w = a[max(0, i - 3) : i + 1]
        n = np.isfinite(w).sum(0)
        out[i] = np.where(n > 0, np.nansum(w, 0) / np.maximum(n, 1), np.nan)
    return out


A = {k: anom4(v) for k, v in S.items()}

# LGA masks (all_touched, as in analyze_nga_lga.py)
adm2 = stratus.codab.load_codab_from_blob("nga", admin_level=2)
masks = [~geometry_mask([g], (NY, NX), affine, all_touched=True) for g in adm2.geometry]


def lga_series(x):
    return np.stack([[np.nanmean(x[t][m]) if np.isfinite(x[t][m]).any() else np.nan for m in masks] for t in range(len(x))])


sf_l = lga_series(SF)
smax = np.nanmax(sf_l, 0)
rows = []
for k in A:
    al = lga_series(A[k])
    for j, (name, pc) in enumerate(zip(adm2["ADM2_EN"], adm2["ADM2_PCODE"])):
        ok = np.isfinite(al[:, j]) & np.isfinite(sf_l[:, j])
        r = np.corrcoef(al[ok, j], sf_l[ok, j])[0, 1] if ok.sum() >= 60 and np.std(sf_l[ok, j]) > 0 else np.nan
        rows.append({"source": k, "lga": name, "pcode": pc, "sfed_max": smax[j], "r": r, "n": int(ok.sum())})
df = pd.DataFrame(rows)

print(f"\nLGA level: median r (4-day anomaly vs SFED, zero lag, >= 60 shared days) by SFED seasonal max")
print(f"{'band':12s} {'n LGAs':>7s} " + " ".join(f"{k:>10s}" for k in A))
for lo, hi in BANDS:
    sub = df[(df.sfed_max >= lo) & (df.sfed_max < hi)]
    meds = [sub[sub.source == k].r.median() for k in A]
    print(f"{lo:.2f}-{'' if hi == 9 else f'{hi:.2f}':5s} {sub[sub.source == 'ours'].r.notna().sum():7d} " + " ".join(f"{m:10.3f}" for m in meds))

print("\nreference LGAs (r):")
for name in REF:
    vals = [df[(df.source == k) & (df.lga == name)].r.iloc[0] if ((df.source == k) & (df.lga == name)).any() else np.nan for k in A]
    print(f"  {name:10s} " + " ".join(f"{k}={v:.3f}" for k, v in zip(A, vals)))

flooded = np.nanmax(SF, 0) > 0.2
print(f"\npixel level, flooded pixels (SFED max > 0.2, n = {int(flooded.sum())}): median per-pixel r(4-day anomaly, SFED)")
for k in A:
    rs = []
    for y, x in zip(*np.where(flooded)):
        a, s = A[k][:, y, x], SF[:, y, x]
        ok = np.isfinite(a) & np.isfinite(s)
        if ok.sum() >= 60 and np.std(s[ok]) > 0:
            rs.append(np.corrcoef(a[ok], s[ok])[0, 1])
    print(f"  {k:10s} median r = {np.median(rs):.3f}  (n px = {len(rs)})")

out = Path("outputs/gfds_rebuild")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "nga2022_lga_r_by_source.csv", index=False)
print(f"\nwrote {out / 'nga2022_lga_r_by_source.csv'}")

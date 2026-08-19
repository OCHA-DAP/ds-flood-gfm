"""Nigeria 2022: GFDS vs FloodScan comparison — stage 1 (sanity + pixel level).

Loads cached .npy stacks, computes GFDS static-baseline anomaly, checks the
Lokoja pixel against the known event timeline, regrids SFED to the GFDS grid,
and maps pixel-level correlation.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

CACHE = Path("data/gdacs_gfds/nga2022")
OUT = Path("outputs/gfds_validation")
OUT.mkdir(parents=True, exist_ok=True)
SENTINELS = (-32000, -2147483648)
LOKOJA = (6.74, 7.80)  # lon, lat


def gfds_coords():
    gt = np.load(CACHE / "geotransform.npy")  # a, b, c, d, e, f (rasterio order)
    a, b, c, d, e, f = gt
    shape = np.load(sorted(CACHE.glob("signal_*.npy"))[0]).shape
    xs = c + (np.arange(shape[1]) + 0.5) * a
    ys = f + (np.arange(shape[0]) + 0.5) * e
    return xs, ys


def load_stack(prefix, scale, signal_like):
    files = sorted(CACHE.glob(f"{prefix}_2022*.npy"))
    dates, arrs = [], []
    for p in files:
        dates.append(pd.Timestamp(p.stem.split("_")[-1]))
        a = np.load(p).astype("float64")
        bad = np.isin(a, SENTINELS)
        if signal_like:
            bad |= a <= 0
        a[bad] = np.nan
        arrs.append(a / scale)
    xs, ys = gfds_coords()
    da = xr.DataArray(np.stack(arrs), dims=("time", "y", "x"),
                      coords={"time": dates, "y": ys, "x": xs})
    missing = len(list(CACHE.glob(f"{prefix}_2022*.missing")))
    print(f"{prefix}: {len(files)} days loaded, {missing} recorded missing")
    return da


def load_sfed():
    files = sorted(CACHE.glob("sfed_2022*.npy"))
    co = np.load(CACHE / "sfed_coords.npz")
    dates = [pd.Timestamp(p.stem.split("_")[-1]) for p in files]
    da = xr.DataArray(np.stack([np.load(p) for p in files]),
                      dims=("time", "y", "x"),
                      coords={"time": dates, "y": co["y"], "x": co["x"]})
    print(f"sfed: {len(files)} days loaded")
    return da


signal = load_stack("signal", 1_000_000, signal_like=True)
mag = load_stack("mag", 1_000, signal_like=False)
avg = np.load(CACHE / "baseline_avg.npy").astype("float64")
sd = np.load(CACHE / "baseline_sd.npy").astype("float64")
for b in (avg, sd):
    b[np.isin(b, SENTINELS) | (b <= 0)] = np.nan
avg /= 1_000_000
sd /= 1_000_000
xs, ys = gfds_coords()
avg = xr.DataArray(avg, dims=("y", "x"), coords={"y": ys, "x": xs})
sd = xr.DataArray(sd, dims=("y", "x"), coords={"y": ys, "x": xs})

anom = ((avg - signal) / sd.clip(min=0.005)).transpose("time", "y", "x")  # flood-positive
sfed = load_sfed()
sfed_on_gfds = sfed.interp(x=signal.x, y=signal.y, method="linear")

# ---- 1. Lokoja sanity check --------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.5))
pt = dict(x=LOKOJA[0], y=LOKOJA[1], method="nearest")
anom.sel(**pt).plot(ax=ax, label="GFDS DIY anomaly (sigma)", color="tab:red")
mag.sel(**pt).plot(ax=ax, label="JRC magnitude (sigma)", color="tab:orange", alpha=0.6)
ax2 = ax.twinx()
sfed_on_gfds.sel(**pt).plot(ax=ax2, label="FloodScan SFED", color="tab:blue")
ax2.set_ylabel("SFED fraction")
for dt, lab in [("2022-09-13", "Lagdo release"), ("2022-10-07", "Kogi peak wk")]:
    ax.axvline(pd.Timestamp(dt), ls="--", color="k", alpha=0.5)
    ax.text(pd.Timestamp(dt), ax.get_ylim()[1], lab, rotation=90, va="top", fontsize=8)
ax.set_title("Lokoja pixel (Niger-Benue confluence), 2022")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
fig.savefig(OUT / "nga_lokoja_timeseries.png", dpi=150, bbox_inches="tight")

# ---- 2. pixel-level correlation map ------------------------------------
sfed_i, anom_i = xr.align(sfed_on_gfds, anom)
sf = sfed_i.values.reshape(sfed_i.sizes["time"], -1)
an = anom_i.values.reshape(anom_i.sizes["time"], -1)
both = np.isfinite(sf) & np.isfinite(an)
n = both.sum(0)
corr = np.full(sf.shape[1], np.nan)
enough = n >= 60
for j in np.where(enough)[0]:
    m = both[:, j]
    if sf[m, j].std() > 1e-6 and an[m, j].std() > 1e-6:
        corr[j] = np.corrcoef(sf[m, j], an[m, j])[0, 1]
corr2d = corr.reshape(anom_i.shape[1:])
cda = xr.DataArray(corr2d, dims=("y", "x"),
                   coords={"y": anom_i.y, "x": anom_i.x})
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
cda.plot(ax=axes[0], vmin=-1, vmax=1, cmap="RdBu_r")
axes[0].set_title("corr(GFDS anomaly, SFED) per pixel, Jun-Dec 2022")
sfed_i.max("time").plot(ax=axes[1], vmin=0, vmax=1, cmap="Blues")
axes[1].set_title("SFED season max (where floods were)")
fig.savefig(OUT / "nga_pixel_corr_map.png", dpi=150, bbox_inches="tight")

# where it matters: flooded pixels only
flooded_px = (sfed_i.max("time") > 0.2).values.ravel()
print("\n--- pixel-level summary ---")
print(f"pixels with corr computed: {int(np.isfinite(corr).sum())}")
print(f"median corr, all pixels:      {np.nanmedian(corr):.2f}")
print(f"median corr, flooded pixels (SFED max>0.2): "
      f"{np.nanmedian(corr[flooded_px]):.2f} (n={int(np.isfinite(corr[flooded_px]).sum())})")

# JRC mag as alternative layer (align on common dates first)
sfed_m, mag_m = xr.align(sfed_on_gfds, mag)
if sfed_m.sizes["time"] >= 60:
    sfm = sfed_m.values.reshape(sfed_m.sizes["time"], -1)
    mg = mag_m.values.reshape(mag_m.sizes["time"], -1)
    bothm = np.isfinite(sfm) & np.isfinite(mg)
    corr_mag = np.full(sfm.shape[1], np.nan)
    for j in np.where(bothm.sum(0) >= 60)[0]:
        m = bothm[:, j]
        if sfm[m, j].std() > 1e-6 and mg[m, j].std() > 1e-6:
            corr_mag[j] = np.corrcoef(sfm[m, j], mg[m, j])[0, 1]
    print(f"median corr using JRC magnitude instead, flooded px: "
          f"{np.nanmedian(corr_mag[flooded_px]):.2f}")
else:
    print(f"JRC magnitude: only {mag_m.sizes['time']} common days cached - "
          "comparison deferred until the mag fetch completes")

# 4-day smoothed DIY anomaly (GFDS is swath-noisy day to day)
anom4 = anom_i.rolling(time=4, min_periods=1).mean()
an4 = anom4.values.reshape(anom4.sizes["time"], -1)
corr4 = np.full(sf.shape[1], np.nan)
both4 = np.isfinite(sf) & np.isfinite(an4)
for j in np.where(both4.sum(0) >= 60)[0]:
    m = both4[:, j]
    if sf[m, j].std() > 1e-6 and an4[m, j].std() > 1e-6:
        corr4[j] = np.corrcoef(sf[m, j], an4[m, j])[0, 1]
print(f"median corr using 4-day-mean DIY anomaly, flooded px: "
      f"{np.nanmedian(corr4[flooded_px]):.2f}")

anom.to_netcdf(OUT / "nga_anom_static.nc")
anom4.to_netcdf(OUT / "nga_anom_static_4day.nc")
sfed_on_gfds.to_netcdf(OUT / "nga_sfed_on_gfds.nc")
mag.to_netcdf(OUT / "nga_mag.nc")
print("\nwrote stacks + 2 PNGs to", OUT)

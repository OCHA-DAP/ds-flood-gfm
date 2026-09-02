"""POC step 1: calibrate GFDS anomaly -> flooded fraction at 10 km.

Per-pixel monotone quantile mapping from the 4-day GFDS anomaly to FloodScan
SFED, fit and evaluated on interleaved day splits (odd/even matched days) so
every score below is out-of-sample. Baseline to beat: predicting each pixel's
training-mean SFED every day.

Outputs (outputs/gfds_downscaling_poc/):
- calibrated fraction stack (netcdf) for the full Nigeria window
- skill table printed + saved
Run: uv run python experiments/gfds_downscaling_poc/01_calibrate_fraction.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CACHE = Path("data/gdacs_gfds/nga2022")
OUT = Path("outputs/gfds_downscaling_poc")
OUT.mkdir(parents=True, exist_ok=True)
SENTINELS = (-32000, -2147483648)


def gfds_coords(shape):
    a, b, c, d, e, f = np.load(CACHE / "geotransform.npy")
    xs = c + (np.arange(shape[1]) + 0.5) * a
    ys = f + (np.arange(shape[0]) + 0.5) * e
    return xs, ys


def load_stack(prefix, scale, positive_only):
    files = sorted(CACHE.glob(f"{prefix}_2022*.npy"))
    arrs, dates = [], []
    for p in files:
        dates.append(pd.Timestamp(p.stem.split("_")[-1]))
        a = np.load(p).astype("float64")
        bad = np.isin(a, SENTINELS)
        if positive_only:
            bad |= a <= 0
        a[bad] = np.nan
        arrs.append(a / scale)
    xs, ys = gfds_coords(arrs[0].shape)
    return xr.DataArray(np.stack(arrs), dims=("time", "y", "x"),
                        coords={"time": dates, "y": ys, "x": xs})


signal = load_stack("signal", 1_000_000, positive_only=True)
avg = np.load(CACHE / "baseline_avg.npy").astype("float64")
sd = np.load(CACHE / "baseline_sd.npy").astype("float64")
for b in (avg, sd):
    b[np.isin(b, SENTINELS) | (b <= 0)] = np.nan
xs, ys = gfds_coords(avg.shape)
avg = xr.DataArray(avg / 1e6, dims=("y", "x"), coords={"y": ys, "x": xs})
sd = xr.DataArray(sd / 1e6, dims=("y", "x"), coords={"y": ys, "x": xs})
anom = ((avg - signal) / sd.clip(min=0.005)).transpose("time", "y", "x")
anom4 = anom.rolling(time=4, min_periods=1).mean()

co = np.load(CACHE / "sfed_coords.npz")
sfiles = sorted(CACHE.glob("sfed_2022*.npy"))
sfed = xr.DataArray(
    np.stack([np.load(p) for p in sfiles]), dims=("time", "y", "x"),
    coords={"time": [pd.Timestamp(p.stem.split("_")[-1]) for p in sfiles],
            "y": co["y"], "x": co["x"]},
).interp(x=signal.x, y=signal.y, method="linear")

sfed_i, anom_i = xr.align(sfed, anom4)
T = sfed_i.sizes["time"]
A = anom_i.values.reshape(T, -1)
W = sfed_i.values.reshape(T, -1)
print(f"matched days: {T} | pixels: {A.shape[1]}")

train = np.arange(T) % 2 == 0  # interleaved split
test = ~train


def quantile_map_fit_predict(a_tr, w_tr, a_te):
    """Empirical CDF match: rank of a in training anomalies -> same rank in
    training SFED. Monotone by construction."""
    ok = np.isfinite(a_tr) & np.isfinite(w_tr)
    if ok.sum() < 20:
        return np.full_like(a_te, np.nan)
    a_s = np.sort(a_tr[ok])
    w_s = np.sort(w_tr[ok])
    # rank of each test anomaly within training anomalies, in [0, 1]
    p = np.searchsorted(a_s, a_te, side="right") / len(a_s)
    p = np.clip(p, 0, 1)
    idx = p * (len(w_s) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    frac = idx - lo
    out = w_s[lo] * (1 - frac) + w_s[hi] * frac
    out[~np.isfinite(a_te)] = np.nan
    return out


pred = np.full_like(W, np.nan)
base = np.full_like(W, np.nan)
for j in range(A.shape[1]):
    pred[test, j] = quantile_map_fit_predict(A[train, j], W[train, j], A[test, j])
    ok_tr = np.isfinite(W[train, j])
    if ok_tr.sum() >= 20:
        base[test, j] = np.nanmean(W[train, j])

# also produce full-period calibrated stack (fit on train days, apply to all)
pred_all = np.full_like(W, np.nan)
for j in range(A.shape[1]):
    pred_all[:, j] = quantile_map_fit_predict(A[train, j], W[train, j], A[:, j])

ok = np.isfinite(pred) & np.isfinite(W)
active = np.nanmax(W, axis=0) > 0.05  # pixels that actually flooded a bit
ok_act = ok & active[None, :]

def rmse(p, w, m):
    return float(np.sqrt(np.nanmean((p[m] - w[m]) ** 2)))

res = {
    "test days": int(test.sum()),
    "RMSE quantile-map (active px)": rmse(pred, W, ok_act),
    "RMSE mean-baseline (active px)": rmse(base, W, ok_act & np.isfinite(base)),
    "bias quantile-map (active px)": float(np.nanmean(pred[ok_act] - W[ok_act])),
}
# basin-scale flooded area time series on test days (the headline number)
px_area_km2 = (0.09 * 111.32) ** 2  # rough, ignores cos(lat); fine at 4-14N for POC
area_gfds = np.nansum(np.where(ok, pred, 0), axis=1) * px_area_km2
area_sfed = np.nansum(np.where(ok, W, 0), axis=1) * px_area_km2
td = test & (area_sfed > 0)
res["area corr (test days)"] = float(np.corrcoef(area_gfds[td], area_sfed[td])[0, 1])
res["area ratio gfds/sfed at SFED peak"] = float(
    area_gfds[np.nanargmax(area_sfed)] / np.nanmax(area_sfed))

for k, v in res.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

frac = xr.DataArray(pred_all.reshape(sfed_i.shape), dims=("time", "y", "x"),
                    coords=sfed_i.coords, name="gfds_fraction")
frac.to_netcdf(OUT / "gfds_calibrated_fraction.nc")
sfed_i.to_netcdf(OUT / "sfed_on_gfds.nc")
pd.Series(res).to_csv(OUT / "calibration_skill.csv")
print("wrote", OUT / "gfds_calibrated_fraction.nc")

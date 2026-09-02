"""POC step 6: three FloodScan-independent routes from GFDS signal to flooded
fraction, scored against SFED as a fair external benchmark.

Routes (none uses FloodScan for calibration):
  R1 physics    f = f_perm + (s_dry - s) / (s_dry * K), K = 1 - eps_w/eps_land
                (literature emissivity contrast at 36 GHz H-pol, K ~ 0.35)
  R2 gsw-anchor per-pixel linear map pinned by two free optical anchors:
                dry signal quantile <-> GSW permanent fraction, and
                wettest signal <-> GSW historical-water envelope fraction
  R3 gfm-anchor like R2 but the wet anchor is Sentinel-1 extent aggregated to
                the cell for the Oct peak window (sparse cross-sensor anchor)
Ceiling:
  R0 sfed-trained per-pixel quantile map from script 01 (trained ON SFED)

Domain: confluence AOI coarse cells (where the GSW prior exists).
Run: uv run python experiments/gfds_downscaling_poc/06_independent_fractions.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

CACHE = Path("data/gdacs_gfds/nga2022")
OUT = Path("outputs/gfds_downscaling_poc")
SENTINELS = (-32000, -2147483648)
K = 0.35  # 1 - eps_water/eps_land at 36 GHz H-pol (literature ~0.30-0.40)


def gfds_coords(shape):
    a, b, c, d, e, f = np.load(CACHE / "geotransform.npy")
    return (c + (np.arange(shape[1]) + 0.5) * a,
            f + (np.arange(shape[0]) + 0.5) * e)


def load_stack(prefix, scale, positive_only):
    files = sorted(CACHE.glob(f"{prefix}_2022*.npy"))
    arrs, dates = [], []
    for fp in files:
        dates.append(pd.Timestamp(fp.stem.split("_")[-1]))
        a = np.load(fp).astype("float64")
        bad = np.isin(a, SENTINELS)
        if positive_only:
            bad |= a <= 0
        a[bad] = np.nan
        arrs.append(a / scale)
    xs, ys = gfds_coords(arrs[0].shape)
    return xr.DataArray(np.stack(arrs), dims=("time", "y", "x"),
                        coords={"time": dates, "y": ys, "x": xs})


signal = load_stack("signal", 1_000_000, positive_only=True)
sig4 = signal.rolling(time=4, min_periods=1).mean()
sfed = xr.open_dataarray(OUT / "sfed_on_gfds.nc")
r0 = xr.open_dataarray(OUT / "gfds_calibrated_fraction.nc")
prior = np.load(OUT / "gsw_prior_aoi.npz")
occ, px, py = prior["occ"], prior["x"], prior["y"]
maps = np.load(OUT / "downscale_maps.npz")
gfm_wet, dom = maps["gfm_wet"], maps["dom"]

# ---- coarse cells inside the prior extent + their GSW/GFM aggregates ------
cells = []
for iy, yc in enumerate(signal.y.values):
    for ix, xc in enumerate(signal.x.values):
        if (px.min() + 0.045 < xc < px.max() - 0.045
                and py.min() + 0.045 < yc < py.max() - 0.045):
            i0 = np.searchsorted(-py, -(yc + 0.045))
            i1 = np.searchsorted(-py, -(yc - 0.045))
            j0 = np.searchsorted(px, xc - 0.045)
            j1 = np.searchsorted(px, xc + 0.045)
            o = occ[i0:i1, j0:j1]
            g = gfm_wet[i0:i1, j0:j1]
            d = dom[i0:i1, j0:j1]
            cells.append({
                "iy": iy, "ix": ix,
                "f_perm": float((o > 80).mean()),
                "f_env": float((o >= 5).mean()),   # historical water envelope
                "f_gfm": float(g[d].mean()) if d.sum() > 100 else np.nan,
            })
C = pd.DataFrame(cells)
iy, ix = C.iy.values, C.ix.values
S = sig4.values[:, iy, ix]          # (time, cell) 4-day signal
W = sfed.values[:, iy, ix]          # benchmark
R0 = r0.values[:, iy, ix]

s_dry = np.nanquantile(S, 0.85, axis=0)
s_wet = np.nanquantile(S, 0.02, axis=0)
peak = (sig4.time >= pd.Timestamp("2022-10-06")) & (sig4.time <= pd.Timestamp("2022-10-13"))
s_peak = np.nanmin(S[peak.values], axis=0)

f_perm, f_env, f_gfm = C.f_perm.values, C.f_env.values, C.f_gfm.values

def clip01(a):
    return np.clip(a, 0.0, 1.0)

R1 = clip01(f_perm[None, :] + (s_dry - S) / (s_dry * K))
den2 = np.where(s_dry - s_wet > 1e-4, s_dry - s_wet, np.nan)
R2 = clip01(f_perm[None, :] + (s_dry - S) / den2 * (f_env - f_perm)[None, :])
den3 = np.where(s_dry - s_peak > 1e-4, s_dry - s_peak, np.nan)
R3 = clip01(f_perm[None, :] + (s_dry - S) / den3 * (f_gfm - f_perm)[None, :])

# ---- score against SFED ----------------------------------------------------
wet_cells = np.nanmax(W, axis=0) > 0.05

def score(P, name):
    ok = np.isfinite(P) & np.isfinite(W)
    okw = ok & wet_cells[None, :]
    r = np.corrcoef(P[okw], W[okw])[0, 1]
    rmse = float(np.sqrt(np.nanmean((P[okw] - W[okw]) ** 2)))
    bias = float(np.nanmean(P[okw] - W[okw]))
    a_p = np.nansum(np.where(ok, P, 0), axis=1)
    a_w = np.nansum(np.where(ok, W, 0), axis=1)
    pk = np.nanargmax(a_w)
    return {"route": name, "r (wet cells)": round(float(r), 3),
            "RMSE": round(rmse, 4), "bias": round(bias, 4),
            "area ratio at SFED peak": round(float(a_p[pk] / a_w[pk]), 2)}

res = pd.DataFrame([
    score(R1, "R1 physics (literature K)"),
    score(R2, "R2 GSW-anchored"),
    score(R3, "R3 GFM-anchored"),
    score(R0, "R0 SFED-trained (ceiling, not independent)"),
]).set_index("route")
print(res.to_string())
res.to_csv(OUT / "independent_fraction_skill.csv")

# ---- area time series figure ----------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.5))
t = sig4.time.values
pxa = (0.09 * 111.32) ** 2
ok_all = np.isfinite(W)
for P, lab, col in [(R1, "R1 physics", "tab:green"),
                    (R2, "R2 GSW-anchored", "tab:red"),
                    (R3, "R3 GFM-anchored", "tab:orange"),
                    (R0, "R0 SFED-trained (ceiling)", "tab:grey")]:
    okp = np.isfinite(P) & ok_all
    ax.plot(t, np.nansum(np.where(okp, P, 0), axis=1) * pxa, label=lab,
            color=col, lw=1.2)
ax.plot(t, np.nansum(np.where(ok_all, W, 0), axis=1) * pxa,
        label="FloodScan SFED (benchmark)", color="tab:blue", lw=2)
ax.set_ylabel("flooded area (km²), AOI")
ax.legend(fontsize=8)
ax.set_title("Confluence AOI: independent fraction routes vs SFED, 2022")
fig.tight_layout()
fig.savefig(OUT / "independent_fraction_areas.png", dpi=140, bbox_inches="tight")
np.savez_compressed(OUT / "independent_fractions.npz",
                    R1=R1, R2=R2, R3=R3, R0=R0, W=W, time=t.astype("datetime64[D]"),
                    iy=iy, ix=ix, wet_cells=wet_cells)
print("wrote independent_fraction_areas.png + independent_fractions.npz")

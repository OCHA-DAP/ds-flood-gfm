"""POC step 4: downscale the calibrated GFDS fraction to ~83 m and validate
against GFM Sentinel-1 extent.

Downscaling = conservative flood-fill: inside each 0.09 deg cell, rank the
~83 m cells by GSW occurrence (desc) then distance-to-historical-water (asc),
and mark the top fraction*n cells wet. Permanent water (occ>80) is excluded
from both prediction and scoring.

Maps compared against GFM (max extent, same Oct window):
- downscaled GFDS (the POC)
- downscaled SFED (upper bound: same fill, licensed input)
- prior-only control (same total area as GFDS, allocated by prior alone,
  ignoring WHERE GFDS put the water at 10 km)
Run: uv run python experiments/gfds_downscaling_poc/04_downscale_validate.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

OUT = Path("outputs/gfds_downscaling_poc")

frac = xr.open_dataarray(OUT / "gfds_calibrated_fraction.nc")
sfed = xr.open_dataarray(OUT / "sfed_on_gfds.nc")
prior = np.load(OUT / "gsw_prior_aoi.npz")
occ, perm = prior["occ"], prior["perm"]
px, py = prior["x"], prior["y"]  # fine coords (x asc, y desc)
gfm = xr.open_dataset(OUT / "gfm_extent_peak.nc")["gfm_flood"]
dates = pd.to_datetime(json.loads((OUT / "gfm_extent_dates.json").read_text()))

# window value = max over the GFM observation dates (same definition per product)
have = [d for d in dates if d in frac.time]
if len(have) < 3:
    raise RuntimeError(f"only {len(have)} GFM dates overlap the GFDS stack: {have}")
w_gfds = frac.sel(time=have).max("time")
w_sfed = sfed.sel(time=have).max("time")
print(f"window: {len(have)} shared dates {have[0].date()}..{have[-1].date()}")

# secondary ranking key: distance to any historical water
dist = distance_transform_edt(occ <= 0)
order_key = np.lexsort(( dist.ravel(), -occ.ravel() ))  # occ desc, then dist asc

# coarse cells fully inside the prior extent
xmin, xmax = px.min(), px.max()
ymin, ymax = py.min(), py.max()
cells = [(iy, ix) for iy, y in enumerate(w_gfds.y.values)
         for ix, x in enumerate(w_gfds.x.values)
         if xmin + 0.045 < x < xmax - 0.045 and ymin + 0.045 < y < ymax - 0.045]

def downscale(wmap):
    out = np.zeros(occ.shape, dtype=bool)
    for iy, ix in cells:
        xc = float(wmap.x[ix]); yc = float(wmap.y[iy])
        w = float(wmap.values[iy, ix])
        if not np.isfinite(w) or w <= 0:
            continue
        i0 = np.searchsorted(-py, -(yc + 0.045))   # py descending
        i1 = np.searchsorted(-py, -(yc - 0.045))
        j0 = np.searchsorted(px, xc - 0.045)
        j1 = np.searchsorted(px, xc + 0.045)
        sub_occ = occ[i0:i1, j0:j1]
        sub_perm = perm[i0:i1, j0:j1]
        sub_dist = dist[i0:i1, j0:j1]
        valid = ~sub_perm
        n_wet = int(round(min(w, 1.0) * valid.sum()))
        if n_wet == 0:
            continue
        flat_order = np.lexsort((sub_dist.ravel(), -sub_occ.ravel()))
        flat_order = flat_order[valid.ravel()[flat_order]]
        sel = flat_order[:n_wet]
        block = np.zeros(sub_occ.size, dtype=bool)
        block[sel] = True
        out[i0:i1, j0:j1] |= block.reshape(sub_occ.shape)
    return out

ds_gfds = downscale(w_gfds)
ds_sfed = downscale(w_sfed)

# prior-only control: same total wet count, allocated AOI-wide by prior rank
n_total = int(ds_gfds.sum())
ctrl = np.zeros(occ.size, dtype=bool)
elig = order_key[~perm.ravel()[order_key]]
ctrl[elig[:n_total]] = True
ctrl = ctrl.reshape(occ.shape)

# GFM (20 m, 0/1/nan) binned onto the 83 m prior grid
gy = gfm.y.values; gx = gfm.x.values
vals = gfm.values.astype("float32")
finite = np.isfinite(vals)
yy, xx = np.meshgrid(gy, gx, indexing="ij")
ye = np.concatenate([py + 0.000375, [py[-1] - 0.000375]])[::-1]  # asc edges
xe = np.concatenate([px - 0.000375, [px[-1] + 0.000375]])
wet_sum, _, _ = np.histogram2d(yy[finite], xx[finite], bins=(ye, xe),
                               weights=(vals[finite] > 0).astype("float32"))
cnt, _, _ = np.histogram2d(yy[finite], xx[finite], bins=(ye, xe))
gfm_frac = np.full(occ.shape, np.nan, dtype="float32")
with np.errstate(invalid="ignore"):
    gfm_frac[:] = (wet_sum / np.where(cnt > 0, cnt, np.nan))[::-1]  # back to y desc
gfm_wet = gfm_frac >= 0.5
gfm_valid = np.isfinite(gfm_frac) & (cnt[::-1] >= 8)  # >=8 of ~17 20m px seen

# scoring domain: valid GFM, non-permanent, inside scored coarse cells
scored = np.zeros(occ.shape, dtype=bool)
for iy, ix in cells:
    if np.isfinite(w_gfds.values[iy, ix]):
        yc = float(w_gfds.y[iy]); xc = float(w_gfds.x[ix])
        i0 = np.searchsorted(-py, -(yc + 0.045)); i1 = np.searchsorted(-py, -(yc - 0.045))
        j0 = np.searchsorted(px, xc - 0.045); j1 = np.searchsorted(px, xc + 0.045)
        scored[i0:i1, j0:j1] = True
dom = gfm_valid & ~perm & scored
print(f"scoring domain: {dom.sum():,} cells | GFM wet in domain: {int((gfm_wet & dom).sum()):,}")

def skill(pred):
    h = int((pred & gfm_wet & dom).sum())
    f = int((pred & ~gfm_wet & dom).sum())
    m = int((~pred & gfm_wet & dom).sum())
    pod = h / (h + m) if h + m else np.nan
    far = f / (h + f) if h + f else np.nan
    csi = h / (h + m + f) if h + m + f else np.nan
    return {"POD": round(pod, 3), "FAR": round(far, 3), "CSI": round(csi, 3),
            "wet_km2": round(int((pred & dom).sum()) * 0.0835 ** 2, 0)}

res = pd.DataFrame({
    "downscaled GFDS (POC)": skill(ds_gfds),
    "downscaled SFED (upper bound)": skill(ds_sfed),
    "prior-only control": skill(ctrl),
}).T
print(res.to_string())
res.to_csv(OUT / "downscale_skill.csv")

# compact artifact for the book chapter: the four maps + domain + coords
np.savez_compressed(
    OUT / "downscale_maps.npz",
    gfm_wet=gfm_wet, ds_gfds=ds_gfds, ds_sfed=ds_sfed, ctrl=ctrl,
    dom=dom, x=px, y=py,
    window=np.array([str(d.date()) for d in have]),
)

fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
ext = [px.min(), px.max(), py.min(), py.max()]
show = lambda ax, m, t, cmap: (ax.imshow(np.where(dom, m, np.nan), extent=ext,
                               cmap=cmap, vmin=0, vmax=1, interpolation="none"),
                               ax.set_title(t, fontsize=11))
show(axes[0, 0], gfm_wet.astype(float), "GFM Sentinel-1 max extent (truth here)", "Blues")
show(axes[0, 1], ds_gfds.astype(float), "downscaled GFDS (free, this POC)", "Reds")
show(axes[1, 0], ds_sfed.astype(float), "downscaled SFED (licensed input)", "Purples")
show(axes[1, 1], ctrl.astype(float), "prior-only control (no 10 km info)", "Greys")
for ax in axes.ravel():
    ax.set_aspect("equal")
fig.suptitle(f"Niger-Benue confluence, {have[0].date()} to {have[-1].date()}, ~83 m",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "downscale_maps.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "downscale_maps.png")

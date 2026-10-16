"""Step 0: FloodScan's core flooded-fraction retrieval on AMSR2 L3 36 GHz (V and H), Nigeria 2022.

Follows AER's Algorithm Description Document, AFED V05R00 (docs/floodscan-algorithm-notes.md):
  footprint:  V and H smoothed with the 22D weighting W(x) = exp(-a (x/D)^2.4), D = 22 km, W(D/2) = 0.5  (eq. 3)
  signal:     PRI = (V - H)/(V + H);  Q = sqrt(PRI + 0.11) - 0.58 PRI                         (eqs. 9-10)
  cross-cal:  AMSR2 -> AMSR-E, 37 GHz: Q' = A Q + B  (asc 1.0466/-0.0154, desc 1.0438/-0.0141)  (Table 19)
  mixing:     ff = (Q' - Q_pow f_pow - Q_dry (1 - f_pow)) / (Q_flood - Q_dry),  Q_flood = Q_pow  (eq. 13)
              Q_pow, 37 GHz: asc 0.457, desc 0.455                                            (Table 22)
  daily:      weighted mean of passes, today 0.67 and yesterday 0.33 (f_hop = 0); a lone value only
              if it is today's; clipped to [0, 1]                                              (s. 3.8.3)

PLACEHOLDERS (step 0, not AER's method), each a stated departure:
- Q_dry: per-pixel, per-pass 10th percentile of Q' over the window (AER: climatology blended with
  an expected 61-day median, needing years of history). A per-pixel constant Q_dry rescales and
  shifts each pixel's ff linearly, so temporal correlation is unaffected; magnitudes are not.
- f_pow = 0 (no persistent-water term); no atmospheric correction; no rain/frozen/flash flags;
  no MDFF; L3 day-mean grids instead of footprint-matched swaths.

Output: data/floodscan_repro/nga2022_core_l3.npz (daily ff on the 0.1-deg grid, region with margin).

    uv run --no-project --with h5py --with numpy python experiments/floodscan_repro/01_core_l3.py
"""

import datetime as dt
from pathlib import Path

import h5py
import numpy as np

SRC = Path("data/gdacs_gfds/rebuild/amsr2_l3_tb36")
OUT = Path("data/floodscan_repro/nga2022_core_l3.npz")
START, END = dt.date(2022, 8, 15), dt.date(2022, 11, 30)
LON0, LON1, LAT0, LAT1 = 2.0, 15.5, 3.0, 15.0  # comparison box 2.67-14.75E, 3.9-14N plus a 0.3+ deg margin
FILL = 65534
D_KM, R_CELLS = 22.0, 3
A_22D = np.log(2) / 0.5**2.4
XCAL = {"A": (1.0466, -0.0154), "D": (1.0438, -0.0141)}
QPOW = {"A": 0.457, "D": 0.455}
DRY_PCT = 10

# rolled L3 grid (-180..180): row r centred at 89.95 - 0.1 r, column c at -179.95 + 0.1 c
r0, r1 = int(round((90 - LAT1) / 0.1)), int(round((90 - LAT0) / 0.1))
c0, c1 = int(round((LON0 + 180) / 0.1)), int(round((LON1 + 180) / 0.1))
lat = 89.95 - 0.1 * np.arange(r0, r1)
lon = -179.95 + 0.1 * np.arange(c0, c1)


def read_vh(path):
    out = []
    with h5py.File(path) as h:
        for pol in ("V", "H"):
            ds = h[f"Brightness Temperature ({pol})"]
            raw = np.roll(ds[:], 1800, axis=1)[r0:r1, c0:c1]
            tb = raw.astype("float64") * float(ds.attrs["SCALE FACTOR"][0])
            tb[raw >= FILL] = np.nan
            out.append(tb)
    return out


def smooth_22d(tb):
    """Normalised convolution with the 22D weighting; NaN where < half the kernel weight is observed."""
    ny, nx = tb.shape
    num = np.zeros_like(tb)
    den = np.zeros_like(tb)
    tot = np.zeros(ny)
    ok = np.isfinite(tb)
    t0 = np.where(ok, tb, 0.0)
    coslat = np.cos(np.deg2rad(lat))
    for di in range(-R_CELLS, R_CELLS + 1):
        for dj in range(-R_CELLS, R_CELLS + 1):
            x = np.sqrt((di * 11.12) ** 2 + (dj * 11.12 * coslat) ** 2)  # km, per row
            w = np.exp(-A_22D * (x / D_KM) ** 2.4)[:, None]
            sh_t = np.roll(np.roll(t0, di, 0), dj, 1)
            sh_ok = np.roll(np.roll(ok, di, 0), dj, 1)
            num += w * sh_t
            den += w * sh_ok
            tot += w[:, 0]
    out = num / np.where(den > 0, den, np.nan)
    out[den < 0.5 * tot[:, None]] = np.nan
    out[:R_CELLS], out[-R_CELLS:], out[:, :R_CELLS], out[:, -R_CELLS:] = np.nan, np.nan, np.nan, np.nan
    return out


days = [START + dt.timedelta(k) for k in range((END - START).days + 1)]
Q = {p: np.full((len(days), r1 - r0, c1 - c0), np.nan) for p in "AD"}
for i, d in enumerate(days):
    for p in "AD":
        files = list((SRC / f"{d:%Y}" / f"{d:%m}").glob(f"GW1AM2_{d:%Y%m%d}_01D_EQM{p}_*.h5"))
        if len(files) != 1:
            raise FileNotFoundError(f"{d} pass {p}: expected 1 L3 file, found {len(files)}")
        v, h = (smooth_22d(x) for x in read_vh(files[0]))
        pri = (v - h) / (v + h)
        with np.errstate(invalid="ignore"):
            q = np.sqrt(pri + 0.11) - 0.58 * pri
        a, b = XCAL[p]
        Q[p][i] = a * q + b

FF = {}
for p in "AD":
    qdry = np.nanpercentile(Q[p], DRY_PCT, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        FF[p] = (Q[p] - qdry) / (QPOW[p] - qdry)

# daily: today's passes weight 0.67, yesterday's 0.33; a lone value counts only if it is today's
ff = np.full(Q["A"].shape, np.nan)
for i in range(len(days)):
    vals, wts, today_n = [], [], np.zeros(ff.shape[1:])
    for k, w in ((i, 0.67), (i - 1, 0.33)):
        if k < 0:
            continue
        for p in "AD":
            x = FF[p][k]
            vals.append(np.where(np.isfinite(x), x, 0.0) * w)
            wts.append(np.isfinite(x) * w)
            if k == i:
                today_n += np.isfinite(x)
    W = np.sum(wts, 0)
    n_all = np.sum([wt > 0 for wt in wts], 0)
    m = np.where(W > 0, np.sum(vals, 0) / np.where(W > 0, W, 1), np.nan)
    m[(n_all == 1) & (today_n == 0)] = np.nan
    ff[i] = np.clip(m, 0, 1)

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, ff=ff.astype("float32"), lat=lat, lon=lon,
                    days=np.array([d.isoformat() for d in days]))
valid = np.isfinite(ff).mean()
print(f"wrote {OUT}: ff {ff.shape}, valid share {valid:.3f}, "
      f"median ff {np.nanmedian(ff):.4f}, 99th pct {np.nanpercentile(ff, 99):.3f}")

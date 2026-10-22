"""Step 1a: AER's static tables (ADD s. 4.11-4.20) for the West Africa box, from CSU AMSR2 grids.

Training years: 2023 and 2025, June-December (2024 is the test year and is left out).
Inputs: data/floodscan_repro/csu_amsr2_grid/YYYY/YYYYMMDD.npz (03, bands 36,18,23,89),
        data/floodscan_repro/static_0p25.tif (00).
Output: data/floodscan_repro/static_tables.npz.

Per frequency (36, 18 GHz) and pass (asc, dsc), on the 0.25-deg grid, north-up:
  SI0         s. 3.7: rain-flag scattering threshold climatology (rainflags.py), from the same years.
              Rain-flagged passes carry no Q into any table. Frozen flags do not apply at 0-20 N; no flash flags.
  Q_pow       s. 4.11 DEPARTURE: median Q over open sea (the box has no inland water the radiometer sees
              as water; see the comment at the fit). Re-derived because the CSU calibration differs.
  cells with f_pow > 0.5 (essentially Lake Chad) are excluded throughout
  Q61c, Q31c  s. 4.12: mean over years of the time-centred 61-/31-day median of land Q,
              Q_land = (Q - f_pow Q_pow) / (1 - f_pow)
  var61       s. 4.13: year-to-year variance of Q61, spatial median ("smooth2") everywhere
              (AER: smooth2 only in floodplains and on coasts; with two years the point-wise value is unusable)
  b31         s. 4.15: Q31c(d) - Q31c(d-15), spatial median
  b61         s. 4.14 with a61 = 1: mean over years of (centred Q61 - retrospective Q61), spatial median
              (AER fits a61 per day by regression; two years cannot support it)
  QdLUT       s. 4.16: 75th percentile of land Q in NDVI x f_tree bins of 0.05 (20% subsample)
  wetland     OUR HEURISTIC, not AER's (its seven rule groups are not published):
              Q61c - QdLUT > WET_THR on that day
  Qdc         s. 4.18 eq. 25 with Qdci = QdLUT (no local-extremes step); wetland rules as in (a) and (d)
  MDFF        s. 4.20: sigma_Q from the 7-day variance climatology (spatial median away from the near-coast
              cells, 31-day average), ff(sigma) = sigma / (Q_pow - Qdc), MDFF0 = 3 ff, capped 0.25 where
              f_pow < 0.49, MDFF = (MDFF0 + max over available days) / 2
  near_coast  s. 4.20 rule: f_pow > 0.01 and >= 3 cells with f_pow > 0.75 in the MDFF window (saved for 05)
  fhop        s. 4.19 (one table, 36 GHz): in-sample initial point-wise FF per pass minus its 15-day running
              mean, negatives to 0, time derivative; Pearson kurtosis k of the derivatives per day-of-year over
              a 61-day window across years and passes; fhop = clip((log10 k - 0.75) / 0.75, 0, 1);
              near-coast cells: fhop = min(5 f_pow, 1)
Spatial windows are AER's rescaled from 5-arcminute to 0.25-deg cells (W_VAR, W_B, W_MDFF below).

    uv run --no-project --with numpy --with scipy --with rasterio python experiments/floodscan_repro/04_static_tables.py
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import median_filter, uniform_filter
from scipy.stats import kurtosis

from aer_chain import FREQS, PASSES, clim_doy, initial_ff, load_year, rain_flags, running_mean, running_median
from rainflags import scattering_index, si0_from_training

SRC = Path("data/floodscan_repro/csu_amsr2_grid")
STATIC = Path("data/floodscan_repro/static_0p25.tif")
OUT = Path("data/floodscan_repro/static_tables.npz")
YEARS = (2023, 2025)
ATMOS = None  # a MERRA-2 directory (09) switches on the earth-leaving correction (atmos.py) before Q
if len(sys.argv) > 1:  # `... 04_static_tables.py 2023,2025 /path/out.npz [atmos]`; years 'default' keeps the defaults
    if sys.argv[1] != "default":
        YEARS = tuple(int(y) for y in sys.argv[1].split(","))
    OUT = Path(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] == "atmos":
        ATMOS = Path("data/floodscan_repro/merra2")
WET_THR = 0.02
Q0 = 0.32  # eq. 25 constant, AMSR-E scale (kept as published; our calibration differs)
# AER's spatial windows are counted in 5-arcminute cells; one 0.25-deg cell is 3 of them. Until 2026-10-21
# the same cell counts were applied here, i.e. windows three times wider on the ground than AER's.
# Rescaled to the nearest odd width: 9x9 -> 3x3 (var61), 7x7 -> 3x3 (b31, b61), 11x11 -> 5x5 (MDFF median
# and the near-coast test); AER's 3x3 mean before the MDFF median is one cell here and is dropped.
W_VAR, W_B, W_MDFF = 3, 3, 5

with rasterio.open(STATIC) as ds:
    names = list(ds.descriptions)
    st = {n: ds.read(i + 1).astype("float64") for i, n in enumerate(names)}
fpow, ftree, focean = st["fpow"], st["ftree"], st["focean"]
ndvi_slots = np.stack([st[f"ndvi_s{k:02d}"] for k in range(1, 24)])
slot_doy = 16 * np.arange(23) + 8.5
inland = focean < 0.01
# AER s. 4.20 near-coastline test: f_pow > 0.01 and at least three cells with f_pow > 0.75 in the MDFF window
# (f_pow here includes sea, as in AER's land-water mask). No spatial filtering of the MDFF variance there.
near_coast = (fpow > 0.01) & (uniform_filter((fpow > 0.75).astype(float), W_MDFF, mode="constant") * W_MDFF ** 2 >= 3)
DOYS = np.arange(1, 366)
ndvi_doy = np.stack([np.array([np.interp(d, slot_doy, ndvi_slots[:, i, j], period=365)
                               for d in DOYS]).reshape(-1) for i, j in np.ndindex(80, 80)], 1).reshape(365, 80, 80)

data = {y: load_year(SRC / str(y), dt.date(y, 6, 1), dt.date(y, 12, 31), atmos_dir=ATMOS) for y in YEARS}
print(f"training years {YEARS}; atmospheric correction {'on' if ATMOS else 'off'}; output {OUT}")
tables = {"fpow": fpow, "ftree": ftree, "focean": focean, "near_coast": near_coast, "ndvi_doy": ndvi_doy, "years": np.array(YEARS)}


def on_doy_of(days):
    doy = np.array([clim_doy(d) for d in days])
    return lambda a: np.stack([a[np.where(doy == dc)[0][0]] if dc in doy else np.full((80, 80), np.nan) for dc in DOYS])


# --- rain flags (s. 3.7; rainflags.py). SI0 is built from these same training years; the 61-day SI median
# is clipped at the record's ends so June and December get a threshold too (31+ days in the window).
si = {y: {p: scattering_index(data[y][2][("18v", p)], data[y][2][("23v", p)], data[y][2][("89va", p)]) for p in PASSES} for y in YEARS}
si0 = si0_from_training({p: np.stack([on_doy_of(data[y][0])(running_median(si[y][p], -30, 30, clip=True)) for y in YEARS]) for p in PASSES})
for p in PASSES:
    tables[f"SI0_{p}"] = si0[p]
for y in YEARS:
    days, q, tb = data[y]
    doy = np.array([clim_doy(d) for d in days])
    flags = rain_flags(tb, si0, doy)
    for p in PASSES:
        for f in FREQS:
            q[(f, p)][flags[p]] = np.nan
        valid = np.isfinite(tb[("18v", p)]) & inland[None]
        in_season = (doy >= 213) & (doy <= 334)
        print(f"{y} {p}: rain-flagged {flags[p][valid].mean():.3f} of valid inland cell-days "
              f"(Aug-Nov {flags[p][in_season][:, inland].mean():.3f})")

for f in FREQS:
    for p in PASSES:
        key = f"{f}_{p}"
        # --- Q_pow (s. 4.11). AER fit Q against f_pow at Lake Victoria. This box has no inland open
        # water visible to the radiometer: MOD44W 'persistent water' cells (Lake Chad, reservoirs) show
        # dry-land Q (dry run 2026-10-21: f_pow 0.2-0.45 cells median Q 0.339 vs 0.343 at f_pow 0;
        # Lake Chad f_pow 0.9 Q 0.36-0.38), so the inland fit extrapolates to a nonsense value below dry
        # land. DEPARTURE: Q_pow = median Q over open sea (f_ocean > 0.99, Gulf of Guinea). At 37 GHz
        # salinity changes water emissivity little (it matters mainly below ~10 GHz). No atmospheric
        # correction here, so the value sits below AER's earth-leaving 0.457, consistently with our Q.
        med = np.nanmedian(np.concatenate([data[y][1][(f, p)] for y in YEARS]), 0)
        sea = focean > 0.99
        qpow = float(np.nanmedian(med[sea]))
        sel = inland & np.isfinite(med) & (fpow < 0.4)
        b1, b0 = np.polyfit(fpow[sel], med[sel], 1)
        tables[f"Qpow_{key}"] = qpow
        print(f"{key}: Q_pow (open sea, n {int(sea.sum())}) = {qpow:.4f}; inland dry-land intercept {b0:.4f} "
              f"(inland fit slope {b1:+.3f}, diagnostic only)")

        # --- land Q, centred and retrospective medians per year (s. 4.12)
        q61c_y, q31c_y, q61r_y, var7_y = [], [], [], []
        for y in YEARS:
            days, q, _ = data[y]
            with np.errstate(invalid="ignore", divide="ignore"):
                ql = (q[(f, p)] - fpow * qpow) / (1 - fpow)
            ql[:, fpow > 0.5] = np.nan  # mostly-water cells (Lake Chad): the land correction is meaningless there
            on_doy = on_doy_of(days)
            q61c_y.append(on_doy(running_median(ql, -30, 30)))
            q31c_y.append(on_doy(running_median(ql, -15, 15)))
            q61r_y.append(on_doy(running_median(ql, -60, 0)))
            var7 = np.full_like(ql, np.nan)
            for i in range(3, len(ql) - 3):
                w = ql[i - 3 : i + 4]
                with np.errstate(all="ignore"):
                    var7[i] = np.where(np.isfinite(w).sum(0) >= 4, np.nanvar(w, 0), np.nan)
            var7_y.append(on_doy(var7))
        with np.errstate(all="ignore"):
            q61c = np.nanmean(q61c_y, 0)
            q31c = np.nanmean(q31c_y, 0)
            var_pw = np.nanvar(q61c_y, 0, ddof=1)
            b61 = np.nanmean(np.array(q61c_y) - np.array(q61r_y), 0)
        var61 = np.stack([median_filter(np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(v).any() else 0), W_VAR) for v in var_pw])
        var61 = np.maximum(var61, 1e-6)
        b31 = np.full_like(q31c, np.nan)
        b31[15:] = q31c[15:] - q31c[:-15]
        b31 = np.stack([median_filter(np.nan_to_num(v), W_B) if np.isfinite(v).any() else v for v in b31])
        b61 = np.stack([median_filter(np.nan_to_num(v), W_B) if np.isfinite(v).any() else v for v in b61])

        # --- QdLUT(NDVI, f_tree): 75th percentile of land Q in 0.05 bins, 20% subsample (s. 4.16)
        rng = np.random.default_rng(0)
        qs, ns, ts = [], [], []
        for y in YEARS:
            days, q, _ = data[y]
            with np.errstate(invalid="ignore", divide="ignore"):
                ql = (q[(f, p)] - fpow * qpow) / (1 - fpow)
            for i, d in enumerate(days):
                m = np.isfinite(ql[i]) & inland & (fpow < 0.2) & (rng.random((80, 80)) < 0.2)
                qs.append(ql[i][m]); ns.append(ndvi_doy[clim_doy(d) - 1][m]); ts.append(ftree[m])
        qs, ns, ts = map(np.concatenate, (qs, ns, ts))
        nb = np.clip((ns / 0.05).astype(int), 0, 19); tb_ = np.clip((ts / 0.05).astype(int), 0, 19)
        lut = np.full((20, 20), np.nan)
        for a in range(20):
            for b in range(20):
                m = (nb == a) & (tb_ == b)
                if m.sum() >= 50:
                    lut[a, b] = np.percentile(qs[m], 75)
        # fill empty bins from the nearest filled bin in NDVI, then in f_tree
        for b in range(20):
            col = lut[:, b]
            if np.isfinite(col).any():
                lut[:, b] = np.interp(np.arange(20), np.where(np.isfinite(col))[0], col[np.isfinite(col)])
        for a in range(20):
            row = lut[a]
            if np.isfinite(row).any():
                lut[a] = np.interp(np.arange(20), np.where(np.isfinite(row))[0], row[np.isfinite(row)])
        qdlut = lut[np.clip((ndvi_doy / 0.05).astype(int), 0, 19), np.clip((ftree / 0.05).astype(int), 0, 19)[None]]

        # --- wetland heuristic and Qdc (s. 4.17-4.18)
        wet = (q61c - qdlut) > WET_THR
        qbar = np.nanmean(q61c, 0)
        w_dc = np.where(q61c <= qdlut, 1 / (25 + 200 * np.maximum(qbar - Q0, 0)), 1 / 25)
        qdc = (w_dc * qdlut + q61c) / (w_dc + 1)
        qdc = np.where(wet & (q61c <= qdlut), q61c, qdc)
        qdc = np.where(wet & (q61c > qdlut), np.minimum(q61c, qdlut), qdc)

        # --- MDFF (s. 4.20)
        with np.errstate(all="ignore"):
            v7 = np.nanmean(var7_y, 0)
        v7f = np.full_like(v7, np.nan)
        for k in range(365):
            v = v7[k]
            if not np.isfinite(v).any():
                continue
            fill = np.nan_to_num(v, nan=np.nanmedian(v))
            sm = median_filter(fill, W_MDFF)
            v7f[k] = np.where(near_coast, fill, sm)
        sig = np.sqrt(np.stack([np.nanmean(v7f[max(0, k - 15) : k + 16], 0) if np.isfinite(v7f[max(0, k - 15) : k + 16]).any() else np.full((80, 80), np.nan) for k in range(365)]))
        with np.errstate(all="ignore"):
            mdff0 = 3 * sig / (qpow - qdc)
        mdff0 = np.where(fpow < 0.49, np.minimum(mdff0, 0.25), mdff0)
        mdff = (mdff0 + np.nanmax(mdff0, 0)) / 2

        for name, arr in (("Q61c", q61c), ("Q31c", q31c), ("var61", var61), ("b31", b31), ("b61", b61),
                          ("QdLUT", qdlut), ("wet", wet), ("Qdc", qdc), ("MDFF", mdff)):
            tables[f"{name}_{key}"] = arr
        ok = np.isfinite(mdff[200])
        print(f"   DOY 200 (Jul 19): median Qdc {np.nanmedian(qdc[200]):.4f}, wet share {wet[200][inland].mean():.3f}, "
              f"median MDFF {np.nanmedian(mdff[200][ok]):.3f}, sqrt(var61) median {np.sqrt(np.nanmedian(var61[200])):.4f}")

# --- f_hop (s. 4.19): kurtosis of the derivatives of the in-sample initial point-wise FF, 36 GHz, both passes
series = []  # (day-of-year per sample, FF derivative per sample) for each year and pass
for y in YEARS:
    days, q, _ = data[y]
    doy = np.array([clim_doy(d) for d in days])
    ff = initial_ff(q, tables, doy, fpow, freqs=("36",))["pw"]
    for p in PASSES:
        x = ff[("36", p)]
        x = np.maximum(x - running_mean(x, 7), 0)  # 15-day running mean removed, negatives to zero
        # derivative between successive valid observations of each cell (a pass sees a cell every ~2 days
        # at these latitudes, so consecutive-day differences would discard most of the record)
        d = np.full_like(x, np.nan)
        last, t_last = np.full((80, 80), np.nan), np.full((80, 80), np.nan)
        for i in range(len(days)):
            ok = np.isfinite(x[i])
            with np.errstate(invalid="ignore"):
                d[i] = np.where(ok & np.isfinite(last), (x[i] - last) / (i - t_last), np.nan)
            last, t_last = np.where(ok, x[i], last), np.where(ok, i, t_last)
        series.append((doy, d))
fhop = np.full((365, 80, 80), np.nan)
FHOP_MIN_N = 8  # fewer derivative samples than this in a cell's 61-day window: fhop 0 (AER's typical value)
fhop_defaulted = 0  # worst day-of-year count of such land cells, printed below
for dc in DOYS:
    parts = [d[(doy >= dc - 30) & (doy <= dc + 30)] for doy, d in series]
    sample = np.concatenate(parts)
    if np.median(np.isfinite(sample).sum(0)[fpow <= 0.5]) < 20:
        continue  # day-of-year outside the training season: no table (05 raises if it needs it)
    with np.errstate(all="ignore"):
        k = kurtosis(sample, axis=0, fisher=False, nan_policy="omit")
        k = np.where(np.nanvar(sample, 0) > 0, k, 1.0)  # a flat series has no outliers: fhop 0
        k = np.where(np.isfinite(sample).sum(0) >= FHOP_MIN_N, k, 1.0)  # too few samples: fhop 0 (counted below)
        fhop[dc - 1] = np.clip((np.log10(k) - 0.75) / 0.75, 0, 1)
    few = (np.isfinite(sample).sum(0) < FHOP_MIN_N) & (fpow <= 0.5) & ~near_coast
    fhop_defaulted = max(fhop_defaulted, int(few.sum()))
fhop = np.where(near_coast[None], np.minimum(5 * fpow, 1)[None], fhop)
tables["fhop"] = fhop
in_season = slice(212, 334)
print(f"fhop Aug-Nov: inland median {np.nanmedian(fhop[in_season][:, inland]):.3f}, share > 0: "
      f"{np.nanmean(fhop[in_season][:, inland] > 0):.3f}, share = 1: {np.nanmean(fhop[in_season][:, inland] >= 1):.3f}; "
      f"land cells defaulted to 0 for lack of samples (worst day): {fhop_defaulted}")

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, **tables)
print(f"wrote {OUT}")

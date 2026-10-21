"""Step 1b: AER's flooded-fraction chain (ADD s. 3.8) for 2024, on the CSU AMSR2 0.25-deg grid.

Uses the static tables from 04 (built from 2023 and 2025; 2024 is the test year).
Per pass (dsc then asc each day, AMSR2 ~01:30 and ~13:30 local) and frequency (36, 18 GHz):
  Q_land            = (Q - f_pow Q_pow) / (1 - f_pow)
  Q31R, Q61R        retrospective medians of Q_land (d-30..d, d-60..d)
  Q31E, Q61E        Q31R + b31(dc), Q61R + b61(dc)                      (eqs. 14-15, a61 = 1)
  Q61Epw            (Q31E + Q61E) / 2                                    (eq. 16)
  Q_dry             (Qdc / var61 + Q61Epw / 0.0012) / (1 / var61 + 1 / 0.0012); wetland: Q_dry = Qdc   (eq. 17)
  ff                (Q - Q_pow f_pow - Q_dry (1 - f_pow)) / (Q_pow - Q_dry)                          (eq. 13)
Flash false positives (s. 3.8.2), simplified:
  newly flooded = ff18 > 0.05 where the previous pass's ff36 < 0.05; candidates where > 75% of a
  4-cell-diameter circle (~93 km, AER's 10 px at 5') is newly flooded; drop wetland cells, cells with
  ff36 below MDFF, and isolated cells; dilate over connected cells with ff36 > 0.025; drop those values.
  (No t_flash/f_hop screen: f_hop is not built, so it is 0.)
Daily (s. 3.8.3): today's passes weight 0.67, yesterday's 0.33 (f_hop = 0); a lone value only if it is
today's; clip to [0, 1 - f_pow]; ff < MDFF(dc) -> 0 (mean of asc/dsc MDFF). No fill-in, no outlier
test, no spatial (11x11) variant, no rain/frozen flags (stage-1 data has no 23/89 GHz).

Variants written alongside, for before/after: 'placeholder' (step-0 dry reference: per-pixel 10th
percentile of Q in Jun-Dec 2024, no MDFF, no flash) and 'noflash' (full chain without flash detection).

    uv run --no-project --with numpy --with scipy python experiments/floodscan_repro/05_ff_chain.py
"""

import datetime as dt
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, convolve, label

SRC = Path("data/floodscan_repro/csu_amsr2_grid/2024")
TAB = Path("data/floodscan_repro/static_tables.npz")
OUT = Path("data/floodscan_repro/ff_2024.npz")
D0, D1 = dt.date(2024, 6, 1), dt.date(2024, 12, 31)
T0, T1 = dt.date(2024, 8, 1), dt.date(2024, 11, 30)
W_EXP = 1 / 0.0012

t = np.load(TAB)
fpow = t["fpow"]


def clim_doy(d):
    return dt.date(2023, d.month, d.day).timetuple().tm_yday


def running_median(x, lo, hi, min_frac=0.5):
    out = np.full_like(x, np.nan)
    n = hi - lo + 1
    for i in range(len(x)):
        a, b = i + lo, i + hi
        if a < 0 or b >= len(x):
            continue
        w = x[a : b + 1]
        ok = np.isfinite(w).sum(0) >= min_frac * n
        with np.errstate(all="ignore"):
            out[i] = np.where(ok, np.nanmedian(w, 0), np.nan)
    return out


days = [D0 + dt.timedelta(k) for k in range((D1 - D0).days + 1)]
miss_file = SRC.parent / "missing.txt"  # days 03 found absent upstream: no data, not an error
ABSENT = {line.split()[0] for line in miss_file.read_text().splitlines()} if miss_file.exists() else set()
Q = {}
for f in ("36", "18"):
    for p in ("asc", "dsc"):
        arr = np.full((len(days), 80, 80), np.nan)
        for i, d in enumerate(days):
            path = SRC / f"{d:%Y%m%d}.npz"
            if not path.exists():
                if d.isoformat() in ABSENT:
                    continue
                raise FileNotFoundError(f"{path} missing and not recorded absent: run 03 for 2024 first")
            z = np.load(path)
            v, h = np.flipud(z[f"fcdr_tb{f}v_{p}"]), np.flipud(z[f"fcdr_tb{f}h_{p}"])
            pri = (v - h) / (v + h)
            with np.errstate(invalid="ignore"):
                arr[i] = np.sqrt(pri + 0.11) - 0.58 * pri
        Q[(f, p)] = arr

dcs = np.array([clim_doy(d) for d in days])
FF, FFPH = {}, {}
for (f, p), q in Q.items():
    key = f"{f}_{p}"
    qpow = float(t[f"Qpow_{key}"])
    with np.errstate(invalid="ignore", divide="ignore"):
        ql = (q - fpow * qpow) / (1 - fpow)
    q31r, q61r = running_median(ql, -30, 0), running_median(ql, -60, 0)
    b31, b61 = t[f"b31_{key}"][dcs - 1], t[f"b61_{key}"][dcs - 1]
    qdc, var61, wet = t[f"Qdc_{key}"][dcs - 1], t[f"var61_{key}"][dcs - 1], t[f"wet_{key}"][dcs - 1]
    q61epw = ((q31r + b31) + (q61r + b61)) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        qdry = (qdc / var61 + W_EXP * q61epw) / (1 / var61 + W_EXP)
        qdry = np.where(wet, qdc, qdry)
        FF[(f, p)] = (q - qpow * fpow - qdry * (1 - fpow)) / (qpow - qdry)
        qph = np.nanpercentile(q, 10, axis=0)
        FFPH[(f, p)] = (q - qph) / (qpow - qph)
    FF[(f, p)][:, fpow > 0.5] = np.nan  # mostly-water cells, excluded as in 04
    FFPH[(f, p)][:, fpow > 0.5] = np.nan

mdff = (t["MDFF_36_asc"] + t["MDFF_36_dsc"]) / 2
wet36 = t["wet_36_asc"] | t["wet_36_dsc"]
yy, xx = np.mgrid[-2:3, -2:3]
circle = ((yy ** 2 + xx ** 2) <= 2.0 ** 2).astype(float)
circle /= circle.sum()
neigh8 = np.ones((3, 3)); neigh8[1, 1] = 0

# flash detection over passes in time order (dsc then asc each day)
flagged = {k: np.zeros(FF[k].shape, bool) for k in (("36", "asc"), ("36", "dsc"))}
prev36 = np.full((80, 80), np.nan)
n_flash = 0
for i in range(len(days)):
    dc = dcs[i] - 1
    for p in ("dsc", "asc"):
        ff36, ff18 = FF[("36", p)][i], FF[("18", p)][i]
        newly = (ff18 > 0.05) & (prev36 < 0.05)
        frac = convolve(newly.astype(float), circle, mode="constant")
        cand = newly & (frac > 0.75) & ~wet36[dc] & ~(ff36 < mdff[dc])
        cand &= convolve(cand.astype(float), neigh8, mode="constant") > 0  # drop isolated
        if cand.any():
            lab, _ = label(ff36 > 0.025)
            keep = np.isin(lab, np.unique(lab[cand & (lab > 0)]))
            flash = (cand | keep) & np.isfinite(ff36)
            flagged[("36", p)][i] = flash
            n_flash += int(flash.sum())
        prev36 = np.where(np.isfinite(ff36) & ~flagged[("36", p)][i], ff36, prev36)
print(f"flash detection flagged {n_flash} pass-cells over {len(days)} days")


def daily(ffd, use_flags, apply_mdff):
    out = []
    for i, d in enumerate(days):
        if not (T0 <= d <= T1):
            continue
        vals, wts, today_n = [], [], np.zeros((80, 80))
        for k, w in ((i, 0.67), (i - 1, 0.33)):
            for p in ("asc", "dsc"):
                x = ffd[("36", p)][k].copy()
                if use_flags:
                    x[flagged[("36", p)][k]] = np.nan
                vals.append(np.where(np.isfinite(x), x, 0.0) * w)
                wts.append(np.isfinite(x) * w)
                if k == i:
                    today_n += np.isfinite(x)
        W = np.sum(wts, 0)
        n_all = np.sum([wt > 0 for wt in wts], 0)
        m = np.where(W > 0, np.sum(vals, 0) / np.where(W > 0, W, 1), np.nan)
        m[(n_all == 1) & (today_n == 0)] = np.nan
        m = np.clip(m, 0, 1 - fpow)
        if apply_mdff:
            m = np.where(m < mdff[dcs[i] - 1], 0.0, m)
        out.append(m)
    return np.stack(out).astype("float32")


test_days = [d for d in days if T0 <= d <= T1]
out = {"full": daily(FF, True, True), "noflash": daily(FF, False, True), "placeholder": daily(FFPH, False, False)}
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, days=np.array([d.isoformat() for d in test_days]), fpow=fpow, focean=t["focean"], **out)
for k, v in out.items():
    print(f"{k:12s}: valid {np.isfinite(v).mean():.3f}, mean ff {np.nanmean(v):.4f}, share of cell-days > 0: {(v > 0).mean():.3f}")
print(f"wrote {OUT}")

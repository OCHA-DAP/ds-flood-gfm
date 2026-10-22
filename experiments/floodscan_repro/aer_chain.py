"""Shared pieces of AER's flooded-fraction chain (ADD s. 3.8.1) used by 04 (tables, f_hop) and 05 (daily run).

Grid: the CSU 0.25-deg West Africa box (80 x 80, north-up after flipud). Day-of-year is on a 365-day
calendar (clim_doy). Tables come from 04 (static_tables.npz).
"""

import datetime as dt
from pathlib import Path

import numpy as np

from rainflags import final_rain, scattering_index

FREQS, PASSES = ("36", "18"), ("asc", "dsc")
FLAG_CHANNELS = ("18v", "18h", "23v", "89va")  # 19 V/H, 22 V, 85 V in AER's terms
W_EXP = 1 / 0.0012  # weight on the expected 61-day median in Q_dry (eq. 17)
W_SP = 5  # AER's 11 x 11 spatial window at 5' (0.92 deg) -> 5 x 5 at 0.25 deg; the 3 x 3 centre is excluded
# Minimum valid share of a running-median window. AER states none. AMSR2 sees ~55% of the box's cells per
# pass per day (swath gaps near the equator) and rain flags remove more, so 0.5 left many cells without a
# 61-day median (2026-10-22 in-sample test); 0.25 is >= 15 of 61 days and >= 8 of 31.
MIN_FRAC = 0.25


def clim_doy(d):
    return dt.date(2023, d.month, d.day).timetuple().tm_yday


def running_median(x, lo, hi, min_frac=MIN_FRAC, clip=False):
    """nanmedian over days [i+lo, i+hi]; NaN where the window leaves the record (unless clip: the window is
    cut at the record's ends, the min_frac test still counts against the full width) or < min_frac is valid."""
    out = np.full_like(x, np.nan)
    n = hi - lo + 1
    for i in range(len(x)):
        a, b = i + lo, i + hi
        if a < 0 or b >= len(x):
            if not clip:
                continue
            a, b = max(a, 0), min(b, len(x) - 1)
        w = x[a : b + 1]
        ok = np.isfinite(w).sum(0) >= min_frac * n
        with np.errstate(all="ignore"):
            out[i] = np.where(ok, np.nanmedian(w, 0), np.nan)
    return out


def running_mean(x, half):
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - half) : i + half + 1]
        with np.errstate(all="ignore"):
            out[i] = np.nanmean(w, 0)
    return out


def load_year(src, d0, d1):
    """Q (36, 18 GHz) and the rain-flag channels per pass for days d0..d1 from 03's npz files.
    Days listed in missing.txt stay NaN (upstream absence); any other missing day raises."""
    src = Path(src)
    days = [d0 + dt.timedelta(k) for k in range((d1 - d0).days + 1)]
    miss_file = src.parent / "missing.txt"
    absent = {line.split()[0] for line in miss_file.read_text().splitlines()} if miss_file.exists() else set()
    q = {(f, p): np.full((len(days), 80, 80), np.nan) for f in FREQS for p in PASSES}
    tb = {(c, p): np.full((len(days), 80, 80), np.nan) for c in FLAG_CHANNELS for p in PASSES}
    for i, d in enumerate(days):
        path = src / f"{d:%Y%m%d}.npz"
        if not path.exists():
            if d.isoformat() in absent:
                continue
            raise FileNotFoundError(f"{path} missing and not recorded absent: run 03 for {d.year} first")
        z = np.load(path)
        for p in PASSES:
            for f in FREQS:
                v, h = np.flipud(z[f"fcdr_tb{f}v_{p}"]), np.flipud(z[f"fcdr_tb{f}h_{p}"])  # CSU rows run south->north
                pri = (v - h) / (v + h)
                with np.errstate(invalid="ignore"):
                    q[(f, p)][i] = np.sqrt(pri + 0.11) - 0.58 * pri
            for c in FLAG_CHANNELS:
                if f"fcdr_tb{c}_{p}" not in z.files:
                    raise KeyError(f"{path}: no {c} GHz channel; run 03 with bands 36,18,23,89 for {d.year}")
                tb[(c, p)][i] = np.flipud(z[f"fcdr_tb{c}_{p}"])
    return days, q, tb


def rain_flags(tb, si0, dcs):
    """Final rain flag per pass (rainflags.py) from the channel stack and the SI0 climatology {pass: (365, ny, nx)}."""
    out = {}
    for p in PASSES:
        si = scattering_index(tb[("18v", p)], tb[("23v", p)], tb[("89va", p)])
        out[p] = final_rain(si, si0[p][dcs - 1], tb[("18v", p)], tb[("18h", p)], tb[("23v", p)], tb[("89va", p)])
    return out


def spatial_departure(q61epw, q61c):
    """Q61Esp - Q61c (s. 3.8.1): median over the W_SP window minus its 3 x 3 centre of (Q61Epw - Q61c),
    over cells where Q61Epw > Q61c. NaN where no neighbour qualifies."""
    r = W_SP // 2
    with np.errstate(invalid="ignore"):
        dev = np.where(q61epw > q61c, q61epw - q61c, np.nan)
    pad = np.pad(dev, ((0, 0), (r, r), (r, r)), constant_values=np.nan)
    shifts = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if max(abs(dy), abs(dx)) <= 1:
                continue
            shifts.append(pad[:, r + dy : r + dy + dev.shape[1], r + dx : r + dx + dev.shape[2]])
    with np.errstate(all="ignore"):
        return np.nanmedian(np.stack(shifts), 0)


def initial_ff(q, tables, dcs, fpow, freqs=FREQS, w_exp=W_EXP, qpow_offset=0.0):
    """Initial flooded fractions per frequency and pass (s. 3.8.1): {'pw': ff_pw, '61': ff_61, 'sp': ff_sp}.
    q: {(f, p): (ndays, ny, nx)} with rain-flagged passes already NaN.
    w_exp and qpow_offset are the fit knobs of 07 (AER: 1/0.0012 and the table value)."""
    out = {k: {} for k in ("pw", "61", "sp")}
    for f in freqs:
        for p in PASSES:
            key = f"{f}_{p}"
            qpow = float(tables[f"Qpow_{key}"]) + qpow_offset
            x = q[(f, p)]
            with np.errstate(invalid="ignore", divide="ignore"):
                ql = (x - fpow * qpow) / (1 - fpow)
            q31r, q61r = running_median(ql, -30, 0), running_median(ql, -60, 0)
            b31, b61 = tables[f"b31_{key}"][dcs - 1], tables[f"b61_{key}"][dcs - 1]
            qdc, var61, wet = tables[f"Qdc_{key}"][dcs - 1], tables[f"var61_{key}"][dcs - 1], tables[f"wet_{key}"][dcs - 1]
            q61c = tables[f"Q61c_{key}"][dcs - 1]
            q61e = q61r + b61  # eq. 15, a61 = 1
            q61epw = ((q31r + b31) + q61e) / 2  # eqs. 14, 16
            q61esp = q61c + np.nan_to_num(spatial_departure(q61epw, q61c), nan=0.0)  # no qualifying neighbour: Q61c
            for name, qexp in (("61", q61e), ("pw", q61epw), ("sp", q61esp)):
                with np.errstate(invalid="ignore", divide="ignore"):
                    qdry = (qdc / var61 + w_exp * qexp) / (1 / var61 + w_exp)  # eq. 17
                    qdry = np.where(wet, qdc, qdry)
                    ff = (x - qpow * fpow - qdry * (1 - fpow)) / (qpow - qdry)  # eq. 13
                ff[:, fpow > 0.5] = np.nan  # mostly-water cells, excluded as in 04
                out[name][(f, p)] = ff
    return out

"""Step 1b: AER's flooded-fraction chain (ADD s. 3.8) for 2024, on the CSU AMSR2 0.25-deg grid.

Uses the static tables from 04 (built from 2023 and 2025; 2024 is the test year) and aer_chain.py.
Per pass (dsc then asc each day, AMSR2 ~01:30 and ~13:30 local):
  rain flags (s. 3.7)        rainflags.py with the SI0 climatology; flagged cells carry no Q for that pass
  initial FF (s. 3.8.1)      aer_chain.initial_ff: ff_pw, ff_61, ff_sp per frequency from Q_dry built on
                             Q61Epw, Q61E and the spatial Q61Esp (5x5 window minus its 3x3 centre)
Flash false positives (s. 3.8.2), on ff_pw: newly flooded = ff18 > 0.05 where the previous pass's ff36 < 0.05;
  candidates where > 75% of a 4-cell-diameter circle (~93 km, AER's 10 px at 5') is newly flooded; screens:
  a valid prior FF within t_flash = 1.5 + 4.75 fhop days, not wetland, ff36 >= MDFF, not isolated; dilation
  over connected cells with ff36 > 0.025 or rain-flagged cells whose last test was flash- or rain-flagged.
Daily finishing (s. 3.8.3), 36 GHz only, per FF type:
  weights today 0.67 - 0.33 fhop, 1 day prior 0.33, 2 days prior 0.33 fhop; group outliers (OUR TEST: more
  than OUTLIER_DEV from the group median) at half weight; no prior-day data: the last earlier instantaneous
  value at half the prior-day weight; a lone value only if it is today's.
  merge (eq. 18): where the final FF was above MDFF on >= 12 of the last 15 days, a contiguous region with
  ff_sp and ff_61 above MDFF takes ff = max(w ff_sp + (1 - w) ff_pw, ff_pw), w = clip((N - 11) / 3, 0, 1).
  fill-in for days without input: f0 = f-1 - a max(T0 - T-1, 0) (f-1 - f_floor), a = 0.05 if f-1 < f-2 + 0.015
  else 0.2, f_floor = f-2 if f-1 > f-2 + 0.015 else 0 (OUR READING: the ADD's typeset formula is garbled in
  extraction; this is the form that matches its stated behaviour).
  clip to [0, 1 - f_pow]; ff < MDFF -> 0, except (s. 3.8.3) near-coast cells next to a cell above MDFF, which
  take min(ff, half the lowest such neighbour) (OUR READING of "interpolation between zero and the neighbour").
MDFF is the mean of the asc/dsc tables.

Variants written for before/after (06 scores every key):
  full        everything above
  noflash     no flash filter
  nomerge     no spatial merge (point-wise FF only)
  nofhop      fhop = 0 (fixed 0.67/0.33 weights, no 2-day term)
  nocoast     no coastal MDFF exception
  clim_raw    no MDFF at all (and so no merge)
  mdff_half   MDFF halved
  norain      no daily rain flags (the training tables keep theirs), no flash filter

    uv run --no-project --with numpy --with scipy python experiments/floodscan_repro/05_ff_chain.py
"""

import datetime as dt
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve, label

from aer_chain import PASSES, clim_doy, initial_ff, load_year, rain_flags

SRC = Path("data/floodscan_repro/csu_amsr2_grid/2024")
TAB = Path("data/floodscan_repro/static_tables.npz")
OUT = Path("data/floodscan_repro/ff_2024.npz")
if len(sys.argv) > 1:  # e.g. an in-sample code test: `... 05_ff_chain.py /tmp/tables_2024.npz /tmp/ff.npz`
    TAB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
D0, D1 = dt.date(2024, 6, 1), dt.date(2024, 12, 31)
T0, T1 = dt.date(2024, 8, 1), dt.date(2024, 11, 30)
# Fit knobs (07): a JSON file as argv[3] with any of w_exp, qpow_offset, mdff_scale, outlier_dev, variants.
P = {"w_exp": 1 / 0.0012, "qpow_offset": 0.0, "mdff_scale": 1.0, "outlier_dev": 0.10, "variants": None}
if len(sys.argv) > 3:
    overrides = json.loads(Path(sys.argv[3]).read_text())
    unknown = set(overrides) - set(P)
    if unknown:
        raise KeyError(f"unknown fit parameters {sorted(unknown)}")
    P.update(overrides)
OUTLIER_DEV = P["outlier_dev"]

t = np.load(TAB)
fpow, near_coast = t["fpow"], t["near_coast"]
days, Q, TB = load_year(SRC, D0, D1)
dcs = np.array([clim_doy(d) for d in days])
NT = len(days)

# --- rain flags per pass (s. 3.7)
RAIN = rain_flags(TB, {p: t[f"SI0_{p}"] for p in PASSES}, dcs)
for p in PASSES:
    valid = np.isfinite(TB[("18v", p)]) & (t["focean"] <= 0.001)[None]
    print(f"2024 {p}: rain-flagged {RAIN[p][valid].mean():.3f} of valid inland cell-days")
Q_flagged = {k: np.where(RAIN[k[1]], np.nan, q) for k, q in Q.items()}

FF = initial_ff(Q_flagged, t, dcs, fpow, w_exp=P["w_exp"], qpow_offset=P["qpow_offset"])  # {'pw' | '61' | 'sp': {(f, p): ...}}
FF_NR = initial_ff(Q, t, dcs, fpow, w_exp=P["w_exp"], qpow_offset=P["qpow_offset"])

mdff = (t["MDFF_36_asc"] + t["MDFF_36_dsc"]) / 2 * P["mdff_scale"]
fhop_tab = t["fhop"]
if not np.isfinite(fhop_tab[dcs[(days.index(T0)) : (days.index(T1) + 1)] - 1][:, fpow <= 0.5]).all():
    raise ValueError("fhop table has gaps inside the test window")
wet36 = t["wet_36_asc"] | t["wet_36_dsc"]
yy, xx = np.mgrid[-2:3, -2:3]
circle = ((yy ** 2 + xx ** 2) <= 2.0 ** 2).astype(float)
circle /= circle.sum()
neigh8 = np.ones((3, 3)); neigh8[1, 1] = 0

# --- flash detection over passes in time order (dsc then asc each day), s. 3.8.2
flagged = {p: np.zeros((NT, 80, 80), bool) for p in PASSES}
prev36 = np.full((80, 80), np.nan)
last_valid_t = np.full((80, 80), -np.inf)  # time (days) of the last valid ff36
prev_flash_or_rain = np.zeros((80, 80), bool)  # last test result at each cell (step 3b)
n_flash = 0
for i in range(NT):
    dc = dcs[i] - 1
    t_flash = 1.5 + 4.75 * np.nan_to_num(fhop_tab[dc], nan=0.0)
    for p, tnow in (("dsc", i + 0.0), ("asc", i + 0.5)):
        ff36, ff18, rain = FF["pw"][("36", p)][i], FF["pw"][("18", p)][i], RAIN[p][i]
        newly = (ff18 > 0.05) & (prev36 < 0.05)
        frac = convolve(newly.astype(float), circle, mode="constant")
        cand = newly & (frac > 0.75) & (tnow - last_valid_t <= t_flash) & ~wet36[dc] & ~(ff36 < mdff[dc])
        cand &= convolve(cand.astype(float), neigh8, mode="constant") > 0  # drop isolated
        flash = np.zeros((80, 80), bool)
        if cand.any():
            lab, _ = label((ff36 > 0.025) | (rain & prev_flash_or_rain))
            keep = np.isin(lab, np.unique(lab[cand & (lab > 0)]))
            flash = (cand | keep) & np.isfinite(ff36)
            flagged[p][i] = flash
            n_flash += int(flash.sum())
        observed = np.isfinite(TB[("18v", p)][i])
        prev_flash_or_rain = np.where(observed, flash | rain, prev_flash_or_rain)
        good = np.isfinite(ff36) & ~flash
        prev36 = np.where(good, ff36, prev36)
        last_valid_t = np.where(good, tnow, last_valid_t)
print(f"flash detection flagged {n_flash} pass-cells over {NT} days")


def neighbour_min_above(x, thr):
    """Lowest 8-neighbour value that is >= thr (inf where none)."""
    y = np.where(x >= thr, x, np.inf)
    pad = np.pad(y, 1, constant_values=np.inf)
    out = np.full_like(x, np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out = np.minimum(out, pad[1 + dy : 81 + dy, 1 + dx : 81 + dx])
    return out


def finish(ffsets, use_flash=True, mdff_scale=1.0, use_merge=True, use_fhop=True, coastal=True):
    """Daily final FF (s. 3.8.3) for the test window. ffsets: {'pw' | '61' | 'sp': {(f, p): ...}}."""
    hist = deque(maxlen=15)  # final FF above MDFF, per day
    f1, f2, t1 = np.full((80, 80), np.nan), np.full((80, 80), np.nan), np.full((80, 80), np.nan)
    last_inst, last_inst_i = np.full((80, 80), np.nan), np.full((80, 80), -1)
    out = []
    for i in range(NT):
        dc = dcs[i] - 1
        fh = np.nan_to_num(fhop_tab[dc], nan=0.0) if use_fhop else np.zeros((80, 80))
        w_day = {0: 0.67 - 0.33 * fh, 1: np.full((80, 80), 0.33), 2: 0.33 * fh}
        avg = {}
        for kind in ("pw", "61", "sp"):
            vals, wts = [], []
            for back in (0, 1, 2):
                k = i - back
                if k < 0 or (back == 2 and not use_fhop):
                    continue
                for p in PASSES:
                    x = ffsets[kind][("36", p)][k].copy()
                    if use_flash:
                        x[flagged[p][k]] = np.nan
                    vals.append(x)
                    wts.append(np.where(np.isfinite(x), w_day[back], 0.0))
            if kind == "pw":
                today_valid = np.isfinite(vals[0]) | np.isfinite(vals[1])
                prior_valid = (np.isfinite(vals[2]) | np.isfinite(vals[3])) if len(vals) > 3 else np.zeros((80, 80), bool)
            # no prior-day data: the last earlier instantaneous value at half the prior-day weight
            if i >= 1:
                fallback = ~prior_valid & np.isfinite(last_inst) & (last_inst_i < i - 1)
                vals.append(np.where(fallback, last_inst, np.nan))
                wts.append(np.where(fallback, 0.33 / 2, 0.0))
            V, Wt = np.stack(vals), np.stack(wts)
            with np.errstate(all="ignore"):
                med = np.nanmedian(V, 0)
                outlier = np.abs(V - med) > OUTLIER_DEV
                Wt = np.where(outlier, Wt / 2, Wt)
                W = Wt.sum(0)
                m = np.where(W > 0, np.nansum(np.nan_to_num(V) * Wt, 0) / np.where(W > 0, W, 1), np.nan)
            n_valid = np.isfinite(V).sum(0)
            m[(n_valid == 1) & ~today_valid] = np.nan
            avg[kind] = m
        if i >= 1:  # instantaneous record for the fallback (the previous day's passes, in order)
            for p in ("dsc", "asc"):
                x = ffsets["pw"][("36", p)][i - 1]
                ok = np.isfinite(x) & (~flagged[p][i - 1] if use_flash else True)
                last_inst = np.where(ok, x, last_inst)
                last_inst_i = np.where(ok, i - 1, last_inst_i)
        thr = mdff[dc] * mdff_scale if mdff_scale else None
        final = avg["pw"].copy()
        if use_merge and thr is not None and len(hist) == 15:
            nff = np.sum(hist, 0)
            trig = nff >= 12
            with np.errstate(invalid="ignore"):
                region = (avg["sp"] > thr) & (avg["61"] > thr)
            if (trig & region).any():
                lab, n = label(region)
                wsp = np.zeros((80, 80))
                for r in np.unique(lab[trig & region]):
                    if r > 0:
                        wsp[lab == r] = np.clip((nff[trig & (lab == r)].max() - 11) / 3, 0, 1)
                with np.errstate(invalid="ignore"):
                    merged = np.fmax(wsp * avg["sp"] + (1 - wsp) * avg["pw"], avg["pw"])
                final = np.where(wsp > 0, merged, final)
        # fill-in from the last two valid 3-day averages
        gap = ~np.isfinite(final) & np.isfinite(f1)
        if gap.any():
            with np.errstate(invalid="ignore"):
                a = np.where(f1 < f2 + 0.015, 0.05, 0.2)
                floor = np.where(f1 > f2 + 0.015, f2, 0.0)
                a, floor = np.where(np.isfinite(f2), a, 0.2), np.where(np.isfinite(f2), floor, 0.0)
                fill = f1 - a * np.maximum(i - t1, 0) * (f1 - floor)
                final = np.where(gap, np.maximum(fill, floor), final)
        final = np.clip(final, 0, 1 - fpow)
        if thr is not None:
            with np.errstate(invalid="ignore"):
                below = final < thr
                if coastal:
                    nmin = neighbour_min_above(np.nan_to_num(final, nan=-1.0), thr)
                    excepted = below & near_coast & np.isfinite(nmin)
                    final = np.where(excepted, np.minimum(final, 0.5 * nmin), final)
                    below &= ~excepted
            final = np.where(below, 0.0, final)
            hist.append(np.nan_to_num(final, nan=0.0) >= thr)
        valid_avg = np.isfinite(avg["pw"])
        f2 = np.where(valid_avg, f1, f2)
        f1 = np.where(valid_avg, avg["pw"], f1)
        t1 = np.where(valid_avg, i, t1)
        if T0 <= days[i] <= T1:
            out.append(final)
    return np.stack(out).astype("float32")


test_days = [d for d in days if T0 <= d <= T1]
recipes = {
    "full": lambda: finish(FF),
    "noflash": lambda: finish(FF, use_flash=False),
    "nomerge": lambda: finish(FF, use_merge=False),
    "nofhop": lambda: finish(FF, use_fhop=False),
    "nocoast": lambda: finish(FF, coastal=False),
    "clim_raw": lambda: finish(FF, mdff_scale=0),
    "mdff_half": lambda: finish(FF, mdff_scale=0.5),
    "norain": lambda: finish(FF_NR, use_flash=False),
}
wanted = P["variants"] or list(recipes)
unknown = set(wanted) - set(recipes)
if unknown:
    raise KeyError(f"unknown variants {sorted(unknown)}")
out = {k: recipes[k]() for k in wanted}
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, days=np.array([d.isoformat() for d in test_days]), fpow=fpow, focean=t["focean"], **out)
for k, v in out.items():
    print(f"{k:12s}: valid {np.isfinite(v).mean():.3f}, mean ff {np.nanmean(v):.4f}, share of cell-days > 0: {(v > 0).mean():.3f}")
print(f"wrote {OUT}")

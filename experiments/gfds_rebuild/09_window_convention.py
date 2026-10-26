"""Is GFDS's calibration pixel C computed on the 0.09-deg grid, and by which percentile convention?

Uses a day where JRC's per-sensor files are complete (2016-08-20: SourceTiffs T_M, CalibrationTiffs T_C,
PositionTiffs, SignalTiffs). If C comes from a 9x9 window of *grid cells* of T_M, recomputing it from JRC's
own T_M must reproduce T_C exactly for some convention; the position file then decodes as a window offset.
Also checks s = T_M / T_C against the signal file (scaling and rounding).

    uv run --no-project --with numpy --with rasterio python experiments/gfds_rebuild/09_window_convention.py 2016-08-20
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import rasterio
from numpy.lib.stride_tricks import sliding_window_view
from rasterio.windows import Window

RES, WIN, H = 0.09, 9, 4
LON0, LON1, LAT0, LAT1 = -2.0, 22.0, -2.0, 22.0  # box plus a margin so the window is complete inside 0-20
R0, R1 = int((90 - LAT1) / RES), int(np.ceil((90 - LAT0) / RES))
C0, C1 = int((LON0 + 180) / RES), int(np.ceil((LON1 + 180) / RES))
NR, NC = R1 - R0, C1 - C0
day = dt.date.fromisoformat(sys.argv[1])
KEY = Path(f"data/gdacs_gfds/rebuild/answer_key_{day:%Y%m%d}")


def crop(pattern):
    path = next(KEY.glob(pattern))
    with rasterio.open(path) as ds:
        a = ds.read(1, window=Window(C0, R0, NC, NR)).astype("float64")
    a[(a == -32000) | (a == np.iinfo("int32").min)] = np.nan
    return a, path.name


tm, n1 = crop(f"{day:%Y%m%d}_HD5*.tif")
tc, n2 = crop(f"calibration_{day:%Y%m%d}*.tif")
pos, n3 = crop(f"position_{day:%Y%m%d}*.tif")
sig, n4 = crop(f"signal_{day:%Y%m%d}*.tif")
print("files:", n1, n2, n3, n4)
for name, a in (("T_M", tm), ("T_C", tc), ("position", pos), ("signal", sig)):
    v = a[np.isfinite(a)]
    print(f"  {name:9s} valid {v.size:7d}  min {v.min():.0f}  median {np.median(v):.0f}  max {v.max():.0f}  distinct {len(np.unique(v))}")
inner = np.zeros((NR, NC), bool)
inner[H:-H, H:-H] = True
ok = inner & np.isfinite(tm) & np.isfinite(tc)

# --- s = T_M / T_C ?
with np.errstate(invalid="ignore", divide="ignore"):
    s = tm / tc
m = ok & np.isfinite(sig)
d = s[m] - sig[m] / 1e6
print(f"signal vs T_M/T_C: r {np.corrcoef(s[m], sig[m] / 1e6)[0, 1]:.6f}, mad {np.median(np.abs(d)):.7f}, exact(<1e-6) {(np.abs(d) < 1e-6).mean():.4f}")

# --- C from a 9x9 window of grid cells of T_M, several conventions
pad = np.pad(tm, H, constant_values=np.nan)
win = sliding_window_view(pad, (WIN, WIN)).reshape(NR, NC, WIN * WIN)
n = np.isfinite(win).sum(-1)
srt = np.sort(np.where(np.isfinite(win), win, np.inf), axis=-1)  # NaN last
conventions = {}
with np.errstate(all="ignore"):
    conventions["numpy linear p95"] = np.nanpercentile(win, 95, axis=-1)
    for label, k in (("rank ceil(0.95n)", np.ceil(0.95 * n).astype(int) - 1), ("rank floor(0.95n)", np.floor(0.95 * n).astype(int) - 1),
                     ("rank round(0.95(n-1))", np.round(0.95 * (n - 1)).astype(int)), ("rank floor(0.95(n-1))", np.floor(0.95 * (n - 1)).astype(int)),
                     ("5th brightest", n - 5), ("4th brightest", n - 4), ("max", n - 1)):
        kk = np.clip(k, 0, WIN * WIN - 1)
        conventions[label] = np.take_along_axis(srt, kk[..., None], axis=-1)[..., 0]
for label, c in conventions.items():
    mm = ok & np.isfinite(c) & (n == WIN * WIN)
    d = c[mm] - tc[mm]
    print(f"{label:24s} full windows n {int(mm.sum()):6d}  exact {(np.abs(d) < 0.5).mean():.4f}  mad {np.median(np.abs(d)):.2f}  bias {d.mean():+.2f}")

# --- position: decode as the window offset of the pixel equal to T_C
best = None
for label in ("5th brightest", "rank ceil(0.95n)"):
    pass
eq = np.isclose(win, tc[..., None]) & np.isfinite(win)
first = np.argmax(eq, axis=-1)  # first window offset whose value equals T_C (row-major, 0..80, centre 40)
has = eq.any(-1) & ok & np.isfinite(pos)
pv = pos[has].astype(int)
print(f"position: cells where some window cell equals T_C {has.sum()} of {int(ok.sum())}; distinct position values {len(np.unique(pv))}, "
      f"range {pv.min()}..{pv.max()}")
for name, cand in (("row-major offset 0..80", first), ("offset + 1", first + 1), ("col-major offset", (first % WIN) * WIN + first // WIN),
                   ("dr*10+dc (signed)", (first // WIN - H) * 10 + (first % WIN - H))):
    print(f"  {name:24s} matches {np.mean(cand[has] == pv):.4f}")

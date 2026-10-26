"""Exactness test: GFDS's AMSR2 signal from raw orbit footprints, against JRC's own answer-key files for one day.

JRC (GFDS tech note): per footprint, M = 36.5 GHz H-pol Tb, C = the 95th percentile of the 9x9 footprint
neighbourhood in swath geometry, s = M/C; per day, each 0.09-deg cell (4000x2000 from 180W/90N) takes the
mean of its samples. JRC's per-sensor files for 2013 to 2018-09 hold the pieces: SourceTiffs (T_M),
CalibrationTiffs (T_C), PositionTiffs (which of the 81 window cells was C), SignalTiffs (s x 1e6), and
MagTiffs (m x 1e3) with the baseline in AveragesAndSd. So each step can be checked on its own.

Input: CSU AMSR2 FCDR orbit files (per-variable range reads; the 36 GHz H channel, geolocation, scan
time) for the orbits dated d-1 and d; footprints are assigned to the UTC date of their scan. Only scan rows
near the West Africa box (0-20E, 0-20N) are processed, with 4 extra rows each side for the window.
Two percentile conventions are computed, since the note does not say: 'lin' (numpy linear interpolation
over the valid window cells) and 'rank' (the 5th brightest of 81: nearest rank, ceil(0.95 n) - 1 ascending).
Two daily aggregations: mean of per-footprint s, and mean(M) / mean(C).

Output: data/gdacs_gfds/rebuild/swath_signal/YYYYMMDD.npz (box crop on JRC's grid) and a printed comparison
with JRC's five files for the day (data/gdacs_gfds/rebuild/answer_key_YYYYMMDD/).

    uv run --no-project --with h5py --with fsspec --with aiohttp --with numpy --with scipy --with rasterio \
        python experiments/gfds_rebuild/07_signal_from_swath.py 2018-08-20
"""

import datetime as dt
import re
import sys
from pathlib import Path

import aiohttp
import fsspec
import h5py
import numpy as np
import rasterio
from numpy.lib.stride_tricks import sliding_window_view
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "floodscan_repro"))
from swath import BUCKET, _orbits_dated, get, read_var  # noqa: E402

LON0, LON1, LAT0, LAT1 = 0.0, 20.0, 0.0, 20.0
RES = 0.09
R0, R1 = int((90 - LAT1) / RES), int(np.ceil((90 - LAT0) / RES))  # JRC rows 777..1000
C0, C1 = int((LON0 + 180) / RES), int(np.ceil((LON1 + 180) / RES))  # cols 2000..2223
NR, NC = R1 - R0, C1 - C0
WIN, H = 9, 4
day = dt.date.fromisoformat(sys.argv[1])
KEY = Path(f"data/gdacs_gfds/rebuild/answer_key_{day:%Y%m%d}")
OUT = Path("data/gdacs_gfds/rebuild/swath_signal")
MASK = Path("data/gdacs_gfds/rebuild/ALL/oceanmask.tif")
BASE = Path("data/gdacs_gfds/rebuild/AMSR2/AveragesAndSd")


def chunk_info_units(key, variables):
    url = f"{BUCKET}/{key}"
    opts = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=300, sock_read=120)}}
    info, units = {}, {}
    with fsspec.open(url, "rb", block_size=2**16, cache_type="blockcache", **opts) as fo, h5py.File(fo) as h:
        for v in variables:
            ds = h[v]
            if ds.id.get_num_chunks() != 1:
                raise ValueError(f"{key}:{v}: {ds.id.get_num_chunks()} chunks, expected one")
            ci = ds.id.get_chunk_info(0)
            info[v] = (ci.byte_offset, ci.size, ds.dtype, ds.shape)
            units[v] = ds.attrs.get("units", b"").decode() if isinstance(ds.attrs.get("units", b""), bytes) else str(ds.attrs.get("units", ""))
    return info, units


def epoch_of(units):
    m = re.match(r"seconds since (\d{4}-\d{2}-\d{2})[T ]?(\d{2}:\d{2}:\d{2})?", units)
    if not m:
        raise ValueError(f"scan_time units not understood: {units!r}")
    return dt.datetime.fromisoformat(m.group(1) + "T" + (m.group(2) or "00:00:00"))


def window_stats(tb):
    """Per footprint: C by two conventions and the rank pixel's window offset (0-80, row-major)."""
    pad = np.pad(tb, H, constant_values=np.nan)
    win = sliding_window_view(pad, (WIN, WIN)).reshape(tb.shape[0], tb.shape[1], WIN * WIN)
    n = np.isfinite(win).sum(-1)
    with np.errstate(all="ignore"):
        c_lin = np.nanpercentile(win, 95, axis=-1)
    order = np.argsort(np.where(np.isfinite(win), win, -np.inf), axis=-1)  # NaN sorted first as -inf
    k = np.clip(np.ceil(0.95 * n).astype(int) - 1, 0, WIN * WIN - 1)  # nearest rank among the valid
    idx = np.take_along_axis(order, (WIN * WIN - n + k)[..., None], axis=-1)[..., 0]  # skip the NaN (-inf) slots
    c_rank = np.take_along_axis(win, idx[..., None], axis=-1)[..., 0]
    ok = n >= 41
    return np.where(ok, c_lin, np.nan), np.where(ok, c_rank, np.nan), np.where(ok, idx, -1), n


# --- footprints from the orbits dated d-1, d and d+1. The grids below use the UTC day; the saved footprints
# span [d - 12 h, d + 36 h) so 08 can test other day windows (JRC's grouping is not documented).
keys = []
for dd in (day - dt.timedelta(1), day, day + dt.timedelta(1)):
    kind, ks = _orbits_dated(dd)
    if not ks:
        raise FileNotFoundError(f"no AMSR2 orbit files listed for {dd}")
    keys += ks
SPAN = (dt.timedelta(hours=-12), dt.timedelta(hours=36))
sums = {k: np.zeros((NR, NC)) for k in ("m", "c_lin", "c_rank", "s_lin", "s_rank")}
cnt = np.zeros((NR, NC))
pos_hist = np.zeros(WIN * WIN, int)
fp_samples = []  # per footprint: lat, lon, M, C_lin, C_rank, seconds since 00 UTC, ascending flag (saved for offline gridding tests)
mb, used = 0.0, 0
t_lo, t_hi = dt.datetime(day.year, day.month, day.day), dt.datetime(day.year, day.month, day.day) + dt.timedelta(1)
for key in keys:
    info, units = chunk_info_units(key, ["spacecraft_lat", "spacecraft_lon", "scan_time", "lat_tb36", "lon_tb36", "fcdr_tb36h"])
    sc_lat, sc_lon = read_var(key, *info["spacecraft_lat"]), read_var(key, *info["spacecraft_lon"])
    sc_lon = np.where(sc_lon > 180, sc_lon - 360, sc_lon)
    mb += (info["spacecraft_lat"][1] + info["spacecraft_lon"][1]) / 1e6
    if not ((sc_lat >= LAT0 - 8) & (sc_lat <= LAT1 + 8) & (sc_lon >= LON0 - 10) & (sc_lon <= LON1 + 10)).any():
        continue
    tsec = read_var(key, *info["scan_time"])
    epoch = epoch_of(units["scan_time"])
    tscan = np.array([epoch + dt.timedelta(seconds=float(s)) for s in tsec])
    in_day = (tscan >= t_lo) & (tscan < t_hi)
    in_span = (tscan >= t_lo + SPAN[0]) & (tscan < t_lo + SPAN[1])
    lat, lon = read_var(key, *info["lat_tb36"]), read_var(key, *info["lon_tb36"])
    lon = np.where(lon > 180, lon - 360, lon)
    th = read_var(key, *info["fcdr_tb36h"])
    mb += sum(info[v][1] for v in ("scan_time", "lat_tb36", "lon_tb36", "fcdr_tb36h")) / 1e6
    th[th <= 0] = np.nan
    rows = np.where(((lat >= LAT0) & (lat < LAT1) & (lon >= LON0) & (lon < LON1)).any(1) & in_span)[0]
    if len(rows) == 0:
        continue
    r0, r1 = max(rows.min() - H, 0), min(rows.max() + H + 1, th.shape[0])
    c_lin, c_rank, pos, n = window_stats(th[r0:r1])
    sel = (lat[r0:r1] >= LAT0) & (lat[r0:r1] < LAT1) & (lon[r0:r1] >= LON0) & (lon[r0:r1] < LON1) & in_span[r0:r1, None] & np.isfinite(c_lin) & np.isfinite(th[r0:r1])
    rr = ((90 - lat[r0:r1][sel]) / RES).astype(int) - R0
    cc = ((lon[r0:r1][sel] + 180) / RES).astype(int) - C0
    inside = (rr >= 0) & (rr < NR) & (cc >= 0) & (cc < NC)  # a footprint exactly on the box edge rounds outside
    rr, cc = rr[inside], cc[inside]
    m = th[r0:r1][sel][inside]
    cl, cr = c_lin[sel][inside], c_rank[sel][inside]
    today = np.broadcast_to(in_day[r0:r1, None], sel.shape)[sel][inside]
    for name, val in (("m", m), ("c_lin", cl), ("c_rank", cr), ("s_lin", m / cl), ("s_rank", m / cr)):
        np.add.at(sums[name], (rr[today], cc[today]), val[today])
    np.add.at(cnt, (rr[today], cc[today]), 1)
    np.add.at(pos_hist, pos[sel][inside][today], 1)
    tsel = np.broadcast_to((tscan[r0:r1] - t_lo)[:, None], sel.shape)[sel][inside]
    asc_scan = np.gradient(sc_lat)[r0:r1]
    fp_samples.append(np.column_stack([lat[r0:r1][sel][inside], lon[r0:r1][sel][inside], m, cl, cr,
                                       np.array([t.total_seconds() for t in tsel]),
                                       np.broadcast_to((asc_scan > 0)[:, None], sel.shape)[sel][inside].astype(float)]))
    n_today = int(today.sum())
    print(f"  {key.split('/')[-1]}: {int(sel.sum())} footprints in the box within the span, {n_today} on the UTC day", flush=True)
    used += 1
    print(f"  {key.split('/')[-1]}: {int(sel.sum())} footprints in the box on {day}", flush=True)
print(f"{day}: {used} orbit files used, {mb:.0f} MB streamed; cells with samples {int((cnt > 0).sum())} of {NR * NC}")
with np.errstate(invalid="ignore"):
    ours = {k: np.where(cnt > 0, v / cnt, np.nan) for k, v in sums.items()}
    ours["s_ratio_lin"] = ours["m"] / ours["c_lin"]
    ours["s_ratio_rank"] = ours["m"] / ours["c_rank"]
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / f"{day:%Y%m%d}.npz", cnt=cnt, pos_hist=pos_hist, footprints=np.concatenate(fp_samples), **ours)

# --- JRC's files for the day, cropped to the box
if not KEY.exists():
    raise FileNotFoundError(f"{KEY}: download JRC's answer-key files first")


def crop(path):
    with rasterio.open(path) as ds:
        if (ds.height, ds.width) != (2000, 4000):
            raise ValueError(f"{path.name}: grid {(ds.height, ds.width)}")
        a = ds.read(1, window=Window(C0, R0, NC, NR)).astype("float64")
    a[(a == -32000) | (a == np.iinfo("int32").min)] = np.nan
    return a


with rasterio.open(MASK) as ds:
    land = ds.read(1, window=Window(C0, R0, NC, NR)) == 0
def key_file(prefix):  # 2016 files carry a _GZ suffix, 2018 files do not
    hits = sorted(KEY.glob(f"{prefix}{day:%Y%m%d}_HD5*.tif"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{KEY}: expected one {prefix}{day:%Y%m%d}_HD5*.tif, found {hits}")
    return hits[0]


jrc = {"signal": crop(key_file("signal_")) / 1e6, "source": crop(key_file("")),
       "calib": crop(key_file("calibration_")), "position": crop(key_file("position_")),
       "mag": crop(key_file("mag_signal_")) / 1e3}
for k in ("source", "calib"):  # Tb scaling unknown: bring the land median to ~250 K and say what was assumed
    med = np.nanmedian(jrc[k][land])
    scale = 10 ** -round(np.log10(med / 250)) if med > 0 else 1.0
    print(f"JRC {k}: raw land median {med:.1f}, assumed scale {scale:g}")
    jrc[k] = jrc[k] * scale
jrc["signal"][jrc["signal"] <= 0] = np.nan
print(f"JRC position values (box, land): {np.unique(jrc['position'][land & np.isfinite(jrc['position'])])[:20]} ...")
print(f"our rank-pixel window offset histogram (top 5): {sorted(enumerate(pos_hist), key=lambda t: -t[1])[:5]}")


def report(name, a, b):
    ok = np.isfinite(a) & np.isfinite(b) & land
    if ok.sum() < 10:
        print(f"{name:28s} n {int(ok.sum())}")
        return
    d = a[ok] - b[ok]
    print(f"{name:28s} n {int(ok.sum()):6d}  r {np.corrcoef(a[ok], b[ok])[0, 1]:.4f}  bias {d.mean():+.5f}  "
          f"mad {np.median(np.abs(d)):.5f}  p95|d| {np.percentile(np.abs(d), 95):.5f}")


print(f"coverage (land cells): JRC signal {np.isfinite(jrc['signal'])[land].mean():.3f}, ours {np.isfinite(ours['m'])[land].mean():.3f}")
report("M: ours vs JRC source", ours["m"], jrc["source"])
report("C lin: ours vs JRC calib", ours["c_lin"], jrc["calib"])
report("C rank: ours vs JRC calib", ours["c_rank"], jrc["calib"])
report("s lin (mean s)", ours["s_lin"], jrc["signal"])
report("s rank (mean s)", ours["s_rank"], jrc["signal"])
report("s lin (mean M / mean C)", ours["s_ratio_lin"], jrc["signal"])
report("s rank (mean M / mean C)", ours["s_ratio_rank"], jrc["signal"])
report("JRC source/calib vs signal", jrc["source"] / jrc["calib"], jrc["signal"])
if (BASE / "bt_signal_avg.tif").exists():
    avg = crop(BASE / "bt_signal_avg.tif") / 1e6
    sd = crop(BASE / "bt_signal_sd_sf_min_to_0.5.tif") / 1e6
    with np.errstate(invalid="ignore", divide="ignore"):
        report("mag from JRC signal + baseline", (jrc["signal"] - avg) / sd, jrc["mag"])
        report("mag from our s (rank) + baseline", (ours["s_rank"] - avg) / sd, jrc["mag"])

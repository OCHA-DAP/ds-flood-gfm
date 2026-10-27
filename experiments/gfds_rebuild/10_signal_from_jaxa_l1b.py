"""GFDS signal from JAXA's own AMSR2 L1B granules (what JRC read), for the exactness test against JRC's files.

Same method as 07 (M = 36.5 GHz H-pol Tb per footprint, C = 95th percentile of the 9x9 footprint window,
s = M / C; daytime pass first, night pass fills the rest, per 0.09-deg cell), but on JAXA's calibration
instead of NOAA/CSU's. Granules are fetched from G-Portal by SFTP into data/gdacs_gfds/rebuild/jaxa_l1b/
(one scene ~70-100 MB; only scenes whose start time can cover the box's passes are taken).

JAXA L1B HDF5 (AMSR2 L1 format spec): 'Brightness Temperature (36.5GHz,H)' int16 x 0.01 K (scans x 243),
'Latitude of Observation Point for 89A' / 'Longitude ...' (scans x 486); the low-frequency footprint
positions are the odd 89A pixels (index 1, 3, ...) per the spec; 'Scan Time' seconds since 1993-01-01.
Any name that is not found is reported with the file's dataset list, so a wrong assumption cannot pass.

    uv run --no-project --with h5py --with numpy --with scipy --with rasterio --with paramiko --with python-dotenv \
        python experiments/gfds_rebuild/10_signal_from_jaxa_l1b.py 2016-08-20
"""

import datetime as dt
import stat
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import paramiko
import rasterio
from dotenv import dotenv_values
from numpy.lib.stride_tricks import sliding_window_view
from rasterio.windows import Window
from scipy.spatial import cKDTree

ENV = Path(__file__).resolve().parents[2] / ".env"
ROOT = "/standard/GCOM-W/GCOM-W.AMSR2/L1B/2"
CACHE = Path("data/gdacs_gfds/rebuild/jaxa_l1b")
LON0, LON1, LAT0, LAT1 = 0.0, 20.0, 0.0, 20.0
RES, WIN, H = 0.09, 9, 4
R0, R1 = int((90 - LAT1) / RES), int(np.ceil((90 - LAT0) / RES))
C0, C1 = int((LON0 + 180) / RES), int(np.ceil((LON1 + 180) / RES))
NR, NC = R1 - R0, C1 - C0
TB_NAME = "Brightness Temperature (36.5GHz,H)"
LAT_NAME, LON_NAME, TIME_NAME = "Latitude of Observation Point for 89A", "Longitude of Observation Point for 89A", "Scan Time"
EPOCH = dt.datetime(1993, 1, 1)
day = dt.date.fromisoformat(sys.argv[1])
KEY = Path(f"data/gdacs_gfds/rebuild/answer_key_{day:%Y%m%d}")
cfg = dotenv_values(ENV)
# G-Portal moved to a new system on 2026-10-01 (old host ftp.gportal.jaxa.jp:2051 with password login shut down
# 2026-11-04). The new SFTP host sftp.gportal.jaxa.jp:22 accepts public-key login only; the key is registered on
# the website under "User account services". .env: GPORTAL_USER, GPORTAL_KEY (private key path); GPORTAL_HOST /
# GPORTAL_PORT override the defaults; GPORTAL_PASSWORD is used only when no key is configured (old host).
HOST, PORT = cfg.get("GPORTAL_HOST", "sftp.gportal.jaxa.jp"), int(cfg.get("GPORTAL_PORT", "22"))
if not cfg.get("GPORTAL_USER"):
    sys.exit(f"GPORTAL_USER missing from {ENV}")
if cfg.get("GPORTAL_KEY"):
    KEY_PATH = Path(cfg["GPORTAL_KEY"]).expanduser()
    if not KEY_PATH.exists():
        sys.exit(f"GPORTAL_KEY points to {KEY_PATH}, which does not exist")
    AUTH = {"pkey": paramiko.PKey.from_path(str(KEY_PATH))}
elif cfg.get("GPORTAL_PASSWORD"):
    AUTH = {"password": cfg["GPORTAL_PASSWORD"]}
else:
    sys.exit(f"neither GPORTAL_KEY nor GPORTAL_PASSWORD set in {ENV}; the new host needs a registered SSH key")


def connect():
    for attempt in range(6):
        try:
            t = paramiko.Transport((HOST, PORT))
            t.banner_timeout = 60
            t.connect(username=cfg["GPORTAL_USER"], **AUTH)
            s = paramiko.SFTPClient.from_transport(t)
            s.get_channel().settimeout(120)
            return t, s
        except (paramiko.SSHException, OSError) as e:
            if attempt == 5:
                raise
            print(f"  G-Portal connect failed ({type(e).__name__}); retrying in {60 * (attempt + 1)} s", flush=True)
            time.sleep(60 * (attempt + 1))


def granules_for(d):
    """L1B scene files dated d: (name, size, start datetime, A/D) from the file name GW1AM2_YYYYMMDDHHMM_PPPA_..."""
    t, s = connect()
    out = []
    try:
        for e in s.listdir_attr(f"{ROOT}/{d:%Y/%m/%d}"):
            if stat.S_ISDIR(e.st_mode) or not e.filename.endswith(".h5"):
                continue
            stamp = e.filename.split("_")[1]
            out.append((e.filename, e.st_size, dt.datetime.strptime(stamp, "%Y%m%d%H%M"), e.filename.split("_")[2][-1]))
    finally:
        s.close(); t.close()
    return sorted(out, key=lambda x: x[2])


def fetch(d, name):
    local = CACHE / f"{d:%Y}" / name
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(".part")
    for attempt in range(4):
        t, s = connect()
        try:
            s.get(f"{ROOT}/{d:%Y/%m/%d}/{name}", str(tmp))
            tmp.rename(local)
            return local
        except (paramiko.SSHException, OSError) as e:
            print(f"  {name}: transfer dropped ({type(e).__name__}); retrying", flush=True)
            if attempt == 3:
                raise
        finally:
            s.close(); t.close()


def window_stats(tb):
    pad = np.pad(tb, H, constant_values=np.nan)
    win = sliding_window_view(pad, (WIN, WIN)).reshape(tb.shape[0], tb.shape[1], WIN * WIN)
    n = np.isfinite(win).sum(-1)
    with np.errstate(all="ignore"):
        c = np.nanpercentile(win, 95, axis=-1)
    return np.where(n >= 41, c, np.nan)


def read_granule(path):
    with h5py.File(path) as h:
        missing = [k for k in (TB_NAME, LAT_NAME, LON_NAME, TIME_NAME) if k not in h]
        if missing:
            raise KeyError(f"{path.name}: datasets {missing} not found; file has {list(h.keys())}")
        tb = h[TB_NAME][:].astype("float64") * float(h[TB_NAME].attrs["SCALE FACTOR"][0]) if "SCALE FACTOR" in h[TB_NAME].attrs else h[TB_NAME][:].astype("float64") * 0.01
        lat89, lon89 = h[LAT_NAME][:].astype("float64"), h[LON_NAME][:].astype("float64")
        tsec = h[TIME_NAME][:].astype("float64")
    if lat89.shape[1] != 2 * tb.shape[1]:
        raise ValueError(f"{path.name}: 89A geolocation {lat89.shape} does not pair with Tb {tb.shape}")
    lat, lon = lat89[:, 1::2], lon89[:, 1::2]  # low-frequency footprints sit at the odd 89A pixels (spec)
    tb[tb <= 0] = np.nan
    return tb, lat, lon, tsec


# --- scenes covering the box: the day's and the neighbours', by file start time within [-12 h, +36 h)
t_lo = dt.datetime(day.year, day.month, day.day)
scenes = []
for d in (day - dt.timedelta(1), day, day + dt.timedelta(1)):
    scenes += [(d, *g) for g in granules_for(d)]
scenes = [sc for sc in scenes if t_lo - dt.timedelta(hours=12) <= sc[3] < t_lo + dt.timedelta(hours=36)]
print(f"{day}: {len(scenes)} L1B scenes in the span, {sum(sc[2] for sc in scenes) / 1e6:.0f} MB listed")
samples = []  # lat, lon, M, C, hours since 00 UTC, ascending
for d, name, size, t0, ad in scenes:
    # a scene lasts ~50 min; skip those whose ground track cannot reach the box: decided after reading the
    # geolocation of a cheap neighbour is not possible over SFTP, so use the start-time heuristic below
    path = fetch(d, name)
    tb, lat, lon, tsec = read_granule(path)
    inbox = (lat >= LAT0) & (lat < LAT1) & (lon >= LON0) & (lon < LON1)
    if not inbox.any():
        continue
    rows = np.where(inbox.any(1))[0]
    r0, r1 = max(rows.min() - H, 0), min(rows.max() + H + 1, tb.shape[0])
    c = window_stats(tb[r0:r1])
    sel = inbox[r0:r1] & np.isfinite(c) & np.isfinite(tb[r0:r1])
    hours = (np.array([EPOCH + dt.timedelta(seconds=float(x)) for x in tsec[r0:r1]]) - t_lo)
    hours = np.array([x.total_seconds() / 3600 for x in hours])
    asc = ad == "A"
    samples.append(np.column_stack([lat[r0:r1][sel], lon[r0:r1][sel], tb[r0:r1][sel], c[sel],
                                    np.broadcast_to(hours[:, None], sel.shape)[sel], np.full(int(sel.sum()), float(asc))]))
    print(f"  {name}: {int(sel.sum())} footprints in the box ({'asc' if asc else 'dsc'}, start {t0:%m-%d %H:%M} UTC)", flush=True)
fp = np.concatenate(samples)
lat, lon, M, C, hour, asc = fp.T
asc = asc > 0.5
np.savez_compressed(CACHE / f"footprints_{day:%Y%m%d}.npz", footprints=fp)

# --- grid: daytime (ascending) pass of the day first, night pass before it fills the rest; mean of samples
KM, cos10 = 111.2, np.cos(np.deg2rad(10.0))


def cell_mean(sel, val):
    rr = ((90 - lat[sel]) / RES).astype(int) - R0
    cc = ((lon[sel] + 180) / RES).astype(int) - C0
    ok = (rr >= 0) & (rr < NR) & (cc >= 0) & (cc < NC)
    s, n = np.zeros((NR, NC)), np.zeros((NR, NC))
    np.add.at(s, (rr[ok], cc[ok]), val[sel][ok]); np.add.at(n, (rr[ok], cc[ok]), 1)
    return np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)


asc_day = asc & (hour >= 12) & (hour < 24)
dsc_before = ~asc & (hour >= -12) & (hour < 12)
utc_day = (hour >= 0) & (hour < 24)
ours = {}
for label, val in (("m", M), ("c", C), ("s", M / C)):
    a, d = cell_mean(asc_day, val), cell_mean(dsc_before, val)
    ours[label] = np.where(np.isfinite(a), a, d)
    ours[label + "_utc"] = cell_mean(utc_day, val)


def crop(prefix):
    hits = sorted(KEY.glob(f"{prefix}{day:%Y%m%d}_HD5*.tif"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{KEY}: expected one {prefix}{day:%Y%m%d}_HD5*.tif, found {hits}")
    with rasterio.open(hits[0]) as ds:
        a = ds.read(1, window=Window(C0, R0, NC, NR)).astype("float64")
    a[(a == -32000) | (a == np.iinfo("int32").min)] = np.nan
    return a


with rasterio.open("data/gdacs_gfds/rebuild/ALL/oceanmask.tif") as ds:
    land = ds.read(1, window=Window(C0, R0, NC, NR)) == 0
jrc = {"m": crop("") * 0.01, "c": crop("calibration_") * 0.01, "s": crop("signal_") / 1e6}
jrc["s"][jrc["s"] <= 0] = np.nan


def report(name, a, b, tol):
    ok = np.isfinite(a) & np.isfinite(b) & land
    d = a[ok] - b[ok]
    print(f"{name:36s} n {int(ok.sum()):6d}  r {np.corrcoef(a[ok], b[ok])[0, 1]:.4f}  bias {d.mean():+.5f}  "
          f"mad {np.median(np.abs(d)):.5f}  within {tol:g}: {(np.abs(d) <= tol).mean():.3f}")


for label, tol in (("m", 0.05), ("c", 0.05), ("s", 1e-6)):
    report(f"{label}: JAXA, day pass first, vs JRC", ours[label], jrc[label], tol)
    report(f"{label}: JAXA, UTC-day mean, vs JRC", ours[label + "_utc"], jrc[label], tol)

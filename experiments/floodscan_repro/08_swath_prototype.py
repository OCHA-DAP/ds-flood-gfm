"""Swath-route prototype, one day: CSU AMSR2 FCDR orbit files -> 36 GHz Q on AER's 5-arcminute grid.

Why: FloodScan works on 22 km composite footprints regridded to 5' (ADD s. 3.3-3.5); our 0.25-deg grids
dilute narrow floodplains. The CSU bucket has AMSR2 swath files from 2012 (one per orbit, ~74 MB, each
variable one whole-file gzip chunk, so a range read per variable fetches only what is asked for).

Per orbit file of the day: read the spacecraft track (tiny) and skip orbits that never come near the box;
otherwise fetch lat/lon and 36 GHz V/H (4 chunks, a few MB each). Footprints in the box are split by
scan direction (ascending = spacecraft latitude increasing) and footprint-matched onto the 5' grid with a
Gaussian of FWHM 22 km (AER's Backus-Gilbert coefficients are not published), earliest swath winning where
two overlap (s. 3.5). Q = sqrt(PRI + 0.11) - 0.58 PRI. The 5' Q is aggregated 3x3 and compared with the
CSU 0.25-deg grid's Q for the same day and pass, as a check of the gridding.

    uv run --no-project --with h5py --with fsspec --with aiohttp --with numpy --with scipy \
        python experiments/floodscan_repro/08_swath_prototype.py 2024-09-01
"""

import datetime as dt
import http.client
import re
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import aiohttp
import fsspec
import h5py
import numpy as np
from scipy.spatial import cKDTree

BUCKET = "https://s3.us-east-1.amazonaws.com/noaa-cdr-microwave-imager-brit-temp-csu-pds"
LON0, LON1, LAT0, LAT1 = 0.0, 20.0, 0.0, 20.0
RES = 1 / 12  # 5 arcmin
NY, NX = int((LAT1 - LAT0) / RES), int((LON1 - LON0) / RES)
FWHM_KM = 22.0
SIGMA_KM = FWHM_KM / 2.355
GRID_SRC = Path("data/floodscan_repro/csu_amsr2_grid")
OUT = Path("data/floodscan_repro/swath_proto")
day = dt.date.fromisoformat(sys.argv[1])


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (OSError, http.client.HTTPException):
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))


def list_orbits(d):
    """Orbit files of the day: FCDR (final) where it exists, else ICDR (interim; 2024 is ICDR-only)."""
    for kind in ("FCDR", "ICDR"):
        xml = get(f"{BUCKET}/?list-type=2&prefix=data/{kind}/{d:%Y}/CSU_AMSR2_{kind}_V02R00_GCOMW1_D{d:%Y%m%d}&max-keys=100").decode()
        keys = re.findall(r"<Key>([^<]+)</Key>", xml)
        if keys:
            print(f"{d}: {kind} orbit files")
            return keys
    return []


def chunk_info(key, variables):
    url = f"{BUCKET}/{key}"
    opts = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=300, sock_read=120)}}
    info = {}
    with fsspec.open(url, "rb", block_size=2**16, cache_type="blockcache", **opts) as fo, h5py.File(fo) as h:
        for v in variables:
            ds = h[v]
            if ds.id.get_num_chunks() != 1:
                raise ValueError(f"{key}:{v}: {ds.id.get_num_chunks()} chunks, expected one")
            ci = ds.id.get_chunk_info(0)
            info[v] = (ci.byte_offset, ci.size, ds.dtype, ds.shape)
    return info


def read_var(key, off, size, dtype, shape):
    raw = get(f"{BUCKET}/{key}", {"Range": f"bytes={off}-{off + size - 1}"})
    if len(raw) != size:
        raise OSError(f"{key}: short read {len(raw)} of {size}")
    return np.frombuffer(zlib.decompress(raw), dtype=dtype).reshape(shape)


keys = list_orbits(day)
if not keys:
    raise FileNotFoundError(f"no AMSR2 orbit files listed for {day}")
print(f"{day}: {len(keys)} orbit files listed")
samples = {"asc": [], "dsc": []}  # (time, lat, lon, v, h) arrays per pass
mb = 0.0
for key in keys:
    info = chunk_info(key, ["spacecraft_lat", "spacecraft_lon", "scan_time", "lat_tb36", "lon_tb36", "fcdr_tb36v", "fcdr_tb36h"])
    sc_lat = read_var(key, *info["spacecraft_lat"]).astype(float)
    sc_lon = read_var(key, *info["spacecraft_lon"]).astype(float)
    sc_lon = np.where(sc_lon > 180, sc_lon - 360, sc_lon)
    mb += (info["spacecraft_lat"][1] + info["spacecraft_lon"][1]) / 1e6
    near = (sc_lat >= LAT0 - 8) & (sc_lat <= LAT1 + 8) & (sc_lon >= LON0 - 10) & (sc_lon <= LON1 + 10)
    if not near.any():
        continue
    lat = read_var(key, *info["lat_tb36"]).astype(float)
    lon = read_var(key, *info["lon_tb36"]).astype(float)
    lon = np.where(lon > 180, lon - 360, lon)
    tv = read_var(key, *info["fcdr_tb36v"]).astype(float)
    th = read_var(key, *info["fcdr_tb36h"]).astype(float)
    tsec = read_var(key, *info["scan_time"]).astype(float)
    mb += sum(info[v][1] for v in ("lat_tb36", "lon_tb36", "fcdr_tb36v", "fcdr_tb36h", "scan_time")) / 1e6
    inbox = (lat >= LAT0 - 0.5) & (lat <= LAT1 + 0.5) & (lon >= LON0 - 0.5) & (lon <= LON1 + 0.5) & (tv > 0) & (th > 0)
    if not inbox.any():
        continue
    dlat = np.gradient(sc_lat)
    asc_scan = dlat > 0
    for p, sel in (("asc", asc_scan), ("dsc", ~asc_scan)):
        m = inbox & sel[:, None]
        if m.any():
            tt = np.broadcast_to(tsec[:, None], lat.shape)[m]
            samples[p].append((tt, lat[m], lon[m], tv[m], th[m]))
            print(f"  {key.split('/')[-1]}: {p} {int(m.sum())} footprints in the box")
print(f"streamed {mb:.1f} MB")

# --- footprint matching onto the 5' grid, earliest swath wins where two overlap
glat = LAT1 - (np.arange(NY) + 0.5) * RES
glon = LON0 + (np.arange(NX) + 0.5) * RES
GY, GX = np.meshgrid(glat, glon, indexing="ij")
KM_LAT = 111.2
q5 = {}
for p in ("asc", "dsc"):
    if not samples[p]:
        print(f"{p}: no footprints")
        continue
    tt, lat, lon, tv, th = (np.concatenate([s[i] for s in samples[p]]) for i in range(5))
    coslat = np.cos(np.deg2rad(10.0))
    pts = np.column_stack([lat * KM_LAT, lon * KM_LAT * coslat])
    tree = cKDTree(pts)
    gpts = np.column_stack([GY.ravel() * KM_LAT, GX.ravel() * KM_LAT * coslat])
    groups = tree.query_ball_point(gpts, r=3 * SIGMA_KM)
    v_out, h_out = np.full(NY * NX, np.nan), np.full(NY * NX, np.nan)
    for gi, idx in enumerate(groups):
        if len(idx) < 3:
            continue
        idx = np.asarray(idx)
        # earliest swath: keep the footprints from the earliest hour-long cluster of times
        t0 = tt[idx].min()
        idx = idx[tt[idx] - t0 < 3600]
        d2 = ((pts[idx] - gpts[gi]) ** 2).sum(1)
        w = np.exp(-0.5 * d2 / SIGMA_KM ** 2)
        if w.sum() < 0.5:
            continue
        v_out[gi] = (w * tv[idx]).sum() / w.sum()
        h_out[gi] = (w * th[idx]).sum() / w.sum()
    pri = (v_out - h_out) / (v_out + h_out)
    with np.errstate(invalid="ignore"):
        q5[p] = (np.sqrt(pri + 0.11) - 0.58 * pri).reshape(NY, NX)
    print(f"{p}: {len(tt)} footprints -> 5' cells valid {np.isfinite(q5[p]).mean():.3f}")

# --- check against the CSU 0.25-deg grid for the same day
z = np.load(GRID_SRC / f"{day:%Y}" / f"{day:%Y%m%d}.npz")
for p, q in q5.items():
    v, h = np.flipud(z[f"fcdr_tb36v_{p}"]), np.flipud(z[f"fcdr_tb36h_{p}"])
    pri = (v - h) / (v + h)
    with np.errstate(invalid="ignore"):
        qg = np.sqrt(pri + 0.11) - 0.58 * pri
    q3 = np.nanmean(q.reshape(80, 3, 80, 3), (1, 3))
    ok = np.isfinite(q3) & np.isfinite(qg)
    r = np.corrcoef(q3[ok], qg[ok])[0, 1]
    print(f"{p}: 5'->0.25 vs CSU grid: n {int(ok.sum())}, r {r:.4f}, bias {np.mean(q3[ok] - qg[ok]):+.4f}, "
          f"mad {np.median(np.abs(q3[ok] - qg[ok])):.4f}; grid valid {np.isfinite(qg).mean():.3f}, ours {np.isfinite(q3).mean():.3f}")
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / f"q36_5min_{day:%Y%m%d}.npz", **{f"q_{p}": q for p, q in q5.items()}, lat=glat, lon=glon)
print(f"wrote {OUT / f'q36_5min_{day:%Y%m%d}.npz'}")

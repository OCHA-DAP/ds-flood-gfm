"""CSU AMSR2 orbit files -> footprint-matched brightness temperatures on AER's 5-arcminute grid (ADD s. 3.3-3.5).

One file per orbit (~74 MB; every variable a single whole-file gzip chunk, so one HTTP range request per
variable). Per day: list the orbits (FCDR, else ICDR), skip those whose track never nears the box, fetch
geolocation and the requested channels, split footprints by scan direction (ascending = spacecraft latitude
increasing), footprint-match onto the 5' grid with a Gaussian (FWHM 22 km for 36 GHz, 27 km for the flag
channels: AER's composite footprints; its Backus-Gilbert coefficients are unpublished), earliest swath
winning where two overlap. A day runs from 21:00 UTC the day before (DAY_START_UTC), as AER does (s. 3.5),
so one night's descending pass over the box is not split between two days.
"""

import datetime as dt
import http.client
import re
import time
import urllib.request
import zlib

import aiohttp
import fsspec
import h5py
import numpy as np
from scipy.spatial import cKDTree

BUCKET = "https://s3.us-east-1.amazonaws.com/noaa-cdr-microwave-imager-brit-temp-csu-pds"
LON0, LON1, LAT0, LAT1 = 0.0, 20.0, 0.0, 20.0
RES = 1 / 12
NY, NX = int((LAT1 - LAT0) / RES), int((LON1 - LON0) / RES)
GLAT = LAT1 - (np.arange(NY) + 0.5) * RES
GLON = LON0 + (np.arange(NX) + 0.5) * RES
KM_LAT = 111.2
COSLAT = np.cos(np.deg2rad(10.0))
FWHM = {"36": 22.0, "18": 27.0, "23": 27.0, "89": 27.0}
LOWRES = {"36": ("fcdr_tb36v", "fcdr_tb36h", "lat_tb36", "lon_tb36"),
          "18": ("fcdr_tb18v", "fcdr_tb18h", "lat_tb18", "lon_tb18"),
          "23": ("fcdr_tb23v", "fcdr_tb23v", "lat_tb23", "lon_tb23"),
          "89": ("fcdr_tb89v_ascan", "fcdr_tb89h_ascan", "lat_tb89_ascan", "lon_tb89_ascan")}


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


DAY_START_UTC = 21  # AER (s. 3.5): the day starts at 21:00 UTC the day before, so a night's descending pass is not split


def _orbits_dated(d):
    for kind in ("FCDR", "ICDR"):
        xml = get(f"{BUCKET}/?list-type=2&prefix=data/{kind}/{d:%Y}/CSU_AMSR2_{kind}_V02R00_GCOMW1_D{d:%Y%m%d}&max-keys=100").decode()
        keys = re.findall(r"<Key>([^<]+)</Key>", xml)
        if keys:
            return kind, keys
    return None, []


def _start_hour(key):
    return int(re.search(r"_S(\d{2})\d{2}_E", key).group(1))


def list_orbits(d):
    """Orbit files of AER's day d: those dated d-1 starting at or after DAY_START_UTC plus those dated d
    starting before it. The product kind (FCDR/ICDR) is whatever each date has."""
    kind_prev, prev = _orbits_dated(d - dt.timedelta(1))
    kind, cur = _orbits_dated(d)
    keys = [k for k in prev if _start_hour(k) >= DAY_START_UTC] + [k for k in cur if _start_hour(k) < DAY_START_UTC]
    return kind or kind_prev, keys


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
    return np.frombuffer(zlib.decompress(raw), dtype=dtype).reshape(shape).astype(float)


def day_footprints(d, bands=("36",)):
    """Footprint samples per pass and band: {pass: {band: (time, lat, lon, v, h)}}; also the MB streamed."""
    kind, keys = list_orbits(d)
    if not keys:
        raise FileNotFoundError(f"no AMSR2 orbit files listed for {d}")
    samples = {p: {b: [] for b in bands} for p in ("asc", "dsc")}
    mb = 0.0
    for key in keys:
        variables = ["spacecraft_lat", "spacecraft_lon", "scan_time"]
        info = chunk_info(key, variables + [v for b in bands for v in LOWRES[b]])
        sc_lat = read_var(key, *info["spacecraft_lat"])
        sc_lon = read_var(key, *info["spacecraft_lon"])
        sc_lon = np.where(sc_lon > 180, sc_lon - 360, sc_lon)
        mb += (info["spacecraft_lat"][1] + info["spacecraft_lon"][1]) / 1e6
        if not ((sc_lat >= LAT0 - 8) & (sc_lat <= LAT1 + 8) & (sc_lon >= LON0 - 10) & (sc_lon <= LON1 + 10)).any():
            continue
        tsec = read_var(key, *info["scan_time"])
        asc_scan = np.gradient(sc_lat) > 0
        mb += info["scan_time"][1] / 1e6
        for b in bands:
            vn, hn, latn, lonn = LOWRES[b]
            lat, lon = read_var(key, *info[latn]), read_var(key, *info[lonn])
            lon = np.where(lon > 180, lon - 360, lon)
            tv, th = read_var(key, *info[vn]), read_var(key, *info[hn])
            mb += sum(info[v][1] for v in {vn, hn, latn, lonn}) / 1e6
            inbox = (lat >= LAT0 - 0.5) & (lat <= LAT1 + 0.5) & (lon >= LON0 - 0.5) & (lon <= LON1 + 0.5) & (tv > 0) & (th > 0)
            if lat.shape[0] != asc_scan.shape[0]:
                raise ValueError(f"{key}:{b}: scan count differs from the spacecraft track")
            for p, sel in (("asc", asc_scan), ("dsc", ~asc_scan)):
                m = inbox & sel[:, None]
                if m.any():
                    tt = np.broadcast_to(tsec[:, None], lat.shape)[m]
                    samples[p][b].append((tt, lat[m], lon[m], tv[m], th[m]))
    return samples, mb


def footprint_match(tt, lat, lon, tv, th, fwhm_km):
    """Gaussian footprint matching onto the 5' grid; earliest swath wins. Returns (V, H) arrays (NY, NX)."""
    sigma = fwhm_km / 2.355
    pts = np.column_stack([lat * KM_LAT, lon * KM_LAT * COSLAT])
    tree = cKDTree(pts)
    gy, gx = np.meshgrid(GLAT, GLON, indexing="ij")
    gpts = np.column_stack([gy.ravel() * KM_LAT, gx.ravel() * KM_LAT * COSLAT])
    groups = tree.query_ball_point(gpts, r=3 * sigma)
    v_out, h_out = np.full(NY * NX, np.nan), np.full(NY * NX, np.nan)
    for gi, idx in enumerate(groups):
        if len(idx) < 3:
            continue
        idx = np.asarray(idx)
        t0 = tt[idx].min()
        idx = idx[tt[idx] - t0 < 3600]  # earliest swath (one orbit's visit is well under an hour)
        d2 = ((pts[idx] - gpts[gi]) ** 2).sum(1)
        w = np.exp(-0.5 * d2 / sigma ** 2)
        if w.sum() < 0.5:
            continue
        v_out[gi] = (w * tv[idx]).sum() / w.sum()
        h_out[gi] = (w * th[idx]).sum() / w.sum()
    return v_out.reshape(NY, NX), h_out.reshape(NY, NX)


def grid_day(d, bands=("36",)):
    """{pass: {band: (V, H)}} on the 5' grid for one day, plus MB streamed."""
    samples, mb = day_footprints(d, bands)
    out = {}
    for p in ("asc", "dsc"):
        out[p] = {}
        for b in bands:
            if not samples[p][b]:
                out[p][b] = (np.full((NY, NX), np.nan), np.full((NY, NX), np.nan))
                continue
            tt, lat, lon, tv, th = (np.concatenate([s[i] for s in samples[p][b]]) for i in range(5))
            out[p][b] = footprint_match(tt, lat, lon, tv, th, FWHM[b])
    return out, mb


def q_of(v, h):
    pri = (v - h) / (v + h)
    with np.errstate(invalid="ignore"):
        return np.sqrt(pri + 0.11) - 0.58 * pri

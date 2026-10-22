"""Fetch AMSR2 brightness temperatures from NOAA's CSU microwave CDR (V02R00) on AWS, region subset only.

Source: public S3 bucket noaa-cdr-microwave-imager-brit-temp-csu-pds (us-east-1, anonymous HTTPS).
Daily 0.25-deg grids, one file per day with every channel and pass:
  data/FCDR-GRID/YYYY/CSU_AMSR2_FCDR-GRID_V02R00_GCOMW1_DYYYYMMDD.nc  (final; 2023, 2025)
  data/ICDR-GRID/YYYY/CSU_AMSR2_ICDR-GRID_V02R00_GCOMW1_DYYYYMMDD.nc  (interim; 2024, 2026, part of 2023)
FCDR is used where it exists, else ICDR; the choice is recorded per day. Calibration: CSU's absolute
calibration to GPM GMI, not JAXA's, so AER's AMSR2->AMSR-E constants do not apply as published.

Each variable is a single zlib chunk (about 1.5 MB compressed), so HTTP range reads fetch only the
channels asked for. Only the region box is kept: data/floodscan_repro/csu_amsr2_grid/YYYY/YYYYMMDD.npz.
Resumable: a day whose file already holds every requested variable is skipped; a day fetched earlier
with fewer bands gets the missing ones added from the same upstream file (a changed upstream key
raises). A day with no file in either product is recorded in missing.txt (listed absence); any read
error raises.

    uv run --no-project --with h5py --with fsspec --with aiohttp --with numpy \
        python experiments/floodscan_repro/03_fetch_csu_amsr2_grid.py 2023-06-01 2023-12-31 36,18,23,89
"""

import datetime as dt
import http.client
import re
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
import fsspec
import fsspec.exceptions
import h5py
import numpy as np

# Path-style regional URL: the virtual-hosted name (<bucket>.s3.amazonaws.com) stopped resolving on
# this machine mid-session on 2026-10-16 while s3.us-east-1.amazonaws.com kept working.
BUCKET = "https://s3.us-east-1.amazonaws.com/noaa-cdr-microwave-imager-brit-temp-csu-pds"
OUT = Path("data/floodscan_repro/csu_amsr2_grid")
LON0, LON1, LAT0, LAT1 = 0.0, 20.0, 0.0, 20.0  # West Africa box (cell edges)
# grid: lat centres -89.875..89.875 ascending, lon centres 0.125..359.875
R0, R1 = int((LAT0 + 90) / 0.25), int((LAT1 + 90) / 0.25)
C0, C1 = int(LON0 / 0.25), int(LON1 / 0.25)
CHANNELS = {"36": ("36v", "36h"), "18": ("18v", "18h"), "23": ("23v",), "89": ("89va", "89ha")}

start, end = (dt.date.fromisoformat(a) for a in sys.argv[1:3])
bands = sys.argv[3].split(",") if len(sys.argv) > 3 else ["36", "18"]
VARS = [f"fcdr_tb{c}_{p}" for b in bands for c in CHANNELS[b] for p in ("asc", "dsc")]


def list_keys(kind, year):
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix=data/{kind}/{year}/CSU_AMSR2&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token)
        for attempt in range(5):  # a reset on the listing killed the 2024 run at start (2026-10-21)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    xml = r.read().decode()
                break
            except (OSError, http.client.HTTPException):
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def day_of(key):
    return dt.datetime.strptime(re.search(r"_D(\d{8})\.nc$", key).group(1), "%Y%m%d").date()


index = {}
for year in range(start.year, end.year + 1):
    for kind in ("ICDR-GRID", "FCDR-GRID"):  # FCDR second so it overrides ICDR
        for k in list_keys(kind, year):
            index[day_of(k)] = (kind, k)


def read_day(key, variables):
    """Header through a small block cache, then one exact range request per variable chunk.

    Reading the data through h5py with a single-block read-ahead cache re-downloaded 2 MB blocks on
    every scattered read: on a slow link 4 parallel days pulled > 130 MB without finishing (2026-10-21).
    Explicit timeouts: without them a dropped connection hung the fetch with no error.
    """
    url = f"{BUCKET}/{key}"
    opts = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=300, sock_read=120)}}
    info = {}
    with fsspec.open(url, "rb", block_size=2**16, cache_type="blockcache", **opts) as fo, h5py.File(fo) as h:
        lat, lon = h["lat"][:], h["lon"][:]
        if not (np.isclose(lat[R0], LAT0 + 0.125) and np.isclose(lon[C0], LON0 + 0.125)):
            raise ValueError(f"{key}: unexpected grid origin lat {lat[R0]} lon {lon[C0]}")
        for v in variables:
            ds = h[v]
            if ds.id.get_num_chunks() != 1 or tuple(ds.chunks) != ds.shape or set(ds._filters) != {"gzip"}:
                raise ValueError(f"{key}:{v}: expected one zlib chunk, got chunks {ds.chunks}, filters {ds._filters}")
            ci = ds.id.get_chunk_info(0)
            info[v] = (ci.byte_offset, ci.size, ds.dtype, ds.shape)
    arrays = {}
    for v, (off, size, dtype, shape) in info.items():
        req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + size - 1}"})
        for attempt in range(5):  # per-chunk retry: resets hit ~1 in 3 requests on a flaky link (2026-10-21)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    raw = r.read()
                if len(raw) != size:
                    raise OSError(f"{key}:{v}: short read {len(raw)} of {size} bytes")
                break
            except (OSError, http.client.HTTPException):
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        a = np.frombuffer(zlib.decompress(raw), dtype=dtype).reshape(shape)[R0:R1, C0:C1].astype("float32")
        a[a < 0] = np.nan  # _FillValue -9999.9
        arrays[v] = a
    return arrays


def fetch(day):
    out = OUT / f"{day:%Y}" / f"{day:%Y%m%d}.npz"
    have = {}
    if out.exists():
        with np.load(out) as z:
            have = {k: z[k] for k in z.files}
        if all(v in have for v in VARS):
            return "skipped"
        # a day fetched earlier with fewer bands: add the missing ones, keep what is there
    if day not in index:
        if have:
            raise RuntimeError(f"{day}: have a local file but the day is no longer listed upstream")
        with open(OUT / "missing.txt", "a") as fh:
            fh.write(f"{day} not listed in FCDR-GRID or ICDR-GRID\n")
        return "absent"
    kind, key = index[day]
    if have and str(have["key"]) != key:
        raise RuntimeError(f"{day}: local file came from {have['key']}, upstream now lists {key}; delete the local file to refetch")
    need = [v for v in VARS if v not in have]
    for attempt in range(3):  # network drops only; a bad grid or file still raises at once
        try:
            arrays = read_day(key, need)
            break
        except (OSError, TimeoutError, http.client.HTTPException, aiohttp.ClientError, fsspec.exceptions.FSTimeoutError) as e:
            if attempt == 2:
                raise
            print(f"  {day}: network error ({type(e).__name__}); retrying in 30 s", flush=True)
            time.sleep(30)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.npz")
    have.update(arrays)
    np.savez_compressed(tmp, source=np.array(kind), key=np.array(key), **{k: v for k, v in have.items() if k not in ("source", "key")})
    tmp.rename(out)
    return kind if len(need) == len(VARS) else "extended"


OUT.mkdir(parents=True, exist_ok=True)
days = [start + dt.timedelta(n) for n in range((end - start).days + 1)]
with ThreadPoolExecutor(4) as ex:
    results = list(ex.map(fetch, days))
summary = {k: results.count(k) for k in sorted(set(results))}
print(f"{start}..{end} bands {bands}: {summary}")

"""Fetch GFDS daily signal+mag windows over Nigeria, Jun-Dec 2022, plus static baselines.

Resumable: per-day .npy cache; 404 recorded as .missing marker; any other
error raises (fail loudly). Throttled ~1 req/s.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import os
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "2097152")
os.environ.setdefault("GDAL_INGESTED_BYTES_AT_OPEN", "65536")
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import from_bounds

BBOX = (2.7, 4.0, 14.7, 14.0)  # Nigeria
CACHE = Path("data/gdacs_gfds/nga2022")
CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://www.gdacs.org/flooddetection/DATA"
KINDS = {"signal": ("SignalTiffs", "signal"), "mag": ("MagTiffs", "mag_signal")}


def read_window(url, attempts=3):
    last = None
    for i in range(attempts):
        try:
            return _read_window_once(url)
        except RasterioIOError as e:
            msg = str(e)
            if "404" in msg or "does not exist" in msg or "not recognized" in msg:
                raise  # a 404 is an answer, not a failure - no retry
            last = e
            wait = 15 * (i + 1)
            print(f"retry {i+1}/{attempts} after error on {url}: {e} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise last


def _read_window_once(url):
    with rasterio.open(f"/vsicurl/{url}") as src:
        win = from_bounds(*BBOX, src.transform)
        arr = src.read(1, window=win)
        transform = src.window_transform(win)
    return arr, transform


# static baselines first (one-off)
for which in ("avg", "sd"):
    out = CACHE / f"baseline_{which}.npy"
    if not out.exists():
        arr, transform = read_window(f"{BASE}/ALL/AveragesAndSd/bt_signal_{which}.tif")
        np.save(out, arr)
        np.save(CACHE / "geotransform.npy", np.array(transform)[:6])
        print(f"baseline {which}: {arr.shape}", flush=True)
        time.sleep(1)

priority = pd.date_range("2022-08-15", "2022-11-30", freq="D")
rest = [d for d in pd.date_range("2022-06-01", "2022-12-31", freq="D")
        if d not in set(priority)]
n_fetched = 0
# event window first, both kinds; then the shoulders, signal only
work = [("signal", priority), ("mag", priority), ("signal", rest)]
for kind, dates in work:
    folder, prefix = KINDS[kind]
    for ts in dates:
        d = ts.date()
        npy = CACHE / f"{kind}_{d:%Y%m%d}.npy"
        miss = CACHE / f"{kind}_{d:%Y%m%d}.missing"
        if npy.exists() or miss.exists():
            continue
        url = f"{BASE}/ALL/{folder}/{d:%Y}/{d:%m}/{prefix}_{d:%Y%m%d}_ALL.tif"
        try:
            arr, _ = read_window(url)
        except RasterioIOError as e:
            msg = str(e)
            if "404" in msg or "does not exist" in msg or "not recognized" in msg:
                miss.touch()
                print(f"MISSING upstream: {kind} {d}", flush=True)
            else:
                print(f"FETCH ERROR (not a 404) on {url}: {e}", file=sys.stderr, flush=True)
                raise
        else:
            np.save(npy, arr)
            n_fetched += 1
            if n_fetched % 25 == 0:
                print(f"{n_fetched} files fetched (latest {kind} {d})", flush=True)
        time.sleep(0.5)

n_sig = len(list(CACHE.glob("signal_*.npy")))
n_mag = len(list(CACHE.glob("mag_*.npy")))
n_miss = len(list(CACHE.glob("*.missing")))
print(f"DONE: {n_sig} signal days, {n_mag} mag days, {n_miss} missing markers", flush=True)

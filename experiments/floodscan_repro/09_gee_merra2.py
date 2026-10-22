"""Atmospheric inputs for the earth-leaving Tb correction (ADD s. 3.6): MERRA-2 hourly single-level fields
from Earth Engine, sampled at the AMSR2 overpass hours on the 0.25-deg West Africa box.

AER drives its radiative-transfer model with NCEP/GFS profiles. Here (atmos.py) the RSS absorption
model (Wentz & Meissner 2000) needs only columnar water vapour V (mm) and an effective air temperature,
so MERRA-2 M2T1NXSLV suffices: TQV (kg m-2 = mm), T2M (K), PS (Pa). GEE asset NASA/GSFC/MERRA/slv/2,
hourly means stamped at HH:30. Overpass: descending ~01:30 local = the 00:30 UTC hour over 0-20 E,
ascending ~13:30 local = the 12:30 UTC hour.

Output: data/floodscan_repro/merra2/YYYYMM.npz with arrays (ndays, 80, 80) per field and pass, north-up.
One download per month (a stacked image), cached; a missing hour raises.

    uv run python experiments/floodscan_repro/09_gee_merra2.py 2023 2024 2025
"""

import calendar
import datetime as dt
import http.client
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import ee
import numpy as np
import rasterio

PROJECT = "ee-zackarno"
OUT = Path("data/floodscan_repro/merra2")
FIELDS = ("TQV", "T2M", "PS")
HOURS = {"dsc": 0, "asc": 12}  # UTC hour of the MERRA-2 mean (stamped HH:30) nearest the overpass
X0, Y1, N = 0.0, 20.0, 80
GRID = {"crs": "EPSG:4326", "crs_transform": [0.25, 0, X0, 0, -0.25, Y1]}

ee.Initialize(project=PROJECT)
REGION = ee.Geometry.Rectangle([X0, Y1 - 0.25 * N, X0 + 0.25 * N, Y1], None, False)
col = ee.ImageCollection("NASA/GSFC/MERRA/slv/2").select(list(FIELDS))


def month_stack(year, month):
    ndays = calendar.monthrange(year, month)[1]
    imgs, names = [], []
    for d in range(1, ndays + 1):
        for p, h in HOURS.items():
            t0 = dt.datetime(year, month, d, h)
            img = col.filterDate(t0.isoformat(), (t0 + dt.timedelta(hours=1)).isoformat()).first()
            imgs.append(ee.Image(img))
            names += [f"{f}_{p}_{d:02d}" for f in FIELDS]
    stack = ee.ImageCollection(imgs).toBands().rename(names)
    # fail loudly on a missing hour: toBands on a null image raises server-side with a clear message
    return stack, names


def download(stack):
    url = stack.getDownloadURL({"region": REGION, "format": "GEO_TIFF", **GRID})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=900) as r:
                return r.read()
        except (http.client.RemoteDisconnected, urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                raise
            print(f"  download dropped ({e!r}); retrying", flush=True)
            time.sleep(30)


OUT.mkdir(parents=True, exist_ok=True)
for year in (int(a) for a in sys.argv[1:]):
    for month in range(6, 13):
        out = OUT / f"{year}{month:02d}.npz"
        if out.exists():
            continue
        stack, names = month_stack(year, month)
        raw = download(stack)
        with rasterio.open(io.BytesIO(raw)) as ds:
            if (ds.height, ds.width, ds.count) != (N, N, len(names)):
                raise ValueError(f"{year}-{month:02d}: unexpected export {(ds.height, ds.width, ds.count)} vs {(N, N, len(names))}")
            a = ds.read().astype("float32")
        if not np.isfinite(a).all():
            raise ValueError(f"{year}-{month:02d}: non-finite MERRA-2 values in the export")
        ndays = calendar.monthrange(year, month)[1]
        arrays = {}
        for f in FIELDS:
            for p in HOURS:
                idx = [names.index(f"{f}_{p}_{d:02d}") for d in range(1, ndays + 1)]
                arrays[f"{f}_{p}"] = a[idx]
        tmp = out.with_suffix(".part.npz")
        np.savez_compressed(tmp, days=np.array([dt.date(year, month, d).isoformat() for d in range(1, ndays + 1)]), **arrays)
        tmp.rename(out)
        print(f"{year}-{month:02d}: TQV asc median {np.median(arrays['TQV_asc']):.1f} mm, T2M asc median "
              f"{np.median(arrays['T2M_asc']):.1f} K, PS median {np.median(arrays['PS_asc']) / 100:.0f} hPa", flush=True)
print("done")

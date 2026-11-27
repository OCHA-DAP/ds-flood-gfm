"""One CSU AMSR2 swath FCDR file: size, variables, chunking, and whether a range read can take only the
scans crossing our box (0-20E, 0-20N)."""
import urllib.request

import aiohttp
import fsspec
import h5py
import numpy as np

url = ("https://s3.us-east-1.amazonaws.com/noaa-cdr-microwave-imager-brit-temp-csu-pds/"
       "data/FCDR/2016/CSU_AMSR2_FCDR_V02R00_GCOMW1_D20160101_S0003_E0142_R019274.nc")
req = urllib.request.Request(url, method="HEAD")
with urllib.request.urlopen(req, timeout=60) as r:
    print("size MB", int(r.headers["Content-Length"]) / 1e6)
opts = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=300, sock_read=120)}}
with fsspec.open(url, "rb", block_size=2**16, cache_type="blockcache", **opts) as fo, h5py.File(fo) as h:
    print("attrs:", {k: (str(v)[:80]) for k, v in h.attrs.items() if k in ("title", "summary", "spatial_resolution", "geospatial_lat_resolution")})
    for k in h:
        ds = h[k]
        print(f"{k:28s} {ds.shape} {ds.dtype} chunks={ds.chunks} filters={ds._filters} "
              f"{dict(ds.attrs).get('long_name', b'')!s:.70}")
    lat_name = [k for k in h if k.startswith("lat")][0]
    lat = h[lat_name][:]
    lon = h[[k for k in h if k.startswith("lon")][0]][:]
    print(lat_name, lat.shape)
    inbox = (lat >= 0) & (lat <= 20) & (lon >= 0) & (lon <= 20)
    rows = np.where(inbox.any(axis=1))[0]
    print("scan rows in box:", len(rows), "of", lat.shape[0], "range", (rows.min(), rows.max()) if len(rows) else None)

"""List the variables in one CSU AMSR2 gridded file (header only)."""
import aiohttp
import fsspec
import h5py

url = ("https://s3.us-east-1.amazonaws.com/noaa-cdr-microwave-imager-brit-temp-csu-pds/"
       "data/ICDR-GRID/2024/CSU_AMSR2_ICDR-GRID_V02R00_GCOMW1_D20240901.nc")
opts = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=300, sock_read=120)}}
with fsspec.open(url, "rb", block_size=2**16, cache_type="blockcache", **opts) as fo, h5py.File(fo) as h:
    for k in h:
        ds = h[k]
        print(k, ds.shape, ds.dtype, dict(ds.attrs).get("long_name", b""))

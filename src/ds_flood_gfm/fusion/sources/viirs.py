"""NOAA/GMU VIIRS flood map (v2r0, 1-day global composite) on the 0.01-degree grid.

Archive: s3://noaa-jpss/JPSS_Blended_Products/VFM_1day_GLB/TIF/YYYY/MM/DD/
(public, no credentials; 2012-2020 and Aug 2023 onward; nothing for 2021-2022).
136 AOI tiles (GLB001..GLB136), 15 x 15 degrees, 375 m, uint8, one-row strips.

Legend (from each file's WaterDetection:TypeDescription):
  1 no valid data | 15 floodwater without fraction | 16 bare land | 17 vegetation
  20 snow/ice | 27 river/lake ice | 30 cloud | 38 supra-snow/ice water | 50 shadow
  99 normal water | 100-200 floodwater fraction (value - 100 percent)

Mapping to the value rule (fixed decision 1):
  NoData  : 1, 20, 27, 30, 38, 50 (no data, snow/ice, cloud, shadow masks)
  observed: 15, 16, 17, 99, 100-200; flood fraction 1.0 for 15, (v-100)/100 for 100-200, 0 otherwise
A cell is observed iff observed pixels cover >= observed_min_frac of it, then
flood iff mean flood fraction over observed pixels >= flood_threshold.
Several files for one AOI and day (different overpass spans) are pooled.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import Config
from ds_flood_gfm.fusion.sources import DRY, FLOOD, NODATA

BUCKET = "https://noaa-jpss.s3.amazonaws.com"
PREFIX = "JPSS_Blended_Products/VFM_1day_GLB/TIF"
NODATA_CODES = (1, 20, 27, 30, 38, 50)
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIRANGE="YES")
AOI_RE = re.compile(r"VIIRS-Flood-1day-(GLB\d{3})_")


def _list_keys(prefix: str) -> list[str]:
    keys, token = [], None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token, safe="")
        from ds_flood_gfm.fusion.net import retry

        def _get(u=url):
            with urllib.request.urlopen(u, timeout=120) as r:
                return r.read()

        root = ET.fromstring(retry(_get, f"VIIRS listing {prefix}"))
        keys += [c.find("s3:Key", ns).text for c in root.findall("s3:Contents", ns)]
        nxt = root.find("s3:NextContinuationToken", ns)
        if nxt is None:
            return keys
        token = nxt.text


def flood_fraction(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(observed mask, flood fraction in [0,1]) per VIIRS pixel."""
    observed = ~np.isin(codes, NODATA_CODES)
    frac = np.zeros(codes.shape, "float32")
    frac[codes == 15] = 1.0
    w = (codes >= 100) & (codes <= 200)
    frac[w] = (codes[w].astype("float32") - 100.0) / 100.0
    return observed, frac


class VIIRSSource:
    name = "viirs"

    def __init__(self, cfg: Config, threads: int = 6) -> None:
        self.cfg = cfg
        self.s = cfg["sources"]["viirs"]
        self.threads = threads
        self._aoi = self._aoi_bounds()

    def _aoi_bounds(self) -> dict[str, tuple[float, float, float, float]]:
        path = self.cfg.local("raw", "tile_index", "viirs_aoi_bounds.json")
        if path.exists():
            return {k: tuple(v) for k, v in json.loads(path.read_text()).items()}
        ref = self.s["aoi_reference_day"]
        keys = _list_keys(f"{PREFIX}/{ref[:4]}/{ref[5:7]}/{ref[8:10]}/")
        first: dict[str, str] = {}
        for k in keys:
            m = AOI_RE.search(k)
            if m and m.group(1) not in first:
                first[m.group(1)] = k
        out = {}
        with rasterio.Env(**GDAL_ENV):
            for aoi, k in sorted(first.items()):
                with rasterio.open(f"/vsicurl/{BUCKET}/{k}") as ds:
                    b = ds.bounds
                    out[aoi] = (b.left, b.bottom, b.right, b.top)
        if len(out) < 100:
            raise RuntimeError(f"only {len(out)} VIIRS AOI tiles found on reference day {ref}")
        path.write_text(json.dumps(out))
        return out

    def day_keys(self, day: dt.date) -> dict[str, list[str]]:
        """AOI -> file keys for one day (cached listing). Empty dict = no files that day (upstream absence)."""
        path = self.cfg.local("raw", "viirs_listing", f"{day:%Y%m%d}.json")
        if path.exists():
            return json.loads(path.read_text())
        keys = _list_keys(f"{PREFIX}/{day:%Y/%m/%d}/")
        out: dict[str, list[str]] = {}
        for k in keys:
            m = AOI_RE.search(k)
            if m and k.endswith(".tif"):
                out.setdefault(m.group(1), []).append(k)
        path.write_text(json.dumps(out))
        return out

    def _read(self, key: str, win: grid.Window):
        minx, miny, maxx, maxy = win.bounds
        with rasterio.Env(**GDAL_ENV), rasterio.open(f"/vsicurl/{BUCKET}/{key}") as ds:
            b = ds.bounds
            x0, y0, x1, y1 = max(minx, b.left), max(miny, b.bottom), min(maxx, b.right), min(maxy, b.top)
            if x0 >= x1 or y0 >= y1:
                return None
            rw = from_bounds(x0, y0, x1, y1, ds.transform).round_offsets().round_lengths()
            if rw.width <= 0 or rw.height <= 0:
                return None
            a = ds.read(1, window=rw)
            t = ds.window_transform(rw)
        lon = t.c + (np.arange(a.shape[1]) + 0.5) * t.a
        lat = t.f + (np.arange(a.shape[0]) + 0.5) * t.e
        rows = grid.row_of(lat) - win.row0
        cols = grid.col_of(lon) - win.col0
        rin = (rows >= 0) & (rows < win.shape[0])
        cin = (cols >= 0) & (cols < win.shape[1])
        a = a[np.ix_(rin, cin)]
        rr, cc = np.meshgrid(rows[rin], cols[cin], indexing="ij")
        obs, frac = flood_fraction(a)
        n_all = grid.bincount_2d(rr.ravel(), cc.ravel(), win.shape)
        n_obs = grid.bincount_2d(rr[obs], cc[obs], win.shape)
        f_sum = grid.bincount_2d(rr[obs], cc[obs], win.shape, weights=frac[obs])
        return n_all, n_obs, f_sum

    def observe_day(self, win: grid.Window, day: dt.date) -> tuple[np.ndarray, np.ndarray]:
        out = np.full(win.shape, NODATA, "uint8")
        frac = np.full(win.shape, np.nan, "float32")
        listing = self.day_keys(day)
        if not listing:
            return out, frac
        minx, miny, maxx, maxy = win.bounds
        keys = [k for aoi, ks in listing.items() if aoi in self._aoi
                for k in ks
                if self._aoi[aoi][0] < maxx and minx < self._aoi[aoi][2] and self._aoi[aoi][1] < maxy and miny < self._aoi[aoi][3]]
        # pool only the files that observe the cell (a cloudy second overpass must not
        # dilute a clear first one)
        n_obs = np.zeros(win.shape)
        f_sum = np.zeros(win.shape)
        with ThreadPoolExecutor(self.threads) as ex:
            from ds_flood_gfm.fusion.net import retry

            for res in ex.map(lambda k: retry(lambda: self._read(k, win), f"VIIRS {k}"), keys):
                if res is None:
                    continue
                a_i, o_i, f_i = res
                with np.errstate(invalid="ignore", divide="ignore"):
                    ok_i = np.where(a_i > 0, o_i / a_i, 0.0) >= self.s["observed_min_frac"]
                n_obs[ok_i] += o_i[ok_i]
                f_sum[ok_i] += f_i[ok_i]
        observed = n_obs > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(observed, f_sum / n_obs, np.nan)
        frac[observed] = f[observed]
        out[observed] = DRY
        out[observed & (f >= self.s["flood_threshold"])] = FLOOD
        return out, frac

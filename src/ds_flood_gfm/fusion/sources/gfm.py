"""Copernicus GFM (Sentinel-1) on the 0.01-degree grid.

Per acquisition day, every GFM scene intersecting the window is read at COG
overview level 1 (80 m; validated against full resolution in D-08). Each 80 m
pixel centre is assigned to its 0.01-degree cell:

    observed pixel = ensemble_flood_extent != 255 and exclusion_mask == 0
                     and advisory_flags not in {1, 3}
    flood pixel    = observed and ensemble_flood_extent == 1

advisory_flags 1/3 mark "low regional backscatter: snow-covered, frozen or dry
soil" (GFM PUM); the exclusion mask does not cover snow, so this flag is GFM's
own snow/frozen-ground mask and is treated as NoData (fixed decision 1, D-17).

A cell is observed iff observed pixels cover >= observed_min_frac of the cell
(expected pixel count from cell area); then it is flood iff
flood / observed >= flood_threshold. Excluded pixels read 0 in the flood
layer, so they are dropped explicitly rather than counted as dry.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pystac_client
import rasterio
import shapely
from pyproj import Transformer
from rasterio.errors import RasterioIOError
from rasterio.windows import from_bounds

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import Config
from ds_flood_gfm.fusion.sources import DRY, FLOOD, NODATA

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="5",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
)


class _UnusableScene(Exception):
    """A scene whose flood or exclusion layer is not listed in STAC, or is an empty file upstream."""


def _empty_remote(href: str) -> bool:
    """True if the server reports a zero-length file (seen in the EODC archive: an empty
    advisory file with a 1970 timestamp). Network errors propagate."""
    import urllib.request

    req = urllib.request.Request(href, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.headers.get("Content-Length") == "0"


@lru_cache(maxsize=64)
def _transformer(crs_wkt: str) -> Transformer:
    return Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)


class GFMSource:
    name = "gfm"

    def __init__(self, cfg: Config, threads: int = 4) -> None:
        self.s = cfg["sources"]["gfm"]
        self.threads = threads
        self._client = None
        self._index_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # scenes dropped because one of their asset files is absent upstream (HTTP 404)
        self.dropped_scenes: list[str] = []
        self.scenes_without_advisory: list[str] = []
        if self.s["on_missing_asset"] not in ("drop_scene", "raise"):
            raise ValueError(f"sources.gfm.on_missing_asset must be drop_scene or raise, got {self.s['on_missing_asset']!r}")

    @property
    def client(self):
        if self._client is None:
            self._client = pystac_client.Client.open(self.s["stac_url"])
        return self._client

    def search(self, win: grid.Window, start: dt.date, end: dt.date) -> dict[dt.date, list]:
        """GFM items intersecting `win` between start and end (inclusive), grouped by UTC acquisition day.

        EODC's STAC returns items far outside the query bbox (reprex/), so each
        item's own bbox is re-checked against the window.
        """
        minx, miny, maxx, maxy = win.bounds
        box = shapely.box(minx, miny, maxx, maxy)
        search = self.client.search(
            collections=[self.s["collection"]],
            bbox=[minx, miny, maxx, maxy],
            datetime=f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        )
        from ds_flood_gfm.fusion.net import retry

        items = retry(lambda: list(search.items()), f"GFM STAC search {win.key()} {start}..{end}")
        out: dict[dt.date, list] = defaultdict(list)
        for it in items:
            if it.bbox is None or not shapely.box(*it.bbox).intersects(box):
                continue
            out[it.datetime.astimezone(dt.UTC).date()].append(it)
        return dict(out)

    def _cell_index(self, crs_wkt: str, transform, shape: tuple[int, int], win: grid.Window):
        key = (crs_wkt, tuple(transform)[:6], shape, win.key())
        hit = self._index_cache.get(key)
        if hit is not None:
            return hit
        h, w = shape
        xs = transform.c + (np.arange(w) + 0.5) * transform.a
        ys = transform.f + (np.arange(h) + 0.5) * transform.e
        X, Y = np.meshgrid(xs, ys)
        lon, lat = _transformer(crs_wkt).transform(X, Y)
        r, c, inside = grid.cell_index(lon, lat, win)
        hit = (r, c, inside)
        if len(self._index_cache) > 256:
            self._index_cache.clear()
        self._index_cache[key] = hit
        return hit

    def _read_item(self, item, win: grid.Window) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Per-cell (observed_count, flood_count, pixel size m) from one scene, or None if no overlap."""
        missing = [a for a in (self.s["flood_asset"], self.s["exclusion_asset"]) if a not in item.assets]
        if missing:
            # without the exclusion mask, excluded pixels (which read 0) cannot be told from dry:
            # the scene cannot be used without inventing dry observations (D-29)
            raise _UnusableScene(f"{item.id} lists no {missing}")
        flood_href = item.assets[self.s["flood_asset"]].href
        excl_href = item.assets[self.s["exclusion_asset"]].href
        adv_asset = item.assets.get(self.s["advisory_asset"])
        if adv_asset is None:
            # this product version has no advisory layer: there is no GFM snow/frozen flag
            # to apply for this scene (recorded, D-17); 404s on listed files are handled apart
            if self.s["on_unlisted_advisory"] != "read_without_flag":
                raise KeyError(f"{item.id} has no {self.s['advisory_asset']} asset")
            self.scenes_without_advisory.append(item.id)
        adv_href = adv_asset.href if adv_asset is not None else None
        lvl = self.s["overview_level"]
        minx, miny, maxx, maxy = win.bounds
        ring = shapely.segmentize(shapely.box(minx, miny, maxx, maxy), 0.05)
        with rasterio.Env(**GDAL_ENV):
            try:
                fds_ctx = rasterio.open(flood_href, overview_level=lvl)
            except RasterioIOError as e:
                if "not recognized" in str(e) and _empty_remote(flood_href):
                    raise _UnusableScene(f"{item.id}: empty flood file upstream") from e
                raise
            with fds_ctx as fds:
                crs_wkt = fds.crs.to_wkt()
                tr = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)
                xx, yy = tr.transform(*np.asarray(ring.exterior.coords).T)
                b = fds.bounds
                x0, y0, x1, y1 = max(xx.min(), b.left), max(yy.min(), b.bottom), min(xx.max(), b.right), min(yy.max(), b.top)
                if x0 >= x1 or y0 >= y1:
                    return None
                rw = from_bounds(x0, y0, x1, y1, fds.transform).round_offsets().round_lengths()
                if rw.width <= 0 or rw.height <= 0:
                    return None
                flood = fds.read(1, window=rw)
                wt = fds.window_transform(rw)
            # exclusion / advisory files do not always carry overviews (newer advisory
            # layers have none), so read them at full resolution over the same bounds,
            # decimated (nearest) onto the flood overview grid.
            wb = rasterio.windows.bounds(rw, fds.transform)
            try:
                excl = self._read_on_grid(excl_href, wb, flood.shape)
            except RasterioIOError as e:
                if "not recognized" in str(e) and _empty_remote(excl_href):
                    raise _UnusableScene(f"{item.id}: empty exclusion file upstream") from e
                raise
            adv = None
            if adv_href:
                try:
                    adv = self._read_on_grid(adv_href, wb, flood.shape)
                except RasterioIOError as e:
                    # older archive scenes list an advisory file that is not there (404) or is empty
                    # (0 bytes); flood and exclusion layers are fine, so read the scene without the
                    # flag and record it (D-29)
                    missing = "HTTP response code: 404" in str(e) or ("not recognized" in str(e) and _empty_remote(adv_href))
                    if not missing or self.s["on_unlisted_advisory"] != "read_without_flag":
                        raise
                    self.scenes_without_advisory.append(item.id)
            if adv is None:
                adv = np.zeros_like(flood)
        if not (flood.shape == excl.shape == adv.shape):
            raise RuntimeError(f"GFM asset grids differ for {item.id}: {flood.shape} {excl.shape} {adv.shape}")
        r, c, inside = self._cell_index(crs_wkt, wt, flood.shape, win)
        snow_frozen = np.isin(adv, self.s["advisory_nodata_values"])
        observed = (flood != self.s["nodata"]) & (excl == 0) & ~snow_frozen & inside
        is_flood = observed & (flood == 1)
        obs = grid.bincount_2d(r[observed], c[observed], win.shape)
        fl = grid.bincount_2d(r[is_flood], c[is_flood], win.shape)
        return obs, fl, abs(wt.a)

    @staticmethod
    def _read_on_grid(href: str, bounds, shape: tuple[int, int]) -> np.ndarray:
        from rasterio.enums import Resampling

        with rasterio.open(href) as ds:
            w = from_bounds(*bounds, ds.transform).round_offsets().round_lengths()
            return ds.read(1, window=w, out_shape=shape, resampling=Resampling.nearest)

    def _read_item_checked(self, item, win: grid.Window):
        """`_read_item`, except that a 404 on any asset drops the scene (recorded) when configured.

        A 404 means the file is absent upstream (STAC lists it, the archive does
        not have it). Any other error is a fetch failure and propagates.
        """
        from ds_flood_gfm.fusion.net import retry

        try:
            return retry(lambda: self._read_item(item, win), f"GFM scene {item.id}")
        except _UnusableScene:
            if self.s["on_missing_asset"] != "drop_scene":
                raise
            self.dropped_scenes.append(item.id)
            return None
        except RasterioIOError as e:
            if "HTTP response code: 404" in str(e) and self.s["on_missing_asset"] == "drop_scene":
                self.dropped_scenes.append(item.id)
                return None
            raise

    def observe_day(self, win: grid.Window, items: list) -> tuple[np.ndarray, np.ndarray]:
        """(value raster 1/0/255, flooded fraction float32 with NaN where unobserved) for one day."""
        obs = np.zeros(win.shape)
        fl = np.zeros(win.shape)
        px = None
        with ThreadPoolExecutor(self.threads) as ex:
            for res in ex.map(lambda it: self._read_item_checked(it, win), items):
                if res is None:
                    continue
                o, f, px = res
                obs += o
                fl += f
        out = np.full(win.shape, NODATA, "uint8")
        frac = np.full(win.shape, np.nan, "float32")
        if px is None:
            return out, frac
        # expected pixel centres per cell from cell area (Equi7 is near-equidistant)
        lat = win.lat()[:, None]
        cell_area_m2 = (grid.RES * 111_320.0) ** 2 * np.cos(np.deg2rad(lat))
        expected = cell_area_m2 / (px * px)
        observed_cell = obs >= self.s["observed_min_frac"] * expected
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(obs > 0, fl / obs, np.nan)
        frac[observed_cell] = f[observed_cell]
        out[observed_cell] = DRY
        out[observed_cell & (f >= self.s["flood_threshold"])] = FLOOD
        return out, frac


def expected_samples(lat: float, px_m: float = 80.0) -> float:
    return (grid.RES * 111_320.0) ** 2 * math.cos(math.radians(lat)) / (px_m * px_m)

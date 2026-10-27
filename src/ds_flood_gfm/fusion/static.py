"""Static layers on the 0.01-degree grid: floodable domain, permanent water, land cover, snow season, strata.

Each fine-resolution input (HAND 30 m, GSW 30 m, WorldCover 10 m) is read as
a decimated window (every `sample_read_step`-th pixel, served from COG
overviews where they exist). Every sampled pixel centre is binned into its
0.01-degree cell, and per-cell fractions are formed from the counts. That is
a sampling estimate of the area fraction with 80-900 samples per cell.

Tile absence is a real state (ocean has no HAND tile) and is decided against a
bucket listing, never by catching a failed read: a tile that is listed but
cannot be read raises.
"""

from __future__ import annotations

import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import Config

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="5",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    VSI_CACHE="TRUE",
)

# Stratum codes: landcover (0 open, 1 vegetated, 2 urban) * 2 + snow (0 no, 1 yes)
STRATUM_NAMES = {
    0: "open_nosnow",
    1: "open_snow",
    2: "vegetated_nosnow",
    3: "vegetated_snow",
    4: "urban_nosnow",
    5: "urban_snow",
}
LANDCOVER_NAMES = {0: "open", 1: "vegetated", 2: "urban"}
NODATA_U8 = 255


# --------------------------------------------------------------------------- tile indexes

def _list_s3(bucket_url: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        url = f"{bucket_url}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token, safe="")
        with urllib.request.urlopen(url, timeout=120) as r:
            root = ET.fromstring(r.read())
        keys += [c.find("s3:Key", ns).text for c in root.findall("s3:Contents", ns)]
        nxt = root.find("s3:NextContinuationToken", ns)
        if nxt is None:
            return keys
        token = nxt.text


def _list_gcs(bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    marker = ""
    ns = {"g": "http://doc.s3.amazonaws.com/2006-03-01"}
    while True:
        url = f"https://storage.googleapis.com/{bucket}?prefix={prefix}&marker={urllib.request.quote(marker)}"
        with urllib.request.urlopen(url, timeout=120) as r:
            root = ET.fromstring(r.read())
        page = [c.find("g:Key", ns).text for c in root.findall("g:Contents", ns)]
        keys += page
        trunc = root.find("g:IsTruncated", ns)
        if trunc is None or trunc.text != "true" or not page:
            return keys
        marker = page[-1]


def tile_index(cfg: Config, layer: str) -> set[str]:
    """Set of tile file names that exist upstream for `layer`, cached locally as a text file."""
    path = cfg.local("raw", "tile_index", f"{layer}.txt")
    if path.exists():
        return set(path.read_text().split())
    if layer == "hand":
        keys = _list_s3("https://glo-30-hand.s3.amazonaws.com", "v1/2021/")
    elif layer == "worldcover":
        keys = _list_s3("https://esa-worldcover.s3.eu-central-1.amazonaws.com", "v200/2021/map/")
    elif layer == "gsw":
        keys = _list_gcs("global-surface-water", "downloads2021/occurrence/")
    else:
        raise ValueError(layer)
    names = sorted({k.rsplit("/", 1)[-1] for k in keys if k.endswith(".tif")})
    if not names:
        raise RuntimeError(f"tile listing for {layer} returned no .tif keys; refusing to treat everything as absent")
    path.write_text("\n".join(names))
    return set(names)


# --------------------------------------------------------------------------- tile url builders

def _hand_tiles(cfg: Config, bounds) -> list[str]:
    minx, miny, maxx, maxy = bounds
    tmpl = cfg["static"]["hand"]["url_template"]
    out = []
    for lat in range(math.floor(miny), math.ceil(maxy)):
        for lon in range(math.floor(minx), math.ceil(maxx)):
            la = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            lo = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            out.append(tmpl.format(lat=la, lon=lo))
    return out


def _worldcover_tiles(cfg: Config, bounds) -> list[str]:
    minx, miny, maxx, maxy = bounds
    tmpl = cfg["static"]["worldcover"]["url_template"]
    out = []
    for lat in range(3 * math.floor(miny / 3), math.ceil(maxy / 3) * 3, 3):
        for lon in range(3 * math.floor(minx / 3), math.ceil(maxx / 3) * 3, 3):
            t = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            out.append(tmpl.format(tile=t))
    return out


def _gsw_tiles(cfg: Config, bounds) -> list[str]:
    minx, miny, maxx, maxy = bounds
    tmpl = cfg["static"]["gsw"]["url_template"]
    out = []
    # GSW tiles are named by their top-left corner: tile (left, top) covers
    # lon [left, left + 10), lat [top - 10, top); zero is written "0E" / "0N".
    for k in range(math.floor(miny / 10), math.floor((maxy - 1e-9) / 10) + 1):
        top = 10 * k + 10
        for left in range(10 * math.floor(minx / 10), 10 * math.ceil(maxx / 10), 10):
            lo = f"{abs(left)}{'E' if left >= 0 else 'W'}"
            la = f"{abs(top)}{'N' if top >= 0 else 'S'}"
            out.append(tmpl.format(lon=lo, lat=la))
    return out


# --------------------------------------------------------------------------- sampler

def _accumulate(url: str, win: grid.Window, step: int, classify, counts: dict[str, np.ndarray]) -> None:
    """Read `url` over `win` at 1/step resolution and add per-cell counts produced by `classify`.

    `classify(arr) -> dict[name, bool array]` must include key "valid".
    """
    minx, miny, maxx, maxy = win.bounds
    with rasterio.Env(**GDAL_ENV), rasterio.open("/vsicurl/" + url) as ds:
        b = ds.bounds
        ix0, iy0, ix1, iy1 = max(minx, b.left), max(miny, b.bottom), min(maxx, b.right), min(maxy, b.top)
        if ix0 >= ix1 or iy0 >= iy1:
            return
        rw = from_bounds(ix0, iy0, ix1, iy1, ds.transform).round_offsets().round_lengths()
        if rw.width <= 0 or rw.height <= 0:
            return
        oh, ow = max(1, int(rw.height // step)), max(1, int(rw.width // step))
        arr = ds.read(1, window=rw, out_shape=(oh, ow), resampling=Resampling.nearest)
        t = ds.window_transform(rw)
    dx = t.a * rw.width / ow
    dy = t.e * rw.height / oh
    lon = t.c + (np.arange(ow) + 0.5) * dx
    lat = t.f + (np.arange(oh) + 0.5) * dy
    rows = grid.row_of(lat) - win.row0
    cols = grid.col_of(lon) - win.col0
    rin = (rows >= 0) & (rows < win.shape[0])
    cin = (cols >= 0) & (cols < win.shape[1])
    arr = arr[np.ix_(rin, cin)]
    rr, cc = np.meshgrid(rows[rin], cols[cin], indexing="ij")
    for name, mask in classify(arr).items():
        counts[name] += grid.bincount_2d(rr[mask], cc[mask], win.shape)
    counts["_samples"] += grid.bincount_2d(rr.ravel(), cc.ravel(), win.shape)


def _layer_counts(cfg: Config, win: grid.Window, layer: str, urls: list[str], classify, names: list[str]) -> dict[str, np.ndarray]:
    index = tile_index(cfg, layer)
    step = cfg["static"]["sample_read_step"]
    counts = {n: np.zeros(win.shape, "float64") for n in [*names, "valid", "_samples"]}
    for url in urls:
        if url.rsplit("/", 1)[-1] not in index:
            continue  # tile does not exist upstream (ocean / outside product): cells stay unsampled
        _accumulate(url, win, step, classify, counts)
    return counts


def _frac(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan).astype("float32")


def compute_static(cfg: Config, win: grid.Window) -> xr.Dataset:
    """Floodable / permanent-water / land-cover fractions and classes for one window."""
    s = cfg["static"]
    hand_thr = s["hand"]["threshold_m"]
    hand = _layer_counts(
        cfg, win, "hand", _hand_tiles(cfg, win.bounds),
        lambda a: {"valid": np.isfinite(a) & (a > -1e30), "low": np.isfinite(a) & (a < hand_thr)},
        ["low"],
    )
    occ_thr = s["gsw"]["permanent_occurrence"]
    gsw = _layer_counts(
        cfg, win, "gsw", _gsw_tiles(cfg, win.bounds),
        lambda a: {"valid": a != 255, "perm": (a != 255) & (a >= occ_thr)},
        ["perm"],
    )
    wc_s = s["worldcover"]
    wc = _layer_counts(
        cfg, win, "worldcover", _worldcover_tiles(cfg, win.bounds),
        lambda a: {"valid": a != 0, "tree": a == wc_s["tree_class"], "built": a == wc_s["built_class"]},
        ["tree", "built"],
    )
    hand_low = _frac(hand["low"], hand["valid"])
    perm = _frac(gsw["perm"], gsw["valid"])
    tree = _frac(wc["tree"], wc["valid"])
    built = _frac(wc["built"], wc["valid"])

    floodable = np.nan_to_num(hand_low, nan=0.0) >= s["hand"]["floodable_min_frac"]
    perm_water = np.nan_to_num(perm, nan=0.0) >= s["gsw"]["permanent_min_frac"]
    has_lc = wc["valid"] > 0
    landcover = np.full(win.shape, NODATA_U8, "uint8")
    landcover[has_lc] = 0
    landcover[has_lc & (np.nan_to_num(tree) >= wc_s["tree_min_frac"])] = 1
    landcover[has_lc & (np.nan_to_num(built) >= wc_s["built_min_frac"])] = 2  # urban takes precedence
    domain = floodable & ~perm_water & has_lc

    coords = {"lat": win.lat(), "lon": win.lon()}
    dims = ("lat", "lon")
    ds = xr.Dataset(
        {
            "hand_low_frac": (dims, hand_low),
            "perm_water_frac": (dims, perm),
            "tree_frac": (dims, tree),
            "built_frac": (dims, built),
            "floodable": (dims, floodable),
            "perm_water": (dims, perm_water),
            "landcover": (dims, landcover),
            "domain": (dims, domain),
        },
        coords=coords,
        attrs={"window": win.key(), "row0": win.row0, "row1": win.row1, "col0": win.col0, "col1": win.col1},
    )
    return ds


def static_for_window(cfg: Config, win: grid.Window) -> xr.Dataset:
    """Cached wrapper around `compute_static`, one NetCDF per window (idempotent)."""
    path = cfg.local("static", f"{win.key()}.nc")
    if path.exists():
        return xr.load_dataset(path, engine="netcdf4")
    ds = compute_static(cfg, win)
    tmp = path.with_suffix(".part.nc")
    ds.to_netcdf(tmp, engine="netcdf4")
    tmp.rename(path)
    return ds


# --------------------------------------------------------------------------- snow season

_SNOW_CLIM: dict[str, xr.DataArray] = {}


class SnowClimatology:
    """Monthly snow-season flag from ERA5-Land monthly mean snow cover (%).

    snow_season(month, lat, lon) = climatological mean snow cover for that
    calendar month >= threshold. Nearest 0.25-degree cell; ocean (NaN) = no snow.
    The 12-month climatology is computed once from the monthly series, cached
    on disk next to it and in memory per process.
    """

    def __init__(self, cfg: Config) -> None:
        s = cfg["static"]["snow"]
        y0, y1 = s["years"]
        path = cfg.local("raw", "era5_land", f"era5land_snow_cover_monthly_{y0}_{y1}_{s['grid_deg']}deg.nc")
        clim_path = cfg.local("static", f"snow_climatology_{y0}_{y1}_{s['grid_deg']}deg.nc")
        self.threshold = s["snow_season_min_cover_pct"]
        key = str(clim_path)
        if key in _SNOW_CLIM:
            self.clim = _SNOW_CLIM[key]
            return
        if clim_path.exists():
            self.clim = xr.load_dataarray(clim_path, engine="netcdf4")
            _SNOW_CLIM[key] = self.clim
            return
        if not path.exists():
            from ds_flood_gfm.fusion.config import pull_raw

            if not pull_raw(cfg, path, "era5_land"):
                raise FileNotFoundError(f"snow climatology input missing locally and on blob raw/era5_land: {path.name}; "
                                        "run scripts/fusion/fetch_snow_climatology.py")
        ds = xr.open_dataset(path, engine="netcdf4")
        var = "snowc" if "snowc" in ds else next(iter(ds.data_vars))
        da = ds[var]
        tdim = "valid_time" if "valid_time" in da.dims else "time"
        clim = da.groupby(f"{tdim}.month").mean(tdim)
        lon_name = "longitude" if "longitude" in clim.coords else "lon"
        lat_name = "latitude" if "latitude" in clim.coords else "lat"
        clim = clim.assign_coords({lon_name: (((clim[lon_name] + 180) % 360) - 180)}).sortby(lon_name)
        clim = clim.rename({lon_name: "lon", lat_name: "lat"})
        clim = clim.drop_vars([c for c in clim.coords if c not in ("month", "lat", "lon")]).load()
        tmp = clim_path.with_suffix(".part.nc")
        clim.to_netcdf(tmp, engine="netcdf4")
        tmp.rename(clim_path)
        self.clim = clim
        _SNOW_CLIM[key] = clim

    def snow_season(self, month: int, win: grid.Window) -> np.ndarray:
        c = self.clim.sel(month=month).sel(lat=xr.DataArray(win.lat(), dims="lat"), lon=xr.DataArray(win.lon(), dims="lon"), method="nearest")
        return np.nan_to_num(c.values, nan=0.0) >= self.threshold


def stratum(landcover: np.ndarray, snow: np.ndarray) -> np.ndarray:
    """6-class stratum code (see STRATUM_NAMES); 255 where land cover is unknown."""
    out = np.full(landcover.shape, NODATA_U8, "uint8")
    ok = landcover != NODATA_U8
    out[ok] = landcover[ok] * 2 + snow[ok].astype("uint8")
    return out


def hand_tile_name_check(name: str) -> bool:
    return re.match(r"Copernicus_DSM_COG_10_[NS]\d{2}_00_[EW]\d{3}_00_HAND\.tif", name) is not None


__all__ = [
    "STRATUM_NAMES",
    "LANDCOVER_NAMES",
    "SnowClimatology",
    "compute_static",
    "static_for_window",
    "stratum",
    "tile_index",
    "Path",
]

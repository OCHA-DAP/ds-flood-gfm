"""Seasonal "normally wet" baseline from JRC Global Surface Water monthly recurrence (D-38).

Data: gs://global-surface-water/downloads2021/monthlyRecurrence2021/monthlyRecurrence{1..12}/
{row_off:010d}-{col_off:010d}.tif, int16 percent 0-100, 10 x 10 degree tiles of 40000 x 40000
pixels at 0.00025 degree, the grid's top edge at 80 N (row_off = (80 - top) / 0.00025,
col_off = (left + 180) / 0.00025). Value = share of the years with a valid observation in that
calendar month in which the pixel was water (Landsat, 1984-2021).

Caveat (checked 2026-10-13 on the Inner Niger Delta): the export's has_observations layers are
identical for every month and match "occurrence > 0" for 99% of pixels. They mark "water ever
seen", not "this month was observed". A month that was never observed (persistently cloudy)
therefore reads 0 and cannot be told apart from a dry month. The baseline can miss
cloudy-season water. It errs towards masking too little, never too much.

A 0.01-degree cell is exactly 40 x 40 pixels, so cells and pixels align. A cell is normally wet
in month m when at least `normal_wet_min_frac` of its pixels have recurrence >= `wet_recurrence_pct`.
"""

from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.windows import Window as RWindow

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import Config

URL = "https://storage.googleapis.com/global-surface-water/downloads2021/monthlyRecurrence2021"
PX = 0.00025
TILE_DEG = 10
TOP = 80.0
PER_CELL = 40  # 0.01 / 0.00025
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3")


def tiles_for_bounds(minx: float, miny: float, maxx: float, maxy: float) -> list[tuple[float, float]]:
    """(left, top) of each 10-degree recurrence tile intersecting the bounds."""
    if maxy > TOP:
        raise ValueError(f"window reaches {maxy} N, beyond the recurrence grid's top edge at {TOP} N")
    lefts = range(math.floor(minx / TILE_DEG) * TILE_DEG, math.ceil(maxx / TILE_DEG) * TILE_DEG, TILE_DEG)
    tops = range(math.ceil(maxy / TILE_DEG) * TILE_DEG, math.floor(miny / TILE_DEG) * TILE_DEG, -TILE_DEG)
    return [(float(left), float(top)) for top in tops for left in lefts]


def tile_name(left: float, top: float) -> str:
    return f"{round((TOP - top) / PX):010d}-{round((left + 180) / PX):010d}.tif"


def wet_share(rec: np.ndarray, wet_recurrence_pct: float) -> np.ndarray:
    """Share of each 40 x 40 pixel block with recurrence >= wet_recurrence_pct (rec dims multiples of 40)."""
    r, c = rec.shape[0] // PER_CELL, rec.shape[1] // PER_CELL
    w = (rec >= wet_recurrence_pct).reshape(r, PER_CELL, c, PER_CELL)
    return w.mean(axis=(1, 3)).astype("float32")


def read_month(win: grid.Window, month: int) -> np.ndarray:
    """Recurrence (percent) for the window at 0.00025 degree, shape (rows*40, cols*40), from every tile it touches."""
    minx, miny, maxx, maxy = win.bounds
    out = np.zeros((win.shape[0] * PER_CELL, win.shape[1] * PER_CELL), "int16")
    with rasterio.Env(**GDAL_ENV):
        for left, top in tiles_for_bounds(minx, miny, maxx, maxy):
            x0, x1 = max(minx, left), min(maxx, left + TILE_DEG)
            y0, y1 = max(miny, top - TILE_DEG), min(maxy, top)
            if x0 >= x1 or y0 >= y1:
                continue
            col0, row0 = round((x0 - left) / PX), round((top - y1) / PX)
            ncol, nrow = round((x1 - x0) / PX), round((y1 - y0) / PX)
            from ds_flood_gfm.fusion.net import retry

            def _read(url=f"/vsicurl/{URL}/monthlyRecurrence{month}/{tile_name(left, top)}"):
                with rasterio.open(url) as ds:
                    return ds.read(1, window=RWindow(col0, row0, ncol, nrow))

            a = retry(_read, f"GSW monthly recurrence m{month} {tile_name(left, top)}")
            orow, ocol = round((maxy - y1) / PX), round((x0 - minx) / PX)
            out[orow:orow + nrow, ocol:ocol + ncol] = a
    return out


def normally_wet(cfg: Config, win: grid.Window, month: int) -> np.ndarray:
    """Boolean (rows, cols): cell normally wet in this calendar month. Cached per window and month."""
    b = cfg["baseline"]
    path = cfg.local("static", "baseline", f"{win.key()}_m{month:02d}.npy")
    if path.exists():
        share = np.load(path)
    else:
        share = wet_share(read_month(win, month), b["wet_recurrence_pct"])
        np.save(path, share)
    return share >= b["normal_wet_min_frac"]

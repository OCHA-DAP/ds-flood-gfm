"""AER FloodScan SFED on the 0.01-degree grid.

SFED is already a flooded fraction per 300-arcsec (~9 km) cell, so each
0.01-degree cell takes the SFED of the FloodScan cell containing its centre
and is flood iff SFED >= flood_threshold (fixed decision 1). 0.01 degrees does
not nest in 300 arcsec; the containing-cell rule is the documented resampling.

Regions and archives (docs/inventory.md):
  Africa         prod raster  floodscan/daily/v5/processed/ daily COGs (current)
  Asia-Australia dev raster   raw SFED NetCDF 1998-01-12 .. 2026-03-31
  N. America     dev raster   raw SFED NetCDF 1998-01-12 .. 2026-03-31
  S. America     dev raster   raw SFED NetCDF 1998-01-12 .. 2026-03-31
Africa wins where grids overlap (50-60E). Outside every region, or after a
region's archive ends, the cell is NoData (not covered), never 0.
FloodScan has no cloud/snow mask of its own and no NaN in these files.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import rasterio
import xarray as xr

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import Config
from ds_flood_gfm.fusion.sources import DRY, FLOOD, NODATA

FS_RES = 1.0 / 12.0


@dataclass(frozen=True)
class Region:
    name: str
    left: float
    top: float
    ncols: int
    nrows: int
    last_day: dt.date | None  # None = current daily archive

    def index(self, lon: np.ndarray, lat: np.ndarray):
        c = np.floor((lon - self.left) / FS_RES + 1e-9).astype(int)
        r = np.floor((self.top - lat) / FS_RES + 1e-9).astype(int)
        ok = (c >= 0) & (c < self.ncols) & (r >= 0) & (r < self.nrows)
        return r, c, ok


class FloodScanSource:
    name = "floodscan"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.s = cfg["sources"]["floodscan"]
        self.thr = self.s["flood_threshold"]
        self._nc: dict[str, xr.Dataset] = {}
        self.africa = Region("africa", -30.0, 40.0, 1080, 1080, None)
        self.regions: list[Region] = [self.africa]
        for name, meta in self.s["regional_nc"]["files"].items():
            self.regions.append(Region(name, float(meta["left"]), float(meta["top"]), int(meta["ncols"]), int(meta["nrows"]),
                                       dt.date.fromisoformat(meta["last_day"])))
        self._africa_cache: dict[dt.date, np.ndarray] = {}
        # days the Africa archive has no file for (upstream absence); callers record them
        self.absent_days: set[dt.date] = set()

    def _nc_for(self, reg: Region) -> xr.Dataset:
        """Open a regional NetCDF on first use (downloaded once) and check its grid against the config."""
        if reg.name in self._nc:
            return self._nc[reg.name]
        meta = self.s["regional_nc"]["files"][reg.name]
        ds = self._open_nc(reg.name, meta["blob"])
        lat, lon = ds.lat.values, ds.lon.values
        got = (round(float(lon[0] - FS_RES / 2), 4), round(float(lat[0] + FS_RES / 2), 4), len(lon), len(lat), str(ds.time.values[-1])[:10])
        want = (reg.left, reg.top, reg.ncols, reg.nrows, reg.last_day.isoformat())
        if got != want:
            raise ValueError(f"FloodScan {reg.name}: file grid/end {got} != config {want}")
        return ds

    def _open_nc(self, name: str, blob_name: str) -> xr.Dataset:
        local = self.cfg.local("raw", "floodscan", blob_name.rsplit("/", 1)[-1])
        if not local.exists():
            import ocha_stratus as stratus

            rc = self.s["regional_nc"]
            cc = stratus.get_container_client(stage=rc["stage"], container_name=rc["container"])
            tmp = local.with_suffix(".part")
            with open(tmp, "wb") as f:
                cc.get_blob_client(blob_name).download_blob(max_concurrency=4).readinto(f)
            tmp.rename(local)
        ds = xr.open_dataset(local, engine="netcdf4")
        if not (ds.lat.values[1] < ds.lat.values[0]):
            raise ValueError(f"{local}: expected descending latitude")
        self._nc[name] = ds
        return ds

    def _africa_day(self, day: dt.date) -> np.ndarray | None:
        """SFED grid for Africa on `day`; None if the archive has no file for that day."""
        if day in self._africa_cache:
            return self._africa_cache[day]
        a = self.s["africa"]
        name = a["blob_template"].format(date=day.isoformat())
        local = self.cfg.local("raw", "floodscan", "africa_daily", name.rsplit("/", 1)[-1])
        if not local.exists():
            import ocha_stratus as stratus
            from azure.core.exceptions import ResourceNotFoundError

            cc = stratus.get_container_client(stage=a["stage"], container_name=a["container"])
            try:
                data = cc.get_blob_client(name).download_blob().readall()
            except ResourceNotFoundError:
                self._africa_cache[day] = None
                self.absent_days.add(day)
                return None  # upstream absence for this day: NoData, and listed in absent_days
            import os
            import threading

            tmp = local.with_name(f"{local.name}.{os.getpid()}.{threading.get_ident()}.part")
            tmp.write_bytes(data)
            tmp.rename(local)  # atomic; a concurrent writer of the same day produces identical bytes
        with rasterio.open(local) as ds:
            names = list(ds.descriptions)
            if a["band"] not in names:
                raise ValueError(f"{local}: band {a['band']!r} not in {names}")
            arr = ds.read(names.index(a["band"]) + 1)
            if (ds.transform.c, ds.transform.f) != (self.africa.left, self.africa.top) or arr.shape != (self.africa.nrows, self.africa.ncols):
                raise ValueError(f"{local}: unexpected grid {ds.transform} {arr.shape}")
        if len(self._africa_cache) > 400:
            self._africa_cache.clear()
        self._africa_cache[day] = arr
        return arr

    def observe_days(self, win: grid.Window, days: list[dt.date]) -> tuple[np.ndarray, np.ndarray]:
        """(values[day, row, col] 1/0/255, sfed[day, row, col] float32) for a window."""
        lon2, lat2 = np.meshgrid(win.lon(), win.lat())
        out = np.full((len(days), *win.shape), NODATA, "uint8")
        sfed = np.full((len(days), *win.shape), np.nan, "float32")
        claimed = np.zeros(win.shape, bool)
        for reg in self.regions:
            r, c, ok = reg.index(lon2, lat2)
            ok &= ~claimed
            if not ok.any():
                continue
            claimed |= ok
            rr, cc_ = r[ok], c[ok]
            if reg.name == "africa":
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(8) as ex:  # per-day COG downloads are independent (I/O bound)
                    arrs = list(ex.map(self._africa_day, days))
                for i, arr in enumerate(arrs):
                    if arr is None:
                        continue
                    sfed[i][ok] = arr[rr, cc_]
            else:
                ds = self._nc_for(reg)
                usable = [i for i, d in enumerate(days) if d <= reg.last_day]
                if not usable:
                    continue
                r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc_.min(), cc_.max() + 1
                tsel = np.array([np.datetime64(days[i]) for i in usable])
                block = ds.SFED_AREA.isel(lat=slice(r0, r1), lon=slice(c0, c1)).sel(time=tsel).transpose("time", "lat", "lon").values
                for k, i in enumerate(usable):
                    sfed[i][ok] = block[k][rr - r0, cc_ - c0]
        valid = np.isfinite(sfed)
        out[valid] = DRY
        out[valid & (sfed >= self.thr)] = FLOOD
        return out, sfed

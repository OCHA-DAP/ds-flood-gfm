"""Static layers for the FloodScan reproduction, built in Earth Engine on the CSU AMSR2 0.25-deg grid.

Grid: West Africa box 0-20E, 0-20N, 80 x 80 cells, edges on multiples of 0.25 deg (matches the CSU
CDR grid). Every layer is footprint-weighted the way AER describes (ADD s. 4.3-4.6): a Gaussian of
half-weight diameter 27 km (sigma = 27 / 2.355 km), applied on a 1 km intermediate grid, then sampled
at the 0.25-deg cell centres.

Bands written to data/floodscan_repro/static_0p25.tif:
  ndvi_s01..ndvi_s23  MODIS MOD13A2 v061 NDVI, mean of 2013-2022 per 16-day slot (slot k starts on DOY 16k+1)
  ftree               MODIS MOD44B v061 Percent_Tree_Cover, mean 2018-2022, / 100
  fpow                persistent open water: MODIS MOD44W v006 water_mask, water in every year 2011-2015
  focean              not inside any LSIB 2017 country polygon (ocean), for coastal flags

Departures from AER: NDVI from MODIS 2013-2022 (AER: VIP Lab 2000-2010 long-term average); tree cover
from MODIS VCF (AER: Hansen treecover2000); persistent water from MOD44W (AER: Hansen OR SRTM water
bodies with basin-level exceptions, MOD44W only as gap filler). The 30 m sources exceeded Earth
Engine's interactive memory limit; at a ~27 km footprint the source resolution is immaterial.

    uv run python experiments/floodscan_repro/00_gee_static.py   # needs `earthengine authenticate`
"""

import urllib.request
from pathlib import Path

import ee

PROJECT = "ee-zackarno"  # the project the team's other repos initialise with
OUT = Path("data/floodscan_repro/static_0p25.tif")
X0, Y1, N = 0.0, 20.0, 80
SIGMA_M = 27_000 / 2.355
GRID = {"crs": "EPSG:4326", "crsTransform": [0.25, 0, X0, 0, -0.25, Y1]}  # for ee.Image.reproject

ee.Initialize(project=PROJECT)
kernel = ee.Kernel.gaussian(radius=3 * SIGMA_M, sigma=SIGMA_M, units="meters", normalize=True)
km = ee.Projection("EPSG:4326").atScale(1000)


def footprint(img, native=True):
    """Mean to 1 km (native-resolution inputs) or rasterise at 1 km (painted inputs), Gaussian footprint
    weighting, then sample at 0.25-deg centres."""
    coarse = img.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(km) if native else img.reproject(km)
    return coarse.convolve(kernel).reproject(**GRID)


ndvi = ee.ImageCollection("MODIS/061/MOD13A2").filterDate("2013-01-01", "2023-01-01").select("NDVI")
modis_proj = ndvi.first().projection()  # collection means lose their projection; restore MODIS's 1 km grid
slots = []
for k in range(23):
    doy0 = 16 * k + 1
    sl = ndvi.filter(ee.Filter.calendarRange(doy0, doy0 + 15, "day_of_year")).mean().multiply(0.0001)
    slots.append(footprint(sl.setDefaultProjection(modis_proj)).rename(f"ndvi_s{k + 1:02d}"))

# 250 m MODIS sources: averaging 30 m Hansen/GSW over a 5-deg tile exceeded Earth Engine's interactive
# memory limit ("User memory limit exceeded", 2026-10-21). At a ~27 km footprint the source detail is immaterial.
vcf = ee.ImageCollection("MODIS/061/MOD44B").filterDate("2018-01-01", "2023-01-01").select("Percent_Tree_Cover")
tree = vcf.map(lambda im: im.updateMask(im.lte(100))).mean().divide(100).setDefaultProjection(vcf.first().projection())
ftree = footprint(tree.unmask(0)).rename("ftree")  # >100 codes water/fill: treated as no trees
wm = ee.ImageCollection("MODIS/006/MOD44W").filterDate("2011-01-01", "2016-01-01").select("water_mask")
water = wm.min().setDefaultProjection(wm.first().projection())  # water in every year 2011-2015 = persistent
fpow = footprint(water.unmask(0).toFloat()).rename("fpow")
land = ee.Image(0).paint(ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017"), 1)
focean = footprint(ee.Image(1).subtract(land).toFloat(), native=False).rename("focean")

stack = ee.Image.cat(slots + [ftree, fpow, focean]).toFloat()

# One request over the whole box reads > 2^31 pixels of the 30 m layers ("Number of pixels requested
# ... exceeds the maximum allowed"), so request 5-deg tiles (20 x 20 cells) and stitch them locally.
import io  # noqa: E402

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

import http.client  # noqa: E402

TILE = 20
TILES = OUT.parent / "gee_tiles"  # finished tiles are cached so a rerun redoes only the missing ones
TILES.mkdir(parents=True, exist_ok=True)
mosaic, names = None, None
for ty in range(N // TILE):
    for tx in range(N // TILE):
        cache = TILES / f"tile_{tx}_{ty}.tif"
        if not cache.exists():
            x0, y1 = X0 + 0.25 * TILE * tx, Y1 - 0.25 * TILE * ty
            tile_grid = {"crs": "EPSG:4326", "crs_transform": [0.25, 0, x0, 0, -0.25, y1]}
            url = stack.getDownloadURL({"region": ee.Geometry.Rectangle([x0, y1 - 0.25 * TILE, x0 + 0.25 * TILE, y1], None, False),
                                        "format": "GEO_TIFF", **tile_grid})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=900) as r:
                        raw = r.read()
                    break
                except urllib.error.HTTPError as e:  # Earth Engine puts the reason in the body: not retried
                    raise RuntimeError(f"Earth Engine download failed for tile {tx},{ty}: HTTP {e.code}: "
                                       f"{e.read().decode(errors='replace')[:500]}") from e
                except (http.client.RemoteDisconnected, urllib.error.URLError, TimeoutError) as e:
                    if attempt == 2:
                        raise
                    print(f"tile {tx},{ty}: connection dropped ({e!r}); retrying", flush=True)
            tmp = cache.with_suffix(".part.tif")
            tmp.write_bytes(raw)
            tmp.rename(cache)
        with rasterio.open(cache) as ds:
            if (ds.height, ds.width) != (TILE, TILE):
                raise ValueError(f"tile {tx},{ty}: unexpected shape {(ds.height, ds.width)}")
            if mosaic is None:
                names = ds.descriptions
                mosaic = np.full((ds.count, N, N), np.nan, "float32")
            mosaic[:, ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE] = ds.read()
        print(f"tile {tx},{ty} ok", flush=True)

# Earth Engine's GeoTIFF download carries no band names, so name them from the stack order above;
# masked pixels (NDVI over the sea) arrive as -inf, written here as NaN.
names = tuple([f"ndvi_s{k + 1:02d}" for k in range(23)] + ["ftree", "fpow", "focean"])
if mosaic.shape[0] != len(names):
    raise ValueError(f"expected {len(names)} bands, got {mosaic.shape[0]}")
mosaic[~np.isfinite(mosaic)] = np.nan
OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = OUT.with_suffix(".part.tif")
with rasterio.open(tmp, "w", driver="GTiff", height=N, width=N, count=mosaic.shape[0], dtype="float32",
                   crs="EPSG:4326", transform=Affine(0.25, 0, X0, 0, -0.25, Y1), nodata=np.nan) as dst:
    dst.write(mosaic)
    dst.descriptions = names
tmp.rename(OUT)

with rasterio.open(OUT) as ds:
    if (ds.height, ds.width) != (N, N):
        raise ValueError(f"unexpected export grid {(ds.height, ds.width)}")
    print(f"wrote {OUT}: {ds.count} bands, transform {tuple(round(v, 4) for v in ds.transform[:6])}")
    for i, name in enumerate(ds.descriptions, 1):
        if name in ("ndvi_s15", "ftree", "fpow", "focean"):
            a = ds.read(i)
            print(f"  {name:9s} min {a.min():.3f} max {a.max():.3f} mean {a.mean():.3f}")

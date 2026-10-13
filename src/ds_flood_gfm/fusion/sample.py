"""Phase 2 stratified random sample of floodable cells (no gold), for prior v1 and EM.

Design (D-23):
  1. Candidate 0.25-degree blocks on land are drawn at random within each
     Koppen main group (A, B, C, D, E; Beck et al. 2018, 0.083 deg) x continent
     cell, `per_cell` blocks each (seeded).
  2. Urban strata are rare, so city-centred blocks (Natural Earth populated
     places, largest first) are added, split by Koppen group so that
     snow-climate cities (D) are included.
  3. Static layers are computed for every candidate; cell-days per stratum
     over the sample period are tallied from the monthly snow flag. Blocks are
     added from a reserve list until every stratum reaches the target, or
     the reserve runs out (reported, never silently accepted).
Every day in the sample period is fetched for every block (sources as in the
gold cube; no truth).
"""

from __future__ import annotations

import datetime as dt
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

from ds_flood_gfm.fusion import grid, static
from ds_flood_gfm.fusion.config import Config

KOPPEN_GROUP = {**{k: "A" for k in (1, 2, 3)}, **{k: "B" for k in (4, 5, 6, 7)}, **{k: "C" for k in range(8, 17)},
                **{k: "D" for k in range(17, 29)}, **{k: "E" for k in (29, 30)}}


def _koppen(cfg: Config):
    zpath = cfg.local("raw", "koppen", "Beck_KG_V1.zip")
    tif = zpath.parent / "Beck_KG_V1_present_0p083.tif"
    if not tif.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extract("Beck_KG_V1_present_0p083.tif", zpath.parent)
    with rasterio.open(tif) as ds:
        return ds.read(1), ds.transform


def koppen_at(cfg: Config, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    arr, t = _koppen(cfg)
    c = np.floor((lon - t.c) / t.a).astype(int)
    r = np.floor((lat - t.f) / t.e).astype(int)
    ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    out = np.zeros(lon.shape, int)
    out[ok] = arr[r[ok], c[ok]]
    return out


def _continents(cfg: Config, lon, lat) -> np.ndarray:
    path = cfg.local("raw", "natural_earth", "ne_50m_admin_0_countries.zip")
    ne = gpd.read_file(path)[["CONTINENT", "geometry"]]
    ne = ne[ne.CONTINENT != "Seven seas (open ocean)"].to_crs(8857)
    pts = gpd.GeoDataFrame(geometry=[shapely.Point(x, y) for x, y in zip(lon, lat)], crs="EPSG:4326").to_crs(8857)
    j = gpd.sjoin_nearest(pts, ne, how="left")
    j = j[~j.index.duplicated()]
    return j.CONTINENT.values


def candidate_blocks(cfg: Config, per_cell: int, seed: int) -> pd.DataFrame:
    s = cfg["sample"]
    bd = s["block_deg"]
    lons = np.arange(-180 + bd / 2, 180, bd)
    lats = np.arange(-60 + bd / 2, 75, bd)
    LON, LAT = np.meshgrid(lons, lats)
    LON, LAT = LON.ravel(), LAT.ravel()
    kg = koppen_at(cfg, LON, LAT)
    land = kg > 0
    LON, LAT, kg = LON[land], LAT[land], kg[land]
    grp = np.array([KOPPEN_GROUP[k] for k in kg])
    rng = np.random.default_rng(seed)
    # thin to a manageable random subset before the (slow) continent join
    idx = rng.permutation(len(LON))[: 40_000]
    df = pd.DataFrame({"lon": LON[idx], "lat": LAT[idx], "koppen": kg[idx], "kgroup": grp[idx]})
    df["continent"] = _continents(cfg, df.lon.values, df.lat.values)
    df = df[df.continent != "Antarctica"]
    picks = []
    for (_, _), g in df.groupby(["kgroup", "continent"]):
        picks.append(g.sample(n=min(per_cell * 3, len(g)), random_state=int(rng.integers(1 << 31))))
    cand = pd.concat(picks).reset_index(drop=True)
    cand["rank"] = cand.groupby(["kgroup", "continent"]).cumcount()
    cand["kind"] = np.where(cand["rank"] < per_cell, "random", "reserve")
    return cand


def snow_reserve_blocks(cfg: Config, n: int, seed: int) -> pd.DataFrame:
    """Extra reserve candidates in snow climates (Koppen D/E), for topping up the snow strata."""
    bd = cfg["sample"]["block_deg"]
    lons = np.arange(-180 + bd / 2, 180, bd)
    lats = np.arange(30 + bd / 2, 72, bd)
    LON, LAT = (a.ravel() for a in np.meshgrid(lons, lats))
    kg = koppen_at(cfg, LON, LAT)
    keep = np.isin(kg, [k for k, g in KOPPEN_GROUP.items() if g in ("D", "E")])
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.where(keep)[0], size=n, replace=False)
    df = pd.DataFrame({"lon": LON[idx], "lat": LAT[idx], "koppen": kg[idx]})
    df["kgroup"] = [KOPPEN_GROUP[k] for k in df.koppen]
    df["continent"] = _continents(cfg, df.lon.values, df.lat.values)
    df["kind"] = "reserve"
    df["rank"] = 99
    return df


def city_blocks(cfg: Config, n_snow: int, n_other: int) -> pd.DataFrame:
    path = cfg.local("raw", "natural_earth", "ne_50m_populated_places.zip")
    if not path.exists():
        import urllib.request

        urllib.request.urlretrieve("https://naciscdn.org/naturalearth/50m/cultural/ne_50m_populated_places.zip", path)
    pp = gpd.read_file(path)[["NAME", "POP_MAX", "geometry"]].sort_values("POP_MAX", ascending=False)
    pp["lon"], pp["lat"] = pp.geometry.x, pp.geometry.y
    pp["koppen"] = koppen_at(cfg, pp.lon.values, pp.lat.values)
    pp = pp[pp.koppen > 0]
    pp["kgroup"] = [KOPPEN_GROUP[k] for k in pp.koppen]
    snow = pp[pp.kgroup.isin(["D", "E"])].head(n_snow * 2)
    other = pp[~pp.kgroup.isin(["D", "E"])].head(n_other * 2)
    out = pd.concat([snow.assign(rank=np.arange(len(snow))), other.assign(rank=np.arange(len(other)))])
    out["continent"] = _continents(cfg, out.lon.values, out.lat.values)
    out["kind"] = np.where(((out.kgroup.isin(["D", "E"])) & (out["rank"] < n_snow)) | ((~out.kgroup.isin(["D", "E"])) & (out["rank"] < n_other)), "city", "city_reserve")
    return out[["NAME", "lon", "lat", "koppen", "kgroup", "continent", "kind", "rank"]].rename(columns={"NAME": "city"})


def block_window(cfg: Config, lon: float, lat: float, city: bool = False) -> grid.Window:
    h = cfg["sample"]["city_block_deg" if city else "block_deg"] / 2
    return grid.window_from_bounds(lon - h, lat - h, lon + h, lat + h)


def stratum_cell_days(cfg: Config, win: grid.Window, st, snow: static.SnowClimatology) -> dict[str, int]:
    y0, y1 = cfg["sample"]["years"]
    days = pd.date_range(f"{y0}-01-01", f"{y1}-12-31")
    per_month = days.month.value_counts()
    dom = st.domain.values.astype(bool)
    lc = st.landcover.values
    out = {n: 0 for n in static.STRATUM_NAMES.values()}
    for m, ndays in per_month.items():
        strat = static.stratum(lc, snow.snow_season(m, win))
        for code, name in static.STRATUM_NAMES.items():
            out[name] += int(((strat == code) & dom).sum()) * int(ndays)
    return out


def sample_days(cfg: Config) -> list[dt.date]:
    y0, y1 = cfg["sample"]["years"]
    return [d.date() for d in pd.date_range(f"{y0}-01-01", f"{y1}-12-31")]

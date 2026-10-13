"""Writing fusion rasters as COGs and mirroring them to blob storage.

Layout (KB convention {PROJECT_PREFIX}/{raw|processed}/{datasource}/{filename},
dev account, `projects` container; D-25):
  local  data/fusion/out/static/{key}.tif
         data/fusion/out/daily/{key}/{key}_{YYYY-MM-DD}.tif
  blob   ds-flood-gfm/processed/flood_fusion_v1/static/{key}.tif
         ds-flood-gfm/processed/flood_fusion_v1/daily/{key}/{key}_{YYYY-MM-DD}.tif
Daily COG bands (float32, NaN = outside floodable domain):
  1 p_flood  2 abs_logodds  3 n_sources_observed  4 days_since_obs (65535 = never)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import rasterio

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import Config

DAILY_BANDS = ["p_flood", "abs_logodds", "n_sources_observed", "days_since_obs"]


def _write_cog(path: Path, bands: list[np.ndarray], names: list[str], win: grid.Window, dtype: str, nodata, tags: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.tif")
    profile = dict(driver="COG", width=win.shape[1], height=win.shape[0], count=len(bands), dtype=dtype,
                   crs="EPSG:4326", transform=win.transform, nodata=nodata, compress="DEFLATE", blocksize=256)
    with rasterio.open(tmp, "w", **profile) as dst:
        for i, (b, n) in enumerate(zip(bands, names), start=1):
            dst.write(b.astype(dtype), i)
            dst.set_band_description(i, n)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    tmp.rename(path)


def static_cog_path(cfg: Config, key: str) -> Path:
    return cfg.local("out", "static", f"{key}.tif")


def static_cog_exists(cfg: Config, key: str) -> bool:
    return static_cog_path(cfg, key).exists()


def write_static_cogs(cfg: Config, key: str, win: grid.Window, st, months: dict[int, np.ndarray]) -> Path:
    """One uint8 COG per window: domain, HAND<15 m fraction (%), permanent water, land cover, stratum for months 1..12."""
    hand = np.where(np.isfinite(st.hand_low_frac.values), np.round(st.hand_low_frac.values * 100), 255)
    bands = [st.domain.values.astype("uint8"), hand, st.perm_water.values.astype("uint8"), st.landcover.values]
    names = ["domain", "hand_lt15m_pct", "permanent_water", "landcover_0open_1veg_2urban"]
    for m in range(1, 13):
        bands.append(months[m])
        names.append(f"stratum_m{m:02d}")
    path = static_cog_path(cfg, key)
    _write_cog(path, bands, names, win, "uint8", 255, tags={"version": cfg["version"], "stratum_codes": "0 open_nosnow,1 open_snow,2 vegetated_nosnow,3 vegetated_snow,4 urban_nosnow,5 urban_snow"})
    return path


def daily_cog_path(cfg: Config, key: str, day: dt.date) -> Path:
    return cfg.local("out", "daily", key, f"{key}_{day.isoformat()}.tif")


def write_daily_cog(cfg: Config, key: str, day: dt.date, win: grid.Window, out, sources_present: list[str]) -> Path:
    bands = [
        out.p_flood,
        out.abs_logodds,
        np.where(out.n_sources == 255, np.nan, out.n_sources).astype("float32"),
        np.where(np.isnan(out.p_flood), np.nan, out.days_since_obs.astype("float32")),
    ]
    path = daily_cog_path(cfg, key, day)
    _write_cog(path, bands, DAILY_BANDS, win, "float32", np.nan,
               tags={"valid_time": day.isoformat(), "version": cfg["version"], "sources_present": ",".join(sources_present),
                     "days_since_obs_never": 65535})
    return path


def upload_tree(cfg: Config, local_dir: Path, blob_subdir: str, overwrite: bool = False) -> int:
    """Mirror every .tif/.csv/.json/.parquet under `local_dir` to the processed blob prefix. Returns files uploaded."""
    import ocha_stratus as stratus

    b = cfg["paths"]["blob"]
    cc = stratus.get_container_client(container_name=b["container"], stage=b["stage"], write=True)
    existing = {x.name for x in cc.list_blobs(name_starts_with=cfg.blob_path("processed", blob_subdir))} if not overwrite else set()
    n = 0
    for p in sorted(local_dir.rglob("*")):
        if p.is_dir() or p.suffix not in (".tif", ".csv", ".json", ".parquet", ".png", ".md") or ".part" in p.name:
            continue
        name = cfg.blob_path("processed", blob_subdir, *p.relative_to(local_dir).parts)
        if name in existing:
            continue
        with open(p, "rb") as f:
            cc.upload_blob(name, f, overwrite=True)
        n += 1
    return n

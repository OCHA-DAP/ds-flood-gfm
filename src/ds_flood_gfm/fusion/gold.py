"""Gold labels: Copernicus EMS and UNOSAT flood maps on the 0.01-degree grid.

Label sets come from the gold corpora built in ds-geospatial-impact-estimates
(dev account, container `global`). One label set = one mapped AOI on one
acquisition day. An *event* is one activation code (EMSR... or UNOSAT code);
its windows are clusters of its label-set footprints.

Truth per cell and gold day (see docs/decisions.md D-10..D-14):
  valid  iff >= valid_min_frac of the cell is inside the label set's valid mask
  flood  iff flooded fraction of the valid area >= flood_threshold
  unlabelled iff "flood possible" fraction >= possible_exclude_frac
Truth on dry-window days: every valid cell of the event footprint is dry.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from rasterio.features import rasterize

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import Config

FLOOD, DRY, NOTRUTH = 1, 0, 255

SAR_SENSOR_TOKENS = ("sentinel-1", "cosmo", "radarsat", "paz", "terrasar", "tandem", "iceye", "capella", "risat", "alos", "saocom", "rcm", "kompsat-5", "sar")


def _container(cfg: Config):
    import ocha_stratus as stratus

    g = cfg["gold"]
    return stratus.get_container_client(stage=g["blob_stage"], container_name=g["blob_container"])


def _read_parquet_blob(cc, name: str) -> pd.DataFrame:
    return pq.read_table(io.BytesIO(cc.get_blob_client(name).download_blob().readall())).to_pandas()


def _sensor_class_cems(sensor: str | None) -> str:
    if sensor is None or (isinstance(sensor, float) and np.isnan(sensor)):
        return "unknown"
    s = str(sensor).lower()
    if any(t in s for t in SAR_SENSOR_TOKENS):
        return "sar"
    return "optical"


def _is_s1(sensor) -> bool:
    return sensor is not None and not (isinstance(sensor, float) and np.isnan(sensor)) and "sentinel-1" in str(sensor).lower()


def load_label_index(cfg: Config) -> pd.DataFrame:
    """Both gold indexes in one table with a common schema (one row per label set)."""
    cc = _container(cfg)
    g = cfg["gold"]
    ce = _read_parquet_blob(cc, g["cems_index"])
    ce = pd.DataFrame(
        {
            "source": "cems",
            "code": ce.code,
            "aoi": ce.aoi,
            "label_day": pd.to_datetime(ce.label_day, errors="coerce"),
            "acq_start": pd.to_datetime(ce.acq_start, errors="coerce"),
            "acq_precision": ce.acq_precision,
            "sensor": ce.sensor,
            "sensor_class": [_sensor_class_cems(s) for s in ce.sensor],
            "product_classes": "flood",
            "valid_basis": ce.valid_basis,
            "countries": ce.countries,
            "minx": ce.minx, "miny": ce.miny, "maxx": ce.maxx, "maxy": ce.maxy,
        }
    )
    un = _read_parquet_blob(cc, g["unosat_index"])
    un = pd.DataFrame(
        {
            "source": "unosat",
            "code": un.code,
            "aoi": un.aoi.fillna(""),
            "label_day": pd.to_datetime(un.label_day, errors="coerce"),
            "acq_start": pd.to_datetime(un.acq_start, errors="coerce"),
            "acq_precision": un.acq_precision,
            "sensor": un.sensor,
            "sensor_class": un.sensor_class.fillna("unknown"),
            "product_classes": un.product_classes.fillna(""),
            "valid_basis": un.valid_basis,
            "countries": un.countries,
            "minx": un.minx, "miny": un.miny, "maxx": un.maxx, "maxy": un.maxy,
        }
    )
    df = pd.concat([ce, un], ignore_index=True)
    df["is_s1"] = [_is_s1(s) for s in df.sensor]
    return df


def select_label_sets(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the v1 gold filters. Returns (kept, excluded-with-reason); nothing is dropped silently."""
    g = cfg["gold"]
    reasons = pd.Series("", index=df.index)

    def flag(mask, why):
        nonlocal reasons
        reasons = reasons.where(~(mask & (reasons == "")), why)

    flag(df.label_day.isna(), "no label_day")
    flag(df.label_day.dt.year < g["min_year"], f"before {g['min_year']} (no GFM archive)")
    flag(~df.acq_precision.isin(g["acq_precision"]), "acquisition is a window, not a day")
    flag(df.valid_basis.isin(["none"]) | df.valid_basis.isna(), "no valid mask (dry cells unknowable)")
    flag(df.sensor_class.isin(g["exclude_sensor_class"]), "coarse optical label (candidate input, not gold)")
    flag(~df.product_classes.str.contains("flood", na=False) | (df.product_classes.str.strip() == "flood_possible"), "no flood layer")
    flag((df.maxx <= df.minx) | (df.maxy <= df.miny), "degenerate bbox")
    kept = df[reasons == ""].copy()
    excluded = df[reasons != ""].assign(reason=reasons[reasons != ""])
    return kept, excluded


def label_partition_sizes(cfg: Config) -> pd.Series:
    """Bytes of each code's label partition (cost proxy for ordering and the v1 cap)."""
    cc = _container(cfg)
    sizes: dict[str, int] = {}
    for prefix in ("copernicus_ems/flood/gold/labels/", "unosat/gold/labels/"):
        for b in cc.list_blobs(name_starts_with=prefix):
            if b.size and "code=" in b.name:
                code = b.name.split("code=")[1].split("/")[0]
                sizes[code] = sizes.get(code, 0) + b.size
    return pd.Series(sizes, name="label_bytes")


# --------------------------------------------------------------------------- continents

def continent_lookup(cfg: Config):
    """Return f(lon, lat) -> continent using Natural Earth 1:50m countries (nearest country offshore)."""
    import geopandas as gpd

    path = cfg.local("raw", "natural_earth", "ne_50m_admin_0_countries.zip")
    if not path.exists():
        import urllib.request

        urllib.request.urlretrieve("https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip", path)
    ne = gpd.read_file(path)[["CONTINENT", "geometry"]]
    ne = ne[ne.CONTINENT != "Seven seas (open ocean)"]

    def f(lon: float, lat: float) -> str:
        p = gpd.GeoDataFrame(geometry=[shapely.Point(lon, lat)], crs="EPSG:4326")
        j = gpd.sjoin_nearest(p.to_crs(8857), ne.to_crs(8857), how="left")
        return str(j.CONTINENT.iloc[0])

    return f


# --------------------------------------------------------------------------- events and windows

@dataclass(frozen=True)
class EventWindow:
    code: str
    idx: int
    window: grid.Window

    @property
    def key(self) -> str:
        return f"{self.code}_w{self.idx}"


def _merge_boxes(boxes: list[tuple[float, float, float, float]], buffer: float) -> list[tuple[float, float, float, float]]:
    """Merge overlapping (buffered) boxes until stable."""
    out = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        merged: list[list[float]] = []
        for b in out:
            for m in merged:
                if b[0] - buffer < m[2] and m[0] - buffer < b[2] and b[1] - buffer < m[3] and m[1] - buffer < b[3]:
                    m[0], m[1], m[2], m[3] = min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])
                    changed = True
                    break
            else:
                merged.append(list(b))
        out = merged
    return [tuple(m) for m in out]


def event_table(sets: pd.DataFrame, cfg: Config, sizes: pd.Series) -> pd.DataFrame:
    """One row per event code with dates, windows, continent and cost proxies."""
    cont = continent_lookup(cfg)
    g = cfg["gold"]
    rows = []
    for code, s in sets.groupby("code"):
        boxes = list(zip(s.minx, s.miny, s.maxx, s.maxy))
        wins = [grid.window_from_bounds(*b) for b in _merge_boxes(boxes, buffer=0.05)]
        start, end = s.label_day.min().normalize(), s.label_day.max().normalize()
        cx = float((s.minx.min() + s.maxx.max()) / 2)
        cy = float((s.miny.min() + s.maxy.max()) / 2)
        rows.append(
            {
                "code": code,
                "source": s.source.iloc[0],
                "n_sets": len(s),
                "n_days": s.label_day.dt.normalize().nunique(),
                "start": start,
                "end": end,
                "window_start": start - pd.Timedelta(days=g["dry_pre_days"]),
                "window_end": end + pd.Timedelta(days=g["dry_post_gap_days"] + g["dry_post_days"]),
                "lon": cx,
                "lat": cy,
                "continent": cont(cx, cy),
                "n_windows": len(wins),
                "cells": int(sum(w.shape[0] * w.shape[1] for w in wins)),
                "windows": json.dumps([[w.row0, w.row1, w.col0, w.col1] for w in wins]),
                "frac_s1_sets": float(s.is_s1.mean()),
                "label_bytes": int(sizes.get(code, 0)),
            }
        )
    return pd.DataFrame(rows)


def event_windows(row: pd.Series) -> list[EventWindow]:
    return [EventWindow(row.code, i, grid.Window(*w)) for i, w in enumerate(json.loads(row.windows))]


# --------------------------------------------------------------------------- rasterisation

def _polygonal(wkb) -> shapely.Geometry | None:
    """2-D polygonal part of a WKB geometry (CEMS gold v1 has 3-D coordinates and line fragments)."""
    if wkb is None:
        return None
    geom = shapely.force_2d(shapely.from_wkb(wkb))
    if geom.is_empty:
        return None
    parts = shapely.get_parts(geom)
    polys = [p for p in parts if p.geom_type in ("Polygon", "MultiPolygon")]
    nested = [q for p in parts if p.geom_type == "GeometryCollection" for q in shapely.get_parts(p) if q.geom_type in ("Polygon", "MultiPolygon")]
    polys += nested
    if not polys:
        return None
    return shapely.union_all(polys) if len(polys) > 1 else polys[0]


def burn_fraction(geom: shapely.Geometry | None, win: grid.Window, burn_res: float) -> np.ndarray:
    """Fraction of each 0.01-degree cell covered by `geom`, from a burn at `burn_res` (block-sum)."""
    out = np.zeros(win.shape, "float32")
    if geom is None:
        return out
    k = round(grid.RES / burn_res)
    # rasterise one-degree chunks to bound memory on large windows
    for sub in grid.tiles_for_window(win, 1):
        minx, miny, maxx, maxy = sub.bounds
        clip = shapely.clip_by_rect(geom, minx, miny, maxx, maxy)
        if clip.is_empty:
            continue
        h, w = sub.shape[0] * k, sub.shape[1] * k
        from affine import Affine

        t = Affine(burn_res, 0, minx, 0, -burn_res, maxy)
        fine = rasterize([(clip, 1)], out_shape=(h, w), transform=t, fill=0, dtype="uint8", all_touched=False)
        frac = fine.reshape(sub.shape[0], k, sub.shape[1], k).mean(axis=(1, 3))
        rs, cs = win.slices_of(sub)
        out[rs, cs] = frac
    return out


def read_label_rows(cfg: Config, source: str, code: str) -> pd.DataFrame:
    cc = _container(cfg)
    g = cfg["gold"]
    if source == "cems":
        df = _read_parquet_blob(cc, g["cems_labels"].format(code=code))
        return pd.DataFrame(
            {
                "aoi": df.aoi,
                "label_day": pd.to_datetime(df.label_day),
                "flood": df.geometry,
                "valid": df.valid_geometry,
                "possible": None,
            }
        )
    df = _read_parquet_blob(cc, f"{g['unosat_labels_prefix']}labels/code={code}/data.parquet")
    return pd.DataFrame(
        {
            "aoi": df.aoi.fillna(""),
            "label_day": pd.to_datetime(df.label_day),
            "flood": df.geom_flood,
            "valid": df.geom_valid,
            "possible": df.geom_possible,
        }
    )


def truth_for_window(cfg: Config, ew: EventWindow, label_rows: pd.DataFrame, kept_sets: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Gold truth per gold day on one event window.

    Returns (days, truth[day, row, col] in {1, 0, 255}, footprint[row, col]).
    Several label sets on the same day are combined: flood wins over dry.
    """
    g = cfg["gold"]
    keep = kept_sets[kept_sets.code == ew.code][["aoi", "label_day"]].assign(label_day=lambda d: d.label_day.dt.normalize())
    rows = label_rows.assign(label_day=label_rows.label_day.dt.normalize()).merge(keep.drop_duplicates(), on=["aoi", "label_day"], how="inner")
    days = pd.DatetimeIndex(sorted(rows.label_day.unique()))
    truth = np.full((len(days), *ew.window.shape), NOTRUTH, "uint8")
    footprint = np.zeros(ew.window.shape, bool)
    for i, day in enumerate(days):
        for _, r in rows[rows.label_day == day].iterrows():
            valid = burn_fraction(_polygonal(r.valid), ew.window, g["burn_res_deg"])
            vmask = valid >= g["valid_min_frac"]
            if not vmask.any():
                continue
            flood = burn_fraction(_polygonal(r.flood), ew.window, g["burn_res_deg"])
            with np.errstate(invalid="ignore", divide="ignore"):
                flood_of_valid = np.where(valid > 0, np.minimum(flood, valid) / valid, 0.0)
            possible = burn_fraction(_polygonal(r.possible), ew.window, g["burn_res_deg"]) if r.possible is not None else np.zeros_like(valid)
            t = truth[i]
            is_flood = vmask & (flood_of_valid >= g["flood_threshold"])
            is_dry = vmask & ~is_flood & (possible < g["possible_exclude_frac"])
            t[is_flood] = FLOOD
            t[is_dry & (t != FLOOD)] = DRY
            footprint |= vmask
    return days, truth, footprint


def corrected_dry_windows(ds) -> tuple[np.ndarray, np.ndarray]:
    """D-39 variant of an event cube's truth: a cell flooded on any gold day is NOTRUTH (not dry) in the
    dry windows; gold days and every other footprint cell are unchanged. Returns (truth, gold-day mask)."""
    days = pd.to_datetime(ds.day.values)
    gd = (days >= pd.Timestamp(ds.attrs["event_start"])) & (days <= pd.Timestamp(ds.attrs["event_end"]))
    t = ds.truth.values.copy()
    ever = (t[gd] == FLOOD).any(axis=0)
    t[~gd] = np.where(ever[None, :, :], NOTRUTH, t[~gd])
    return t, np.asarray(gd)

"""An independent referee for ordinary days: Sentinel-2 surface water on the random sample (2026-10-26).

Why: every ordinary-day precision so far is judged by GFM, which misses 86% of flood. Sentinel-2's Level-2A
scene classification (SCL) marks water at 20 m from optical imagery, independent of all four inputs
(GFM is radar; VIIRS is a different optical sensor and product). Source: Element84 Earth Search STAC,
collection sentinel-2-l2a, SCL COGs on AWS, no login.

Rules, fixed before running:
  - population: the 30 in-scope random-sample blocks, every day of 2023-2025 with a Sentinel-2 scene whose
    scene-level cloud cover is under 20%; every scene tile touching the block that day is read and the valid
    pixels combined;
  - per 0.01-degree cell, the same rules as every other layer: observed if at least 50% of its 20 m pixels are
    valid (SCL not 0 no data, 1 saturated, 3 cloud shadow, 8/9 cloud, 10 cirrus); water if at least 30% of the
    observed pixels are SCL 6 (water); snow/ice (11) counts as not water;
  - two referee labels per observed cell-day: "water" (as above) and "unusual water" = water on a cell that is
    not normally wet in that calendar month (the D-38 JRC rule), the closest optical proxy for flood;
  - products on the same cell-days: fusion (rates fitted on all 170 events, as in cv_ordinary_days), fusion
    with FloodScan alone, the cascade, and each single product; per product: precision and recall against
    water, and against unusual water, all cells (no data = not flagged), by land type;
  - caveats stated up front: cloud-free days are drier than average (floods come with cloud), so this
    measures ordinary-day behaviour, not flood-day skill; water under trees is invisible to SCL; a 5-day
    revisit means most days are unlabelled.

Outputs: results/experiment_s2_referee.csv (+ per-block rows in data/fusion/metrics/). Resumable per block-day.
Run: nice -n 19 uv run python -u scripts/fusion/experiment_s2_referee.py [--blocks N] [--max-days N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr
from rasterio.warp import reproject, Resampling, transform_bounds

from ds_flood_gfm.fusion import baseline, crossval, engine, grid
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.validate import SINGLE

STAC = "https://earth-search.aws.element84.com/v1/search"
CASCADE = "cascade: VIIRS, else GFM, else microwave"
FS_ONLY = "fusion, FloodScan alone"
INVALID = {0, 1, 3, 8, 9, 10}
WATER = 6
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3")


def scenes(bbox, start, end) -> pd.DataFrame:
    items, token = [], None
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200,
            "query": {"eo:cloud_cover": {"lt": 20}}, "fields": {"include": ["id", "properties.datetime", "assets.scl.href"]}}
    while True:
        r = requests.post(STAC, json=body | ({"token": token} if token else {}), timeout=120)
        r.raise_for_status()
        j = r.json()
        items += [{"day": i["properties"]["datetime"][:10], "scl": i["assets"]["scl"]["href"]} for i in j["features"]]
        nxt = [lk for lk in j.get("links", []) if lk.get("rel") == "next"]
        if not nxt:
            break
        token = nxt[0]["body"]["token"] if "body" in nxt[0] else None
        if token is None:
            break
    return pd.DataFrame(items)


def read_scl_on_grid(hrefs, win: grid.Window) -> tuple[np.ndarray, np.ndarray]:
    """Per cell: share of 20 m pixels valid, and share of valid pixels that are water, from every scene that day."""
    minx, miny, maxx, maxy = win.bounds
    rows, cols = win.shape
    fine = 10  # 0.001 degree sub-pixels (~110 m): fine enough for 50%/30% rules, cheap to warp
    valid_any = np.zeros((rows * fine, cols * fine), bool)
    water_any = np.zeros_like(valid_any)
    dst_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, cols * fine, rows * fine)
    with rasterio.Env(**GDAL_ENV):
        for href in hrefs:
            with rasterio.open(href) as src:
                b = transform_bounds("EPSG:4326", src.crs, minx, miny, maxx, maxy)
                w = rasterio.windows.from_bounds(*b, src.transform).round_offsets().round_lengths()
                if w.width <= 0 or w.height <= 0:
                    continue
                a = src.read(1, window=w, boundless=True, fill_value=0)
                t = src.window_transform(w)
                out = np.zeros((rows * fine, cols * fine), "uint8")
                reproject(a, out, src_transform=t, src_crs=src.crs, dst_transform=dst_transform, dst_crs="EPSG:4326", resampling=Resampling.nearest, dst_nodata=0)
            v = ~np.isin(out, list(INVALID))
            valid_any |= v
            water_any |= v & (out == WATER)
    vs = valid_any.reshape(rows, fine, cols, fine).mean(axis=(1, 3))
    ws = water_any.reshape(rows, fine, cols, fine).sum(axis=(1, 3)) / np.maximum(valid_any.reshape(rows, fine, cols, fine).sum(axis=(1, 3)), 1)
    return vs, ws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=None, help="first N blocks (smoke test)")
    ap.add_argument("--max-days", type=int, default=None, help="first N scene-days per block (smoke test)")
    a = ap.parse_args()
    cfg = load_config()
    f, thr, rule = cfg["fusion"], cfg["fusion"]["prob_threshold"], cfg["gold"]["dry_window_rule"]
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = counts[counts.rule == rule].drop(columns="rule")
    comp = pd.read_parquet(cfg.local("metrics", "cv_fit_counts_components.parquet"))
    prior = pd.read_csv(cfg.results_dir / "cv" / "prior.csv")
    clip = tuple(f["sens_spec_clip"])
    fs_srcs = ["gfm", "floodscan", "viirs"]
    m_all = engine.Model.from_tables(crossval.fit_table(counts, folds.code), prior, MODEL_SOURCES, f["decay"], clip)
    m_fs = engine.Model.from_tables(crossval.fit_table(pd.concat([counts, comp[comp.source == "floodscan"]]), folds.code, sources=fs_srcs), prior, fs_srcs, f["decay"], clip)
    blocks = pd.read_parquet(cfg.local("sample", "blocks.parquet"))
    blocks = blocks[blocks.continent.isin(cfg["scope"]["continents"])]
    if a.blocks:
        blocks = blocks.head(a.blocks)
    part = cfg.local("metrics", "s2_referee_rows.parquet")
    rows = pd.read_parquet(part).to_dict("records") if part.exists() else []
    done = {(r["block_id"], r["day"]) for r in rows}
    t0 = time.time()
    for _, b in blocks.iterrows():
        ds = xr.load_dataset(cfg.local("cube", "sample", f"{b.block_id}.nc"), engine="netcdf4")
        win = grid.Window(int(b.row0), int(b.row1), int(b.col0), int(b.col1))
        dom = ds.domain.values.astype(bool)
        if not dom.any():
            continue
        days = pd.to_datetime(ds.day.values)
        lt = crossval.landtype_codes(ds.stratum.values)
        v = {s: ds[s].values for s in SINGLE}
        p = engine.run_sequence({s: v[s] for s in MODEL_SOURCES}, ds.stratum.values, dom, m_all)["p_flood"]
        pf = engine.run_sequence({s: v[s] for s in fs_srcs}, ds.stratum.values, dom, m_fs)["p_flood"]
        cas = np.full(v["viirs"].shape, NODATA, "uint8")
        for s in ["microwave", "gfm", "viirs"]:
            cas = np.where(v[s] != NODATA, v[s], cas)
        calls = {"fusion": np.isfinite(p) & (p >= thr), FS_ONLY: np.isfinite(pf) & (pf >= thr), CASCADE: cas == 1} | {s: v[s] == 1 for s in SINGLE}
        minx, miny, maxx, maxy = win.bounds
        sc = scenes([minx, miny, maxx, maxy], days[0].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d"))
        if a.max_days:
            sc = sc[sc.day.isin(sorted(sc.day.unique())[: a.max_days])]
        nw = {m: baseline.normally_wet(cfg, win, m) for m in range(1, 13)}
        for day, g in sc.groupby("day"):
            if (b.block_id, day) in done:
                continue
            i = int(np.where(days == pd.Timestamp(day))[0][0]) if (days == pd.Timestamp(day)).any() else -1
            if i < 0:
                continue
            vs, ws = read_scl_on_grid(g.scl.tolist(), win)
            obs = (vs >= 0.5) & dom & (lt[i] != NODATA)
            water = obs & (ws >= 0.30)
            unusual = water & ~nw[pd.Timestamp(day).month]
            for k, ltn in crossval.LANDTYPES.items():
                m = obs & (lt[i] == k)
                if not m.any():
                    continue
                for name, c in calls.items():
                    cc = c[i] & m
                    rows.append({"block_id": b.block_id, "continent": b.continent, "day": day, "landtype": ltn, "product": name,
                                 "observed": int(m.sum()), "water": int((water & m).sum()), "unusual": int((unusual & m).sum()),
                                 "calls": int(cc.sum()), "calls_water": int((cc & water).sum()), "calls_unusual": int((cc & unusual).sum())})
            pd.DataFrame(rows).to_parquet(part)
        print(f"{b.block_id}: {sc.day.nunique()} scene-days, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = []
    for by in ([], ["landtype"], ["continent"]):
        g = df.groupby(["product", *by], as_index=False)[["observed", "water", "unusual", "calls", "calls_water", "calls_unusual"]].sum()
        g["call_rate"], g["water_rate"], g["unusual_rate"] = g.calls / g.observed, g.water / g.observed, g.unusual / g.observed
        g["precision_vs_water"], g["recall_vs_water"] = g.calls_water / g.calls, g.calls_water / g.water
        g["precision_vs_unusual_water"], g["recall_vs_unusual_water"] = g.calls_unusual / g.calls, g.calls_unusual / g.unusual
        g["view"] = ",".join(by) if by else "overall"
        out.append(g)
    out = pd.concat(out, ignore_index=True)
    out.to_csv(cfg.results_dir / "experiment_s2_referee.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"labelled cell-days: {df[df['product'] == 'fusion'].observed.sum():,} on {df.day.nunique()} scene-days, {df.block_id.nunique()} blocks")
    print(out[out.view == "overall"][["product", "call_rate", "water_rate", "unusual_rate", "precision_vs_water", "recall_vs_water", "precision_vs_unusual_water", "recall_vs_unusual_water"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

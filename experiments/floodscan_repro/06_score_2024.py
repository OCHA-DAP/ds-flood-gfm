"""Step 1c: score every 2024 flooded-fraction variant (05) against FloodScan SFED area, West Africa box and Nigeria LGAs.

FloodScan: the fusion work's local cache of the Africa daily COGs (band SFED, 1/12 deg, origin -30E 40N),
main checkout data/fusion/raw/floodscan/africa_daily/. Our box 0-20E, 0-20N is rows 240-480, cols 360-600,
which nests 3x3 into our 0.25-deg cells.
SFED area is a fraction of unmasked (non-water) area, so ours is compared as ff / (1 - f_pow)
(cells with f_pow > 0.5 dropped, as in 04-05).

Cell-level metrics use one common population: cell-days valid in every variant (the flash filter
turns cells into NaN, so per-variant populations would differ). Per variant:
  ratio    total ours / total SFED
  never    share of our total in cells SFED never floods (max < 0.02)
  r_pool   pooled r over 0.25-deg cell-days
  r_cell   median per-cell temporal r, cells with SFED max > 0.2
  r_blk    median per-block temporal r, 0.5-deg blocks with SFED max > 0.2
  r_lga1/2 median LGA temporal r, SFED seasonal max 0.10-0.20 / > 0.20 (small n: see counts)
  peak     median ours / SFED at each LGA's SFED peak, LGAs with SFED max > 0.10
  peak_mm  median max(ours) / max(SFED) over the window, same LGAs (height regardless of day)
  lag      median (day of our max - day of SFED max), same LGAs; positive = ours later

    uv run python experiments/floodscan_repro/06_score_2024.py
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import ocha_stratus as stratus
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window

FS = Path("/Users/zackarno/Documents/CHD/repos/ds-flood-gfm/data/fusion/raw/floodscan/africa_daily")  # main checkout cache
FF = Path("data/floodscan_repro/ff_2024.npz")
OUT = Path("results/floodscan_repro")
if len(sys.argv) > 1:  # e.g. an in-sample code test: `... 06_score_2024.py /tmp/ff.npz /tmp/results_dir`
    FF, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = np.load(FF)
days = [dt.date.fromisoformat(s) for s in z["days"]]
fpow = z["fpow"]
VARIANTS = [k for k in z.files if k not in ("days", "fpow", "focean")]

sf_native = []
for d in days:
    p = FS / f"aer_area_300s_v{d:%Y-%m-%d}_v05r01.tif"
    if not p.exists():
        raise FileNotFoundError(f"FloodScan cache lacks {p.name}")
    with rasterio.open(p) as ds:
        if not (np.isclose(ds.transform.c, -30) and np.isclose(ds.transform.f, 40) and np.isclose(ds.transform.a, 1 / 12)):
            raise ValueError(f"{p.name}: unexpected grid {ds.transform}")
        sf_native.append(ds.read(list(ds.descriptions).index("SFED") + 1, window=Window(360, 240, 240, 240)).astype("float64"))
SFN = np.stack(sf_native)
SF = SFN.reshape(len(days), 80, 3, 80, 3).mean((2, 4))

ours = {}
for v in VARIANTS:
    with np.errstate(invalid="ignore", divide="ignore"):
        ours[v] = np.where(fpow <= 0.5, z[v] / (1 - fpow), np.nan)
common = np.isfinite(SF) & np.logical_and.reduce([np.isfinite(o) for o in ours.values()])
smax = np.nanmax(np.where(common, SF, np.nan), 0)
print(f"FloodScan {days[0]}..{days[-1]}; common cell-days {common.sum():,} of {SF.size:,} "
      f"({common.mean():.2f}); variants: {', '.join(VARIANTS)}")


def temporal_r(a, b, sel, min_n=60):
    rs = []
    for i, j in zip(*np.where(sel)):
        x, y = a[:, i, j], b[:, i, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= min_n and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
    return (np.median(rs) if rs else np.nan), len(rs)


adm2 = stratus.codab.load_codab_from_blob("nga", admin_level=2)
m_ours = [~geometry_mask([g], (80, 80), Affine(0.25, 0, 0, 0, -0.25, 20), all_touched=True) for g in adm2.geometry]
m_sf = [~geometry_mask([g], (240, 240), Affine(1 / 12, 0, 0, 0, -1 / 12, 20), all_touched=True) for g in adm2.geometry]


def lga(x, masks):
    return np.stack([[np.nanmean(x[t][m]) if np.isfinite(x[t][m]).any() else np.nan for m in masks] for t in range(len(x))])


sf_l = lga(SFN, m_sf)
smax_l = np.nanmax(sf_l, 0)
summary, rows = [], []
for v in VARIANTS:
    o = np.where(common, ours[v], np.nan)
    s = np.where(common, SF, np.nan)
    ok = np.isfinite(o)
    rc, nc = temporal_r(o, s, smax > 0.2)
    ob = np.nanmean(o.reshape(len(days), 40, 2, 40, 2), (2, 4))
    sb = np.nanmean(s.reshape(len(days), 40, 2, 40, 2), (2, 4))
    rb, nb = temporal_r(ob, sb, np.nanmax(sb, 0) > 0.2)
    o_l = lga(ours[v], m_ours)
    lr = []
    for j, pc in enumerate(adm2["ADM2_PCODE"]):
        x, y = o_l[:, j], sf_l[:, j]
        okl = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[okl], y[okl])[0, 1] if okl.sum() >= 60 and np.std(y[okl]) > 0 and np.std(x[okl]) > 0 else np.nan
        pk = int(np.nanargmax(y)) if np.isfinite(y).any() else None
        ratio = x[pk] / y[pk] if pk is not None and y[pk] > 0 else np.nan
        mm = np.nanmax(x) / y[pk] if pk is not None and y[pk] > 0 and np.isfinite(x).any() else np.nan
        lag = int(np.nanargmax(x)) - pk if pk is not None and np.isfinite(x).any() else np.nan
        lr.append((r, ratio, mm, lag))
        rows.append({"variant": v, "lga": adm2["ADM2_EN"].iloc[j], "pcode": pc, "state": adm2["ADM1_EN"].iloc[j],
                     "sfed_max": smax_l[j], "r": r, "peak_ratio": ratio, "max_over_max": mm, "lag_days": lag})
    lr = np.array(lr)
    b1 = (smax_l >= 0.10) & (smax_l < 0.20)
    b2 = smax_l >= 0.20
    own = np.isfinite(ours[v]) & np.isfinite(SF)  # the variant's own population (rain flags shrink 'common')
    summary.append({
        "variant": v,
        "ratio": o[ok].sum() / s[ok].sum(),
        "never": np.nansum(np.where((smax < 0.02)[None], o, 0)) / np.nansum(o),
        "ratio_own": ours[v][own].sum() / SF[own].sum(),
        "never_own": np.nansum(np.where((smax < 0.02)[None] & own, ours[v], 0)) / np.nansum(np.where(own, ours[v], 0)),
        "r_pool": np.corrcoef(o[ok], s[ok])[0, 1],
        "r_cell": rc, "r_blk": rb,
        "r_lga1": np.nanmedian(lr[b1, 0]), "r_lga2": np.nanmedian(lr[b2, 0]),
        "peak": np.nanmedian(lr[smax_l >= 0.10, 1]),
        "peak_mm": np.nanmedian(lr[smax_l >= 0.10, 2]),
        "lag": np.nanmedian(lr[smax_l >= 0.10, 3]),
    })
tab = pd.DataFrame(summary).set_index("variant")
print(f"\ncounts: cells with SFED max > 0.2: {nc}; 0.5-deg blocks: {nb}; LGAs 0.10-0.20: {int(b1.sum())}, > 0.20: {int(b2.sum())}")
print(tab.round(3).to_string())

# Inland cells only (no ocean in the footprint): the coast holds most of our false mass, and FloodScan's
# users guide says floods "within 5-10 km of coastlines" are usually not depicted.
inland = z["focean"] <= 0.001
summary_in = []
for v in VARIANTS:
    o = np.where(common & inland[None], ours[v], np.nan)
    s = np.where(common & inland[None], SF, np.nan)
    ok = np.isfinite(o)
    rc, nci = temporal_r(o, s, (smax > 0.2) & inland)
    summary_in.append({"variant": v, "ratio": o[ok].sum() / s[ok].sum(),
                       "never": np.nansum(np.where((smax < 0.02)[None], o, 0)) / np.nansum(o),
                       "r_pool": np.corrcoef(o[ok], s[ok])[0, 1], "r_cell": rc})
tab_in = pd.DataFrame(summary_in).set_index("variant")
print(f"\ninland cells only ({inland.mean():.2f} of cells; cells with SFED max > 0.2: {nci}):")
print(tab_in.round(3).to_string())
OUT.mkdir(parents=True, exist_ok=True)
tab.round(4).to_csv(OUT / "nga2024_variant_summary.csv")
tab_in.round(4).to_csv(OUT / "nga2024_variant_summary_inland.csv")
pd.DataFrame(rows).to_csv(OUT / "nga2024_lga_by_variant.csv", index=False)
print(f"wrote {OUT / 'nga2024_variant_summary.csv'} and {OUT / 'nga2024_lga_by_variant.csv'}")

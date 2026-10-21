"""Step 1c: score the 2024 flooded fraction (05) against FloodScan SFED area, West Africa box and Nigeria LGAs.

FloodScan: the fusion work's local cache of the Africa daily COGs (band SFED, 1/12 deg, origin -30E 40N),
main checkout data/fusion/raw/floodscan/africa_daily/. Our box 0-20E, 0-20N is rows 240-480, cols 360-600,
which nests 3x3 into our 0.25-deg cells.
SFED area is a fraction of unmasked (non-water) area, so ours is compared as ff / (1 - f_pow)
(cells with f_pow > 0.5 dropped, as in 04-05).

Per variant (full, noflash, placeholder):
- 0.25-deg cell-days: pooled r, orthogonal slope, mean ours / mean SFED; per-cell temporal r for cells
  with SFED max > 0.05 and > 0.2; share of our total where SFED never floods (max < 0.02); monthly means
- 0.5-deg blocks: pooled r; per-block temporal r where SFED max > 0.2
- Nigeria LGAs (each product on its own grid, all_touched): median r by SFED seasonal-max band and
  median ours / SFED at each LGA's SFED peak; the ten LGAs with the largest SFED peak.

    uv run python experiments/floodscan_repro/06_score_2024.py
"""

import datetime as dt
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
BANDS = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9)]
VARIANTS = ("full", "noflash", "placeholder")

z = np.load(FF)
days = [dt.date.fromisoformat(s) for s in z["days"]]
fpow = z["fpow"]

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
print(f"FloodScan: {len(days)} days ({days[0]}..{days[-1]}), box mean SFED {SF.mean():.4f}")


def orth_slope(x, y):
    _, v = np.linalg.eigh(np.cov(np.vstack([x - x.mean(), y - y.mean()])))
    return v[1, 1] / v[0, 1]


def temporal_r(a, b, sel, min_n=60):
    rs = []
    for i, j in zip(*np.where(sel)):
        x, y = a[:, i, j], b[:, i, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= min_n and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
    return np.median(rs) if rs else np.nan, len(rs)


adm2 = stratus.codab.load_codab_from_blob("nga", admin_level=2)
aff_ours = Affine(0.25, 0, 0, 0, -0.25, 20)
aff_sf = Affine(1 / 12, 0, 0, 0, -1 / 12, 20)
m_ours = [~geometry_mask([g], (80, 80), aff_ours, all_touched=True) for g in adm2.geometry]
m_sf = [~geometry_mask([g], (240, 240), aff_sf, all_touched=True) for g in adm2.geometry]


def lga(x, masks):
    return np.stack([[np.nanmean(x[t][m]) if np.isfinite(x[t][m]).any() else np.nan for m in masks] for t in range(len(x))])


sf_l = lga(SFN, m_sf)
smax_l = np.nanmax(sf_l, 0)
rows = []
for var in VARIANTS:
    with np.errstate(invalid="ignore", divide="ignore"):
        ours = np.where(fpow <= 0.5, z[var] / (1 - fpow), np.nan)
    both = np.isfinite(ours) & np.isfinite(SF)
    o1, s1 = ours[both], SF[both]
    smax = np.nanmax(SF, 0)
    print(f"\n===== {var} =====")
    print(f"0.25-deg cell-days (n {both.sum():,}): r {np.corrcoef(o1, s1)[0, 1]:.3f}, orthogonal slope {orth_slope(s1, o1):.2f}, "
          f"mean ours / SFED {o1.mean() / s1.mean():.2f}")
    for thr in (0.05, 0.2):
        r, n = temporal_r(ours, SF, smax > thr)
        print(f"   per-cell temporal r, SFED max > {thr}: median {r:.3f} (n {n})")
    never = smax < 0.02
    print(f"   share of our total where SFED never floods (max < 0.02, {never.mean():.2f} of cells): "
          f"{np.nansum(np.where(both & never[None], ours, 0)) / np.nansum(np.where(both, ours, 0)):.2f}")
    print("   monthly mean ours / SFED: " + ", ".join(
        f"{m}: {np.nanmean(np.where(both[t], ours[t], np.nan)):.4f}/{np.nanmean(SF[t]):.4f}"
        for m in (8, 9, 10, 11) for t in [np.array([d.month == m for d in days])]))
    ob = ours.reshape(len(days), 40, 2, 40, 2)
    ob = np.where(np.isfinite(ob).sum((2, 4)) >= 2, np.nanmean(ob, (2, 4)), np.nan)
    sb = SF.reshape(len(days), 40, 2, 40, 2).mean((2, 4))
    bb = np.isfinite(ob)
    r, n = temporal_r(ob, sb, np.nanmax(sb, 0) > 0.2)
    print(f"0.5-deg blocks: pooled r {np.corrcoef(ob[bb], sb[bb])[0, 1]:.3f}; per-block temporal r (SFED max > 0.2) median {r:.3f} (n {n})")

    o_l = lga(ours, m_ours)
    for j, (name, pc) in enumerate(zip(adm2["ADM2_EN"], adm2["ADM2_PCODE"])):
        x, y = o_l[:, j], sf_l[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() >= 60 and np.std(y[ok]) > 0 and np.std(x[ok]) > 0 else np.nan
        pk = int(np.nanargmax(y)) if np.isfinite(y).any() else None
        ratio = x[pk] / y[pk] if pk is not None and y[pk] > 0 else np.nan
        rows.append({"variant": var, "lga": name, "pcode": pc, "state": adm2["ADM1_EN"].iloc[j],
                     "sfed_max": smax_l[j], "r": r, "peak_ratio": ratio})
    df = pd.DataFrame([r for r in rows if r["variant"] == var])
    print("Nigeria LGAs, median r / median peak ratio (ours/SFED) by SFED seasonal max:")
    for lo, hi in BANDS:
        s = df[(df.sfed_max >= lo) & (df.sfed_max < hi)]
        print(f"   {lo:.2f}-{'' if hi == 9 else f'{hi:.2f}':5s} n {s.r.notna().sum():3d}: r {s.r.median():.3f}, peak ratio {s.peak_ratio.median():.2f}")
    top = df.sort_values("sfed_max", ascending=False).head(10)
    print("   top-10 SFED LGAs: " + "; ".join(f"{t.lga} ({t.state}) {t.r:.2f}" for t in top.itertuples()))

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "nga2024_lga_by_variant.csv", index=False)
print(f"\nwrote {OUT / 'nga2024_lga_by_variant.csv'}")

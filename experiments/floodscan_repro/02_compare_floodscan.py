"""Step 0 scoring: our FloodScan-core flooded fraction (01) against FloodScan SFED area (300"), Nigeria 2022.

Each product is aggregated on its own native grid:
- 0.5-deg blocks: 5x5 of our 0.1-deg cells, 6x6 of SFED's 1/12-deg cells (both nest exactly);
  a block counts if >= 50% of its cells are valid.
- Nigeria LGAs: mean over cells touching the LGA (all_touched, as in experiments/gfds_nga2022).
LGA severity bands use the SFED seasonal max from results/gfds_rebuild/nga2022_lga_r_by_source.csv,
so the GFDS-style result (06) and this one are scored on the same LGAs.

Variants: no MDFF, and a uniform MDFF of 0.06 (AER's worked example value; AER's MDFF is per pixel
and per day of year). Magnitude: orthogonal slope and mean ratio on 0.5-deg blocks; median
ours/SFED ratio at each LGA's SFED peak day.

    uv run python experiments/floodscan_repro/02_compare_floodscan.py
"""

import datetime as dt
from pathlib import Path

import numpy as np
import ocha_stratus as stratus
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import Affine

NGA = Path("/Users/zackarno/Documents/CHD/repos/ds-flood-gfm/data/gdacs_gfds/nga2022")  # main checkout cache
CORE = Path("data/floodscan_repro/nga2022_core_l3.npz")
GFDS_CSV = Path("results/gfds_rebuild/nga2022_lga_r_by_source.csv")
BANDS = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9)]
REF = ["Ibaji", "Sagbama", "Kogi", "Lokoja", "Bassa", "Ajaokuta", "Ekeremor"]

z = np.load(CORE)
ff0, lat, lon = z["ff"].astype("float64"), z["lat"], z["lon"]
days = [dt.date.fromisoformat(s) for s in z["days"]]
ours_aff = Affine(0.1, 0, round(lon[0] - 0.05, 6), 0, -0.1, round(lat[0] + 0.05, 6))

sc = np.load(NGA / "sfed_coords.npz")
sx, sy = sc["x"], sc["y"]
res = 1 / 12
sfed_aff = Affine(res, 0, sx[0] - res / 2, 0, -res, sy[0] + res / 2)
SF = np.stack([np.load(NGA / f"sfed_{d:%Y%m%d}.npy").astype("float64") for d in days])
print(f"SFED: {len(days)} days, grid {SF.shape[1:]}; ours: grid {ff0.shape[1:]}")

# 0.5-deg block indices: lon 3.0-14.5, lat 14.0-4.0
oj0 = int(round((3.0 - (lon[0] - 0.05)) / 0.1)); oi0 = int(round(((lat[0] + 0.05) - 14.0) / 0.1))
sj0 = int(round((3.0 - (sx[0] - res / 2)) / res)); si0 = int(round(((sy[0] + res / 2) - 14.0) / res))
NBY, NBX = 20, 23


def blocks(x, i0, j0, k):
    sub = x[:, i0 : i0 + NBY * k, j0 : j0 + NBX * k].reshape(x.shape[0], NBY, k, NBX, k)
    n = np.isfinite(sub).sum((2, 4))
    s = np.nansum(sub, (2, 4))
    return np.where(n >= 0.5 * k * k, s / np.maximum(n, 1), np.nan)


def r_series(a, b, min_n=60):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n or np.std(b[ok]) == 0 or np.std(a[ok]) == 0:
        return np.nan
    return np.corrcoef(a[ok], b[ok])[0, 1]


adm2 = stratus.codab.load_codab_from_blob("nga", admin_level=2)
m_ours = [~geometry_mask([g], ff0.shape[1:], ours_aff, all_touched=True) for g in adm2.geometry]
m_sfed = [~geometry_mask([g], SF.shape[1:], sfed_aff, all_touched=True) for g in adm2.geometry]


def lga_means(x, masks):
    return np.stack([[np.nanmean(x[t][m]) if np.isfinite(x[t][m]).any() else np.nan for m in masks] for t in range(len(x))])


sf_l = lga_means(SF, m_sfed)
gfds = pd.read_csv(GFDS_CSV)
band_of = gfds[gfds.source == "ours"].set_index("pcode")["sfed_max"]
gfds_r = gfds[gfds.source == "ours"].set_index("pcode")["r"]
sb = blocks(SF, si0, sj0, 6)

for label, ff in (("no MDFF", ff0), ("MDFF 0.06 uniform", np.where(ff0 < 0.06, 0.0, ff0))):
    print(f"\n===== FloodScan core, {label} =====")
    ob = blocks(ff, oi0, oj0, 5)
    both = np.isfinite(ob) & np.isfinite(sb)
    o1, s1 = ob[both], sb[both]
    _, v = np.linalg.eigh(np.cov(np.vstack([s1 - s1.mean(), o1 - o1.mean()])))
    print(f"0.5-deg blocks, pooled block-days (n = {both.sum():,}): r = {np.corrcoef(o1, s1)[0, 1]:.3f}, "
          f"orthogonal slope ours~SFED = {v[1, 1] / v[0, 1]:.2f}, mean ours / mean SFED = {o1.mean() / s1.mean():.2f}")
    smax_b = np.nanmax(sb, 0)
    for thr in (0.05, 0.2):
        rs = [r_series(ob[:, i, j], sb[:, i, j]) for i, j in zip(*np.where(smax_b > thr))]
        rs = np.array([r for r in rs if np.isfinite(r)])
        print(f"   per-block temporal r, blocks with SFED max > {thr}: median {np.median(rs):.3f} (n {len(rs)})")

    of_l = lga_means(ff, m_ours)
    rows = []
    for j, pc in enumerate(adm2["ADM2_PCODE"]):
        if pc not in band_of.index:
            continue
        pk = np.nanargmax(sf_l[:, j]) if np.isfinite(sf_l[:, j]).any() else None
        ratio = of_l[pk, j] / sf_l[pk, j] if pk is not None and sf_l[pk, j] > 0 else np.nan
        rows.append({"pcode": pc, "lga": adm2["ADM2_EN"].iloc[j], "sfed_max": band_of[pc],
                     "r_core": r_series(of_l[:, j], sf_l[:, j]), "r_gfds": gfds_r[pc], "peak_ratio": ratio})
    df = pd.DataFrame(rows)
    print("LGA level, median r vs SFED (same LGAs per band):")
    print(f"   {'band':12s} {'n':>4s} {'FloodScan core':>15s} {'GFDS-style (06)':>16s} {'peak ours/SFED':>15s}")
    for lo, hi in BANDS:
        s = df[(df.sfed_max >= lo) & (df.sfed_max < hi)]
        print(f"   {lo:.2f}-{'' if hi == 9 else f'{hi:.2f}':5s} {s.r_core.notna().sum():4d} {s.r_core.median():15.3f} "
              f"{s.r_gfds.median():16.3f} {s.peak_ratio.median():15.2f}")
    print("   reference LGAs (core / GFDS-style):",
          ", ".join(f"{n} {df[df.lga == n].r_core.iloc[0]:.2f}/{df[df.lga == n].r_gfds.iloc[0]:.2f}" for n in REF))
    if label == "no MDFF":
        out = Path("results/floodscan_repro")
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "nga2022_core_l3_lga.csv", index=False)
        print(f"   wrote {out / 'nga2022_core_l3_lga.csv'}")

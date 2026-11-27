"""Checks before phase 2: (1) our Q_pow vs AER's; (2) timing vs amplitude in the LGA peak metric.
Run from the worktree with the main venv (ocha_stratus)."""
import datetime as dt
from pathlib import Path

import numpy as np
import ocha_stratus as stratus
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window

t = np.load("data/floodscan_repro/static_tables.npz")
print("Q_pow ours (CSU AMSR2, open sea):", {k: round(float(t[k]), 4) for k in t.files if k.startswith("Qpow")})
# AER: AMSR-E 37 GHz 0.457 asc / 0.455 dsc; AMSR2 -> AMSR-E Q' = A Q + B, asc 1.0466/-0.0154, dsc 1.0438/-0.0141
print("AER Q_pow 37 GHz on AMSR2-native scale (JAXA cal): asc", round((0.457 + 0.0154) / 1.0466, 4), "dsc", round((0.455 + 0.0141) / 1.0438, 4))
print("AER Q_pow 19 GHz AMSR-E: asc 0.465 dsc 0.463 (no AMSR2 cross-cal at 19 GHz in the notes)")

FS = Path("/Users/zackarno/Documents/CHD/repos/ds-flood-gfm/data/fusion/raw/floodscan/africa_daily")
z = np.load("data/floodscan_repro/ff_2024.npz")
days = [dt.date.fromisoformat(s) for s in z["days"]]
fpow = z["fpow"]
sfn = []
for d in days:
    with rasterio.open(FS / f"aer_area_300s_v{d:%Y-%m-%d}_v05r01.tif") as ds:
        sfn.append(ds.read(list(ds.descriptions).index("SFED") + 1, window=Window(360, 240, 240, 240)).astype("float64"))
SFN = np.stack(sfn)
adm2 = stratus.codab.load_codab_from_blob("nga", admin_level=2)
m_ours = [~geometry_mask([g], (80, 80), Affine(0.25, 0, 0, 0, -0.25, 20), all_touched=True) for g in adm2.geometry]
m_sf = [~geometry_mask([g], (240, 240), Affine(1 / 12, 0, 0, 0, -1 / 12, 20), all_touched=True) for g in adm2.geometry]


def lga(x, masks):
    return np.stack([[np.nanmean(x[k][m]) if np.isfinite(x[k][m]).any() else np.nan for m in masks] for k in range(len(x))])


sf_l = lga(SFN, m_sf)
smax = np.nanmax(sf_l, 0)
for v in ("clim_raw", "noflash", "mdff_half"):
    with np.errstate(invalid="ignore", divide="ignore"):
        o = np.where(fpow <= 0.5, z[v] / (1 - fpow), np.nan)
    o_l = lga(o, m_ours)
    rows = []
    for j in range(len(adm2)):
        if smax[j] < 0.10:
            continue
        y, x = sf_l[:, j], o_l[:, j]
        pk = int(np.nanargmax(y))
        lag = int(np.nanargmax(x)) - pk if np.isfinite(x).any() else np.nan
        rows.append({"lga": adm2["ADM2_EN"].iloc[j], "state": adm2["ADM1_EN"].iloc[j], "sfed_max": smax[j],
                     "at_peak": x[pk] / y[pk], "max_over_max": np.nanmax(x) / y[pk], "lag_days": lag})
    df = pd.DataFrame(rows)
    print(f"\n{v}: LGAs with SFED max >= 0.10: {len(df)}")
    print(df[["at_peak", "max_over_max", "lag_days"]].describe().loc[["25%", "50%", "75%"]].round(2).to_string())
    print(" by state (median max/max, n):")
    print(df.groupby("state")["max_over_max"].agg(["median", "count"]).round(2).sort_values("count", ascending=False).head(8).to_string())

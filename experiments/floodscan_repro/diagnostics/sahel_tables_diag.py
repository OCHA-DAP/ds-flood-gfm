"""Why does the 2023 hold-out flood the Sahel band in Oct-Nov? Compare the dry references of the two table sets
(2023+2025 vs 2024+2025) in the band 12.5-17.5N, and the yearly Q cycle there."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments/floodscan_repro")
from aer_chain import load_year  # noqa: E402

T = {"2325": np.load("data/floodscan_repro/static_tables.npz"), "2425": np.load("data/floodscan_repro/static_tables_2425.npz")}
land = T["2325"]["focean"] < 0.5
band = np.zeros((80, 80), bool); band[10:30] = True; band &= land
south = np.zeros((80, 80), bool); south[30:80] = True; south &= land
for k in ("Qdc_36_asc", "Q61c_36_asc", "MDFF_36_asc", "var61_36_asc", "wet_36_asc", "Qpow_36_asc", "fhop"):
    for name, t in T.items():
        if k in t:
            print(f"{k:14s} {name}: shape {t[k].shape} dtype {t[k].dtype}")


def at(t, k, doy, mask):
    a = t[k]
    if a.ndim == 3:
        return np.nanmedian(a[doy - 1][mask])
    return np.nanmedian(a[mask])


print("\nband 12.5-17.5N medians (asc): DOY 230 (Aug 18), 288 (Oct 15), 319 (Nov 15)  | south of 12.5N")
for k in ("Qdc_36_asc", "Q61c_36_asc", "MDFF_36_asc", "wet_36_asc"):
    for name, t in T.items():
        print(f"  {k:12s} {name}: " + "  ".join(f"{at(t, k, d, band):.4f}" for d in (230, 288, 319))
              + "   | " + "  ".join(f"{at(t, k, d, south):.4f}" for d in (230, 288, 319)))

print("\nband-median Q (36 asc) by 10-day block from Jun 1, per year")
for y in (2023, 2024, 2025):
    days, Q, TB = load_year(Path(f"data/floodscan_repro/csu_amsr2_grid/{y}"), dt.date(y, 6, 1), dt.date(y, 12, 31))
    q = Q[("36", "asc")]
    med = np.array([np.nanmedian(f[band]) for f in q])
    print(f"  {y} band : " + " ".join(f"{np.nanmean(med[i:i+10]):.4f}" for i in range(0, 214, 10)))
    meds = np.array([np.nanmedian(f[south]) for f in q])
    print(f"  {y} south: " + " ".join(f"{np.nanmean(meds[i:i+10]):.4f}" for i in range(0, 214, 10)))

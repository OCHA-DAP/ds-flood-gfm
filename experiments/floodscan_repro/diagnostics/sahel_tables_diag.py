"""Why does the 2023 hold-out flood the Sahel band in Oct-Nov? Compare the dry references of the two table sets
(2023+2025 vs 2024+2025) in the band 12.5-17.5N, and the yearly Q cycle there. Prints the tables and writes
results/floodscan_repro/holdout_2023/sahel_band_references.csv and sahel_band_q_cycle.csv (loaded by the book).

    uv run --no-project --with numpy --with scipy --with rasterio python experiments/floodscan_repro/diagnostics/sahel_tables_diag.py
"""
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments/floodscan_repro")
from aer_chain import load_year  # noqa: E402

OUT = Path("results/floodscan_repro/holdout_2023")
T = {"2023+2025": np.load("data/floodscan_repro/static_tables.npz"), "2024+2025": np.load("data/floodscan_repro/static_tables_2425.npz")}
land = T["2023+2025"]["focean"] < 0.5
band = np.zeros((80, 80), bool); band[10:30] = True; band &= land      # rows 10-29 = 12.5-17.5N in the 0-20N box
south = np.zeros((80, 80), bool); south[30:80] = True; south &= land   # south of 12.5N
DOYS = {230: "Aug 18", 288: "Oct 15", 319: "Nov 15"}


def at(t, k, doy, mask):
    a = t[k]
    return float(np.nanmedian(a[doy - 1][mask])) if a.ndim == 3 else float(np.nanmedian(a[mask]))


print("band 12.5-17.5N medians (asc): DOY 230 (Aug 18), 288 (Oct 15), 319 (Nov 15)  | south of 12.5N")
ref_rows = []
for k in ("Qdc_36_asc", "Q61c_36_asc", "MDFF_36_asc", "wet_36_asc"):
    for name, t in T.items():
        vals = {d: (at(t, k, d, band), at(t, k, d, south)) for d in DOYS}
        for d, (b, s) in vals.items():
            ref_rows.append({"table": k, "training_years": name, "doy": d, "date": DOYS[d], "band_12.5_17.5N": round(b, 4), "south_of_12.5N": round(s, 4)})
        print(f"  {k:12s} {name}: " + "  ".join(f"{vals[d][0]:.4f}" for d in DOYS) + "   | " + "  ".join(f"{vals[d][1]:.4f}" for d in DOYS))

print("\nband-median Q (36 asc) by 10-day block from Jun 1, per year")
cycle_rows = []
for y in (2023, 2024, 2025):
    days, Q, TB = load_year(Path(f"data/floodscan_repro/csu_amsr2_grid/{y}"), dt.date(y, 6, 1), dt.date(y, 12, 31))
    q = Q[("36", "asc")]
    for region, mask in (("band_12.5_17.5N", band), ("south_of_12.5N", south)):
        med = np.array([np.nanmedian(f[mask]) for f in q])
        blocks = [float(np.nanmean(med[i:i + 10])) for i in range(0, 214, 10)]
        for i, v in enumerate(blocks):
            cycle_rows.append({"year": y, "region": region, "block_start": (dt.date(y, 6, 1) + dt.timedelta(10 * i)).isoformat(), "median_Q_36_asc": round(v, 4)})
        print(f"  {y} {region:16s}: " + " ".join(f"{v:.4f}" for v in blocks))

OUT.mkdir(parents=True, exist_ok=True)
for fname, rows in (("sahel_band_references.csv", ref_rows), ("sahel_band_q_cycle.csv", cycle_rows)):
    with (OUT / fname).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT / fname}")

"""Monthly median of Q (36 GHz, asc/dsc) and of the 18 GHz Tb over land cells, by year, from the CSU grid npz.
Looks for a level shift between years (e.g. FCDR vs ICDR calibration). Prints the table and writes it to
results/floodscan_repro/holdout_2023/q_level_by_year.csv (loaded by the book).

    uv run --no-project --with numpy --with scipy --with rasterio python experiments/floodscan_repro/diagnostics/q_level_by_year.py
"""
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments/floodscan_repro")
from aer_chain import load_year  # noqa: E402

OUT = Path("results/floodscan_repro/holdout_2023/q_level_by_year.csv")
static = np.load("data/floodscan_repro/static_tables.npz")
land = static["focean"] < 0.5
rows = []
for year in (2023, 2024, 2025):
    src = Path(f"data/floodscan_repro/csu_amsr2_grid/{year}")
    days, Q, TB = load_year(src, dt.date(year, 6, 1), dt.date(year, 12, 31))
    print(f"\n{year}: Q keys {list(Q.keys())}, TB keys {list(TB.keys())}, days {len(days)}")
    months = np.array([d.month for d in days])
    for key in (("36", "asc"), ("36", "dsc")):
        q = Q[key]
        line = []
        for m in (6, 7, 8, 9, 10, 11, 12):
            qm = q[months == m][:, land]
            med, valid = float(np.nanmedian(qm)), float(np.isfinite(qm).mean())
            rows.append({"year": year, "variable": f"Q_{key[0]}_{key[1]}", "month": m, "median": round(med, 4), "valid_share": round(valid, 3)})
            line.append(f"{m:02d}:{med:.4f}/{valid:.2f}")
        print(f"  {key} median Q / valid share  ", "  ".join(line))
    for key in [k for k in TB.keys() if k[0] in ("18v", "18h")]:
        t = TB[key]
        line = []
        for m in (6, 7, 8, 9, 10, 11, 12):
            tm = t[months == m][:, land]
            med = float(np.nanmedian(tm))
            rows.append({"year": year, "variable": f"Tb_{key[0]}_{key[1]}", "month": m, "median": round(med, 2), "valid_share": round(float(np.isfinite(tm).mean()), 3)})
            line.append(f"{m:02d}:{med:.2f}")
        print(f"  {key} median Tb  ", "  ".join(line))
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"\nwrote {OUT}")

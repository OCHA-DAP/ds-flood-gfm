"""Monthly median of Q (36 GHz, asc/dsc) and of the 36V/36H Tb over inland land cells, by year, from the CSU grid npz.
Looks for a level shift between years (e.g. FCDR vs ICDR calibration)."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments/floodscan_repro")
from aer_chain import load_year  # noqa: E402

static = np.load("data/floodscan_repro/static_tables.npz")
keys = list(static.keys())
land = static["focean"] < 0.5
print("static keys:", keys[:40])
for year in (2023, 2024, 2025):
    src = Path(f"data/floodscan_repro/csu_amsr2_grid/{year}")
    days, Q, TB = load_year(src, dt.date(year, 6, 1), dt.date(year, 12, 31))
    print(f"\n{year}: Q keys {list(Q.keys())}, TB keys {list(TB.keys())}, days {len(days)}")
    months = np.array([d.month for d in days])
    for key in (("36", "asc"), ("36", "dsc")):
        q = Q[key]
        line = []
        for m in (6, 7, 8, 9, 10, 11, 12):
            sel = months == m
            qm = q[sel]
            if land is not None:
                qm = qm[:, land]
            line.append(f"{m:02d}:{np.nanmedian(qm):.4f}/{np.isfinite(qm).mean():.2f}")
        print(f"  {key} median Q / valid share  ", "  ".join(line))
    for key in [k for k in TB.keys() if k[0] in ("18v", "18h")]:
        t = TB[key]
        line = []
        for m in (6, 7, 8, 9, 10, 11, 12):
            sel = months == m
            tm = t[sel]
            if land is not None:
                tm = tm[:, land]
            line.append(f"{m:02d}:{np.nanmedian(tm):.2f}")
        print(f"  {key} median Tb  ", "  ".join(line))

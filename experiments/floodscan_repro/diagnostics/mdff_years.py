"""Is MDFF inflated by having only two training years? Compare one-year vs two-year tables."""
import numpy as np

S = "data/floodscan_repro/"  # one-year tables: `04 2023 data/floodscan_repro/static_tables_2023only.npz`, same for 2025
both = np.load("data/floodscan_repro/static_tables.npz")
y23 = np.load(S + "static_tables_2023only.npz")
y25 = np.load(S + "static_tables_2025only.npz")
inland = both["focean"] <= 0.001 if "focean" in both.files else np.ones((80, 80), bool)
fpow = both["fpow"]
sel = inland & (fpow <= 0.5)
for key in ("MDFF_36_asc", "MDFF_36_dsc", "var61_36_asc"):
    for name, t in (("2023+2025", both), ("2023 only", y23), ("2025 only", y25)):
        a = t[key][:, sel]  # (doy, cells)
        # Aug-Nov day-of-year 213..334 (2023 calendar)
        a = a[212:334]
        print(f"{key:14s} {name:10s} median {np.nanmedian(a):.4f}  p25 {np.nanpercentile(a, 25):.4f}  p75 {np.nanpercentile(a, 75):.4f}  p90 {np.nanpercentile(a, 90):.4f}")
    print()

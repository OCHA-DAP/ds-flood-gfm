"""Why are f_hop and MDFF NaN at inland cells in the in-sample 2024 tables?"""
import datetime as dt
import sys

import numpy as np
from scipy.stats import kurtosis

sys.path.insert(0, "experiments/floodscan_repro")
from aer_chain import clim_doy, initial_ff, load_year, running_mean  # noqa: E402

t = np.load(sys.argv[1])
fpow = t["fpow"]
land = fpow <= 0.5
# --- MDFF NaN cells
m = t["MDFF_36_asc"][250]
bad = ~np.isfinite(m) & land
print("MDFF NaN land cells:", bad.sum())
for name in ("Qdc_36_asc", "QdLUT_36_asc", "Q61c_36_asc", "var61_36_asc"):
    print(f"  {name} NaN among them: {(~np.isfinite(t[name][250][bad])).sum()}")
print("  ndvi_doy NaN among them:", (~np.isfinite(t["ndvi_doy"][250][bad])).sum(), "ftree NaN:", (~np.isfinite(t["ftree"][bad])).sum())
print("  ndvi NaN land cells total:", (~np.isfinite(t["ndvi_doy"][250]) & land).sum())

# --- f_hop
days, q, _ = load_year("data/floodscan_repro/csu_amsr2_grid/2024", dt.date(2024, 6, 1), dt.date(2024, 12, 31))
doy = np.array([clim_doy(d) for d in days])
ff = initial_ff(q, t, doy, fpow, freqs=("36",))["pw"]
x = ff[("36", "asc")]
print("ff valid share per month:", {mo: round(float(np.isfinite(x[[d.month == mo for d in days]][:, land]).mean()), 2) for mo in range(6, 13)})
x = np.maximum(x - running_mean(x, 7), 0)
d = np.full_like(x, np.nan)
d[1:] = x[1:] - x[:-1]
for dc in (230, 250, 300):
    sample = d[(doy >= dc - 30) & (doy <= dc + 30)]
    nvalid = np.isfinite(sample).sum(0)
    k = kurtosis(sample, axis=0, fisher=False, nan_policy="omit")
    print(f"dc {dc}: sample {sample.shape}, median n valid (land) {np.median(nvalid[land])}, k type {type(k).__name__}, "
          f"k NaN land {np.isnan(np.asarray(k, dtype=float)[land]).sum()}, masked {np.ma.count_masked(k) if np.ma.isMaskedArray(k) else 'n/a'}, "
          f"var0 land {(np.nanvar(sample, 0)[land] == 0).sum()}")
    kf = np.asarray(k, dtype=float)
    print("   k land quantiles:", np.nanpercentile(kf[land], [5, 50, 95]).round(2))
